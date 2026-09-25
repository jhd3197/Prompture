"""The local companion server: the companion API over Prompture's own ledger.

``prompture companion`` runs this. It speaks the same endpoints as
prompture-hub's companion API, so a desktop companion can show this machine's
Prompture usage with no hub at all:

- ``GET /v1/companion/info`` — public; version, API version, features.
- ``GET /v1/live`` — Server-Sent Events (calls as they finish).
- ``GET /v1/spend?period=day|week|month`` and ``GET /v1/limits``.
- ``GET /v1/alerts`` — always empty; alert rules live in the hub.

With a :class:`~.coding_tools.CodingToolSource` it also counts the calls local
coding agents (Claude Code, Codex, Kimi Code, Gemini CLI, …) log on disk, adds
their plan windows to the limits, and serves ``GET /v1/tools?period=`` — each
agent's usage plus which agents are installed.

It listens on ``127.0.0.1`` only, on a free port the OS picks, and writes the
address and a random bearer token to ``~/.prompture/companion.json`` (readable
only by the current user on POSIX). Readers take both from there. Only the
Python standard library is used, so a plain ``pip install prompture`` is enough.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .coding_tools import CodingToolSource
from .live import LiveBus, get_bus, sse_event
from .local import LedgerSource
from .summary import COMPANION_API_VERSION, PERIODS, UsageRow, account_limits, provider_limits, summarize_spend

logger = logging.getLogger("prompture.companion")

STATE_FILE = Path.home() / ".prompture" / "companion.json"
HEARTBEAT_SECONDS = 15.0

#: What a client can rely on from this server. The hub advertises more.
FEATURES = {
    "live": "/v1/live",
    "limits": "/v1/limits",
    "spend": "/v1/spend",
    "alerts": "/v1/alerts",
    "tools": "/v1/tools",
    "activity": "/v1/activity",
}
CAPABILITIES = {
    "running_calls": False,  # the ledger only sees calls after they finish
    "projects": True,
    "key_controls": False,
    "provider_controls": False,
    "alert_rules": False,
    "coding_tools": False,
    "activity": True,
}


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("prompture")
    except Exception:
        return "0"


def info(coding_tools: bool = False) -> dict[str, Any]:
    return {
        "service": "prompture",
        "mode": "local",
        "version": _version(),
        "api_version": COMPANION_API_VERSION,
        "features": dict(FEATURES),
        "capabilities": {**CAPABILITIES, "coding_tools": coding_tools},
    }


def read_state(path: Path = STATE_FILE) -> dict[str, Any] | None:
    """The running companion's ``{url, token, pid}``, or ``None``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("url") and data.get("token") else None


def _write_state(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    if os.name == "posix":
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _tz_offset(query: dict[str, str]) -> int:
    """``tz_offset`` (minutes behind UTC, as JavaScript reports it); 0 = UTC windows."""
    try:
        return min(840, max(-840, int(query.get("tz_offset", "0"))))
    except ValueError:
        return 0


class _Handler(BaseHTTPRequestHandler):
    server: CompanionServer  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("companion %s", fmt % args)

    def _json(self, status: int, body: Any) -> None:
        raw = json.dumps(body, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        return bool(token) and secrets.compare_digest(token, self.server.token)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if not self._authorized():
            return self._json(401, {"detail": "Missing or wrong companion token (see ~/.prompture/companion.json)."})
        if url.path == "/v1/tools/claude-plan" and self.server.coding_tools is not None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._json(400, {"detail": "Body must be JSON."})
            if not isinstance(body, dict) or not isinstance(body.get("enabled"), bool):
                return self._json(422, {"detail": 'Send {"enabled": true|false}.'})
            self.server.coding_tools.set_claude_plan_usage(body["enabled"])
            return self._json(200, {"claude_plan_usage": self.server.coding_tools.plan_usage})
        return self._json(404, {"detail": "Not found"})

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        if url.path == "/v1/companion/info":
            return self._json(200, info(self.server.coding_tools is not None))
        if not self._authorized():
            return self._json(401, {"detail": "Missing or wrong companion token (see ~/.prompture/companion.json)."})
        if url.path == "/v1/spend":
            period = query.get("period", "day")
            if period not in PERIODS:
                return self._json(422, {"detail": "period must be day, week or month"})
            sources = query.get("sources", "all")
            if sources not in ("all", "api"):
                return self._json(422, {"detail": "sources must be all or api"})
            # "api": only calls made through Prompture, billed per token — what budgets measure.
            offset = _tz_offset(query)
            rows = self.server.rows(period, api_only=sources == "api", offset_minutes=offset)
            return self._json(200, summarize_spend(rows, period, offset_minutes=offset))
        if url.path == "/v1/limits":
            return self._json(200, self.server.limits(accounts=query.get("accounts", "true") != "false"))
        if url.path == "/v1/alerts":
            return self._json(200, [])
        if url.path == "/v1/activity":
            try:
                days = min(400, max(1, int(query.get("days", "371"))))
                offset = min(840, max(-840, int(query.get("tz_offset", "0"))))
            except ValueError:
                return self._json(422, {"detail": "days and tz_offset must be integers"})
            return self._json(200, self.server.activity(days, offset))
        if url.path == "/v1/tools":
            if self.server.coding_tools is None:
                return self._json(
                    404, {"detail": "Coding-tool usage is off (companion started with --no-coding-tools)."}
                )
            period = query.get("period", "day")
            if period not in PERIODS:
                return self._json(422, {"detail": "period must be day, week or month"})
            return self._json(200, self.server.coding_tools.tools(period, offset_minutes=_tz_offset(query)))
        if url.path == "/v1/live":
            return self._live(query)
        return self._json(404, {"detail": "Not found"})

    def _live(self, query: dict[str, str]) -> None:
        bus = self.server.bus
        header_id = self.headers.get("Last-Event-ID")
        resume = int(header_id) if header_id and header_id.isdigit() else None
        if resume is None and query.get("after", "").isdigit():
            resume = int(query["after"])
        limit = int(query["limit"]) if query.get("limit", "").isdigit() else None
        sub = bus.subscribe_thread()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        sent_id, count = 0, 0

        def send(chunk: str) -> None:
            self.wfile.write(chunk.encode())
            self.wfile.flush()

        try:
            send("retry: 3000\n\n")
            if resume is not None:
                initial = bus.replay(resume)
            else:
                sent_id = bus.last_id()
                initial = [{"type": "snapshot", "running": bus.running(), "last_id": sent_id}]
            for event in initial:
                sent_id = max(sent_id, event.get("id", 0))
                send(sse_event(event))
                count += 1
                if limit and count >= limit:
                    return
            while not self.server.stopping.is_set():
                try:
                    event = sub.queue.get(timeout=HEARTBEAT_SECONDS)
                except queue.Empty:
                    send(": keep-alive\n\n")
                    continue
                if sub.overflowed:
                    sub.overflowed = False
                    send(sse_event({"type": "resync", "reason": "client fell behind"}))
                if event["id"] <= sent_id:
                    continue
                sent_id = event["id"]
                send(sse_event(event))
                count += 1
                if limit and count >= limit:
                    return
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        finally:
            bus.unsubscribe(sub)


class CompanionServer(ThreadingHTTPServer):
    """Threaded HTTP server bound to localhost; see the module docstring."""

    daemon_threads = True

    def __init__(
        self,
        ledger: LedgerSource | None = None,
        *,
        port: int = 0,
        token: str | None = None,
        bus: LiveBus | None = None,
        state_path: Path | None = STATE_FILE,
        coding_tools: CodingToolSource | None = None,
    ) -> None:
        super().__init__(("127.0.0.1", port), _Handler)
        self.ledger = ledger or LedgerSource()
        self.token = token or secrets.token_urlsafe(32)
        self.bus = bus or get_bus()
        self.state_path = state_path
        self.coding_tools = coding_tools
        self.stopping = threading.Event()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def rows(self, period: str, *, api_only: bool = False, offset_minutes: int = 0) -> list[UsageRow]:
        """The ledger's rows for *period*, plus coding-tool calls when enabled (unless ``api_only``)."""
        rows = self.ledger.rows(period, offset_minutes=offset_minutes)
        if self.coding_tools is not None and not api_only:
            rows += self.coding_tools.rows(period, offset_minutes=offset_minutes)
        return rows

    def rate_limits(self) -> dict[str, dict[str, Any]]:
        limits = self.ledger.rate_limits()
        if self.coding_tools is not None:
            try:
                limits.update(self.coding_tools.rate_limits())
            except Exception:  # coding-tool limits are extra context, never a failure
                logger.debug("coding tool limits failed", exc_info=True)
        return limits

    def _tail(self) -> None:
        threading.Thread(target=self.ledger.tail, args=(self.bus, self.stopping), daemon=True).start()
        if self.coding_tools is not None:
            threading.Thread(target=self.coding_tools.tail, args=(self.bus, self.stopping), daemon=True).start()

    def activity(self, days: int = 371, offset_minutes: int = 0) -> dict[str, Any]:
        """Per-day totals for the last ``days`` local days: Prompture calls plus coding tools.

        Only days with activity are listed; each names what contributed
        (``sources``: "Prompture" and each coding agent, by tokens).
        """
        from datetime import datetime, timedelta, timezone

        local_now = datetime.now(timezone.utc) - timedelta(minutes=offset_minutes)
        first_day = local_now.date() - timedelta(days=days - 1)
        since = datetime.combine(first_day, datetime.min.time(), timezone.utc) + timedelta(minutes=offset_minutes)
        merged: dict[str, dict[str, Any]] = {}

        def add(day: str, requests: int, tokens: int, cost: float, name: str) -> None:
            d = merged.setdefault(day, {"date": day, "requests": 0, "tokens": 0, "cost_usd": 0.0, "sources": {}})
            d["requests"] += requests
            d["tokens"] += tokens
            d["cost_usd"] += cost
            s = d["sources"].setdefault(name, {"name": name, "requests": 0, "tokens": 0})
            s["requests"] += requests
            s["tokens"] += tokens

        for day, t in self.ledger.daily(since, offset_minutes).items():
            add(day, t["requests"], t["tokens"], t["cost_usd"], "Prompture")
        if self.coding_tools is not None:
            names = self.coding_tools.names
            for day, t in self.coding_tools.usage.daily(since, offset_minutes).items():
                for agent, a in t["agents"].items():
                    share = a["tokens"] / t["tokens"] if t["tokens"] else 0.0
                    add(day, a["requests"], a["tokens"], t["cost_usd"] * share, names.get(agent, agent))
        out = []
        for day in sorted(merged):
            if day < first_day.isoformat():
                continue
            d = merged[day]
            d["cost_usd"] = round(d["cost_usd"], 6)
            d["sources"] = sorted(d["sources"].values(), key=lambda s: -s["tokens"])
            out.append(d)
        return {"start": first_day.isoformat(), "end": local_now.date().isoformat(), "days": out}

    def limits(self, *, accounts: bool = True) -> dict[str, Any]:
        from datetime import datetime, timezone

        body: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "keys": [],
            "providers": provider_limits(self.rate_limits()),
            "accounts": None,
            "paused_providers": [],
        }
        if accounts:
            try:
                body["accounts"] = account_limits()
            except Exception:  # accounts are optional context, never a failure
                logger.debug("account lookup failed", exc_info=True)
                body["accounts"] = []
        return body

    def start_background(self) -> threading.Thread:
        """Start tailing the ledger and serving on background threads (for tests and embedding)."""
        self._tail()
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread

    def run(self) -> None:
        """Serve until interrupted, advertising the address in the state file."""
        if self.state_path is not None:
            _write_state(self.state_path, {"url": self.url, "token": self.token, "pid": os.getpid()})
        self._tail()
        try:
            self.serve_forever()
        finally:
            self.shutdown_companion()

    def shutdown_companion(self) -> None:
        self.stopping.set()
        if self.state_path is not None:
            state = read_state(self.state_path)
            if state and state.get("pid") == os.getpid():
                with contextlib.suppress(OSError):
                    self.state_path.unlink()
        self.server_close()


def running_instance(path: Path = STATE_FILE, timeout: float = 2.0) -> dict[str, Any] | None:
    """The state of a companion that is already up and answering, else ``None``.

    The state file is only trusted to name a plain-HTTP loopback address, so a
    tampered file can't point this check at another host or a ``file:`` URL.
    """
    import http.client

    state = read_state(path)
    if not state:
        return None
    url = urlparse(str(state.get("url", "")))
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"} or not url.port:
        return None
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    try:
        conn.request("GET", "/v1/companion/info")
        return state if conn.getresponse().status == 200 else None
    except OSError:
        return None
    finally:
        conn.close()


def process_alive(pid: int) -> bool:
    """Whether *pid* is a running process (best effort, no extra dependencies)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        synchronize, wait_timeout = 0x00100000, 0x00000102
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_when_process_exits(server: CompanionServer, pid: int, interval: float = 3.0) -> threading.Thread:
    """Shut *server* down once process *pid* is gone."""

    def watch() -> None:
        while not server.stopping.wait(interval):
            if not process_alive(pid):
                logger.info("Owner process %s exited; stopping the companion.", pid)
                server.shutdown()
                return

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    return thread
