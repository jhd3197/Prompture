"""The local companion server: the companion API over Prompture's own ledger.

``prompture companion`` runs this. It speaks the same endpoints as
prompture-hub's companion API, so a desktop companion can show this machine's
Prompture usage with no hub at all:

- ``GET /v1/companion/info`` — public; version, API version, features.
- ``GET /v1/live`` — Server-Sent Events (calls as they finish).
- ``GET /v1/spend?period=day|week|month`` and ``GET /v1/limits``.
- ``GET /v1/alerts`` — always empty; alert rules live in the hub.

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

from .live import LiveBus, get_bus, sse_event
from .local import LedgerSource
from .summary import COMPANION_API_VERSION, PERIODS, account_limits, provider_limits, summarize_spend

logger = logging.getLogger("prompture.companion")

STATE_FILE = Path.home() / ".prompture" / "companion.json"
HEARTBEAT_SECONDS = 15.0

#: What a client can rely on from this server. The hub advertises more.
FEATURES = {"live": "/v1/live", "limits": "/v1/limits", "spend": "/v1/spend", "alerts": "/v1/alerts"}
CAPABILITIES = {
    "running_calls": False,  # the ledger only sees calls after they finish
    "projects": True,
    "key_controls": False,
    "provider_controls": False,
    "alert_rules": False,
}


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("prompture")
    except Exception:
        return "0"


def info() -> dict[str, Any]:
    return {
        "service": "prompture",
        "mode": "local",
        "version": _version(),
        "api_version": COMPANION_API_VERSION,
        "features": dict(FEATURES),
        "capabilities": dict(CAPABILITIES),
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

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        if url.path == "/v1/companion/info":
            return self._json(200, info())
        if not self._authorized():
            return self._json(401, {"detail": "Missing or wrong companion token (see ~/.prompture/companion.json)."})
        if url.path == "/v1/spend":
            period = query.get("period", "day")
            if period not in PERIODS:
                return self._json(422, {"detail": "period must be day, week or month"})
            return self._json(200, summarize_spend(self.server.ledger.rows(period), period))
        if url.path == "/v1/limits":
            return self._json(200, self.server.limits(accounts=query.get("accounts", "true") != "false"))
        if url.path == "/v1/alerts":
            return self._json(200, [])
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
    ) -> None:
        super().__init__(("127.0.0.1", port), _Handler)
        self.ledger = ledger or LedgerSource()
        self.token = token or secrets.token_urlsafe(32)
        self.bus = bus or get_bus()
        self.state_path = state_path
        self.stopping = threading.Event()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def limits(self, *, accounts: bool = True) -> dict[str, Any]:
        from datetime import datetime, timezone

        body: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "keys": [],
            "providers": provider_limits(self.ledger.rate_limits()),
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
        threading.Thread(target=self.ledger.tail, args=(self.bus, self.stopping), daemon=True).start()
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread

    def run(self) -> None:
        """Serve until interrupted, advertising the address in the state file."""
        if self.state_path is not None:
            _write_state(self.state_path, {"url": self.url, "token": self.token, "pid": os.getpid()})
        threading.Thread(target=self.ledger.tail, args=(self.bus, self.stopping), daemon=True).start()
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
    """The state of a companion that is already up and answering, else ``None``."""
    import urllib.request

    state = read_state(path)
    if not state:
        return None
    try:
        with urllib.request.urlopen(
            f"{state['url']}/v1/companion/info", timeout=timeout
        ) as resp:  # loopback URL we wrote ourselves
            if resp.status == 200:
                return state
    except OSError:
        return None
    return None
