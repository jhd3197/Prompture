"""The local companion server: the companion API over Prompture's own ledger.

``prompture companion`` runs this. It speaks the same endpoints as
prompture-hub's companion API, so a desktop companion can show this machine's
Prompture usage with no hub at all:

- ``GET /v1/companion/info`` — public; version, API version, features.
- ``GET /v1/live`` — Server-Sent Events (calls as they finish).
- ``GET /v1/spend?period=day|week|month`` and ``GET /v1/limits``.
- ``GET /v1/alerts`` — always empty; alert rules live in the hub.
- ``POST /v1/shutdown`` — stop cleanly (CLI configs put back first).

With an :class:`~.automations.Automations` it also runs queued coding-agent
steps one after another (``/v1/automations``).

With a :class:`~.coding_tools.CodingToolSource` it also counts the calls local
coding agents (Claude Code, Codex, Kimi Code, Gemini CLI, …) log on disk, adds
their plan windows to the limits, and serves ``GET /v1/tools?period=`` — each
agent's usage plus which agents are installed.

With a :class:`~.router.Router` it routes coding CLIs: ``/tools/<tool>/…``
takes Claude Code's and Codex's model traffic (passed through to the vendor,
or sent to a Prompture model), and ``/v1/router`` turns routing, request
rules, presets and Claude Code's live-state hooks (``POST /v1/hooks``) on and
off. ``/v1/router/calls`` lists what each call cost and why it went where it
did, ``/v1/router/savings`` adds them up per tool, project and rule, and
``/v1/router/tasks`` shows each session's destination, failures and budget.

With a :class:`~.memory.MemoryService`, ``/v1/memory`` keeps each project's
notes (decisions, conventions, commands, verified fixes), shows what each
session was given, and lists skill proposals mined from finished tasks.

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
from urllib.parse import parse_qs, unquote, urlparse

from .automations import AutomationError, Automations, roadmap_steps
from .capacity import PlanCapacity
from .coding_tools import CodingToolSource
from .live import LiveBus, get_bus, sse_event
from .local import LedgerSource
from .memory import MemoryService
from .router import Router
from .summary import COMPANION_API_VERSION, PERIODS, UsageRow, account_limits, provider_limits, summarize_spend
from .tool_routing import RoutingError, ToolRouting

logger = logging.getLogger("prompture.companion")

STATE_FILE = Path.home() / ".prompture" / "companion.json"
#: The port ``prompture companion`` asks for first, so routed CLIs find it at the same address.
DEFAULT_PORT = 47811
HEARTBEAT_SECONDS = 15.0

#: What a client can rely on from this server. The hub advertises more.
FEATURES = {
    "live": "/v1/live",
    "limits": "/v1/limits",
    "spend": "/v1/spend",
    "alerts": "/v1/alerts",
    "tools": "/v1/tools",
    "activity": "/v1/activity",
    "recent": "/v1/recent",
    "shutdown": "/v1/shutdown",
}
CAPABILITIES = {
    "running_calls": False,  # the ledger only sees calls after they finish
    "projects": True,
    "key_controls": False,
    "provider_controls": False,
    "alert_rules": False,
    "coding_tools": False,
    "activity": True,
    "automations": False,
}


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("prompture")
    except Exception:
        return "0"


def info(
    coding_tools: bool = False, automations: bool = False, router: bool = False, memory: bool = False
) -> dict[str, Any]:
    features = dict(FEATURES)
    if automations:
        features["automations"] = "/v1/automations"
    if router:
        features["router"] = "/v1/router"
    if memory:
        features["memory"] = "/v1/memory"
    return {
        "service": "prompture",
        "mode": "local",
        "version": _version(),
        "api_version": COMPANION_API_VERSION,
        "features": features,
        "capabilities": {
            **CAPABILITIES,
            # Coding agents' turns and routed requests show while they run.
            "running_calls": coding_tools or router,
            "coding_tools": coding_tools,
            "agent_turns": coding_tools,
            "automations": automations,
            "router": router,
            # Call records, savings, presets, fallback and task controls (/v1/router/calls, …).
            "router_calls": router,
            "gemini_routing": router,
            "memory": memory,
        },
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

    def _body(self) -> Any:
        """The request's JSON body; raises ``ValueError`` when it isn't JSON."""
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _routed(self, method: str) -> bool:
        """Hand ``/tools/…`` to the router; ``True`` when it took the request."""
        url = urlparse(self.path)
        if not url.path.startswith("/tools/") or self.server.router is None:
            return False
        self.server.router.handle(self, method, url.path, url.query)
        return True

    def do_PUT(self) -> None:
        if not self._routed("PUT"):
            self._json(404, {"detail": "Not found"})

    def do_DELETE(self) -> None:
        if self._routed("DELETE"):
            return
        url = urlparse(self.path)
        if not self._authorized():
            return self._json(401, {"detail": "Missing or wrong companion token (see ~/.prompture/companion.json)."})
        parts = [unquote(p) for p in url.path.strip("/").split("/")]
        memory = self.server.memory
        if memory is not None and len(parts) == 6 and parts[:3] == ["v1", "memory", "projects"] and parts[4] == "facts":
            if memory.memory.delete(parts[3], parts[5]):
                return self._json(200, {"deleted": parts[5]})
            return self._json(404, {"detail": "No such note."})
        self._json(404, {"detail": "Not found"})

    def do_POST(self) -> None:
        if self._routed("POST"):
            return
        url = urlparse(self.path)
        if not self._authorized():
            return self._json(401, {"detail": "Missing or wrong companion token (see ~/.prompture/companion.json)."})
        if url.path.startswith("/v1/automations") and self.server.automations is not None:
            try:
                body = self._body()
            except ValueError:
                return self._json(400, {"detail": "Body must be JSON."})
            return self._automations_post(url.path, body if isinstance(body, dict) else {})
        if url.path == "/v1/tools/claude-plan" and self.server.coding_tools is not None:
            try:
                body = self._body()
            except ValueError:
                return self._json(400, {"detail": "Body must be JSON."})
            if not isinstance(body, dict) or not isinstance(body.get("enabled"), bool):
                return self._json(422, {"detail": 'Send {"enabled": true|false}.'})
            self.server.coding_tools.set_claude_plan_usage(body["enabled"])
            return self._json(200, {"claude_plan_usage": self.server.coding_tools.plan_usage})
        if url.path == "/v1/hooks" and self.server.coding_tools is not None:
            try:
                body = self._body()
            except ValueError:
                return self._json(400, {"detail": "Body must be JSON."})
            fields = [body.get(k) if isinstance(body, dict) else None for k in ("agent", "event", "session")]
            if not all(isinstance(f, str) and f for f in fields):
                return self._json(422, {"detail": "Send agent, event and session."})
            agent, event, session = (str(f) for f in fields)
            project = body.get("project") if isinstance(body.get("project"), str) else None
            if project:
                self.server.remember_project(agent, session, project)
            self.server.coding_tools.hook_event(self.server.bus, agent, event, session)
            if self.server.router is not None:
                self.server.router.hook(agent, event, session, project)
            answer: dict[str, Any] = {"ok": True}
            memory = self.server.memory
            if memory is not None:
                cwd = body.get("cwd") if isinstance(body.get("cwd"), str) else None
                memory.remember_folder(project, cwd)
                if event == "UserPromptSubmit":
                    prompt = body.get("prompt") if isinstance(body.get("prompt"), str) else ""
                    context = memory.inject(agent, session, project, prompt, via="hook")
                    if context:
                        answer["context"] = context
            return self._json(200, answer)
        if url.path == "/v1/shutdown":
            # How an app that started the companion stops it: routing is put back first,
            # which a killed process can't do.
            self._json(202, {"stopping": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return None
        if url.path.startswith("/v1/memory") and self.server.memory is not None:
            try:
                body = self._body()
            except ValueError:
                return self._json(400, {"detail": "Body must be JSON."})
            return self._memory_post(url.path, body if isinstance(body, dict) else {})
        if url.path.startswith("/v1/router/") and self.server.tool_routing is not None:
            try:
                body = self._body()
            except ValueError:
                return self._json(400, {"detail": "Body must be JSON."})
            return self._router_post(url.path, body if isinstance(body, dict) else {})
        return self._json(404, {"detail": "Not found"})

    def _router_post(self, path: str, body: dict[str, Any]) -> None:
        routing, router = self.server.tool_routing, self.server.router
        assert routing is not None and router is not None
        parts = path.strip("/").split("/")[2:]  # after v1/router
        try:
            if len(parts) == 4 and parts[0] == "sessions" and parts[3] == "switch":
                to = body.get("to")
                if not isinstance(to, str) or not to.strip():
                    return self._json(
                        422, {"detail": 'Send {"to": "native:<model>" | "<provider/model>" | "original"}.'}
                    )
                switched = router.switch(parts[1], parts[2], to.strip(), now=body.get("now") is True)
                if switched is None:
                    return self._json(404, {"detail": "No such task; it may have ended."})
                return self._json(200, {"task": switched.to_dict()})
            if len(parts) == 4 and parts[0] == "sessions" and parts[3] == "escalate":
                decision = router.escalate(parts[1], parts[2], str(body.get("reason") or ""))
                task = router.policy.find(parts[1], parts[2])
                if task is None:
                    return self._json(404, {"detail": "No such task; it may have ended."})
                return self._json(200, {"escalated": decision is not None, "task": task.to_dict()})
            if parts == ["routes"]:
                router.routes.save(body)
            elif parts == ["hooks"] and isinstance(body.get("enabled"), bool):
                routing.set_hooks(body["enabled"])
            elif len(parts) == 2 and parts[0] == "tools" and isinstance(body.get("enabled"), bool):
                routing.set_enabled(parts[1], body["enabled"], self.server.url)
            else:
                return self._json(422, {"detail": 'Send {"enabled": true|false}, or rules to /v1/router/routes.'})
        except RoutingError as exc:
            return self._json(409, {"detail": str(exc)})
        return self._json(200, self.server.router_state())

    def do_GET(self) -> None:
        if self._routed("GET"):
            return
        url = urlparse(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        if url.path == "/v1/companion/info":
            return self._json(
                200,
                info(
                    self.server.coding_tools is not None,
                    self.server.automations is not None,
                    self.server.router is not None,
                    self.server.memory is not None,
                ),
            )
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
        if url.path == "/v1/recent":
            try:
                minutes = min(24 * 60, max(1, int(query.get("minutes", "30"))))
            except ValueError:
                return self._json(422, {"detail": "minutes must be an integer"})
            return self._json(200, self.server.recent(minutes))
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
        if url.path == "/v1/router" and self.server.tool_routing is not None:
            return self._json(200, self.server.router_state())
        if url.path.startswith("/v1/router/") and self.server.router is not None:
            return self._router_get(url.path, query)
        if url.path.startswith("/v1/memory") and self.server.memory is not None:
            return self._memory_get(url.path, query)
        if url.path.startswith("/v1/automations") and self.server.automations is not None:
            return self._automations_get(url.path, query)
        return self._json(404, {"detail": "Not found"})

    def _memory_post(self, path: str, body: dict[str, Any]) -> None:
        memory = self.server.memory
        assert memory is not None
        parts = [unquote(p) for p in path.strip("/").split("/")][2:]  # after v1/memory
        if parts == ["settings"]:
            return self._json(200, memory.save_settings(body))
        if len(parts) >= 3 and parts[0] == "projects":
            project = parts[1]
            if parts[2:] == ["facts"]:
                try:
                    fact = memory.memory.add(
                        project,
                        str(body.get("content") or ""),
                        kind=str(body.get("kind") or "fact"),
                        source=body.get("source") if isinstance(body.get("source"), str) else None,
                        verified=bool(body.get("verified")),
                        pinned=bool(body.get("pinned")),
                        agent=str(body.get("agent") or "you"),
                    )
                except ValueError as exc:
                    return self._json(422, {"detail": str(exc)})
                return self._json(200, _fact(fact))
            if len(parts) == 4 and parts[2] == "facts":
                updated = memory.memory.update(project, parts[3], **body)
                return self._json(200, _fact(updated)) if updated else self._json(404, {"detail": "No such note."})
            if len(parts) == 5 and parts[2] == "skills" and parts[4] == "save":
                try:
                    return self._json(200, {"path": memory.save_skill(project, parts[3])})
                except KeyError as exc:
                    return self._json(404, {"detail": str(exc.args[0])})
                except (FileNotFoundError, FileExistsError, OSError) as exc:
                    return self._json(409, {"detail": str(exc)})
        return self._json(404, {"detail": "Not found"})

    def _memory_get(self, path: str, query: dict[str, str]) -> None:
        memory = self.server.memory
        assert memory is not None
        parts = [unquote(p) for p in path.strip("/").split("/")][2:]
        if not parts:
            return self._json(200, {"settings": memory.settings(), "projects": memory.projects()})
        if parts == ["injections"]:
            return self._json(200, memory.injections(query.get("project"), query.get("session")))
        if len(parts) == 2 and parts[0] == "projects":
            project = parts[1]
            return self._json(
                200,
                {
                    "project": project,
                    "folder": memory.folder(project),
                    "facts": [_fact(f) for f in memory.memory.notes(project)],
                    "injections": memory.injections(project, limit=20),
                    "skills": memory.skills(project),
                },
            )
        return self._json(404, {"detail": "Not found"})

    def _router_get(self, path: str, query: dict[str, str]) -> None:
        from .calls import summarize_calls
        from .summary import window_start

        router = self.server.router
        assert router is not None
        parts = path.strip("/").split("/")[2:]  # after v1/router
        period = query.get("period", "day")
        if period not in PERIODS:
            return self._json(422, {"detail": "period must be day, week or month"})
        start = window_start(period, offset_minutes=_tz_offset(query))
        if parts == ["savings"]:
            return self._json(200, {"period": period, **summarize_calls(router.calls.calls(start), start)})
        if parts == ["tasks"]:
            return self._json(200, [t.to_dict() for t in router.policy.tasks()])
        if parts == ["calls"]:
            try:
                limit = min(1000, max(1, int(query.get("limit", "200"))))
            except ValueError:
                return self._json(422, {"detail": "limit must be an integer"})
            calls = router.calls.calls(start)
            for key in ("tool", "session", "project", "route"):
                if query.get(key):
                    calls = [c for c in calls if getattr(c, key) == query[key]]
            if query.get("routed") == "true":
                calls = [c for c in calls if c.route != "passthrough"]
            return self._json(200, [c.to_dict() for c in reversed(calls[-limit:])])
        if len(parts) == 2 and parts[0] == "calls":
            call = router.calls.get(parts[1])
            if call is None:
                return self._json(404, {"detail": "No such call."})
            task = router.policy.find(call.tool, call.session) if call.session else None
            return self._json(200, {**call.to_dict(), "task": task.to_dict() if task else None})
        return self._json(404, {"detail": "Not found"})

    def _automations_get(self, path: str, query: dict[str, str]) -> None:
        auto = self.server.automations
        assert auto is not None
        parts = path.strip("/").split("/")[2:]  # after v1/automations
        try:
            if not parts:
                return self._json(200, auto.state())
            if parts == ["roadmap"]:
                return self._json(200, {"steps": roadmap_steps(query.get("cwd", ""))})
            if len(parts) == 2 and parts[0] == "runs":
                return self._json(200, auto.get_run(parts[1]))
            if len(parts) == 5 and parts[0] == "runs" and parts[2] == "steps" and parts[4] == "log":
                return self._json(200, {"lines": auto.log(parts[1], parts[3])})
        except AutomationError as exc:
            return self._json(exc.status, {"detail": exc.detail})
        return self._json(404, {"detail": "Not found"})

    def _automations_post(self, path: str, body: dict[str, Any]) -> None:
        auto = self.server.automations
        assert auto is not None
        parts = path.strip("/").split("/")[2:]
        try:
            if not parts:
                return self._json(200, auto.start(body))
            if len(parts) == 2 and parts[0] == "current":
                action = parts[1]
                if action == "steps":
                    return self._json(200, auto.set_steps(body.get("steps")))
                if action == "answer":
                    return self._json(200, auto.answer(str(body.get("text") or "")))
                if action in ("pause", "resume", "skip", "stop"):
                    return self._json(200, getattr(auto, action)())
        except AutomationError as exc:
            return self._json(exc.status, {"detail": exc.detail})
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
    # On Windows, SO_REUSEADDR would let two servers share a port.
    allow_reuse_address = os.name != "nt"

    def __init__(
        self,
        ledger: LedgerSource | None = None,
        *,
        port: int = 0,
        token: str | None = None,
        bus: LiveBus | None = None,
        state_path: Path | None = STATE_FILE,
        coding_tools: CodingToolSource | None = None,
        automations: Automations | None = None,
        tool_routing: ToolRouting | None = None,
        router: Router | None = None,
        memory: MemoryService | None = None,
    ) -> None:
        super().__init__(("127.0.0.1", port), _Handler)
        self.ledger = ledger or LedgerSource()
        self.token = token or secrets.token_urlsafe(32)
        self.bus = bus or get_bus()
        self.state_path = state_path
        self.coding_tools = coding_tools
        self.automations = automations
        self.tool_routing = tool_routing
        self._projects: dict[tuple[str, str], str] = {}
        self.memory = memory
        if router is None and tool_routing is not None:
            router = Router(self.bus, upstream=tool_routing.upstream, project_for=self.project_for, memory=memory)
        self.router = router
        #: Plan windows read from replies through the router (fresher than the CLIs' logs).
        self.capacity = PlanCapacity()
        if router is not None:
            router.capacity = self.capacity
        self.stopping = threading.Event()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def remember_project(self, agent: str, session: str, project: str) -> None:
        """A hook named the folder a session works in."""
        self._projects[(agent, session)] = project

    def project_for(self, agent: str, session: str) -> str | None:
        """The project a coding-agent session works in: from its hooks, else its logs."""
        project = self._projects.get((agent, session))
        if project is None and self.coding_tools is not None:
            try:
                project = self.coding_tools.project_for(agent, session)
            except Exception:
                logger.debug("project lookup failed", exc_info=True)
            if project:
                self._projects[(agent, session)] = project
        return project

    def router_state(self) -> dict[str, Any]:
        """``/v1/router``: each CLI's routing switch, the request rules, and the hooks switch."""
        assert self.tool_routing is not None and self.router is not None
        from .router import KINDS
        from .routing_policy import BACKGROUND_KINDS, NATIVE_ONLY_KINDS, PRESETS

        return {
            **self.tool_routing.status(self.url),
            "routes": self.router.routes.data(),
            "kinds": list(KINDS),
            "background_kinds": sorted(BACKGROUND_KINDS - NATIVE_ONLY_KINDS),
            "presets": {name: dict(kinds) for name, kinds in PRESETS.items()},
            "settings": self.router.policy.settings(),
        }

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
        for target, snap in self.capacity.limits().items():
            known = limits.get(target)
            if not known or float(known.get("observed_at") or 0) <= float(snap.get("observed_at") or 0):
                limits[target] = snap
        return limits

    def _tail(self) -> None:
        threading.Thread(target=self.ledger.tail, args=(self.bus, self.stopping), daemon=True).start()
        if self.coding_tools is not None:
            threading.Thread(target=self.coding_tools.tail, args=(self.bus, self.stopping), daemon=True).start()

    def recent(self, minutes: int = 30, limit: int = 500) -> list[dict[str, Any]]:
        """Calls that finished in the last ``minutes``, oldest first, as ``request.finished`` payloads.

        Lets a client that just connected fill its recent-calls view before new
        calls stream in over ``/v1/live``.
        """
        from datetime import datetime, timedelta, timezone

        since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        events = self.ledger.events_since(since, limit)
        if self.coding_tools is not None:
            events += self.coding_tools.events_since(since, limit)
        events.sort(key=lambda e: str(e.get("ts") or ""))
        return [{"type": "request.finished", **e} for e in events[-limit:]]

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
            s = d["sources"].setdefault(name, {"name": name, "requests": 0, "tokens": 0, "cost_usd": 0.0})
            s["requests"] += requests
            s["tokens"] += tokens
            s["cost_usd"] += cost

        for day, t in self.ledger.daily(since, offset_minutes).items():
            add(day, t["requests"], t["tokens"], t["cost_usd"], "Prompture")
        if self.coding_tools is not None:
            names = self.coding_tools.names
            for day, t in self.coding_tools.usage.daily(since, offset_minutes).items():
                for agent, a in t["agents"].items():
                    add(day, a["requests"], a["tokens"], a["cost_usd"], names.get(agent, agent))
        out = []
        for day in sorted(merged):
            if day < first_day.isoformat():
                continue
            d = merged[day]
            d["cost_usd"] = round(d["cost_usd"], 6)
            d["sources"] = sorted(
                ({**s, "cost_usd": round(s["cost_usd"], 6)} for s in d["sources"].values()), key=lambda s: -s["tokens"]
            )
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
        """Serve until interrupted, advertising the address in the state file.

        SIGTERM (a plain ``kill``) stops it like Ctrl+C does, so routed CLIs
        are put back; only a forced kill skips that.
        """
        _stop_on_sigterm()
        if self.state_path is not None:
            _write_state(self.state_path, {"url": self.url, "token": self.token, "pid": os.getpid()})
        if self.tool_routing is not None:
            for problem in self.tool_routing.apply_enabled(self.url):
                logger.warning("Routing: %s", problem)
            try:
                self.tool_routing.refresh_hooks()
            except RoutingError as exc:
                logger.warning("Hooks: %s", exc)
        if self.memory is not None:
            # The first prompt of a session waits on this; load it before anyone asks.
            threading.Thread(target=self.memory.warm, daemon=True).start()
        self._tail()
        try:
            self.serve_forever()
        finally:
            self.shutdown_companion()

    def shutdown_companion(self) -> None:
        self.stopping.set()
        if self.tool_routing is not None:
            self.tool_routing.restore_all()  # never leave a CLI pointing at a stopped companion
        if self.automations is not None:
            self.automations.shutdown()
        if self.state_path is not None:
            state = read_state(self.state_path)
            if state and state.get("pid") == os.getpid():
                with contextlib.suppress(OSError):
                    self.state_path.unlink()
        self.server_close()


def _fact(fact: Any) -> dict[str, Any]:
    """A project note as the API shows it."""
    meta = fact.metadata or {}
    return {
        "id": fact.id,
        "kind": fact.kind,
        "content": fact.content,
        "source": meta.get("source"),
        "verified": bool(meta.get("verified")),
        "pinned": bool(meta.get("pinned")),
        "agent": meta.get("agent"),
        "ts": fact.ts,
        "updated": meta.get("updated") or fact.ts,
    }


def _stop_on_sigterm() -> None:
    """Turn SIGTERM into KeyboardInterrupt, so ``run()``'s cleanup runs (main thread only)."""
    import signal

    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGTERM"):
        return

    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGTERM, interrupt)


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
