"""The companion's router: coding CLIs send their model traffic through Prompture.

A CLI pointed at ``<companion>/tools/<tool>`` (Claude Code through
``ANTHROPIC_BASE_URL``, Codex through a ``model_providers`` entry; see
:mod:`.tool_routing`) reaches this module instead of its vendor. Each request
is either:

- **passed through** (the default): forwarded unchanged to the vendor with the
  CLI's own credentials, so API keys and subscription logins keep working, and
  streamed back as it arrives;
- sent to another **native** model: still the vendor and the CLI's login, but
  a different model (a preset putting titles on the vendor's small model); or
- **routed** to any Prompture model (``ollama/qwen3``, ``combo/…``,
  ``auto/cheap``, an alias) through Prompture's drivers, translated to and from
  the CLI's wire format by :mod:`prompture.gateway`.

Which one is decided per request by :class:`~.routing_policy.RoutePolicy`
from ``~/.prompture/routes.json``::

    {"preset": "balanced",
     "tools": {"claude-code": {
        "models": {"claude-haiku-*": "ollama/qwen3:8b"},
        "kinds": {"background": "ollama/qwen3:8b"},
        "preset": "economy"}},
     "projects": {"my-app": {"preset": "quality"}},
     "fallback": {"models": ["auto/cheap"], "allow_paid": false},
     "escalation": {"enabled": true, "after_failures": 3},
     "budget": {"task_usd": 2.0, "task_attempts": 6}}

``kinds`` matches what a request is for (:func:`anthropic_kind`,
:func:`responses_kind`): ``main`` turns, ``tool_result`` follow-ups, and the
``background`` ones (``title``, ``compaction``) that a small model handles as
well as a big one. Quota checks (``probe``) always reach the vendor.

Every request shows live on the bus: ``request.started`` (with ``tool``,
``kind`` and, when routed, ``routed_to``), ``request.first_token`` and
``request.ended`` (with where it went, its cost and savings), and is recorded
in the :class:`~.calls.CallLog` for ``/v1/router/calls`` and
``/v1/router/savings``. Usage isn't added to the spend views here, so nothing
is counted twice: passed-through calls are counted from the CLI's own logs,
routed ones by the usage ledger every driver call writes to. Routed replies
carry a ``msg_prompture_`` / ``resp_prompture_`` id, which the log readers skip.

Only localhost may call the router: requests naming another host, or coming
from a web page (an ``Origin`` header), are refused.
"""

from __future__ import annotations

import contextlib
import fnmatch
import http.client
import json
import logging
import re
import ssl
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .calls import CallLog, RoutedCall, Usage, UsageSniffer, billing_of, settle
from .live import LiveBus, new_request_id
from .routing_policy import (
    BACKGROUND_KINDS,
    PASS,
    PRESET_NAMES,
    TASK_KINDS,
    Decision,
    Needs,
    RoutePolicy,
    Task,
)

logger = logging.getLogger("prompture.companion.router")

ROUTES_FILE = Path.home() / ".prompture" / "routes.json"
#: A route target meaning "the vendor, unchanged".
PASSTHROUGH = "passthrough"
#: Id prefixes of replies Prompture produced; the coding-agent log readers skip them.
MESSAGE_PREFIX = "msg_prompture_"
RESPONSE_PREFIX = "resp_prompture_"
UPSTREAM_TIMEOUT = 600.0

ANTHROPIC_UPSTREAM = "https://api.anthropic.com"
OPENAI_UPSTREAM = "https://api.openai.com/v1"
#: Where Codex sends requests when signed in with ChatGPT.
CHATGPT_UPSTREAM = "https://chatgpt.com/backend-api/codex"

KINDS = ("main", "tool_result", "title", "probe", "compaction")
#: Vendor answers a fallback may step in for: rate limits, overload, outages.
FALLBACK_STATUSES = {429, 500, 502, 503, 504, 529}

_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
}


@dataclass(frozen=True)
class Tool:
    """A CLI the router serves. ``dialect`` is the wire format it speaks."""

    id: str
    name: str
    agent: str
    dialect: str  # "anthropic" | "openai"

    @property
    def vendor(self) -> str:
        """The Prompture provider of the CLI's own vendor."""
        return "claude" if self.dialect == "anthropic" else "openai"


TOOLS: dict[str, Tool] = {
    "claude-code": Tool("claude-code", "Claude Code", "claude", "anthropic"),
    "codex": Tool("codex", "Codex", "codex", "openai"),
}


# ------------------------------------------------------------------ request kinds

_TITLE_HINTS = (
    "new conversation topic",
    "isnewtopic",
    "title for",
    "in under 50 characters",
    "5-10 word title",
    "summarize this coding conversation",
)
_COMPACTION_HINTS = (
    "create a detailed summary of the conversation",
    "summary of the conversation so far",
    "your task is to create a detailed summary",
)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("type") in ("text", "input_text")
        )
    return ""


def anthropic_kind(body: dict[str, Any]) -> str:
    """What an Anthropic Messages request is for: one of :data:`KINDS`."""
    if body.get("max_tokens") == 1:
        return "probe"  # a quota or connectivity check
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    last = messages[-1] if messages and isinstance(messages[-1], dict) else {}
    content = last.get("content")
    if (
        last.get("role") == "user"
        and isinstance(content, list)
        and content
        and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    ):
        return "tool_result"
    prompt = _text(content).lower()
    if any(h in prompt for h in _COMPACTION_HINTS):
        return "compaction"
    system = _text(body.get("system")).lower()
    if any(h in system or h in prompt for h in _TITLE_HINTS):
        return "title"
    return "main"


def responses_kind(body: dict[str, Any], path: str = "") -> str:
    """What an OpenAI Responses request is for: one of :data:`KINDS`."""
    if path.endswith("/compact"):
        return "compaction"
    items = body.get("input") if isinstance(body.get("input"), list) else []
    last = items[-1] if items and isinstance(items[-1], dict) else {}
    if last.get("type") in ("function_call_output", "custom_tool_call_output"):
        return "tool_result"
    return "main"


# ------------------------------------------------------------------ sessions and billing

_SESSION_IN_USER_ID = re.compile(r"_session_([\w-]+)$")


def session_of(tool: Tool, headers: Any, body: dict[str, Any]) -> str | None:
    """The CLI session (task) a request belongs to, from what the CLI sends anyway."""
    if tool.dialect == "anthropic":
        sid = headers.get("x-claude-code-session-id")
        if sid:
            return str(sid)
        user_id = (body.get("metadata") or {}).get("user_id") if isinstance(body.get("metadata"), dict) else None
        if isinstance(user_id, str):
            if user_id.startswith("{"):
                with contextlib.suppress(ValueError, AttributeError):
                    sid = json.loads(user_id).get("session_id")
                    if sid:
                        return str(sid)
            found = _SESSION_IN_USER_ID.search(user_id)
            if found:
                return found.group(1)
        return None
    sid = headers.get("session_id") or headers.get("conversation_id") or body.get("prompt_cache_key")
    return str(sid) if sid else None


def billing_of_request(tool: Tool, headers: Any) -> str:
    """How the CLI's own path bills this request, from the login it sends: its plan, or an API key."""
    if tool.dialect == "anthropic":
        auth = str(headers.get("Authorization") or "")
        if auth.lower().startswith("bearer sk-ant-oat"):
            return "subscription"
        return "api" if headers.get("x-api-key") or auth else "unknown"
    return "subscription" if headers.get("chatgpt-account-id") else "api"


# ------------------------------------------------------------------ routes


class Routes:
    """``routes.json``: rules, presets, fallback, escalation and budgets. See the module docstring."""

    def __init__(self, path: str | Path | None = ROUTES_FILE) -> None:
        self.path = Path(path) if path else None
        self._cache: tuple[float, dict[str, Any]] | None = None
        self._memory: dict[str, Any] = {"tools": {}}
        self._lock = threading.Lock()

    def data(self) -> dict[str, Any]:
        if self.path is None:
            return self._memory
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return {"tools": {}}
        with self._lock:
            if self._cache is None or self._cache[0] != mtime:
                try:
                    loaded = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    loaded = {}
                self._cache = (mtime, normalize_routes(loaded))
            return self._cache[1]

    def save(self, data: Any) -> dict[str, Any]:
        clean = normalize_routes(data)
        if self.path is None:
            self._memory = clean
            return clean
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(clean, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        with self._lock:
            self._cache = None
        return clean

    def target(self, tool: str, model: str, kind: str) -> str | None:
        """The Prompture model a request goes to by the tool's own rules, or ``None`` (unchanged)."""
        rules = self.data()["tools"].get(tool) or {}
        kinds = rules.get("kinds") or {}
        target = kinds.get(kind) or (kinds.get("background") if kind in BACKGROUND_KINDS else None)
        if not target:
            for pattern, to in (rules.get("models") or {}).items():
                if fnmatch.fnmatchcase(model.lower(), pattern.lower()):
                    target = to
                    break
        return None if not target or target == PASSTHROUGH else target


def _tool_rules(rules: Any) -> dict[str, Any]:
    rules = rules if isinstance(rules, dict) else {}
    models = {
        str(p).strip(): str(t).strip()
        for p, t in (rules.get("models") or {}).items()
        if str(p).strip() and isinstance(t, str) and t.strip()
    }
    kinds = {
        k: t.strip()
        for k, t in (rules.get("kinds") or {}).items()
        if (k in KINDS or k == "background") and isinstance(t, str) and t.strip()
    }
    out: dict[str, Any] = {}
    if models or kinds:
        out = {"models": models, "kinds": kinds}
    if rules.get("preset") in PRESET_NAMES:
        out["preset"] = rules["preset"]
    return out


def _number(value: Any, *, integer: bool = False) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return int(value) if integer else float(value)


def normalize_routes(data: Any) -> dict[str, Any]:
    """Keep only well-formed settings: string patterns and targets, known kinds and presets."""
    data = data if isinstance(data, dict) else {}
    tools: dict[str, Any] = {}
    raw = data.get("tools")
    for tool, rules in raw.items() if isinstance(raw, dict) else ():
        if isinstance(tool, str) and (clean := _tool_rules(rules)):
            tools[tool] = clean
    out: dict[str, Any] = {"tools": tools}
    if data.get("preset") in PRESET_NAMES:
        out["preset"] = data["preset"]
    projects: dict[str, Any] = {}
    raw_projects = data.get("projects")
    for name, proj in raw_projects.items() if isinstance(raw_projects, dict) else ():
        if not isinstance(name, str) or not name.strip() or not isinstance(proj, dict):
            continue
        entry: dict[str, Any] = {}
        if proj.get("preset") in PRESET_NAMES:
            entry["preset"] = proj["preset"]
        ptools = {t: c for t, r in (proj.get("tools") or {}).items() if isinstance(t, str) and (c := _tool_rules(r))}
        if ptools:
            entry["tools"] = ptools
        if entry:
            projects[name.strip()] = entry
    if projects:
        out["projects"] = projects
    fb = data.get("fallback")
    if isinstance(fb, dict):
        models = [m.strip() for m in fb.get("models") or [] if isinstance(m, str) and m.strip()]
        entry = {"models": models, "allow_paid": bool(fb.get("allow_paid"))}
        if models or entry["allow_paid"]:
            out["fallback"] = entry
    esc = data.get("escalation")
    if isinstance(esc, dict):
        entry = {}
        if isinstance(esc.get("enabled"), bool):
            entry["enabled"] = esc["enabled"]
        for key in ("after_failures", "after_repeats"):
            if (n := _number(esc.get(key), integer=True)) is not None:
                entry[key] = n
        if isinstance(esc.get("to"), str) and esc["to"].strip():
            entry["to"] = esc["to"].strip()
        if entry:
            out["escalation"] = entry
    budget = data.get("budget")
    if isinstance(budget, dict):
        entry = {}
        if (usd := _number(budget.get("task_usd"))) is not None and usd > 0:
            entry["task_usd"] = usd
        if (n := _number(budget.get("task_attempts"), integer=True)) is not None:
            entry["task_attempts"] = n
        if budget.get("on_exceed") in ("native", "stop"):
            entry["on_exceed"] = budget["on_exceed"]
        if entry:
            out["budget"] = entry
    return out


# ------------------------------------------------------------------ plumbing


def record_usage(driver: Any, outcome: Any, elapsed_ms: float) -> None:
    """Write a routed call to the usage ledger, like a driver's own hooked calls do.

    The gateway calls the driver's plain methods, which don't record; the
    ledger is where the companion's spend views (and budgets) read it back.
    """
    record = getattr(driver, "_auto_record_usage", None)
    if record is None:
        return
    error = outcome.error if isinstance(outcome.error, Exception) else None
    record({"meta": outcome.meta}, elapsed_ms, status="error" if error else "success", error=error)


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    if "chunked" in handler.headers.get("Transfer-Encoding", "").lower():
        chunks = []
        while True:
            size = int(handler.rfile.readline().split(b";")[0].strip() or b"0", 16)
            if size == 0:
                handler.rfile.readline()
                break
            chunks.append(handler.rfile.read(size))
            handler.rfile.readline()
        return b"".join(chunks)
    length = int(handler.headers.get("Content-Length") or 0)
    return handler.rfile.read(length) if length else b""


def _error_body(dialect: str, message: str, kind: str = "api_error") -> dict[str, Any]:
    if dialect == "anthropic":
        return {"type": "error", "error": {"type": kind, "message": message}}
    return {"error": {"message": message, "type": kind, "code": None}}


def _send_json(handler: BaseHTTPRequestHandler, status: int, body: Any) -> None:
    raw = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _start_stream(handler: BaseHTTPRequestHandler) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.close_connection = True


def local_request(handler: BaseHTTPRequestHandler) -> bool:
    """Whether a request came from a local program, not a web page or a rebound hostname."""
    if handler.headers.get("Origin"):
        return False
    host = (handler.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
    return host in ("127.0.0.1", "localhost", "::1")


#: Stream events that mean the reply has really started (content, or a tool call).
_CONTENT_EVENTS = ("event: content_block_start", "event: content_block_delta", "event: response.output")
_FAILED_EVENTS = ("event: error", "event: response.failed")


def _prefetch(frames: Iterator[str], limit: int = 64) -> tuple[list[str], str | None]:
    """Frames up to the first real content: ``(frames, None)``, or ``([], why)`` when the stream failed first.

    The gateway opens every stream with a start event before calling the
    driver, and turns a driver failure into an error event rather than
    raising, so a failed attempt shows here and nothing has reached the CLI.
    """
    head: list[str] = []
    for frame in frames:
        if frame.startswith(_FAILED_EVENTS):
            try:
                data = json.loads(frame.split("data:", 1)[1])
                err = data.get("error") or (data.get("response") or {}).get("error") or {}
                return [], str(err.get("message") or "the model failed")
            except (IndexError, ValueError, AttributeError):
                return [], "the model failed"
        head.append(frame)
        if frame.startswith(_CONTENT_EVENTS) or len(head) >= limit:
            break
    return head, None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _Request:
    """One model request in flight, and what it has done so far."""

    tool: Tool
    suffix: str
    body: dict[str, Any]
    model: str
    kind: str
    session: str | None
    project: str | None
    billing: str  # the original path's
    needs: Needs
    task: Task | None
    decision: Decision
    rid: str
    started: float
    call: RoutedCall


# ------------------------------------------------------------------ router


class Router:
    """Serves ``/tools/<tool>/…`` on the companion; see the module docstring.

    ``upstream(tool_id)`` names a custom vendor URL for a tool (the gateway it
    pointed at before routing was turned on), or ``None`` for the vendor's own.
    ``project_for(agent, session)`` names the project a CLI session works in.
    """

    def __init__(
        self,
        bus: LiveBus,
        routes: Routes | None = None,
        *,
        upstream: Callable[[str], str | None] | None = None,
        driver_for: Callable[[str], Any] | None = None,
        calls: CallLog | None = None,
        project_for: Callable[[str, str], str | None] | None = None,
    ) -> None:
        self.bus = bus
        self.routes = routes or Routes()
        self.upstream = upstream or (lambda tool: None)
        self._driver_for = driver_for
        if calls is None:
            # Kept beside routes.json (``~/.prompture/router/calls.jsonl``); in memory without one.
            calls = CallLog(self.routes.path.parent / "router" / "calls.jsonl" if self.routes.path else None)
        self.calls = calls
        self.project_for = project_for or (lambda agent, session: None)
        self.policy = RoutePolicy(self.routes.data)

    def driver(self, model: str) -> Any:
        if self._driver_for is not None:
            return self._driver_for(model)
        from ..drivers import get_driver_for_model

        return get_driver_for_model(model)

    # -- entry point ----------------------------------------------------------

    def handle(self, handler: BaseHTTPRequestHandler, method: str, path: str, query: str) -> None:
        parts = path.split("/", 3)  # "", "tools", "<tool>", "v1/…"
        tool = TOOLS.get(parts[2]) if len(parts) > 2 else None
        if tool is None:
            return _send_json(handler, 404, {"detail": "Unknown tool; use /tools/claude-code or /tools/codex."})
        if not local_request(handler):
            return _send_json(handler, 403, _error_body(tool.dialect, "Only local programs may use the router."))
        suffix = "/" + parts[3] if len(parts) > 3 else "/"
        raw = _read_body(handler)
        body: dict[str, Any] = {}
        if raw and method == "POST":
            try:
                loaded = json.loads(raw)
                body = loaded if isinstance(loaded, dict) else {}
            except ValueError:
                body = {}
        model = str(body.get("model") or "")
        kind = self._kind(tool, suffix, body) if method == "POST" else None
        if not model or not kind:
            return self._pass(handler, tool, method, suffix, query, raw, None)
        req = self._begin(handler, tool, suffix, body, model, kind)
        if req.decision.stop:
            self._finish(req, Usage(), error=req.decision.reason)
            return _send_json(
                handler, 429, _error_body(tool.dialect, f"Prompture: {req.decision.reason}", "rate_limit_error")
            )
        if req.decision.target:
            return self._route(handler, req, [req.decision.target], query)
        if req.decision.native:
            rewritten = json.dumps({**body, "model": req.decision.native}).encode()
            return self._pass(handler, tool, method, suffix, query, rewritten, req)
        return self._pass(handler, tool, method, suffix, query, raw, req)

    @staticmethod
    def _kind(tool: Tool, suffix: str, body: dict[str, Any]) -> str | None:
        if suffix == "/v1/messages":
            return anthropic_kind(body)
        if suffix.startswith("/v1/responses"):
            return responses_kind(body, suffix)
        if suffix == "/v1/chat/completions":
            return "main"
        return None  # counting tokens, listing models: not a model call

    @staticmethod
    def _routable(suffix: str) -> bool:
        return suffix in ("/v1/messages", "/v1/responses")

    def _begin(
        self, handler: BaseHTTPRequestHandler, tool: Tool, suffix: str, body: dict[str, Any], model: str, kind: str
    ) -> _Request:
        session = session_of(tool, handler.headers, body)
        project = self.project_for(tool.agent, session) if session else None
        billing = billing_of_request(tool, handler.headers)
        needs = Needs.of(tool.dialect, body)
        task = self.policy.task(tool.id, session, project) if kind in TASK_KINDS else None
        decision = PASS
        if self._routable(suffix):
            if kind == "tool_result":
                self.policy.observe(tool.id, tool.dialect, body, task, model)
            decision = self.policy.decide(
                tool.id,
                tool.dialect,
                model,
                kind,
                project=project,
                session=session,
                needs=needs,
                original_billing=billing,
            )
        requested = f"{tool.vendor}/{model}"
        served = decision.target or (f"{tool.vendor}/{decision.native}" if decision.native else requested)
        stream = bool(body.get("stream"))
        started = time.perf_counter()
        rid = self._started(tool, model, suffix, kind, stream, decision.target, decision, project)
        call = RoutedCall(
            id=rid,
            ts=_now(),
            tool=tool.id,
            kind=kind,
            endpoint=suffix,
            requested=requested,
            served=served,
            route=decision.route,
            rule=decision.rule(),
            billing=billing_of(decision.target) if decision.target else billing,
            original_billing=billing,
            session=session,
            project=project,
        )
        return _Request(
            tool, suffix, body, model, kind, session, project, billing, needs, task, decision, rid, started, call
        )

    def _finish(self, req: _Request, usage: Usage, error: str | None = None) -> None:
        """Settle the call's numbers, record it, and tell the bus it ended."""
        call = req.call
        call.status = "error" if error else "ok"
        call.error = error
        call.latency_ms = int((time.perf_counter() - req.started) * 1000)
        task = req.task
        if usage.model and call.route != "routed":
            call.served = f"{req.tool.vendor}/{usage.model}"
        settle(call, usage, vendor=req.tool.vendor, cache_ratio=task.cache_ratio if task else None)
        if task is not None:
            if call.route == "passthrough" and usage.input_tokens and call.status == "ok":
                task.cache_ratio = usage.cache_read_tokens / usage.input_tokens
            if task.served is not None and task.served != call.served and call.status == "ok":
                call.switched = True
            if call.status == "ok":
                task.served = call.served
            self.policy.add_spend(task, call.cost_usd if call.billing == "api" else 0.0)
        if not call.project and call.session:
            call.project = self.project_for(req.tool.agent, call.session)
        try:
            self.calls.add(call)
        except Exception:  # the record is context; never fail the request over it
            logger.debug("router: could not record call", exc_info=True)
        self._ended(req.rid, req.tool, req.started, error, call)

    # -- live events ----------------------------------------------------------

    def _started(
        self,
        tool: Tool,
        model: str,
        suffix: str,
        kind: str | None,
        stream: bool,
        routed_to: str | None,
        decision: Decision | None = None,
        project: str | None = None,
    ) -> str:
        rid = new_request_id()
        vendor = "claude" if tool.dialect == "anthropic" else "openai"
        self.bus.publish(
            "request.started",
            {
                "request_id": rid,
                "key_id": None,
                "key_name": tool.name,
                "model": f"{vendor}/{model}" if model else vendor,
                "routed_to": routed_to or (f"{vendor}/{decision.native}" if decision and decision.native else None),
                "project": project,
                "endpoint": suffix,
                "stream": stream,
                "tool": tool.agent,
                "kind": kind,
                "rule": decision.reason if decision and decision.route != "passthrough" else None,
            },
        )
        return rid

    def _ended(
        self, rid: str, tool: Tool, started: float, error: str | None = None, call: RoutedCall | None = None
    ) -> None:
        self.bus.publish(
            "request.ended",
            {
                "request_id": rid,
                "tool": tool.agent,
                "key_name": tool.name,
                "status": "error" if error else "ok",
                "error": error,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                **(call.event() if call else {}),
            },
        )

    def _first_token(self, req: _Request) -> None:
        ttft = int((time.perf_counter() - req.started) * 1000)
        req.call.ttft_ms = ttft
        self.bus.publish("request.first_token", {"request_id": req.rid, "ttft_ms": ttft})

    # -- pass through ---------------------------------------------------------

    def upstream_url(self, tool: Tool, suffix: str, headers: Any) -> str:
        """Where a passed-through request goes."""
        custom = self.upstream(tool.id)
        if tool.dialect == "anthropic":
            return (custom or ANTHROPIC_UPSTREAM).rstrip("/") + suffix
        rest = suffix[3:] if suffix.startswith("/v1/") else suffix
        if headers.get("chatgpt-account-id"):
            return CHATGPT_UPSTREAM + rest
        return (custom or OPENAI_UPSTREAM).rstrip("/") + rest

    def _pass(
        self,
        handler: BaseHTTPRequestHandler,
        tool: Tool,
        method: str,
        suffix: str,
        query: str,
        raw: bytes,
        req: _Request | None,
    ) -> None:
        url = urlsplit(self.upstream_url(tool, suffix, handler.headers))
        headers = {k: v for k, v in handler.headers.items() if k.lower() not in _HOP_HEADERS}
        headers["Host"] = url.netloc
        headers["Accept-Encoding"] = "identity"  # read as it streams, never recompressed
        if raw or method == "POST":
            headers["Content-Length"] = str(len(raw))
        sniffer = UsageSniffer(tool.dialect)
        conn: http.client.HTTPConnection
        if url.scheme == "https":
            conn = http.client.HTTPSConnection(
                url.hostname or "", url.port, timeout=UPSTREAM_TIMEOUT, context=ssl.create_default_context()
            )
        else:
            conn = http.client.HTTPConnection(url.hostname or "", url.port, timeout=UPSTREAM_TIMEOUT)
        error: str | None = None
        handed_off = False
        try:
            target = url.path + (f"?{query}" if query else "")
            conn.request(method, target, body=raw or None, headers=headers)
            resp = conn.getresponse()
            error = None if resp.status < 400 else f"HTTP {resp.status}"
            if req is not None and resp.status in FALLBACK_STATUSES:
                fallbacks = self.policy.fallbacks(req.decision, req.needs, req.task, req.billing)
                if fallbacks and self._routable(suffix):
                    # The vendor is limited or down: try Prompture models before the CLI sees an error.
                    resp.read()
                    req.call.attempts.append(
                        {"model": req.call.served, "status": "error", "error": error, "cost_usd": 0.0}
                    )
                    self.policy.count_attempt(req.task)
                    handed_off = True
                    reason = f"{req.call.served} answered {error}; fell back to {fallbacks[0]}."
                    req.call.rule = {**req.call.rule, "source": "fallback", "reason": reason}
                    return self._route(handler, req, fallbacks, query, vendor_last=False)
            handler.send_response(resp.status, resp.reason)
            for key, value in resp.getheaders():
                if key.lower() not in _HOP_HEADERS:
                    handler.send_header(key, value)
            if "text/event-stream" not in (resp.getheader("Content-Type") or ""):
                data = resp.read()
                sniffer.feed(data, stream=False)
                handler.send_header("Content-Length", str(len(data)))
                handler.end_headers()
                handler.wfile.write(data)
                return
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.close_connection = True
            first = True
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                if first and req is not None:
                    self._first_token(req)
                    first = False
                sniffer.feed(chunk, stream=True)
                handler.wfile.write(chunk)
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            error = "client disconnected"
        except (OSError, http.client.HTTPException) as exc:
            error = f"upstream unreachable: {exc}"
            logger.warning("router: %s %s failed: %s", method, url.geturl(), exc)
            with contextlib.suppress(OSError):
                _send_json(handler, 502, _error_body(tool.dialect, f"Prompture router: {error}"))
        finally:
            conn.close()
            if req is not None and not handed_off:
                if req.call.attempts:
                    req.call.attempts.append(
                        {
                            "model": req.call.served,
                            "status": "error" if error else "ok",
                            "error": error,
                            "cost_usd": 0.0,
                        }
                    )
                self._finish(req, sniffer.finish(), error)

    # -- route to a Prompture model -------------------------------------------

    def _route(
        self,
        handler: BaseHTTPRequestHandler,
        req: _Request,
        targets: list[str],
        query: str,
        *,
        vendor_last: bool = True,
    ) -> None:
        """Answer from the first of *targets* that works; then the vendor's own path (``vendor_last``)."""
        from ..gateway import (
            ChatOutcome,
            anthropic_message,
            anthropic_sse,
            anthropic_to_driver,
            get_reasoning_cache,
            live_events_for,
            response_object,
            responses_sse,
            responses_to_driver,
            run_chat,
            stream_anthropic_events,
            stream_responses_events,
        )

        tool = req.tool
        anthropic = req.suffix == "/v1/messages"
        stream = bool(req.body.get("stream"))
        if targets and targets[0] == req.decision.target:
            targets = [*targets, *self.policy.fallbacks(req.decision, req.needs, req.task, req.billing)]
        reply_id = (MESSAGE_PREFIX if anthropic else RESPONSE_PREFIX) + uuid.uuid4().hex[:24]
        last_error: str | None = None

        for i, target in enumerate(targets):
            if i > 0:
                self.policy.count_attempt(req.task)
            attempt_started = time.perf_counter()
            outcome_box: list[Any] = []
            try:
                msgs, tools, options = anthropic_to_driver(req.body) if anthropic else responses_to_driver(req.body)
                msgs = get_reasoning_cache().restore(msgs)
                driver = self.driver(target)
            except Exception as exc:
                last_error = f"{target}: {exc}"
                req.call.attempts.append({"model": target, "status": "error", "error": last_error, "cost_usd": 0.0})
                continue

            def remember(
                outcome: ChatOutcome,
                driver: Any = driver,
                began: float = attempt_started,
                box: list[Any] = outcome_box,
            ) -> None:
                if outcome.error is None:
                    get_reasoning_cache().remember(outcome)
                record_usage(driver, outcome, (time.perf_counter() - began) * 1000)
                box.append(outcome)

            req.call.served = target
            req.call.route = "routed"
            req.call.billing = billing_of(target)
            if i > 0 and req.call.rule.get("source") != "fallback":
                req.call.rule = {
                    **req.call.rule,
                    "source": "fallback",
                    "reason": f"{targets[i - 1]} failed ({last_error}); fell back to {target}.",
                }
            try:
                if not stream:
                    try:
                        outcome = run_chat(driver, msgs, options, tools=tools or None)
                    except Exception as exc:
                        remember(ChatOutcome(error=exc))
                        raise
                    remember(outcome)
                    self._routed_attempt(req, target, outcome, "ok")
                    body = (
                        anthropic_message(outcome, model=req.model, message_id=reply_id)
                        if anthropic
                        else response_object(outcome, model=req.model, response_id=reply_id)
                    )
                    _send_json(handler, 200, body)
                    return self._finish(req, Usage.from_meta(outcome.meta, target))
                events = live_events_for(driver, msgs, tools, options)
                frames: Iterator[str]
                if anthropic:
                    frames = (
                        anthropic_sse(name, data)
                        for name, data in stream_anthropic_events(
                            events, model=req.model, message_id=reply_id, on_complete=remember
                        )
                    )
                else:
                    frames = (
                        responses_sse(event)
                        for event in stream_responses_events(
                            events, model=req.model, response_id=reply_id, on_complete=remember
                        )
                    )
                # The driver call starts here; until the first content arrives, a failure can still fall back.
                head, failure = _prefetch(frames)
                if failure is not None:
                    raise RuntimeError(failure)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return self._finish(req, Usage(), "client disconnected")
            except Exception as exc:
                last_error = f"{target}: {exc}"
                logger.warning("router: %s via %s failed: %s", req.model, target, exc)
                req.call.attempts.append({"model": target, "status": "error", "error": str(exc), "cost_usd": 0.0})
                continue
            return self._stream_rest(handler, req, target, head, frames, outcome_box)

        # Every Prompture target failed: the CLI's own vendor answers, as it would have without routing.
        if vendor_last:
            req.call.served = f"{tool.vendor}/{req.model}"
            req.call.route = "passthrough"
            req.call.billing = req.billing
            req.call.rule = {
                **req.call.rule,
                "source": "fallback",
                "reason": f"{targets[-1] if targets else 'The route'} failed ({last_error}); fell back to {req.model}.",
            }
            self.policy.count_attempt(req.task)
            raw = json.dumps(req.body).encode()
            return self._pass(handler, tool, "POST", req.suffix, query, raw, req)
        self._finish(req, Usage(), last_error)
        with contextlib.suppress(OSError):
            _send_json(handler, 502, _error_body(tool.dialect, f"Prompture route {last_error}"))

    def _stream_rest(
        self,
        handler: BaseHTTPRequestHandler,
        req: _Request,
        target: str,
        head: list[str],
        frames: Iterator[str],
        outcome_box: list[Any],
    ) -> None:
        error: str | None = None
        try:
            _start_stream(handler)
            first = True
            for part in head, frames:
                for f in part:
                    if first and "delta" in f:
                        self._first_token(req)
                        first = False
                    handler.wfile.write(f.encode())
                    handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            error = "client disconnected"
        except Exception as exc:
            error = f"{target}: {exc}"
            logger.warning("router: %s via %s failed mid-stream: %s", req.model, target, exc)
        outcome = outcome_box[-1] if outcome_box else None
        self._routed_attempt(req, target, outcome, "error" if error else "ok", error)
        self._finish(req, Usage.from_meta(outcome.meta if outcome else None, target), error)

    def _routed_attempt(self, req: _Request, target: str, outcome: Any, status: str, error: str | None = None) -> None:
        meta = (outcome.meta if outcome is not None else None) or {}
        cost = float(meta.get("cost") or 0.0)
        if cost:
            req.call.cost_usd = cost
            req.call.cost_source = str(meta.get("cost_source") or "reported")
        raw_route = meta.get("route")
        route: dict[str, Any] = raw_route if isinstance(raw_route, dict) else {}
        if route.get("served_by") and target.split("/", 1)[0] in ("auto", "combo"):
            req.call.served = f"{target} → {route['served_by']}"
        if req.call.attempts:
            req.call.attempts.append({"model": target, "status": status, "error": error, "cost_usd": round(cost, 6)})

    # -- tasks ----------------------------------------------------------------

    def hook(self, agent: str, event: str, session: str, project: str | None = None) -> None:
        """A CLI hook event (see :mod:`.hook`): permission prompts mark the task as waiting on the user."""
        tool = next((t for t in TOOLS.values() if t.agent == agent), None)
        if tool is None:
            return
        if project:
            self.policy.task(tool.id, session, project)
        if event == "Notification":
            self.policy.set_waiting(tool.id, session, True)
        elif event in ("UserPromptSubmit", "PostToolUse", "Stop"):
            self.policy.set_waiting(tool.id, session, False)

    def escalate(self, tool_id: str, session: str, reason: str) -> Decision | None:
        """A reviewer (or the user) says the task's answers aren't good enough: move it up."""
        tool = TOOLS.get(tool_id)
        task = self.policy.find(tool_id, session) if tool else None
        if tool is None or task is None:
            return None
        requested = self._last_requested(tool, session)
        return self.policy.escalate(
            tool_id, tool.dialect, task, requested or "", reason or "a review rejected the result"
        )

    def _last_requested(self, tool: Tool, session: str) -> str | None:
        for call in reversed(self.calls.session_calls(tool.id, session)):
            return call.requested.split("/", 1)[-1]
        return None
