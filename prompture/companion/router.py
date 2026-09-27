"""The companion's router: coding CLIs send their model traffic through Prompture.

A CLI pointed at ``<companion>/tools/<tool>`` (Claude Code through
``ANTHROPIC_BASE_URL``, Codex through a ``model_providers`` entry; see
:mod:`.tool_routing`) reaches this module instead of its vendor. Each request
is either:

- **passed through** (the default): forwarded unchanged to the vendor with the
  CLI's own credentials, so API keys and subscription logins keep working, and
  streamed back as it arrives; or
- **routed** to any Prompture model (``ollama/qwen3``, ``combo/…``,
  ``auto/cheap``, an alias) through Prompture's drivers, translated to and from
  the CLI's wire format by :mod:`prompture.gateway`.

Which one is decided per tool by ``~/.prompture/routes.json``::

    {"tools": {"claude-code": {
        "models": {"claude-haiku-*": "ollama/qwen3:8b"},
        "kinds": {"background": "ollama/qwen3:8b"}}}}

``kinds`` matches what a request is for (:func:`anthropic_kind`,
:func:`responses_kind`): ``main`` turns, ``tool_result`` follow-ups, and the
``background`` ones (``title``, ``probe``, ``compaction``) that a small model
handles as well as a big one. A kind rule wins over a model pattern; anything
unmatched passes through.

Every request shows live on the bus: ``request.started`` (with ``tool``,
``kind`` and, when routed, ``routed_to``), ``request.first_token`` and
``request.ended``. Usage isn't reported here, so nothing is counted twice:
passed-through calls are counted from the CLI's own logs, routed ones by the
usage ledger every driver call writes to. Routed replies carry a
``msg_prompture_`` / ``resp_prompture_`` id, which the log readers skip.

Only localhost may call the router: requests naming another host, or coming
from a web page (an ``Origin`` header), are refused.
"""

from __future__ import annotations

import contextlib
import fnmatch
import http.client
import json
import logging
import ssl
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .live import LiveBus, new_request_id

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

#: Request kinds a small model handles as well as a big one.
BACKGROUND_KINDS = {"title", "probe", "compaction"}
KINDS = ("main", "tool_result", "title", "probe", "compaction")

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


# ------------------------------------------------------------------ routes


class Routes:
    """``routes.json``: which requests each tool sends to which Prompture model."""

    def __init__(self, path: str | Path | None = ROUTES_FILE) -> None:
        self.path = Path(path) if path else None
        self._cache: tuple[float, dict[str, Any]] | None = None
        self._lock = threading.Lock()

    def data(self) -> dict[str, Any]:
        if self.path is None:
            return {"tools": {}}
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
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(clean, indent=2), encoding="utf-8")
            tmp.replace(self.path)
            with self._lock:
                self._cache = None
        return clean

    def target(self, tool: str, model: str, kind: str) -> str | None:
        """The Prompture model a request goes to, or ``None`` to pass it through."""
        rules = self.data()["tools"].get(tool) or {}
        kinds = rules.get("kinds") or {}
        target = kinds.get(kind) or (kinds.get("background") if kind in BACKGROUND_KINDS else None)
        if not target:
            for pattern, to in (rules.get("models") or {}).items():
                if fnmatch.fnmatchcase(model.lower(), pattern.lower()):
                    target = to
                    break
        return None if not target or target == PASSTHROUGH else target


def normalize_routes(data: Any) -> dict[str, Any]:
    """Keep only well-formed rules: string patterns and targets, known kinds."""
    tools: dict[str, Any] = {}
    raw = data.get("tools") if isinstance(data, dict) else None
    for tool, rules in (raw or {}).items() if isinstance(raw, dict) else ():
        if not isinstance(tool, str) or not isinstance(rules, dict):
            continue
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
        if models or kinds:
            tools[tool] = {"models": models, "kinds": kinds}
    return {"tools": tools}


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


# ------------------------------------------------------------------ router


class Router:
    """Serves ``/tools/<tool>/…`` on the companion; see the module docstring.

    ``upstream(tool_id)`` names a custom vendor URL for a tool (the gateway it
    pointed at before routing was turned on), or ``None`` for the vendor's own.
    """

    def __init__(
        self,
        bus: LiveBus,
        routes: Routes | None = None,
        *,
        upstream: Callable[[str], str | None] | None = None,
        driver_for: Callable[[str], Any] | None = None,
    ) -> None:
        self.bus = bus
        self.routes = routes or Routes()
        self.upstream = upstream or (lambda tool: None)
        self._driver_for = driver_for

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
        target = self.routes.target(tool.id, model, kind) if model and kind else None
        if target and self._routable(tool, suffix):
            return self._route(handler, tool, suffix, body, model, kind or "main", target)
        return self._pass(handler, tool, method, suffix, query, raw, model, kind)

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
    def _routable(tool: Tool, suffix: str) -> bool:
        return suffix in ("/v1/messages", "/v1/responses")

    # -- live events ----------------------------------------------------------

    def _started(self, tool: Tool, model: str, suffix: str, kind: str, stream: bool, routed_to: str | None) -> str:
        rid = new_request_id()
        vendor = "claude" if tool.dialect == "anthropic" else "openai"
        self.bus.publish(
            "request.started",
            {
                "request_id": rid,
                "key_id": None,
                "key_name": tool.name,
                "model": f"{vendor}/{model}" if model else vendor,
                "routed_to": routed_to,
                "project": None,
                "endpoint": suffix,
                "stream": stream,
                "tool": tool.agent,
                "kind": kind,
            },
        )
        return rid

    def _ended(self, rid: str, tool: Tool, started: float, error: str | None = None) -> None:
        self.bus.publish(
            "request.ended",
            {
                "request_id": rid,
                "tool": tool.agent,
                "key_name": tool.name,
                "status": "error" if error else "ok",
                "error": error,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    def _first_token(self, rid: str, started: float) -> None:
        self.bus.publish(
            "request.first_token", {"request_id": rid, "ttft_ms": int((time.perf_counter() - started) * 1000)}
        )

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
        model: str,
        kind: str | None,
    ) -> None:
        url = urlsplit(self.upstream_url(tool, suffix, handler.headers))
        headers = {k: v for k, v in handler.headers.items() if k.lower() not in _HOP_HEADERS}
        headers["Host"] = url.netloc
        headers["Accept-Encoding"] = "identity"  # read as it streams, never recompressed
        if raw or method == "POST":
            headers["Content-Length"] = str(len(raw))
        started = time.perf_counter()
        rid = self._started(tool, model, suffix, kind, True, None) if kind else None
        conn: http.client.HTTPConnection
        if url.scheme == "https":
            conn = http.client.HTTPSConnection(
                url.hostname or "", url.port, timeout=UPSTREAM_TIMEOUT, context=ssl.create_default_context()
            )
        else:
            conn = http.client.HTTPConnection(url.hostname or "", url.port, timeout=UPSTREAM_TIMEOUT)
        error: str | None = None
        try:
            target = url.path + (f"?{query}" if query else "")
            conn.request(method, target, body=raw or None, headers=headers)
            resp = conn.getresponse()
            error = None if resp.status < 400 else f"HTTP {resp.status}"
            handler.send_response(resp.status, resp.reason)
            for key, value in resp.getheaders():
                if key.lower() not in _HOP_HEADERS:
                    handler.send_header(key, value)
            if "text/event-stream" not in (resp.getheader("Content-Type") or ""):
                data = resp.read()
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
                if first and rid:
                    self._first_token(rid, started)
                    first = False
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
            if rid:
                self._ended(rid, tool, started, error)

    # -- route to a Prompture model -------------------------------------------

    def _route(
        self,
        handler: BaseHTTPRequestHandler,
        tool: Tool,
        suffix: str,
        body: dict[str, Any],
        model: str,
        kind: str,
        target: str,
    ) -> None:
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

        stream = bool(body.get("stream"))
        started = time.perf_counter()
        rid = self._started(tool, model, suffix, kind, stream, target)
        anthropic = suffix == "/v1/messages"
        error: str | None = None

        try:
            if anthropic:
                msgs, tools, options = anthropic_to_driver(body)
            else:
                msgs, tools, options = responses_to_driver(body)
            msgs = get_reasoning_cache().restore(msgs)
            driver = self.driver(target)
        except Exception as exc:
            error = f"{target}: {exc}"
            self._ended(rid, tool, started, error)
            return _send_json(handler, 502, _error_body(tool.dialect, f"Prompture route {error}"))

        def remember(outcome: ChatOutcome) -> None:
            if outcome.error is None:
                get_reasoning_cache().remember(outcome)
            record_usage(driver, outcome, (time.perf_counter() - started) * 1000)

        reply_id = (MESSAGE_PREFIX if anthropic else RESPONSE_PREFIX) + uuid.uuid4().hex[:24]
        try:
            if not stream:
                try:
                    outcome = run_chat(driver, msgs, options, tools=tools or None)
                except Exception as exc:
                    remember(ChatOutcome(error=exc))
                    raise
                remember(outcome)
                if anthropic:
                    return _send_json(handler, 200, anthropic_message(outcome, model=model, message_id=reply_id))
                return _send_json(handler, 200, response_object(outcome, model=model, response_id=reply_id))
            _start_stream(handler)
            events = live_events_for(driver, msgs, tools, options)
            frames: Iterator[str]
            if anthropic:
                frames = (
                    anthropic_sse(name, data)
                    for name, data in stream_anthropic_events(
                        events, model=model, message_id=reply_id, on_complete=remember
                    )
                )
            else:
                frames = (
                    responses_sse(event)
                    for event in stream_responses_events(
                        events, model=model, response_id=reply_id, on_complete=remember
                    )
                )
            first = True
            for frame in frames:
                if first and "delta" in frame:
                    self._first_token(rid, started)
                    first = False
                handler.wfile.write(frame.encode())
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            error = "client disconnected"
        except Exception as exc:
            error = f"{target}: {exc}"
            logger.warning("router: %s via %s failed: %s", model, target, exc)
            if not stream:
                with contextlib.suppress(OSError):
                    _send_json(handler, 502, _error_body(tool.dialect, f"Prompture route {error}"))
        finally:
            self._ended(rid, tool, started, error)
