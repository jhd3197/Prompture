"""Minimal synchronous MCP client over streamable HTTP.

Just enough JSON-RPC to call one tool on a remote MCP server without the
``mcp`` package: ``initialize`` → ``notifications/initialized`` →
``tools/call``. Responses may come back as plain JSON or as a
``text/event-stream`` body; both are handled. The ``Mcp-Session-Id`` header
returned by ``initialize`` is echoed on later requests, and an expired
session (404) triggers one re-initialize.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import threading
from typing import Any

import requests

from ...capabilities.errors import CapabilityError
from ...capabilities.http import proxies_for
from ._common import API_USER_AGENT, header, raise_for_status

PROTOCOL_VERSION = "2025-03-26"


class MCPError(CapabilityError):
    """JSON-RPC error or ``isError`` tool result from an MCP server."""

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(f"mcp error{f' {code}' if code is not None else ''}: {message}")
        self.code = code


def parse_sse_messages(text: str) -> list[dict[str, Any]]:
    """Return the JSON objects carried by ``data:`` lines of an SSE body."""
    messages: list[dict[str, Any]] = []
    data: list[str] = []

    def flush() -> None:
        if data:
            try:
                obj = json.loads("\n".join(data))
            except ValueError:
                obj = None
            if isinstance(obj, dict):
                messages.append(obj)
            elif isinstance(obj, list):
                messages.extend(o for o in obj if isinstance(o, dict))
            data.clear()

    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if not line:
            flush()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    flush()
    return messages


class MCPHttpClient:
    """Call tools on a streamable-HTTP MCP endpoint.

    Args:
        url: Endpoint URL (e.g. ``https://mcp.exa.ai/mcp``).
        session: Optional ``requests.Session``.
        headers: Extra headers sent on every request.
        timeout: Per-request timeout in seconds.
        proxy_backend: Backend name used to pick a per-backend proxy.
    """

    def __init__(
        self,
        url: str,
        *,
        session: requests.Session | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        proxy_backend: str | None = None,
        client_name: str = "prompture",
    ) -> None:
        self.url = url
        self.session = session or requests.Session()
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.proxy_backend = proxy_backend
        self.client_name = client_name
        self.session_id: str | None = None
        self._initialized = False
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": API_USER_AGENT,
            **self.headers,
        }
        if self._initialized:
            h["MCP-Protocol-Version"] = PROTOCOL_VERSION
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        return h

    def _post(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        resp = self.session.post(
            self.url,
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
            proxies=proxies_for(self.proxy_backend),
        )
        sid = header(resp, "Mcp-Session-Id")
        if sid:
            self.session_id = sid
        if "id" not in payload:  # notification
            status = getattr(resp, "status_code", 202)
            if isinstance(status, int) and status >= 400 and status != 404:
                raise_for_status(resp, self.proxy_backend or "mcp", reject=())
            return None
        raise_for_status(resp, self.proxy_backend or "mcp", reject=(400, 422))
        ctype = (header(resp, "Content-Type") or "").lower()
        if "text/event-stream" in ctype:
            raw = getattr(resp, "content", None)
            # SSE bodies carry no charset; requests would guess ISO-8859-1.
            text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else resp.text
            messages = parse_sse_messages(text)
        else:
            body = resp.json()
            messages = body if isinstance(body, list) else [body]
        msg = next((m for m in messages if m.get("id") == payload["id"]), None)
        if msg is None:
            msg = next((m for m in messages if "result" in m or "error" in m), None)
        if msg is None:
            raise MCPError("empty response from MCP server")
        if msg.get("error"):
            err = msg["error"]
            raise MCPError(str(err.get("message") or err), code=err.get("code"))
        result = msg.get("result")
        return result if isinstance(result, dict) else {"value": result}

    def _request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": next(self._ids), "method": method}
        if params is not None:
            payload["params"] = params
        result = self._post(payload)
        return result or {}

    def initialize(self) -> dict[str, Any]:
        """Run the MCP handshake (idempotent)."""
        with self._lock:
            if self._initialized:
                return {}
            self.session_id = None
            result = self._request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": self.client_name, "version": "1"},
                },
            )
            self._initialized = True
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
            return result

    def _with_session(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        self.initialize()
        try:
            return self._request(method, params)
        except requests.HTTPError as exc:
            # 404 on a session-bound request means the server dropped our session.
            if getattr(getattr(exc, "response", None), "status_code", None) == 404 and self.session_id:
                self._initialized = False
                self.initialize()
                return self._request(method, params)
            raise

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self._with_session("tools/list", {}).get("tools", []))

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call tool *name* and return the raw ``result`` (``content`` list etc.)."""
        result = self._with_session("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise MCPError(tool_text(result) or f"tool {name} failed")
        return result

    def close(self) -> None:
        """Best-effort session termination (``DELETE`` with the session id)."""
        if not self.session_id:
            return
        with contextlib.suppress(Exception):
            self.session.delete(self.url, headers=self._headers(), timeout=5)
        self.session_id = None
        self._initialized = False


def tool_text(result: dict[str, Any]) -> str:
    """Join the ``text`` blocks of a ``tools/call`` result."""
    parts = []
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]))
    return "\n\n".join(parts)
