"""MCP client — let Prompture agents consume external MCP servers as tools.

Connects to an MCP server, lists its tools, and adapts each into a Prompture
:class:`~prompture.agents.tools_schema.ToolDefinition` (whose function is async
and calls the remote tool). Drop the resulting definitions into a ``ToolRegistry``
and any Prompture agent can use the remote tools. Requires ``prompture[mcp]``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from ..agents.tools_schema import ToolDefinition, ToolRegistry

__all__ = [
    "fetch_mcp_tool_definitions",
    "http_session",
    "load_mcp_registry",
    "mcp_tool_to_definition",
    "stdio_session",
]

#: Default seconds for HTTP connect/write; long-lived SSE reads get ``sse_read_timeout``.
DEFAULT_HTTP_TIMEOUT = 30.0
DEFAULT_SSE_READ_TIMEOUT = 300.0


def _extract_content(result: Any) -> Any:
    """Normalize an MCP ``CallToolResult`` to a plain Python value."""
    content = getattr(result, "content", None)
    if content is None:
        return result
    texts = [getattr(c, "text", None) for c in content]
    texts = [t for t in texts if t is not None]
    if len(texts) == 1:
        return texts[0]
    return texts or content


def mcp_tool_to_definition(tool: Any, session: Any) -> ToolDefinition:
    """Adapt one remote MCP tool descriptor into a Prompture :class:`ToolDefinition`.

    The returned definition's ``function`` is an async callable that invokes the
    remote tool via ``session.call_tool`` (use it through ``ToolRegistry.aexecute``).
    """
    name = tool.name
    schema = getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}}
    description = getattr(tool, "description", None) or name

    async def _call(**kwargs: Any) -> Any:
        result = await session.call_tool(name, kwargs)
        return _extract_content(result)

    _call.__name__ = name
    return ToolDefinition(name=name, description=description, parameters=schema, function=_call)


async def fetch_mcp_tool_definitions(session: Any) -> list[ToolDefinition]:
    """List the tools on a connected MCP *session* and adapt them to ToolDefinitions."""
    listing = await session.list_tools()
    tools = getattr(listing, "tools", listing)
    return [mcp_tool_to_definition(t, session) for t in tools]


async def load_mcp_registry(session: Any, registry: ToolRegistry | None = None) -> ToolRegistry:
    """Fetch a session's tools into a :class:`ToolRegistry` (new one if not given)."""
    registry = registry or ToolRegistry()
    for td in await fetch_mcp_tool_definitions(session):
        registry.add(td)
    return registry


def _require_client() -> tuple[Any, Any, Any]:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "The MCP client requires the 'mcp' package. Install it with: pip install prompture[mcp]"
        ) from exc
    return ClientSession, StdioServerParameters, stdio_client


@asynccontextmanager
async def stdio_session(command: str, args: list[str] | None = None, env: dict[str, str] | None = None):
    """Async context manager yielding an initialized MCP ``ClientSession`` over stdio.

    Example::

        async with stdio_session("python", ["-m", "some_mcp_server"]) as session:
            reg = await load_mcp_registry(session)
    """
    ClientSession, StdioServerParameters, stdio_client = _require_client()
    params = StdioServerParameters(command=command, args=args or [], env=env)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


def _require_http_client() -> tuple[Any, Any]:
    """Return ``(ClientSession, streamable_http module)`` or raise with an install hint."""
    try:
        from mcp import ClientSession
        from mcp.client import streamable_http
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "The MCP HTTP client requires the 'mcp' package. Install it with: pip install prompture[mcp]"
        ) from exc
    return ClientSession, streamable_http


@asynccontextmanager
async def _http_transport(
    module: Any,
    url: str,
    headers: dict[str, str] | None,
    timeout: float,
    sse_read_timeout: float,
):
    """Open a streamable-HTTP transport across ``mcp`` SDK generations.

    Newer SDKs expose ``streamable_http_client(url, http_client=...)`` and take
    headers/timeouts from a pre-built HTTP client; older ones expose
    ``streamablehttp_client(url, headers=..., timeout=...)``. Both yield a tuple
    whose first two items are the read and write streams.
    """
    new_style = getattr(module, "streamable_http_client", None)
    if new_style is not None:
        from mcp.shared import _httpx_utils as http_utils

        http_lib = getattr(http_utils, "httpx2", None) or getattr(http_utils, "httpx", None)
        client_timeout = http_lib.Timeout(timeout, read=sse_read_timeout) if http_lib is not None else None
        client = http_utils.create_mcp_http_client(headers=headers or None, timeout=client_timeout)
        async with client, new_style(url, http_client=client) as streams:
            yield streams[0], streams[1]
        return
    legacy = module.streamablehttp_client
    async with legacy(url, headers=headers or None, timeout=timeout, sse_read_timeout=sse_read_timeout) as streams:
        yield streams[0], streams[1]


@asynccontextmanager
async def http_session(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    *,
    sse_read_timeout: float = DEFAULT_SSE_READ_TIMEOUT,
):
    """Async context manager yielding an initialized MCP ``ClientSession`` over streamable HTTP.

    Args:
        url: The server's MCP endpoint (e.g. ``https://mcp.example.com/mcp``).
        headers: Extra request headers (``Authorization`` etc.). Resolve any
            secrets before calling; values are sent as given.
        timeout: Seconds for connect/write operations.
        sse_read_timeout: Seconds a server-sent event stream may stay idle.

    Example::

        async with http_session("https://mcp.example.com/mcp") as session:
            reg = await load_mcp_registry(session)
    """
    ClientSession, module = _require_http_client()
    async with (
        _http_transport(module, url, headers, timeout, sse_read_timeout) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session
