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
    "load_mcp_registry",
    "mcp_tool_to_definition",
    "stdio_session",
]


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
