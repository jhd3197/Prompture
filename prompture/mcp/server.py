"""MCP server — expose Prompture's drivers (and workflows) as MCP tools.

Builds a `FastMCP <https://modelcontextprotocol.io>`_ server from Prompture's
canonical :class:`ToolDefinition` set so any MCP client (Claude Desktop, Cursor,
Windsurf, …) can call Prompture image/video/music generation, text generation,
and cost estimation. Requires the optional ``mcp`` dependency
(``pip install prompture[mcp]``).
"""

from __future__ import annotations

from typing import Any

from .tools import _coerce_tool_definitions

__all__ = ["build_mcp_server", "register_tools", "serve"]


def _require_fastmcp() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "The MCP server requires the 'mcp' package. Install it with: pip install prompture[mcp]"
        ) from exc
    return FastMCP


def register_tools(server: Any, tools: Any = None) -> Any:
    """Register Prompture tools onto an existing FastMCP *server* and return it.

    *tools* may be a list of :class:`ToolDefinition`, a ``ToolRegistry``, or ``None``
    (Prompture's default capability set). Each tool's underlying function is
    registered — FastMCP derives the input schema from its type hints + docstring.
    """
    for td in _coerce_tool_definitions(tools):
        server.add_tool(td.function, name=td.name, description=td.description)
    return server


def build_mcp_server(name: str = "prompture", *, tools: Any = None, **fastmcp_kwargs: Any) -> Any:
    """Create a FastMCP server exposing Prompture's tools.

    Args:
        name: Server name advertised to clients.
        tools: ToolDefinitions / ToolRegistry / None (defaults).
        fastmcp_kwargs: Passed through to ``FastMCP(...)``.

    Returns:
        A ``FastMCP`` instance (call ``.run(...)`` or use :func:`serve`).
    """
    FastMCP = _require_fastmcp()
    server = FastMCP(name, **fastmcp_kwargs)
    return register_tools(server, tools)


def serve(
    *,
    name: str = "prompture",
    tools: Any = None,
    transport: str = "stdio",
    **run_kwargs: Any,
) -> None:
    """Build and run a Prompture MCP server (blocking).

    Args:
        name: Server name.
        tools: ToolDefinitions / ToolRegistry / None.
        transport: ``"stdio"`` (default), ``"sse"``, or ``"streamable-http"``.
        run_kwargs: Passed through to ``FastMCP.run(...)``.
    """
    server = build_mcp_server(name, tools=tools)
    server.run(transport=transport, **run_kwargs)
