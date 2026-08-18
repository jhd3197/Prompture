"""Model Context Protocol (MCP) runtime for Prompture.

Two directions, both reusing Prompture's existing ``ToolDefinition`` layer:

- **Server** — expose Prompture's generation/discovery/pricing drivers as MCP
  tools so any MCP client can call them: :func:`build_mcp_server` / :func:`serve`.
- **Client** — consume an external MCP server and adapt its tools into Prompture
  :class:`ToolDefinition`s for agents: :func:`load_mcp_registry` /
  :func:`stdio_session` / :func:`fetch_mcp_tool_definitions`.

The canonical capability set is :func:`prompture_tool_definitions` (SDK-independent).
Server/client require the optional ``mcp`` package (``pip install prompture[mcp]``).
"""

from __future__ import annotations

from .client import (
    fetch_mcp_tool_definitions,
    load_mcp_registry,
    mcp_tool_to_definition,
    stdio_session,
)
from .server import build_mcp_server, register_tools, serve
from .tools import (
    prompture_estimate_cost,
    prompture_generate,
    prompture_tool_definitions,
)

__all__ = [
    # server
    "build_mcp_server",
    "fetch_mcp_tool_definitions",
    # client
    "load_mcp_registry",
    "mcp_tool_to_definition",
    "prompture_estimate_cost",
    "prompture_generate",
    # tools (SDK-independent)
    "prompture_tool_definitions",
    "register_tools",
    "serve",
    "stdio_session",
]
