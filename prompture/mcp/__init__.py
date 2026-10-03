"""Model Context Protocol (MCP) runtime for Prompture.

Two directions, both reusing Prompture's existing ``ToolDefinition`` layer:

- **Server** — expose Prompture's generation/discovery/pricing drivers as MCP
  tools so any MCP client can call them: :func:`build_mcp_server` / :func:`serve`.
- **Client** — consume an external MCP server and adapt its tools into Prompture
  :class:`ToolDefinition`s for agents: :func:`load_mcp_registry` /
  :func:`stdio_session` / :func:`http_session` / :func:`fetch_mcp_tool_definitions`.
- **Hub** — named servers in ``~/.prompture/mcp.json`` (+ project
  ``.prompture/mcp.json``), mounted by name with ``tools=["mcp:<name>"]`` or
  :func:`load_mcp_registry_by_name`; pooled sessions, prefixed tool names,
  presets and opt-in editor import (:mod:`prompture.mcp.hub`).

The canonical capability set is :func:`prompture_tool_definitions` (SDK-independent).
Server/client require the optional ``mcp`` package (``pip install prompture[mcp]``).
"""

from __future__ import annotations

from .client import (
    fetch_mcp_tool_definitions,
    http_session,
    load_mcp_registry,
    mcp_tool_to_definition,
    stdio_session,
)
from .hub import (
    PRESETS,
    InlineSecretError,
    MCPConfigError,
    MCPConnectionError,
    MCPHub,
    MCPHubError,
    MCPServerConfig,
    MCPSessionPool,
    MCPTimeoutError,
    MissingEnvError,
    check_server,
    get_pool,
    get_preset,
    list_presets,
    load_mcp_registry_by_name,
    load_mcp_registry_by_name_sync,
    resolve_mcp_tools,
)
from .server import build_mcp_server, register_tools, serve
from .tools import (
    prompture_estimate_cost,
    prompture_generate,
    prompture_tool_definitions,
)

__all__ = [
    # hub
    "PRESETS",
    "InlineSecretError",
    "MCPConfigError",
    "MCPConnectionError",
    "MCPHub",
    "MCPHubError",
    "MCPServerConfig",
    "MCPSessionPool",
    "MCPTimeoutError",
    "MissingEnvError",
    # server
    "build_mcp_server",
    "check_server",
    "fetch_mcp_tool_definitions",
    "get_pool",
    "get_preset",
    "http_session",
    "list_presets",
    # client
    "load_mcp_registry",
    "load_mcp_registry_by_name",
    "load_mcp_registry_by_name_sync",
    "mcp_tool_to_definition",
    "prompture_estimate_cost",
    "prompture_generate",
    # tools (SDK-independent)
    "prompture_tool_definitions",
    "register_tools",
    "resolve_mcp_tools",
    "serve",
    "stdio_session",
]
