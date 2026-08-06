"""Bridge between MCP (Model Context Protocol) servers and Prompture's tool registry.

Discovers tools exposed by an MCP server over a ``mcp.ClientSession`` and
registers them as Prompture :class:`ToolDefinition` objects, so agents can
call MCP tools exactly like native Prompture tools or tukuy skills.

The ``mcp`` package is an optional dependency — install it with::

    pip install 'prompture[mcp]'

All ``mcp`` imports are lazy to avoid import-time errors when the package
is not installed.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from ..agents.tools_schema import ToolDefinition, ToolRegistry

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import mcp

_MCP_IMPORT_MESSAGE = "The 'mcp' package is required for MCP integration. Install it with: pip install 'prompture[mcp]'"

# Tool names must match this pattern to be accepted by LLM providers.
_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def _import_mcp() -> Any:
    """Import the ``mcp`` package lazily, raising a clear error if missing."""
    try:
        import mcp
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_MCP_IMPORT_MESSAGE) from exc
    return mcp


def _sanitize_name(name: str, prefix: str | None = None) -> str:
    """Sanitize an MCP tool name to ``^[a-zA-Z0-9_-]{1,64}$``.

    Invalid characters are replaced with ``_`` and the result is truncated
    to 64 characters.  When *prefix* is given, it is prepended as
    ``{prefix}_{name}`` before sanitization.
    """
    raw = f"{prefix}_{name}" if prefix else name
    sanitized = _INVALID_NAME_CHARS.sub("_", raw)[:64]
    if not sanitized:
        sanitized = "mcp_tool"
    if not _NAME_PATTERN.match(sanitized):  # pragma: no cover - defensive
        sanitized = "mcp_tool"
    return sanitized


def _serialize_content_block(block: Any) -> str:
    """Serialize a single MCP content block to a string.

    Text blocks contribute their text; non-text blocks (images, resources,
    ...) are represented by a short placeholder since the tool-result path
    is string-only today.
    """
    text = getattr(block, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(block, str):
        return block
    block_type = getattr(block, "type", None) or type(block).__name__
    return f"[{block_type} content block]"


def _serialize_call_result(result: Any) -> str:
    """Serialize an MCP ``CallToolResult`` to a single string for the tool-result path."""
    if isinstance(result, str):
        return result
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if not content:
        return ""
    if not isinstance(content, (list, tuple)):
        content = [content]
    return "\n".join(_serialize_content_block(block) for block in content)


def _make_mcp_tool_function(session: mcp.ClientSession, mcp_name: str) -> Any:
    """Build the sync/async callables that dispatch a tool call to the MCP session."""

    async def _acall(**arguments: Any) -> str:
        try:
            result = await session.call_tool(mcp_name, arguments)
        except Exception as exc:
            # MCP call errors become LLM-friendly error strings (registry convention).
            return f"Error calling MCP tool '{mcp_name}': {exc}"
        text = _serialize_call_result(result)
        if getattr(result, "isError", False):
            return f"Error from MCP tool '{mcp_name}': {text}"
        return text

    def _call(**arguments: Any) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_acall(**arguments))
        raise RuntimeError(
            f"MCP tool '{mcp_name}' was called synchronously from within a running event loop. "
            "Use AsyncAgent/AsyncConversation (which awaits the tool via its async path), "
            "or call it from a thread without a running loop."
        )

    _call._async_fn = _acall  # type: ignore[attr-defined]
    return _call


async def register_mcp_tools(
    registry: ToolRegistry,
    session: mcp.ClientSession,
    *,
    prefix: str | None = None,
) -> list[str]:
    """Register every tool exposed by an MCP session into *registry*.

    Calls ``session.list_tools()``, converts each MCP tool to a
    :class:`ToolDefinition` (the MCP ``inputSchema`` is already JSON Schema
    and is passed through unchanged, with names sanitized to
    ``^[a-zA-Z0-9_-]{1,64}$`` and optionally prefixed), and registers it so
    execution dispatches to ``session.call_tool(name, arguments)``.

    Each tool is registered with an ``_async_fn`` hook, so
    :class:`AsyncAgent`/:class:`AsyncConversation` await it natively.  The
    sync execution path uses ``asyncio.run`` and raises a clear error when
    called from within a running event loop.

    Args:
        registry: The :class:`ToolRegistry` to register tools into.
        session: An initialized ``mcp.ClientSession``.
        prefix: Optional prefix prepended to every tool name as
            ``{prefix}_{name}`` (useful to namespace several MCP servers).

    Returns:
        The list of registered (sanitized) tool names.
    """
    list_result = await session.list_tools()
    tools = getattr(list_result, "tools", list_result)

    registered: list[str] = []
    for tool in tools:
        mcp_name = getattr(tool, "name", None)
        if not mcp_name:
            continue
        name = _sanitize_name(str(mcp_name), prefix)
        description = getattr(tool, "description", None) or f"MCP tool {mcp_name}"
        parameters = getattr(tool, "inputSchema", None)
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        registry.add(
            ToolDefinition(
                name=name,
                description=str(description),
                parameters=parameters,
                function=_make_mcp_tool_function(session, str(mcp_name)),
            )
        )
        registered.append(name)
    return registered


def register_mcp_tools_sync(
    registry: ToolRegistry,
    session: mcp.ClientSession,
    *,
    prefix: str | None = None,
) -> list[str]:
    """Synchronous wrapper around :func:`register_mcp_tools`.

    Uses ``asyncio.run`` when no event loop is running; raises a clear
    error otherwise.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(register_mcp_tools(registry, session, prefix=prefix))
    raise RuntimeError(
        "register_mcp_tools_sync() cannot be called from within a running event loop. "
        "Use 'await register_mcp_tools(...)' instead."
    )


@asynccontextmanager
async def mcp_session_from_stdio(
    command: str,
    args: list[str] | None = None,
    env: dict | None = None,
) -> AsyncIterator[mcp.ClientSession]:
    """One-liner async context manager yielding an initialized MCP stdio session.

    Example::

        async with mcp_session_from_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]) as session:
            await register_mcp_tools(registry, session)

    Args:
        command: The command used to start the MCP server process.
        args: Optional command arguments.
        env: Optional environment variables for the server process.
    """
    mcp = _import_mcp()
    from mcp.client.stdio import stdio_client

    server_params = mcp.StdioServerParameters(command=command, args=args or [], env=env)
    async with stdio_client(server_params) as (read, write), mcp.ClientSession(read, write) as session:
        await session.initialize()
        yield session


__all__ = [
    "mcp_session_from_stdio",
    "register_mcp_tools",
    "register_mcp_tools_sync",
]
