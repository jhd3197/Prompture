"""Named tool specs: ``tools=["mcp:search", "pack:finance", "cli:gh"]``.

Agents accept strings of the form ``"<namespace>:<name>"`` next to callables
and :class:`ToolDefinition` objects. Each namespace has a resolver that turns
the name into a list of tool definitions. Built-in namespaces are imported
lazily so an unused one costs nothing:

* ``mcp`` — named MCP servers from the MCP hub registry.
* ``pack`` — curated domain tool packs (finance, news, dev, places).
* ``cli`` — wrapped command-line tools (``gh``, ``yt-dlp``, user-defined).
* ``web`` — web toolkit (``web:all``, ``web:search``, ``web:fetch``, ...).
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Iterable
from typing import Any

from ..agents.tools_schema import ToolDefinition, tool_from_function

Resolver = Callable[[str], list[ToolDefinition]]

_resolvers: dict[str, Resolver] = {}
_lock = threading.Lock()

# namespace → "module:function", imported on first use.
_BUILTIN_RESOLVERS: dict[str, str] = {
    "mcp": "prompture.mcp.hub:resolve_mcp_tools",
    "pack": "prompture.tools.packs:resolve_pack_tools",
    "cli": "prompture.tools.cli:resolve_cli_tools",
    "web": "prompture.tools.web:resolve_web_tools",
}


def register_tool_namespace(namespace: str, resolver: Resolver) -> None:
    """Register *resolver* for ``"<namespace>:<name>"`` tool specs."""
    with _lock:
        _resolvers[namespace.lower()] = resolver


def unregister_tool_namespace(namespace: str) -> None:
    with _lock:
        _resolvers.pop(namespace.lower(), None)


def _resolver_for(namespace: str) -> Resolver:
    ns = namespace.lower()
    with _lock:
        found = _resolvers.get(ns)
    if found is not None:
        return found
    target = _BUILTIN_RESOLVERS.get(ns)
    if target is None:
        known = sorted(set(_resolvers) | set(_BUILTIN_RESOLVERS))
        raise ValueError(f"Unknown tool namespace {namespace!r}. Known: {', '.join(known)}")
    module, func = target.split(":", 1)
    resolver = getattr(importlib.import_module(module), func)
    register_tool_namespace(ns, resolver)
    return resolver


def is_tool_spec(item: Any) -> bool:
    return isinstance(item, str) and ":" in item and not item.startswith(":")


def resolve_tool_spec(spec: str) -> list[ToolDefinition]:
    """Resolve one ``"<namespace>:<name>"`` spec into tool definitions."""
    if not is_tool_spec(spec):
        raise ValueError(f"Tool spec must look like 'namespace:name', got {spec!r}")
    namespace, name = spec.split(":", 1)
    return list(_resolver_for(namespace)(name))


def expand_tool_specs(tools: Iterable[Any]) -> list[Any]:
    """Replace string specs in *tools* with their tool definitions; keep everything else."""
    out: list[Any] = []
    for item in tools:
        if isinstance(item, str):
            out.extend(resolve_tool_spec(item))
        else:
            out.append(item)
    return out


def to_tool_definition(item: Any) -> ToolDefinition:
    """Coerce a callable or :class:`ToolDefinition` into a definition."""
    if isinstance(item, ToolDefinition):
        return item
    if callable(item):
        return tool_from_function(item)
    raise TypeError(f"Unsupported tool type: {type(item).__name__}")
