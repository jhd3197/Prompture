"""Node-type registry — the extensibility seam, modeled on ``drivers/registry``.

A node ``type`` string maps to a :class:`NodeType` bundling a ``run`` callable
plus declared handles. Custom node types register exactly like custom drivers::

    register_node_type("my_node", my_runner, output_handles=(Handle("result"),))
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .model import Handle

__all__ = [
    "NodeType",
    "get_node_type",
    "is_node_type_registered",
    "list_node_types",
    "register_builtin_node_types",
    "register_node_type",
    "unregister_node_type",
]


@dataclass(frozen=True)
class NodeType:
    """A registered node behavior: a ``run`` callable + declared handles."""

    name: str
    run: Callable[..., Any]  # run(node, resolved_config, ctx) -> NodeResult
    input_handles: tuple[Handle, ...] = ()
    output_handles: tuple[Handle, ...] = ()
    produces_job: bool = False
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


_REGISTRY: dict[str, NodeType] = {}
_LOCK = threading.Lock()
_builtins_loaded = False


def register_node_type(
    name: str,
    run: Callable[..., Any],
    *,
    input_handles: tuple[Handle, ...] = (),
    output_handles: tuple[Handle, ...] = (),
    produces_job: bool = False,
    description: str = "",
    overwrite: bool = False,
) -> NodeType:
    """Register a node type. Raises if *name* exists and ``overwrite`` is False."""
    with _LOCK:
        if name in _REGISTRY and not overwrite:
            raise ValueError(f"Node type {name!r} is already registered (pass overwrite=True)")
        nt = NodeType(
            name=name,
            run=run,
            input_handles=input_handles,
            output_handles=output_handles,
            produces_job=produces_job,
            description=description,
        )
        _REGISTRY[name] = nt
        return nt


def get_node_type(name: str) -> NodeType:
    with _LOCK:
        if name not in _REGISTRY:
            raise KeyError(f"Unknown node type {name!r}. Registered: {sorted(_REGISTRY)}")
        return _REGISTRY[name]


def unregister_node_type(name: str) -> bool:
    with _LOCK:
        return _REGISTRY.pop(name, None) is not None


def list_node_types() -> list[str]:
    with _LOCK:
        return sorted(_REGISTRY)


def is_node_type_registered(name: str) -> bool:
    with _LOCK:
        return name in _REGISTRY


def register_builtin_node_types() -> None:
    """Register the built-in node runners (idempotent). Called at package import."""
    global _builtins_loaded
    with _LOCK:
        if _builtins_loaded:
            return
        _builtins_loaded = True
    from . import nodes  # lazy import to avoid a cycle (nodes imports this module)

    nodes.register_all()
