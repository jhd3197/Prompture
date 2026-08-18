"""Exception hierarchy for the workflow engine.

All errors subclass :class:`~prompture.exceptions.PromptureError` (and the
matching stdlib base, mirroring ``DriverError`` / ``ExtractionError``) so callers
can catch them broadly or narrowly.
"""

from __future__ import annotations

from ..exceptions import PromptureError

__all__ = [
    "CycleError",
    "GraphValidationError",
    "HandleTypeError",
    "NodeExecutionError",
    "ReferenceResolutionError",
    "WorkflowError",
]


class WorkflowError(PromptureError):
    """Base class for all workflow-engine errors."""


class GraphValidationError(WorkflowError, ValueError):
    """A graph failed structural validation (bad ids, dangling edges/refs)."""


class CycleError(GraphValidationError):
    """The graph is not acyclic."""

    def __init__(self, message: str, *, nodes: list[str] | None = None) -> None:
        super().__init__(message)
        self.nodes = nodes or []


class ReferenceResolutionError(WorkflowError, KeyError):
    """A ``{{...}}`` template reference could not be resolved."""

    def __init__(self, raw: str, message: str | None = None) -> None:
        self.raw = raw
        # KeyError repr quotes its arg; pass a clean message string.
        WorkflowError.__init__(self, message or f"Could not resolve reference {raw!r}")

    def __str__(self) -> str:  # avoid KeyError's quoting
        return self.args[0] if self.args else f"Could not resolve reference {self.raw!r}"


class HandleTypeError(WorkflowError, TypeError):
    """A value did not satisfy a handle's declared type (strict mode)."""


class NodeExecutionError(WorkflowError, RuntimeError):
    """A node's runner raised. Carries the offending ``node_id`` and chains the cause."""

    def __init__(self, node_id: str, message: str) -> None:
        self.node_id = node_id
        super().__init__(f"Node {node_id!r}: {message}")
