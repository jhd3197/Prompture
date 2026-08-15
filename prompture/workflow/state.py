"""Execution state, status, and result types for the workflow engine.

``NodeStatus`` / ``RunStatus`` are SUPERSETS of :class:`~prompture.jobs.JobStatus`
(they add READY/SKIPPED/SUSPENDED etc.). ``NodeResult`` / ``RunResult`` mirror the
shape of ``PipelineResult`` / ``GroupResult``. ``NodeResultStore`` + ``ResultView``
are the lookup the reference resolver reads — a ``ResultView`` makes
``{{n.outputs[0].value}}`` / ``{{n.outputs.caption}}`` / ``{{n.value}}`` /
``{{n.status}}`` all resolve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import ReferenceResolutionError

__all__ = [
    "NodeResult",
    "NodeResultStore",
    "NodeStatus",
    "ResultView",
    "RunResult",
    "RunStatus",
    "WorkflowCallbacks",
    "aggregate_usage",
]


class NodeStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

    @property
    def is_terminal(self) -> bool:
        return self in {NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.SKIPPED}

    @property
    def is_success(self) -> bool:
        return self is NodeStatus.COMPLETED


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {RunStatus.COMPLETED, RunStatus.FAILED}


def aggregate_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum per-node usage dicts (driver/extraction shape: ``cost`` + token keys)."""
    total: dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "call_count": 0,
    }
    for u in usages:
        if not u:
            continue
        total["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
        total["completion_tokens"] += int(u.get("completion_tokens", 0) or 0)
        total["total_tokens"] += int(u.get("total_tokens", 0) or 0)
        total["cost"] += float(u.get("cost", 0.0) or 0.0)
        total["call_count"] += 1
    total["cost"] = round(total["cost"], 6)
    return total


@dataclass
class NodeResult:
    node_id: str
    status: NodeStatus
    outputs: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    strategy: str | None = None
    error: str | None = None
    error_type: str | None = None
    elapsed_ms: float = 0.0
    raw: dict[str, Any] | None = None

    @property
    def done(self) -> bool:
        return self.status.is_terminal

    def output(self, name: str) -> Any:
        return self.outputs.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "outputs": self.outputs,
            "usage": self.usage,
            "cost": self.cost,
            "strategy": self.strategy,
            "error": self.error,
            "error_type": self.error_type,
            "elapsed_ms": self.elapsed_ms,
        }


class _Output:
    """A single (name, value) output pair, addressable as ``.value`` / ``.name``."""

    __slots__ = ("name", "value")

    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.value = value


class _OutputsView:
    """Outputs addressable by int (positional → ``_Output``) or name (→ value)."""

    def __init__(self, outputs: dict[str, Any]) -> None:
        self._o = outputs

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            items = list(self._o.items())
            name, value = items[key]
            return _Output(name, value)
        return self._o[key]

    def __getattr__(self, name: str) -> Any:
        try:
            return self._o[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class ResultView:
    """Resolver-facing facade over a :class:`NodeResult`."""

    def __init__(self, result: NodeResult) -> None:
        self._r = result

    @property
    def outputs(self) -> _OutputsView:
        return _OutputsView(self._r.outputs)

    @property
    def value(self) -> Any:
        outs = self._r.outputs
        if len(outs) != 1:
            raise ReferenceResolutionError(
                self._r.node_id, f"'.value' needs exactly one output on {self._r.node_id!r} (has {len(outs)})"
            )
        return next(iter(outs.values()))

    @property
    def status(self) -> str:
        return self._r.status.value

    @property
    def usage(self) -> dict[str, Any]:
        return self._r.usage

    @property
    def cost(self) -> float:
        return self._r.cost

    def __getitem__(self, key: Any) -> Any:
        return _OutputsView(self._r.outputs)[key]


@dataclass
class NodeResultStore:
    """node_id → NodeResult lookup; produces :class:`ResultView`s for the resolver."""

    results: dict[str, NodeResult] = field(default_factory=dict)

    def get(self, node_id: str) -> NodeResult | None:
        return self.results.get(node_id)

    def set(self, node_id: str, result: NodeResult) -> None:
        self.results[node_id] = result

    def view(self, node_id: str) -> ResultView:
        return ResultView(self.results[node_id])

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.results


@dataclass
class RunResult:
    graph_id: str
    status: RunStatus
    node_results: dict[str, NodeResult] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    total_usage: dict[str, Any] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.COMPLETED

    def __getitem__(self, node_id: str) -> NodeResult:
        return self.node_results[node_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "status": self.status.value,
            "node_results": {k: v.to_dict() for k, v in self.node_results.items()},
            "outputs": self.outputs,
            "total_usage": self.total_usage,
            "order": self.order,
            "errors": self.errors,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class WorkflowCallbacks:
    """Optional observability hooks, shaped like ``GroupCallbacks``."""

    on_node_start: Callable[[str], None] | None = None
    on_node_complete: Callable[[str, NodeResult], None] | None = None
    on_node_error: Callable[[str, Exception], None] | None = None
    on_graph_complete: Callable[[RunResult], None] | None = None
