"""Synchronous execution engine internals: the status-aware topological run loop.

One algorithm backs both :func:`run_graph` (full DAG) and :func:`run_node`
(single node). Nodes run in :meth:`Graph.topo_order`; a node whose predecessor
failed or was skipped is itself SKIPPED (skip-propagation); ``fail_fast`` stops
scheduling after the first failure while ``continue_on_error`` lets independent
branches finish.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from ..groups.types import ErrorPolicy
from .errors import NodeExecutionError
from .model import Graph, Node
from .references import resolve_template
from .registry import get_node_type
from .state import (
    NodeResult,
    NodeResultStore,
    NodeStatus,
    RunResult,
    RunStatus,
    WorkflowCallbacks,
    aggregate_usage,
)

__all__ = ["ExecutionContext", "run_graph", "run_node"]


class _Scope:
    """Resolver scope: ``inputs`` reserved root + each node id → its ResultView."""

    def __init__(self, store: NodeResultStore, inputs: dict[str, Any]) -> None:
        self._store = store
        self._inputs = inputs

    def __contains__(self, key: str) -> bool:
        return key == "inputs" or key in self._store

    def __getitem__(self, key: str) -> Any:
        if key == "inputs":
            return self._inputs
        return self._store.view(key)


@dataclass
class ExecutionContext:
    """Read-only context threaded into each node runner."""

    graph: Graph
    inputs: dict[str, Any]
    store: NodeResultStore
    options: dict[str, Any] = field(default_factory=dict)
    callbacks: WorkflowCallbacks | None = None

    def scope(self) -> _Scope:
        return _Scope(self.store, self.inputs)

    def resolve(self, value: Any, *, strict: bool = True) -> Any:
        return resolve_template(value, self.scope(), strict=strict)


def _emit(callbacks: WorkflowCallbacks | None, event: str, *args: Any) -> None:
    if callbacks is None:
        return
    cb = getattr(callbacks, event, None)
    if cb is None:
        return
    with contextlib.suppress(Exception):
        cb(*args)


def _run_one(node: Node, ctx: ExecutionContext) -> NodeResult:
    nt = get_node_type(node.type)
    resolved = ctx.resolve(node.config)
    result = nt.run(node, resolved, ctx)
    if not isinstance(result, NodeResult):
        raise NodeExecutionError(node.id, f"node type {node.type!r} returned {type(result).__name__}, not NodeResult")
    return result


def run_graph(
    graph: Graph,
    inputs: dict[str, Any] | None = None,
    *,
    error_policy: ErrorPolicy = ErrorPolicy.fail_fast,
    options: dict[str, Any] | None = None,
    callbacks: WorkflowCallbacks | None = None,
) -> RunResult:
    """Execute every node of *graph* in topological order and return a RunResult."""
    graph.validate()
    inputs = dict(inputs or {})
    store = NodeResultStore()
    ctx = ExecutionContext(graph=graph, inputs=inputs, store=store, options=dict(options or {}), callbacks=callbacks)

    node_map = {n.id: n for n in graph.nodes}
    order = graph.topo_order()
    results: dict[str, NodeResult] = {}
    errors: list[tuple[str, str]] = []
    any_failed = False
    t_run = perf_counter()

    for nid in order:
        node = node_map[nid]
        preds = graph.predecessors(nid)
        pred_blocked = any((p in results and not results[p].status.is_success) for p in preds)
        if pred_blocked or (any_failed and error_policy is ErrorPolicy.fail_fast):
            res = NodeResult(nid, NodeStatus.SKIPPED)
            store.set(nid, res)
            results[nid] = res
            continue

        _emit(callbacks, "on_node_start", nid)
        t0 = perf_counter()
        try:
            res = _run_one(node, ctx)
            res.elapsed_ms = (perf_counter() - t0) * 1000
        except Exception as exc:
            res = NodeResult(
                nid,
                NodeStatus.FAILED,
                error=str(exc),
                error_type=type(exc).__name__,
                elapsed_ms=(perf_counter() - t0) * 1000,
            )
            store.set(nid, res)
            results[nid] = res
            errors.append((nid, str(exc)))
            any_failed = True
            _emit(callbacks, "on_node_error", nid, exc)
            if error_policy is ErrorPolicy.raise_on_error:
                raise NodeExecutionError(nid, str(exc)) from exc
            continue

        store.set(nid, res)
        results[nid] = res
        _emit(callbacks, "on_node_complete", nid, res)

    outputs = _collect_outputs(graph, ctx, results)
    status = RunStatus.FAILED if any_failed else RunStatus.COMPLETED
    run = RunResult(
        graph_id=graph.id,
        status=status,
        node_results=results,
        outputs=outputs,
        total_usage=aggregate_usage([r.usage for r in results.values()]),
        order=order,
        errors=errors,
        error=errors[0][1] if errors else None,
        elapsed_ms=(perf_counter() - t_run) * 1000,
    )
    _emit(callbacks, "on_graph_complete", run)
    return run


def _collect_outputs(graph: Graph, ctx: ExecutionContext, results: dict[str, NodeResult]) -> dict[str, Any]:
    scope = _Scope(ctx.store, ctx.inputs)
    outputs: dict[str, Any] = {}
    for name, template in graph.outputs.items():
        outputs[name] = resolve_template(template, scope, strict=False)
    for node in graph.nodes:
        if node.type == "output":
            res = results.get(node.id)
            if res is not None and res.status.is_success:
                outputs.setdefault(node.config.get("name", node.id), res.outputs.get("value"))
    return outputs


def run_node(
    graph: Graph,
    node_id: str,
    *,
    inputs: dict[str, Any] | None = None,
    upstream: dict[str, dict[str, Any]] | None = None,
    options: dict[str, Any] | None = None,
) -> NodeResult:
    """Run a single node, given its graph-level ``inputs`` and ``upstream`` outputs.

    ``upstream`` maps an upstream node id to its outputs dict; each is wrapped in a
    synthetic completed :class:`NodeResult` so references resolve. Raises
    :class:`NodeExecutionError` if the node's runner fails.
    """
    store = NodeResultStore()
    for up_id, out in (upstream or {}).items():
        store.set(up_id, NodeResult(up_id, NodeStatus.COMPLETED, outputs=dict(out)))
    ctx = ExecutionContext(
        graph=graph, inputs=dict(inputs or {}), store=store, options=dict(options or {}), callbacks=None
    )
    node = graph.node(node_id)
    t0 = perf_counter()
    try:
        res = _run_one(node, ctx)
    except NodeExecutionError:
        raise
    except Exception as exc:
        raise NodeExecutionError(node_id, str(exc)) from exc
    res.elapsed_ms = (perf_counter() - t0) * 1000
    return res
