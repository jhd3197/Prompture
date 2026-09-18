"""Durable workflow runs: crash, resume, and reconcile.

Demonstrates :mod:`prompture.workflow.recovery` by deliberately killing a run in
each of the three crash windows a side-effecting workflow actually has:

1. before the external action — safe to retry;
2. after the action succeeded but before the result was saved — genuinely
   ambiguous, so the run refuses to guess;
3. after the result was saved — the node is skipped entirely.

The "external system" here is
:class:`~prompture.execution.sandbox.InventoryWorld`, a deterministic local
world. That is the point: a shadow or fault-injection run must never duplicate a
real external effect, so the example uses a sandbox rather than a live API.

No model, no API key, and no network are involved — this example runs anywhere.

Run it
~~~~~~

::

    python examples/durable_workflow_example.py
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompture.execution.sandbox import InventoryWorld
from prompture.workflow import Graph, Handle, Node, register_node_type, unregister_node_type
from prompture.workflow.recovery import (
    FileWorkflowStateStore,
    HumanInputRequired,
    OperationLedger,
    OperationStatus,
    ReconciliationRequired,
    ResumableGraphRunner,
    RunLifecycle,
    operation_id,
)
from prompture.workflow.state import NodeResult, NodeStatus

# =============================================================================
# A tiny world plus the nodes that act on it
# =============================================================================

WORLD = InventoryWorld({"stock": {"SKU-1": {"A": 10, "B": 0}}})
CRASH_AT: dict[str, str] = {}  # node id -> "before" | "after_effect"
RUNS: dict[str, int] = {}


class SimulatedCrash(RuntimeError):
    """Stands in for the process dying mid-node."""


def plan_node(node: Any, config: dict[str, Any], ctx: Any) -> NodeResult:
    """A cheap, side-effect-free step."""
    RUNS[node.id] = RUNS.get(node.id, 0) + 1
    return NodeResult(
        node_id=node.id,
        status=NodeStatus.COMPLETED,
        outputs={"value": "move 1 unit of SKU-1 from A to B"},
        usage={"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28, "cost": 0.0004},
        cost=0.0004,
    )


def transfer_node(node: Any, config: dict[str, Any], ctx: Any) -> NodeResult:
    """The side-effecting step, guarded by the operation ledger."""
    RUNS[node.id] = RUNS.get(node.id, 0) + 1
    ledger: OperationLedger = ctx.options["operation_ledger"]
    op = operation_id(ctx.options["run_id"], node.id, tool="move_stock")

    if ledger.already_succeeded(op):
        return NodeResult(node.id, NodeStatus.COMPLETED, outputs={"value": ledger.result_of(op)})

    if CRASH_AT.get(node.id) == "before":
        raise SimulatedCrash("process died before dispatching the transfer")

    ledger.begin(op, node_id=node.id, tool="move_stock", arguments={"qty": 1})
    result = WORLD.move_stock("SKU-1", "A", "B", 1, operation_id=op)

    if CRASH_AT.get(node.id) == "after_effect":
        raise SimulatedCrash("process died after the transfer succeeded, before saving it")

    ledger.succeed(op, result)
    return NodeResult(node.id, NodeStatus.COMPLETED, outputs={"value": result})


def approval_node(node: Any, config: dict[str, Any], ctx: Any) -> NodeResult:
    """A step that pauses for a person."""
    answer = (ctx.options.get("human_input") or {}).get(node.id)
    if answer is None:
        raise HumanInputRequired("Approve the stock transfer?", node_id=node.id, schema={"type": "boolean"})
    return NodeResult(node.id, NodeStatus.COMPLETED, outputs={"value": f"approved={answer}"})


def build_graph(*, with_approval: bool = False) -> Graph:
    graph = Graph(id="transfer")
    graph.add_node(Node(id="plan", type="ex_plan"))
    if with_approval:
        graph.add_node(Node(id="approve", type="ex_approve", config={"plan": "{{plan.outputs.value}}"}))
        graph.add_node(Node(id="transfer", type="ex_transfer", config={"ok": "{{approve.outputs.value}}"}))
    else:
        graph.add_node(Node(id="transfer", type="ex_transfer", config={"plan": "{{plan.outputs.value}}"}))
    graph.outputs = {"result": "{{transfer.outputs.value}}"}
    return graph


def reset_world() -> None:
    global WORLD
    WORLD = InventoryWorld({"stock": {"SKU-1": {"A": 10, "B": 0}}})
    CRASH_AT.clear()
    RUNS.clear()


def show(runner: ResumableGraphRunner, run_id: str, label: str) -> None:
    state = runner.state(run_id)
    print(f"    {label}")
    print(f"      lifecycle : {state.status.value}")
    print(f"      stock     : {WORLD.snapshot()['stock']}")
    print(f"      node runs : {dict(RUNS)}")
    print(f"      cost      : ${state.usage.get('cost', 0.0):.6f}")
    unsettled = [r.operation_id[:8] for r in state.ledger().unsettled()]
    print(f"      unsettled : {unsettled or 'none'}")


# =============================================================================
# The three crash windows
# =============================================================================


def section_crash_before(runner: ResumableGraphRunner) -> None:
    print("\n" + "=" * 78)
    print("1. Crash BEFORE the external action - safe to retry")
    print("=" * 78)
    reset_world()
    CRASH_AT["transfer"] = "before"

    runner.start(build_graph(), run_id="run-before")
    show(runner, "run-before", "after the crash:")

    CRASH_AT.clear()
    result = runner.resume(build_graph(), "run-before")
    show(runner, "run-before", "after the resume:")
    print(f"      outputs   : {result.outputs}")
    print("      -> the transfer happened exactly once, and 'plan' never re-ran")


def section_crash_after_effect(runner: ResumableGraphRunner) -> None:
    print("\n" + "=" * 78)
    print("2. Crash AFTER the action, BEFORE saving - ambiguous, so the run stops")
    print("=" * 78)
    reset_world()
    CRASH_AT["transfer"] = "after_effect"

    runner.start(build_graph(), run_id="run-ambiguous")
    show(runner, "run-ambiguous", "after the crash (the world changed; the run does not know it):")

    CRASH_AT.clear()
    try:
        runner.resume(build_graph(), "run-ambiguous")
    except ReconciliationRequired as exc:
        print(f"      refused   : {exc}")
    show(runner, "run-ambiguous", "after the refused resume:")
    print("      -> no blind repeat; the stock is still 9/1, not 8/2")

    def reconcile(record):
        # A real reconciler queries the external system. The sandbox happens to
        # remember which operation ids it already applied, and replaying under
        # the same id returns the original result rather than moving stock again
        # -- so the lost return value is recovered along with the verdict.
        if record.operation_id not in WORLD.applied_operation_ids:
            return OperationStatus.FAILED
        recovered = WORLD.move_stock("SKU-1", "A", "B", 1, operation_id=record.operation_id)
        return OperationStatus.SUCCEEDED, recovered

    result = runner.resume(build_graph(), "run-ambiguous", reconcile=reconcile)
    show(runner, "run-ambiguous", "after reconciling:")
    print(f"      outputs   : {result.outputs}")


def section_crash_after_save(runner: ResumableGraphRunner) -> None:
    print("\n" + "=" * 78)
    print("3. Crash AFTER saving - the settled node is skipped entirely")
    print("=" * 78)
    reset_world()

    graph = build_graph()
    graph.add_node(Node(id="notify", type="ex_plan", config={"of": "{{transfer.outputs.value}}"}))
    graph.outputs = {"result": "{{transfer.outputs.value}}", "notified": "{{notify.outputs.value}}"}
    CRASH_AT["notify"] = "before"

    # `notify` reuses the plan runner, so make it crash by name.
    def crashing_plan(node, config, ctx):
        if CRASH_AT.get(node.id) == "before":
            RUNS[node.id] = RUNS.get(node.id, 0) + 1
            raise SimulatedCrash("process died in the notify step")
        return plan_node(node, config, ctx)

    register_node_type("ex_plan", crashing_plan, output_handles=(Handle("value"),), overwrite=True)
    try:
        runner.start(graph, run_id="run-after-save")
        show(runner, "run-after-save", "after the crash:")

        CRASH_AT.clear()
        result = runner.resume(graph, "run-after-save")
        show(runner, "run-after-save", "after the resume:")
        print(f"      outputs   : {result.outputs}")
        print("      -> 'transfer' ran once in total; its cost was not counted twice")
    finally:
        register_node_type("ex_plan", plan_node, output_handles=(Handle("value"),), overwrite=True)


def section_pause(runner: ResumableGraphRunner) -> None:
    print("\n" + "=" * 78)
    print("4. Pausing for a person, across a restart")
    print("=" * 78)
    reset_world()
    graph = build_graph(with_approval=True)

    runner.start(graph, run_id="run-approval")
    state = runner.state("run-approval")
    print(f"    lifecycle : {state.status.value}")
    print(f"    waiting on: {state.paused_node} - {state.pause_prompt}")
    print(f"    schema    : {state.pause_schema}")
    print(f"    stock     : {WORLD.snapshot()['stock']}  (nothing downstream ran)")

    result = runner.resume(graph, "run-approval", human_input=True)
    print(f"    resumed   : {result.outputs}")
    print(f"    stock     : {WORLD.snapshot()['stock']}")


def section_definition_change(runner: ResumableGraphRunner) -> None:
    print("\n" + "=" * 78)
    print("5. The graph changed while the run was down")
    print("=" * 78)
    reset_world()
    CRASH_AT["transfer"] = "before"
    runner.start(build_graph(), run_id="run-changed")
    CRASH_AT.clear()

    changed = build_graph()
    changed.add_node(Node(id="audit", type="ex_plan", config={"note": "{{transfer.outputs.value}}"}))

    from prompture.workflow.recovery import WorkflowVersionMismatch

    try:
        runner.resume(changed, "run-changed")
    except WorkflowVersionMismatch as exc:
        print(f"    refused   : {exc}")

    result = runner.resume(changed, "run-changed", on_mismatch="force")
    print(f"    forced    : {result.status.value}, outputs={result.outputs}")
    state = runner.state("run-changed")
    print(f"    recorded  : {state.metadata.get('version_mismatch_forced')}")


def main() -> None:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(errors="replace")

    print("Prompture - durable workflow recovery")
    register_node_type("ex_plan", plan_node, output_handles=(Handle("value"),), overwrite=True)
    register_node_type("ex_transfer", transfer_node, output_handles=(Handle("value"),), overwrite=True)
    register_node_type("ex_approve", approval_node, output_handles=(Handle("value"),), overwrite=True)

    directory = Path(tempfile.mkdtemp(prefix="prompture-runs-"))
    print(f"  persisting run state to {directory}")
    runner = ResumableGraphRunner(store=FileWorkflowStateStore(directory))

    try:
        section_crash_before(runner)
        section_crash_after_effect(runner)
        section_crash_after_save(runner)
        section_pause(runner)
        section_definition_change(runner)

        print("\n" + "=" * 78)
        print("Persisted runs")
        print("=" * 78)
        for state in runner.store.list_runs():
            marker = "!" if state.status is RunLifecycle.RECONCILE_REQUIRED else " "
            print(f"  {marker} {state.run_id:<16} {state.status.value:<20} ${state.usage.get('cost', 0.0):.6f}")

        print(
            "\nExactly-once external effects are NOT promised - the underlying service\n"
            "generally cannot provide them. What is promised: no silent loss, no blind\n"
            "repetition, and an explicit reconciliation state when the truth is unknown."
        )
    finally:
        for name in ("ex_plan", "ex_transfer", "ex_approve"):
            unregister_node_type(name)


if __name__ == "__main__":
    main()
