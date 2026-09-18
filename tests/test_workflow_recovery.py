"""Phase F contract checks: durable workflow state, resume, reconciliation.

The three crash windows the roadmap names each get a test: before the external
action, after it succeeded but before the result was saved, and after it was
saved.  The middle one is the interesting case — the run must surface a
reconciliation state rather than silently repeating or silently skipping.
"""

from __future__ import annotations

import pytest

from prompture.execution.sandbox import InventoryWorld
from prompture.groups.types import ErrorPolicy
from prompture.workflow import Graph, Node, register_node_type, unregister_node_type
from prompture.workflow.errors import WorkflowError
from prompture.workflow.model import Handle
from prompture.workflow.recovery import (
    FileWorkflowStateStore,
    HumanInputRequired,
    InMemoryWorkflowStateStore,
    OperationLedger,
    OperationStatus,
    ReconciliationRequired,
    ResumableGraphRunner,
    RunLifecycle,
    SQLiteWorkflowStateStore,
    WorkflowRunState,
    WorkflowVersionMismatch,
    graph_version,
    operation_id,
)
from prompture.workflow.state import NodeResult, NodeStatus

# ---------------------------------------------------------------------------
# Test node types
# ---------------------------------------------------------------------------

CALLS: dict[str, int] = {}
CRASH_ON: dict[str, str] = {}  # node_id -> "before" | "after_effect" | ""
WORLD = InventoryWorld({"stock": {"SKU-1": {"A": 10, "B": 0}}})


class _Boom(RuntimeError):
    """Simulates the process dying mid-node."""


def _counting_node(node, config, ctx):
    """A pure node that records how many times it actually executed."""
    CALLS[node.id] = CALLS.get(node.id, 0) + 1
    if CRASH_ON.get(node.id) == "before":
        raise _Boom(f"crash before {node.id} did anything")
    return NodeResult(
        node_id=node.id,
        status=NodeStatus.COMPLETED,
        outputs={"value": f"{config.get('label', node.id)}#{CALLS[node.id]}"},
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
        cost=0.001,
    )


def _effect_node(node, config, ctx):
    """A node whose work has a real external effect, guarded by the ledger."""
    CALLS[node.id] = CALLS.get(node.id, 0) + 1
    ledger: OperationLedger = ctx.options["operation_ledger"]
    run_id = ctx.options["run_id"]
    op = operation_id(run_id, node.id, tool="move_stock")

    if ledger.already_succeeded(op):
        return NodeResult(node.id, NodeStatus.COMPLETED, outputs={"value": ledger.result_of(op)})

    if CRASH_ON.get(node.id) == "before":
        raise _Boom("crashed before dispatching the external call")

    ledger.begin(op, node_id=node.id, tool="move_stock", arguments={"qty": 1})
    result = WORLD.move_stock("SKU-1", "A", "B", 1, operation_id=op)

    if CRASH_ON.get(node.id) == "after_effect":
        # The external system acted; the process dies before the outcome is
        # written back. This is the ambiguous window.
        raise _Boom("crashed after the external call succeeded")

    ledger.succeed(op, result)
    return NodeResult(node.id, NodeStatus.COMPLETED, outputs={"value": result})


def _pausing_node(node, config, ctx):
    CALLS[node.id] = CALLS.get(node.id, 0) + 1
    answer = (ctx.options.get("human_input") or {}).get(node.id)
    if answer is None:
        raise HumanInputRequired("Approve the transfer?", node_id=node.id, schema={"type": "boolean"})
    return NodeResult(node.id, NodeStatus.COMPLETED, outputs={"value": f"approved={answer}"})


@pytest.fixture(autouse=True)
def _register_test_nodes():
    global WORLD
    CALLS.clear()
    CRASH_ON.clear()
    WORLD = InventoryWorld({"stock": {"SKU-1": {"A": 10, "B": 0}}})
    register_node_type("t_count", _counting_node, output_handles=(Handle("value"),), overwrite=True)
    register_node_type("t_effect", _effect_node, output_handles=(Handle("value"),), overwrite=True)
    register_node_type("t_pause", _pausing_node, output_handles=(Handle("value"),), overwrite=True)
    yield
    for name in ("t_count", "t_effect", "t_pause"):
        unregister_node_type(name)


def _chain(*node_specs) -> Graph:
    """Build a linear graph from ``(id, type)`` pairs, wired by reference."""
    graph = Graph(id="chain")
    previous = None
    for node_id, node_type in node_specs:
        config = {"label": node_id}
        if previous is not None:
            config["upstream"] = f"{{{{{previous}.outputs.value}}}}"
        graph.add_node(Node(id=node_id, type=node_type, config=config))
        previous = node_id
    graph.outputs = {"final": f"{{{{{previous}.outputs.value}}}}"}
    return graph


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def test_a_completed_run_persists_definition_inputs_results_and_usage():
    graph = _chain(("a", "t_count"), ("b", "t_count"))
    runner = ResumableGraphRunner()

    result = runner.start(graph, {"topic": "x"}, run_id="run-1")

    assert result.ok is True
    state = runner.state("run-1")
    assert state.status is RunLifecycle.COMPLETED
    assert state.graph_version == graph_version(graph)
    assert state.inputs == {"topic": "x"}
    assert set(state.completed_nodes) == {"a", "b"}
    assert state.pending == []
    assert state.usage["cost"] == pytest.approx(0.002)
    assert state.usage["call_count"] == 2


def test_state_is_persisted_before_the_first_node_runs():
    """Even a crash inside node one leaves a resumable record."""
    graph = _chain(("a", "t_count"))
    store = InMemoryWorkflowStateStore()
    CRASH_ON["a"] = "before"

    ResumableGraphRunner(store=store).start(graph, run_id="run-1")

    state = store.load("run-1")
    assert state is not None
    assert state.graph_version == graph_version(graph)
    assert state.status is RunLifecycle.FAILED


@pytest.mark.parametrize(
    "store_factory",
    [
        lambda tmp: InMemoryWorkflowStateStore(),
        lambda tmp: FileWorkflowStateStore(tmp / "runs"),
        lambda tmp: SQLiteWorkflowStateStore(tmp / "runs.db"),
    ],
    ids=["memory", "file", "sqlite"],
)
def test_every_store_round_trips_a_run(tmp_path, store_factory):
    store = store_factory(tmp_path)
    graph = _chain(("a", "t_count"), ("b", "t_count"))
    ResumableGraphRunner(store=store).start(graph, {"k": "v"}, run_id="run-1")

    loaded = store.load("run-1")
    assert loaded.run_id == "run-1"
    assert loaded.inputs == {"k": "v"}
    assert loaded.status is RunLifecycle.COMPLETED
    assert [r.run_id for r in store.list_runs()] == ["run-1"]
    assert store.delete("run-1") is True
    assert store.load("run-1") is None


def test_the_in_memory_store_snapshots_rather_than_aliasing():
    store = InMemoryWorkflowStateStore()
    state = WorkflowRunState(run_id="r", graph_id="g", graph_version="v")
    store.save(state)

    state.inputs["mutated"] = True  # after "persisting"

    assert store.load("r").inputs == {}, "a persisted snapshot must not change under us"


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_resume_does_not_re_run_completed_nodes():
    graph = _chain(("a", "t_count"), ("b", "t_count"), ("c", "t_count"))
    store = InMemoryWorkflowStateStore()
    runner = ResumableGraphRunner(store=store)

    CRASH_ON["c"] = "before"
    first = runner.start(graph, run_id="run-1")
    assert first.ok is False
    assert CALLS == {"a": 1, "b": 1, "c": 1}

    CRASH_ON.clear()
    second = runner.resume(graph, "run-1")

    assert second.ok is True
    assert CALLS["a"] == 1, "a completed node must never run twice"
    assert CALLS["b"] == 1
    assert CALLS["c"] == 2, "only the failed node is retried"


def test_resumed_outputs_use_the_persisted_upstream_results():
    graph = _chain(("a", "t_count"), ("b", "t_count"))
    runner = ResumableGraphRunner()
    CRASH_ON["b"] = "before"
    runner.start(graph, run_id="run-1")

    CRASH_ON.clear()
    result = runner.resume(graph, "run-1")

    assert result.outputs["final"] == "b#2"
    assert result.node_results["a"].outputs["value"] == "a#1", "restored from state, not recomputed"


def test_cost_is_not_double_counted_across_a_resume():
    graph = _chain(("a", "t_count"), ("b", "t_count"), ("c", "t_count"))
    runner = ResumableGraphRunner()
    CRASH_ON["c"] = "before"
    runner.start(graph, run_id="run-1")

    CRASH_ON.clear()
    result = runner.resume(graph, "run-1")

    assert result.total_usage["cost"] == pytest.approx(0.003), "three nodes, three costs, one each"
    assert result.total_usage["call_count"] == 3


def test_resuming_an_unknown_run_raises():
    with pytest.raises(KeyError):
        ResumableGraphRunner().resume(_chain(("a", "t_count")), "nope")


def test_a_completed_run_cannot_be_resumed():
    graph = _chain(("a", "t_count"))
    runner = ResumableGraphRunner()
    runner.start(graph, run_id="run-1")

    with pytest.raises(WorkflowError, match="already completed"):
        runner.resume(graph, "run-1")


# ---------------------------------------------------------------------------
# Definition changes
# ---------------------------------------------------------------------------


def test_a_changed_definition_is_refused_by_default():
    graph = _chain(("a", "t_count"), ("b", "t_count"))
    runner = ResumableGraphRunner()
    CRASH_ON["b"] = "before"
    runner.start(graph, run_id="run-1")

    changed = _chain(("a", "t_count"), ("b", "t_count"), ("c", "t_count"))
    CRASH_ON.clear()

    with pytest.raises(WorkflowVersionMismatch) as excinfo:
        runner.resume(changed, "run-1")
    assert excinfo.value.persisted != excinfo.value.current


def test_restart_discards_completed_results_from_the_old_definition():
    graph = _chain(("a", "t_count"), ("b", "t_count"))
    runner = ResumableGraphRunner()
    CRASH_ON["b"] = "before"
    runner.start(graph, run_id="run-1")

    changed = _chain(("a", "t_count"), ("b", "t_count"), ("c", "t_count"))
    CRASH_ON.clear()
    result = runner.resume(changed, "run-1", on_mismatch="restart")

    assert result.ok is True
    assert CALLS["a"] == 2, "restart re-runs everything against the new definition"


def test_force_keeps_old_results_and_records_that_it_did():
    graph = _chain(("a", "t_count"), ("b", "t_count"))
    runner = ResumableGraphRunner()
    CRASH_ON["b"] = "before"
    runner.start(graph, run_id="run-1")

    changed = _chain(("a", "t_count"), ("b", "t_count"), ("c", "t_count"))
    CRASH_ON.clear()
    result = runner.resume(changed, "run-1", on_mismatch="force")

    assert result.ok is True
    assert CALLS["a"] == 1, "force keeps the previous definition's result"
    state = runner.state("run-1")
    assert state.metadata["version_mismatch_forced"], "the compromise is recorded, not hidden"
    assert state.graph_version == graph_version(changed)


def test_an_unknown_mismatch_policy_is_rejected():
    graph = _chain(("a", "t_count"))
    runner = ResumableGraphRunner()
    CRASH_ON["a"] = "before"
    runner.start(graph, run_id="run-1")

    with pytest.raises(ValueError, match="on_mismatch must be"):
        runner.resume(_chain(("a", "t_count"), ("b", "t_count")), "run-1", on_mismatch="whatever")


# ---------------------------------------------------------------------------
# Pause and cancel
# ---------------------------------------------------------------------------


def test_a_node_can_pause_the_run_for_human_input():
    graph = _chain(("a", "t_count"), ("gate", "t_pause"), ("c", "t_count"))
    runner = ResumableGraphRunner()

    runner.start(graph, run_id="run-1")

    state = runner.state("run-1")
    assert state.status is RunLifecycle.PAUSED
    assert state.paused_node == "gate"
    assert state.pause_prompt == "Approve the transfer?"
    assert state.pause_schema == {"type": "boolean"}
    assert "c" not in state.node_results, "downstream work must not run while paused"


def test_resuming_a_paused_run_needs_the_answer():
    graph = _chain(("a", "t_count"), ("gate", "t_pause"))
    runner = ResumableGraphRunner()
    runner.start(graph, run_id="run-1")

    with pytest.raises(WorkflowError, match="waiting for input"):
        runner.resume(graph, "run-1")

    result = runner.resume(graph, "run-1", human_input=True)

    assert result.ok is True
    assert result.node_results["gate"].outputs["value"] == "approved=True"
    assert CALLS["a"] == 1, "the pause did not cost the earlier node a second run"


def test_cancellation_is_persisted_and_stops_further_scheduling():
    graph = _chain(("a", "t_count"), ("b", "t_count"))
    store = InMemoryWorkflowStateStore()
    runner = ResumableGraphRunner(store=store)
    CRASH_ON["b"] = "before"
    runner.start(graph, run_id="run-1")

    state = runner.cancel("run-1", reason="operator stopped it")

    assert state.status is RunLifecycle.CANCELLED
    assert state.error == "operator stopped it"
    with pytest.raises(WorkflowError, match="already cancelled"):
        runner.resume(graph, "run-1")


# ---------------------------------------------------------------------------
# The three crash windows
# ---------------------------------------------------------------------------


def test_crash_before_the_action_is_safe_to_retry():
    graph = _chain(("effect", "t_effect"))
    runner = ResumableGraphRunner()
    CRASH_ON["effect"] = "before"

    runner.start(graph, run_id="run-1")

    assert WORLD.snapshot()["stock"] == {"SKU-1": {"A": 10}}, "no external effect happened"
    state = runner.state("run-1")
    assert state.ledger().records() == [], "nothing was even dispatched"

    CRASH_ON.clear()
    result = runner.resume(graph, "run-1")

    assert result.ok is True
    assert WORLD.snapshot()["stock"] == {"SKU-1": {"A": 9, "B": 1}}, "the action ran exactly once"


def test_crash_after_the_action_but_before_saving_requires_reconciliation():
    graph = _chain(("effect", "t_effect"))
    runner = ResumableGraphRunner()
    CRASH_ON["effect"] = "after_effect"

    runner.start(graph, run_id="run-1")

    # The external world changed even though the run recorded no success.
    assert WORLD.snapshot()["stock"] == {"SKU-1": {"A": 9, "B": 1}}
    state = runner.state("run-1")
    unsettled = state.ledger().unsettled()
    assert len(unsettled) == 1
    assert unsettled[0].status is OperationStatus.PENDING

    CRASH_ON.clear()
    with pytest.raises(ReconciliationRequired) as excinfo:
        runner.resume(graph, "run-1")

    assert len(excinfo.value.operations) == 1
    assert runner.state("run-1").status is RunLifecycle.RECONCILE_REQUIRED
    assert WORLD.snapshot()["stock"] == {"SKU-1": {"A": 9, "B": 1}}, "the run did not blindly repeat it"


def test_a_reconciler_can_resolve_the_ambiguous_window():
    graph = _chain(("effect", "t_effect"))
    runner = ResumableGraphRunner()
    CRASH_ON["effect"] = "after_effect"
    runner.start(graph, run_id="run-1")
    CRASH_ON.clear()

    def reconcile(record):
        # A real reconciler queries the external system. Here the sandbox's
        # idempotent operation ledger already knows the truth.
        return (
            OperationStatus.SUCCEEDED if record.operation_id in WORLD.applied_operation_ids else OperationStatus.FAILED
        )

    result = runner.resume(graph, "run-1", reconcile=reconcile)

    assert result.ok is True
    assert WORLD.snapshot()["stock"] == {"SKU-1": {"A": 9, "B": 1}}, "still exactly one effect"


def test_a_reconciler_that_still_cannot_tell_leaves_the_run_blocked():
    graph = _chain(("effect", "t_effect"))
    runner = ResumableGraphRunner()
    CRASH_ON["effect"] = "after_effect"
    runner.start(graph, run_id="run-1")
    CRASH_ON.clear()

    with pytest.raises(ReconciliationRequired):
        runner.resume(graph, "run-1", reconcile=lambda record: OperationStatus.RECONCILE_REQUIRED)

    assert runner.state("run-1").status is RunLifecycle.RECONCILE_REQUIRED


def test_crash_after_saving_skips_the_completed_node():
    graph = _chain(("effect", "t_effect"), ("after", "t_count"))
    runner = ResumableGraphRunner()
    CRASH_ON["after"] = "before"

    runner.start(graph, run_id="run-1")

    assert WORLD.snapshot()["stock"] == {"SKU-1": {"A": 9, "B": 1}}
    assert runner.state("run-1").ledger().unsettled() == [], "the operation was confirmed before the crash"

    CRASH_ON.clear()
    result = runner.resume(graph, "run-1")

    assert result.ok is True
    assert CALLS["effect"] == 1, "a settled node is not re-executed at all"
    assert WORLD.snapshot()["stock"] == {"SKU-1": {"A": 9, "B": 1}}, "no repeated effect"


# ---------------------------------------------------------------------------
# Operation ledger and identifiers
# ---------------------------------------------------------------------------


def test_operation_ids_are_stable_across_processes():
    first = operation_id("run-1", "node-a", tool="charge", index=0)
    second = operation_id("run-1", "node-a", tool="charge", index=0)

    assert first == second
    assert first != operation_id("run-1", "node-a", tool="charge", index=1)
    assert first != operation_id("run-2", "node-a", tool="charge", index=0)


def test_the_ledger_records_intent_before_the_outcome():
    ledger = OperationLedger()
    op = operation_id("r", "n", tool="charge")

    ledger.begin(op, node_id="n", tool="charge", arguments={"amount": 10})
    record = ledger.get(op)
    assert record.status is OperationStatus.PENDING
    assert ledger.unsettled() == [record]

    ledger.succeed(op, {"receipt": "abc"})
    assert ledger.already_succeeded(op) is True
    assert ledger.result_of(op) == {"receipt": "abc"}
    assert ledger.unsettled() == []


def test_a_ledger_round_trips_through_the_persisted_state():
    ledger = OperationLedger()
    op = operation_id("r", "n")
    ledger.begin(op, node_id="n", tool="charge")
    ledger.succeed(op, "ok")

    restored = OperationLedger.from_list(ledger.to_list())

    assert restored.already_succeeded(op) is True
    assert restored.result_of(op) == "ok"


def test_a_failed_operation_is_distinct_from_an_unconfirmed_one():
    ledger = OperationLedger()
    failed, unknown = operation_id("r", "a"), operation_id("r", "b")
    ledger.begin(failed, node_id="a")
    ledger.fail(failed, "rejected by the API before any effect")
    ledger.begin(unknown, node_id="b")

    assert [r.operation_id for r in ledger.unsettled()] == [unknown]
    assert ledger.get(failed).status is OperationStatus.FAILED


# ---------------------------------------------------------------------------
# Error policy
# ---------------------------------------------------------------------------


def test_continue_on_error_lets_an_independent_branch_finish():
    graph = Graph(id="fan")
    graph.add_node(Node(id="bad", type="t_count", config={"label": "bad"}))
    graph.add_node(Node(id="good", type="t_count", config={"label": "good"}))
    graph.outputs = {"good": "{{good.outputs.value}}"}
    CRASH_ON["bad"] = "before"

    runner = ResumableGraphRunner(error_policy=ErrorPolicy.continue_on_error)
    result = runner.start(graph, run_id="run-1")

    assert result.status.value == "failed"
    assert result.node_results["good"].status is NodeStatus.COMPLETED
    assert result.outputs["good"] == "good#1"


def test_fail_fast_skips_the_rest_and_persists_the_skips():
    graph = _chain(("a", "t_count"), ("b", "t_count"), ("c", "t_count"))
    CRASH_ON["a"] = "before"
    runner = ResumableGraphRunner()

    runner.start(graph, run_id="run-1")

    state = runner.state("run-1")
    assert state.node_results["b"]["status"] == NodeStatus.SKIPPED.value
    assert state.node_results["c"]["status"] == NodeStatus.SKIPPED.value
    assert state.status is RunLifecycle.FAILED


def test_a_reconciler_can_return_the_result_it_recovered():
    """The crash window loses the return value, not just the verdict."""
    graph = _chain(("effect", "t_effect"))
    runner = ResumableGraphRunner()
    CRASH_ON["effect"] = "after_effect"
    runner.start(graph, run_id="run-1")
    CRASH_ON.clear()

    def reconcile(record):
        if record.operation_id not in WORLD.applied_operation_ids:
            return OperationStatus.FAILED
        # Replaying under the same id is a read for an idempotent world.
        recovered = WORLD.move_stock("SKU-1", "A", "B", 1, operation_id=record.operation_id)
        return OperationStatus.SUCCEEDED, recovered

    result = runner.resume(graph, "run-1", reconcile=reconcile)

    assert result.ok is True
    assert result.node_results["effect"].outputs["value"] == {
        "moved": 1,
        "source_remaining": 9,
        "destination_total": 1,
    }
    assert WORLD.snapshot()["stock"] == {"SKU-1": {"A": 9, "B": 1}}, "still exactly one effect"


@pytest.mark.parametrize("settle", [False, True])
@pytest.mark.parametrize("backend", ["file", "sqlite"])
def test_operation_intent_survives_uncaught_process_exit(tmp_path, settle, backend):
    """Intent and settlement are durable before the node returns or raises."""
    store = (
        FileWorkflowStateStore(tmp_path / "states")
        if backend == "file"
        else SQLiteWorkflowStateStore(tmp_path / "states.db")
    )
    effects = []

    def abrupt_node(node, config, ctx):
        ledger = ctx.options["operation_ledger"]
        op = operation_id(ctx.options["run_id"], node.id)
        if ledger.already_succeeded(op):
            return NodeResult(node.id, NodeStatus.COMPLETED, outputs={"value": ledger.result_of(op)})
        ledger.begin(op, node_id=node.id)
        assert store.load("abrupt").ledger().get(op).status is OperationStatus.PENDING
        effects.append("applied")
        if settle:
            ledger.succeed(op, "done")
        raise SystemExit("process exits without an Exception handler")

    register_node_type("t_abrupt", abrupt_node, output_handles=(Handle("value"),), overwrite=True)
    try:
        graph = _chain(("action", "t_abrupt"))
        runner = ResumableGraphRunner(store=store)
        with pytest.raises(SystemExit):
            runner.start(graph, run_id="abrupt")
        assert len(effects) == 1
        if settle:
            runner.resume(graph, "abrupt")
            assert runner.state("abrupt").status is RunLifecycle.COMPLETED
        else:
            with pytest.raises(ReconciliationRequired):
                runner.resume(graph, "abrupt")
        assert len(effects) == 1
    finally:
        unregister_node_type("t_abrupt")
