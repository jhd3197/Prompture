"""Phase C contract checks for bounded adaptive selection and measured routing.

The mechanism is tested here; its *benefit* is not, and cannot be, established
offline.  These tests assert that selection stays inside its constraints, that
escalation is bounded and non-universal, that stopping is explicit, and that a
heuristic fallback is never reported as a measurement.
"""

from __future__ import annotations

import time

import pytest

from prompture.execution import (
    EvidencePassage,
    ExecutionRequest,
    ExecutionResult,
    InventoryWorld,
    OutcomeRecord,
    ResourceLimits,
    TaskCategory,
    TerminationReason,
    UsageAccounting,
    ValidationReport,
)
from prompture.execution.policy import (
    AdaptivePolicy,
    EscalationRule,
    MeasurementKey,
    MeasurementStore,
    SelectionRule,
    shadow_compare,
)
from prompture.execution.strategies.base import ExecutionStrategy

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _ScriptedStrategy(ExecutionStrategy):
    """Returns pre-built results in order and records the requests it saw."""

    name = "scripted"
    version = "1"

    def __init__(self, results, *, name="scripted", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self._results = list(results)
        self.requests: list[ExecutionRequest] = []

    def _execute(self, ctx):  # pragma: no cover - run() is overridden
        raise NotImplementedError

    def run(self, request):
        self.requests.append(request)
        if not self._results:
            raise AssertionError(f"{self.name} called more times than scripted")
        result = self._results.pop(0)
        result.strategy = self.name
        result.model = request.model or ""
        return result


def _ok(cost=0.001, calls=1):
    usage = UsageAccounting()
    for _ in range(calls):
        usage.record({"cost": cost, "total_tokens": 10}, model="stub/m")
    return ExecutionResult(
        termination=TerminationReason.COMPLETED,
        answer="fine",
        usage=usage,
        validation=ValidationReport(checked=True, schema_valid=True),
        elapsed_ms=5.0,
    )


def _validation_failed(fields=("age",)):
    usage = UsageAccounting()
    usage.record({"cost": 0.001, "total_tokens": 10}, model="stub/m")
    return ExecutionResult(
        termination=TerminationReason.VALIDATION_FAILED,
        usage=usage,
        validation=ValidationReport(
            checked=True,
            schema_valid=False,
            field_errors={f: "missing" for f in fields},
        ),
        elapsed_ms=5.0,
    )


def _no_evidence():
    usage = UsageAccounting()
    usage.record({"cost": 0.001, "total_tokens": 10}, model="stub/m")
    return ExecutionResult(termination=TerminationReason.INSUFFICIENT_EVIDENCE, usage=usage, elapsed_ms=5.0)


def _rejected():
    usage = UsageAccounting()
    usage.record({"cost": 0.001, "total_tokens": 10}, model="stub/m")
    return ExecutionResult(termination=TerminationReason.REVIEW_REJECTED, usage=usage, elapsed_ms=5.0)


def _failed_validation(result):
    return result.termination is TerminationReason.VALIDATION_FAILED


def _record(*, strategy="direct", model="a/m", score=0.9, cost=0.002, ts=None, latency=100.0, unpriced=False):
    usage = UsageAccounting()
    usage.record({"cost": None if unpriced else cost, "total_tokens": 10}, model=model)
    return OutcomeRecord(
        task_id="t",
        task_category=TaskCategory.EXTRACTION,
        strategy=strategy,
        strategy_version="1",
        model=model,
        score=score,
        correct=score >= 0.99,
        usage=usage,
        elapsed_ms=latency,
        ts=ts if ts is not None else time.time(),
    )


# ---------------------------------------------------------------------------
# MeasurementStore
# ---------------------------------------------------------------------------


def test_store_aggregates_quality_cost_and_latency_per_key():
    store = MeasurementStore(min_observations=2)
    store.extend([_record(score=1.0, cost=0.002), _record(score=0.6, cost=0.004)])

    stats = store.get(MeasurementKey(TaskCategory.EXTRACTION, "direct", "a/m", "1"))

    assert stats.observations == 2
    assert stats.mean_score == pytest.approx(0.8)
    assert stats.mean_cost == pytest.approx(0.003)
    assert stats.mean_latency_ms == pytest.approx(100.0)
    assert stats.success_rate == pytest.approx(1.0)
    assert stats.cost_complete is True


def test_store_refuses_to_answer_from_too_few_observations():
    store = MeasurementStore(min_observations=20)
    store.observe(_record())

    stats, usable, why = store.usable(MeasurementKey(TaskCategory.EXTRACTION, "direct", "a/m", "1"))

    assert usable is False
    assert "need 20" in why
    assert stats.mean_score is not None, "the number exists; it is just not enough of them"


def test_store_treats_old_observations_as_stale():
    old = time.time() - 60 * 24 * 3600
    store = MeasurementStore(min_observations=1, stale_after_seconds=30 * 24 * 3600)
    store.observe(_record(ts=old))

    _stats, usable, why = store.usable(MeasurementKey(TaskCategory.EXTRACTION, "direct", "a/m", "1"))

    assert usable is False
    assert "day(s) old" in why


def test_measurements_are_keyed_by_model_and_strategy_version():
    store = MeasurementStore(min_observations=1)
    store.observe(_record(model="a/m"))

    other = MeasurementKey(TaskCategory.EXTRACTION, "direct", "b/other", "1")
    assert store.get(other).observations == 0, "another model's history is not evidence about this one"


def test_best_picks_by_the_requested_objective():
    store = MeasurementStore(min_observations=2)
    store.extend([_record(model="cheap/m", score=0.7, cost=0.001, latency=50.0) for _ in range(2)])
    store.extend([_record(model="good/m", score=0.95, cost=0.010, latency=500.0) for _ in range(2)])
    keys = [
        MeasurementKey(TaskCategory.EXTRACTION, "direct", "cheap/m", "1"),
        MeasurementKey(TaskCategory.EXTRACTION, "direct", "good/m", "1"),
    ]

    assert store.best(keys, objective="quality")[0].model == "good/m"
    assert store.best(keys, objective="cost")[0].model == "cheap/m"
    assert store.best(keys, objective="latency")[0].model == "cheap/m"


def test_best_returns_none_with_reasons_when_nothing_is_usable():
    store = MeasurementStore(min_observations=50)
    store.observe(_record(model="a/m"))

    key, stats, reason = store.best([MeasurementKey(TaskCategory.EXTRACTION, "direct", "a/m", "1")])

    assert key is None and stats is None
    assert "need 50" in reason


def test_best_flags_a_cost_comparison_built_on_unpriced_calls():
    store = MeasurementStore(min_observations=2)
    store.extend([_record(model="a/m", unpriced=True) for _ in range(2)])

    _key, _stats, reason = store.best([MeasurementKey(TaskCategory.EXTRACTION, "direct", "a/m", "1")], objective="cost")

    assert "LOWER BOUND" in reason


def test_store_round_trips_through_a_jsonl_outcome_file(tmp_path):
    from prompture.execution import JsonlOutcomeStore

    jsonl = JsonlOutcomeStore(tmp_path / "outcomes.jsonl")
    jsonl.extend([_record(score=0.5), _record(score=1.0)])

    store = MeasurementStore.from_jsonl(tmp_path / "outcomes.jsonl", min_observations=2)
    stats = store.get(MeasurementKey(TaskCategory.EXTRACTION, "direct", "a/m", "1"))

    assert stats.observations == 2
    assert stats.mean_score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _policy(**kwargs):
    kwargs.setdefault(
        "strategies",
        {
            "direct": _ScriptedStrategy([_ok()], name="direct"),
            "retrieve_verify": _ScriptedStrategy([_ok()], name="retrieve_verify"),
        },
    )
    kwargs.setdefault("default_strategy", "direct")
    return AdaptivePolicy(**kwargs)


def test_an_evidence_task_selects_the_grounded_strategy_by_rule():
    policy = _policy()
    request = ExecutionRequest(
        task="what does the policy say?",
        category=TaskCategory.DOCUMENT_QA,
        passages=(EvidencePassage(id="p1", text="14 days."),),
    )

    result = policy.run(request)

    assert result.strategy == "retrieve_verify"
    assert any("rule evidence_required" in d for d in result.decisions)


def test_no_matching_rule_falls_back_to_the_configured_cold_start_default():
    policy = _policy()
    result = policy.run(ExecutionRequest(task="plain task"))

    assert result.strategy == "direct"
    assert any("cold-start default" in d for d in result.decisions)


def test_the_policy_marks_itself_experimental_in_its_own_decision_log():
    result = _policy().run(ExecutionRequest(task="x"))
    assert any("EXPERIMENTAL" in d for d in result.decisions)

    quiet = _policy(experimental=False).run(ExecutionRequest(task="x"))
    assert not any("EXPERIMENTAL" in d for d in quiet.decisions)


def test_a_pinned_model_outside_the_eligible_list_is_not_used():
    policy = _policy(eligible_models=["allowed/one"])
    result = policy.run(ExecutionRequest(task="x", model="forbidden/two"))

    assert result.model == "allowed/one"
    assert policy.strategies["direct"].requests[0].model == "allowed/one"


def test_a_pinned_eligible_model_is_respected():
    policy = _policy(eligible_models=["allowed/one", "allowed/two"])
    result = policy.run(ExecutionRequest(task="x", model="allowed/two"))

    assert result.model == "allowed/two"
    assert any("pinned by the caller" in d for d in result.decisions)


def test_without_measurements_model_choice_is_reported_as_configured_not_measured():
    policy = _policy(eligible_models=["a/one", "b/two"])
    result = policy.run(ExecutionRequest(task="x"))

    reasons = " ".join(result.decisions)
    assert "no measurement store configured" in reasons
    assert "measured:" not in reasons


def test_a_usable_measurement_drives_and_labels_the_model_choice():
    store = MeasurementStore(min_observations=2)
    store.extend([_record(strategy="direct", model="b/two", score=0.95) for _ in range(2)])
    store.extend([_record(strategy="direct", model="a/one", score=0.40) for _ in range(2)])

    policy = _policy(eligible_models=["a/one", "b/two"], measurements=store)
    result = policy.run(ExecutionRequest(task="x", category=TaskCategory.EXTRACTION))

    assert result.model == "b/two"
    assert any("measured:" in d for d in result.decisions)


def test_a_heuristic_fallback_is_labelled_as_a_heuristic(monkeypatch):
    policy = _policy(eligible_models=["a/one", "b/two"], use_heuristic_router=True)
    monkeypatch.setattr(
        AdaptivePolicy,
        "_heuristic_model",
        lambda self, request, allowed: ("b/two", "price tier guess"),
    )

    result = policy.run(ExecutionRequest(task="x"))

    reasons = " ".join(result.decisions)
    assert result.model == "b/two"
    assert "heuristic router chose b/two" in reasons
    assert "measured:" not in reasons


def test_custom_selection_rules_take_precedence_in_order():
    policy = _policy(
        selection_rules=(
            SelectionRule(
                name="always_retrieve",
                when=lambda r: True,
                strategy="retrieve_verify",
                reason="testing rule ordering",
            ),
        )
    )
    result = policy.run(ExecutionRequest(task="x"))

    assert result.strategy == "retrieve_verify"
    assert any("always_retrieve" in d for d in result.decisions)


def test_a_rule_naming_an_unconfigured_strategy_is_skipped():
    policy = _policy(
        selection_rules=(SelectionRule(name="ghost", when=lambda r: True, strategy="not_configured", reason="x"),)
    )
    assert policy.run(ExecutionRequest(task="x")).strategy == "direct"


def test_a_raising_rule_predicate_does_not_kill_the_run():
    def boom(_request):
        raise RuntimeError("bad predicate")

    policy = _policy(selection_rules=(SelectionRule("boom", boom, "retrieve_verify", "x"),))
    assert policy.run(ExecutionRequest(task="x")).strategy == "direct"


def test_a_policy_needs_a_configured_default():
    with pytest.raises(ValueError, match="not among the configured"):
        AdaptivePolicy(strategies={"direct": _ScriptedStrategy([])}, default_strategy="nope")


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


def test_failed_field_validation_escalates_to_one_targeted_repair():
    direct = _ScriptedStrategy([_validation_failed(), _ok()], name="direct")
    policy = AdaptivePolicy(strategies={"direct": direct}, default_strategy="direct")

    result = policy.run(ExecutionRequest(task="x"))

    assert result.termination is TerminationReason.COMPLETED
    assert len(direct.requests) == 2
    assert any("repair_failed_fields" in d for d in result.decisions)


def test_escalation_merges_usage_and_steps_from_both_attempts():
    direct = _ScriptedStrategy([_validation_failed(), _ok(calls=2)], name="direct")
    policy = AdaptivePolicy(strategies={"direct": direct}, default_strategy="direct")

    result = policy.run(ExecutionRequest(task="x"))

    assert result.usage.call_count == 3, "the failed attempt is still billed and still counted"
    assert result.usage.cost == pytest.approx(0.003)


def test_each_escalation_rule_fires_at_most_once_per_run():
    """A rule that keeps matching cannot re-trigger itself into a loop."""
    direct = _ScriptedStrategy([_validation_failed(), _validation_failed()], name="direct")
    policy = AdaptivePolicy(strategies={"direct": direct}, default_strategy="direct", max_escalations=5)

    result = policy.run(ExecutionRequest(task="x"))

    assert len(direct.requests) == 2, "one attempt plus exactly one escalation, despite max_escalations=5"
    assert result.termination is TerminationReason.VALIDATION_FAILED


def test_the_escalation_bound_stops_a_chain_of_different_rules():
    first = _ScriptedStrategy([_validation_failed()], name="direct")
    second = _ScriptedStrategy([_validation_failed()], name="second")
    third = _ScriptedStrategy([_ok()], name="third")
    policy = AdaptivePolicy(
        strategies={"direct": first, "second": second, "third": third},
        default_strategy="direct",
        max_escalations=1,
        escalation_rules=(
            EscalationRule("to_second", _failed_validation, "second", "first hop"),
            EscalationRule("to_third", _failed_validation, "third", "second hop"),
        ),
    )

    result = policy.run(ExecutionRequest(task="x"))

    assert len(second.requests) == 1
    assert third.requests == [], "the second hop is outside the bound"
    assert result.termination is TerminationReason.VALIDATION_FAILED
    assert any("escalation bound reached" in d for d in result.decisions)


def test_max_escalations_zero_disables_escalation_entirely():
    direct = _ScriptedStrategy([_validation_failed()], name="direct")
    policy = AdaptivePolicy(strategies={"direct": direct}, default_strategy="direct", max_escalations=0)

    result = policy.run(ExecutionRequest(task="x"))

    assert len(direct.requests) == 1
    assert result.termination is TerminationReason.VALIDATION_FAILED


def test_missing_evidence_escalates_to_retrieval_exactly_once():
    direct = _ScriptedStrategy([_no_evidence()], name="direct")
    retrieve = _ScriptedStrategy([_ok()], name="retrieve_verify")
    policy = AdaptivePolicy(
        strategies={"direct": direct, "retrieve_verify": retrieve},
        default_strategy="direct",
    )

    result = policy.run(ExecutionRequest(task="x"))

    assert len(retrieve.requests) == 1
    assert result.strategy == "retrieve_verify"
    assert any("ground_missing_evidence" in d for d in result.decisions)


def test_there_is_no_universal_escalation_chain():
    """A rejected review does not summon a debate or a pricier model by default."""
    direct = _ScriptedStrategy([_rejected()], name="direct")
    expensive = _ScriptedStrategy([_ok()], name="expensive")
    policy = AdaptivePolicy(
        strategies={"direct": direct, "expensive": expensive},
        default_strategy="direct",
    )

    result = policy.run(ExecutionRequest(task="x"))

    assert expensive.requests == [], "no default rule escalates a review rejection"
    assert result.termination is TerminationReason.REVIEW_REJECTED


def test_a_host_can_add_its_own_escalation_rule():
    direct = _ScriptedStrategy([_rejected()], name="direct")
    second = _ScriptedStrategy([_ok()], name="second")
    policy = AdaptivePolicy(
        strategies={"direct": direct, "second": second},
        default_strategy="direct",
        escalation_rules=(
            EscalationRule(
                name="retry_rejected",
                when=lambda r: r.termination is TerminationReason.REVIEW_REJECTED,
                strategy="second",
                reason="this host wants one more pass after a rejection",
            ),
        ),
    )

    result = policy.run(ExecutionRequest(task="x"))

    assert result.termination is TerminationReason.COMPLETED
    assert any("retry_rejected" in d for d in result.decisions)


# ---------------------------------------------------------------------------
# Stopping and hard constraints
# ---------------------------------------------------------------------------


def test_an_impossible_budget_abstains_before_spending_anything():
    direct = _ScriptedStrategy([_ok()], name="direct")
    policy = AdaptivePolicy(strategies={"direct": direct}, default_strategy="direct")

    result = policy.run(ExecutionRequest(task="x", limits=ResourceLimits(max_llm_calls=0)))

    assert result.termination is TerminationReason.ABSTAINED
    assert direct.requests == []
    assert any("no attempt is possible" in d for d in result.decisions)


def test_an_evidence_task_with_no_evidence_source_abstains():
    direct = _ScriptedStrategy([_ok()], name="direct")
    policy = AdaptivePolicy(strategies={"direct": direct}, default_strategy="direct")

    result = policy.run(ExecutionRequest(task="x", category=TaskCategory.DOCUMENT_QA))

    assert result.termination is TerminationReason.ABSTAINED
    assert any("neither passages nor a retriever" in d for d in result.decisions)


def test_narrow_tools_never_widens_the_callers_allow_list():
    policy = _policy()
    request = ExecutionRequest(task="x", allowed_tools=frozenset({"read"}))

    assert policy.narrow_tools(request, ["read", "write"]) == frozenset({"read"})
    assert policy.narrow_tools(request, ["write"]) == frozenset()

    open_request = ExecutionRequest(task="x")
    assert policy.narrow_tools(open_request, ["read"]) == frozenset({"read"})


def test_the_policy_is_usable_directly_as_a_harness_executor():
    from prompture.execution import Split, load_bundled_fixture_set, run_fixture_set

    fixtures = load_bundled_fixture_set("extraction_contacts")
    n = len(fixtures.dev())
    policy = AdaptivePolicy(
        strategies={"direct": _ScriptedStrategy([_ok() for _ in range(n)], name="direct")},
        default_strategy="direct",
    )

    report = run_fixture_set(fixtures, policy, split=Split.DEV, repeats=1, label="adaptive")

    assert report.summaries[0].runs == n


# ---------------------------------------------------------------------------
# Shadow comparison
# ---------------------------------------------------------------------------


def test_shadow_compare_refuses_a_request_with_real_write_tools():
    from prompture.agents.tools_schema import ToolRegistry

    registry = ToolRegistry()

    def send_invoice(customer: str) -> str:
        """Send an invoice.

        Args:
            customer: Who to bill.
        """
        return "sent"

    registry.register(send_invoice, metadata={"is_write": True})
    request = ExecutionRequest(task="bill them", tools=registry)

    with pytest.raises(ValueError, match="perform each action twice"):
        shadow_compare(request, _ScriptedStrategy([_ok()]), _ScriptedStrategy([_ok()]))


def test_shadow_compare_allows_a_sandboxed_world():
    world = InventoryWorld({"stock": {"SKU-1": {"A": 3}}})
    request = ExecutionRequest(task="move one", tools=world.as_tool_registry())

    results = shadow_compare(
        request,
        _ScriptedStrategy([_ok()], name="baseline"),
        _ScriptedStrategy([_ok()], name="candidate"),
    )

    assert set(results) == {"baseline", "candidate"}


def test_shadow_compare_can_be_forced_for_idempotent_tools():
    from prompture.agents.tools_schema import ToolRegistry

    registry = ToolRegistry()

    def upsert(key: str) -> str:
        """Idempotent upsert.

        Args:
            key: The key.
        """
        return "ok"

    registry.register(upsert, metadata={"is_write": True})
    request = ExecutionRequest(task="x", tools=registry)

    results = shadow_compare(
        request,
        _ScriptedStrategy([_ok()]),
        _ScriptedStrategy([_ok()]),
        allow_side_effects=True,
    )
    assert len(results) == 2
