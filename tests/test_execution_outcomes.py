"""Phase A contract checks: outcome records, fixtures, scoring, gates, harness.

Everything here is deterministic and offline.  These tests verify the
*accounting and reporting contract* — that an unknown price never reads as zero,
that a held-out split cannot be optimised against, that an unanswerable question
is only scored correct when the run abstains.  They say nothing about model
quality, which is measured separately by a live benchmark.
"""

from __future__ import annotations

import json

import pytest

from prompture.execution import (
    DEFAULT_GATES,
    CaseScore,
    EvidenceReport,
    ExecutionResult,
    FixtureCase,
    FixtureSet,
    GateVerdict,
    InventoryWorld,
    JsonlOutcomeStore,
    OutcomeRecord,
    ResourceLimits,
    Split,
    TaskCategory,
    TerminationReason,
    UsageAccounting,
    ValidationReport,
    assert_optimization_safe,
    build_request,
    evaluate_gate,
    load_bundled_fixture_set,
    run_fixture_set,
    save_fixture_set,
    score_case,
    score_document_qa,
    score_extraction,
    score_tool_task,
    summarize_runs,
)
from prompture.execution.harness import CaseRun, percentile
from prompture.execution.types import BudgetLedger, CancellationToken, Cancelled, EvidencePassage

# ---------------------------------------------------------------------------
# UsageAccounting — unknown price must never read as zero
# ---------------------------------------------------------------------------


def test_priced_call_is_counted_and_cost_stays_complete():
    usage = UsageAccounting()
    usage.record({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.002}, model="openai/x")

    assert usage.call_count == 1
    assert usage.unpriced_calls == 0
    assert usage.cost_complete is True
    assert usage.cost == pytest.approx(0.002)
    assert usage.per_model["openai/x"]["calls"] == 1


def test_missing_cost_key_marks_accounting_incomplete():
    usage = UsageAccounting()
    usage.record({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, model="local/mystery")

    assert usage.call_count == 1
    assert usage.unpriced_calls == 1
    assert usage.cost_complete is False
    # The cost is a lower bound, not a total.
    assert usage.cost == 0.0
    assert usage.per_model["local/mystery"]["unpriced"] == 1


def test_explicit_zero_cost_is_priced_not_unknown():
    """A provider that really is free is different from one we can't price."""
    usage = UsageAccounting()
    usage.record({"cost": 0.0, "total_tokens": 3}, model="ollama/llama")

    assert usage.unpriced_calls == 0
    assert usage.cost_complete is True


def test_empty_meta_still_counts_a_call():
    usage = UsageAccounting()
    usage.record(None)
    usage.record({})

    assert usage.call_count == 2
    assert usage.unpriced_calls == 2
    assert usage.cost_complete is False


def test_merge_preserves_incompleteness():
    priced = UsageAccounting()
    priced.record({"cost": 0.01, "total_tokens": 10}, model="a/b")
    unpriced = UsageAccounting()
    unpriced.record({"total_tokens": 4}, model="c/d")

    priced.merge(unpriced)

    assert priced.call_count == 2
    assert priced.cost_complete is False
    assert priced.total_tokens == 14


def test_usage_round_trips_through_dict():
    usage = UsageAccounting()
    usage.record({"cost": 0.5, "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}, model="m")
    usage.record({"prompt_tokens": 1})

    restored = UsageAccounting.from_dict(usage.to_dict())

    assert restored.cost == usage.cost
    assert restored.unpriced_calls == usage.unpriced_calls
    assert restored.cost_complete is False


# ---------------------------------------------------------------------------
# OutcomeRecord
# ---------------------------------------------------------------------------


def _sample_record() -> OutcomeRecord:
    usage = UsageAccounting()
    usage.record({"cost": 0.003, "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}, model="p/m")
    return OutcomeRecord(
        task_id="case-1",
        task_category=TaskCategory.EXTRACTION,
        strategy="direct",
        strategy_version="1",
        model="p/m",
        structured_output_strategy="provider_native",
        validation=ValidationReport(checked=True, schema_valid=True),
        evidence=EvidenceReport(checked=False),
        correct=True,
        score=1.0,
        elapsed_ms=42.0,
        usage=usage,
        termination=TerminationReason.COMPLETED,
        decisions=["selected direct: no evidence requirement"],
    )


def test_outcome_record_round_trips():
    record = _sample_record()
    restored = OutcomeRecord.from_json(record.to_json())

    assert restored.task_id == record.task_id
    assert restored.termination is TerminationReason.COMPLETED
    assert restored.validation.schema_valid is True
    assert restored.usage.cost == pytest.approx(0.003)
    assert restored.decisions == record.decisions


def test_validation_ok_is_none_when_nothing_was_checked():
    assert ValidationReport().ok is None
    assert ValidationReport(checked=True, schema_valid=False).ok is False


def test_termination_reason_knows_which_states_carry_output():
    assert TerminationReason.COMPLETED.produced_output is True
    assert TerminationReason.VALIDATION_FAILED.produced_output is True
    assert TerminationReason.INSUFFICIENT_EVIDENCE.produced_output is False
    assert TerminationReason.BUDGET_EXHAUSTED.produced_output is False


def test_jsonl_store_appends_and_reads(tmp_path):
    store = JsonlOutcomeStore(tmp_path / "outcomes.jsonl")
    store.append(_sample_record())
    store.append(_sample_record())

    records = list(store.read())
    assert len(records) == 2
    assert all(r.task_id == "case-1" for r in records)
    assert len(store) == 2


def test_jsonl_store_skips_a_truncated_trailing_line(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    store = JsonlOutcomeStore(path)
    store.append(_sample_record())
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"task_id": "half-writ')  # simulate a crash mid-append

    assert len(list(store.read())) == 1


def test_jsonl_store_rewrite_is_atomic_and_replaces_content(tmp_path):
    store = JsonlOutcomeStore(tmp_path / "outcomes.jsonl")
    store.append(_sample_record())
    store.append(_sample_record())

    store.rewrite([_sample_record()])

    assert len(list(store.read())) == 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(["extraction_contacts", "document_qa_policies", "tool_task_inventory"]))
def test_bundled_fixture_sets_load_and_split(name):
    fixture_set = load_bundled_fixture_set(name)

    assert len(fixture_set) > 0
    assert fixture_set.dev(), "every bundled set needs a dev split"
    assert fixture_set.heldout(), "every bundled set needs a held-out split"
    dev_ids = {c.id for c in fixture_set.dev()}
    heldout_ids = {c.id for c in fixture_set.heldout()}
    assert dev_ids.isdisjoint(heldout_ids)
    assert all(c.category is fixture_set.category for c in fixture_set)


def test_fixture_checksum_is_stable_and_content_sensitive():
    original = load_bundled_fixture_set("extraction_contacts")
    same = load_bundled_fixture_set("extraction_contacts")
    assert original.checksum() == same.checksum()

    edited = FixtureSet(
        name=original.name,
        version=original.version,
        category=original.category,
        cases=original.cases[:-1],
    )
    assert edited.checksum() != original.checksum()


def test_fixture_set_rejects_duplicate_ids():
    case = FixtureCase(id="dup", category=TaskCategory.EXTRACTION, split=Split.DEV)
    with pytest.raises(ValueError, match="duplicate case ids"):
        FixtureSet(name="x", version="1", category=TaskCategory.EXTRACTION, cases=(case, case))


def test_fixture_set_rejects_category_mismatch():
    case = FixtureCase(id="a", category=TaskCategory.DOCUMENT_QA, split=Split.DEV)
    with pytest.raises(ValueError, match="disagree"):
        FixtureSet(name="x", version="1", category=TaskCategory.EXTRACTION, cases=(case,))


def test_assert_optimization_safe_blocks_heldout_leakage():
    fixture_set = load_bundled_fixture_set("extraction_contacts")

    assert_optimization_safe(fixture_set.dev())  # must not raise

    with pytest.raises(ValueError, match="Held-out cases must not be used"):
        assert_optimization_safe(fixture_set.cases)


def test_fixture_set_survives_a_save_load_round_trip(tmp_path):
    original = load_bundled_fixture_set("tool_task_inventory")
    path = save_fixture_set(original, tmp_path / "copy.json")

    from prompture.execution import load_fixture_set

    restored = load_fixture_set(path)
    assert restored.checksum() == original.checksum()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _extraction_case() -> FixtureCase:
    return load_bundled_fixture_set("extraction_contacts").case("contact-dev-01")


def test_extraction_scoring_gives_partial_credit():
    case = _extraction_case()
    expected = dict(case.expected["fields"])
    expected["company"] = "Wrong Corp"

    score = score_extraction(case, expected)

    assert score.correct is False
    assert score.score == pytest.approx(4 / 5)
    assert score.details["wrong_fields"] == ["company"]


def test_extraction_scoring_normalizes_case_and_phone_formatting():
    case = _extraction_case()
    fields = dict(case.expected["fields"])
    fields["name"] = "  maria GOMEZ "
    fields["phone"] = "+1 (555) 014-2"

    score = score_extraction(case, fields)

    assert score.correct is True


def test_extraction_scoring_treats_absent_expectation_as_must_be_absent():
    case = load_bundled_fixture_set("extraction_contacts").case("contact-dev-03")

    invented = dict(case.expected["fields"])
    invented["company"] = "Aurora"
    assert score_extraction(case, invented).details["per_field"]["company"] is False

    for nullish in (None, "", "N/A", "unknown"):
        fields = dict(case.expected["fields"])
        fields["company"] = nullish
        assert score_extraction(case, fields).details["per_field"]["company"] is True


def test_extraction_scoring_with_no_output_is_incorrect_not_unknown():
    score = score_extraction(_extraction_case(), None)
    assert score.correct is False
    assert score.score == 0.0


def test_unanswerable_question_is_only_correct_when_the_run_abstains():
    case = load_bundled_fixture_set("document_qa_policies").case("policy-dev-03")

    abstained = score_document_qa(case, answer=None, abstained=True)
    assert abstained.correct is True

    plausible = score_document_qa(
        case,
        answer="The company offers a three-month paid sabbatical after five years.",
        cited_sources=["hb-benefits-1"],
        abstained=False,
    )
    assert plausible.correct is False
    assert plausible.score == 0.0


def test_answerable_question_needs_both_content_and_attribution():
    case = load_bundled_fixture_set("document_qa_policies").case("policy-dev-01")

    right = score_document_qa(case, answer="At least 14 calendar days.", cited_sources=["hb-leave-1"])
    assert right.correct is True
    assert right.score == pytest.approx(1.0)

    wrong_source = score_document_qa(case, answer="At least 14 calendar days.", cited_sources=["hb-expenses-1"])
    assert wrong_source.correct is False
    assert wrong_source.score == pytest.approx(0.5)

    wrong_content = score_document_qa(case, answer="Thirty days.", cited_sources=["hb-leave-1"])
    assert wrong_content.correct is False


def test_abstaining_on_an_answerable_question_is_wrong():
    case = load_bundled_fixture_set("document_qa_policies").case("policy-dev-01")
    score = score_document_qa(case, answer=None, abstained=True)
    assert score.correct is False


def test_tool_task_is_graded_on_world_state_not_prose():
    case = load_bundled_fixture_set("tool_task_inventory").case("inv-dev-01")
    world = InventoryWorld(case.inputs["initial_state"])

    lying = score_tool_task(case, world.snapshot(), answer="Moved 3 units of SKU-100 from A to B.")
    assert lying.correct is False

    world.move_stock("SKU-100", "A", "B", 3)
    honest = score_tool_task(case, world.snapshot(), answer="done")
    assert honest.correct is True


def test_impossible_tool_task_requires_both_no_change_and_an_abstention():
    case = load_bundled_fixture_set("tool_task_inventory").case("inv-dev-05")
    world = InventoryWorld(case.inputs["initial_state"])

    silent = score_tool_task(case, world.snapshot(), answer="ok", abstained=False)
    assert silent.correct is False
    assert silent.details["state_ok"] is True

    reported = score_tool_task(case, world.snapshot(), answer="not enough stock", abstained=True)
    assert reported.correct is True


def test_score_case_dispatches_by_category():
    case = _extraction_case()
    dispatched = score_case(case, fields=dict(case.expected["fields"]))
    assert isinstance(dispatched, CaseScore)
    assert dispatched.correct is True


# ---------------------------------------------------------------------------
# Sandbox world
# ---------------------------------------------------------------------------


def test_world_rejects_a_move_it_cannot_satisfy_and_leaves_state_untouched():
    from prompture.execution.sandbox import InventoryError

    world = InventoryWorld({"stock": {"SKU-1": {"A": 2}}})
    before = world.snapshot()

    with pytest.raises(InventoryError, match="insufficient stock"):
        world.move_stock("SKU-1", "A", "B", 5)

    assert world.snapshot() == before
    assert world.writes[-1].ok is False
    assert world.writes[-1].applied is False


def test_world_operation_ids_make_a_replayed_call_idempotent():
    world = InventoryWorld({"stock": {"SKU-1": {"A": 5}}})

    first = world.move_stock("SKU-1", "A", "B", 2, operation_id="op-1")
    second = world.move_stock("SKU-1", "A", "B", 2, operation_id="op-1")

    assert first == second
    assert world.snapshot()["stock"] == {"SKU-1": {"A": 3, "B": 2}}
    assert world.calls[-1].replayed is True


def test_world_tool_registry_honours_an_include_list():
    world = InventoryWorld({"stock": {"SKU-1": {"A": 1}}})
    registry = world.as_tool_registry(include={"get_stock"})

    assert registry.names == ["get_stock"]
    assert "move_stock" not in registry


def test_normalize_state_drops_zero_quantities():
    from prompture.execution.sandbox import normalize_inventory_state

    assert normalize_inventory_state({"stock": {"S": {"A": 0, "B": 3}}})["stock"] == {"S": {"B": 3}}
    assert normalize_inventory_state({"reservations": {"O": {}}})["reservations"] == {}
    assert normalize_inventory_state({"prices": {"S": 2}})["prices"] == {"S": 2.0}


# ---------------------------------------------------------------------------
# Budget ledger and cancellation
# ---------------------------------------------------------------------------


def test_ledger_blocks_further_work_once_the_call_limit_is_reached():
    ledger = BudgetLedger(ResourceLimits(max_llm_calls=2))
    ledger.record({"cost": 0.001})
    assert ledger.check() is None
    ledger.record({"cost": 0.001})

    assert "call limit reached" in (ledger.check() or "")


def test_ledger_uses_a_preflight_estimate_for_the_cost_limit():
    ledger = BudgetLedger(ResourceLimits(max_cost_usd=0.01))
    ledger.record({"cost": 0.009})

    assert ledger.check() is None, "observed spend is still under the ceiling"
    assert "cost limit reached" in (ledger.check(estimated_next_cost=0.005) or "")


def test_ledger_says_when_cost_is_only_a_lower_bound():
    ledger = BudgetLedger(ResourceLimits(max_cost_usd=0.01))
    ledger.record({"total_tokens": 100})  # no price available

    assert ledger.usage.cost_complete is False
    assert "lower bound" in ledger.enforcement_note()


def test_ledger_reports_the_in_flight_overshoot_window():
    ledger = BudgetLedger(ResourceLimits(max_cost_usd=0.01))
    assert ledger.overshoot_possible is False

    ledger.reserve(estimated_cost=0.004)
    assert ledger.overshoot_possible is True

    ledger.record({"cost": 0.004})
    assert ledger.overshoot_possible is False
    assert "in-flight call is still billed" in ledger.enforcement_note()


def test_cancellation_token_raises_only_after_cancel():
    token = CancellationToken()
    token.raise_if_cancelled()

    token.cancel("user pressed stop")
    assert token.cancelled is True
    with pytest.raises(Cancelled, match="user pressed stop"):
        token.raise_if_cancelled()


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


def test_build_request_maps_document_qa_passages():
    case = load_bundled_fixture_set("document_qa_policies").case("policy-dev-01")
    request = build_request(case)

    assert request.task.startswith("How many days")
    assert len(request.passages) == 3
    assert isinstance(request.passages[0], EvidencePassage)
    assert request.passages[0].id == "hb-leave-1"
    assert request.requires_evidence is True


def test_build_request_gives_tool_tasks_a_sandbox_registry():
    case = load_bundled_fixture_set("tool_task_inventory").case("inv-dev-01")
    request = build_request(case)

    assert "move_stock" in request.tools
    assert isinstance(request.metadata["world"], InventoryWorld)


def test_allowed_tools_narrow_the_registry_before_the_model_sees_it():
    case = load_bundled_fixture_set("tool_task_inventory").case("inv-dev-01")
    request = build_request(case, allowed_tools=frozenset({"get_stock"}))

    effective = request.effective_tools()
    assert effective.names == ["get_stock"]
    assert "move_stock" not in effective


# ---------------------------------------------------------------------------
# Harness + gates
# ---------------------------------------------------------------------------


def _perfect_executor(fixture_set):
    """An executor that answers every dev case correctly, with priced usage."""

    def execute(request):
        case = fixture_set.case(request.task_id)
        usage = UsageAccounting()
        usage.record({"cost": 0.0005, "prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}, model="stub/x")
        result = ExecutionResult(
            strategy="stub",
            model="stub/x",
            usage=usage,
            elapsed_ms=5.0,
            validation=ValidationReport(checked=True, schema_valid=True),
        )
        if case.category is TaskCategory.DOCUMENT_QA:
            if not case.expected.get("answerable", True):
                result.termination = TerminationReason.INSUFFICIENT_EVIDENCE
            else:
                result.answer = " ".join(case.expected.get("answer_contains") or ["ok"])
                result.evidence = EvidenceReport(
                    checked=True,
                    sources=list(case.expected.get("supporting_source_ids") or []),
                    sufficient=True,
                    support_score=1.0,
                    checker="stub",
                )
        elif case.category is TaskCategory.EXTRACTION:
            result.output = dict(case.expected["fields"])
        return result

    execute.name = "perfect-stub"
    return execute


def test_harness_reports_sample_counts_versions_and_repeat_variability():
    fixture_set = load_bundled_fixture_set("document_qa_policies")
    report = run_fixture_set(fixture_set, _perfect_executor(fixture_set), repeats=3, label="perfect")

    summary = report.summaries[0]
    assert summary.n_cases == len(fixture_set.dev())
    assert summary.runs == summary.n_cases * 3
    assert summary.repeats == 3
    assert summary.fixture_version == fixture_set.version
    assert summary.fixture_checksum == fixture_set.checksum()
    assert summary.mean_repeat_stddev == pytest.approx(0.0)
    assert report.provenance["fixtures"]["checksum"] == fixture_set.checksum()


def test_harness_separates_quality_schema_evidence_and_cost():
    fixture_set = load_bundled_fixture_set("document_qa_policies")
    summary = run_fixture_set(fixture_set, _perfect_executor(fixture_set), repeats=2).summaries[0]

    assert summary.mean_score == pytest.approx(1.0)
    assert summary.accuracy == pytest.approx(1.0)
    assert summary.schema_valid_rate == pytest.approx(1.0)
    # 3 of the 10 dev cases are unanswerable, so evidence is only checked on 7.
    assert summary.evidence_checked_runs == 14
    assert summary.abstain_rate == pytest.approx(0.3)
    assert summary.cost_complete is True
    assert summary.mean_cost_per_task == pytest.approx(0.0005)


def test_harness_marks_cost_as_a_lower_bound_when_a_call_is_unpriced():
    fixture_set = load_bundled_fixture_set("extraction_contacts")

    def unpriced(request):
        usage = UsageAccounting()
        usage.record({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}, model="local/mystery")
        case = fixture_set.case(request.task_id)
        return ExecutionResult(
            strategy="unpriced",
            model="local/mystery",
            usage=usage,
            elapsed_ms=1.0,
            output=dict(case.expected["fields"]),
            validation=ValidationReport(checked=True, schema_valid=True),
        )

    report = run_fixture_set(fixture_set, unpriced, repeats=1)
    summary = report.summaries[0]

    assert summary.cost_complete is False
    assert summary.unpriced_calls == summary.total_calls
    assert any("lower bound" in note for note in summary.incomplete_measurements)

    evaluation = report.gate_evaluations[0]
    cost_criterion = next(c for c in evaluation.criteria if c.name == "cost_per_task")
    assert cost_criterion.verdict is GateVerdict.UNKNOWN
    assert evaluation.passed is False, "an unknown cost must never clear a ceiling"


def test_harness_records_a_provider_error_instead_of_aborting_the_sweep():
    fixture_set = load_bundled_fixture_set("extraction_contacts")
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("provider exploded")
        return ExecutionResult(strategy="flaky", model="stub/x", elapsed_ms=1.0)

    summary = run_fixture_set(fixture_set, flaky, repeats=1).summaries[0]

    assert summary.runs == len(fixture_set.dev())
    assert summary.termination_counts.get(TerminationReason.PROVIDER_ERROR.value) == 1


def test_gate_stays_undetermined_below_the_declared_sample_minimum():
    fixture_set = load_bundled_fixture_set("extraction_contacts")
    subset = fixture_set.dev()[:2]
    report = run_fixture_set(fixture_set, _perfect_executor(fixture_set), cases=subset, repeats=1)

    evaluation = report.gate_evaluations[0]
    assert evaluation.passed is False
    assert "quality" in evaluation.undetermined
    assert any("min_samples" in note for note in evaluation.notes)


def test_gate_passes_only_when_every_criterion_passes():
    fixture_set = load_bundled_fixture_set("document_qa_policies")
    report = run_fixture_set(fixture_set, _perfect_executor(fixture_set), repeats=3, label="perfect")
    evaluation = report.gate_evaluations[0]

    assert evaluation.passed is True
    assert {c.name for c in evaluation.criteria} == {"quality", "cost_per_task", "latency_p95"}


def test_gate_regression_criterion_needs_a_baseline():
    fixture_set = load_bundled_fixture_set("document_qa_policies")
    gate = DEFAULT_GATES["document_qa_policies"]
    good = run_fixture_set(fixture_set, _perfect_executor(fixture_set), repeats=3).summaries[0]

    without = evaluate_gate(gate, good)
    assert all(c.name != "regression" for c in without.criteria)
    assert any("no baseline supplied" in n for n in without.notes)

    weakened = summarize_runs(
        [
            CaseRun(
                case_id="x",
                repeat=0,
                score=CaseScore(correct=False, score=0.5),
                result=ExecutionResult(strategy="weak", model="stub/x", elapsed_ms=1.0),
                outcome=OutcomeRecord(task_id="x"),
            )
        ],
        workload="document_qa_policies",
        split="dev",
        repeats=1,
    )
    with_baseline = evaluate_gate(gate, weakened, baseline=good)
    regression = next(c for c in with_baseline.criteria if c.name == "regression")
    assert regression.verdict is GateVerdict.FAIL


def test_report_serializes_to_json(tmp_path):
    fixture_set = load_bundled_fixture_set("extraction_contacts")
    report = run_fixture_set(fixture_set, _perfect_executor(fixture_set), repeats=1)

    path = report.save(tmp_path / "report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["provenance"]["fixtures"]["name"] == "extraction_contacts"
    assert len(payload["runs"]) == len(fixture_set.dev())
    assert "formatted" not in payload  # no accidental console text in the data
    assert report.format()


def test_outcome_store_receives_one_record_per_run(tmp_path):
    fixture_set = load_bundled_fixture_set("extraction_contacts")
    store = JsonlOutcomeStore(tmp_path / "outcomes.jsonl")

    run_fixture_set(fixture_set, _perfect_executor(fixture_set), repeats=2, outcome_store=store)

    records = list(store.read())
    assert len(records) == len(fixture_set.dev()) * 2
    assert {r.task_category for r in records} == {TaskCategory.EXTRACTION}
    assert all(r.correct is True for r in records)


def test_percentile_handles_empty_and_single_values():
    assert percentile([], 0.95) is None
    assert percentile([7.0], 0.95) == 7.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 100.0], 0.95) == 100.0
