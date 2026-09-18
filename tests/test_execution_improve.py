"""Phase E contract checks: candidates, optimizers, scorecards, promotion.

The two acceptance behaviours the roadmap names are the first two sections here:
a deliberately weak candidate is rejected, and an improving one is reproducible
and reversible.  The rest verifies that nothing changes production implicitly and
that the splits stay separate.
"""

from __future__ import annotations

import pytest

from prompture.execution import (
    TerminationReason,
    UsageAccounting,
    load_bundled_fixture_set,
)
from prompture.execution.gates import DEFAULT_GATES
from prompture.execution.harness import CaseRun, summarize_runs
from prompture.execution.improve import (
    CandidateKind,
    CandidateSearchOptimizer,
    CandidateStatus,
    CandidateStore,
    GEPAOptimizerAdapter,
    OptimizerUnavailable,
    PromotionError,
    PromotionLedger,
    Scorecard,
    candidate_from_failures,
    candidate_from_skill_proposal,
    compare_optimizers,
    content_version,
)
from prompture.execution.outcomes import OutcomeRecord
from prompture.execution.scoring import CaseScore
from prompture.execution.types import ExecutionResult

FIXTURES = load_bundled_fixture_set("extraction_contacts")
GATE = DEFAULT_GATES["extraction_contacts"]

BASE_PROMPT = "Extract the contact details."
GOOD_PROMPT = "Extract the contact details. Use null for anything the text does not state."
WEAK_PROMPT = "Guess the contact details, filling every field with your best invention."


# ---------------------------------------------------------------------------
# Helpers: synthetic summaries
# ---------------------------------------------------------------------------


def _summary(mean_score, *, n=10, cost=0.001, latency=100.0, cost_complete=True, workload="extraction_contacts"):
    """Build a WorkloadSummary from synthetic runs at a fixed score."""
    runs = []
    for i in range(n):
        usage = UsageAccounting()
        usage.record({"cost": cost if cost_complete else None, "total_tokens": 10}, model="stub/m")
        runs.append(
            CaseRun(
                case_id=f"case-{i}",
                repeat=0,
                score=CaseScore(correct=mean_score >= 0.999, score=mean_score),
                result=ExecutionResult(
                    termination=TerminationReason.COMPLETED,
                    strategy="direct",
                    model="stub/m",
                    usage=usage,
                    elapsed_ms=latency,
                ),
                outcome=OutcomeRecord(task_id=f"case-{i}"),
            )
        )
    return summarize_runs(runs, workload=workload, split="dev", repeats=3, n_cases=n)


def _scorecard(candidate, *, dev_gain=0.05, heldout_gain=0.05, gate=GATE, record_heldout=True, cost_complete=True):
    card = Scorecard(
        candidate_id=candidate.id,
        target=candidate.target,
        candidate_version=candidate.version,
        base_version=candidate.base_version,
    )
    card.record_dev(
        _summary(0.90, cost_complete=cost_complete),
        _summary(0.90 + dev_gain, cost_complete=cost_complete),
        gate=gate,
    )
    if record_heldout:
        card.record_heldout(
            _summary(0.90, cost_complete=cost_complete),
            _summary(0.90 + heldout_gain, cost_complete=cost_complete),
            gate=gate,
        )
    return card


def _candidate(content=GOOD_PROMPT, *, base=BASE_PROMPT, target="extract_prompt"):
    return candidate_from_failures(
        target,
        base_content=base,
        proposed_content=content,
        failures=[
            OutcomeRecord(task_id="contact-dev-03", correct=False, termination=TerminationReason.VALIDATION_FAILED),
            OutcomeRecord(task_id="contact-dev-07", correct=False, termination=TerminationReason.COMPLETED),
        ],
        rationale="the model invented a company for project-only mentions",
    )


# ---------------------------------------------------------------------------
# Acceptance: a weak candidate is rejected
# ---------------------------------------------------------------------------


def test_a_weak_candidate_is_rejected_and_production_is_untouched():
    ledger = PromotionLedger()
    ledger.set_baseline("extract_prompt", BASE_PROMPT)
    weak = _candidate(WEAK_PROMPT)
    card = _scorecard(weak, dev_gain=0.05, heldout_gain=-0.20)

    ok, why = card.promotable()
    assert ok is False
    assert "below the required" in why

    with pytest.raises(PromotionError, match="Refusing to promote"):
        ledger.promote(weak, card)

    assert ledger.active_content("extract_prompt") == BASE_PROMPT
    ledger.reject(weak, card)
    assert weak.status is CandidateStatus.REJECTED
    assert "below the required" in weak.metadata["rejection_reason"]


def test_a_dev_split_win_alone_is_not_enough_to_promote():
    ledger = PromotionLedger()
    ledger.set_baseline("extract_prompt", BASE_PROMPT)
    candidate = _candidate()
    card = _scorecard(candidate, dev_gain=0.09, record_heldout=False)

    ok, why = card.promotable()
    assert ok is False
    assert "held-out" in why
    assert card.improves_on_dev is True
    assert card.improves_on_heldout is None

    with pytest.raises(PromotionError):
        ledger.promote(candidate, card)


def test_an_undetermined_gate_criterion_blocks_promotion():
    """An unpriced cost cannot clear a cost ceiling, so it cannot clear the gate."""
    ledger = PromotionLedger()
    ledger.set_baseline("extract_prompt", BASE_PROMPT)
    candidate = _candidate()
    card = _scorecard(candidate, cost_complete=False)

    ok, why = card.promotable()
    assert ok is False
    assert "did not pass" in why
    assert "cost_per_task" in why

    with pytest.raises(PromotionError):
        ledger.promote(candidate, card)


# ---------------------------------------------------------------------------
# Acceptance: an improving candidate is reproducible and reversible
# ---------------------------------------------------------------------------


def test_an_improving_candidate_is_promoted_and_can_be_rolled_back():
    ledger = PromotionLedger()
    ledger.set_baseline("extract_prompt", BASE_PROMPT)
    candidate = _candidate()
    card = _scorecard(candidate, heldout_gain=0.05)

    ok, why = card.promotable(min_gain=0.02)
    assert ok is True, why

    active = ledger.promote(candidate, card, min_gain=0.02)
    assert active.content == GOOD_PROMPT
    assert ledger.active_content("extract_prompt") == GOOD_PROMPT
    assert candidate.status is CandidateStatus.PROMOTED

    restored = ledger.rollback("extract_prompt")
    assert restored.content == BASE_PROMPT
    assert ledger.active_content("extract_prompt") == BASE_PROMPT


def test_the_ledger_records_why_each_change_happened():
    ledger = PromotionLedger()
    ledger.set_baseline("extract_prompt", BASE_PROMPT)
    candidate = _candidate()
    ledger.promote(candidate, _scorecard(candidate))
    ledger.rollback("extract_prompt", reason="regression in production")

    records = ledger.records("extract_prompt")
    actions = [r.action for r in records]
    assert actions == ["baseline", "promote", "rollback"]
    assert records[1].scorecard is not None
    assert records[1].candidate_id == candidate.id
    assert records[2].reason == "regression in production"


def test_a_promotion_is_reproducible_because_versions_are_content_addressed():
    first = _candidate()
    second = _candidate()

    assert first.version == second.version == content_version(GOOD_PROMPT)
    assert first.id != second.id, "same content, different proposal instances"


def test_rollback_without_a_retained_version_is_refused():
    ledger = PromotionLedger()
    with pytest.raises(PromotionError, match="nothing to roll back"):
        ledger.rollback("never-seen")


def test_a_candidate_measured_against_a_stale_base_is_refused():
    ledger = PromotionLedger()
    ledger.set_baseline("extract_prompt", BASE_PROMPT)
    candidate = _candidate()

    # Someone else changes the active prompt after the candidate was measured.
    ledger.set_baseline("extract_prompt", "A different prompt entirely.")

    with pytest.raises(PromotionError, match="active now"):
        ledger.promote(candidate, _scorecard(candidate))


def test_forcing_a_promotion_requires_a_reason_and_is_marked_forever():
    ledger = PromotionLedger()
    ledger.set_baseline("extract_prompt", BASE_PROMPT)
    weak = _candidate(WEAK_PROMPT)
    card = _scorecard(weak, heldout_gain=-0.5)

    with pytest.raises(PromotionError, match="requires force_reason"):
        ledger.promote(weak, card, force=True)

    ledger.promote(weak, card, force=True, force_reason="incident mitigation, reverting later today")
    record = ledger.records("extract_prompt")[-1]
    assert record.forced is True
    assert "FORCED" in record.reason


def test_the_ledger_survives_a_restart(tmp_path):
    path = tmp_path / "promotions.json"
    ledger = PromotionLedger(path)
    ledger.set_baseline("extract_prompt", BASE_PROMPT)
    candidate = _candidate()
    ledger.promote(candidate, _scorecard(candidate))

    reopened = PromotionLedger(path)
    assert reopened.active_content("extract_prompt") == GOOD_PROMPT
    assert reopened.rollback("extract_prompt").content == BASE_PROMPT


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def test_creating_a_candidate_changes_nothing():
    candidate = _candidate()

    assert candidate.status is CandidateStatus.PROPOSED
    assert candidate.kind is CandidateKind.PROMPT
    assert candidate.base_content == BASE_PROMPT
    # Nothing was registered, promoted, or written anywhere.
    assert PromotionLedger().active(candidate.target) is None


def test_a_candidate_records_the_failures_that_motivated_it():
    candidate = _candidate()

    assert candidate.motivating_task_ids == ["contact-dev-03", "contact-dev-07"]
    assert candidate.metadata["failure_count"] == 2
    assert candidate.metadata["failure_terminations"]["validation_failed"] == 1
    assert candidate.origin == "eval_failures"


def test_a_candidate_can_show_its_diff():
    diff = _candidate().diff()

    assert "Extract the contact details." in diff
    assert "+" in diff and "-" in diff
    assert "extract_prompt@" in diff


def test_a_mined_skill_proposal_becomes_a_candidate_not_a_registered_skill():
    from prompture.agents.skill_miner import SkillProposal
    from prompture.agents.skills import get_skill_names

    proposal = SkillProposal(
        name="reconcile-invoice",
        description="Match invoice lines to purchase orders",
        instructions="1. Fetch the PO. 2. Compare line items. 3. Flag mismatches.",
        tool_sequence=("get_po", "get_invoice"),
        occurrences=5,
        confidence=0.9,
        rationale="seen in five runs",
    )

    candidate = candidate_from_skill_proposal(proposal, motivating_task_ids=["t1", "t2"])

    assert candidate.kind is CandidateKind.SKILL
    assert candidate.origin == "skill_miner"
    assert candidate.status is CandidateStatus.PROPOSED
    assert candidate.metadata["confidence_is_self_reported"] is True
    assert "reconcile-invoice" not in get_skill_names(), "mining must not register anything"


def test_the_candidate_store_round_trips(tmp_path):
    path = tmp_path / "candidates.json"
    store = CandidateStore(path)
    candidate = store.add(_candidate())

    reopened = CandidateStore(path)
    assert len(reopened) == 1
    assert reopened.get(candidate.id).content == GOOD_PROMPT
    assert reopened.for_target("extract_prompt")[0].id == candidate.id
    assert reopened.with_status(CandidateStatus.PROPOSED)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


def _evaluate(candidate, cases):
    """Deterministic stand-in scorer: longer, more careful prompts score higher."""
    assert cases, "the optimizer must pass the development cases through"
    return 0.9 if "null" in candidate.content else 0.4


def test_candidate_search_ranks_by_measured_dev_score():
    good = _candidate(GOOD_PROMPT)
    weak = _candidate(WEAK_PROMPT)

    result = CandidateSearchOptimizer().optimize([weak, good], FIXTURES.dev(), _evaluate)

    assert result.best is good
    assert result.best_score == pytest.approx(0.9)
    assert result.evaluations == 2
    assert any("development case" in n for n in result.notes)


def test_an_optimizer_refuses_to_look_at_the_heldout_split():
    with pytest.raises(ValueError, match="Held-out cases must not be used"):
        CandidateSearchOptimizer().optimize([_candidate()], FIXTURES.cases, _evaluate)


def test_max_candidates_caps_the_evaluation_bill_without_silently_dropping_work():
    candidates = [_candidate(f"{GOOD_PROMPT} v{i}") for i in range(5)]

    result = CandidateSearchOptimizer(max_candidates=2).optimize(candidates, FIXTURES.dev(), _evaluate)

    assert result.evaluations == 2
    assert any("were not evaluated" in n for n in result.notes)


def test_one_failing_candidate_does_not_abort_the_sweep():
    def flaky(candidate, cases):
        if candidate.content == WEAK_PROMPT:
            raise RuntimeError("scorer blew up")
        return 0.8

    result = CandidateSearchOptimizer().optimize(
        [_candidate(WEAK_PROMPT), _candidate(GOOD_PROMPT)], FIXTURES.dev(), flaky
    )

    assert result.evaluations == 1
    assert any("failed to evaluate" in n for n in result.notes)


def test_the_gepa_adapter_refuses_rather_than_silently_substituting():
    if GEPAOptimizerAdapter.is_available("gepa"):
        pytest.skip("gepa is installed; the unavailable path cannot be exercised")

    with pytest.raises(OptimizerUnavailable, match="not installed"):
        GEPAOptimizerAdapter()

    deferred = GEPAOptimizerAdapter(require=False)
    with pytest.raises(OptimizerUnavailable):
        deferred.optimize([_candidate()], FIXTURES.dev(), _evaluate)


def test_the_gepa_adapter_needs_a_runner_for_the_installed_version():
    adapter = GEPAOptimizerAdapter(module="json", require=False)  # any importable module
    with pytest.raises(OptimizerUnavailable, match="runner"):
        adapter.optimize([_candidate()], FIXTURES.dev(), _evaluate)


def test_a_supplied_runner_drives_the_external_optimizer():
    def runner(module, candidates, cases, evaluate):
        return [(c, evaluate(c, cases)) for c in candidates]

    adapter = GEPAOptimizerAdapter(module="json", runner=runner, require=False)
    result = adapter.optimize([_candidate(WEAK_PROMPT), _candidate(GOOD_PROMPT)], FIXTURES.dev(), _evaluate)

    assert result.optimizer == "gepa"
    assert result.best.content == GOOD_PROMPT


def test_comparing_optimizers_records_an_unavailable_one_instead_of_failing():
    results = compare_optimizers(
        {
            "search": CandidateSearchOptimizer(),
            "gepa": GEPAOptimizerAdapter(module="definitely_not_installed_xyz", require=False),
        },
        [_candidate(GOOD_PROMPT)],
        FIXTURES.dev(),
        _evaluate,
    )

    assert results["search"].best is not None
    assert results["gepa"].best is None
    assert any("unavailable" in n for n in results["gepa"].notes)


# ---------------------------------------------------------------------------
# Scorecards
# ---------------------------------------------------------------------------


def test_the_scorecard_reports_deltas_per_split_and_the_evaluation_cost():
    candidate = _candidate()
    card = _scorecard(candidate, dev_gain=0.05, heldout_gain=0.03)

    assert card.dev.score_delta == pytest.approx(0.05)
    assert card.heldout.score_delta == pytest.approx(0.03)
    assert card.dev.cost_delta == pytest.approx(0.0)
    assert card.evaluation_calls == 40, "every evaluation run is counted"
    assert card.evaluation_cost_complete is True
    assert "promotable" in card.to_dict()
    assert card.format()


def test_a_cost_delta_is_withheld_when_either_side_is_a_lower_bound():
    candidate = _candidate()
    card = _scorecard(candidate, cost_complete=False)

    assert card.dev.cost_delta is None, "a comparison against a lower bound is not a cost comparison"
    assert card.evaluation_cost_complete is False
    assert "unavailable" in card.format()


def test_recording_the_heldout_split_twice_is_flagged():
    candidate = _candidate()
    card = _scorecard(candidate)
    card.record_heldout(_summary(0.90), _summary(0.99), gate=GATE)

    assert any("no longer a clean held-out estimate" in n for n in card.notes)


def test_data_provenance_travels_with_the_scorecard():
    candidate = _candidate()
    card = Scorecard(candidate_id=candidate.id, target=candidate.target)
    card.record_dev(_summary(0.9), _summary(0.95), provenance=FIXTURES.provenance())

    assert card.data_provenance["dev"]["checksum"] == FIXTURES.checksum()
    assert card.data_provenance["dev"]["n_dev"] == len(FIXTURES.dev())


def test_a_scorecard_with_no_gate_still_requires_a_heldout_gain():
    candidate = _candidate()
    card = _scorecard(candidate, heldout_gain=0.01, gate=None)

    assert card.promotable(min_gain=0.0)[0] is True
    assert card.promotable(min_gain=0.05)[0] is False


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_the_full_propose_evaluate_promote_rollback_cycle(tmp_path):
    candidates = CandidateStore(tmp_path / "candidates.json")
    ledger = PromotionLedger(tmp_path / "promotions.json", min_gain=0.02)
    ledger.set_baseline("extract_prompt", BASE_PROMPT)

    proposed = [candidates.add(_candidate(WEAK_PROMPT)), candidates.add(_candidate(GOOD_PROMPT))]

    ranked = CandidateSearchOptimizer().optimize(proposed, FIXTURES.dev(), _evaluate)
    winner = ranked.best
    assert winner.content == GOOD_PROMPT

    card = _scorecard(winner, dev_gain=0.06, heldout_gain=0.04)
    ok, why = card.promotable(min_gain=0.02)
    assert ok, why

    ledger.promote(winner, card)
    assert ledger.active_content("extract_prompt") == GOOD_PROMPT

    loser = next(c for c in proposed if c is not winner)
    ledger.reject(loser, card, reason="lower dev score")
    candidates.update(loser)
    assert CandidateStore(tmp_path / "candidates.json").get(loser.id).status is CandidateStatus.REJECTED

    ledger.rollback("extract_prompt", reason="post-deploy regression")
    assert ledger.active_content("extract_prompt") == BASE_PROMPT
