"""Offline improvement: propose, evaluate, and only then promote.

The pipeline this package implements, end to end::

    evaluation failures  ─┐
    opted-in feedback    ─┼─→ Candidate ─→ Optimizer ─→ Scorecard ─→ PromotionLedger
    mined skill proposals ┘   (inert)      (dev only)   (dev + held-out)  (explicit,
                                                                          reversible)

Three properties hold at every step:

* **Nothing changes production implicitly.**  A candidate is data.  Only
  :meth:`~prompture.execution.improve.promotion.PromotionLedger.promote` makes
  one active, it refuses without a justifying scorecard, and the previous
  version is retained for :meth:`~prompture.execution.improve.promotion.PromotionLedger.rollback`.
* **The splits stay separate.**  Optimizers see the development split and call
  :func:`~prompture.execution.fixtures.assert_optimization_safe` first; the
  held-out comparison is a separate, explicit call whose result is what
  promotion reads.
* **A weak candidate is rejected.**  Promotion requires a held-out gain and a
  passing gate; an ``unknown`` gate criterion — too few samples, or a cost that
  is only a lower bound — blocks it just as a failure does.

Quick start::

    from prompture.execution.improve import (
        CandidateSearchOptimizer, PromotionLedger, Scorecard, candidate_from_failures,
    )

    ledger = PromotionLedger("promotions.json")
    ledger.set_baseline("extract_prompt", CURRENT_PROMPT)

    candidate = candidate_from_failures(
        "extract_prompt",
        base_content=CURRENT_PROMPT,
        proposed_content=REVISED_PROMPT,
        failures=[r for r in outcomes if r.correct is False],
        rationale="the model kept inventing a company for project-only mentions",
    )

    result = CandidateSearchOptimizer().optimize([candidate], fixtures.dev(), evaluate)
    card = Scorecard(candidate_id=candidate.id, target=candidate.target)
    card.record_dev(baseline_dev_summary, candidate_dev_summary, gate=gate)
    card.record_heldout(baseline_heldout_summary, candidate_heldout_summary, gate=gate)

    ok, why = card.promotable(min_gain=0.02)
    ledger.promote(candidate, card) if ok else print("rejected:", why)
"""

from __future__ import annotations

from .candidates import (
    Candidate,
    CandidateKind,
    CandidateStatus,
    CandidateStore,
    candidate_from_failures,
    candidate_from_skill_proposal,
    content_version,
)
from .optimizer import (
    CandidateSearchOptimizer,
    GEPAOptimizerAdapter,
    OptimizationResult,
    Optimizer,
    OptimizerUnavailable,
    compare_optimizers,
)
from .promotion import ActiveVersion, PromotionError, PromotionLedger, PromotionRecord
from .scorecard import Scorecard, SplitResult

__all__ = [
    "ActiveVersion",
    "Candidate",
    "CandidateKind",
    "CandidateSearchOptimizer",
    "CandidateStatus",
    "CandidateStore",
    "GEPAOptimizerAdapter",
    "OptimizationResult",
    "Optimizer",
    "OptimizerUnavailable",
    "PromotionError",
    "PromotionLedger",
    "PromotionRecord",
    "Scorecard",
    "SplitResult",
    "candidate_from_failures",
    "candidate_from_skill_proposal",
    "compare_optimizers",
    "content_version",
]
