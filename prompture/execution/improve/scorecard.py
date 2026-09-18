"""Candidate scorecards: what a change actually did, on which data.

A scorecard answers four questions that must not be collapsed into one number:

1. **Did it help on the data it was tuned on?**  (development split)
2. **Does that survive on data it never saw?**  (held-out split, reported
   separately and only once, at the end)
3. **What did the improvement cost?**  Both the candidate's own per-task cost and
   the evaluation's cost, so a 1% gain bought with 40x spend is visible.
4. **Where did the data come from?**  Fixture name, version and checksum, so two
   scorecards are only comparable when they measured the same thing.

The three splits are kept apart structurally.  :meth:`Scorecard.dev` is what
candidate *search* may look at; :meth:`Scorecard.heldout` is populated by a
separate, explicit call and is what a promotion decision reads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..gates import GateEvaluation, WorkloadGate, evaluate_gate
from ..harness import WorkloadSummary

__all__ = ["Scorecard", "SplitResult"]


@dataclass
class SplitResult:
    """Baseline versus candidate on one data split."""

    split: str
    baseline: WorkloadSummary | None = None
    candidate: WorkloadSummary | None = None
    gate: GateEvaluation | None = None

    @property
    def score_delta(self) -> float | None:
        """Candidate mean score minus baseline mean score."""
        if (
            self.baseline is None
            or self.candidate is None
            or self.baseline.mean_score is None
            or self.candidate.mean_score is None
        ):
            return None
        return self.candidate.mean_score - self.baseline.mean_score

    @property
    def cost_delta(self) -> float | None:
        """Candidate mean cost per completed task minus baseline's.

        ``None`` when either side's cost is incomplete — a comparison against a
        lower bound is not a cost comparison.
        """
        if self.baseline is None or self.candidate is None:
            return None
        if not (self.baseline.cost_complete and self.candidate.cost_complete):
            return None
        if self.baseline.mean_cost_per_task is None or self.candidate.mean_cost_per_task is None:
            return None
        return self.candidate.mean_cost_per_task - self.baseline.mean_cost_per_task

    @property
    def latency_delta(self) -> float | None:
        if self.baseline is None or self.candidate is None:
            return None
        if self.baseline.p95_latency_ms is None or self.candidate.p95_latency_ms is None:
            return None
        return self.candidate.p95_latency_ms - self.baseline.p95_latency_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "score_delta": self.score_delta,
            "cost_delta": self.cost_delta,
            "latency_delta": self.latency_delta,
            "gate": self.gate.to_dict() if self.gate else None,
        }


@dataclass
class Scorecard:
    """The full comparison record for one candidate.

    Attributes:
        candidate_id / target / candidate_version / base_version: What was tested.
        dev: Development-split comparison — what candidate search optimised on.
        heldout: Held-out comparison.  Empty until :meth:`record_heldout` runs.
        evaluation_cost_usd: What producing this scorecard cost.
        evaluation_calls: How many model calls it took.
        evaluation_cost_complete: ``False`` when any evaluation call was unpriced.
        data_provenance: Fixture provenance blocks, keyed by split.
        notes: Caveats that must travel with the numbers.
    """

    candidate_id: str
    target: str = ""
    candidate_version: str = ""
    base_version: str = ""
    dev: SplitResult | None = None
    heldout: SplitResult | None = None
    evaluation_cost_usd: float = 0.0
    evaluation_calls: int = 0
    evaluation_cost_complete: bool = True
    data_provenance: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    # ---- recording ----------------------------------------------------

    def record_dev(
        self,
        baseline: WorkloadSummary,
        candidate: WorkloadSummary,
        *,
        gate: WorkloadGate | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> SplitResult:
        """Record the development-split comparison."""
        evaluation = evaluate_gate(gate, candidate, baseline=baseline) if gate else None
        self.dev = SplitResult(split="dev", baseline=baseline, candidate=candidate, gate=evaluation)
        if provenance:
            self.data_provenance["dev"] = provenance
        self._accumulate_cost(baseline, candidate)
        return self.dev

    def record_heldout(
        self,
        baseline: WorkloadSummary,
        candidate: WorkloadSummary,
        *,
        gate: WorkloadGate | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> SplitResult:
        """Record the held-out comparison.

        Call this **once**, after candidate selection is finished.  Re-running it
        to find a split where the candidate happens to look better is how a
        held-out set stops being held out; the promotion ledger records how many
        times it was written.
        """
        if self.heldout is not None:
            self.notes.append(
                "held-out results were recorded more than once; the later numbers are "
                "no longer a clean held-out estimate"
            )
        evaluation = evaluate_gate(gate, candidate, baseline=baseline) if gate else None
        self.heldout = SplitResult(split="heldout", baseline=baseline, candidate=candidate, gate=evaluation)
        if provenance:
            self.data_provenance["heldout"] = provenance
        self._accumulate_cost(baseline, candidate)
        return self.heldout

    def _accumulate_cost(self, *summaries: WorkloadSummary) -> None:
        for summary in summaries:
            self.evaluation_cost_usd += summary.total_cost
            self.evaluation_calls += summary.total_calls
            if not summary.cost_complete:
                self.evaluation_cost_complete = False

    # ---- verdicts -----------------------------------------------------

    @property
    def improves_on_dev(self) -> bool | None:
        delta = self.dev.score_delta if self.dev else None
        return None if delta is None else delta > 0

    @property
    def improves_on_heldout(self) -> bool | None:
        delta = self.heldout.score_delta if self.heldout else None
        return None if delta is None else delta > 0

    def promotable(self, *, min_gain: float = 0.0) -> tuple[bool, str]:
        """Whether this scorecard justifies promoting the candidate.

        Requires, in order:

        1. A held-out comparison exists.  A dev-split win alone is a result about
           the data the candidate was tuned on.
        2. The held-out gain is at least ``min_gain``.
        3. The held-out gate — when one was supplied — passed.  An ``unknown``
           criterion (an unpriced cost, too few samples) blocks promotion, which
           is the point of the three-valued verdict.

        Returns ``(ok, reason)``.
        """
        if self.heldout is None:
            return False, "no held-out evaluation was recorded; a dev-split win is not evidence of generalisation"
        delta = self.heldout.score_delta
        if delta is None:
            return False, "the held-out comparison produced no comparable scores"
        if delta < min_gain:
            return False, f"held-out gain {delta:+.4f} is below the required {min_gain:+.4f}"
        gate = self.heldout.gate
        if gate is not None and not gate.passed:
            undetermined = ", ".join(gate.undetermined)
            detail = f"undetermined criteria: {undetermined}" if undetermined else "a criterion failed"
            return False, f"the held-out gate did not pass ({detail})"
        return True, f"held-out gain {delta:+.4f} with the declared gate satisfied"

    # ---- reporting ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        ok, reason = self.promotable()
        return {
            "candidate_id": self.candidate_id,
            "target": self.target,
            "candidate_version": self.candidate_version,
            "base_version": self.base_version,
            "dev": self.dev.to_dict() if self.dev else None,
            "heldout": self.heldout.to_dict() if self.heldout else None,
            "evaluation_cost_usd": self.evaluation_cost_usd,
            "evaluation_calls": self.evaluation_calls,
            "evaluation_cost_complete": self.evaluation_cost_complete,
            "data_provenance": self.data_provenance,
            "promotable": ok,
            "promotable_reason": reason,
            "notes": list(self.notes),
            "created_at": self.created_at,
        }

    def format(self) -> str:
        def delta(value: float | None, fmt: str = "{:+.4f}") -> str:
            return "n/a" if value is None else fmt.format(value)

        lines = [f"Scorecard for candidate {self.candidate_id} ({self.target} v{self.candidate_version[:8]})"]
        for split in (self.dev, self.heldout):
            if split is None:
                lines.append(f"  {'heldout' if self.heldout is None else 'dev'}: not evaluated")
                continue
            lines.append(
                f"  {split.split:<8}: score {delta(split.score_delta)} "
                f"cost {delta(split.cost_delta, '{:+.6f}')} "
                f"latency {delta(split.latency_delta, '{:+.1f}ms')}"
            )
            if split.cost_delta is None and split.baseline is not None:
                lines.append("            cost comparison unavailable (an unpriced call makes it a lower bound)")
            if split.gate is not None:
                lines.append(f"            gate: {'PASS' if split.gate.passed else 'NOT PASSED'}")
        cost_note = "" if self.evaluation_cost_complete else " (LOWER BOUND)"
        lines.append(
            f"  evaluation cost: ${self.evaluation_cost_usd:.6f}{cost_note} over {self.evaluation_calls} call(s)"
        )
        ok, reason = self.promotable()
        lines.append(f"  promotable: {ok} - {reason}")
        lines.extend(f"  NOTE: {n}" for n in self.notes)
        return "\n".join(lines)
