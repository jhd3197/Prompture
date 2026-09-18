"""Predeclared acceptance gates.

A gate is written down **before** anything is tuned against it.  That is the
whole point: a threshold chosen after seeing the numbers is not a gate, it is a
description.  :data:`DEFAULT_GATES` carries the thresholds declared for the three
bundled workloads at the start of this work, together with the rationale and the
date, so a later report can be checked against what was actually promised.

A gate evaluation has three verdicts, not two:

``pass`` / ``fail``
    The measurement exists and is on the right side of the threshold.
``unknown``
    The measurement could not be made — too few samples, or a cost figure that
    is only a lower bound because some calls had no resolvable price.  An
    ``unknown`` never counts as a pass.  This is how "we did not measure it"
    stays distinguishable from "we measured it and it was fine".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from .outcomes import TaskCategory

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .harness import WorkloadSummary

__all__ = [
    "DEFAULT_GATES",
    "CriterionResult",
    "GateEvaluation",
    "GateVerdict",
    "WorkloadGate",
    "evaluate_gate",
    "gate_for_workload",
]


class GateVerdict(str, Enum):
    """Outcome of one gate criterion."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class WorkloadGate:
    """Acceptance thresholds declared in advance for one workload.

    Attributes:
        workload: The fixture-set name the gate applies to.
        category: The workload's task category.
        quality_floor: Minimum mean :class:`~prompture.execution.scoring.CaseScore`
            score across the evaluated split, in ``0.0..1.0``.
        max_cost_per_task_usd: Ceiling on mean USD per *successfully completed*
            task.  ``None`` disables the criterion.
        max_latency_p95_ms: Ceiling on the 95th-percentile wall clock per task.
            ``None`` disables the criterion.
        max_regression: Largest tolerated absolute drop in mean score against a
            declared baseline, when a baseline is supplied.
        min_samples: Fewest scored cases before any verdict other than
            ``unknown`` may be issued.
        min_repeats: Fewest repeat runs per case before repeat-variability is
            considered established.  Below this, variability is reported but
            the gate stays honest about the single-shot nature of the numbers.
        declared_on: ISO date the thresholds were fixed.
        rationale: Why these numbers, in one or two sentences.
    """

    workload: str
    category: TaskCategory
    quality_floor: float
    max_cost_per_task_usd: float | None = None
    max_latency_p95_ms: float | None = None
    max_regression: float = 0.02
    min_samples: int = 8
    min_repeats: int = 3
    declared_on: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value
        return data


#: Thresholds declared on 2026-09-08, before any strategy was implemented or
#: measured.  They are calibrated for the small synthetic bundled sets; a host
#: benchmarking its own data should declare its own gate and say so in the
#: report.
DEFAULT_GATES: dict[str, WorkloadGate] = {
    "extraction_contacts": WorkloadGate(
        workload="extraction_contacts",
        category=TaskCategory.EXTRACTION,
        quality_floor=0.85,
        max_cost_per_task_usd=0.01,
        max_latency_p95_ms=15_000.0,
        max_regression=0.02,
        min_samples=8,
        min_repeats=3,
        declared_on="2026-09-08",
        rationale=(
            "Field-level partial credit; 0.85 leaves room for the two "
            "deliberately paraphrased role labels while still failing a model "
            "that invents absent fields. Cost and latency ceilings are set for a "
            "small-model single-call extraction, so a strategy that silently "
            "escalates to a premium model trips the cost criterion."
        ),
    ),
    "document_qa_policies": WorkloadGate(
        workload="document_qa_policies",
        category=TaskCategory.DOCUMENT_QA,
        quality_floor=0.80,
        max_cost_per_task_usd=0.02,
        max_latency_p95_ms=25_000.0,
        max_regression=0.03,
        min_samples=8,
        min_repeats=3,
        declared_on="2026-09-08",
        rationale=(
            "Three of ten dev cases are unanswerable, so a system that always "
            "answers cannot reach 0.80. Attribution is half the per-case credit, "
            "which means a right answer citing the wrong passage also fails to "
            "clear the floor on its own."
        ),
    ),
    "tool_task_inventory": WorkloadGate(
        workload="tool_task_inventory",
        category=TaskCategory.TOOL_TASK,
        quality_floor=0.90,
        max_cost_per_task_usd=0.05,
        max_latency_p95_ms=40_000.0,
        max_regression=0.02,
        min_samples=6,
        min_repeats=3,
        declared_on="2026-09-08",
        rationale=(
            "Graded on world state, so the floor is high: these are short, "
            "deterministic tasks with an unambiguous correct end state. The two "
            "impossible cases require a refusal, so a model that always acts "
            "cannot clear it."
        ),
    ),
}


def gate_for_workload(workload: str) -> WorkloadGate | None:
    """Look up a declared gate by fixture-set name."""
    return DEFAULT_GATES.get(workload)


@dataclass(frozen=True)
class CriterionResult:
    """One criterion's verdict with the numbers behind it."""

    name: str
    verdict: GateVerdict
    observed: float | None
    threshold: float | None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "verdict": self.verdict.value,
            "observed": self.observed,
            "threshold": self.threshold,
            "note": self.note,
        }


@dataclass(frozen=True)
class GateEvaluation:
    """The result of checking a summary against its declared gate."""

    gate: WorkloadGate
    criteria: tuple[CriterionResult, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """``True`` only when **every** criterion passed.

        An ``unknown`` criterion prevents a pass, on purpose.
        """
        return bool(self.criteria) and all(c.verdict is GateVerdict.PASS for c in self.criteria)

    @property
    def failed(self) -> bool:
        return any(c.verdict is GateVerdict.FAIL for c in self.criteria)

    @property
    def undetermined(self) -> tuple[str, ...]:
        """Names of criteria that could not be measured."""
        return tuple(c.name for c in self.criteria if c.verdict is GateVerdict.UNKNOWN)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate.to_dict(),
            "passed": self.passed,
            "criteria": [c.to_dict() for c in self.criteria],
            "undetermined": list(self.undetermined),
            "notes": list(self.notes),
        }

    def format(self) -> str:
        """Render as a short text block for a console report."""
        head = "PASS" if self.passed else ("FAIL" if self.failed else "UNDETERMINED")
        lines = [f"Gate {self.gate.workload}: {head}"]
        for c in self.criteria:
            observed = "n/a" if c.observed is None else f"{c.observed:.4f}"
            threshold = "n/a" if c.threshold is None else f"{c.threshold:.4f}"
            lines.append(f"  {c.verdict.value:<8} {c.name}: observed={observed} threshold={threshold}")
            if c.note:
                lines.append(f"           {c.note}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def evaluate_gate(
    gate: WorkloadGate,
    summary: WorkloadSummary,
    *,
    baseline: WorkloadSummary | None = None,
) -> GateEvaluation:
    """Check a :class:`~prompture.execution.harness.WorkloadSummary` against *gate*.

    Args:
        gate: The thresholds declared in advance.
        summary: The measured aggregate.
        baseline: Optional earlier summary to check the regression tolerance
            against.  Omitted means the regression criterion is not evaluated at
            all — not that it passed.
    """
    criteria: list[CriterionResult] = []
    notes: list[str] = []

    enough_samples = summary.scored_cases >= gate.min_samples
    if not enough_samples:
        notes.append(
            f"only {summary.scored_cases} scored case(s); the gate declares "
            f"min_samples={gate.min_samples}, so every criterion is undetermined"
        )

    # -- quality -------------------------------------------------------
    if summary.mean_score is None:
        criteria.append(
            CriterionResult("quality", GateVerdict.UNKNOWN, None, gate.quality_floor, "no case produced a score")
        )
    elif not enough_samples:
        criteria.append(
            CriterionResult(
                "quality",
                GateVerdict.UNKNOWN,
                summary.mean_score,
                gate.quality_floor,
                "sample count below the declared minimum",
            )
        )
    else:
        verdict = GateVerdict.PASS if summary.mean_score >= gate.quality_floor else GateVerdict.FAIL
        criteria.append(CriterionResult("quality", verdict, summary.mean_score, gate.quality_floor))

    # -- cost ----------------------------------------------------------
    if gate.max_cost_per_task_usd is not None:
        if not summary.cost_complete:
            criteria.append(
                CriterionResult(
                    "cost_per_task",
                    GateVerdict.UNKNOWN,
                    summary.mean_cost_per_task,
                    gate.max_cost_per_task_usd,
                    (
                        f"{summary.unpriced_calls} call(s) had no resolvable price; "
                        "the observed figure is a lower bound and cannot clear a ceiling"
                    ),
                )
            )
        elif summary.mean_cost_per_task is None or not enough_samples:
            criteria.append(
                CriterionResult(
                    "cost_per_task",
                    GateVerdict.UNKNOWN,
                    summary.mean_cost_per_task,
                    gate.max_cost_per_task_usd,
                    "insufficient data",
                )
            )
        else:
            verdict = GateVerdict.PASS if summary.mean_cost_per_task <= gate.max_cost_per_task_usd else GateVerdict.FAIL
            criteria.append(
                CriterionResult("cost_per_task", verdict, summary.mean_cost_per_task, gate.max_cost_per_task_usd)
            )

    # -- latency -------------------------------------------------------
    if gate.max_latency_p95_ms is not None:
        observed = summary.p95_latency_ms
        if observed is None or not enough_samples:
            criteria.append(
                CriterionResult(
                    "latency_p95",
                    GateVerdict.UNKNOWN,
                    observed,
                    gate.max_latency_p95_ms,
                    "insufficient data",
                )
            )
        else:
            verdict = GateVerdict.PASS if observed <= gate.max_latency_p95_ms else GateVerdict.FAIL
            criteria.append(CriterionResult("latency_p95", verdict, observed, gate.max_latency_p95_ms))

    # -- regression ----------------------------------------------------
    if baseline is not None:
        if summary.mean_score is None or baseline.mean_score is None:
            criteria.append(
                CriterionResult("regression", GateVerdict.UNKNOWN, None, gate.max_regression, "no comparable scores")
            )
        else:
            drop = baseline.mean_score - summary.mean_score
            verdict = GateVerdict.PASS if drop <= gate.max_regression else GateVerdict.FAIL
            criteria.append(
                CriterionResult(
                    "regression",
                    verdict,
                    drop,
                    gate.max_regression,
                    f"baseline mean_score={baseline.mean_score:.4f}",
                )
            )
    else:
        notes.append("no baseline supplied; the regression criterion was not evaluated")

    if summary.repeats < gate.min_repeats:
        notes.append(
            f"each case ran {summary.repeats} time(s); the gate declares "
            f"min_repeats={gate.min_repeats}, so repeat-run variability is not established"
        )

    return GateEvaluation(gate=gate, criteria=tuple(criteria), notes=tuple(notes))


@dataclass
class GateBook:
    """A mutable collection of declared gates, for hosts with their own workloads."""

    gates: dict[str, WorkloadGate] = field(default_factory=lambda: dict(DEFAULT_GATES))

    def declare(self, gate: WorkloadGate) -> WorkloadGate:
        """Record a gate.  Overwrites any prior declaration for the workload."""
        self.gates[gate.workload] = gate
        return gate

    def get(self, workload: str) -> WorkloadGate | None:
        return self.gates.get(workload)
