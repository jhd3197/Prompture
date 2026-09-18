"""The reproducible benchmark harness.

Runs an executor over a fixture split, repeatedly, and produces a report that
separates the things the roadmap insists must stay separate:

* **quality** (mean scorer credit, and strict task accuracy)
* **schema validity** (a structural signal, never quality)
* **evidence support** (a grounding signal, never quality)
* **latency** (mean and p95 wall clock)
* **cost per successfully completed task** — and, when any call had no
  resolvable price, an explicit statement that the figure is a lower bound
  rather than a total

Every summary carries its sample count, its repeat count, the fixture set's
version and checksum, and a list of measurements that could not be made.  A
report that cannot establish something says so instead of printing a zero.

The harness never calls a provider itself.  It calls whatever ``executor`` it is
given — a fixed strategy, an adaptive policy, or a stub — which is what lets the
deterministic contract checks run offline while the same code path produces real
numbers when a live model is wired in.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .fixtures import FixtureCase, FixtureSet, Split
from .gates import GateEvaluation, WorkloadGate, evaluate_gate, gate_for_workload
from .outcomes import JsonlOutcomeStore, OutcomeRecord, TaskCategory, TerminationReason
from .sandbox import InventoryWorld
from .scoring import CaseScore, score_case
from .types import EvidencePassage, ExecutionRequest, ExecutionResult, ResourceLimits

__all__ = [
    "BenchmarkReport",
    "CaseRun",
    "WorkloadSummary",
    "build_request",
    "percentile",
    "run_fixture_set",
    "score_result",
    "summarize_runs",
]

#: Anything that can execute one request.  A strategy instance, a bound method,
#: or a plain function all qualify.
Executor = Callable[[ExecutionRequest], ExecutionResult]


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile.  ``None`` for an empty sequence."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered) + 0.5) - 1))
    return float(ordered[index])


def build_request(
    case: FixtureCase,
    *,
    model: str | None = None,
    output_model: Any = None,
    limits: ResourceLimits | None = None,
    world: InventoryWorld | None = None,
    allowed_tools: frozenset[str] | None = None,
    **overrides: Any,
) -> ExecutionRequest:
    """Turn a :class:`FixtureCase` into an :class:`ExecutionRequest`.

    The per-category mapping is the fixture contract in code:

    * ``extraction`` — ``inputs["text"]`` becomes the task.
    * ``document_qa`` — ``inputs["question"]`` becomes the task and
      ``inputs["passages"]`` become pre-supplied
      :class:`~prompture.execution.types.EvidencePassage` objects, so the
      workload runs without a vector store.
    * ``tool_task`` — ``inputs["instruction"]`` becomes the task and a fresh
      :class:`~prompture.execution.sandbox.InventoryWorld` (built from
      ``inputs["initial_state"]``) supplies the tools.

    Args:
        case: The fixture case.
        model: Model string to pin, or ``None`` to leave routing to the caller.
        output_model: Pydantic class for typed extraction.
        limits: Resource bounds for the run.
        world: Pre-built sandbox world (tool tasks only).  When omitted, one is
            constructed from the case's initial state.
        allowed_tools: Hard allow-list handed to the request.
        **overrides: Extra :class:`ExecutionRequest` fields.
    """
    common: dict[str, Any] = {
        "task_id": case.id,
        "category": case.category,
        "limits": limits or ResourceLimits(),
        "metadata": {"fixture_case": case.id, "tags": list(case.tags)},
    }
    common.update(overrides)
    if model is not None:
        common["model"] = model

    if case.category is TaskCategory.EXTRACTION:
        common.setdefault("output_model", output_model)
        common.setdefault("instruction", "Extract the contact details from the following text:")
        return ExecutionRequest(task=str(case.inputs.get("text", "")), **common)

    if case.category is TaskCategory.DOCUMENT_QA:
        passages = tuple(EvidencePassage.from_any(p, index=i) for i, p in enumerate(case.inputs.get("passages") or ()))
        common.setdefault("passages", passages)
        return ExecutionRequest(task=str(case.inputs.get("question", "")), **common)

    if case.category is TaskCategory.TOOL_TASK:
        world = world or InventoryWorld(case.inputs.get("initial_state"))
        common["tools"] = world.as_tool_registry()
        if allowed_tools is not None:
            common["allowed_tools"] = allowed_tools
        common["metadata"] = {**common["metadata"], "world": world}
        return ExecutionRequest(task=str(case.inputs.get("instruction", "")), **common)

    return ExecutionRequest(task=str(case.inputs.get("task", "")), **common)


def score_result(
    case: FixtureCase,
    result: ExecutionResult,
    *,
    world: InventoryWorld | None = None,
) -> CaseScore:
    """Unpack an :class:`ExecutionResult` and grade it against *case*.

    The tool-task branch reads the sandbox world, never the model's prose — a
    run that says it moved the stock but did not scores zero.
    """
    if case.category is TaskCategory.EXTRACTION:
        return score_case(case, fields=_as_field_dict(result.output))
    if case.category is TaskCategory.DOCUMENT_QA:
        return score_case(
            case,
            answer=result.answer,
            cited_sources=tuple(result.evidence.sources),
            abstained=result.abstained,
        )
    if case.category is TaskCategory.TOOL_TASK:
        final_state = world.snapshot() if world is not None else None
        return score_case(
            case,
            final_state=final_state,
            answer=result.answer,
            abstained=result.abstained or result.termination is not TerminationReason.COMPLETED,
        )
    return score_case(case)


def _as_field_dict(output: Any) -> dict[str, Any] | None:
    """Coerce a strategy output into a plain ``{field: value}`` dict."""
    if output is None:
        return None
    if isinstance(output, dict):
        return dict(output)
    dump = getattr(output, "model_dump", None)
    if callable(dump):
        try:
            return dict(dump())
        except Exception:  # pragma: no cover - defensive
            return None
    return None


@dataclass
class CaseRun:
    """One (case, repeat) execution with its score and durable outcome record."""

    case_id: str
    repeat: int
    score: CaseScore
    result: ExecutionResult
    outcome: OutcomeRecord

    @property
    def latency_ms(self) -> float:
        return self.result.elapsed_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "repeat": self.repeat,
            "correct": self.score.correct,
            "score": self.score.score,
            "termination": self.result.termination.value,
            "schema_valid": self.result.validation.ok,
            "evidence_sufficient": self.result.evidence.sufficient,
            "latency_ms": self.result.elapsed_ms,
            "cost": self.result.usage.cost,
            "cost_complete": self.result.usage.cost_complete,
            "calls": self.result.usage.call_count,
        }


@dataclass
class WorkloadSummary:
    """Aggregate statistics for one (workload, split, configuration) run.

    Read the docstrings, not just the numbers:

    * :attr:`mean_score` is scorer credit, not a claim about a model's general
      ability.
    * :attr:`schema_valid_rate` is structural only.
    * :attr:`mean_cost_per_task` is over *successfully completed* tasks and is a
      **lower bound** whenever :attr:`cost_complete` is ``False``.
    * :attr:`incomplete_measurements` lists everything the run could not
      establish.  It is the honest part of the report.
    """

    workload: str
    split: str
    label: str = ""
    fixture_version: str = ""
    fixture_checksum: str = ""
    strategy: str = ""
    model: str = ""
    n_cases: int = 0
    repeats: int = 1
    runs: int = 0
    scored_cases: int = 0

    mean_score: float | None = None
    score_stddev: float | None = None
    accuracy: float | None = None
    schema_checked_runs: int = 0
    schema_valid_rate: float | None = None
    evidence_checked_runs: int = 0
    evidence_supported_rate: float | None = None
    mean_support_score: float | None = None
    abstain_rate: float = 0.0

    mean_latency_ms: float | None = None
    p95_latency_ms: float | None = None

    total_cost: float = 0.0
    mean_cost_per_task: float | None = None
    mean_cost_per_attempt: float | None = None
    total_calls: int = 0
    unpriced_calls: int = 0
    cost_complete: bool = True
    cost_estimated: bool = False

    mean_repeat_stddev: float | None = None
    max_repeat_stddev: float | None = None

    termination_counts: dict[str, int] = field(default_factory=dict)
    incomplete_measurements: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items()}
        return data

    def format(self) -> str:
        """Render as a compact console block."""

        def num(value: float | None, fmt: str = "{:.4f}") -> str:
            return "n/a" if value is None else fmt.format(value)

        cost_line = f"${self.total_cost:.6f}"
        if not self.cost_complete:
            cost_line += f" (LOWER BOUND - {self.unpriced_calls}/{self.total_calls} call(s) unpriced)"
        elif self.cost_estimated:
            cost_line += " (includes pre-flight estimates)"

        lines = [
            f"{self.workload} [{self.split}] {self.label or self.strategy or ''}".rstrip(),
            f"  fixtures      : v{self.fixture_version} ({self.fixture_checksum[:12]}) "
            f"{self.n_cases} case(s) x {self.repeats} repeat(s) = {self.runs} run(s)",
            f"  quality       : mean_score={num(self.mean_score)} accuracy={num(self.accuracy)} "
            f"(scored {self.scored_cases} run(s), stddev={num(self.score_stddev)})",
            f"  schema        : valid_rate={num(self.schema_valid_rate)} over {self.schema_checked_runs} checked run(s)",
            f"  evidence      : supported_rate={num(self.evidence_supported_rate)} "
            f"mean_support={num(self.mean_support_score)} over {self.evidence_checked_runs} checked run(s)",
            f"  abstention    : {self.abstain_rate:.4f}",
            f"  latency       : mean={num(self.mean_latency_ms, '{:.1f}')}ms p95={num(self.p95_latency_ms, '{:.1f}')}ms",
            f"  cost          : total={cost_line} per_completed_task={num(self.mean_cost_per_task, '${:.6f}')}",
            f"  repeat spread : mean_stddev={num(self.mean_repeat_stddev)} max_stddev={num(self.max_repeat_stddev)}",
            f"  terminations  : {json.dumps(self.termination_counts, sort_keys=True)}",
        ]
        for note in self.incomplete_measurements:
            lines.append(f"  NOT MEASURED  : {note}")
        return "\n".join(lines)


def summarize_runs(
    runs: Sequence[CaseRun],
    *,
    workload: str,
    split: str,
    label: str = "",
    fixture_version: str = "",
    fixture_checksum: str = "",
    n_cases: int = 0,
    repeats: int = 1,
    config: dict[str, Any] | None = None,
) -> WorkloadSummary:
    """Aggregate :class:`CaseRun` records into a :class:`WorkloadSummary`."""
    summary = WorkloadSummary(
        workload=workload,
        split=split,
        label=label,
        fixture_version=fixture_version,
        fixture_checksum=fixture_checksum,
        n_cases=n_cases or len({r.case_id for r in runs}),
        repeats=repeats,
        runs=len(runs),
        config=dict(config or {}),
    )
    if not runs:
        summary.incomplete_measurements.append("no runs were executed")
        return summary

    summary.strategy = runs[0].result.strategy
    models = {r.result.model for r in runs if r.result.model}
    summary.model = next(iter(models)) if len(models) == 1 else ",".join(sorted(models))

    scored = [r for r in runs if r.score.correct is not None]
    summary.scored_cases = len(scored)
    if scored:
        scores = [r.score.score for r in scored]
        summary.mean_score = statistics.fmean(scores)
        summary.score_stddev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        summary.accuracy = sum(1 for r in scored if r.score.correct) / len(scored)
    else:
        summary.incomplete_measurements.append("no run produced a gradeable score")

    schema_checked = [r for r in runs if r.result.validation.checked]
    summary.schema_checked_runs = len(schema_checked)
    if schema_checked:
        summary.schema_valid_rate = sum(1 for r in schema_checked if r.result.validation.schema_valid) / len(
            schema_checked
        )
    else:
        summary.incomplete_measurements.append("schema validity was never checked")

    evidence_checked = [r for r in runs if r.result.evidence.checked]
    summary.evidence_checked_runs = len(evidence_checked)
    if evidence_checked:
        sufficient = [r for r in evidence_checked if r.result.evidence.sufficient is not None]
        if sufficient:
            summary.evidence_supported_rate = sum(1 for r in sufficient if r.result.evidence.sufficient) / len(
                sufficient
            )
        support_scores = [
            r.result.evidence.support_score for r in evidence_checked if r.result.evidence.support_score is not None
        ]
        if support_scores:
            summary.mean_support_score = statistics.fmean(support_scores)
    else:
        summary.incomplete_measurements.append("evidence support was never checked")

    summary.abstain_rate = sum(1 for r in runs if r.result.abstained) / len(runs)

    latencies = [r.latency_ms for r in runs if r.latency_ms > 0]
    if latencies:
        summary.mean_latency_ms = statistics.fmean(latencies)
        summary.p95_latency_ms = percentile(latencies, 0.95)
    else:
        summary.incomplete_measurements.append("no wall-clock latency was recorded")

    summary.total_cost = sum(r.result.usage.cost for r in runs)
    summary.total_calls = sum(r.result.usage.call_count for r in runs)
    summary.unpriced_calls = sum(r.result.usage.unpriced_calls for r in runs)
    summary.cost_complete = summary.unpriced_calls == 0
    summary.cost_estimated = any(r.result.usage.cost_estimated for r in runs)
    completed = [r for r in runs if r.result.termination is TerminationReason.COMPLETED]
    if completed:
        summary.mean_cost_per_task = statistics.fmean([r.result.usage.cost for r in completed])
    summary.mean_cost_per_attempt = statistics.fmean([r.result.usage.cost for r in runs])
    if not summary.cost_complete:
        summary.incomplete_measurements.append(
            f"cost is a lower bound: {summary.unpriced_calls} of {summary.total_calls} call(s) had no resolvable price"
        )
    if summary.total_calls == 0:
        summary.incomplete_measurements.append("no provider calls were made - these are contract numbers, not quality")

    by_case: dict[str, list[float]] = {}
    for run in runs:
        by_case.setdefault(run.case_id, []).append(run.score.score)
    spreads = [statistics.pstdev(v) for v in by_case.values() if len(v) > 1]
    if spreads:
        summary.mean_repeat_stddev = statistics.fmean(spreads)
        summary.max_repeat_stddev = max(spreads)
    elif repeats < 2:
        summary.incomplete_measurements.append(
            "repeat-run variability not measured (each case ran once); repeats>=2 is required"
        )

    counts: dict[str, int] = {}
    for run in runs:
        key = run.result.termination.value
        counts[key] = counts.get(key, 0) + 1
    summary.termination_counts = dict(sorted(counts.items()))

    return summary


@dataclass
class BenchmarkReport:
    """A full benchmark run: summaries, provenance, and gate verdicts."""

    summaries: list[WorkloadSummary] = field(default_factory=list)
    gate_evaluations: list[GateEvaluation] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    runs: list[CaseRun] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "provenance": self.provenance,
            "summaries": [s.to_dict() for s in self.summaries],
            "gates": [g.to_dict() for g in self.gate_evaluations],
            "runs": [r.to_dict() for r in self.runs],
            "notes": list(self.notes),
        }

    def save(self, path: str | Path) -> Path:
        """Write the report as JSON.  Returns the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
        return target

    def format(self) -> str:
        blocks = [s.format() for s in self.summaries]
        blocks += [g.format() for g in self.gate_evaluations]
        if self.notes:
            blocks.append("Notes:\n" + "\n".join(f"  - {n}" for n in self.notes))
        return "\n\n".join(blocks)


def run_fixture_set(
    fixture_set: FixtureSet,
    executor: Executor,
    *,
    split: Split = Split.DEV,
    repeats: int = 1,
    label: str = "",
    model: str | None = None,
    output_model: Any = None,
    limits: ResourceLimits | None = None,
    allowed_tools: frozenset[str] | None = None,
    gate: WorkloadGate | None = None,
    baseline: WorkloadSummary | None = None,
    outcome_store: JsonlOutcomeStore | None = None,
    cases: Iterable[FixtureCase] | None = None,
    request_overrides: dict[str, Any] | None = None,
) -> BenchmarkReport:
    """Execute a fixture split and produce a :class:`BenchmarkReport`.

    Args:
        fixture_set: The set to run.
        executor: Anything callable with an :class:`ExecutionRequest`.
        split: Which split to run.  Held-out runs are for final reporting.
        repeats: How many times to run each case.  Two or more is required
            before repeat-run variability means anything.
        label: Free-text label for the configuration under test
            (``"direct/gpt-4o-mini"``).
        model: Model to pin on every request.
        output_model: Pydantic class for extraction workloads.
        limits: Resource bounds applied to every request.
        allowed_tools: Hard allow-list for tool workloads.
        gate: Declared thresholds.  Defaults to the workload's entry in
            :data:`~prompture.execution.gates.DEFAULT_GATES` when one exists.
        baseline: Earlier summary for the regression criterion.
        outcome_store: When given, every :class:`OutcomeRecord` is appended.
        cases: Explicit case list, overriding *split*.
        request_overrides: Extra fields merged into every built request.

    A provider error inside ``executor`` is caught and recorded as a
    ``PROVIDER_ERROR`` run rather than aborting the sweep, so one flaky call
    does not discard an entire benchmark.
    """
    selected = (
        tuple(cases) if cases is not None else (fixture_set.dev() if split is Split.DEV else fixture_set.heldout())
    )
    runs: list[CaseRun] = []
    overrides = dict(request_overrides or {})

    for case in selected:
        for repeat in range(repeats):
            world = (
                InventoryWorld(case.inputs.get("initial_state")) if case.category is TaskCategory.TOOL_TASK else None
            )
            request = build_request(
                case,
                model=model,
                output_model=output_model,
                limits=limits,
                world=world,
                allowed_tools=allowed_tools,
                **overrides,
            )
            started = time.perf_counter()
            try:
                result = executor(request)
            except Exception as exc:
                result = ExecutionResult(
                    termination=TerminationReason.PROVIDER_ERROR,
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    strategy=getattr(executor, "name", ""),
                    model=model or "",
                )
            if result.elapsed_ms <= 0:
                result.elapsed_ms = (time.perf_counter() - started) * 1000

            score = score_result(case, result, world=world)
            outcome = result.to_outcome(
                task_id=case.id,
                category=case.category,
                correct=score.correct,
                score=score.score,
                metadata={
                    "workload": fixture_set.name,
                    "fixture_version": fixture_set.version,
                    "split": case.split.value,
                    "repeat": repeat,
                    "label": label,
                    "score_details": score.details,
                    "scorer": score.scorer,
                },
            )
            if outcome_store is not None:
                outcome_store.append(outcome)
            runs.append(CaseRun(case_id=case.id, repeat=repeat, score=score, result=result, outcome=outcome))

    summary = summarize_runs(
        runs,
        workload=fixture_set.name,
        split=split.value if cases is None else "custom",
        label=label,
        fixture_version=fixture_set.version,
        fixture_checksum=fixture_set.checksum(),
        n_cases=len(selected),
        repeats=repeats,
        config={
            "model": model,
            "limits": limits.__dict__ if limits else None,
            "allowed_tools": sorted(allowed_tools) if allowed_tools else None,
            "executor": getattr(executor, "name", type(executor).__name__),
        },
    )

    resolved_gate = gate or gate_for_workload(fixture_set.name)
    evaluations = [evaluate_gate(resolved_gate, summary, baseline=baseline)] if resolved_gate else []

    notes: list[str] = []
    if resolved_gate is None:
        notes.append(f"no acceptance gate is declared for workload {fixture_set.name!r}")
    if summary.total_calls == 0:
        notes.append(
            "This run made no provider calls. The numbers below verify the contract "
            "(accounting, termination, scoring) and say nothing about model quality."
        )

    return BenchmarkReport(
        summaries=[summary],
        gate_evaluations=evaluations,
        provenance={"fixtures": fixture_set.provenance(), "label": label, "repeats": repeats},
        runs=runs,
        notes=notes,
    )
