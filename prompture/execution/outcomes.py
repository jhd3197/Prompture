"""Outcome records — the durable, comparable unit of "one task was executed".

Every execution strategy (:mod:`prompture.execution.strategies`) and every
benchmark run (:mod:`prompture.execution.harness`) emits one
:class:`OutcomeRecord` per task attempt.  The record is deliberately flat and
JSON-round-trippable so it can be appended to a JSONL file, diffed between
runs, and re-scored offline without re-executing anything.

Design notes
------------

* **Separate signals.** Schema validity, evidence support, reviewer approval and
  task correctness are four different fields.  Nothing in this module infers one
  from another — a schema-valid answer is not "correct", and an approving
  reviewer is not "evidence".
* **Unknown cost stays unknown.** :class:`UsageAccounting` tracks how many calls
  had usable pricing.  ``cost_complete`` is ``False`` whenever any call went
  unpriced, and ``cost`` is then a *lower bound*, never a total.
* **Explicit termination.** :class:`TerminationReason` distinguishes "produced an
  answer", "refused for lack of evidence", "ran out of budget" and "the provider
  failed".  Callers must not have to guess from an empty output.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "EvidenceReport",
    "JsonlOutcomeStore",
    "OutcomeRecord",
    "TaskCategory",
    "TerminationReason",
    "UsageAccounting",
    "ValidationReport",
    "merge_usage_accounting",
]

#: Schema version for :meth:`OutcomeRecord.to_dict`.  Bump on incompatible shape
#: changes so an old JSONL file can still be identified.
OUTCOME_SCHEMA_VERSION = 1


class TaskCategory(str, Enum):
    """The three representative workloads this roadmap measures.

    ``EXTRACTION``
        Typed extraction: text in, a Pydantic/JSON-schema object out.  Scored by
        schema validity plus field-level agreement with the fixture's expected
        object.
    ``DOCUMENT_QA``
        A question answered from supplied passages.  Scored by answer match *and*
        by whether the cited evidence actually supports the answer.
    ``TOOL_TASK``
        A bounded task whose completion is checked by inspecting deterministic
        tool state rather than by reading the model's prose.
    ``OTHER``
        Anything a host wants to record that is not one of the three.
    """

    EXTRACTION = "extraction"
    DOCUMENT_QA = "document_qa"
    TOOL_TASK = "tool_task"
    OTHER = "other"


class TerminationReason(str, Enum):
    """Why an execution stopped.  Always set; never inferred by the caller."""

    #: Produced an output that passed every configured check.
    COMPLETED = "completed"
    #: Produced an output, but schema/field validation failed after all repairs.
    VALIDATION_FAILED = "validation_failed"
    #: Deliberately declined to answer because the retrieved evidence did not
    #: support one.  This is a *successful refusal*, not an error.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    #: The reviewer never approved within the iteration limit.
    REVIEW_REJECTED = "review_rejected"
    #: A resource budget (cost, tokens, wall clock, or step count) was reached.
    BUDGET_EXHAUSTED = "budget_exhausted"
    #: The strategy's own iteration/step bound was reached.
    MAX_STEPS = "max_steps"
    #: The caller cancelled the run.
    CANCELLED = "cancelled"
    #: The provider or an adapter raised.  ``error`` carries the message.
    PROVIDER_ERROR = "provider_error"
    #: A policy chose not to run at all (e.g. no eligible model, precondition
    #: unmet).  Distinct from ``INSUFFICIENT_EVIDENCE``, which happens mid-run.
    ABSTAINED = "abstained"

    @property
    def produced_output(self) -> bool:
        """True for reasons that normally come with a usable output."""
        return self in {TerminationReason.COMPLETED, TerminationReason.VALIDATION_FAILED}


@dataclass
class ValidationReport:
    """Result of *deterministic* checks on the produced output.

    This is a structural verdict only.  ``schema_valid=True`` says the output
    matched the requested shape; it says nothing about whether the content is
    factually right.

    Attributes:
        checked: ``False`` when no validation was configured — then
            ``schema_valid`` is meaningless and :attr:`ok` returns ``None``.
        schema_valid: Whether the output satisfied its JSON schema / Pydantic
            model.
        errors: Human-readable validation error strings.
        field_errors: ``{field_path: message}`` for field-level failures, used by
            targeted-repair escalation in :mod:`prompture.execution.policy`.
        repair_attempts: How many repair passes ran before this verdict.
    """

    checked: bool = False
    schema_valid: bool = False
    errors: list[str] = field(default_factory=list)
    field_errors: dict[str, str] = field(default_factory=dict)
    repair_attempts: int = 0

    @property
    def ok(self) -> bool | None:
        """``None`` when nothing was checked, else the schema verdict."""
        return self.schema_valid if self.checked else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationReport:
        data = data or {}
        return cls(
            checked=bool(data.get("checked", False)),
            schema_valid=bool(data.get("schema_valid", False)),
            errors=list(data.get("errors") or []),
            field_errors=dict(data.get("field_errors") or {}),
            repair_attempts=int(data.get("repair_attempts", 0)),
        )


@dataclass
class EvidenceReport:
    """Whether retrieved passages actually support the produced answer.

    Kept separate from :class:`ValidationReport` on purpose: an answer can be
    perfectly schema-valid and completely unsupported.

    Attributes:
        checked: ``False`` when no evidence check ran.
        sources: Identifiers of the passages given to the model, in the order
            they were supplied.  These are attribution handles, not content.
        supported_claims / unsupported_claims / contradicted_claims: Claim
            strings as classified by the evidence checker.
        support_score: Fraction of claims supported (0.0–1.0), or ``None`` when
            the checker could not produce one.
        sufficient: The checker's verdict on whether the evidence was enough to
            answer at all.  ``None`` when unchecked.
        checker: Name of the mechanism that produced this report (e.g.
            ``"faithfulness_evaluator"``, ``"substring"``), so a cheap
            deterministic check is never mistaken for an LLM judgement.
    """

    checked: bool = False
    sources: list[str] = field(default_factory=list)
    supported_claims: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    contradicted_claims: list[str] = field(default_factory=list)
    support_score: float | None = None
    sufficient: bool | None = None
    checker: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceReport:
        data = data or {}
        score = data.get("support_score")
        return cls(
            checked=bool(data.get("checked", False)),
            sources=list(data.get("sources") or []),
            supported_claims=list(data.get("supported_claims") or []),
            unsupported_claims=list(data.get("unsupported_claims") or []),
            contradicted_claims=list(data.get("contradicted_claims") or []),
            support_score=float(score) if isinstance(score, (int, float)) else None,
            sufficient=data.get("sufficient"),
            checker=data.get("checker", ""),
        )


@dataclass
class UsageAccounting:
    """Token / cost / call accounting with an explicit completeness flag.

    A driver response whose provider pricing is unknown contributes ``0.0`` to
    the running cost — which is indistinguishable from a genuinely free call
    unless we count it.  :attr:`unpriced_calls` counts exactly those, and
    :attr:`cost_complete` is ``False`` whenever any exist.  Report ``cost`` as a
    *lower bound* in that case.

    Attributes:
        prompt_tokens / completion_tokens / total_tokens: Summed token counts.
        cost: Summed USD across calls that had pricing.
        call_count: Total LLM/driver calls attributed to this task, including
            routing probes, retrieval generation, reviewers, repairs and any
            nested sub-calls.
        unpriced_calls: Calls whose provider pricing could not be resolved.
        cost_estimated: ``True`` when any part of ``cost`` came from a
            pre-flight estimate rather than an observed provider figure.
        per_model: ``{model_string: {...}}`` breakdown.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    call_count: int = 0
    unpriced_calls: int = 0
    cost_estimated: bool = False
    per_model: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def cost_complete(self) -> bool:
        """``False`` when at least one call had no usable price."""
        return self.unpriced_calls == 0

    def record(
        self,
        meta: dict[str, Any] | None,
        *,
        model: str | None = None,
        estimated: bool = False,
        price_known: bool | None = None,
    ) -> UsageAccounting:
        """Fold one driver ``meta``/usage dict into this accounting.

        A call is treated as *unpriced* when its meta carries no ``cost`` key at
        all, or carries a non-numeric one.

        A reported ``0.0`` is ambiguous on its own: it means "this provider is
        free" for a local model and "we had no rate card" everywhere else,
        because a driver with no pricing data still has to put *something* in
        the field.  ``price_known`` resolves that ambiguity — pass ``False``
        when the model has no resolvable rates, and the zero is booked as
        unknown rather than as a total.  Leaving it ``None`` keeps the naive
        behaviour of trusting whatever the meta said.

        Returns ``self`` so calls can be chained.
        """
        if not meta:
            # A call happened but told us nothing — count it, and count it as
            # unpriced so `cost` is never mistaken for a total.
            self.call_count += 1
            self.unpriced_calls += 1
            return self

        self.prompt_tokens += _as_int(meta.get("prompt_tokens"))
        self.completion_tokens += _as_int(meta.get("completion_tokens"))
        total = meta.get("total_tokens")
        if isinstance(total, (int, float)):
            self.total_tokens += int(total)
        else:
            self.total_tokens += _as_int(meta.get("prompt_tokens")) + _as_int(meta.get("completion_tokens"))

        cost = meta.get("cost")
        priced = isinstance(cost, (int, float)) and not isinstance(cost, bool)
        if priced and price_known is False and not cost:
            # A zero from a model we have no rate card for is not a price.
            priced = False
        if priced:
            self.cost += float(cost)
        else:
            self.unpriced_calls += 1
        if estimated:
            self.cost_estimated = True
        self.call_count += 1

        name = model or meta.get("model_name") or meta.get("model") or "unknown"
        bucket = self.per_model.setdefault(
            str(name),
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0, "calls": 0, "unpriced": 0},
        )
        bucket["prompt_tokens"] += _as_int(meta.get("prompt_tokens"))
        bucket["completion_tokens"] += _as_int(meta.get("completion_tokens"))
        bucket["total_tokens"] += _as_int(meta.get("total_tokens"))
        bucket["calls"] += 1
        if priced:
            bucket["cost"] += float(cost)
        else:
            bucket["unpriced"] += 1
        return self

    def merge(self, other: UsageAccounting | None) -> UsageAccounting:
        """Fold another accounting (e.g. from a nested call) into this one."""
        if other is None:
            return self
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cost += other.cost
        self.call_count += other.call_count
        self.unpriced_calls += other.unpriced_calls
        self.cost_estimated = self.cost_estimated or other.cost_estimated
        for name, bucket in other.per_model.items():
            target = self.per_model.setdefault(
                name,
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0, "calls": 0, "unpriced": 0},
            )
            for key, value in bucket.items():
                target[key] = target.get(key, 0) + value
        return self

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cost_complete"] = self.cost_complete
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageAccounting:
        data = data or {}
        return cls(
            prompt_tokens=_as_int(data.get("prompt_tokens")),
            completion_tokens=_as_int(data.get("completion_tokens")),
            total_tokens=_as_int(data.get("total_tokens")),
            cost=float(data.get("cost") or 0.0),
            call_count=_as_int(data.get("call_count")),
            unpriced_calls=_as_int(data.get("unpriced_calls")),
            cost_estimated=bool(data.get("cost_estimated", False)),
            per_model=dict(data.get("per_model") or {}),
        )


def _as_int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def merge_usage_accounting(items: Iterable[UsageAccounting | None]) -> UsageAccounting:
    """Sum any number of accountings into a fresh one."""
    total = UsageAccounting()
    for item in items:
        total.merge(item)
    return total


@dataclass
class OutcomeRecord:
    """One executed task, with everything needed to compare runs offline.

    Attributes:
        task_id: Stable id of the *task* (usually a fixture case id).  Repeated
            runs of the same task share it — that is what makes repeat-run
            variability measurable.
        run_id: Unique id of this particular attempt.
        task_category: See :class:`TaskCategory`.
        strategy: Execution-strategy name (``"direct"``, ``"retrieve_verify"``,
            ``"draft_critique"``, or a host's own).
        strategy_version / prompt_version / skill_version: Version handles for
            the exact configuration that ran.  Offline optimisation
            (:mod:`prompture.execution.improve`) keys candidates off these.
        model: The model string actually used, after any routing.
        structured_output_strategy: The *provider-compatibility* strategy value
            (``provider_native`` / ``tool_call`` / ``prompted_repair``).  Kept
            distinct from ``strategy`` on purpose.
        validation / evidence: The two independent quality signals.
        correct: The task-level verdict, when a deterministic checker or a human
            supplied one.  ``None`` means *unknown* — never default it to the
            schema verdict.
        score: Optional graded score in 0.0–1.0 for tasks with partial credit.
        elapsed_ms: Wall clock for the whole task attempt.
        usage: See :class:`UsageAccounting`.
        termination: See :class:`TerminationReason`.
        error: Provider/adapter error message when ``termination`` is
            ``PROVIDER_ERROR``.
        decisions: Ordered, human-readable reasons for strategy/model selection
            and escalation, produced by :mod:`prompture.execution.policy`.
        feedback: Optional opted-in human feedback payload.
        metadata: Free-form host annotations.
        ts: Unix timestamp of record creation.
    """

    task_id: str
    task_category: TaskCategory = TaskCategory.OTHER
    strategy: str = ""
    strategy_version: str = ""
    prompt_version: str = ""
    skill_version: str = ""
    model: str = ""
    structured_output_strategy: str = ""
    validation: ValidationReport = field(default_factory=ValidationReport)
    evidence: EvidenceReport = field(default_factory=EvidenceReport)
    correct: bool | None = None
    score: float | None = None
    elapsed_ms: float = 0.0
    usage: UsageAccounting = field(default_factory=UsageAccounting)
    termination: TerminationReason = TerminationReason.COMPLETED
    error: str | None = None
    decisions: list[str] = field(default_factory=list)
    feedback: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)
    version: int = OUTCOME_SCHEMA_VERSION

    @property
    def cost_complete(self) -> bool:
        """Whether every call in this task had resolvable pricing."""
        return self.usage.cost_complete

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task_category": self.task_category.value,
            "strategy": self.strategy,
            "strategy_version": self.strategy_version,
            "prompt_version": self.prompt_version,
            "skill_version": self.skill_version,
            "model": self.model,
            "structured_output_strategy": self.structured_output_strategy,
            "validation": self.validation.to_dict(),
            "evidence": self.evidence.to_dict(),
            "correct": self.correct,
            "score": self.score,
            "elapsed_ms": self.elapsed_ms,
            "usage": self.usage.to_dict(),
            "termination": self.termination.value,
            "error": self.error,
            "decisions": list(self.decisions),
            "feedback": dict(self.feedback),
            "metadata": dict(self.metadata),
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutcomeRecord:
        return cls(
            task_id=data["task_id"],
            task_category=TaskCategory(data.get("task_category", TaskCategory.OTHER.value)),
            strategy=data.get("strategy", ""),
            strategy_version=data.get("strategy_version", ""),
            prompt_version=data.get("prompt_version", ""),
            skill_version=data.get("skill_version", ""),
            model=data.get("model", ""),
            structured_output_strategy=data.get("structured_output_strategy", ""),
            validation=ValidationReport.from_dict(data.get("validation") or {}),
            evidence=EvidenceReport.from_dict(data.get("evidence") or {}),
            correct=data.get("correct"),
            score=data.get("score"),
            elapsed_ms=float(data.get("elapsed_ms", 0.0)),
            usage=UsageAccounting.from_dict(data.get("usage") or {}),
            termination=TerminationReason(data.get("termination", TerminationReason.COMPLETED.value)),
            error=data.get("error"),
            decisions=list(data.get("decisions") or []),
            feedback=dict(data.get("feedback") or {}),
            metadata=dict(data.get("metadata") or {}),
            run_id=data.get("run_id", uuid.uuid4().hex),
            ts=float(data.get("ts", time.time())),
            version=int(data.get("version", OUTCOME_SCHEMA_VERSION)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, raw: str) -> OutcomeRecord:
        return cls.from_dict(json.loads(raw))


class JsonlOutcomeStore:
    """Append-only JSONL store for :class:`OutcomeRecord`.

    One record per line, appended under a lock so concurrent strategy runs in
    the same process cannot interleave partial lines.  Reading tolerates
    malformed trailing lines (a crash mid-append) by skipping them, which keeps
    a long benchmark run's earlier results usable.

    Example::

        store = JsonlOutcomeStore("./outcomes/dev.jsonl")
        store.append(record)
        for rec in store.read():
            print(rec.task_id, rec.termination.value)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: OutcomeRecord) -> None:
        """Append one record."""
        line = record.to_json()
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def extend(self, records: Iterable[OutcomeRecord]) -> int:
        """Append many records; returns how many were written."""
        count = 0
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            for record in records:
                fh.write(record.to_json() + "\n")
                count += 1
        return count

    def read(self) -> Iterator[OutcomeRecord]:
        """Yield every parseable record, oldest first."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield OutcomeRecord.from_json(line)
                except (ValueError, KeyError):
                    continue

    def __len__(self) -> int:
        return sum(1 for _ in self.read())

    def rewrite(self, records: Iterable[OutcomeRecord]) -> None:
        """Atomically replace the file's contents with *records*.

        Used by retention/compaction tooling.  Writes to a temp file in the same
        directory then ``os.replace``s it, so a crash never leaves a truncated
        store.
        """
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    for record in records:
                        fh.write(record.to_json() + "\n")
                os.replace(tmp, self.path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
