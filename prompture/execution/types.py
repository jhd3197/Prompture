"""The shared execution contract: request, result envelope, limits, cancellation.

Everything in :mod:`prompture.execution` speaks these types.  A strategy takes an
:class:`ExecutionRequest` and returns an :class:`ExecutionResult`; the benchmark
harness, the adaptive policy, and the workflow adapter all consume the same
envelope, which is what makes fixed and adaptive runs comparable.

Nothing here calls a model or imports a driver.  The types are deliberately
plain so a host can construct them, serialise them, and assert on them without
provider access.

Relationship to existing Prompture return types
-----------------------------------------------

This envelope does **not** replace ``ExtractResult``, ``RAGAnswer``,
``AgentResult`` or ``ReviewLoopResult``.  Each strategy adapter wraps its
underlying component and keeps the component's own result reachable through
:attr:`ExecutionResult.raw`, so existing callers of those APIs are untouched and
callers who opt into a strategy can still reach backend-specific fields.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .outcomes import (
    EvidenceReport,
    OutcomeRecord,
    TaskCategory,
    TerminationReason,
    UsageAccounting,
    ValidationReport,
)

__all__ = [
    "BudgetLedger",
    "CancellationToken",
    "Cancelled",
    "EvidencePassage",
    "ExecutionRequest",
    "ExecutionResult",
    "ResourceLimits",
    "StepKind",
    "StepRecord",
]


class Cancelled(RuntimeError):
    """Raised inside a strategy when its :class:`CancellationToken` is set."""


class CancellationToken:
    """Cooperative cancellation shared between a caller and a strategy.

    Strategies check :meth:`raise_if_cancelled` at step boundaries — before
    starting a model call, and after each one returns.  Cancellation is
    therefore *cooperative*: it stops the next step, it does not abort an
    in-flight HTTP request, and any call already dispatched is still billed.
    That limitation is reported honestly rather than papered over.

    Example::

        token = CancellationToken()
        token.cancel()          # from another thread, a signal handler, a UI
        assert token.cancelled
    """

    __slots__ = ("_event", "_reason")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason: str = ""

    def cancel(self, reason: str = "cancelled by caller") -> None:
        """Request cancellation.  Idempotent."""
        self._reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def raise_if_cancelled(self) -> None:
        """Raise :class:`Cancelled` when cancellation has been requested."""
        if self._event.is_set():
            raise Cancelled(self._reason or "cancelled by caller")


@dataclass(frozen=True)
class ResourceLimits:
    """Hard bounds a strategy must respect.

    ``None`` means "no limit for this dimension".  Limits are checked *before*
    starting additional work, using a pre-flight estimate where one is
    available; see :class:`BudgetLedger` for what that can and cannot promise.

    Attributes:
        max_cost_usd: Ceiling on total attributed USD.
        max_total_tokens: Ceiling on total tokens.
        max_llm_calls: Ceiling on model calls, including reviewers and repairs.
        max_steps: Ceiling on strategy steps (an iteration of draft+critique is
            two steps).
        max_seconds: Wall-clock ceiling, checked at step boundaries.
    """

    max_cost_usd: float | None = None
    max_total_tokens: int | None = None
    max_llm_calls: int | None = None
    max_steps: int | None = None
    max_seconds: float | None = None

    @property
    def unbounded(self) -> bool:
        return all(
            v is None
            for v in (
                self.max_cost_usd,
                self.max_total_tokens,
                self.max_llm_calls,
                self.max_steps,
                self.max_seconds,
            )
        )


class BudgetLedger:
    """Tracks consumption against :class:`ResourceLimits` and answers "may I?".

    What this is honest about
    -------------------------

    * **Enforcement is pre-flight, not mid-flight.**  :meth:`check` is consulted
      before dispatching work.  Once a call is in flight it runs to completion
      and is billed, so the observed total can overshoot ``max_cost_usd`` by up
      to the cost of one call.  :attr:`overshoot_possible` says whether that
      window is currently open.
    * **Estimates are labelled.**  :meth:`reserve` records an *estimated* cost
      for an about-to-start call; when the real usage arrives, :meth:`record`
      releases the reservation and books the observed figure.  A ledger that
      only ever saw estimates reports ``cost_estimated=True``.
    * **Unknown price is not zero.**  A call whose meta carries no usable cost
      increments ``usage.unpriced_calls``, so ``usage.cost_complete`` goes
      ``False`` and the total is a lower bound.
    """

    def __init__(self, limits: ResourceLimits | None = None, *, started_at: float | None = None) -> None:
        self.limits = limits or ResourceLimits()
        self.usage = UsageAccounting()
        self.steps = 0
        self.started_at = started_at if started_at is not None else time.monotonic()
        self._reserved_cost = 0.0
        self._reserved_calls = 0
        self._lock = threading.RLock()

    # ---- consumption --------------------------------------------------

    def reserve(self, estimated_cost: float = 0.0) -> None:
        """Record that a call is about to start, at an estimated cost."""
        with self._lock:
            self._reserved_cost += max(0.0, float(estimated_cost or 0.0))
            self._reserved_calls += 1

    def release(self) -> None:
        """Drop the outstanding reservation without booking anything.

        Used when a reserved call never dispatched (cancelled, short-circuited).
        """
        with self._lock:
            self._reserved_cost = 0.0
            self._reserved_calls = max(0, self._reserved_calls - 1)

    def record(
        self,
        meta: dict[str, Any] | None,
        *,
        model: str | None = None,
        estimated: bool = False,
        price_known: bool | None = None,
    ) -> None:
        """Book an observed call and clear its reservation."""
        with self._lock:
            self.usage.record(meta, model=model, estimated=estimated, price_known=price_known)
            self._reserved_cost = 0.0
            self._reserved_calls = max(0, self._reserved_calls - 1)

    def record_step(self) -> None:
        with self._lock:
            self.steps += 1

    # ---- interrogation ------------------------------------------------

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def overshoot_possible(self) -> bool:
        """``True`` while a reserved call may still land after a limit is hit."""
        return self._reserved_calls > 0

    def check(self, *, estimated_next_cost: float = 0.0) -> str | None:
        """Return a reason string when more work must not start, else ``None``.

        Args:
            estimated_next_cost: Pre-flight USD estimate for the call being
                considered.  Pass ``0.0`` when no estimate is available — the
                cost limit then only trips on already-observed spend, which is
                weaker but never blocks work on a fabricated number.
        """
        limits = self.limits
        with self._lock:
            if limits.max_steps is not None and self.steps >= limits.max_steps:
                return f"step limit reached ({self.steps}/{limits.max_steps})"
            if limits.max_llm_calls is not None and self.usage.call_count >= limits.max_llm_calls:
                return f"call limit reached ({self.usage.call_count}/{limits.max_llm_calls})"
            if limits.max_total_tokens is not None and self.usage.total_tokens >= limits.max_total_tokens:
                return f"token limit reached ({self.usage.total_tokens}/{limits.max_total_tokens})"
            if limits.max_cost_usd is not None:
                projected = self.usage.cost + self._reserved_cost + max(0.0, float(estimated_next_cost or 0.0))
                if projected > limits.max_cost_usd:
                    detail = "" if self.usage.cost_complete else " (observed cost is a lower bound: unpriced calls)"
                    return f"cost limit reached (projected ${projected:.6f} > ${limits.max_cost_usd:.6f}){detail}"
            if limits.max_seconds is not None and self.elapsed_seconds >= limits.max_seconds:
                return f"time limit reached ({self.elapsed_seconds:.2f}s/{limits.max_seconds:.2f}s)"
        return None

    def enforcement_note(self) -> str:
        """One-line description of how enforcement actually behaved."""
        if self.limits.unbounded:
            return "no resource limits configured"
        parts = [
            f"pre-flight checks against {self._limits_repr()}",
            f"{self.usage.call_count} call(s) observed",
        ]
        if not self.usage.cost_complete:
            parts.append(f"{self.usage.unpriced_calls} call(s) had no resolvable price - cost is a lower bound")
        if self.usage.cost_estimated:
            parts.append("some cost figures are pre-flight estimates")
        parts.append("an in-flight call is still billed after a limit trips")
        return "; ".join(parts)

    def _limits_repr(self) -> str:
        fields = {
            "cost": self.limits.max_cost_usd,
            "tokens": self.limits.max_total_tokens,
            "calls": self.limits.max_llm_calls,
            "steps": self.limits.max_steps,
            "seconds": self.limits.max_seconds,
        }
        return "{" + ", ".join(f"{k}={v}" for k, v in fields.items() if v is not None) + "}"


@dataclass(frozen=True)
class EvidencePassage:
    """One retrievable passage, carrying the id used for attribution.

    Attributes:
        id: Stable identifier quoted in :class:`~prompture.execution.outcomes.EvidenceReport.sources`.
        text: The passage body shown to the model.
        score: Retrieval score, when the retriever produced one.
        metadata: Free-form (document title, page, URI).
    """

    id: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, item: Any, *, index: int = 0) -> EvidencePassage:
        """Coerce a dict, a ``Document``, or a ``VectorSearchResult`` into a passage."""
        if isinstance(item, EvidencePassage):
            return item
        if isinstance(item, dict):
            return cls(
                id=str(item.get("id") or f"passage-{index}"),
                text=str(item.get("text") or item.get("content") or ""),
                score=item.get("score"),
                metadata=dict(item.get("metadata") or {}),
            )
        # rag.VectorSearchResult → .document / .score ; rag.Document → .content
        document = getattr(item, "document", item)
        meta = dict(getattr(document, "metadata", {}) or {})
        identifier = meta.get("id") or getattr(document, "id", None) or f"passage-{index}"
        return cls(
            id=str(identifier),
            text=str(getattr(document, "content", "") or getattr(document, "text", "") or ""),
            score=getattr(item, "score", None),
            metadata=meta,
        )


StepKind = str
"""What a :class:`StepRecord` describes.

Conventional values: ``"generate"``, ``"retrieve"``, ``"validate"``,
``"repair"``, ``"review"``, ``"revise"``, ``"tool"``, ``"evidence_check"``,
``"route"``.  Free-form so a host's own strategy can add its own.
"""


@dataclass
class StepRecord:
    """One observable step inside a strategy run.

    Step records are what make a run explainable without asking the model to
    narrate itself: the sequence of kinds, models and outcomes *is* the
    explanation.
    """

    index: int
    kind: StepKind
    ok: bool = True
    model: str = ""
    detail: str = ""
    usage: UsageAccounting = field(default_factory=UsageAccounting)
    elapsed_ms: float = 0.0
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "ok": self.ok,
            "model": self.model,
            "detail": self.detail,
            "usage": self.usage.to_dict(),
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "data": self.data,
        }


@dataclass
class ExecutionRequest:
    """Everything a strategy needs to execute one task.

    Attributes:
        task: The instruction, question, or source text.
        task_id: Stable id used as ``OutcomeRecord.task_id``.  Defaults to a
            fresh uuid.
        category: Which workload this is.
        output_model: Optional Pydantic model class defining the typed contract.
        output_schema: Optional raw JSON schema, used when no Pydantic model is
            supplied.
        instruction: Optional instruction template prefix for extraction.
        passages: Pre-supplied evidence.  When set, a retrieval strategy skips
            its retriever and grounds on exactly these — which is how the
            document-QA fixtures run without a vector store.
        retriever: Optional object with ``retrieve(query, k=...)``.
        top_k: How many passages to retrieve.
        tools: Optional :class:`~prompture.agents.tools_schema.ToolRegistry`.
        allowed_tools: Hard allow-list.  Enforced by the adapter *before* the
            model sees a catalogue; an adaptive policy can narrow it but never
            widen it.
        model: Model string, or ``None`` to let routing/policy choose.
        persona: Optional :class:`~prompture.agents.persona.Persona`, shared
            unchanged across all three strategies.
        system_prompt: Optional plain system prompt (ignored when ``persona``
            is set).
        options: Driver options passed through unchanged.
        limits: Resource bounds.
        cancel: Optional cooperative cancellation token.
        structured_output_strategy: Optional override for the provider
            compatibility strategy (``provider_native`` / ``tool_call`` /
            ``prompted_repair``).  Deliberately a separate field from the
            execution strategy.
        variables: Persona template variables.
        metadata: Free-form host annotations, copied onto the outcome record.
    """

    task: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    category: TaskCategory = TaskCategory.OTHER
    output_model: Any = None
    output_schema: dict[str, Any] | None = None
    instruction: str = ""
    passages: tuple[EvidencePassage, ...] = ()
    retriever: Any = None
    top_k: int = 4
    tools: Any = None
    allowed_tools: frozenset[str] | None = None
    model: str | None = None
    persona: Any = None
    system_prompt: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    cancel: CancellationToken | None = None
    structured_output_strategy: str | None = None
    variables: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.passages and not isinstance(self.passages, tuple):
            self.passages = tuple(EvidencePassage.from_any(p, index=i) for i, p in enumerate(self.passages))

    @property
    def requires_evidence(self) -> bool:
        """Whether this task must be grounded in retrievable passages."""
        return bool(self.passages) or self.retriever is not None or self.category is TaskCategory.DOCUMENT_QA

    def effective_tools(self) -> Any:
        """The tool registry narrowed to :attr:`allowed_tools`.

        Authorisation is applied here, in Python, before any model sees the
        catalogue — a model cannot talk its way into a tool that was never
        registered for the call.
        """
        if self.tools is None or self.allowed_tools is None:
            return self.tools
        subset = getattr(self.tools, "subset", None)
        if subset is None:
            return self.tools
        available = set(getattr(self.tools, "names", ()) or ())
        return subset(sorted(available & set(self.allowed_tools)))

    def with_overrides(self, **changes: Any) -> ExecutionRequest:
        """Return a shallow copy with *changes* applied."""
        import dataclasses

        return dataclasses.replace(self, **changes)


@dataclass
class ExecutionResult:
    """The common envelope every strategy returns.

    Attributes:
        output: The typed/parsed result (a Pydantic instance, a dict, or
            ``None`` when nothing usable was produced).
        answer: Free-text answer, where the task has one.
        termination: Why the run stopped.  Always set explicitly.
        validation / evidence: The two independent quality signals.
        steps: Ordered step records.
        usage: Aggregated accounting across *every* call, including retrieval,
            reviewers, repairs, and nested sub-calls.
        model: The model actually used (after routing).
        strategy / strategy_version: Which execution strategy ran.
        structured_output_strategy: The provider-compatibility strategy used.
        decisions: Ordered, human-readable reasons for selection, escalation,
            and stopping.
        elapsed_ms: Wall clock.
        error: Message when ``termination`` is ``PROVIDER_ERROR``.
        artifacts: Scoped references to large intermediates (see
            :mod:`prompture.execution.context`), keyed by handle.
        raw: The underlying component result (``ExtractResult``, ``RAGAnswer``,
            ``ReviewLoopResult``, ``AgentResult``) so nothing is lost.
        budget_note: How resource enforcement actually behaved for this run.
    """

    output: Any = None
    answer: str = ""
    termination: TerminationReason = TerminationReason.COMPLETED
    validation: ValidationReport = field(default_factory=ValidationReport)
    evidence: EvidenceReport = field(default_factory=EvidenceReport)
    steps: tuple[StepRecord, ...] = ()
    usage: UsageAccounting = field(default_factory=UsageAccounting)
    model: str = ""
    strategy: str = ""
    strategy_version: str = ""
    structured_output_strategy: str = ""
    decisions: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    budget_note: str = ""

    @property
    def ok(self) -> bool:
        """``True`` only for a completed run whose validation did not fail.

        Note what this does *not* mean: an ``ok`` result is structurally sound
        and terminated normally.  It is not a claim of factual correctness —
        that lives in the fixture scorers and in ``OutcomeRecord.correct``.
        """
        return self.termination is TerminationReason.COMPLETED and self.validation.ok is not False

    @property
    def abstained(self) -> bool:
        """Whether the run deliberately declined to answer."""
        return self.termination in {
            TerminationReason.INSUFFICIENT_EVIDENCE,
            TerminationReason.ABSTAINED,
        }

    def add_step(self, step: StepRecord) -> None:
        self.steps = (*self.steps, step)

    def explain(self) -> str:
        """A short, human-readable account of what happened and why."""
        lines = [f"strategy={self.strategy or '?'} model={self.model or '?'} -> {self.termination.value}"]
        for step in self.steps:
            status = "ok" if step.ok else f"error: {step.error}"
            detail = f" - {step.detail}" if step.detail else ""
            lines.append(f"  [{step.index}] {step.kind}{detail} ({status})")
        for decision in self.decisions:
            lines.append(f"  * {decision}")
        if not self.usage.cost_complete:
            lines.append(
                f"  * cost ${self.usage.cost:.6f} is a LOWER BOUND - "
                f"{self.usage.unpriced_calls} call(s) had no resolvable price"
            )
        if self.budget_note:
            lines.append(f"  * budget: {self.budget_note}")
        return "\n".join(lines)

    def to_outcome(
        self,
        *,
        task_id: str,
        category: TaskCategory = TaskCategory.OTHER,
        correct: bool | None = None,
        score: float | None = None,
        prompt_version: str = "",
        skill_version: str = "",
        metadata: dict[str, Any] | None = None,
        feedback: dict[str, Any] | None = None,
    ) -> OutcomeRecord:
        """Project this envelope onto a durable :class:`OutcomeRecord`.

        ``correct`` and ``score`` come from a scorer, never from this result —
        a strategy has no way to know whether its own answer was right.
        """
        return OutcomeRecord(
            task_id=task_id,
            task_category=category,
            strategy=self.strategy,
            strategy_version=self.strategy_version,
            prompt_version=prompt_version,
            skill_version=skill_version,
            model=self.model,
            structured_output_strategy=self.structured_output_strategy,
            validation=self.validation,
            evidence=self.evidence,
            correct=correct,
            score=score,
            elapsed_ms=self.elapsed_ms,
            usage=self.usage,
            termination=self.termination,
            error=self.error,
            decisions=list(self.decisions),
            feedback=dict(feedback or {}),
            metadata={**(metadata or {}), "steps": [s.to_dict() for s in self.steps]},
        )


#: Signature of anything that can execute one request.  Both
#: :class:`~prompture.execution.strategies.base.ExecutionStrategy` and the
#: adaptive executor satisfy it.
Executor = Callable[[ExecutionRequest], ExecutionResult]
