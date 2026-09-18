"""Bounded adaptive selection and measured routing.  **Experimental.**

Everything in this module is opt-in and off by default.  Implementing a
selection mechanism is not evidence that selecting is better than a fixed
strategy: until a live comparison shows the predeclared benefit on a target
workload (see :mod:`prompture.execution.gates`), :class:`AdaptivePolicy` stays
experimental and :attr:`AdaptivePolicy.experimental` stays ``True``.  The policy
says so in its own decision log, so a run cannot quietly present itself as a
validated default.

What the policy may and may not do
----------------------------------

* It **selects** among strategies and models the caller already configured.  It
  cannot introduce a model that is not in ``eligible_models``, and it cannot add
  a tool: :meth:`AdaptivePolicy.narrow_tools` only ever intersects with the
  request's own ``allowed_tools``.  Authorisation is a hard constraint, not an
  input to a decision.
* It **escalates** along explicit, named rules — a failed field validation may
  trigger a targeted repair; a missing-evidence outcome may trigger retrieval.
  There is deliberately **no** universal escalation chain: a missing fact does
  not summon a debate or a premium model, because nothing shows that helps.
* It **stops**.  When evidence or budget is insufficient, the result is an
  explicit abstention with a reason, not a cheaper guess.
* It **explains**.  Every selection, escalation and stop appends a line to the
  result's ``decisions``, including whether the choice came from a measurement
  or from a heuristic fallback.

Measurements versus heuristics
------------------------------

:class:`MeasurementStore` aggregates real
:class:`~prompture.execution.outcomes.OutcomeRecord` history per
``(task category, strategy, strategy version, model)``.  It refuses to answer
when it has too few observations or when the ones it has are stale — and in that
case the policy falls back to the existing heuristic
:class:`~prompture.pipeline.routing.ModelRouter`, *and says that it did*.  A
price-tier guess and a measured win are never reported the same way.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .outcomes import OutcomeRecord, TaskCategory, TerminationReason
from .strategies.base import ExecutionStrategy
from .types import ExecutionRequest, ExecutionResult

logger = logging.getLogger("prompture.execution.policy")

__all__ = [
    "AdaptivePolicy",
    "EscalationRule",
    "MeasurementKey",
    "MeasurementStore",
    "PerformanceStats",
    "SelectionDecision",
    "SelectionRule",
    "shadow_compare",
]

#: Default: a measurement older than 30 days is not evidence about today's model.
DEFAULT_STALE_AFTER_SECONDS = 30 * 24 * 3600
#: Default: fewer than this many observations is an anecdote, not a measurement.
DEFAULT_MIN_OBSERVATIONS = 20


@dataclass(frozen=True)
class MeasurementKey:
    """What a measurement is *about*.

    Model and strategy version are part of the key on purpose: a score measured
    against ``gpt-4o-mini`` last quarter says nothing about a different model, and
    a strategy whose version changed is a different strategy.
    """

    category: TaskCategory
    strategy: str
    model: str
    strategy_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "strategy": self.strategy,
            "model": self.model,
            "strategy_version": self.strategy_version,
        }


@dataclass
class PerformanceStats:
    """Aggregated per-key performance, with everything needed to distrust it.

    Attributes:
        observations: How many outcome records fed this.
        scored_observations: How many carried a score at all.
        mean_score / score_stddev: Quality, when scores exist.
        success_rate: Fraction terminating as ``COMPLETED``.
        abstain_rate: Fraction that deliberately declined.
        mean_cost / mean_latency_ms: Resource averages.
        cost_complete: ``False`` when any contributing record had an unpriced
            call, which makes ``mean_cost`` a lower bound.
        last_seen: Unix timestamp of the newest contributing record.
    """

    key: MeasurementKey
    observations: int = 0
    scored_observations: int = 0
    mean_score: float | None = None
    score_stddev: float | None = None
    success_rate: float | None = None
    abstain_rate: float = 0.0
    mean_cost: float | None = None
    mean_latency_ms: float | None = None
    cost_complete: bool = True
    last_seen: float = 0.0

    def age_seconds(self, *, now: float | None = None) -> float:
        return max(0.0, (now if now is not None else time.time()) - self.last_seen)

    def is_usable(
        self,
        *,
        min_observations: int = DEFAULT_MIN_OBSERVATIONS,
        stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
        now: float | None = None,
    ) -> tuple[bool, str]:
        """Whether this stat may drive a decision, and why not when it may not."""
        if self.observations < min_observations:
            return False, f"only {self.observations} observation(s), need {min_observations}"
        if self.mean_score is None:
            return False, "no scored observations"
        age = self.age_seconds(now=now)
        if age > stale_after_seconds:
            return False, f"newest observation is {age / 86400:.1f} day(s) old"
        return True, "usable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "observations": self.observations,
            "scored_observations": self.scored_observations,
            "mean_score": self.mean_score,
            "score_stddev": self.score_stddev,
            "success_rate": self.success_rate,
            "abstain_rate": self.abstain_rate,
            "mean_cost": self.mean_cost,
            "mean_latency_ms": self.mean_latency_ms,
            "cost_complete": self.cost_complete,
            "last_seen": self.last_seen,
        }


class MeasurementStore:
    """Per-key aggregates over :class:`OutcomeRecord` history.

    Deliberately simple: means, counts, and a timestamp.  A learned router or a
    contextual bandit is *not* implemented here — the roadmap defers those until
    there is enough representative outcome data to justify them, and a mean over
    a labelled key is enough to answer "has this configuration actually done
    better on this workload?".

    Example::

        store = MeasurementStore.from_jsonl("outcomes.jsonl")
        stats = store.get(MeasurementKey(TaskCategory.EXTRACTION, "direct", "openai/gpt-4o-mini"))
        print(stats.mean_score, stats.is_usable())
    """

    def __init__(
        self,
        *,
        min_observations: int = DEFAULT_MIN_OBSERVATIONS,
        stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    ) -> None:
        self.min_observations = min_observations
        self.stale_after_seconds = stale_after_seconds
        self._records: dict[MeasurementKey, list[OutcomeRecord]] = {}

    # ---- ingestion ----------------------------------------------------

    @staticmethod
    def key_for(record: OutcomeRecord) -> MeasurementKey:
        return MeasurementKey(
            category=record.task_category,
            strategy=record.strategy,
            model=record.model,
            strategy_version=record.strategy_version,
        )

    def observe(self, record: OutcomeRecord) -> None:
        """Add one outcome record."""
        self._records.setdefault(self.key_for(record), []).append(record)

    def extend(self, records: Iterable[OutcomeRecord]) -> int:
        count = 0
        for record in records:
            self.observe(record)
            count += 1
        return count

    @classmethod
    def from_outcomes(cls, records: Iterable[OutcomeRecord], **kwargs: Any) -> MeasurementStore:
        store = cls(**kwargs)
        store.extend(records)
        return store

    @classmethod
    def from_jsonl(cls, path: Any, **kwargs: Any) -> MeasurementStore:
        """Build a store from a :class:`~prompture.execution.outcomes.JsonlOutcomeStore` file."""
        from .outcomes import JsonlOutcomeStore

        return cls.from_outcomes(JsonlOutcomeStore(path).read(), **kwargs)

    # ---- queries ------------------------------------------------------

    def keys(self) -> list[MeasurementKey]:
        return list(self._records)

    def get(self, key: MeasurementKey) -> PerformanceStats:
        """Aggregate the records under *key*.  Empty is a valid, unusable stat."""
        records = self._records.get(key, [])
        stats = PerformanceStats(key=key, observations=len(records))
        if not records:
            return stats

        scores = [r.score for r in records if r.score is not None]
        stats.scored_observations = len(scores)
        if scores:
            stats.mean_score = statistics.fmean(scores)
            stats.score_stddev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        stats.success_rate = sum(1 for r in records if r.termination is TerminationReason.COMPLETED) / len(records)
        stats.abstain_rate = sum(
            1
            for r in records
            if r.termination in {TerminationReason.INSUFFICIENT_EVIDENCE, TerminationReason.ABSTAINED}
        ) / len(records)
        stats.mean_cost = statistics.fmean([r.usage.cost for r in records])
        stats.cost_complete = all(r.usage.cost_complete for r in records)
        latencies = [r.elapsed_ms for r in records if r.elapsed_ms > 0]
        if latencies:
            stats.mean_latency_ms = statistics.fmean(latencies)
        stats.last_seen = max(r.ts for r in records)
        return stats

    def usable(self, key: MeasurementKey, *, now: float | None = None) -> tuple[PerformanceStats, bool, str]:
        """``(stats, usable, why_not)`` for *key*."""
        stats = self.get(key)
        ok, why = stats.is_usable(
            min_observations=self.min_observations,
            stale_after_seconds=self.stale_after_seconds,
            now=now,
        )
        return stats, ok, why

    def best(
        self,
        candidates: Iterable[MeasurementKey],
        *,
        objective: str = "quality",
        now: float | None = None,
    ) -> tuple[MeasurementKey | None, PerformanceStats | None, str]:
        """Pick the best usable candidate, or explain why none was picked.

        Args:
            candidates: Keys to consider.
            objective: ``"quality"`` (highest mean score), ``"cost"`` (lowest
                mean cost among candidates that have a score), or ``"latency"``.
            now: Injectable clock, for tests.

        Returns:
            ``(key, stats, reason)``.  ``key`` is ``None`` when no candidate had
            usable measurements; ``reason`` then lists why each was rejected, so
            the caller can report a heuristic fallback honestly.
        """
        usable: list[tuple[MeasurementKey, PerformanceStats]] = []
        rejections: list[str] = []
        for key in candidates:
            stats, ok, why = self.usable(key, now=now)
            if ok:
                usable.append((key, stats))
            else:
                rejections.append(f"{key.strategy}/{key.model or '?'}: {why}")

        if not usable:
            return None, None, "; ".join(rejections) or "no candidates supplied"

        if objective == "cost":
            chosen = min(usable, key=lambda kv: _or_inf(kv[1].mean_cost))
            metric = f"mean cost ${_or_nan(chosen[1].mean_cost):.6f}"
            if not chosen[1].cost_complete:
                metric += " (LOWER BOUND: some contributing calls were unpriced)"
        elif objective == "latency":
            chosen = min(usable, key=lambda kv: _or_inf(kv[1].mean_latency_ms))
            metric = f"mean latency {_or_nan(chosen[1].mean_latency_ms):.0f}ms"
        else:
            chosen = max(usable, key=lambda kv: _or_neg_inf(kv[1].mean_score))
            metric = f"mean score {_or_nan(chosen[1].mean_score):.3f}"

        key, stats = chosen
        reason = (
            f"measured: {key.strategy}/{key.model or '?'} has the best {objective} "
            f"({metric}) over {stats.observations} observation(s)"
        )
        return key, stats, reason


def _or_inf(value: float | None) -> float:
    return math.inf if value is None else value


def _or_neg_inf(value: float | None) -> float:
    return -math.inf if value is None else value


def _or_nan(value: float | None) -> float:
    return float("nan") if value is None else value


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionRule:
    """An explicit "if the task looks like X, use strategy Y" rule.

    Rules are consulted in order and the first match wins.  They run *before*
    measurements, because an explicit requirement (this task needs evidence) is
    a constraint, not a preference to be optimised away.

    Attributes:
        name: Short identifier used in the decision log.
        when: Predicate over the request.
        strategy: Registered strategy name to use.
        reason: Human-readable justification recorded on the result.
    """

    name: str
    when: Callable[[ExecutionRequest], bool]
    strategy: str
    reason: str


@dataclass(frozen=True)
class EscalationRule:
    """An explicit "if the first attempt ended like X, try Y once" rule.

    Attributes:
        name: Identifier used in the decision log.
        when: Predicate over the *result* of the previous attempt.
        strategy: Strategy to escalate to.  May be the same strategy with
            different options (see ``overrides``).
        reason: Justification recorded on the result.
        overrides: Request field overrides applied to the retry (for example a
            narrower task, or a repair instruction).
        strategy_options: Constructor overrides for the escalated strategy.
    """

    name: str
    when: Callable[[ExecutionResult], bool]
    strategy: str
    reason: str
    overrides: dict[str, Any] = field(default_factory=dict)
    strategy_options: dict[str, Any] = field(default_factory=dict)


def _needs_evidence(request: ExecutionRequest) -> bool:
    return request.requires_evidence


def _failed_field_validation(result: ExecutionResult) -> bool:
    return result.termination is TerminationReason.VALIDATION_FAILED and bool(result.validation.field_errors)


def _missing_evidence(result: ExecutionResult) -> bool:
    return result.termination is TerminationReason.INSUFFICIENT_EVIDENCE


def _review_rejected(result: ExecutionResult) -> bool:
    return result.termination is TerminationReason.REVIEW_REJECTED


#: The default selection rules: evidence-requiring tasks are grounded, and
#: everything else is answered directly.  Nothing here escalates by default.
DEFAULT_SELECTION_RULES: tuple[SelectionRule, ...] = (
    SelectionRule(
        name="evidence_required",
        when=_needs_evidence,
        strategy="retrieve_verify",
        reason="the task supplies passages or a retriever, so the answer must be grounded",
    ),
)

#: The default escalation rules.  Both are narrow and both are bounded by
#: ``AdaptivePolicy.max_escalations``.  Note what is *not* here: no rule sends a
#: missing fact to a debate, a consensus vote, or a more expensive model.
DEFAULT_ESCALATION_RULES: tuple[EscalationRule, ...] = (
    EscalationRule(
        name="repair_failed_fields",
        when=_failed_field_validation,
        strategy="direct",
        reason="specific fields failed validation, so a targeted repair pass is worth one more call",
        strategy_options={"max_repairs": 1},
    ),
    EscalationRule(
        name="ground_missing_evidence",
        when=_missing_evidence,
        strategy="retrieve_verify",
        reason="the answer was unsupported, so retrieval is tried once before giving up",
    ),
)


@dataclass
class SelectionDecision:
    """A record of one selection or escalation, for the decision log."""

    stage: str  # "select" | "escalate" | "route" | "stop"
    strategy: str
    model: str = ""
    source: str = "rule"  # "rule" | "measured" | "heuristic" | "configured"
    reason: str = ""
    rejected: list[str] = field(default_factory=list)

    def format(self) -> str:
        text = f"{self.stage}[{self.source}] -> {self.strategy}"
        if self.model:
            text += f" on {self.model}"
        if self.reason:
            text += f": {self.reason}"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "strategy": self.strategy,
            "model": self.model,
            "source": self.source,
            "reason": self.reason,
            "rejected": list(self.rejected),
        }


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class AdaptivePolicy:
    """Selects a strategy and model, runs it, and escalates within hard bounds.

    A policy is itself an executor: ``policy(request)`` returns an
    :class:`ExecutionResult`, so it drops into the benchmark harness anywhere a
    fixed strategy would go — which is exactly how it gets compared with the
    baselines it hopes to beat.

    Args:
        strategies: ``{name: strategy}`` the policy may choose from.  A name not
            in here can never be selected, however a rule is written.
        default_strategy: Cold-start choice when no rule matches and no
            measurement is usable.  Explicit and configured, never guessed.
        eligible_models: Hard allow-list of model strings.  ``None`` means "use
            whatever the request or the strategy already specifies" — the policy
            still never invents one.
        selection_rules / escalation_rules: See :class:`SelectionRule` and
            :class:`EscalationRule`.
        measurements: Optional :class:`MeasurementStore`.  Without one, model
            choice falls back to the heuristic router and says so.
        objective: ``"quality"``, ``"cost"``, or ``"latency"`` for measured
            model selection.
        max_escalations: Hard cap on escalation attempts (default 1).  This is
            what makes an escalation loop impossible rather than unlikely.
        use_heuristic_router: Allow the existing
            :class:`~prompture.pipeline.routing.ModelRouter` as the documented
            fallback when there is no usable measurement.
        experimental: Marks the policy's own status.  Leave it ``True`` until a
            live comparison clears the workload's declared gate.

    Example::

        policy = AdaptivePolicy(
            strategies={"direct": DirectStrategy(model=m), "retrieve_verify": RetrieveAndVerifyStrategy(model=m)},
            default_strategy="direct",
            eligible_models=["openai/gpt-4o-mini"],
        )
        result = policy.run(request)
        print("\\n".join(result.decisions))
    """

    def __init__(
        self,
        *,
        strategies: dict[str, ExecutionStrategy],
        default_strategy: str = "direct",
        eligible_models: list[str] | None = None,
        selection_rules: Iterable[SelectionRule] = DEFAULT_SELECTION_RULES,
        escalation_rules: Iterable[EscalationRule] = DEFAULT_ESCALATION_RULES,
        measurements: MeasurementStore | None = None,
        objective: str = "quality",
        max_escalations: int = 1,
        use_heuristic_router: bool = False,
        experimental: bool = True,
        name: str = "adaptive",
    ) -> None:
        if not strategies:
            raise ValueError("AdaptivePolicy needs at least one strategy to choose from")
        if default_strategy not in strategies:
            raise ValueError(
                f"default_strategy {default_strategy!r} is not among the configured strategies {sorted(strategies)}"
            )
        if max_escalations < 0:
            raise ValueError(f"max_escalations must be >= 0 (got {max_escalations})")
        self.strategies = dict(strategies)
        self.default_strategy = default_strategy
        self.eligible_models = list(eligible_models) if eligible_models else None
        self.selection_rules = tuple(selection_rules)
        self.escalation_rules = tuple(escalation_rules)
        self.measurements = measurements
        self.objective = objective
        self.max_escalations = max_escalations
        self.use_heuristic_router = use_heuristic_router
        self.experimental = experimental
        self.name = name

    # ---- constraints --------------------------------------------------

    def narrow_tools(self, request: ExecutionRequest, names: Iterable[str]) -> frozenset[str] | None:
        """Intersect *names* with the request's allow-list.  Never widens it.

        Returns the new allow-list, or ``None`` when the request had none and
        the policy is not adding one.
        """
        wanted = frozenset(str(n) for n in names)
        if request.allowed_tools is None:
            return wanted or None
        return request.allowed_tools & wanted

    def _eligible_model(self, request: ExecutionRequest, strategy: ExecutionStrategy) -> tuple[str, str, str]:
        """Return ``(model, source, reason)`` respecting the eligibility list."""
        pinned = request.model or strategy.model
        allowed = self.eligible_models

        if pinned and (allowed is None or pinned in allowed):
            return pinned, "configured", f"model {pinned} was pinned by the caller"
        if pinned and allowed is not None:
            # A hard constraint wins over a caller preference, loudly.
            logger.warning("pinned model %s is not in eligible_models; ignoring the pin", pinned)

        if not allowed:
            return pinned or "", "configured", "no eligible-model list configured; using the strategy default"

        if len(allowed) == 1:
            return allowed[0], "configured", f"{allowed[0]} is the only eligible model"

        if self.measurements is not None:
            keys = [
                MeasurementKey(
                    category=request.category,
                    strategy=strategy.name,
                    model=model,
                    strategy_version=strategy.version,
                )
                for model in allowed
            ]
            key, _stats, reason = self.measurements.best(keys, objective=self.objective)
            if key is not None:
                return key.model, "measured", reason
            fallback_reason = f"no usable measurement ({reason})"
        else:
            fallback_reason = "no measurement store configured"

        if self.use_heuristic_router:
            model, why = self._heuristic_model(request, allowed)
            if model:
                return model, "heuristic", f"{fallback_reason}; heuristic router chose {model} ({why})"

        return allowed[0], "configured", f"{fallback_reason}; falling back to the first eligible model"

    def _heuristic_model(self, request: ExecutionRequest, allowed: list[str]) -> tuple[str, str]:
        """Ask the existing heuristic router, restricted to eligible models."""
        try:
            from ..pipeline.routing import ModelRouter, RoutingConfig

            schema = request.output_schema or {}
            if not schema and request.output_model is not None:
                json_schema = getattr(request.output_model, "model_json_schema", None)
                schema = json_schema() if callable(json_schema) else {}
            router = ModelRouter(RoutingConfig(strategy="balanced"))
            # Restrict the router to the eligible set — a heuristic must not be
            # able to select outside a hard constraint.
            router._get_available_models = lambda: list(allowed)  # type: ignore[method-assign]
            selected, routing = router.select_model(request.task, schema)
            if selected in allowed:
                return selected, routing.reason
        except Exception as exc:  # pragma: no cover - the router is best-effort
            logger.debug("heuristic routing unavailable: %s", exc)
        return "", ""

    # ---- selection ----------------------------------------------------

    def select(self, request: ExecutionRequest) -> SelectionDecision:
        """Choose the first attempt's strategy and model."""
        for rule in self.selection_rules:
            if rule.strategy not in self.strategies:
                continue
            try:
                matched = bool(rule.when(request))
            except Exception as exc:  # pragma: no cover - a bad predicate must not kill the run
                logger.warning("selection rule %s raised: %s", rule.name, exc)
                continue
            if matched:
                strategy = self.strategies[rule.strategy]
                model, source, why = self._eligible_model(request, strategy)
                return SelectionDecision(
                    stage="select",
                    strategy=rule.strategy,
                    model=model,
                    source="rule",
                    reason=f"rule {rule.name}: {rule.reason}; model {source} ({why})",
                )

        strategy = self.strategies[self.default_strategy]
        model, source, why = self._eligible_model(request, strategy)
        return SelectionDecision(
            stage="select",
            strategy=self.default_strategy,
            model=model,
            source="configured",
            reason=f"no selection rule matched; cold-start default; model {source} ({why})",
        )

    def escalation_for(self, result: ExecutionResult, *, already_used: set[str]) -> EscalationRule | None:
        """The first applicable escalation rule that has not fired yet."""
        for rule in self.escalation_rules:
            if rule.name in already_used or rule.strategy not in self.strategies:
                continue
            try:
                if rule.when(result):
                    return rule
            except Exception as exc:  # pragma: no cover
                logger.warning("escalation rule %s raised: %s", rule.name, exc)
        return None

    # ---- execution ----------------------------------------------------

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        """Select, execute, and escalate within bounds.  Returns the final result."""
        decisions: list[SelectionDecision] = []
        preamble: list[str] = []
        if self.experimental:
            preamble.append(
                "EXPERIMENTAL: adaptive selection has not been shown to beat the fixed "
                "baselines on this workload; compare before relying on it"
            )

        chosen = self.select(request)
        decisions.append(chosen)

        blocked = self._insufficient_before_start(request)
        if blocked is not None:
            return self._abstain(request, chosen, preamble, decisions, blocked)

        attempt_request = request.with_overrides(model=chosen.model or request.model)
        result = self.strategies[chosen.strategy].run(attempt_request)

        used_rules: set[str] = set()
        escalations = 0
        while escalations < self.max_escalations:
            rule = self.escalation_for(result, already_used=used_rules)
            if rule is None:
                break
            used_rules.add(rule.name)
            escalations += 1

            strategy = self._strategy_for_escalation(rule)
            model, _source, _why = self._eligible_model(attempt_request, strategy)
            escalation = SelectionDecision(
                stage="escalate",
                strategy=rule.strategy,
                model=model,
                source="rule",
                reason=f"rule {rule.name}: {rule.reason} (previous outcome: {result.termination.value})",
            )
            decisions.append(escalation)

            overrides = dict(rule.overrides)
            overrides.setdefault("model", model or attempt_request.model)
            retry = strategy.run(attempt_request.with_overrides(**overrides))
            retry.usage.merge(result.usage)
            retry.steps = (*result.steps, *retry.steps)
            retry.decisions = [*result.decisions, *retry.decisions]
            result = retry

        if escalations >= self.max_escalations and self.escalation_for(result, already_used=used_rules) is not None:
            decisions.append(
                SelectionDecision(
                    stage="stop",
                    strategy=result.strategy,
                    reason=f"escalation bound reached ({self.max_escalations}); not trying anything further",
                )
            )

        result.decisions = [*preamble, *(d.format() for d in decisions), *result.decisions]
        result.artifacts["policy_decisions"] = [d.to_dict() for d in decisions]
        return result

    def _strategy_for_escalation(self, rule: EscalationRule) -> ExecutionStrategy:
        """The configured strategy, rebuilt with the rule's option overrides.

        Falls back to the configured instance when it cannot be rebuilt — a
        custom strategy with an unusual constructor must not break escalation.
        """
        base = self.strategies[rule.strategy]
        if not rule.strategy_options:
            return base
        try:
            return type(base)(
                model=base.model,
                driver=base._driver,
                driver_factory=base._driver_factory,
                options=dict(base.options),
                **rule.strategy_options,
            )
        except TypeError as exc:
            logger.warning(
                "could not apply strategy_options %s to %s (%s); escalating with the configured instance",
                rule.strategy_options,
                type(base).__name__,
                exc,
            )
            return base

    async def arun(self, request: ExecutionRequest) -> ExecutionResult:
        """Async entry point.  Offloads :meth:`run` so it never blocks a loop."""
        import asyncio

        return await asyncio.to_thread(self.run, request)

    def __call__(self, request: ExecutionRequest) -> ExecutionResult:
        return self.run(request)

    # ---- stop / abstain -----------------------------------------------

    def _insufficient_before_start(self, request: ExecutionRequest) -> str | None:
        """Reason to abstain before spending anything, or ``None``."""
        limits = request.limits
        if limits.max_llm_calls is not None and limits.max_llm_calls < 1:
            return f"the call budget is {limits.max_llm_calls}; no attempt is possible"
        if limits.max_steps is not None and limits.max_steps < 1:
            return f"the step budget is {limits.max_steps}; no attempt is possible"
        if limits.max_cost_usd is not None and limits.max_cost_usd <= 0:
            return f"the cost budget is ${limits.max_cost_usd}; no attempt is possible"
        if request.category is TaskCategory.DOCUMENT_QA and not request.passages and request.retriever is None:
            return "the task requires evidence but neither passages nor a retriever were supplied"
        return None

    def _abstain(
        self,
        request: ExecutionRequest,
        chosen: SelectionDecision,
        preamble: list[str],
        decisions: list[SelectionDecision],
        reason: str,
    ) -> ExecutionResult:
        decisions.append(SelectionDecision(stage="stop", strategy=chosen.strategy, source="rule", reason=reason))
        return ExecutionResult(
            termination=TerminationReason.ABSTAINED,
            strategy=self.name,
            model=chosen.model,
            decisions=[*preamble, *(d.format() for d in decisions)],
            error=None,
        )


# ---------------------------------------------------------------------------
# Shadow comparison
# ---------------------------------------------------------------------------


def shadow_compare(
    request: ExecutionRequest,
    baseline: Any,
    candidate: Any,
    *,
    allow_side_effects: bool = False,
) -> dict[str, ExecutionResult]:
    """Run a baseline and a candidate over the same request, side by side.

    Refuses by default when the request carries write-capable tools, because a
    shadow run would perform the external action twice.  Point the comparison at
    a sandbox (:class:`~prompture.execution.sandbox.InventoryWorld`) or at a
    read-only tool subset instead; ``allow_side_effects=True`` exists only for
    the case where the caller has confirmed the tools are idempotent.

    Returns ``{"baseline": ..., "candidate": ...}``.
    """
    if not allow_side_effects:
        writers = _write_tools(request)
        if writers:
            raise ValueError(
                "Refusing to shadow-run a request whose tools can cause external "
                f"effects: {sorted(writers)}. Duplicating them would perform each "
                "action twice. Use a sandbox world, restrict allowed_tools to "
                "read-only tools, or pass allow_side_effects=True if they are "
                "genuinely idempotent."
            )
    return {
        "baseline": baseline(request.with_overrides()),
        "candidate": candidate(request.with_overrides()),
    }


def _write_tools(request: ExecutionRequest) -> set[str]:
    """Names of tools on the request that declare themselves write-capable."""
    tools = request.effective_tools()
    definitions = getattr(tools, "definitions", None)
    if not definitions:
        return set()
    writers: set[str] = set()
    for definition in definitions:
        metadata = getattr(definition, "metadata", {}) or {}
        if metadata.get("sandbox"):
            continue
        if metadata.get("is_write"):
            writers.add(definition.name)
    return writers
