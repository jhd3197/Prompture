"""The strategy protocol and the scaffolding every strategy shares.

An :class:`ExecutionStrategy` turns an
:class:`~prompture.execution.types.ExecutionRequest` into an
:class:`~prompture.execution.types.ExecutionResult`.  It owns *how many model
steps run and what checks them* — nothing else.  Provider compatibility stays in
:class:`~prompture.extraction.strategy.StructuredOutputStrategy`; prompt
augmentation stays in :mod:`prompture.extraction.reasoning`; model choice stays
in routing and (opt-in) in :mod:`prompture.execution.policy`.

Shared behaviour lives in :class:`StrategyContext`, so all three built-in
strategies agree on:

* **Step records** — every model call, retrieval, validation and review is one
  :class:`~prompture.execution.types.StepRecord`.  The sequence is the
  explanation of the run.
* **Accounting** — every call is booked once, into both the step and the run
  total, including reviewers, repairs and retrieval generation.  A call with no
  resolvable price makes the run's cost a lower bound rather than a total.
* **Budgets** — checked *before* starting more work, using a pre-flight estimate
  when the model has pricing.  An in-flight call is still billed; the result's
  ``budget_note`` says so.
* **Cancellation** — cooperative, checked at step boundaries.
* **Termination** — always explicit, never inferred from an empty output.

Sync / async support
--------------------

:meth:`ExecutionStrategy.run` is the primary entry point for
:class:`DirectStrategy` and :class:`RetrieveAndVerifyStrategy`, which drive
synchronous drivers; their :meth:`arun` offloads to a worker thread.
:class:`DraftAndCritiqueStrategy` is the mirror image: it reuses the async
:class:`~prompture.agents.review_loop.AsyncReviewLoop` engine, so :meth:`arun` is
native and :meth:`run` drives the coroutine to completion — which requires that
no event loop is already running on the calling thread.  Each strategy declares
which side it is native on through :attr:`ExecutionStrategy.native_mode`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from time import perf_counter
from typing import Any

from ..outcomes import TerminationReason, UsageAccounting, ValidationReport
from ..types import (
    BudgetLedger,
    Cancelled,
    ExecutionRequest,
    ExecutionResult,
    StepRecord,
)

logger = logging.getLogger("prompture.execution.strategy")

__all__ = [
    "ExecutionStrategy",
    "StrategyContext",
    "StrategyError",
    "run_sync",
]


class StrategyError(RuntimeError):
    """Raised for a misconfigured strategy (not for a provider failure).

    Provider failures are captured into an
    :class:`~prompture.execution.types.ExecutionResult` with
    ``termination=PROVIDER_ERROR`` instead of propagating, so a benchmark sweep
    survives one bad call.
    """


class _BudgetStop(RuntimeError):
    """Internal: raised to unwind out of a nested loop when budget is exhausted."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def run_sync(coro: Any) -> Any:
    """Drive a coroutine to completion from synchronous code.

    Raises :class:`StrategyError` when an event loop is already running on this
    thread — the caller should ``await`` the strategy's ``arun`` instead.  Being
    explicit beats silently spawning a second loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise StrategyError(
        "This strategy is async-native and run() cannot be called from inside a "
        "running event loop. Await `arun(request)` instead."
    )


class StrategyContext:
    """Per-run state shared by every strategy: ledger, steps, decisions.

    A context is created once per :meth:`ExecutionStrategy.run` call and is not
    reused, so strategies themselves stay stateless and safe to share.
    """

    def __init__(self, request: ExecutionRequest, strategy: ExecutionStrategy) -> None:
        self.request = request
        self.strategy = strategy
        self.ledger = BudgetLedger(request.limits)
        self.steps: list[StepRecord] = []
        self.decisions: list[str] = []
        self.artifacts: dict[str, Any] = {}
        self.model = request.model or ""
        self._price_known_cache: dict[str, bool] = {}
        self.structured_output_strategy = request.structured_output_strategy or ""
        self._started = perf_counter()

    # ---- narration ----------------------------------------------------

    def note(self, message: str) -> None:
        """Record an observable decision reason."""
        self.decisions.append(message)
        logger.debug("[%s] %s", self.strategy.name, message)

    # ---- guards -------------------------------------------------------

    def check_cancelled(self) -> None:
        """Raise :class:`~prompture.execution.types.Cancelled` if requested."""
        if self.request.cancel is not None:
            self.request.cancel.raise_if_cancelled()

    def budget_blocker(self, *, prompt: str = "", model: str = "") -> str | None:
        """Return a reason string when further work must not start.

        A pre-flight cost estimate is used when the model has resolvable
        pricing.  When it does not, the estimate contributes ``0.0`` — the check
        then relies on observed spend only, which is weaker but never blocks
        work on an invented number.
        """
        estimate = 0.0
        target = model or self.model
        if prompt and target and self.request.limits.max_cost_usd is not None:
            with contextlib.suppress(Exception):
                from ...infra.budget import estimate_call_cost

                forecast = estimate_call_cost(target, prompt)
                if forecast.rates_available:
                    estimate = forecast.total_cost
        return self.ledger.check(estimated_next_cost=estimate)

    def stop_if_over_budget(self, *, prompt: str = "", model: str = "") -> None:
        """Raise :class:`_BudgetStop` when the budget forbids further work."""
        reason = self.budget_blocker(prompt=prompt, model=model)
        if reason:
            raise _BudgetStop(reason)

    # ---- steps --------------------------------------------------------

    @contextlib.contextmanager
    def step(self, kind: str, *, model: str = "", detail: str = "") -> Iterator[StepRecord]:
        """Open a step, timing it and marking it failed if the body raises."""
        record = StepRecord(index=len(self.steps), kind=kind, model=model or self.model, detail=detail)
        self.steps.append(record)
        self.ledger.record_step()
        started = perf_counter()
        try:
            yield record
        except Exception as exc:
            record.ok = False
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.elapsed_ms = (perf_counter() - started) * 1000

    def book(self, step: StepRecord, meta: dict[str, Any] | None, *, model: str | None = None) -> None:
        """Book one provider call into both the step and the run total.

        A reported cost of ``0.0`` is only trusted when the model has a
        resolvable rate card; otherwise it is booked as an unknown price, which
        is what keeps the run's total honest about being a lower bound.
        """
        target = model or step.model or self.model
        known = self.price_known(target)
        step.usage.record(meta, model=target, price_known=known)
        self.ledger.record(meta, model=target, price_known=known)

    def price_known(self, model: str) -> bool | None:
        """Whether *model* has resolvable pricing.  ``None`` when undeterminable.

        Cached per run so a multi-step strategy performs at most one rate lookup
        per model.
        """
        if not model:
            return None
        if model in self._price_known_cache:
            return self._price_known_cache[model]
        known: bool | None = None
        provider, _, model_id = model.partition("/")
        if model_id:
            try:
                from ...infra.model_rates import get_model_rates

                rates = get_model_rates(provider, model_id)
                known = bool(rates and (rates.get("input") is not None or rates.get("output") is not None))
            except Exception:  # pragma: no cover - pricing lookup is best-effort
                known = None
        if known is not None:
            self._price_known_cache[model] = known
        return known

    def book_usage(self, step: StepRecord, usage: UsageAccounting | None) -> None:
        """Fold an already-aggregated accounting (from a nested component) in."""
        if usage is None:
            return
        step.usage.merge(usage)
        self.ledger.usage.merge(usage)

    # ---- driver -------------------------------------------------------

    def generate(
        self,
        driver: Any,
        prompt: str,
        options: dict[str, Any],
        step: StepRecord,
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Call a driver through its hook wrapper and book the usage.

        The hook wrapper (``generate_with_hooks``) is preferred over a bare
        ``generate`` so the call is recorded by the usage tracker and reaches any
        configured :class:`~prompture.infra.callbacks.DriverCallbacks` — a bare
        call would run off the books.
        """
        self.check_cancelled()
        generate = getattr(driver, "generate_with_hooks", None) or driver.generate
        response = generate(prompt, options)
        self.book(step, (response or {}).get("meta") or {}, model=model)
        self.check_cancelled()
        return response or {}

    # ---- completion ---------------------------------------------------

    def finish(
        self,
        *,
        termination: TerminationReason,
        output: Any = None,
        answer: str = "",
        validation: ValidationReport | None = None,
        evidence: Any = None,
        error: str | None = None,
        raw: Any = None,
    ) -> ExecutionResult:
        """Assemble the :class:`ExecutionResult` for this run."""
        from ..outcomes import EvidenceReport

        return ExecutionResult(
            output=output,
            answer=answer or "",
            termination=termination,
            validation=validation or ValidationReport(),
            evidence=evidence or EvidenceReport(),
            steps=tuple(self.steps),
            usage=self.ledger.usage,
            model=self.model,
            strategy=self.strategy.name,
            strategy_version=self.strategy.version,
            structured_output_strategy=self.structured_output_strategy,
            decisions=list(self.decisions),
            elapsed_ms=(perf_counter() - self._started) * 1000,
            error=error,
            artifacts=dict(self.artifacts),
            raw=raw,
            budget_note=self.ledger.enforcement_note(),
        )


class ExecutionStrategy(ABC):
    """Base class for a fixed execution strategy.

    Subclasses implement :meth:`_execute` (sync-native) or :meth:`_aexecute`
    (async-native) and set :attr:`native_mode` accordingly.  The public
    :meth:`run` / :meth:`arun` wrappers add the shared error, cancellation and
    budget handling so no strategy has to repeat it.

    Args:
        model: Default model string used when a request does not pin one.
        driver: Pre-built driver to use for every call.  Injecting one is how
            the contract tests run offline.
        driver_factory: Callable resolving a model string to a driver.  Defaults
            to :func:`prompture.drivers.get_driver_for_model`.
        options: Driver options merged under the request's own options.
    """

    #: Stable strategy name, recorded on every outcome.
    name: str = "strategy"
    #: Bump when the strategy's observable behaviour changes.
    version: str = "1"
    #: ``"sync"`` or ``"async"`` — which side the implementation is native on.
    native_mode: str = "sync"

    def __init__(
        self,
        *,
        model: str | None = None,
        driver: Any = None,
        driver_factory: Callable[[str], Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self._driver = driver
        self._driver_factory = driver_factory
        self.options = dict(options or {})

    # ---- driver resolution --------------------------------------------

    def resolve_model(self, request: ExecutionRequest) -> str:
        """The model string this run should use.

        Request wins over strategy default.  When neither is set and a driver
        was injected, its own ``model`` attribute is used, so a stub driver does
        not have to be paired with a model string.
        """
        model = request.model or self.model
        if not model and self._driver is not None:
            model = getattr(self._driver, "model", "") or getattr(self._driver, "model_name", "")
        return model or ""

    def resolve_driver(self, model: str) -> Any:
        """Return the driver for *model*."""
        if self._driver is not None:
            return self._driver
        if self._driver_factory is not None:
            return self._driver_factory(model)
        if not model:
            raise StrategyError(
                f"{self.name} needs a model: pass `model=` on the strategy or the request, or inject a driver."
            )
        from ...drivers import get_driver_for_model

        return get_driver_for_model(model)

    def merged_options(self, request: ExecutionRequest) -> dict[str, Any]:
        """Strategy options with the request's options layered on top."""
        options = dict(self.options)
        options.update(request.options or {})
        return options

    def system_prompt_for(self, request: ExecutionRequest) -> str | None:
        """Render the request's persona, or fall back to its plain system prompt.

        The same persona works with all three strategies, and a persona whose
        system prompt is callable keeps working — ``render`` is called with the
        request's template variables exactly as ``AsyncAgent`` would.
        """
        persona = request.persona
        if persona is None:
            return request.system_prompt
        render = getattr(persona, "render", None)
        if callable(render):
            return render(**(request.variables or {}))
        return str(persona)

    # ---- execution ----------------------------------------------------

    @abstractmethod
    def _execute(self, ctx: StrategyContext) -> ExecutionResult:
        """Sync-native implementation.  Override for ``native_mode='sync'``."""

    async def _aexecute(self, ctx: StrategyContext) -> ExecutionResult:
        """Async-native implementation.  Override for ``native_mode='async'``."""
        raise NotImplementedError

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute *request* synchronously and return the result envelope.

        Never raises for a provider failure, a budget stop, or cancellation —
        each becomes an explicit ``termination``.  A :class:`StrategyError`
        (misconfiguration) does propagate, because retrying it is pointless.
        """
        ctx = StrategyContext(request, self)
        ctx.model = self.resolve_model(request)
        try:
            if self.native_mode == "async":
                return run_sync(self._aexecute(ctx))
            return self._execute(ctx)
        except Exception as exc:
            return self._to_failure(ctx, exc)

    async def arun(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute *request*, awaiting the async-native path where there is one.

        For a sync-native strategy this offloads :meth:`run` to a worker thread,
        so an async caller never blocks its event loop on a synchronous driver.
        """
        if self.native_mode != "async":
            return await asyncio.to_thread(self.run, request)
        ctx = StrategyContext(request, self)
        ctx.model = self.resolve_model(request)
        try:
            return await self._aexecute(ctx)
        except Exception as exc:
            return self._to_failure(ctx, exc)

    def __call__(self, request: ExecutionRequest) -> ExecutionResult:
        """Strategies are usable directly as a harness ``executor``."""
        return self.run(request)

    # ---- failure mapping ----------------------------------------------

    def _to_failure(self, ctx: StrategyContext, exc: Exception) -> ExecutionResult:
        if isinstance(exc, StrategyError):
            raise exc
        if isinstance(exc, Cancelled):
            ctx.note(f"cancelled: {exc}")
            return ctx.finish(termination=TerminationReason.CANCELLED, error=str(exc))
        if isinstance(exc, _BudgetStop):
            ctx.note(f"stopped before further work: {exc.reason}")
            return ctx.finish(termination=TerminationReason.BUDGET_EXHAUSTED, error=exc.reason)
        from ...exceptions import BudgetExceededError

        if isinstance(exc, BudgetExceededError):
            ctx.note(f"budget exceeded: {exc}")
            return ctx.finish(termination=TerminationReason.BUDGET_EXHAUSTED, error=str(exc))
        ctx.note(f"provider error: {type(exc).__name__}")
        logger.warning("[%s] provider error: %s", self.name, exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        return ctx.finish(
            termination=TerminationReason.PROVIDER_ERROR,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Shared validation helper
# ---------------------------------------------------------------------------


def validate_against_model(model_cls: Any, payload: Any, *, attempts: int = 0) -> tuple[Any, ValidationReport]:
    """Validate *payload* against a Pydantic model, producing a report.

    Returns ``(instance_or_None, report)``.  Field-level errors are keyed by
    dotted path so a repair pass can target exactly the fields that failed
    rather than regenerating the whole object.
    """
    report = ValidationReport(checked=True, repair_attempts=attempts)
    if model_cls is None:
        report.checked = False
        return payload, report
    try:
        instance = model_cls.model_validate(payload)
    except Exception as exc:
        report.schema_valid = False
        errors = getattr(exc, "errors", None)
        if callable(errors):
            for error in errors():
                path = ".".join(str(p) for p in error.get("loc", ())) or "<root>"
                message = str(error.get("msg", "invalid"))
                report.field_errors[path] = message
                report.errors.append(f"{path}: {message}")
        else:
            report.errors.append(str(exc))
        return None, report
    report.schema_valid = True
    return instance, report


def schema_for(request: ExecutionRequest) -> dict[str, Any] | None:
    """The JSON schema this request wants, from its model class or raw schema."""
    if request.output_schema is not None:
        return request.output_schema
    model_cls = request.output_model
    if model_cls is None:
        return None
    json_schema = getattr(model_cls, "model_json_schema", None)
    if callable(json_schema):
        return json_schema()
    raise StrategyError(
        f"output_model must be a Pydantic model class (got {type(model_cls).__name__}); "
        "pass output_schema=... for a raw JSON schema."
    )


def compact_json(value: Any, *, limit: int = 2000) -> str:
    """Serialise *value* for inclusion in a prompt, truncated at *limit*."""
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:  # pragma: no cover - defensive
        text = str(value)
    if len(text) > limit:
        return text[:limit] + f"\n… (truncated, {len(text)} chars total)"
    return text
