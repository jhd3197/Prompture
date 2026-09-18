"""``direct`` — one primary attempt, then validate, with bounded repair.

The cheapest of the three strategies and the right default when the task has a
typed contract and no evidence requirement.  It is a thin adapter over
:func:`prompture.extraction.core.ask_for_json`, so it inherits that function's
provider-compatibility handling (native JSON mode, tool-call extraction,
prompted repair), its caching, and its usage metadata — this strategy adds the
result envelope, the accounting, and an explicitly *separate* repair bound.

Two repair budgets, deliberately distinct
-----------------------------------------

``ask_for_json`` already repairs *malformed JSON* internally (its ``ai_cleanup``
pass).  This strategy adds a second, outer loop that repairs *schema-invalid but
well-formed* output by telling the model which fields failed.  They are bounded
separately (``ai_cleanup`` and ``max_repairs``) so a caller can turn one off
without losing the other, and every attempt is its own step record.
"""

from __future__ import annotations

from typing import Any

from ..outcomes import TerminationReason
from ..types import ExecutionRequest, ExecutionResult
from .base import (
    ExecutionStrategy,
    StrategyContext,
    compact_json,
    schema_for,
    validate_against_model,
)

__all__ = ["DirectStrategy"]


_REPAIR_TEMPLATE = """Your previous answer did not satisfy the required schema.

Previous answer:
{previous}

Validation errors (field: problem):
{errors}

Produce a corrected answer. Fix only the listed fields; keep every other value
unchanged. If a value is genuinely unknown, use null rather than inventing one."""


class DirectStrategy(ExecutionStrategy):
    """Execute one primary attempt and validate it.

    Args:
        model: Default model string.
        driver / driver_factory: See :class:`~prompture.execution.strategies.base.ExecutionStrategy`.
        options: Driver options.
        max_repairs: How many *schema-repair* attempts may follow the first
            answer.  ``0`` disables the outer repair loop entirely; the default
            ``1`` gives one targeted correction pass.
        ai_cleanup: Forwarded to ``ask_for_json`` — repairs malformed JSON.
            Independent of ``max_repairs``.
        instruction: Default instruction prefix when the request has none.

    Example::

        from pydantic import BaseModel
        from prompture.execution import ExecutionRequest
        from prompture.execution.strategies import DirectStrategy

        class Contact(BaseModel):
            name: str | None = None
            email: str | None = None

        strategy = DirectStrategy(model="openai/gpt-4o-mini")
        result = strategy.run(ExecutionRequest(task="Call Ana on ana@x.example", output_model=Contact))
        print(result.output, result.termination.value)
    """

    name = "direct"
    version = "1"
    native_mode = "sync"

    def __init__(
        self,
        *,
        max_repairs: int = 1,
        ai_cleanup: bool = True,
        instruction: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if max_repairs < 0:
            raise ValueError(f"max_repairs must be >= 0 (got {max_repairs})")
        self.max_repairs = max_repairs
        self.ai_cleanup = ai_cleanup
        self.instruction = instruction

    # ------------------------------------------------------------------

    def _execute(self, ctx: StrategyContext) -> ExecutionResult:
        request = ctx.request
        schema = schema_for(request)
        if schema is None:
            return self._execute_freeform(ctx)
        return self._execute_typed(ctx, schema)

    # ---- typed path ---------------------------------------------------

    def _execute_typed(self, ctx: StrategyContext, schema: dict[str, Any]) -> ExecutionResult:
        from ...extraction.core import ask_for_json

        request = ctx.request
        driver = self.resolve_driver(ctx.model)
        options = self.merged_options(request)
        system_prompt = self.system_prompt_for(request)
        instruction = request.instruction or self.instruction
        prompt = f"{instruction}\n\n{request.task}" if instruction else request.task

        payload: Any = None
        report = None
        last_json_string = ""
        raw_result: Any = None

        for attempt in range(self.max_repairs + 1):
            ctx.check_cancelled()
            ctx.stop_if_over_budget(prompt=prompt, model=ctx.model)

            kind = "generate" if attempt == 0 else "repair"
            detail = "primary attempt" if attempt == 0 else f"schema repair {attempt}"
            with ctx.step(kind, detail=detail) as step:
                response = ask_for_json(
                    driver,
                    prompt,
                    schema,
                    ai_cleanup=self.ai_cleanup,
                    model_name=ctx.model,
                    options=options,
                    system_prompt=system_prompt,
                    images=None,
                    strategy=request.structured_output_strategy,
                )
                usage = (response or {}).get("usage") or {}
                ctx.book(step, usage)
                payload = (response or {}).get("json_object")
                last_json_string = (response or {}).get("json_string", "") or ""
                raw_result = response
                resolved = usage.get("strategy")
                if resolved:
                    ctx.structured_output_strategy = str(resolved)
                step.data["cache_hit"] = bool(usage.get("cache_hit"))

            with ctx.step("validate", detail=f"attempt {attempt}") as step:
                instance, report = validate_against_model(request.output_model, payload, attempts=attempt)
                step.ok = report.ok is not False
                step.data["field_errors"] = dict(report.field_errors)

            if report.ok is not False:
                ctx.note(f"validated on attempt {attempt}")
                return ctx.finish(
                    termination=TerminationReason.COMPLETED,
                    output=instance if instance is not None else payload,
                    answer=last_json_string,
                    validation=report,
                    raw=raw_result,
                )

            if attempt >= self.max_repairs:
                break

            failed = ", ".join(sorted(report.field_errors)) or "the whole object"
            ctx.note(f"attempt {attempt} failed validation on {failed}; requesting a targeted repair")
            prompt = _REPAIR_TEMPLATE.format(
                previous=compact_json(payload),
                errors="\n".join(f"- {path}: {message}" for path, message in sorted(report.field_errors.items()))
                or "- the output did not match the schema",
            )

        ctx.note(f"exhausted {self.max_repairs} repair attempt(s) without a valid object")
        return ctx.finish(
            termination=TerminationReason.VALIDATION_FAILED,
            output=payload,
            answer=last_json_string,
            validation=report,
            raw=raw_result,
        )

    # ---- free-text path -----------------------------------------------

    def _execute_freeform(self, ctx: StrategyContext) -> ExecutionResult:
        """No schema was requested: one call, no validation claim."""
        request = ctx.request
        driver = self.resolve_driver(ctx.model)
        options = self.merged_options(request)
        system_prompt = self.system_prompt_for(request)
        if system_prompt:
            options = {**options, "system_prompt": system_prompt}

        instruction = request.instruction or self.instruction
        prompt = f"{instruction}\n\n{request.task}" if instruction else request.task

        ctx.stop_if_over_budget(prompt=prompt, model=ctx.model)
        with ctx.step("generate", detail="primary attempt (no schema)") as step:
            response = ctx.generate(driver, prompt, options, step)
        answer = (response or {}).get("text", "") or ""
        ctx.note("no output contract was supplied; schema validity is unchecked")
        return ctx.finish(termination=TerminationReason.COMPLETED, answer=answer, raw=response)


def direct(request: ExecutionRequest, **kwargs: Any) -> ExecutionResult:
    """One-shot convenience wrapper around :class:`DirectStrategy`."""
    return DirectStrategy(**kwargs).run(request)
