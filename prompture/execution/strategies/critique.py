"""``draft_critique`` — produce a candidate, critique it, revise within a bound.

This strategy does not implement a review loop; it adapts the existing one.
:class:`~prompture.agents.review_loop.AsyncReviewLoop` owns the iteration,
approval predicate, feedback folding and event stream.  What this module adds is
the pair of runnables the loop drives (a drafter and a reviewer, both backed by
whatever driver the request resolves to), *structured* review feedback instead of
free prose, and the shared accounting/termination contract.

Why structured feedback
-----------------------

A substring check for the word "approved" is fragile and gives a revision pass
nothing to act on.  The reviewer here is asked for a typed verdict —
``{approved, score, issues[], suggestions[]}`` — so approval is a field rather
than a guess, and the next draft receives the specific issues to fix.  When a run
ends unapproved, the outstanding issues are on the result, which is what makes
``REVIEW_REJECTED`` actionable instead of merely negative.

Reviewer approval is not correctness.  It is recorded as its own signal and is
never folded into schema validity or evidence support.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ..outcomes import TerminationReason, ValidationReport
from ..types import ExecutionRequest, ExecutionResult
from .base import (
    ExecutionStrategy,
    StrategyContext,
    _BudgetStop,
    schema_for,
    validate_against_model,
)

__all__ = ["DraftAndCritiqueStrategy"]


_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean", "description": "True only when the work needs no further change."},
        "score": {"type": "integer", "description": "Quality score from 0 to 10."},
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete problems that must be fixed. Empty when approved.",
        },
        "suggestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional improvements that are not blocking.",
        },
    },
    "required": ["approved", "issues"],
}

_REVIEW_PROMPT = """Review the work below against the original task.

Original task:
{task}

Work to review:
{draft}

Judge only what is present. List concrete, fixable issues. Approve only when you
would ship it as-is."""

_REVISION_PROMPT = """{task}

---
Your previous attempt:
{draft}

The reviewer did not approve it. Issues to fix:
{issues}

{suggestions}Revise the work to address every issue. Change nothing else."""


@dataclass
class _Draft:
    """Loop-facing wrapper for a drafting turn (needs a ``.output`` string)."""

    output: str
    payload: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Review:
    """Loop-facing wrapper for a review turn."""

    output: str
    verdict: dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return bool(self.verdict.get("approved"))

    @property
    def issues(self) -> list[str]:
        return [str(i) for i in self.verdict.get("issues") or []]


class _DriverRunnable:
    """Adapts a driver call to the loop's ``async arun(prompt, **kwargs)`` shape."""

    def __init__(self, ctx: StrategyContext, strategy: DraftAndCritiqueStrategy, role: str) -> None:
        self.ctx = ctx
        self.strategy = strategy
        self.role = role

    async def arun(self, prompt: str, **_: Any) -> Any:
        ctx = self.ctx
        ctx.check_cancelled()
        # Pre-flight: refuse to start another turn we cannot afford.
        ctx.stop_if_over_budget(prompt=prompt, model=ctx.model)
        return await asyncio.to_thread(self._call, prompt)

    def _call(self, prompt: str) -> Any:
        if self.role == "draft":
            return self.strategy._draft(self.ctx, prompt)
        return self.strategy._review(self.ctx, prompt)


class DraftAndCritiqueStrategy(ExecutionStrategy):
    """Draft, obtain structured feedback, and revise within an iteration limit.

    Args:
        model / driver / driver_factory / options: See
            :class:`~prompture.execution.strategies.base.ExecutionStrategy`.
        max_iterations: Hard bound on drafting turns (one review per draft).
            Reaching it without approval terminates as ``REVIEW_REJECTED``.
        reviewer_model: Optional different model for the critique turn.  When
            unset, the drafter's model reviews its own work — cheaper, and
            recorded as such in the decisions so nobody mistakes it for an
            independent check.
        min_score: Optional additional bar; a review that approves but scores
            below this is treated as not approved.
        accept_unapproved: When ``True``, a run that exhausts its iterations
            still returns the last draft as ``COMPLETED``.  Off by default:
            silently accepting rejected work is exactly the failure this
            strategy exists to catch.

    Async note: this strategy is async-native.  ``arun`` is the direct path;
    ``run`` drives the coroutine and therefore cannot be called from inside a
    running event loop.
    """

    name = "draft_critique"
    version = "1"
    native_mode = "async"

    def __init__(
        self,
        *,
        max_iterations: int = 3,
        reviewer_model: str | None = None,
        min_score: int | None = None,
        accept_unapproved: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1 (got {max_iterations})")
        self.max_iterations = max_iterations
        self.reviewer_model = reviewer_model
        self.min_score = min_score
        self.accept_unapproved = accept_unapproved

    # ---- turns --------------------------------------------------------

    def _draft(self, ctx: StrategyContext, prompt: str) -> _Draft:
        request = ctx.request
        driver = self.resolve_driver(ctx.model)
        schema = schema_for(request)
        index = sum(1 for s in ctx.steps if s.kind in {"generate", "revise"})
        kind = "generate" if index == 0 else "revise"

        with ctx.step(kind, detail=f"draft {index}") as step:
            if schema is None:
                options = self.merged_options(request)
                system_prompt = self.system_prompt_for(request)
                if system_prompt:
                    options = {**options, "system_prompt": system_prompt}
                response = ctx.generate(driver, prompt, options, step)
                text = (response or {}).get("text", "") or ""
                return _Draft(output=text, payload=None, meta=(response or {}).get("meta") or {})

            from ...extraction.core import ask_for_json

            response = ask_for_json(
                driver,
                prompt,
                schema,
                ai_cleanup=True,
                model_name=ctx.model,
                options=self.merged_options(request),
                system_prompt=self.system_prompt_for(request),
                strategy=request.structured_output_strategy,
            )
            usage = (response or {}).get("usage") or {}
            ctx.book(step, usage)
            if usage.get("strategy"):
                ctx.structured_output_strategy = str(usage["strategy"])
            return _Draft(
                output=(response or {}).get("json_string", "") or "",
                payload=(response or {}).get("json_object"),
                meta=usage,
            )

    def _review(self, ctx: StrategyContext, prompt: str) -> _Review:
        from ...extraction.core import ask_for_json

        model = self.reviewer_model or ctx.model
        driver = self.resolve_driver(model) if self.reviewer_model else self.resolve_driver(ctx.model)
        index = sum(1 for s in ctx.steps if s.kind == "review")

        with ctx.step("review", model=model, detail=f"review {index}") as step:
            response = ask_for_json(
                driver,
                prompt,
                _REVIEW_SCHEMA,
                ai_cleanup=True,
                model_name=model,
                options=self.merged_options(ctx.request),
                strategy=ctx.request.structured_output_strategy,
            )
            ctx.book(step, (response or {}).get("usage") or {}, model=model)
            verdict = (response or {}).get("json_object") or {}
            step.data["verdict"] = verdict
        return _Review(output=(response or {}).get("json_string", "") or "", verdict=verdict)

    # ---- loop ---------------------------------------------------------

    def _approve_when(self, review: Any) -> bool:
        verdict = getattr(review, "verdict", {}) or {}
        if not verdict.get("approved"):
            return False
        if self.min_score is None:
            return True
        score = verdict.get("score")
        return isinstance(score, (int, float)) and score >= self.min_score

    def _execute(self, ctx: StrategyContext) -> ExecutionResult:  # pragma: no cover - async-native
        raise NotImplementedError("DraftAndCritiqueStrategy is async-native; use arun() or run()")

    async def _aexecute(self, ctx: StrategyContext) -> ExecutionResult:
        from ...agents.review_loop import AsyncReviewLoop

        request = ctx.request
        task = f"{request.instruction}\n\n{request.task}" if request.instruction else request.task
        if self.reviewer_model is None:
            ctx.note("the drafting model also reviews its own work; this is not an independent check")

        loop = AsyncReviewLoop(
            coder=_DriverRunnable(ctx, self, "draft"),
            reviewer=_DriverRunnable(ctx, self, "review"),
            max_iters=self.max_iterations,
            approve_when=self._approve_when,
            review_prompt=lambda draft: _REVIEW_PROMPT.format(task=task, draft=draft),
            feedback_prompt=lambda original, iteration: _revision_prompt(task, iteration),
        )

        last_draft: _Draft | None = None
        last_review: _Review | None = None
        approved = False
        iterations = 0
        budget_reason: str | None = None
        loop_result: Any = None

        stream = loop.astream(task)
        try:
            async for event in stream:
                if event.type == "coder_done":
                    last_draft = event.code_result
                    ctx.artifacts["last_draft"] = getattr(last_draft, "output", "")
                elif event.type == "reviewer_done":
                    last_review = event.review_result
                    iterations = event.index + 1
                    approved = event.approved
                    if not approved and last_review is not None:
                        ctx.note(f"iteration {event.index}: not approved — {last_review.issues}")
                elif event.type == "done":
                    loop_result = event.result
        except _BudgetStop as stop:
            budget_reason = stop.reason
            ctx.note(f"stopped before another turn: {stop.reason}")
        finally:
            await stream.aclose()

        payload = getattr(last_draft, "payload", None)
        text = getattr(last_draft, "output", "") or ""
        issues = last_review.issues if last_review is not None else []

        validation: ValidationReport | None = None
        output: Any = payload
        if payload is not None and (request.output_model is not None or request.output_schema is not None):
            with ctx.step("validate") as step:
                instance, validation = validate_against_model(request.output_model, payload)
                step.ok = validation.ok is not False
                step.data["field_errors"] = dict(validation.field_errors)
            if validation.ok is False:
                ctx.note("final draft did not satisfy the output contract")
                return ctx.finish(
                    termination=TerminationReason.VALIDATION_FAILED,
                    output=payload,
                    answer=text,
                    validation=validation,
                    raw=loop_result,
                )
            output = instance if instance is not None else payload

        ctx.artifacts["review_issues"] = issues
        if last_review is not None:
            ctx.artifacts["review_verdict"] = dict(last_review.verdict)

        if budget_reason is not None:
            return ctx.finish(
                termination=TerminationReason.BUDGET_EXHAUSTED,
                output=output,
                answer=text,
                validation=validation,
                error=budget_reason,
                raw=loop_result,
            )

        if approved:
            ctx.note(f"approved after {iterations} iteration(s)")
            return ctx.finish(
                termination=TerminationReason.COMPLETED,
                output=output,
                answer=text,
                validation=validation,
                raw=loop_result,
            )

        if last_draft is None:
            ctx.note("no draft was produced")
            return ctx.finish(termination=TerminationReason.REVIEW_REJECTED, raw=loop_result)

        if self.accept_unapproved:
            ctx.note(f"iteration limit reached unapproved; returning the last draft ({len(issues)} open issue(s))")
            return ctx.finish(
                termination=TerminationReason.COMPLETED,
                output=output,
                answer=text,
                validation=validation,
                raw=loop_result,
            )

        ctx.note(f"reviewer never approved within {self.max_iterations} iteration(s); {len(issues)} issue(s) remain")
        return ctx.finish(
            termination=TerminationReason.REVIEW_REJECTED,
            output=output,
            answer=text,
            validation=validation,
            raw=loop_result,
        )


def _revision_prompt(task: str, iteration: Any) -> str:
    review = getattr(iteration, "review_result", None)
    issues = getattr(review, "issues", []) if review is not None else []
    suggestions = [str(s) for s in (getattr(review, "verdict", {}) or {}).get("suggestions") or []]
    return _REVISION_PROMPT.format(
        task=task,
        draft=iteration.code_output,
        issues="\n".join(f"- {i}" for i in issues) or "- (the reviewer gave no specific issue)",
        suggestions=("Optional suggestions:\n" + "\n".join(f"- {s}" for s in suggestions) + "\n\n")
        if suggestions
        else "",
    )


def draft_and_critique(request: ExecutionRequest, **kwargs: Any) -> ExecutionResult:
    """One-shot convenience wrapper around :class:`DraftAndCritiqueStrategy`."""
    return DraftAndCritiqueStrategy(**kwargs).run(request)
