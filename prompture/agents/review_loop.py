"""Generic developer → reviewer feedback loop.

Wraps any "do work, then critique it, then revise" pattern as a single
async call.  The loop runs the coder, hands its output to the reviewer,
checks an ``approve_when`` predicate against the review, and either
returns or re-runs the coder with the prior review folded back into the
prompt as feedback.

Both coder and reviewer must implement a minimal protocol::

    async def arun(prompt: str, **kwargs) -> result_with_output

Any object with an ``arun`` returning something whose ``output``
attribute is a string qualifies — :class:`Assistant`,
:class:`AsyncAgent` (via a thin wrapper), or a fake stub in tests.

Quick start::

    from prompture import Assistant, AsyncReviewLoop, Persona

    coder = Assistant(
        name="coder",
        persona=Persona(name="dev", system_prompt="Write Python functions."),
        model="openai/gpt-4o-mini",
    )
    reviewer = Assistant(
        name="reviewer",
        persona=Persona(
            name="rev",
            system_prompt=(
                "Critique the function. End with SCORE: <0-10> on its own line."
            ),
        ),
        model="openai/gpt-4o-mini",
    )

    loop = AsyncReviewLoop(
        coder=coder,
        reviewer=reviewer,
        max_iters=3,
        approve_when=lambda review: "SCORE: 9" in review.output
                                  or "SCORE: 10" in review.output,
    )
    result = await loop.arun("Write a function that reverses a string.")
    print(result.output)
    print(f"Iterations: {result.iterations}, approved: {result.approved}")
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("prompture.review_loop")


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class _HasOutput(Protocol):
    """Any object exposing an ``.output`` string attribute."""

    output: str


@runtime_checkable
class _Runnable(Protocol):
    """Anything with an awaitable ``arun(prompt, **kwargs)`` method.

    The returned object must expose ``.output`` (string).  :class:`Assistant`
    is the canonical implementation, but tests routinely pass small stubs.
    """

    async def arun(self, prompt: str, **kwargs: Any) -> _HasOutput: ...


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ReviewLoopIteration:
    """Per-iteration record from an :class:`AsyncReviewLoop` run."""

    index: int  # 0-based
    code_output: str
    review_output: str
    approved: bool
    code_result: Any = None  # raw coder result for backend-specific fields
    review_result: Any = None


@dataclasses.dataclass(frozen=True)
class ReviewLoopResult:
    """Final result returned by :meth:`AsyncReviewLoop.arun`."""

    output: str  # the final code_output (last iteration's)
    approved: bool  # True iff the last iteration's review approved
    iterations: int  # number of iterations actually run
    history: tuple[ReviewLoopIteration, ...] = ()

    @property
    def last_iteration(self) -> ReviewLoopIteration | None:
        return self.history[-1] if self.history else None


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def _default_approve(review: _HasOutput) -> bool:
    """Default predicate: anything containing the literal word 'approved'."""
    return "approved" in (review.output or "").lower()


def _default_feedback_prompt(
    original_prompt: str,
    iteration: ReviewLoopIteration,
) -> str:
    """Default prompt rewrite that folds review feedback back into the task."""
    return (
        f"{original_prompt}\n\n"
        f"---\n"
        f"Your previous attempt:\n{iteration.code_output}\n\n"
        f"Reviewer feedback:\n{iteration.review_output}\n\n"
        f"Revise your work to address the feedback."
    )


@dataclasses.dataclass(frozen=True)
class AsyncReviewLoop:
    """A developer → reviewer feedback loop.

    Runs ``coder`` on the original prompt, feeds the output to
    ``reviewer``, and re-runs ``coder`` with the review folded back in
    until ``approve_when(review)`` returns True or ``max_iters`` is hit.

    Args:
        coder: Anything with an awaitable ``arun(prompt, **kwargs)`` —
            typically an :class:`Assistant`.
        reviewer: Same shape as ``coder``.  Receives the coder's output
            (via ``review_prompt``) and must produce something callable
            ``approve_when`` can inspect.
        max_iters: Cap on coder runs (one review per coder run).
            Defaults to 3.
        approve_when: Predicate applied to each reviewer result.  When
            True, the loop exits and the latest code is returned.
            Defaults to a substring check for the word "approved".
        review_prompt: Builds the prompt sent to the reviewer.  Defaults
            to wrapping the coder's output in a brief "Review this:" frame.
        feedback_prompt: Builds the next coder prompt from the original
            request + the latest iteration.  Defaults to appending the
            review verbatim as "Reviewer feedback".
    """

    coder: _Runnable
    reviewer: _Runnable
    max_iters: int = 3
    approve_when: Callable[[Any], bool] = _default_approve
    review_prompt: Callable[[str], str] = dataclasses.field(
        default=lambda code: f"Review this work and give feedback:\n\n{code}"
    )
    feedback_prompt: Callable[[str, ReviewLoopIteration], str] = (
        _default_feedback_prompt
    )

    def __post_init__(self) -> None:
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be >= 1 (got {self.max_iters})")

    async def arun(
        self,
        prompt: str,
        *,
        coder_kwargs: dict[str, Any] | None = None,
        reviewer_kwargs: dict[str, Any] | None = None,
    ) -> ReviewLoopResult:
        """Run the loop to completion.

        Args:
            prompt: The initial task for the coder.
            coder_kwargs: Extra kwargs forwarded to every ``coder.arun``
                call (e.g. template variables for an :class:`Assistant`).
            reviewer_kwargs: Extra kwargs forwarded to every
                ``reviewer.arun`` call.
        """
        coder_kwargs = coder_kwargs or {}
        reviewer_kwargs = reviewer_kwargs or {}
        history: list[ReviewLoopIteration] = []
        current_prompt = prompt

        for i in range(self.max_iters):
            code_result = await self.coder.arun(current_prompt, **coder_kwargs)
            code_text = getattr(code_result, "output", "") or ""

            review_result = await self.reviewer.arun(
                self.review_prompt(code_text), **reviewer_kwargs
            )
            review_text = getattr(review_result, "output", "") or ""

            approved = bool(self.approve_when(review_result))
            iteration = ReviewLoopIteration(
                index=i,
                code_output=code_text,
                review_output=review_text,
                approved=approved,
                code_result=code_result,
                review_result=review_result,
            )
            history.append(iteration)
            logger.info(
                "review_loop iter=%d approved=%s code_len=%d review_len=%d",
                i,
                approved,
                len(code_text),
                len(review_text),
            )
            if approved:
                break
            current_prompt = self.feedback_prompt(prompt, iteration)

        last = history[-1]
        return ReviewLoopResult(
            output=last.code_output,
            approved=last.approved,
            iterations=len(history),
            history=tuple(history),
        )


__all__ = [
    "AsyncReviewLoop",
    "ReviewLoopIteration",
    "ReviewLoopResult",
]
