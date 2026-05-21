"""Tests for AsyncReviewLoop using lightweight stub agents."""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from prompture import AsyncReviewLoop


@dataclasses.dataclass
class _StubResult:
    output: str


class _ScriptedAgent:
    """Returns the next entry from `outputs` on each arun() call.

    Records every prompt it was called with in `prompts`.
    """

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    async def arun(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        if not self._outputs:
            raise AssertionError("arun called more times than scripted")
        return _StubResult(self._outputs.pop(0))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_max_iters_must_be_positive():
    with pytest.raises(ValueError, match="max_iters must be >= 1"):
        AsyncReviewLoop(
            coder=_ScriptedAgent([]),
            reviewer=_ScriptedAgent([]),
            max_iters=0,
        )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_exits_on_first_approval():
    coder = _ScriptedAgent(["def foo(): return 1"])
    reviewer = _ScriptedAgent(["Looks good. APPROVED."])
    loop = AsyncReviewLoop(
        coder=coder,
        reviewer=reviewer,
        approve_when=lambda r: "APPROVED" in r.output,
    )
    result = asyncio.run(loop.arun("Write foo()."))
    assert result.approved is True
    assert result.iterations == 1
    assert result.output == "def foo(): return 1"
    assert len(coder.prompts) == 1
    assert "def foo()" in reviewer.prompts[0]


def test_loops_until_approved():
    coder = _ScriptedAgent(["v1", "v2", "v3"])
    reviewer = _ScriptedAgent(["needs work", "still wrong", "APPROVED"])
    loop = AsyncReviewLoop(
        coder=coder,
        reviewer=reviewer,
        max_iters=5,
        approve_when=lambda r: "APPROVED" in r.output,
    )
    result = asyncio.run(loop.arun("Make a thing."))
    assert result.approved is True
    assert result.iterations == 3
    assert result.output == "v3"
    # Each follow-up prompt should include the previous code + review.
    assert "v1" in coder.prompts[1]
    assert "needs work" in coder.prompts[1]
    assert "v2" in coder.prompts[2]
    assert "still wrong" in coder.prompts[2]


def test_stops_at_max_iters_when_never_approved():
    coder = _ScriptedAgent(["v1", "v2"])
    reviewer = _ScriptedAgent(["nope", "still nope"])
    loop = AsyncReviewLoop(
        coder=coder,
        reviewer=reviewer,
        max_iters=2,
        approve_when=lambda r: False,
    )
    result = asyncio.run(loop.arun("Try this."))
    assert result.approved is False
    assert result.iterations == 2
    assert result.output == "v2"
    assert len(result.history) == 2


# ---------------------------------------------------------------------------
# Custom prompts
# ---------------------------------------------------------------------------


def test_custom_review_prompt():
    coder = _ScriptedAgent(["code"])
    reviewer = _ScriptedAgent(["APPROVED"])
    loop = AsyncReviewLoop(
        coder=coder,
        reviewer=reviewer,
        approve_when=lambda r: "APPROVED" in r.output,
        review_prompt=lambda code: f"CUSTOM_REVIEW: {code}",
    )
    asyncio.run(loop.arun("task"))
    assert reviewer.prompts[0] == "CUSTOM_REVIEW: code"


def test_custom_feedback_prompt():
    coder = _ScriptedAgent(["v1", "v2"])
    reviewer = _ScriptedAgent(["bad", "APPROVED"])

    def feedback(original, it):
        return f"RETRY[{original}]<<<{it.review_output}>>>"

    loop = AsyncReviewLoop(
        coder=coder,
        reviewer=reviewer,
        approve_when=lambda r: "APPROVED" in r.output,
        feedback_prompt=feedback,
    )
    asyncio.run(loop.arun("original-task"))
    assert coder.prompts[1] == "RETRY[original-task]<<<bad>>>"


# ---------------------------------------------------------------------------
# kwargs pass-through
# ---------------------------------------------------------------------------


def test_kwargs_passed_through_each_iteration():
    coder = _ScriptedAgent(["v1", "v2"])
    reviewer = _ScriptedAgent(["bad", "APPROVED"])
    loop = AsyncReviewLoop(
        coder=coder,
        reviewer=reviewer,
        approve_when=lambda r: "APPROVED" in r.output,
    )
    asyncio.run(
        loop.arun(
            "task",
            coder_kwargs={"role": "senior"},
            reviewer_kwargs={"strictness": "high"},
        )
    )
    assert all(kw == {"role": "senior"} for kw in coder.kwargs)
    assert all(kw == {"strictness": "high"} for kw in reviewer.kwargs)


# ---------------------------------------------------------------------------
# Default approve predicate
# ---------------------------------------------------------------------------


def test_default_approve_matches_word_approved_case_insensitive():
    coder = _ScriptedAgent(["c"])
    reviewer = _ScriptedAgent(["This is Approved by me."])
    loop = AsyncReviewLoop(coder=coder, reviewer=reviewer)
    result = asyncio.run(loop.arun("task"))
    assert result.approved is True
