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


# ---------------------------------------------------------------------------
# astream (streaming events)
# ---------------------------------------------------------------------------


def _drain_stream(loop, prompt, **kwargs):
    async def go():
        return [event async for event in loop.astream(prompt, **kwargs)]

    return asyncio.run(go())


def test_astream_emits_lifecycle_for_single_iteration():
    coder = _ScriptedAgent(["code-v1"])
    reviewer = _ScriptedAgent(["nice. APPROVED."])
    loop = AsyncReviewLoop(
        coder=coder,
        reviewer=reviewer,
        approve_when=lambda r: "APPROVED" in r.output,
    )
    events = _drain_stream(loop, "task")
    types = [e.type for e in events]
    assert types == [
        "iteration_started",
        "coder_done",
        "reviewer_done",
        "approved",
        "done",
    ]
    # iteration_started carries the prompt the coder will receive.
    assert events[0].prompt == "task"
    # coder_done carries code_output + the raw coder result.
    assert events[1].code_output == "code-v1"
    assert events[1].code_result.output == "code-v1"
    # reviewer_done carries both outputs + approval flag.
    assert events[2].review_output == "nice. APPROVED."
    assert events[2].approved is True
    # done carries the full ReviewLoopResult.
    assert events[-1].result.approved is True
    assert events[-1].result.iterations == 1


def test_astream_emits_rejected_when_max_iters_exceeded():
    coder = _ScriptedAgent(["v1", "v2"])
    reviewer = _ScriptedAgent(["nope", "still nope"])
    loop = AsyncReviewLoop(
        coder=coder,
        reviewer=reviewer,
        max_iters=2,
        approve_when=lambda r: False,
    )
    events = _drain_stream(loop, "task")
    terminal_types = [e.type for e in events if e.type in ("approved", "rejected", "done")]
    assert terminal_types == ["rejected", "done"]
    assert events[-1].result.approved is False
    assert events[-1].result.iterations == 2


def test_astream_emits_two_iterations_then_approval():
    coder = _ScriptedAgent(["v1", "v2"])
    reviewer = _ScriptedAgent(["bad", "APPROVED"])
    loop = AsyncReviewLoop(
        coder=coder,
        reviewer=reviewer,
        approve_when=lambda r: "APPROVED" in r.output,
    )
    events = _drain_stream(loop, "task")
    starts = [e for e in events if e.type == "iteration_started"]
    assert [e.index for e in starts] == [0, 1]
    # First iteration_started gets the original prompt; second gets the
    # feedback-augmented prompt.
    assert starts[0].prompt == "task"
    assert "v1" in starts[1].prompt
    assert "bad" in starts[1].prompt


def test_arun_returns_same_result_as_streaming_done_event():
    coder = _ScriptedAgent(["v1", "v2"])
    reviewer = _ScriptedAgent(["bad", "APPROVED"])
    loop = AsyncReviewLoop(
        coder=coder,
        reviewer=reviewer,
        approve_when=lambda r: "APPROVED" in r.output,
    )
    via_arun = asyncio.run(loop.arun("task"))
    assert via_arun.iterations == 2
    assert via_arun.approved is True
    assert via_arun.output == "v2"
