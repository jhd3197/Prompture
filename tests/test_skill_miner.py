"""Tests for the skill miner (prompture.agents.skill_miner).

The LLM judgment/drafting step is injected (``extract_fn`` / ``aextract_fn``) so these
are fast unit tests with no live model calls. The heuristic gate, recurrence threshold,
in-memory registration, dedupe, and SKILL.md round-trip are all exercised offline.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from prompture.agents.skill_miner import SkillMiner, SkillProposal, default_signature
from prompture.agents.skills import (
    SkillInfo,
    get_skill,
    get_skill_registry_snapshot,
    load_skill,
    register_skill,
    unregister_skill,
)
from prompture.agents.types import AgentResult, AgentState, AgentStep, StepType

# ------------------------------------------------------------------
# Fixtures & helpers
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_skill_registry():
    """Snapshot, clear, and restore the global skill registry around each test."""
    snapshot = get_skill_registry_snapshot()
    for name in list(snapshot):
        unregister_skill(name)
    try:
        yield
    finally:
        for name in list(get_skill_registry_snapshot()):
            unregister_skill(name)
        for skill in snapshot.values():
            register_skill(skill)


def make_result(
    tools,
    *,
    task="do the thing",
    output="done",
    state=AgentState.idle,
    typed_steps=True,
):
    """Build an AgentResult exhibiting the given ordered tool sequence."""
    all_tool_calls = [{"name": t, "arguments": {"i": i}, "id": str(i)} for i, t in enumerate(tools)]
    steps: list[AgentStep] = []
    if typed_steps:
        for i, t in enumerate(tools):
            steps.append(AgentStep(StepType.tool_call, timestamp=0.0, tool_name=t, tool_args={"i": i}))
            steps.append(AgentStep(StepType.tool_result, timestamp=0.0, tool_name=t, tool_result=f"{t}-result"))
    messages = [{"role": "user", "content": task}]
    return AgentResult(
        output=output,
        output_text=output,
        messages=messages,
        usage={},
        steps=steps,
        all_tool_calls=all_tool_calls,
        state=state,
    )


class FakeExtract:
    """Stand-in for extract_with_model that returns a fixed draft and counts calls."""

    def __init__(self, draft):
        self.draft = draft
        self.calls = 0

    def __call__(self, model_cls, **kwargs):
        self.calls += 1
        return {"model": self.draft, "usage": {}}


class AsyncFakeExtract(FakeExtract):
    async def __call__(self, model_cls, **kwargs):  # type: ignore[override]
        self.calls += 1
        return {"model": self.draft, "usage": {}}


def accept_draft(name="web-research", confidence=0.9):
    return SimpleNamespace(
        generalizable=True,
        name=name,
        description="Research a topic and summarize the sources.",
        when_to_use="Use when asked to research and summarize.",
        instructions="1. Search.\n2. Read.\n3. Summarize.",
        confidence=confidence,
    )


def reject_draft():
    return SimpleNamespace(
        generalizable=False,
        name="nope",
        description="",
        when_to_use="",
        instructions="",
        confidence=0.1,
    )


# ------------------------------------------------------------------
# Signature
# ------------------------------------------------------------------


def test_default_signature_from_tool_calls():
    result = make_result(["search", "read", "edit"])
    assert default_signature(result) == ("search", "read", "edit")


def test_default_signature_falls_back_to_steps():
    result = make_result(["search", "read"])
    result.all_tool_calls.clear()  # force the steps fallback path
    assert default_signature(result) == ("search", "read")


# ------------------------------------------------------------------
# Recurrence gate
# ------------------------------------------------------------------


def test_no_proposal_below_threshold():
    fake = FakeExtract(accept_draft())
    miner = SkillMiner(model="x/y", extract_fn=fake)
    assert miner.observe(make_result(["search", "read", "edit"])) is None
    assert fake.calls == 0  # LLM not consulted before the threshold
    assert miner.recurring() == []


def test_recurrence_triggers_and_registers():
    fake = FakeExtract(accept_draft(name="web-research"))
    seen = []
    miner = SkillMiner(model="x/y", extract_fn=fake, on_proposal=seen.append)

    assert miner.observe(make_result(["search", "read", "edit"])) is None
    proposal = miner.observe(make_result(["search", "read", "edit"]))

    assert isinstance(proposal, SkillProposal)
    assert fake.calls == 1
    assert proposal.name == "web-research"
    assert proposal.tool_sequence == ("search", "read", "edit")
    assert proposal.occurrences == 2
    assert seen == [proposal]
    # Confirmed proposal is registered in-memory for the session.
    registered = get_skill("web-research")
    assert isinstance(registered, SkillInfo)
    assert registered.metadata["tool_sequence"] == ["search", "read", "edit"]
    assert registered.metadata["generated_by"] == "prompture.skill_miner"


def test_does_not_write_to_disk_automatically(tmp_path):
    fake = FakeExtract(accept_draft())
    miner = SkillMiner(model="x/y", extract_fn=fake, skills_dir=tmp_path)
    miner.observe(make_result(["search", "read", "edit"]))
    miner.observe(make_result(["search", "read", "edit"]))
    # propose-and-confirm: nothing on disk until save() is called
    assert list(tmp_path.glob("**/SKILL.md")) == []


# ------------------------------------------------------------------
# LLM judgment
# ------------------------------------------------------------------


def test_rejected_draft_yields_no_proposal_and_caches_verdict():
    fake = FakeExtract(reject_draft())
    miner = SkillMiner(model="x/y", extract_fn=fake)
    miner.observe(make_result(["a", "b"]))
    assert miner.observe(make_result(["a", "b"])) is None  # judged, rejected
    assert miner.observe(make_result(["a", "b"])) is None  # verdict cached
    assert fake.calls == 1  # not re-judged after a verdict exists


def test_min_confidence_rejects_low_confidence():
    fake = FakeExtract(accept_draft(confidence=0.3))
    miner = SkillMiner(model="x/y", extract_fn=fake, min_confidence=0.8)
    miner.observe(make_result(["a", "b"]))
    assert miner.observe(make_result(["a", "b"])) is None


def test_no_double_propose():
    fake = FakeExtract(accept_draft())
    miner = SkillMiner(model="x/y", extract_fn=fake)
    miner.observe(make_result(["a", "b"]))
    first = miner.observe(make_result(["a", "b"]))
    third = miner.observe(make_result(["a", "b"]))
    assert isinstance(first, SkillProposal)
    assert third is None
    assert len(miner.proposals) == 1
    assert fake.calls == 1


# ------------------------------------------------------------------
# Heuristic gate
# ------------------------------------------------------------------


def test_single_distinct_tool_never_qualifies():
    fake = FakeExtract(accept_draft())
    miner = SkillMiner(model="x/y", extract_fn=fake)  # min_tools=2 default
    miner.observe(make_result(["a", "a"]))
    assert miner.observe(make_result(["a", "a"])) is None
    assert fake.calls == 0


def test_errored_run_is_ignored():
    fake = FakeExtract(accept_draft())
    miner = SkillMiner(model="x/y", extract_fn=fake)
    miner.observe(make_result(["a", "b"], state=AgentState.errored))
    miner.observe(make_result(["a", "b"], state=AgentState.errored))
    assert fake.calls == 0
    assert miner.recurring() == []


def test_run_without_output_is_ignored():
    fake = FakeExtract(accept_draft())
    miner = SkillMiner(model="x/y", extract_fn=fake)
    miner.observe(make_result(["a", "b"], output=""))
    miner.observe(make_result(["a", "b"], output=""))
    assert fake.calls == 0


def test_already_registered_skill_is_not_remined():
    register_skill(
        SkillInfo(
            name="manual-skill",
            description="hand-written",
            instructions="do x",
            metadata={"tool_sequence": ["a", "b"]},
        )
    )
    fake = FakeExtract(accept_draft())
    miner = SkillMiner(model="x/y", extract_fn=fake)
    miner.observe(make_result(["a", "b"]))
    assert miner.observe(make_result(["a", "b"])) is None
    assert fake.calls == 0  # dedup gate prevents the LLM call entirely


# ------------------------------------------------------------------
# judge=False (offline / deterministic)
# ------------------------------------------------------------------


def test_judge_false_drafts_without_llm():
    miner = SkillMiner(judge=False)  # no model, no extract_fn needed
    miner.observe(make_result(["search", "read", "edit"]))
    proposal = miner.observe(make_result(["search", "read", "edit"]))
    assert isinstance(proposal, SkillProposal)
    assert proposal.tool_sequence == ("search", "read", "edit")
    assert "search" in proposal.name


# ------------------------------------------------------------------
# Batch mining
# ------------------------------------------------------------------


def test_mine_batch_returns_confirmed_proposals():
    fake = FakeExtract(accept_draft())
    miner = SkillMiner(model="x/y", extract_fn=fake)
    runs = [make_result(["a", "b", "c"]) for _ in range(3)]
    proposals = miner.mine(runs)
    assert len(proposals) == 1
    assert proposals[0].occurrences == 2


# ------------------------------------------------------------------
# Persistence (the confirm step)
# ------------------------------------------------------------------


def test_save_roundtrips_through_skill_md(tmp_path):
    fake = FakeExtract(accept_draft(name="web-research"))
    miner = SkillMiner(model="x/y", extract_fn=fake)
    miner.observe(make_result(["search", "read", "edit"]))
    proposal = miner.observe(make_result(["search", "read", "edit"]))

    path = miner.save(proposal, skills_dir=tmp_path)
    assert path == tmp_path / "web-research" / "SKILL.md"
    assert path.exists()

    loaded = load_skill(path)
    assert loaded.name == "web-research"
    assert loaded.description == proposal.description
    assert loaded.metadata["tool_sequence"] == ["search", "read", "edit"]
    assert loaded.instructions.strip() == proposal.instructions.strip()


def test_save_refuses_overwrite_by_default(tmp_path):
    fake = FakeExtract(accept_draft(name="dup"))
    miner = SkillMiner(model="x/y", extract_fn=fake)
    miner.observe(make_result(["a", "b"]))
    proposal = miner.observe(make_result(["a", "b"]))

    miner.save(proposal, skills_dir=tmp_path)
    with pytest.raises(FileExistsError):
        miner.save(proposal, skills_dir=tmp_path)
    # overwrite=True succeeds
    miner.save(proposal, skills_dir=tmp_path, overwrite=True)


# ------------------------------------------------------------------
# Async
# ------------------------------------------------------------------


async def test_aobserve_triggers_with_async_extract():
    fake = AsyncFakeExtract(accept_draft(name="async-skill"))
    miner = SkillMiner(model="x/y", aextract_fn=fake)
    assert await miner.aobserve(make_result(["a", "b"])) is None
    proposal = await miner.aobserve(make_result(["a", "b"]))
    assert isinstance(proposal, SkillProposal)
    assert proposal.name == "async-skill"
    assert fake.calls == 1


# ------------------------------------------------------------------
# Top-level export
# ------------------------------------------------------------------


def test_exported_at_top_level():
    import prompture

    assert prompture.SkillMiner is SkillMiner
    assert prompture.SkillProposal is SkillProposal
