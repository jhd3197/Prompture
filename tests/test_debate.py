"""Tests for DebateGroup and AsyncDebateGroup."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from prompture.agents.types import AgentResult, AgentState
from prompture.groups.debate import (
    AsyncDebateGroup,
    DebateConfig,
    DebateEntry,
    DebateGroup,
    DebateResult,
)
from prompture.groups.types import GroupCallbacks


def _make_mock_agent(name: str, output_text: str = "output") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.description = f"Mock agent {name}"
    agent.output_key = None
    agent.run.return_value = AgentResult(
        output=output_text,
        output_text=output_text,
        messages=[],
        usage={},
        state=AgentState.idle,
        run_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "total_cost": 0.001},
    )
    return agent


def _make_async_mock_agent(name: str, output_text: str = "output") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.description = f"Mock agent {name}"
    agent.output_key = None
    result = AgentResult(
        output=output_text,
        output_text=output_text,
        messages=[],
        usage={},
        state=AgentState.idle,
        run_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "total_cost": 0.001},
    )
    agent.run = AsyncMock(return_value=result)
    return agent


# ---------------------------------------------------------------------------
# Basic debate tests
# ---------------------------------------------------------------------------


class TestDebateGroup:
    def test_two_agents_one_round(self):
        pro = _make_mock_agent("pro_bot", "Arguments for the topic")
        con = _make_mock_agent("con_bot", "Arguments against the topic")

        group = DebateGroup([pro, con], DebateConfig(rounds=1))
        result = group.run("Should AI be regulated?")

        assert isinstance(result, DebateResult)
        assert result.success
        assert result.rounds_completed == 1
        assert len(result.transcript) == 2
        assert result.transcript[0].agent_name == "pro_bot"
        assert result.transcript[1].agent_name == "con_bot"
        assert result.judge_verdict is None

    def test_three_agents_two_rounds(self):
        a = _make_mock_agent("alice", "Alice's argument")
        b = _make_mock_agent("bob", "Bob's argument")
        c = _make_mock_agent("charlie", "Charlie's argument")

        group = DebateGroup([a, b, c], DebateConfig(rounds=2))
        result = group.run("Topic")

        assert result.rounds_completed == 2
        assert len(result.transcript) == 6  # 3 agents * 2 rounds
        # Verify round numbering
        assert all(e.round == 0 for e in result.transcript[:3])
        assert all(e.round == 1 for e in result.transcript[3:])

    def test_with_positions(self):
        pro = _make_mock_agent("pro_bot", "Pro argument")
        con = _make_mock_agent("con_bot", "Con argument")

        config = DebateConfig(
            rounds=1,
            positions={"pro_bot": "FOR regulation", "con_bot": "AGAINST regulation"},
        )
        group = DebateGroup([pro, con], config)
        result = group.run("AI regulation")

        assert result.transcript[0].position == "FOR regulation"
        assert result.transcript[1].position == "AGAINST regulation"

        # Verify position was included in prompt
        call_args = pro.run.call_args[0][0]
        assert "FOR regulation" in call_args

    def test_with_judge(self):
        pro = _make_mock_agent("pro", "Pro argument")
        con = _make_mock_agent("con", "Con argument")
        judge = _make_mock_agent("judge", "The debate was balanced. My verdict: pro wins.")

        config = DebateConfig(rounds=1, judge=judge)
        group = DebateGroup([pro, con], config)
        result = group.run("Topic")

        assert result.judge_verdict == "The debate was balanced. My verdict: pro wins."
        assert "judge" in result.agent_results

    def test_without_judge(self):
        pro = _make_mock_agent("pro", "Argument")
        con = _make_mock_agent("con", "Counter")

        group = DebateGroup([pro, con], DebateConfig(rounds=1))
        result = group.run("Topic")

        assert result.judge_verdict is None

    def test_agent_results_keyed_by_round(self):
        a = _make_mock_agent("alice", "argument")
        b = _make_mock_agent("bob", "rebuttal")

        group = DebateGroup([a, b], DebateConfig(rounds=2))
        result = group.run("Topic")

        assert "alice_round0" in result.agent_results
        assert "bob_round0" in result.agent_results
        assert "alice_round1" in result.agent_results
        assert "bob_round1" in result.agent_results

    def test_aggregate_usage(self):
        a = _make_mock_agent("a", "arg")
        b = _make_mock_agent("b", "arg")

        group = DebateGroup([a, b], DebateConfig(rounds=1))
        result = group.run("Topic")

        assert result.aggregate_usage["total_tokens"] == 30  # 15 * 2
        assert result.aggregate_usage["total_cost"] == pytest.approx(0.002)

    def test_timeline_records(self):
        a = _make_mock_agent("a", "arg")
        group = DebateGroup([a], DebateConfig(rounds=2))
        result = group.run("Topic")

        assert len(result.timeline) == 2
        assert all(s.step_type == "debate_argument" for s in result.timeline)


# ---------------------------------------------------------------------------
# Debate prompt construction
# ---------------------------------------------------------------------------


class TestDebatePrompts:
    def test_first_round_no_transcript(self):
        a = _make_mock_agent("a", "first argument")
        group = DebateGroup([a], DebateConfig(rounds=1))
        group.run("Climate change")

        prompt = a.run.call_args[0][0]
        assert "Climate change" in prompt
        assert "Previous arguments" not in prompt

    def test_second_agent_sees_first_response(self):
        a = _make_mock_agent("a", "A's opening argument")
        b = _make_mock_agent("b", "B's rebuttal")

        group = DebateGroup([a, b], DebateConfig(rounds=1))
        group.run("Topic")

        b_prompt = b.run.call_args[0][0]
        assert "A's opening argument" in b_prompt

    def test_second_round_sees_all_previous(self):
        a = _make_mock_agent("a", "A round 1")
        b = _make_mock_agent("b", "B round 1")

        # For second round, a should see both round 1 responses
        group = DebateGroup([a, b], DebateConfig(rounds=2))
        group.run("Topic")

        # a is called twice — second call should have transcript
        second_call_prompt = a.run.call_args_list[1][0][0]
        assert "A round 1" in second_call_prompt
        assert "B round 1" in second_call_prompt


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class TestDebateCallbacks:
    def test_round_callbacks(self):
        round_starts: list[int] = []
        round_completes: list[int] = []

        callbacks = GroupCallbacks(
            on_round_start=lambda r: round_starts.append(r),
            on_round_complete=lambda r: round_completes.append(r),
        )

        a = _make_mock_agent("a", "arg")
        group = DebateGroup([a], DebateConfig(rounds=3), callbacks=callbacks)
        group.run("Topic")

        assert round_starts == [0, 1, 2]
        assert round_completes == [0, 1, 2]

    def test_agent_callbacks(self):
        starts: list[str] = []
        completes: list[str] = []

        callbacks = GroupCallbacks(
            on_agent_start=lambda n, p: starts.append(n),
            on_agent_complete=lambda n, r: completes.append(n),
        )

        a = _make_mock_agent("a", "arg")
        b = _make_mock_agent("b", "arg")
        group = DebateGroup([a, b], DebateConfig(rounds=1), callbacks=callbacks)
        group.run("Topic")

        assert starts == ["a", "b"]
        assert completes == ["a", "b"]

    def test_judge_callback(self):
        completes: list[str] = []

        callbacks = GroupCallbacks(
            on_agent_complete=lambda n, r: completes.append(n),
        )

        a = _make_mock_agent("a", "arg")
        judge = _make_mock_agent("judge", "verdict")
        group = DebateGroup([a], DebateConfig(rounds=1, judge=judge), callbacks=callbacks)
        group.run("Topic")

        assert "judge" in completes


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestDebateErrors:
    def test_agent_error_recorded(self):
        a = MagicMock()
        a.name = "failing"
        a.output_key = None
        a.run.side_effect = RuntimeError("debate error")

        group = DebateGroup([a], DebateConfig(rounds=1))
        result = group.run("Topic")

        assert not result.success
        assert len(result.errors) == 1
        assert result.errors[0].agent_name == "failing"

    def test_judge_error_recorded(self):
        a = _make_mock_agent("a", "argument")
        judge = MagicMock()
        judge.name = "judge"
        judge.output_key = None
        judge.run.side_effect = RuntimeError("judge error")

        group = DebateGroup([a], DebateConfig(rounds=1, judge=judge))
        result = group.run("Topic")

        assert not result.success
        assert any(e.agent_name == "judge" for e in result.errors)
        assert result.judge_verdict is None


# ---------------------------------------------------------------------------
# Stop behavior
# ---------------------------------------------------------------------------


class TestDebateStop:
    def test_stop_mid_debate(self):
        a = _make_mock_agent("a", "argument")
        b = _make_mock_agent("b", "should not run in round 2")

        group = DebateGroup([a, b], DebateConfig(rounds=3))

        original_run = a.run
        call_count = 0

        def run_and_maybe_stop(prompt):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                group.stop()
            return original_run(prompt)

        a.run = run_and_maybe_stop
        result = group.run("Topic")

        # Should have stopped before completing all 3 rounds
        assert result.rounds_completed < 3


# ---------------------------------------------------------------------------
# DebateConfig
# ---------------------------------------------------------------------------


class TestDebateConfig:
    def test_defaults(self):
        config = DebateConfig()
        assert config.rounds == 2
        assert config.positions is None
        assert config.judge is None
        assert config.show_position_in_prompt is True

    def test_custom_judge_template(self):
        judge = _make_mock_agent("judge", "verdict")
        a = _make_mock_agent("a", "argument")

        config = DebateConfig(
            rounds=1,
            judge=judge,
            judge_prompt_template="Summarize debate on {topic}:\n{transcript}\nVerdict:",
        )
        group = DebateGroup([a], config)
        group.run("AI ethics")

        judge_prompt = judge.run.call_args[0][0]
        assert "AI ethics" in judge_prompt
        assert "argument" in judge_prompt


# ---------------------------------------------------------------------------
# DebateEntry / DebateResult dataclasses
# ---------------------------------------------------------------------------


class TestDebateDataclasses:
    def test_debate_entry(self):
        entry = DebateEntry(
            round=0,
            agent_name="bot_a",
            position="FOR",
            content="My argument",
            timestamp=1000.0,
        )
        assert entry.round == 0
        assert entry.agent_name == "bot_a"
        assert entry.position == "FOR"

    def test_debate_result_extends_group_result(self):
        result = DebateResult(
            transcript=[],
            rounds_completed=2,
            judge_verdict="Pro wins",
        )
        assert result.rounds_completed == 2
        assert result.judge_verdict == "Pro wins"
        assert result.success is True  # inherited default


# ---------------------------------------------------------------------------
# AsyncDebateGroup
# ---------------------------------------------------------------------------


class TestAsyncDebateGroup:
    def test_basic_async_debate(self):
        a = _make_async_mock_agent("a", "async argument")
        b = _make_async_mock_agent("b", "async rebuttal")

        group = AsyncDebateGroup([a, b], DebateConfig(rounds=1))
        result = asyncio.run(group.run("Async topic"))

        assert isinstance(result, DebateResult)
        assert result.success
        assert result.rounds_completed == 1
        assert len(result.transcript) == 2

    def test_async_with_judge(self):
        a = _make_async_mock_agent("a", "argument")
        judge = _make_async_mock_agent("judge", "verdict")

        group = AsyncDebateGroup([a], DebateConfig(rounds=1, judge=judge))
        result = asyncio.run(group.run("Topic"))

        assert result.judge_verdict == "verdict"

    def test_async_with_positions(self):
        pro = _make_async_mock_agent("pro", "for it")
        con = _make_async_mock_agent("con", "against it")

        config = DebateConfig(
            rounds=1,
            positions={"pro": "FOR", "con": "AGAINST"},
        )
        group = AsyncDebateGroup([pro, con], config)
        result = asyncio.run(group.run("Topic"))

        assert result.transcript[0].position == "FOR"
        assert result.transcript[1].position == "AGAINST"
