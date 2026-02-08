"""Tests for async multi-agent group coordination."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from prompture.agents.types import AgentResult, AgentState
from prompture.groups.async_groups import (
    AsyncLoopGroup,
    AsyncRouterAgent,
    AsyncSequentialGroup,
    ParallelGroup,
)
from prompture.groups.types import ErrorPolicy, GroupCallbacks


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_async_mock_agent(
    name: str, output_text: str = "output", output_key: str | None = None
) -> MagicMock:
    """Create a MagicMock async agent that returns a predetermined AgentResult."""
    agent = MagicMock()
    agent.name = name
    agent.description = f"Mock agent {name}"
    agent.output_key = output_key
    agent.run = AsyncMock(
        return_value=AgentResult(
            output=output_text,
            output_text=output_text,
            messages=[],
            usage={},
            state=AgentState.idle,
            run_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "total_cost": 0.001},
        )
    )
    return agent


# ---------------------------------------------------------------------------
# ParallelGroup tests
# ---------------------------------------------------------------------------


class TestParallelGroup:
    @pytest.mark.asyncio
    async def test_basic_parallel(self):
        a = _make_async_mock_agent("a", "output_a", output_key="key_a")
        b = _make_async_mock_agent("b", "output_b", output_key="key_b")

        group = ParallelGroup([a, b])
        result = await group.run_async("test")

        assert result.success is True
        assert "a" in result.agent_results
        assert "b" in result.agent_results
        assert result.shared_state["key_a"] == "output_a"
        assert result.shared_state["key_b"] == "output_b"

    @pytest.mark.asyncio
    async def test_parallel_state_snapshot(self):
        """Agents read from frozen state, not live state."""
        a = _make_async_mock_agent("a", "a_output", output_key="a_key")
        b = _make_async_mock_agent("b", "b_output", output_key="b_key")

        group = ParallelGroup(
            [(a, "State: {initial}"), (b, "State: {initial}")],
            state={"initial": "frozen_value"},
        )
        result = await group.run_async()

        # Both agents should see the frozen initial state
        a.run.assert_called_once_with("State: frozen_value")
        b.run.assert_called_once_with("State: frozen_value")

    @pytest.mark.asyncio
    async def test_parallel_error_handling(self):
        a = _make_async_mock_agent("a", "ok")
        b = MagicMock()
        b.name = "b"
        b.output_key = None
        b.run = AsyncMock(side_effect=RuntimeError("parallel boom"))

        group = ParallelGroup([a, b], error_policy=ErrorPolicy.fail_fast)
        result = await group.run_async("test")

        assert result.success is False
        assert len(result.errors) == 1
        assert result.errors[0].agent_name == "b"

    @pytest.mark.asyncio
    async def test_parallel_aggregate_usage(self):
        a = _make_async_mock_agent("a", "ok")
        b = _make_async_mock_agent("b", "ok")

        group = ParallelGroup([a, b])
        result = await group.run_async("test")

        assert result.aggregate_usage["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_parallel_callbacks(self):
        starts = []
        completes = []

        callbacks = GroupCallbacks(
            on_agent_start=lambda n, p: starts.append(n),
            on_agent_complete=lambda n, r: completes.append(n),
        )

        a = _make_async_mock_agent("a", "ok")
        b = _make_async_mock_agent("b", "ok")

        group = ParallelGroup([a, b], callbacks=callbacks)
        await group.run_async("test")

        assert "a" in starts
        assert "b" in starts
        assert "a" in completes
        assert "b" in completes

    @pytest.mark.asyncio
    async def test_parallel_timeout(self):
        """Test that timeout raises on slow agents."""

        async def slow_run(prompt):
            await asyncio.sleep(10)
            return AgentResult(
                output="late", output_text="late", messages=[], usage={},
                state=AgentState.idle, run_usage={},
            )

        slow = MagicMock()
        slow.name = "slow"
        slow.output_key = None
        slow.run = slow_run

        group = ParallelGroup([slow], timeout_ms=50)
        result = await group.run_async("test")

        assert result.success is False
        assert len(result.errors) == 1

    def test_sync_wrapper(self):
        a = _make_async_mock_agent("a", "sync_output")
        group = ParallelGroup([a])
        result = group.run("test")
        assert result.success is True
        assert "a" in result.agent_results


# ---------------------------------------------------------------------------
# AsyncSequentialGroup tests
# ---------------------------------------------------------------------------


class TestAsyncSequentialGroup:
    @pytest.mark.asyncio
    async def test_basic_async_sequential(self):
        a = _make_async_mock_agent("a", "result_a", output_key="a")
        b = _make_async_mock_agent("b", "result_b", output_key="b")

        group = AsyncSequentialGroup([a, b])
        result = await group.run("test")

        assert result.success is True
        assert "a" in result.agent_results
        assert "b" in result.agent_results

    @pytest.mark.asyncio
    async def test_async_sequential_state_flow(self):
        a = _make_async_mock_agent("a", "researched", output_key="research")
        b = _make_async_mock_agent("b", "analyzed", output_key="analysis")

        group = AsyncSequentialGroup(
            [(a, "Research {topic}"), (b, "Analyze: {research}")],
            state={"topic": "AI"},
        )
        result = await group.run()

        a.run.assert_called_once_with("Research AI")
        b.run.assert_called_once_with("Analyze: researched")
        assert result.shared_state["research"] == "researched"

    @pytest.mark.asyncio
    async def test_async_sequential_error_fail_fast(self):
        a = _make_async_mock_agent("a", "ok")
        b = MagicMock()
        b.name = "b"
        b.output_key = None
        b.run = AsyncMock(side_effect=RuntimeError("async boom"))
        c = _make_async_mock_agent("c", "should not run")

        group = AsyncSequentialGroup([a, b, c], error_policy=ErrorPolicy.fail_fast)
        result = await group.run("test")

        assert result.success is False
        c.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_sequential_max_turns(self):
        agents = [_make_async_mock_agent(f"a{i}", f"out{i}") for i in range(5)]
        group = AsyncSequentialGroup(agents, max_total_turns=2)
        result = await group.run("test")
        assert len(result.agent_results) == 2


# ---------------------------------------------------------------------------
# AsyncLoopGroup tests
# ---------------------------------------------------------------------------


class TestAsyncLoopGroup:
    @pytest.mark.asyncio
    async def test_basic_async_loop(self):
        a = _make_async_mock_agent("a", "loop_out")

        def exit_cond(state, iteration):
            return iteration >= 3

        group = AsyncLoopGroup([a], exit_condition=exit_cond, max_iterations=10)
        result = await group.run("test")

        assert result.success is True
        assert len(result.timeline) == 3

    @pytest.mark.asyncio
    async def test_async_loop_max_iterations(self):
        a = _make_async_mock_agent("a", "ok")

        def never_exit(state, iteration):
            return False

        group = AsyncLoopGroup([a], exit_condition=never_exit, max_iterations=4)
        result = await group.run("test")

        assert a.run.call_count == 4

    @pytest.mark.asyncio
    async def test_async_loop_error(self):
        a = MagicMock()
        a.name = "failing"
        a.output_key = None
        a.run = AsyncMock(side_effect=RuntimeError("loop error"))

        def never_exit(state, iteration):
            return False

        group = AsyncLoopGroup(
            [a], exit_condition=never_exit, max_iterations=5,
            error_policy=ErrorPolicy.fail_fast,
        )
        result = await group.run("test")

        assert result.success is False
        assert len(result.errors) == 1


# ---------------------------------------------------------------------------
# AsyncRouterAgent tests
# ---------------------------------------------------------------------------


class TestAsyncRouterAgent:
    @pytest.mark.asyncio
    async def test_async_routing(self):
        writer = _make_async_mock_agent("writer", "written content")
        writer.description = "Writes content"
        coder = _make_async_mock_agent("coder", "code output")
        coder.description = "Writes code"

        # Mock async conversation for routing
        mock_conv_class = MagicMock()
        mock_conv_instance = MagicMock()
        mock_conv_instance.ask = AsyncMock(return_value="writer")
        mock_conv_instance.messages = []
        mock_conv_instance.usage = {}
        mock_conv_class.return_value = mock_conv_instance

        import prompture.groups.async_groups as ag
        original = ag.__dict__.get("AsyncConversation")

        # Patch at module level
        import prompture.agents.async_conversation

        original_cls = prompture.agents.async_conversation.AsyncConversation

        try:
            prompture.agents.async_conversation.AsyncConversation = mock_conv_class
            router = AsyncRouterAgent(
                model="test/model",
                agents=[writer, coder],
            )
            result = await router.run("Write me a story")

            writer.run.assert_called_once_with("Write me a story")
            assert result.output_text == "written content"
        finally:
            prompture.agents.async_conversation.AsyncConversation = original_cls

    @pytest.mark.asyncio
    async def test_async_router_fuzzy_match(self):
        agent = _make_async_mock_agent("my_agent", "result")

        router = AsyncRouterAgent(model="test/model", agents=[agent])

        # Test fuzzy matching directly
        matched = router._fuzzy_match("my_agent")
        assert matched is not None

        matched_substring = router._fuzzy_match("I think my_agent should handle this")
        assert matched_substring is not None

        no_match = router._fuzzy_match("completely_unrelated")
        assert no_match is None
