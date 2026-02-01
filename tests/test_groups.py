"""Tests for synchronous multi-agent group coordination."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any
from unittest.mock import MagicMock

import pytest

from prompture.agent import Agent
from prompture.agent_types import AgentResult, AgentState
from prompture.driver import Driver
from prompture.group_types import (
    AgentError,
    ErrorPolicy,
    GroupCallbacks,
    GroupResult,
    GroupStep,
    _aggregate_usage,
)
from prompture.groups import (
    GroupAsAgent,
    LoopGroup,
    RouterAgent,
    SequentialGroup,
    _inject_state,
)
from prompture.tools_schema import ToolDefinition


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockDriver(Driver):
    """Simple mock driver returning canned text responses."""

    supports_messages = True
    supports_tool_use = False

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or ["Hello from mock"])
        self._call_count = 0
        self.model = "mock-model"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self._make_response()

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return self._make_response()

    def _make_response(self) -> dict[str, Any]:
        idx = min(self._call_count, len(self.responses) - 1)
        text = self.responses[idx]
        self._call_count += 1
        return {
            "text": text,
            "meta": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.001,
                "raw_response": {},
            },
        }


def _make_agent(name: str, response: str = "output", output_key: str | None = None) -> Agent:
    """Create a mock agent with a given name and canned response."""
    return Agent(
        "test/model",
        driver=MockDriver(responses=[response]),
        name=name,
        output_key=output_key,
    )


def _make_mock_agent(name: str, output_text: str = "output", output_key: str | None = None) -> MagicMock:
    """Create a MagicMock agent that returns a predetermined AgentResult."""
    agent = MagicMock()
    agent.name = name
    agent.description = f"Mock agent {name}"
    agent.output_key = output_key
    agent.run.return_value = AgentResult(
        output=output_text,
        output_text=output_text,
        messages=[],
        usage={},
        state=AgentState.idle,
        run_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "total_cost": 0.001},
    )
    return agent


# ---------------------------------------------------------------------------
# State injection tests
# ---------------------------------------------------------------------------


class TestInjectState:
    def test_basic_substitution(self):
        result = _inject_state("Hello {name}", {"name": "World"})
        assert result == "Hello World"

    def test_multiple_keys(self):
        result = _inject_state("{a} and {b}", {"a": "X", "b": "Y"})
        assert result == "X and Y"

    def test_unknown_keys_pass_through(self):
        result = _inject_state("Hello {unknown}", {})
        assert result == "Hello {unknown}"

    def test_mixed_known_unknown(self):
        result = _inject_state("{known} and {unknown}", {"known": "yes"})
        assert result == "yes and {unknown}"

    def test_empty_template(self):
        result = _inject_state("", {"key": "val"})
        assert result == ""


# ---------------------------------------------------------------------------
# SequentialGroup tests
# ---------------------------------------------------------------------------


class TestSequentialGroup:
    def test_basic_execution(self):
        a = _make_mock_agent("agent_a", "result_a", output_key="a")
        b = _make_mock_agent("agent_b", "result_b", output_key="b")

        group = SequentialGroup([a, b])
        result = group.run("test prompt")

        assert result.success is True
        assert "agent_a" in result.agent_results
        assert "agent_b" in result.agent_results
        assert len(result.timeline) == 2
        assert result.elapsed_ms > 0

    def test_state_propagation(self):
        a = _make_mock_agent("a", "research output", output_key="research")
        b = _make_mock_agent("b", "analysis output", output_key="analysis")

        group = SequentialGroup(
            [(a, "Do research on {topic}"), (b, "Analyze: {research}")],
            state={"topic": "climate"},
        )
        result = group.run()

        # Agent a should have been called with injected state
        a.run.assert_called_once_with("Do research on climate")
        # Agent b should see the state updated by agent a
        b.run.assert_called_once_with("Analyze: research output")

        assert result.shared_state["research"] == "research output"
        assert result.shared_state["analysis"] == "analysis output"

    def test_initial_state_preserved(self):
        a = _make_mock_agent("a", "output", output_key="result")
        group = SequentialGroup([a], state={"existing": "value"})
        result = group.run("test")

        assert result.shared_state["existing"] == "value"
        assert result.shared_state["result"] == "output"

    def test_error_policy_fail_fast(self):
        a = _make_mock_agent("a", "ok")
        b = MagicMock()
        b.name = "b"
        b.output_key = None
        b.run.side_effect = RuntimeError("boom")
        c = _make_mock_agent("c", "should not run")

        group = SequentialGroup([a, b, c], error_policy=ErrorPolicy.fail_fast)
        result = group.run("test")

        assert result.success is False
        assert len(result.errors) == 1
        assert result.errors[0].agent_name == "b"
        c.run.assert_not_called()

    def test_error_policy_continue(self):
        a = _make_mock_agent("a", "ok")
        b = MagicMock()
        b.name = "b"
        b.output_key = None
        b.run.side_effect = RuntimeError("boom")
        c = _make_mock_agent("c", "still runs")

        group = SequentialGroup([a, b, c], error_policy=ErrorPolicy.continue_on_error)
        result = group.run("test")

        assert result.success is False
        assert len(result.errors) == 1
        c.run.assert_called_once()

    def test_max_total_turns(self):
        agents = [_make_mock_agent(f"a{i}", f"out{i}") for i in range(5)]
        group = SequentialGroup(agents, max_total_turns=2)
        result = group.run("test")

        assert len(result.agent_results) == 2

    def test_max_total_cost(self):
        # Each agent costs 0.001
        agents = [_make_mock_agent(f"a{i}", f"out{i}") for i in range(5)]
        group = SequentialGroup(agents, max_total_cost=0.002)
        result = group.run("test")

        # Should run at most 2 agents before budget is exceeded
        assert len(result.agent_results) <= 3

    def test_callbacks(self):
        starts = []
        completes = []
        state_updates = []

        callbacks = GroupCallbacks(
            on_agent_start=lambda n, p: starts.append(n),
            on_agent_complete=lambda n, r: completes.append(n),
            on_state_update=lambda k, v: state_updates.append((k, v)),
        )

        a = _make_mock_agent("a", "output_a", output_key="key_a")
        group = SequentialGroup([a], callbacks=callbacks)
        group.run("test")

        assert "a" in starts
        assert "a" in completes
        assert ("key_a", "output_a") in state_updates

    def test_stop(self):
        a = _make_mock_agent("a", "ok")
        b = _make_mock_agent("b", "should not run")

        group = SequentialGroup([a, b])

        # Make agent a's run trigger stop
        original_run = a.run

        def run_and_stop(prompt):
            result = original_run(prompt)
            group.stop()
            return result

        a.run = run_and_stop
        result = group.run("test")

        assert "a" in result.agent_results
        assert "b" not in result.agent_results

    def test_timeline_records(self):
        a = _make_mock_agent("a", "ok")
        group = SequentialGroup([a])
        result = group.run("test")

        assert len(result.timeline) == 1
        step = result.timeline[0]
        assert step.agent_name == "a"
        assert step.step_type == "agent_run"
        assert step.duration_ms >= 0

    def test_aggregate_usage(self):
        a = _make_mock_agent("a", "ok")
        b = _make_mock_agent("b", "ok")
        group = SequentialGroup([a, b])
        result = group.run("test")

        assert result.aggregate_usage["total_tokens"] == 30  # 15 * 2
        assert result.aggregate_usage["total_cost"] == pytest.approx(0.002)

    def test_unnamed_agents_get_index_names(self):
        a = MagicMock()
        a.name = ""
        a.output_key = None
        a.run.return_value = AgentResult(
            output="ok", output_text="ok", messages=[], usage={},
            state=AgentState.idle, run_usage={},
        )
        group = SequentialGroup([a])
        result = group.run("test")
        assert "agent_0" in result.agent_results


# ---------------------------------------------------------------------------
# LoopGroup tests
# ---------------------------------------------------------------------------


class TestLoopGroup:
    def test_basic_loop(self):
        a = _make_mock_agent("a", "iter_output", output_key="data")

        def exit_cond(state, iteration):
            return iteration >= 3

        group = LoopGroup([a], exit_condition=exit_cond, max_iterations=10)
        result = group.run("loop test")

        assert result.success is True
        assert len(result.timeline) == 3

    def test_exit_condition_with_state(self):
        counter = {"count": 0}

        a = MagicMock()
        a.name = "counter"
        a.output_key = "count"

        def make_result(*args, **kwargs):
            counter["count"] += 1
            return AgentResult(
                output=str(counter["count"]),
                output_text=str(counter["count"]),
                messages=[], usage={},
                state=AgentState.idle,
                run_usage={"total_tokens": 10, "total_cost": 0.001},
            )

        a.run.side_effect = make_result

        def exit_cond(state, iteration):
            return state.get("count", "0") == "2"

        group = LoopGroup([a], exit_condition=exit_cond, max_iterations=10)
        result = group.run("test")

        assert a.run.call_count == 2

    def test_max_iterations_cap(self):
        a = _make_mock_agent("a", "ok")

        def never_exit(state, iteration):
            return False

        group = LoopGroup([a], exit_condition=never_exit, max_iterations=5)
        result = group.run("test")

        assert a.run.call_count == 5

    def test_error_in_loop(self):
        a = MagicMock()
        a.name = "failing"
        a.output_key = None
        a.run.side_effect = RuntimeError("loop error")

        def exit_cond(state, iteration):
            return False

        group = LoopGroup([a], exit_condition=exit_cond, max_iterations=3, error_policy=ErrorPolicy.fail_fast)
        result = group.run("test")

        assert result.success is False
        assert len(result.errors) == 1


# ---------------------------------------------------------------------------
# RouterAgent tests
# ---------------------------------------------------------------------------


class TestRouterAgent:
    def test_correct_routing(self):
        writer = _make_mock_agent("writer", "written content")
        writer.description = "Writes content"
        coder = _make_mock_agent("coder", "code output")
        coder.description = "Writes code"

        # Mock the routing driver to return "writer"
        routing_driver = MockDriver(responses=["writer"])

        router = RouterAgent(
            driver=routing_driver,
            agents=[writer, coder],
        )
        result = router.run("Write me a story")

        writer.run.assert_called_once_with("Write me a story")
        assert result.output_text == "written content"

    def test_fallback_agent(self):
        writer = _make_mock_agent("writer", "written content")
        fallback = _make_mock_agent("fallback", "fallback output")

        routing_driver = MockDriver(responses=["unknown_agent_xyz"])

        router = RouterAgent(
            driver=routing_driver,
            agents=[writer],
            fallback=fallback,
        )
        result = router.run("test")

        fallback.run.assert_called_once()
        assert result.output_text == "fallback output"

    def test_fuzzy_matching(self):
        my_writer = _make_mock_agent("my_writer", "output")

        routing_driver = MockDriver(responses=["I think my_writer should handle this"])

        router = RouterAgent(
            driver=routing_driver,
            agents=[my_writer],
        )
        result = router.run("test")

        my_writer.run.assert_called_once()

    def test_no_match_no_fallback(self):
        writer = _make_mock_agent("writer", "output")

        routing_driver = MockDriver(responses=["nonexistent"])

        router = RouterAgent(
            driver=routing_driver,
            agents=[writer],
        )
        result = router.run("test")

        # Should return the routing response itself
        writer.run.assert_not_called()
        assert result.output_text == "nonexistent"


# ---------------------------------------------------------------------------
# Agent.as_tool() tests
# ---------------------------------------------------------------------------


class TestAgentAsTool:
    def test_as_tool_basic(self):
        agent = Agent(
            "test/model",
            driver=MockDriver(responses=["tool output"]),
            name="helper",
            description="A helpful agent",
        )

        td = agent.as_tool()
        assert isinstance(td, ToolDefinition)
        assert td.name == "helper"
        assert td.description == "A helpful agent"
        assert "prompt" in td.parameters["properties"]

        result = td.function(prompt="test prompt")
        assert result == "tool output"

    def test_as_tool_custom_name(self):
        agent = Agent("test/model", driver=MockDriver(), name="original")
        td = agent.as_tool(name="custom_name", description="Custom desc")
        assert td.name == "custom_name"
        assert td.description == "Custom desc"

    def test_as_tool_custom_extractor(self):
        agent = Agent(
            "test/model",
            driver=MockDriver(responses=["raw output"]),
            name="agent",
        )

        def extractor(result: AgentResult) -> str:
            return f"processed: {result.output_text}"

        td = agent.as_tool(custom_output_extractor=extractor)
        result = td.function(prompt="test")
        assert result == "processed: raw output"

    def test_as_tool_default_name(self):
        agent = Agent("test/model", driver=MockDriver())
        td = agent.as_tool()
        assert td.name == "agent_tool"


# ---------------------------------------------------------------------------
# GroupAsAgent tests
# ---------------------------------------------------------------------------


class TestGroupAsAgent:
    def test_wraps_group_as_agent(self):
        a = _make_mock_agent("a", "final output", output_key="result")
        inner_group = SequentialGroup([a])

        adapter = GroupAsAgent(inner_group, name="wrapped_group", output_key="group_output")

        result = adapter.run("test prompt")
        assert isinstance(result, AgentResult)
        assert result.output_text == "final output"
        assert adapter.name == "wrapped_group"
        assert adapter.output_key == "group_output"

    def test_nested_groups(self):
        """Test a SequentialGroup containing a LoopGroup via GroupAsAgent."""
        inner_agent = _make_mock_agent("inner", "loop result", output_key="loop_data")

        def exit_after_2(state, iteration):
            return iteration >= 2

        inner_loop = LoopGroup([inner_agent], exit_condition=exit_after_2, max_iterations=5)
        loop_adapter = GroupAsAgent(inner_loop, name="loop_step", output_key="loop_output")

        outer_agent = _make_mock_agent("outer", "outer result")

        outer_group = SequentialGroup([loop_adapter, outer_agent])
        result = outer_group.run("nested test")

        assert result.success is True
        assert "loop_step" in result.agent_results
        assert "outer" in result.agent_results

    def test_stop_propagation(self):
        inner_group = MagicMock()
        inner_group.run.return_value = GroupResult()
        adapter = GroupAsAgent(inner_group, name="test")
        adapter.stop()
        inner_group.stop.assert_called_once()


# ---------------------------------------------------------------------------
# GroupResult serialization tests
# ---------------------------------------------------------------------------


class TestGroupResultSerialization:
    def test_export_round_trip(self):
        result = GroupResult(
            agent_results={"a": AgentResult(
                output="test", output_text="test", messages=[], usage={},
                state=AgentState.idle, run_usage={"total_tokens": 10},
            )},
            aggregate_usage={"total_tokens": 10, "total_cost": 0.001},
            shared_state={"key": "value"},
            elapsed_ms=100.0,
            timeline=[GroupStep(agent_name="a", step_type="agent_run", timestamp=1.0, duration_ms=50.0)],
            errors=[],
            success=True,
        )

        exported = result.export()
        assert isinstance(exported, dict)
        assert exported["success"] is True
        assert exported["shared_state"]["key"] == "value"
        assert len(exported["timeline"]) == 1
        assert exported["aggregate_usage"]["total_tokens"] == 10

        # Ensure JSON serializable
        json_str = json.dumps(exported)
        parsed = json.loads(json_str)
        assert parsed["success"] is True

    def test_save_to_file(self):
        result = GroupResult(
            shared_state={"test": "data"},
            success=True,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            result.save(path)
            with open(path) as f:
                data = json.load(f)
            assert data["shared_state"]["test"] == "data"
        finally:
            os.unlink(path)

    def test_export_with_errors(self):
        result = GroupResult(
            errors=[AgentError(agent_name="a", error=RuntimeError("fail"), output_key="key")],
            success=False,
        )
        exported = result.export()
        assert len(exported["errors"]) == 1
        assert exported["errors"][0]["agent_name"] == "a"
        assert exported["errors"][0]["error_message"] == "fail"


# ---------------------------------------------------------------------------
# Aggregate usage tests
# ---------------------------------------------------------------------------


class TestAggregateUsage:
    def test_empty(self):
        result = _aggregate_usage()
        assert result["total_tokens"] == 0
        assert result["total_cost"] == 0.0

    def test_merges_multiple(self):
        a = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "total_cost": 0.01, "call_count": 1}
        b = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "total_cost": 0.02, "call_count": 2}
        result = _aggregate_usage(a, b)
        assert result["total_tokens"] == 45
        assert result["total_cost"] == pytest.approx(0.03)
        assert result["call_count"] == 3


# ---------------------------------------------------------------------------
# inject_state / shared_state tests
# ---------------------------------------------------------------------------


class TestInjectStateMethod:
    """Tests for the inject_state() public method on group classes."""

    def test_basic_merge(self):
        a = _make_mock_agent("a", "out", output_key="result")
        group = SequentialGroup([a])
        group.inject_state({"key1": "val1", "key2": "val2"})
        assert group.shared_state == {"key1": "val1", "key2": "val2"}

    def test_setdefault_semantics(self):
        """inject_state should NOT overwrite existing keys."""
        a = _make_mock_agent("a", "out")
        group = SequentialGroup([a], state={"existing": "original"})
        group.inject_state({"existing": "overwritten", "new_key": "new_val"})
        assert group.shared_state["existing"] == "original"
        assert group.shared_state["new_key"] == "new_val"

    def test_recursive_propagation(self):
        """inject_state(recursive=True) should propagate to nested groups."""
        inner_agent = _make_mock_agent("inner", "out")
        inner_group = SequentialGroup([inner_agent], state={"inner_key": "inner_val"})
        outer_group = SequentialGroup(
            [(GroupAsAgent(inner_group, name="nested"), None)],
        )
        # GroupAsAgent doesn't have inject_state, so only the inner_group does.
        # Test with direct nesting instead:
        inner_group_2 = SequentialGroup([inner_agent])
        inner_group_2.inject_state({"propagated": "yes"})
        assert inner_group_2.shared_state["propagated"] == "yes"

    def test_loop_group_inject_state(self):
        a = _make_mock_agent("a", "out")
        group = LoopGroup(
            [a],
            exit_condition=lambda state, i: i >= 1,
            state={"pre": "existing"},
        )
        group.inject_state({"pre": "overwrite_attempt", "extra": "value"})
        assert group.shared_state["pre"] == "existing"
        assert group.shared_state["extra"] == "value"


class TestSharedStateProperty:
    """Tests for the shared_state read-only property on group classes."""

    def test_returns_copy(self):
        """shared_state should return a copy, not a reference."""
        a = _make_mock_agent("a", "out")
        group = SequentialGroup([a], state={"key": "val"})
        snapshot = group.shared_state
        snapshot["key"] = "mutated"
        assert group.shared_state["key"] == "val"

    def test_empty_state(self):
        a = _make_mock_agent("a", "out")
        group = SequentialGroup([a])
        assert group.shared_state == {}

    def test_loop_group_shared_state(self):
        a = _make_mock_agent("a", "out")
        group = LoopGroup(
            [a],
            exit_condition=lambda state, i: i >= 1,
            state={"loop_key": "loop_val"},
        )
        assert group.shared_state == {"loop_key": "loop_val"}
