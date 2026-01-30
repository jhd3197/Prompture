"""Tests for the Agent framework."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from prompture.agent import Agent
from prompture.agent_types import AgentResult, AgentState, StepType
from prompture.driver import Driver
from prompture.tools_schema import ToolRegistry

# ---------------------------------------------------------------------------
# Mock drivers
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


class MockToolDriver(Driver):
    """Mock driver that supports tool use and returns sequenced responses."""

    supports_messages = True
    supports_tool_use = True

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self._call_idx = 0

    def generate_messages(self, messages, options):
        return self._get_next()

    def generate_messages_with_tools(self, messages, tools, options):
        return self._get_next()

    def _get_next(self):
        resp = self._responses[self._call_idx]
        self._call_idx += 1
        return resp


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestAgentConstruction:
    def test_basic_construction(self):
        agent = Agent("test/model", driver=MockDriver())
        assert agent.state == AgentState.idle

    def test_construction_requires_model_or_driver(self):
        with pytest.raises(ValueError, match="Either model or driver"):
            Agent()

    def test_tools_via_list(self):
        def fn_a(x: str) -> str:
            """Tool A."""
            return x

        def fn_b(y: int) -> int:
            """Tool B."""
            return y

        agent = Agent("test/model", driver=MockDriver(), tools=[fn_a, fn_b])
        assert len(agent._tools) == 2
        assert "fn_a" in agent._tools
        assert "fn_b" in agent._tools

    def test_tools_via_registry(self):
        reg = ToolRegistry()

        @reg.tool
        def my_tool(x: str) -> str:
            """A tool."""
            return x

        agent = Agent("test/model", driver=MockDriver(), tools=reg)
        assert "my_tool" in agent._tools

    def test_tool_decorator(self):
        agent = Agent("test/model", driver=MockDriver())

        @agent.tool
        def decorated(x: int) -> int:
            """Decorated tool."""
            return x * 2

        assert "decorated" in agent._tools
        assert decorated(3) == 6


# ---------------------------------------------------------------------------
# run() without tools
# ---------------------------------------------------------------------------


class TestAgentRunNoTools:
    def test_basic_run(self):
        driver = MockDriver(["The capital of France is Paris."])
        agent = Agent("test/model", driver=driver)
        result = agent.run("What is the capital of France?")

        assert isinstance(result, AgentResult)
        assert result.output == "The capital of France is Paris."
        assert result.output_text == "The capital of France is Paris."
        assert result.state == AgentState.idle

    def test_messages_populated(self):
        driver = MockDriver(["response text"])
        agent = Agent("test/model", driver=driver)
        result = agent.run("hello")

        assert len(result.messages) >= 2
        user_msgs = [m for m in result.messages if m.get("role") == "user"]
        asst_msgs = [m for m in result.messages if m.get("role") == "assistant"]
        assert len(user_msgs) >= 1
        assert len(asst_msgs) >= 1

    def test_usage_populated(self):
        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver)
        result = agent.run("test")

        assert "prompt_tokens" in result.usage
        assert "completion_tokens" in result.usage
        assert "total_tokens" in result.usage
        assert result.usage["total_tokens"] > 0

    def test_steps_has_output(self):
        driver = MockDriver(["just text"])
        agent = Agent("test/model", driver=driver)
        result = agent.run("test")

        assert len(result.steps) >= 1
        output_steps = [s for s in result.steps if s.step_type == StepType.output]
        assert len(output_steps) >= 1

    def test_state_idle_after_run(self):
        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver)
        result = agent.run("test")
        assert agent.state == AgentState.idle
        assert result.state == AgentState.idle

    def test_no_tool_calls_in_result(self):
        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver)
        result = agent.run("test")
        assert result.all_tool_calls == []


# ---------------------------------------------------------------------------
# run() with tools
# ---------------------------------------------------------------------------


class TestAgentRunWithTools:
    def test_tool_use_round_trip(self):
        """Agent with tools executes tool calls and returns final answer."""

        def get_weather(city: str) -> str:
            """Get the weather for a city."""
            return f"Sunny in {city}"

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "The weather in Paris is sunny.",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        driver = MockToolDriver(responses)
        agent = Agent("test/model", driver=driver, tools=[get_weather])
        result = agent.run("What's the weather in Paris?")

        assert result.output == "The weather in Paris is sunny."
        assert len(result.all_tool_calls) == 1
        assert result.all_tool_calls[0]["name"] == "get_weather"

    def test_tool_steps_recorded(self):
        """Steps include tool_call and tool_result entries."""

        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10, "cost": 0.0},
                "tool_calls": [{"id": "call_add", "name": "add", "arguments": {"a": 3, "b": 4}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "The answer is 7.",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        driver = MockToolDriver(responses)
        agent = Agent("test/model", driver=driver, tools=[add])
        result = agent.run("What is 3 + 4?")

        step_types = [s.step_type for s in result.steps]
        assert StepType.tool_call in step_types
        assert StepType.tool_result in step_types
        assert StepType.output in step_types

    def test_tool_function_actually_called(self):
        """Verify the tool function is invoked with correct arguments."""
        call_log = []

        def logger_tool(msg: str) -> str:
            """Log a message."""
            call_log.append(msg)
            return f"logged: {msg}"

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10, "cost": 0.0},
                "tool_calls": [{"id": "call_log", "name": "logger_tool", "arguments": {"msg": "hello"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "Done.",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        driver = MockToolDriver(responses)
        agent = Agent("test/model", driver=driver, tools=[logger_tool])
        agent.run("Log 'hello'")

        assert call_log == ["hello"]


# ---------------------------------------------------------------------------
# run() with output_type
# ---------------------------------------------------------------------------


class City(BaseModel):
    name: str
    country: str
    population: int | None = None


class TestAgentRunWithOutputType:
    def test_output_type_parsed(self):
        """When output_type is set, result.output is a Pydantic model instance."""
        json_resp = json.dumps({"name": "Paris", "country": "France", "population": 2161000})
        driver = MockDriver([json_resp])
        agent = Agent("test/model", driver=driver, output_type=City)
        result = agent.run("Tell me about Paris")

        assert isinstance(result.output, City)
        assert result.output.name == "Paris"
        assert result.output.country == "France"
        assert result.output.population == 2161000
        assert result.output_text == json_resp

    def test_output_type_with_retry(self):
        """On bad JSON, agent retries and succeeds on second attempt."""
        good_json = json.dumps({"name": "Berlin", "country": "Germany"})
        driver = MockDriver(["not valid json {{}", good_json])
        agent = Agent("test/model", driver=driver, output_type=City)
        result = agent.run("Tell me about Berlin")

        assert isinstance(result.output, City)
        assert result.output.name == "Berlin"

    def test_output_type_all_retries_fail(self):
        """When all retries fail, a ValueError is raised."""
        driver = MockDriver(["bad json"] * 5)
        agent = Agent("test/model", driver=driver, output_type=City)

        with pytest.raises(ValueError, match="Failed to parse output"):
            agent.run("Tell me about nowhere")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


class TestAgentSystemPrompt:
    def test_system_prompt_set(self):
        """System prompt is configured on the agent and used internally."""
        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver, system_prompt="Be helpful")
        assert agent._system_prompt == "Be helpful"

        result = agent.run("test")
        # Conversation stores system_prompt separately, not in messages list
        assert result.state == AgentState.idle

    def test_output_type_schema_in_resolved_prompt(self):
        """When output_type is set, resolved system prompt includes schema."""
        json_resp = json.dumps({"name": "Tokyo", "country": "Japan"})
        driver = MockDriver([json_resp])
        agent = Agent(
            "test/model",
            driver=driver,
            system_prompt="You are a geography expert.",
            output_type=City,
        )

        resolved = agent._resolve_system_prompt()
        assert "geography expert" in resolved
        assert "JSON" in resolved
        assert "name" in resolved  # schema property

        result = agent.run("Tell me about Tokyo")
        assert isinstance(result.output, City)

    def test_no_system_prompt(self):
        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver)
        assert agent._resolve_system_prompt() is None


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


class TestAgentStop:
    def test_stop_sets_flag(self):
        agent = Agent("test/model", driver=MockDriver())
        agent.stop()
        assert agent._stop_requested is True


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_state_errored_on_failure(self):
        """Agent state is 'errored' when run() raises."""

        class FailDriver(Driver):
            supports_messages = True

            def generate_messages(self, messages, options):
                raise RuntimeError("boom")

        agent = Agent("test/model", driver=FailDriver())
        with pytest.raises(RuntimeError, match="boom"):
            agent.run("test")

        assert agent.state == AgentState.errored

    def test_multiple_runs_independent(self):
        """Each run() creates a fresh conversation."""
        driver = MockDriver(["first", "second"])
        agent = Agent("test/model", driver=driver)

        r1 = agent.run("prompt 1")
        r2 = agent.run("prompt 2")

        assert r1.output == "first"
        assert r2.output == "second"
        # Messages should not leak between runs
        assert r1.messages != r2.messages


# ---------------------------------------------------------------------------
# Options forwarding
# ---------------------------------------------------------------------------


class TestAgentOptions:
    def test_options_forwarded(self):
        """Agent forwards options to the Conversation."""
        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver, options={"temperature": 0.5})
        result = agent.run("test")
        assert result.state == AgentState.idle
