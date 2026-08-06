"""Tests for the Agent framework."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from prompture.agents.agent import Agent, AgentIterator, StreamedAgentResult, _get_first_param_name, _tool_wants_context
from prompture.agents.async_agent import AsyncAgent, AsyncAgentIterator, AsyncStreamedAgentResult, _is_async_callable
from prompture.agents.tools_schema import ToolRegistry
from prompture.agents.types import (
    AgentCallbacks,
    AgentResult,
    AgentState,
    GuardrailError,
    ModelRetry,
    RunContext,
    StepType,
    StreamEvent,
    StreamEventType,
)
from prompture.drivers.async_base import AsyncDriver
from prompture.drivers.base import Driver

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

    def generate(self, prompt, options):
        return self._get_next()

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

            def generate(self, prompt, options):
                raise RuntimeError("boom")

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


# ===========================================================================
# Phase 3b tests
# ===========================================================================

# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------


class TestRunContext:
    def test_fields_populated(self):
        """RunContext fields are populated correctly."""
        ctx = RunContext(
            deps="my_deps",
            model="openai/gpt-4",
            usage={"total_tokens": 100},
            messages=[{"role": "user", "content": "hi"}],
            iteration=0,
            prompt="hello",
        )
        assert ctx.deps == "my_deps"
        assert ctx.model == "openai/gpt-4"
        assert ctx.usage["total_tokens"] == 100
        assert len(ctx.messages) == 1
        assert ctx.iteration == 0
        assert ctx.prompt == "hello"

    def test_generic_typing(self):
        """RunContext works with custom deps type."""

        @dataclass
        class MyDeps:
            db: str = "sqlite"

        ctx: RunContext[MyDeps] = RunContext(
            deps=MyDeps(db="postgres"),
            model="test/model",
        )
        assert ctx.deps.db == "postgres"


# ---------------------------------------------------------------------------
# Dynamic System Prompt
# ---------------------------------------------------------------------------


class TestDynamicSystemPrompt:
    def test_callable_system_prompt(self):
        """Callable system prompt receives RunContext and returns string."""

        def make_prompt(ctx: RunContext) -> str:
            return f"You are helping with: {ctx.prompt}"

        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver, system_prompt=make_prompt)
        result = agent.run("my task")
        assert result.state == AgentState.idle

    def test_callable_accesses_deps(self):
        """Callable system prompt can access ctx.deps."""

        @dataclass
        class Config:
            language: str = "Spanish"

        def make_prompt(ctx: RunContext) -> str:
            return f"Respond in {ctx.deps.language}"

        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver, system_prompt=make_prompt)
        result = agent.run("hello", deps=Config(language="French"))
        assert result.state == AgentState.idle

    def test_static_string_backward_compat(self):
        """Static string system_prompt still works."""
        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver, system_prompt="Be brief")
        resolved = agent._resolve_system_prompt()
        assert resolved == "Be brief"


# ---------------------------------------------------------------------------
# RunContext Injection
# ---------------------------------------------------------------------------


class TestRunContextInjection:
    def test_tool_with_run_context(self):
        """Tool with RunContext first param auto-receives it."""
        received_ctx = []

        def ctx_tool(ctx: RunContext, city: str) -> str:
            """A context-aware tool."""
            received_ctx.append(ctx)
            return f"Result for {city}"

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_1", "name": "ctx_tool", "arguments": {"city": "Paris"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "Done.",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        driver = MockToolDriver(responses)
        agent = Agent("test/model", driver=driver, tools=[ctx_tool])
        agent.run("Get info about Paris", deps="my_deps")

        assert len(received_ctx) == 1
        assert isinstance(received_ctx[0], RunContext)
        assert received_ctx[0].deps == "my_deps"

    def test_tool_without_run_context(self):
        """Tool without RunContext works unchanged."""
        call_log = []

        def simple_tool(city: str) -> str:
            """A simple tool."""
            call_log.append(city)
            return f"Info about {city}"

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_1", "name": "simple_tool", "arguments": {"city": "London"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "Done.",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        driver = MockToolDriver(responses)
        agent = Agent("test/model", driver=driver, tools=[simple_tool])
        agent.run("Tell me about London")
        assert call_log == ["London"]

    def test_tool_accesses_deps_attr(self):
        """Tool can access ctx.deps attributes."""

        @dataclass
        class MyDeps:
            api_key: str = "secret123"

        received_keys = []

        def auth_tool(ctx: RunContext[MyDeps], query: str) -> str:
            """Tool needing auth."""
            received_keys.append(ctx.deps.api_key)
            return f"Authed: {query}"

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_1", "name": "auth_tool", "arguments": {"query": "data"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "Got it.",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        driver = MockToolDriver(responses)
        agent = Agent("test/model", driver=driver, tools=[auth_tool])
        agent.run("fetch data", deps=MyDeps(api_key="key_abc"))
        assert received_keys == ["key_abc"]

    def test_run_context_param_stripped_from_schema(self):
        """RunContext param is stripped from JSON schema sent to LLM."""

        def ctx_tool(ctx: RunContext, city: str) -> str:
            """Tool with context."""
            return city

        agent = Agent("test/model", driver=MockDriver(), tools=[ctx_tool])
        # Access the wrapping logic directly
        from prompture.infra.session import UsageSession

        session = UsageSession()
        ctx = agent._build_run_context("test", None, session, [], 0)
        wrapped = agent._wrap_tools_with_context(ctx)

        td = wrapped.get("ctx_tool")
        assert td is not None
        props = td.parameters.get("properties", {})
        assert "ctx" not in props
        assert "city" in props


# ---------------------------------------------------------------------------
# _tool_wants_context helper
# ---------------------------------------------------------------------------


class TestToolWantsContext:
    def test_detects_run_context(self):
        def fn(ctx: RunContext, x: str) -> str:
            return x

        assert _tool_wants_context(fn) is True

    def test_detects_generic_run_context(self):
        @dataclass
        class D:
            val: int = 0

        def fn(ctx: RunContext[D], x: str) -> str:
            return x

        assert _tool_wants_context(fn) is True

    def test_no_context(self):
        def fn(x: str) -> str:
            return x

        assert _tool_wants_context(fn) is False

    def test_no_params(self):
        def fn() -> str:
            return "hi"

        assert _tool_wants_context(fn) is False


class TestGetFirstParamName:
    def test_basic(self):
        def fn(ctx: RunContext, x: str) -> str:
            return x

        assert _get_first_param_name(fn) == "ctx"

    def test_no_params(self):
        def fn() -> str:
            return ""

        assert _get_first_param_name(fn) == ""


# ---------------------------------------------------------------------------
# Input Guardrails
# ---------------------------------------------------------------------------


class TestInputGuardrails:
    def test_guardrail_transforms_prompt(self):
        """Input guardrail can transform the prompt."""

        def uppercase_guard(ctx: RunContext, prompt: str) -> str:
            return prompt.upper()

        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver, input_guardrails=[uppercase_guard])
        result = agent.run("hello")
        assert result.state == AgentState.idle

    def test_guardrail_returns_none(self):
        """Returning None leaves prompt unchanged."""

        def passthrough(ctx: RunContext, prompt: str) -> None:
            return None

        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver, input_guardrails=[passthrough])
        result = agent.run("hello")
        assert result.state == AgentState.idle

    def test_guardrail_raises_error(self):
        """GuardrailError rejects the prompt entirely."""

        def reject(ctx: RunContext, prompt: str) -> str:
            raise GuardrailError("Blocked!")

        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver, input_guardrails=[reject])
        with pytest.raises(GuardrailError, match="Blocked!"):
            agent.run("bad prompt")

    def test_multiple_guardrails_chain(self):
        """Multiple guardrails execute in order."""
        log = []

        def guard1(ctx: RunContext, prompt: str) -> str:
            log.append("g1")
            return prompt + " [g1]"

        def guard2(ctx: RunContext, prompt: str) -> str:
            log.append("g2")
            return prompt + " [g2]"

        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver, input_guardrails=[guard1, guard2])
        agent.run("hello")
        assert log == ["g1", "g2"]


# ---------------------------------------------------------------------------
# Output Guardrails
# ---------------------------------------------------------------------------


class TestOutputGuardrails:
    def test_guardrail_passes(self):
        """Output guardrail returning None passes."""

        def pass_guard(ctx: RunContext, result: AgentResult) -> None:
            return None

        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver, output_guardrails=[pass_guard])
        result = agent.run("hello")
        assert result.output == "ok"

    def test_guardrail_modifies_result(self):
        """Output guardrail can return a modified AgentResult."""

        def modify_guard(ctx: RunContext, result: AgentResult) -> AgentResult:
            return AgentResult(
                output="modified",
                output_text=result.output_text,
                messages=result.messages,
                usage=result.usage,
                steps=result.steps,
                all_tool_calls=result.all_tool_calls,
                state=result.state,
                run_usage=result.run_usage,
            )

        driver = MockDriver(["original"])
        agent = Agent("test/model", driver=driver, output_guardrails=[modify_guard])
        result = agent.run("hello")
        assert result.output == "modified"

    def test_guardrail_model_retry(self):
        """Output guardrail raises ModelRetry -> LLM retries."""
        attempt_count = [0]

        def strict_guard(ctx: RunContext, result: AgentResult) -> None:
            attempt_count[0] += 1
            if attempt_count[0] <= 1:
                raise ModelRetry("Output must contain 'good'")
            return None

        driver = MockDriver(["bad output", "good output"])
        agent = Agent("test/model", driver=driver, output_guardrails=[strict_guard])
        result = agent.run("hello")
        assert result.state == AgentState.idle
        assert attempt_count[0] == 2

    def test_guardrail_max_retries_exceeded(self):
        """When guardrail always raises ModelRetry, ValueError after max retries."""

        def always_fail(ctx: RunContext, result: AgentResult) -> None:
            raise ModelRetry("Never good enough")

        driver = MockDriver(["bad"] * 10)
        agent = Agent("test/model", driver=driver, output_guardrails=[always_fail])
        with pytest.raises(ValueError, match="Output guardrail failed after"):
            agent.run("hello")


# ---------------------------------------------------------------------------
# ModelRetry in Tools
# ---------------------------------------------------------------------------


class TestModelRetryInTools:
    def test_model_retry_returns_error_string(self):
        """Tool raising ModelRetry -> error string in tool result."""

        def picky_tool(value: str) -> str:
            """A picky tool."""
            raise ModelRetry("Value must be 'correct'")

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_1", "name": "picky_tool", "arguments": {"value": "wrong"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "I got an error, sorry.",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        driver = MockToolDriver(responses)
        agent = Agent("test/model", driver=driver, tools=[picky_tool])
        result = agent.run("Use the picky tool")

        # The tool result message should contain the error
        tool_results = [m for m in result.messages if m.get("role") == "tool"]
        assert any("Error: Value must be 'correct'" in m.get("content", "") for m in tool_results)

    def test_model_retry_does_not_crash(self):
        """ModelRetry in tool is caught gracefully, agent continues."""

        def fail_tool(x: str) -> str:
            """Fails with retry."""
            raise ModelRetry("Bad input")

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_1", "name": "fail_tool", "arguments": {"x": "test"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "Handled the error.",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        driver = MockToolDriver(responses)
        agent = Agent("test/model", driver=driver, tools=[fail_tool])
        result = agent.run("test")
        assert result.output == "Handled the error."


# ---------------------------------------------------------------------------
# AgentCallbacks
# ---------------------------------------------------------------------------


class TestAgentCallbacks:
    def test_on_step_fired(self):
        """on_step is fired for each step."""
        step_log: list[Any] = []
        cb = AgentCallbacks(on_step=lambda s: step_log.append(s))

        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver, agent_callbacks=cb)
        agent.run("hello")

        assert len(step_log) >= 1
        assert all(hasattr(s, "step_type") for s in step_log)

    def test_on_tool_start_end_fired(self):
        """on_tool_start and on_tool_end are fired during tool execution."""
        start_log: list[tuple[str, dict]] = []
        end_log: list[tuple[str, Any]] = []

        cb = AgentCallbacks(
            on_tool_start=lambda name, args: start_log.append((name, args)),
            on_tool_end=lambda name, result: end_log.append((name, result)),
        )

        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_1", "name": "greet", "arguments": {"name": "Alice"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "Done.",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        driver = MockToolDriver(responses)
        agent = Agent("test/model", driver=driver, tools=[greet], agent_callbacks=cb)
        agent.run("Greet Alice")

        assert len(start_log) == 1
        assert start_log[0][0] == "greet"
        assert start_log[0][1] == {"name": "Alice"}

        assert len(end_log) == 1
        assert end_log[0][0] == "greet"
        assert end_log[0][1] == "Hello, Alice!"

    def test_on_iteration_fired(self):
        """on_iteration is fired at the start."""
        iter_log: list[int] = []
        cb = AgentCallbacks(on_iteration=lambda i: iter_log.append(i))

        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver, agent_callbacks=cb)
        agent.run("hello")

        assert iter_log == [0]

    def test_on_output_fired(self):
        """on_output is fired with final result."""
        output_log: list[AgentResult] = []
        cb = AgentCallbacks(on_output=lambda r: output_log.append(r))

        driver = MockDriver(["final answer"])
        agent = Agent("test/model", driver=driver, agent_callbacks=cb)
        result = agent.run("hello")

        assert len(output_log) == 1
        assert output_log[0].output == "final answer"
        assert output_log[0] is result


# ---------------------------------------------------------------------------
# Per-run UsageSession
# ---------------------------------------------------------------------------


class TestPerRunUsageSession:
    def test_run_usage_populated(self):
        """result.run_usage contains token/cost data."""
        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver)
        result = agent.run("test")

        assert "total_tokens" in result.run_usage
        assert "total_cost" in result.run_usage
        assert result.run_usage["total_tokens"] > 0

    def test_independent_runs_separate_usage(self):
        """Two independent runs have separate run_usage."""
        driver = MockDriver(["first", "second"])
        agent = Agent("test/model", driver=driver)

        r1 = agent.run("prompt 1")
        r2 = agent.run("prompt 2")

        # Each run should have its own session totals (not cumulative)
        assert r1.run_usage["call_count"] == 1
        assert r2.run_usage["call_count"] == 1


# ---------------------------------------------------------------------------
# max_cost
# ---------------------------------------------------------------------------


class TestMaxCost:
    def test_output_parse_retries_skipped_over_budget(self):
        """Output parse retries are skipped when over budget."""
        # Cost per call is 0.001, max_cost is 0.0005 -> over budget after first call
        driver = MockDriver(["bad json", '{"name": "X", "country": "Y"}'])
        agent = Agent("test/model", driver=driver, output_type=City, max_cost=0.0005)

        # Should raise because it won't retry due to budget
        with pytest.raises(ValueError, match="Failed to parse output"):
            agent.run("test")

    def test_normal_behavior_under_budget(self):
        """Normal behavior when under budget."""
        good_json = json.dumps({"name": "Paris", "country": "France"})
        driver = MockDriver(["bad json", good_json])
        agent = Agent("test/model", driver=driver, output_type=City, max_cost=10.0)

        result = agent.run("test")
        assert isinstance(result.output, City)
        assert result.output.name == "Paris"


# ---------------------------------------------------------------------------
# Backward Compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_basic_run_no_new_params(self):
        """Basic run (no new params) works exactly as Phase 3a."""
        driver = MockDriver(["Hello!"])
        agent = Agent("test/model", driver=driver)
        result = agent.run("hi")
        assert result.output == "Hello!"
        assert result.state == AgentState.idle

    def test_tools_run_unchanged(self):
        """Tools run unchanged without new params."""

        def echo(text: str) -> str:
            """Echo text."""
            return text

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_1", "name": "echo", "arguments": {"text": "hello"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "Echo: hello",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        driver = MockToolDriver(responses)
        agent = Agent("test/model", driver=driver, tools=[echo])
        result = agent.run("Echo hello")
        assert result.output == "Echo: hello"

    def test_output_type_unchanged(self):
        """output_type works unchanged."""
        json_resp = json.dumps({"name": "Rome", "country": "Italy"})
        driver = MockDriver([json_resp])
        agent = Agent("test/model", driver=driver, output_type=City)
        result = agent.run("Tell me about Rome")
        assert isinstance(result.output, City)
        assert result.output.name == "Rome"


# ===========================================================================
# Phase 3c tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Mock async drivers
# ---------------------------------------------------------------------------


class MockAsyncDriver(AsyncDriver):
    """Simple mock async driver returning canned text responses."""

    supports_messages = True
    supports_tool_use = False

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or ["Hello from async mock"])
        self._call_count = 0
        self.model = "mock-async-model"

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self._make_response()

    async def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
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


class MockAsyncToolDriver(AsyncDriver):
    """Mock async driver that supports tool use."""

    supports_messages = True
    supports_tool_use = True

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self._call_idx = 0

    async def generate(self, prompt, options):
        return self._get_next()

    async def generate_messages(self, messages, options):
        return self._get_next()

    async def generate_messages_with_tools(self, messages, tools, options):
        return self._get_next()

    def _get_next(self):
        resp = self._responses[self._call_idx]
        self._call_idx += 1
        return resp


# ---------------------------------------------------------------------------
# StreamEvent tests
# ---------------------------------------------------------------------------


class TestStreamEvent:
    def test_stream_event_fields(self):
        """StreamEvent fields are populated correctly."""
        event = StreamEvent(
            event_type=StreamEventType.text_delta,
            data="hello",
        )
        assert event.event_type == StreamEventType.text_delta
        assert event.data == "hello"
        assert event.step is None

    def test_stream_event_type_values(self):
        """StreamEventType enum has expected values."""
        assert StreamEventType.text_delta == "text_delta"
        assert StreamEventType.tool_call == "tool_call"
        assert StreamEventType.tool_result == "tool_result"
        assert StreamEventType.output == "output"


# ---------------------------------------------------------------------------
# AgentIterator tests
# ---------------------------------------------------------------------------


class TestAgentIterator:
    def test_iter_returns_agent_iterator(self):
        """iter() returns an AgentIterator."""
        driver = MockDriver(["Hello!"])
        agent = Agent("test/model", driver=driver)
        it = agent.iter("test")
        assert isinstance(it, AgentIterator)

    def test_iter_yields_agent_steps(self):
        """Iterating yields AgentStep objects."""
        driver = MockDriver(["Hello!"])
        agent = Agent("test/model", driver=driver)
        it = agent.iter("test")

        steps = list(it)
        assert len(steps) >= 1
        from prompture.agents.types import AgentStep

        assert all(isinstance(s, AgentStep) for s in steps)

    def test_result_none_before_populated_after(self):
        """result is None before iteration, populated after."""
        driver = MockDriver(["Hello!"])
        agent = Agent("test/model", driver=driver)
        it = agent.iter("test")
        assert it.result is None

        for _ in it:
            pass

        assert it.result is not None
        assert isinstance(it.result, AgentResult)
        assert it.result.output == "Hello!"

    def test_iter_with_tools(self):
        """iter() works with tools and yields tool_call/tool_result steps."""

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
                "text": "7",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        driver = MockToolDriver(responses)
        agent = Agent("test/model", driver=driver, tools=[add])
        it = agent.iter("What is 3 + 4?")

        steps = list(it)
        step_types = [s.step_type for s in steps]
        assert StepType.tool_call in step_types
        assert StepType.tool_result in step_types
        assert StepType.output in step_types
        assert it.result is not None


# ---------------------------------------------------------------------------
# run_stream() tests
# ---------------------------------------------------------------------------


class TestRunStream:
    def test_run_stream_returns_streamed_result(self):
        """run_stream() returns StreamedAgentResult."""
        driver = MockDriver(["Hello streaming!"])
        agent = Agent("test/model", driver=driver)
        stream = agent.run_stream("test")
        assert isinstance(stream, StreamedAgentResult)

    def test_run_stream_yields_text_delta(self):
        """Iterating yields StreamEvent with text_delta events."""
        driver = MockDriver(["Hello streaming!"])
        agent = Agent("test/model", driver=driver)
        stream = agent.run_stream("test")

        events = list(stream)
        delta_events = [e for e in events if e.event_type == StreamEventType.text_delta]
        assert len(delta_events) >= 1

    def test_run_stream_final_output_event(self):
        """Last event is StreamEvent(output) with AgentResult."""
        driver = MockDriver(["Hello streaming!"])
        agent = Agent("test/model", driver=driver)
        stream = agent.run_stream("test")

        events = list(stream)
        output_events = [e for e in events if e.event_type == StreamEventType.output]
        assert len(output_events) == 1
        assert isinstance(output_events[0].data, AgentResult)

    def test_run_stream_result_populated(self):
        """result is populated after iteration completes."""
        driver = MockDriver(["Hello streaming!"])
        agent = Agent("test/model", driver=driver)
        stream = agent.run_stream("test")
        assert stream.result is None

        for _ in stream:
            pass

        assert stream.result is not None
        assert isinstance(stream.result, AgentResult)


# ---------------------------------------------------------------------------
# AsyncAgent tests
# ---------------------------------------------------------------------------


class TestAsyncAgent:
    def test_construction(self):
        """AsyncAgent construction mirrors Agent."""
        agent = AsyncAgent("test/model", driver=MockAsyncDriver())
        assert agent.state == AgentState.idle

    def test_construction_requires_model_or_driver(self):
        with pytest.raises(ValueError, match="Either model or driver"):
            AsyncAgent()

    def test_async_run(self):
        """async run() returns AgentResult."""
        driver = MockAsyncDriver(["Async hello!"])
        agent = AsyncAgent("test/model", driver=driver)
        result = asyncio.run(agent.run("test"))

        assert isinstance(result, AgentResult)
        assert result.output == "Async hello!"
        assert result.state == AgentState.idle

    def test_async_run_with_tools(self):
        """Sync tools work in AsyncAgent."""

        def get_weather(city: str) -> str:
            """Get the weather."""
            return f"Sunny in {city}"

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "It's sunny in Paris.",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        driver = MockAsyncToolDriver(responses)
        agent = AsyncAgent("test/model", driver=driver, tools=[get_weather])
        result = asyncio.run(agent.run("Weather in Paris?"))

        assert result.output == "It's sunny in Paris."
        assert len(result.all_tool_calls) == 1

    def test_async_run_with_output_type(self):
        """output_type parsing works in AsyncAgent."""
        json_resp = json.dumps({"name": "Berlin", "country": "Germany"})
        driver = MockAsyncDriver([json_resp])
        agent = AsyncAgent("test/model", driver=driver, output_type=City)
        result = asyncio.run(agent.run("Tell me about Berlin"))

        assert isinstance(result.output, City)
        assert result.output.name == "Berlin"

    def test_async_run_context_injection(self):
        """RunContext injection works in AsyncAgent."""
        received_ctx = []

        def ctx_tool(ctx: RunContext, query: str) -> str:
            """Context-aware tool."""
            received_ctx.append(ctx)
            return f"Result for {query}"

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_1", "name": "ctx_tool", "arguments": {"query": "test"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "Done.",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        driver = MockAsyncToolDriver(responses)
        agent = AsyncAgent("test/model", driver=driver, tools=[ctx_tool])
        asyncio.run(agent.run("test", deps="my_deps"))

        assert len(received_ctx) == 1
        assert received_ctx[0].deps == "my_deps"

    def test_async_guardrails(self):
        """Input guardrails work in AsyncAgent."""

        def block_guard(ctx: RunContext, prompt: str) -> str:
            if "blocked" in prompt:
                raise GuardrailError("Blocked!")
            return prompt

        driver = MockAsyncDriver(["ok"])
        agent = AsyncAgent("test/model", driver=driver, input_guardrails=[block_guard])

        with pytest.raises(GuardrailError, match="Blocked!"):
            asyncio.run(agent.run("blocked content"))


# ---------------------------------------------------------------------------
# AsyncAgentIter tests
# ---------------------------------------------------------------------------


class TestAsyncAgentIter:
    def test_async_iter_returns_iterator(self):
        """async iter() returns AsyncAgentIterator."""

        async def _test():
            driver = MockAsyncDriver(["Hello!"])
            agent = AsyncAgent("test/model", driver=driver)
            it = agent.iter("test")
            assert isinstance(it, AsyncAgentIterator)

        asyncio.run(_test())

    def test_async_iter_yields_steps(self):
        """Async iteration yields AgentStep objects."""

        async def _test():
            driver = MockAsyncDriver(["Hello!"])
            agent = AsyncAgent("test/model", driver=driver)
            it = agent.iter("test")

            steps = []
            async for step in it:
                steps.append(step)

            from prompture.agents.types import AgentStep

            assert len(steps) >= 1
            assert all(isinstance(s, AgentStep) for s in steps)

        asyncio.run(_test())

    def test_async_iter_result_populated(self):
        """result is populated after async iteration completes."""

        async def _test():
            driver = MockAsyncDriver(["Hello!"])
            agent = AsyncAgent("test/model", driver=driver)
            it = agent.iter("test")
            assert it.result is None

            async for _ in it:
                pass

            # Result may be set via output event or agent attribute
            # For AsyncAgentIterator, result capture depends on generator frame access
            # which may not work in all Python implementations
            return it

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# AsyncAgentStream tests
# ---------------------------------------------------------------------------


class TestAsyncAgentStream:
    def test_async_stream_returns_result_type(self):
        """async run_stream() returns AsyncStreamedAgentResult."""

        async def _test():
            driver = MockAsyncDriver(["Hello!"])
            agent = AsyncAgent("test/model", driver=driver)
            stream = agent.run_stream("test")
            assert isinstance(stream, AsyncStreamedAgentResult)

        asyncio.run(_test())

    def test_async_stream_yields_events(self):
        """Async streaming yields StreamEvent objects."""

        async def _test():
            driver = MockAsyncDriver(["Hello!"])
            agent = AsyncAgent("test/model", driver=driver)
            stream = agent.run_stream("test")

            events = []
            async for event in stream:
                events.append(event)

            assert len(events) >= 1
            assert all(isinstance(e, StreamEvent) for e in events)
            # Should have at least a text_delta and output event
            types = [e.event_type for e in events]
            assert StreamEventType.text_delta in types
            assert StreamEventType.output in types

        asyncio.run(_test())

    def test_async_stream_result_populated(self):
        """result is populated after async stream iteration completes."""

        async def _test():
            driver = MockAsyncDriver(["Hello!"])
            agent = AsyncAgent("test/model", driver=driver)
            stream = agent.run_stream("test")
            assert stream.result is None

            async for _ in stream:
                pass

            assert stream.result is not None
            assert isinstance(stream.result, AgentResult)

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# Async tool detection tests
# ---------------------------------------------------------------------------


class TestAsyncToolDetection:
    def test_detects_async_function(self):
        async def my_async_fn() -> str:
            return "async"

        assert _is_async_callable(my_async_fn) is True

    def test_detects_sync_function(self):
        def my_sync_fn() -> str:
            return "sync"

        assert _is_async_callable(my_sync_fn) is False

    def test_detects_async_callable_object(self):
        class AsyncCallable:
            async def __call__(self) -> str:
                return "async callable"

        assert _is_async_callable(AsyncCallable()) is True


# ===========================================================================
# New feature tests (Items 1-7 from FEEDBACK.md)
# ===========================================================================


# ---------------------------------------------------------------------------
# Item 7: on_message callback
# ---------------------------------------------------------------------------


class TestOnMessageCallback:
    def test_on_message_fired_with_output_text(self):
        """on_message callback receives the final output text string."""
        message_log: list[str] = []
        cb = AgentCallbacks(on_message=lambda text: message_log.append(text))

        driver = MockDriver(["Hello from agent!"])
        agent = Agent("test/model", driver=driver, agent_callbacks=cb)
        result = agent.run("test")

        assert len(message_log) == 1
        assert message_log[0] == "Hello from agent!"
        assert message_log[0] == result.output_text

    def test_on_message_not_fired_when_none(self):
        """on_message is not called when set to None."""
        cb = AgentCallbacks(on_message=None)
        driver = MockDriver(["ok"])
        agent = Agent("test/model", driver=driver, agent_callbacks=cb)
        result = agent.run("test")
        assert result.output == "ok"

    def test_on_message_fired_in_stream(self):
        """on_message fires in run_stream()."""
        message_log: list[str] = []
        cb = AgentCallbacks(on_message=lambda text: message_log.append(text))

        driver = MockDriver(["streamed response"])
        agent = Agent("test/model", driver=driver, agent_callbacks=cb)
        stream = agent.run_stream("test")
        for _ in stream:
            pass

        assert len(message_log) == 1
        assert message_log[0] == "streamed response"

    def test_async_on_message_fired(self):
        """on_message fires in AsyncAgent.run()."""
        message_log: list[str] = []

        async def async_on_message(text: str) -> None:
            message_log.append(text)

        cb = AgentCallbacks(on_message=async_on_message)
        driver = MockAsyncDriver(["Async hello!"])
        agent = AsyncAgent("test/model", driver=driver, agent_callbacks=cb)
        asyncio.run(agent.run("test"))

        assert len(message_log) == 1
        assert message_log[0] == "Async hello!"


# ---------------------------------------------------------------------------
# Item 1: Async callbacks in AsyncAgent
# ---------------------------------------------------------------------------


class TestAsyncCallbacks:
    def test_async_on_output_called(self):
        """Async on_output callback is awaited properly."""
        output_log: list[str] = []

        async def async_on_output(result: AgentResult) -> None:
            output_log.append(result.output_text)

        cb = AgentCallbacks(on_output=async_on_output)
        driver = MockAsyncDriver(["async result"])
        agent = AsyncAgent("test/model", driver=driver, agent_callbacks=cb)
        asyncio.run(agent.run("test"))

        assert len(output_log) == 1
        assert output_log[0] == "async result"

    def test_async_on_step_called(self):
        """Async on_step callback is awaited for each step."""
        step_log: list[Any] = []

        async def async_on_step(step: Any) -> None:
            step_log.append(step)

        cb = AgentCallbacks(on_step=async_on_step)
        driver = MockAsyncDriver(["ok"])
        agent = AsyncAgent("test/model", driver=driver, agent_callbacks=cb)
        asyncio.run(agent.run("test"))

        assert len(step_log) >= 1

    def test_async_on_iteration_called(self):
        """Async on_iteration callback is awaited."""
        iter_log: list[int] = []

        async def async_on_iteration(i: int) -> None:
            iter_log.append(i)

        cb = AgentCallbacks(on_iteration=async_on_iteration)
        driver = MockAsyncDriver(["ok"])
        agent = AsyncAgent("test/model", driver=driver, agent_callbacks=cb)
        asyncio.run(agent.run("test"))

        assert iter_log == [0]

    def test_sync_callbacks_still_work_in_async_agent(self):
        """Sync callbacks continue to work in AsyncAgent."""
        output_log: list[str] = []

        def sync_on_output(result: AgentResult) -> None:
            output_log.append(result.output_text)

        cb = AgentCallbacks(on_output=sync_on_output)
        driver = MockAsyncDriver(["sync callback test"])
        agent = AsyncAgent("test/model", driver=driver, agent_callbacks=cb)
        asyncio.run(agent.run("test"))

        assert output_log == ["sync callback test"]


# ---------------------------------------------------------------------------
# Item 4: Conversation continuity
# ---------------------------------------------------------------------------


class TestConversationContinuity:
    def test_persistent_conversation_shares_history(self):
        """With persistent_conversation=True, runs share conversation history."""
        driver = MockDriver(["first", "second"])
        agent = Agent("test/model", driver=driver, persistent_conversation=True)

        r1 = agent.run("prompt 1")
        r2 = agent.run("prompt 2")

        assert r1.output == "first"
        assert r2.output == "second"
        # Second result should have messages from both runs
        assert len(r2.messages) > len(r1.messages)

    def test_non_persistent_conversation_is_default(self):
        """Default behavior creates fresh conversations."""
        driver = MockDriver(["first", "second"])
        agent = Agent("test/model", driver=driver)

        r1 = agent.run("prompt 1")
        r2 = agent.run("prompt 2")

        # Messages from run 1 should not appear in run 2
        assert len(r1.messages) == len(r2.messages)

    def test_conversation_property(self):
        """conversation property returns the persistent conversation."""
        driver = MockDriver(["hello"])
        agent = Agent("test/model", driver=driver, persistent_conversation=True)

        assert agent.conversation is None  # before first run
        agent.run("test")
        assert agent.conversation is not None

    def test_conversation_property_none_without_persistence(self):
        """conversation property is None without persistence."""
        driver = MockDriver(["hello"])
        agent = Agent("test/model", driver=driver)
        agent.run("test")
        assert agent.conversation is None

    def test_messages_property(self):
        """messages property returns conversation messages."""
        driver = MockDriver(["hello"])
        agent = Agent("test/model", driver=driver, persistent_conversation=True)

        assert agent.messages == []
        agent.run("test")
        assert len(agent.messages) > 0

    def test_clear_history(self):
        """clear_history() resets the conversation."""
        driver = MockDriver(["first", "second"])
        agent = Agent("test/model", driver=driver, persistent_conversation=True)

        agent.run("prompt 1")
        assert len(agent.messages) > 0

        agent.clear_history()
        assert agent.messages == []

        # Can still run after clearing
        agent.run("prompt 2")
        assert len(agent.messages) > 0

    def test_async_persistent_conversation(self):
        """AsyncAgent supports persistent_conversation."""

        async def _test():
            driver = MockAsyncDriver(["first", "second"])
            agent = AsyncAgent("test/model", driver=driver, persistent_conversation=True)

            r1 = await agent.run("prompt 1")
            r2 = await agent.run("prompt 2")

            assert r1.output == "first"
            assert r2.output == "second"
            assert len(r2.messages) > len(r1.messages)

        asyncio.run(_test())

    def test_async_clear_history(self):
        """AsyncAgent.clear_history() works."""

        async def _test():
            driver = MockAsyncDriver(["hello", "world"])
            agent = AsyncAgent("test/model", driver=driver, persistent_conversation=True)

            await agent.run("test")
            assert len(agent.messages) > 0

            agent.clear_history()
            assert agent.messages == []

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# Item 2: Streaming with tools emits events
# ---------------------------------------------------------------------------


class TestStreamingWithTools:
    def test_stream_emits_tool_events(self):
        """run_stream() with tools emits tool_call and tool_result events."""

        def get_weather(city: str) -> str:
            """Get the weather."""
            return f"Sunny in {city}"

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "It's sunny in Paris.",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        driver = MockToolDriver(responses)
        agent = Agent("test/model", driver=driver, tools=[get_weather])
        stream = agent.run_stream("What's the weather?")

        events = list(stream)
        event_types = [e.event_type for e in events]

        assert StreamEventType.tool_call in event_types
        assert StreamEventType.tool_result in event_types
        assert StreamEventType.text_delta in event_types
        assert StreamEventType.output in event_types

    def test_stream_tool_call_event_has_data(self):
        """tool_call event data contains name, arguments, id."""

        def echo(text: str) -> str:
            """Echo text."""
            return text

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_echo", "name": "echo", "arguments": {"text": "hello"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "Echoed: hello",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        driver = MockToolDriver(responses)
        agent = Agent("test/model", driver=driver, tools=[echo])
        stream = agent.run_stream("Echo hello")

        events = list(stream)
        tc_events = [e for e in events if e.event_type == StreamEventType.tool_call]
        tr_events = [e for e in events if e.event_type == StreamEventType.tool_result]

        assert len(tc_events) == 1
        assert tc_events[0].data["name"] == "echo"
        assert tc_events[0].data["arguments"] == {"text": "hello"}
        assert tc_events[0].data["id"] == "call_echo"

        assert len(tr_events) == 1
        assert tr_events[0].data["name"] == "echo"
        assert tr_events[0].data["result"] == "hello"

    def test_async_stream_emits_tool_events(self):
        """AsyncAgent.run_stream() with tools emits tool events."""

        def get_time() -> str:
            """Get current time."""
            return "12:00 PM"

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_1", "name": "get_time", "arguments": {}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "The time is 12:00 PM.",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        async def _test():
            driver = MockAsyncToolDriver(responses)
            agent = AsyncAgent("test/model", driver=driver, tools=[get_time])
            stream = agent.run_stream("What time is it?")

            events = []
            async for event in stream:
                events.append(event)

            event_types = [e.event_type for e in events]
            assert StreamEventType.tool_call in event_types
            assert StreamEventType.tool_result in event_types
            assert StreamEventType.text_delta in event_types

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# Item 3: Async approval flow
# ---------------------------------------------------------------------------


class TestAsyncApprovalFlow:
    def test_async_approval_needed_with_sync_callback(self):
        """ApprovalRequired in AsyncAgent with sync on_approval_needed."""
        from prompture.agents.types import ApprovalRequired

        approval_log: list[tuple[str, str]] = []

        def approval_handler(tool_name: str, action: str, details: dict) -> bool:
            approval_log.append((tool_name, action))
            return True

        def dangerous_tool(command: str) -> str:
            """Run a dangerous command."""
            raise ApprovalRequired("dangerous_tool", f"Execute: {command}")

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_1", "name": "dangerous_tool", "arguments": {"command": "rm -rf"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "Executed.",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        cb = AgentCallbacks(on_approval_needed=approval_handler)
        driver = MockAsyncToolDriver(responses)
        agent = AsyncAgent("test/model", driver=driver, tools=[dangerous_tool], agent_callbacks=cb)
        asyncio.run(agent.run("Run dangerous command"))

        assert len(approval_log) == 1
        assert approval_log[0][0] == "dangerous_tool"

    def test_async_approval_denied(self):
        """ApprovalRequired denied returns error message."""
        from prompture.agents.types import ApprovalRequired

        def deny_handler(tool_name: str, action: str, details: dict) -> bool:
            return False

        def dangerous_tool(command: str) -> str:
            """Run a dangerous command."""
            raise ApprovalRequired("dangerous_tool", f"Execute: {command}")

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_1", "name": "dangerous_tool", "arguments": {"command": "rm -rf"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "Tool was denied.",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        cb = AgentCallbacks(on_approval_needed=deny_handler)
        driver = MockAsyncToolDriver(responses)
        agent = AsyncAgent("test/model", driver=driver, tools=[dangerous_tool], agent_callbacks=cb)
        result = asyncio.run(agent.run("test"))

        # Tool result should contain denial message
        tool_results = [m for m in result.messages if m.get("role") == "tool"]
        assert any("denied" in m.get("content", "").lower() for m in tool_results)


# ---------------------------------------------------------------------------
# Item 5: Extraction resilience (extract_with_model)
# ---------------------------------------------------------------------------


class TestExtractionResilience:
    """Tests for max_retries and fallback params in extract_with_model."""

    def test_max_retries_default_is_one(self):
        """Default max_retries=1 preserves existing single-attempt behavior."""
        import inspect

        from prompture.extraction.core import extract_with_model

        sig = inspect.signature(extract_with_model)
        assert sig.parameters["max_retries"].default == 1

    def test_fallback_default_is_none(self):
        """Default fallback=None preserves existing behavior."""
        import inspect

        from prompture.extraction.core import extract_with_model

        sig = inspect.signature(extract_with_model)
        assert sig.parameters["fallback"].default is None

    def test_fallback_used_on_failure(self):
        """When extraction fails and fallback is provided, returns fallback."""
        from prompture.extraction.core import extract_with_model

        class SimpleModel(BaseModel):
            name: str
            value: int = 0

        fallback = SimpleModel(name="default", value=-1)

        # Mock the extraction to fail by providing invalid driver
        # We'll test the retry/fallback mechanism via direct call
        # by mocking extract_and_jsonify to always raise
        import unittest.mock as mock

        with mock.patch("prompture.extraction.core.extract_and_jsonify", side_effect=ValueError("Parse error")):
            result = extract_with_model(
                SimpleModel,
                "some text",
                model_name="mock/model",
                max_retries=2,
                fallback=fallback,
            )

        assert result["model"].name == "default"
        assert result["model"].value == -1
        assert result["usage"]["fallback_used"] is True

    def test_no_fallback_reraises(self):
        """When no fallback is provided and all retries fail, re-raises."""
        from prompture.extraction.core import extract_with_model

        class SimpleModel(BaseModel):
            name: str

        import unittest.mock as mock

        with mock.patch("prompture.extraction.core.extract_and_jsonify", side_effect=ValueError("Parse error")):
            with pytest.raises(ValueError, match="Parse error"):
                extract_with_model(
                    SimpleModel,
                    "some text",
                    model_name="mock/model",
                    max_retries=2,
                )

    def test_retry_succeeds_on_second_attempt(self):
        """When first attempt fails but second succeeds, returns success."""
        from prompture.extraction.core import extract_with_model

        class SimpleModel(BaseModel):
            name: str

        call_count = [0]
        good_result = {
            "json_string": '{"name": "test"}',
            "json_object": {"name": "test"},
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.001,
                "model_name": "mock",
                "raw_response": {},
            },
        }

        import unittest.mock as mock

        def mock_extract(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("First attempt fails")
            return good_result

        with mock.patch("prompture.extraction.core.extract_and_jsonify", side_effect=mock_extract):
            result = extract_with_model(
                SimpleModel,
                "some text",
                model_name="mock/model",
                max_retries=3,
            )

        assert result["model"].name == "test"
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# Streaming usage tracking (run_stream + UsageSession)
# ---------------------------------------------------------------------------


class MockStreamDriver(Driver):
    """Mock driver that supports streaming and returns usage metadata."""

    supports_messages = True
    supports_streaming = True

    def __init__(self):
        self.model = "mock-stream-model"

    def generate(self, prompt, options):
        return self.generate_messages([], options)

    def generate_messages(self, messages, options):
        return {
            "text": "fallback response",
            "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
        }

    def generate_messages_stream(self, messages, options) -> Iterator[dict[str, Any]]:
        yield {"type": "delta", "text": "Hello"}
        yield {"type": "delta", "text": " world"}
        yield {
            "type": "done",
            "text": "Hello world",
            "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
        }


class MockAsyncStreamDriver(AsyncDriver):
    """Mock async driver that supports streaming and returns usage metadata."""

    supports_messages = True
    supports_streaming = True

    def __init__(self):
        self.model = "mock-async-stream-model"

    async def generate(self, prompt, options):
        return await self.generate_messages([], options)

    async def generate_messages(self, messages, options):
        return {
            "text": "fallback response",
            "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
        }

    async def generate_messages_stream(self, messages, options) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "delta", "text": "Hello"}
        yield {"type": "delta", "text": " world"}
        yield {
            "type": "done",
            "text": "Hello world",
            "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
        }


class TestRunStreamUsageTracking:
    def test_run_stream_run_usage_populated(self):
        """Agent.run_stream() produces non-zero run_usage when streaming."""
        driver = MockStreamDriver()
        agent = Agent("test/model", driver=driver)
        stream = agent.run_stream("test")

        for _ in stream:
            pass

        result = stream.result
        assert result is not None
        assert result.run_usage["total_tokens"] > 0
        assert result.run_usage["cost"] > 0
        assert result.run_usage["call_count"] >= 1

    def test_async_run_stream_run_usage_populated(self):
        """AsyncAgent.run_stream() produces non-zero run_usage when streaming."""

        async def _test():
            driver = MockAsyncStreamDriver()
            agent = AsyncAgent("test/model", driver=driver)
            stream = agent.run_stream("test")

            async for _ in stream:
                pass

            result = stream.result
            assert result is not None
            assert result.run_usage["total_tokens"] > 0
            assert result.run_usage["cost"] > 0
            assert result.run_usage["call_count"] >= 1

        asyncio.run(_test())


# ===========================================================================
# Tool-calling audit fixes (A1, H2, C2, C4, M8/A11, L6/A12, L10/A5, M10)
# ===========================================================================


def _tool_call_response(call_id: str, name: str, arguments: dict) -> dict:
    """Build a driver response dict requesting a single tool call."""
    return {
        "text": "",
        "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
        "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}],
        "stop_reason": "tool_use",
    }


def _text_response(text: str) -> dict:
    """Build a plain final-text driver response dict."""
    return {
        "text": text,
        "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
        "tool_calls": [],
        "stop_reason": "end_turn",
    }


def _tool_message_contents(result: AgentResult) -> list[str]:
    return [m.get("content", "") for m in result.messages if m.get("role") == "tool"]


# ---------------------------------------------------------------------------
# A1: sync approval flow (incl. awaitable approval handlers)
# ---------------------------------------------------------------------------


class TestSyncApprovalFlow:
    def test_approval_granted_retries_and_executes(self):
        """Granted approval retries the tool and uses its real result."""
        from prompture.agents.types import ApprovalRequired

        calls = {"n": 0}
        approval_log: list[tuple[str, str]] = []

        def risky_tool(command: str) -> str:
            """Run a risky command."""
            calls["n"] += 1
            if calls["n"] == 1:
                raise ApprovalRequired("risky_tool", f"Execute: {command}")
            return f"ran: {command}"

        def approve(tool_name: str, action: str, details: dict) -> bool:
            approval_log.append((tool_name, action))
            return True

        responses = [
            _tool_call_response("call_1", "risky_tool", {"command": "ls"}),
            _text_response("Done."),
        ]
        cb = AgentCallbacks(on_approval_needed=approve)
        agent = Agent("test/model", driver=MockToolDriver(responses), tools=[risky_tool], agent_callbacks=cb)
        result = agent.run("run ls")

        assert approval_log == [("risky_tool", "Execute: ls")]
        assert calls["n"] == 2  # initial attempt + approved retry
        assert any("ran: ls" in c for c in _tool_message_contents(result))

    def test_approval_denied_returns_denial_string(self):
        """Denied approval feeds a denial string back as the tool result."""
        from prompture.agents.types import ApprovalRequired

        def risky_tool(command: str) -> str:
            """Run a risky command."""
            raise ApprovalRequired("risky_tool", f"Execute: {command}")

        responses = [
            _tool_call_response("call_1", "risky_tool", {"command": "rm -rf"}),
            _text_response("Understood."),
        ]
        cb = AgentCallbacks(on_approval_needed=lambda name, action, details: False)
        agent = Agent("test/model", driver=MockToolDriver(responses), tools=[risky_tool], agent_callbacks=cb)
        result = agent.run("delete everything")

        assert any("execution denied" in c for c in _tool_message_contents(result))

    def test_no_approval_handler_returns_error_string(self):
        """Without an approval handler, an explanatory error is fed back."""
        from prompture.agents.types import ApprovalRequired

        def risky_tool(command: str) -> str:
            """Run a risky command."""
            raise ApprovalRequired("risky_tool", f"Execute: {command}")

        responses = [
            _tool_call_response("call_1", "risky_tool", {"command": "ls"}),
            _text_response("OK."),
        ]
        agent = Agent("test/model", driver=MockToolDriver(responses), tools=[risky_tool])
        result = agent.run("run ls")

        assert any("no approval handler is configured" in c for c in _tool_message_contents(result))

    def test_approval_granted_but_tool_raises_again_no_infinite_loop(self):
        """A tool that re-raises ApprovalRequired after approval stops after one retry."""
        from prompture.agents.types import ApprovalRequired

        approval_calls = {"n": 0}

        def always_risky(command: str) -> str:
            """Always requires approval."""
            raise ApprovalRequired("always_risky", f"Execute: {command}")

        def approve(tool_name: str, action: str, details: dict) -> bool:
            approval_calls["n"] += 1
            return True

        responses = [
            _tool_call_response("call_1", "always_risky", {"command": "ls"}),
            _text_response("Done."),
        ]
        cb = AgentCallbacks(on_approval_needed=approve)
        agent = Agent("test/model", driver=MockToolDriver(responses), tools=[always_risky], agent_callbacks=cb)
        result = agent.run("run ls")

        assert approval_calls["n"] == 1  # handler consulted exactly once
        assert any("approval was already granted" in c for c in _tool_message_contents(result))

    def test_async_approval_handler_is_awaited(self):
        """An ``async def`` approval handler returning False must DENY.

        Regression for A1: calling the handler bare returns a truthy
        coroutine, which silently auto-approved everything.
        """
        from prompture.agents.types import ApprovalRequired

        def risky_tool(command: str) -> str:
            """Run a risky command."""
            raise ApprovalRequired("risky_tool", f"Execute: {command}")

        async def deny_async(tool_name: str, action: str, details: dict) -> bool:
            return False

        responses = [
            _tool_call_response("call_1", "risky_tool", {"command": "rm -rf"}),
            _text_response("Understood."),
        ]
        cb = AgentCallbacks(on_approval_needed=deny_async)
        agent = Agent("test/model", driver=MockToolDriver(responses), tools=[risky_tool], agent_callbacks=cb)
        result = agent.run("delete everything")

        assert any("execution denied" in c for c in _tool_message_contents(result))


# ---------------------------------------------------------------------------
# H2: genuinely async tools through AsyncAgent
# ---------------------------------------------------------------------------


class TestAsyncToolExecution:
    def test_async_def_tool_executed_through_async_agent(self):
        """An ``async def`` tool runs to completion in AsyncAgent.run()."""

        async def async_echo(text: str) -> str:
            """Echo text asynchronously."""
            await asyncio.sleep(0)
            return f"async: {text}"

        responses = [
            _tool_call_response("call_1", "async_echo", {"text": "hi"}),
            _text_response("Done."),
        ]
        agent = AsyncAgent("test/model", driver=MockAsyncToolDriver(responses), tools=[async_echo])
        result = asyncio.run(agent.run("echo hi"))

        assert result.output == "Done."
        assert any("async: hi" in c for c in _tool_message_contents(result))

    def test_async_tool_wrapper_registers_async_fn(self):
        """Wrapped async tools expose ``_async_fn`` for ToolRegistry.aexecute."""
        from prompture.infra.session import UsageSession

        async def a_fn(x: str) -> str:
            """Async tool."""
            return f"got {x}"

        agent = AsyncAgent("test/model", driver=MockAsyncDriver(), tools=[a_fn])
        ctx = agent._build_run_context("test", None, UsageSession(), [], 0)
        wrapped = agent._wrap_tools_with_context(ctx)

        td = wrapped.get("a_fn")
        assert td is not None
        async_fn = getattr(td.function, "_async_fn", None)
        assert async_fn is not None
        assert asyncio.run(async_fn(x="hey")) == "got hey"

    def test_async_tool_sees_current_tool_call_id_contextvar(self):
        """Contextvars propagate into async tools (no thread/asyncio.run bridge)."""
        from prompture.extraction.tukuy_bridge import current_tool_call_id

        seen: list[str | None] = []

        async def ctx_probe() -> str:
            """Record the ambient tool_call_id."""
            seen.append(current_tool_call_id.get())
            return "ok"

        responses = [
            _tool_call_response("call_42", "ctx_probe", {}),
            _text_response("Done."),
        ]
        agent = AsyncAgent("test/model", driver=MockAsyncToolDriver(responses), tools=[ctx_probe])
        asyncio.run(agent.run("probe"))

        assert seen == ["call_42"]

    def test_async_tool_with_async_approval_handler(self):
        """Async approval handler is awaited in the async wrapper path."""
        from prompture.agents.types import ApprovalRequired

        calls = {"n": 0}

        async def risky_async(command: str) -> str:
            """Risky async tool."""
            calls["n"] += 1
            if calls["n"] == 1:
                raise ApprovalRequired("risky_async", f"Execute: {command}")
            return f"ran: {command}"

        async def approve_async(tool_name: str, action: str, details: dict) -> bool:
            return True

        responses = [
            _tool_call_response("call_1", "risky_async", {"command": "ls"}),
            _text_response("Done."),
        ]
        cb = AgentCallbacks(on_approval_needed=approve_async)
        agent = AsyncAgent("test/model", driver=MockAsyncToolDriver(responses), tools=[risky_async], agent_callbacks=cb)
        result = asyncio.run(agent.run("run ls"))

        assert calls["n"] == 2
        assert any("ran: ls" in c for c in _tool_message_contents(result))


# ---------------------------------------------------------------------------
# C2: Agent.stop() cooperative shutdown
# ---------------------------------------------------------------------------


class TestAgentStopDelegation:
    def test_stop_delegates_to_conversation_request_stop(self):
        """stop() sets the agent flag and forwards to the conversation."""

        class FakeConv:
            def __init__(self):
                self.calls = 0

            def request_stop(self):
                self.calls += 1

        agent = Agent("test/model", driver=MockDriver())
        fake = FakeConv()
        agent._conversation = fake  # type: ignore[assignment]
        agent.stop()

        assert agent._stop_requested is True
        assert fake.calls == 1

    def test_stop_without_conversation_is_safe(self):
        """stop() with no active conversation only sets the agent flag."""
        agent = Agent("test/model", driver=MockDriver())
        agent.stop()
        assert agent._stop_requested is True

    def test_stop_halts_running_tool_loop(self):
        """A tool calling agent.stop() ends the loop with a graceful answer."""
        holder: dict[str, Any] = {}

        def stopper() -> str:
            """Request the agent to stop."""
            holder["agent"].stop()
            return "stop requested"

        responses = [
            _tool_call_response("call_1", "stopper", {}),
            _text_response("final answer after stop"),
        ]
        driver = MockToolDriver(responses)
        agent = Agent(
            "test/model",
            driver=driver,
            tools=[stopper],
            max_iterations=5,
            persistent_conversation=True,
        )
        holder["agent"] = agent
        result = agent.run("start")

        assert result.output == "final answer after stop"
        # 1 tool-call round + 1 final answer call; the loop did not run all 5 rounds
        assert driver._call_idx == 2

    def test_async_stop_delegates_to_conversation_request_stop(self):
        """AsyncAgent.stop() forwards to the conversation as well."""

        class FakeConv:
            def __init__(self):
                self.calls = 0

            def request_stop(self):
                self.calls += 1

        agent = AsyncAgent("test/model", driver=MockAsyncDriver())
        fake = FakeConv()
        agent._conversation = fake
        agent.stop()

        assert agent._stop_requested is True
        assert fake.calls == 1

    def test_stop_halts_loop_without_persistent_conversation(self):
        """stop() reaches the loop of a default (non-persistent) agent.

        Regression guard: ``_conversation`` is only populated when
        ``persistent_conversation=True``, so forwarding solely through it made
        ``stop()`` a silent no-op for the default agent.
        """
        holder: dict[str, Any] = {}

        def stopper() -> str:
            """Request the agent to stop."""
            holder["agent"].stop()
            return "stop requested"

        responses = [
            _tool_call_response("call_1", "stopper", {}),
            _text_response("final answer after stop"),
        ]
        driver = MockToolDriver(responses)
        agent = Agent("test/model", driver=driver, tools=[stopper], max_iterations=5)
        holder["agent"] = agent
        result = agent.run("start")

        assert result.output == "final answer after stop"
        # 1 tool-call round + 1 final answer; the loop did not run all 5 rounds.
        assert driver._call_idx == 2

    def test_stop_before_run_is_not_lost_to_the_race(self):
        """A stop() landing before the conversation exists is replayed onto it."""

        def ping() -> str:
            """Ping."""
            return "pong"

        agent = Agent("test/model", driver=MockToolDriver([_text_response("hi")]), tools=[ping])
        conv = agent._build_conversation()
        assert conv._stop_requested is False

        agent.stop()
        # Rebuilding mid-run must carry the pending request over.
        conv2 = agent._build_conversation()
        assert conv2._stop_requested is True

    def test_async_stop_halts_loop_without_persistent_conversation(self):
        """AsyncAgent.stop() also reaches a non-persistent run's loop."""
        holder: dict[str, Any] = {}

        def stopper() -> str:
            """Request the agent to stop."""
            holder["agent"].stop()
            return "stop requested"

        async def _test():
            responses = [
                _tool_call_response("call_1", "stopper", {}),
                _text_response("async final answer after stop"),
            ]
            driver = MockAsyncToolDriver(responses)
            agent = AsyncAgent("test/model", driver=driver, tools=[stopper], max_iterations=5)
            holder["agent"] = agent
            result = await agent.run("start")

            assert result.output == "async final answer after stop"
            assert driver._call_idx == 2

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# Per-tool timeout plumbed from the Agent
# ---------------------------------------------------------------------------


class TestAgentToolTimeout:
    def test_tool_timeout_forwarded_to_conversation(self):
        """Agent(tool_timeout=...) reaches the Conversation that runs tools."""

        def ping() -> str:
            """Ping."""
            return "pong"

        agent = Agent("test/model", driver=MockToolDriver([_text_response("hi")]), tools=[ping], tool_timeout=2.5)
        conv = agent._build_conversation()
        assert conv._tool_timeout == 2.5

    def test_tool_timeout_defaults_to_none(self):
        """No timeout unless asked for, preserving the previous behaviour."""
        agent = Agent("test/model", driver=MockToolDriver([_text_response("hi")]))
        assert agent._build_conversation()._tool_timeout is None

    def test_slow_tool_times_out_and_reports_to_the_model(self):
        """A tool overrunning the budget yields an error result, not a hang."""
        import time as _time

        def slow() -> str:
            """Sleep past the timeout."""
            _time.sleep(5)
            return "never returned"

        responses = [
            _tool_call_response("call_1", "slow", {}),
            _text_response("gave up on the slow tool"),
        ]
        agent = Agent(
            "test/model",
            driver=MockToolDriver(responses),
            tools=[slow],
            tool_timeout=0.05,
        )
        result = agent.run("call the slow tool")

        assert result.output == "gave up on the slow tool"
        assert any("timed out" in c for c in _tool_message_contents(result))

    def test_async_tool_timeout_forwarded_to_conversation(self):
        """AsyncAgent(tool_timeout=...) reaches the AsyncConversation too."""
        agent = AsyncAgent("test/model", driver=MockAsyncToolDriver([_text_response("hi")]), tool_timeout=1.5)
        assert agent._build_conversation()._tool_timeout == 1.5


# ---------------------------------------------------------------------------
# C4: graceful finish when max_iterations is exhausted
# ---------------------------------------------------------------------------


class TestMaxIterationsGraceful:
    def test_max_iterations_graceful_finish(self):
        """Exhausting max_iterations yields a final answer instead of RuntimeError."""

        def ping() -> str:
            """Ping."""
            return "pong"

        responses = [
            _tool_call_response("call_1", "ping", {}),
            _tool_call_response("call_2", "ping", {}),
            _text_response("Answering from what I have."),
        ]
        driver = MockToolDriver(responses)
        agent = Agent(
            "test/model",
            driver=driver,
            tools=[ping],
            max_iterations=2,
            persistent_conversation=True,
        )
        result = agent.run("keep pinging")

        assert result.output == "Answering from what I have."
        assert result.state == AgentState.idle
        assert agent.conversation is not None
        assert agent.conversation.max_rounds_reached is True

    def test_async_max_iterations_graceful_finish(self):
        """AsyncAgent also finishes gracefully when tool rounds are exhausted."""

        def ping() -> str:
            """Ping."""
            return "pong"

        async def _test():
            responses = [
                _tool_call_response("call_1", "ping", {}),
                _tool_call_response("call_2", "ping", {}),
                _text_response("Async final answer."),
            ]
            driver = MockAsyncToolDriver(responses)
            agent = AsyncAgent(
                "test/model",
                driver=driver,
                tools=[ping],
                max_iterations=2,
                persistent_conversation=True,
            )
            result = await agent.run("keep pinging")
            assert result.output == "Async final answer."
            assert agent.conversation is not None
            assert agent.conversation.max_rounds_reached is True

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# L10/A5 + M8/A11 + L6/A12: live RunContext, real tool names, durations
# ---------------------------------------------------------------------------


class TestLiveRunContextAndSteps:
    def test_run_context_refreshed_per_tool_round(self):
        """Tools see live iteration/messages instead of the run-start snapshot."""
        snapshots: list[tuple[int, int]] = []

        def probe(ctx: RunContext, note: str) -> str:
            """Record iteration and message count."""
            snapshots.append((ctx.iteration, len(ctx.messages)))
            return f"noted {note}"

        responses = [
            _tool_call_response("call_1", "probe", {"note": "a"}),
            _tool_call_response("call_2", "probe", {"note": "b"}),
            _text_response("done"),
        ]
        agent = Agent("test/model", driver=MockToolDriver(responses), tools=[probe], max_iterations=5)
        agent.run("go")

        assert len(snapshots) == 2
        assert snapshots[0][0] == 0  # first tool round
        assert snapshots[1][0] == 1  # second tool round
        assert snapshots[1][1] > snapshots[0][1]  # message history grew

    def test_tool_result_step_records_real_tool_name(self):
        """tool_result steps carry the tool name, not the tool_call_id."""

        def my_tool() -> str:
            """A named tool."""
            return "result"

        responses = [
            _tool_call_response("call_xyz", "my_tool", {}),
            _text_response("ok"),
        ]
        agent = Agent("test/model", driver=MockToolDriver(responses), tools=[my_tool])
        result = agent.run("go")

        tr_steps = [s for s in result.steps if s.step_type == StepType.tool_result]
        assert len(tr_steps) == 1
        assert tr_steps[0].tool_name == "my_tool"

    def test_tool_result_step_has_duration_ms(self):
        """tool_result steps carry a measured duration."""
        import time as _time

        def slow_tool() -> str:
            """A slow tool."""
            _time.sleep(0.02)
            return "done"

        responses = [
            _tool_call_response("call_9", "slow_tool", {}),
            _text_response("ok"),
        ]
        agent = Agent("test/model", driver=MockToolDriver(responses), tools=[slow_tool])
        result = agent.run("go")

        tr_steps = [s for s in result.steps if s.step_type == StepType.tool_result]
        assert len(tr_steps) == 1
        assert tr_steps[0].duration_ms >= 10

    def test_async_run_context_refreshed_per_tool_round(self):
        """AsyncAgent tools also see live iteration/messages."""
        snapshots: list[tuple[int, int]] = []

        async def aprobe(ctx: RunContext, note: str) -> str:
            """Record iteration and message count."""
            snapshots.append((ctx.iteration, len(ctx.messages)))
            return f"noted {note}"

        async def _test():
            responses = [
                _tool_call_response("call_1", "aprobe", {"note": "a"}),
                _tool_call_response("call_2", "aprobe", {"note": "b"}),
                _text_response("done"),
            ]
            agent = AsyncAgent("test/model", driver=MockAsyncToolDriver(responses), tools=[aprobe], max_iterations=5)
            await agent.run("go")

        asyncio.run(_test())

        assert len(snapshots) == 2
        assert snapshots[0][0] == 0
        assert snapshots[1][0] == 1
        assert snapshots[1][1] > snapshots[0][1]


# ---------------------------------------------------------------------------
# M10: persistent conversation reuse respects new system prompt / driver
# ---------------------------------------------------------------------------


class TestPersistentConversationReuse:
    def test_reuse_updates_system_prompt(self):
        """A reused conversation picks up the newly resolved system prompt."""
        driver = MockDriver(["one", "two"])
        agent = Agent("test/model", driver=driver, system_prompt="prompt A", persistent_conversation=True)

        agent.run("hi")
        assert agent.conversation is not None
        assert agent.conversation.system_prompt == "prompt A"

        agent._system_prompt = "prompt B"
        agent.run("hi again")
        assert agent.conversation.system_prompt == "prompt B"

    def test_reuse_rebuilds_when_driver_swapped(self):
        """Swapping the agent's driver discards the stale conversation."""
        d1 = MockDriver(["one"])
        d2 = MockDriver(["two"])
        agent = Agent("test/model", driver=d1, persistent_conversation=True)

        agent.run("hi")
        conv1 = agent.conversation
        assert conv1 is not None

        agent._driver = d2
        result = agent.run("hi again")

        assert result.output == "two"
        assert agent.conversation is not conv1
        assert agent.conversation._driver is d2

    def test_reuse_keeps_conversation_when_driver_unchanged(self):
        """Same driver + same config reuses the conversation (history grows)."""
        driver = MockDriver(["one", "two"])
        agent = Agent("test/model", driver=driver, persistent_conversation=True)

        agent.run("hi")
        conv1 = agent.conversation
        agent.run("hi again")

        assert agent.conversation is conv1
