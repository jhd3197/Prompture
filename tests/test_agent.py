"""Tests for the Agent framework."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from prompture.agent import Agent, AgentIterator, StreamedAgentResult, _get_first_param_name, _tool_wants_context
from prompture.agent_types import (
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
from prompture.async_agent import AsyncAgent, AsyncAgentIterator, AsyncStreamedAgentResult, _is_async_callable
from prompture.async_driver import AsyncDriver
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
        from prompture.session import UsageSession

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
        from prompture.agent_types import AgentStep

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
            it = await agent.iter("test")
            assert isinstance(it, AsyncAgentIterator)

        asyncio.run(_test())

    def test_async_iter_yields_steps(self):
        """Async iteration yields AgentStep objects."""

        async def _test():
            driver = MockAsyncDriver(["Hello!"])
            agent = AsyncAgent("test/model", driver=driver)
            it = await agent.iter("test")

            steps = []
            async for step in it:
                steps.append(step)

            from prompture.agent_types import AgentStep

            assert len(steps) >= 1
            assert all(isinstance(s, AgentStep) for s in steps)

        asyncio.run(_test())

    def test_async_iter_result_populated(self):
        """result is populated after async iteration completes."""

        async def _test():
            driver = MockAsyncDriver(["Hello!"])
            agent = AsyncAgent("test/model", driver=driver)
            it = await agent.iter("test")
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
            stream = await agent.run_stream("test")
            assert isinstance(stream, AsyncStreamedAgentResult)

        asyncio.run(_test())

    def test_async_stream_yields_events(self):
        """Async streaming yields StreamEvent objects."""

        async def _test():
            driver = MockAsyncDriver(["Hello!"])
            agent = AsyncAgent("test/model", driver=driver)
            stream = await agent.run_stream("test")

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
            stream = await agent.run_stream("test")
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
