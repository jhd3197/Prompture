"""Simulated (prompted) tool calls must appear in AgentResult.

Drivers without ``supports_tool_use`` go through
``Conversation._ask_with_simulated_tools``, which records a tool round as a
plain assistant message holding the protocol JSON plus a plain user message
holding the result.  ``Agent._extract_steps`` used to understand only the
native shape (``msg["tool_calls"]`` / ``role == "tool"``), so on every prompted
driver — LM Studio, LocalHTTP, HuggingFace, AirLLM, DeepSeek, OpenAICompatible
— a run reported zero tool activity and leaked the raw protocol JSON as the
model's answer.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from prompture.agents import Agent
from prompture.agents.types import StepType
from prompture.drivers.base import Driver


def _meta() -> dict[str, Any]:
    return {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "cost": 0.0,
        "raw_response": {},
    }


class _ScriptedDriver(Driver):
    """Replays scripted turns; declares no native tool support."""

    supports_tool_use = False

    def __init__(self, turns: list[str]) -> None:
        self._turns = list(turns)
        self._i = 0
        self.model = "scripted/prompted"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        text = self._turns[self._i] if self._i < len(self._turns) else json.dumps(
            {"type": "final_answer", "content": "done"}
        )
        self._i += 1
        return {"text": text, "meta": _meta()}


class _NativeDriver(Driver):
    """Control: the native path, which already worked."""

    supports_tool_use = True

    def __init__(self) -> None:
        self._calls = 0
        self.model = "scripted/native"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {"text": "", "meta": _meta()}

    def generate_messages_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], options: dict[str, Any]
    ) -> dict[str, Any]:
        self._calls += 1
        if self._calls == 1:
            return {
                "text": "",
                "meta": _meta(),
                "tool_calls": [{"id": "call_1", "name": "get_weather", "arguments": {"city": "Caracas"}}],
                "stop_reason": "tool_use",
            }
        return {"text": "It is sunny in Caracas.", "meta": _meta(), "tool_calls": [], "stop_reason": "end_turn"}


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"sunny in {city}"


def _agent(driver: Driver) -> Agent:
    return Agent(driver=driver, tools=[get_weather], max_iterations=5)


def _simulated_agent() -> tuple[Agent, _ScriptedDriver]:
    driver = _ScriptedDriver(
        [
            json.dumps({"type": "tool_call", "name": "get_weather", "arguments": {"city": "Caracas"}}),
            json.dumps({"type": "final_answer", "content": "It is sunny in Caracas."}),
        ]
    )
    return _agent(driver), driver


def test_simulated_tool_call_is_reported():
    agent, _ = _simulated_agent()

    result = agent.run("What is the weather in Caracas?")

    assert result.all_tool_calls == [
        {"name": "get_weather", "arguments": {"city": "Caracas"}, "id": ""}
    ]


def test_simulated_run_records_call_and_result_steps():
    agent, _ = _simulated_agent()

    result = agent.run("What is the weather in Caracas?")

    kinds = [s.step_type for s in result.steps]
    assert StepType.tool_call in kinds
    assert StepType.tool_result in kinds

    call = next(s for s in result.steps if s.step_type is StepType.tool_call)
    assert call.tool_name == "get_weather"
    assert call.tool_args == {"city": "Caracas"}

    res = next(s for s in result.steps if s.step_type is StepType.tool_result)
    assert res.tool_name == "get_weather"
    assert "Caracas" in res.content


def test_protocol_json_is_not_leaked_as_model_output():
    """The raw {"type": "tool_call", ...} must never read as the answer."""
    agent, _ = _simulated_agent()

    result = agent.run("What is the weather in Caracas?")

    assert result.output_text == "It is sunny in Caracas."
    for step in result.steps:
        if step.step_type is StepType.output:
            assert '"type"' not in step.content
            assert "tool_call" not in step.content


def test_simulated_and_native_agree_on_shape():
    """Both dialects must produce the same step vocabulary for one run."""
    sim_agent, _ = _simulated_agent()
    sim = sim_agent.run("What is the weather in Caracas?")
    native = _agent(_NativeDriver()).run("What is the weather in Caracas?")

    def kinds(r: Any) -> set[StepType]:
        return {s.step_type for s in r.steps}

    assert StepType.tool_call in kinds(sim) and StepType.tool_call in kinds(native)
    assert StepType.tool_result in kinds(sim) and StepType.tool_result in kinds(native)
    assert [c["name"] for c in sim.all_tool_calls] == [c["name"] for c in native.all_tool_calls]
    assert [c["arguments"] for c in sim.all_tool_calls] == [c["arguments"] for c in native.all_tool_calls]


def test_ordinary_prose_is_still_output_not_a_tool_call():
    """A plain answer must not be mistaken for a protocol payload."""
    driver = _ScriptedDriver([json.dumps({"type": "final_answer", "content": "Just prose."})])

    result = _agent(driver).run("hi")

    assert result.all_tool_calls == []
    assert result.output_text == "Just prose."


@pytest.mark.parametrize(
    "payload",
    [
        '{"not_a_tool": true}',
        '{"name": "get_weather"}',  # no arguments
        '{"type": "tool_call", "arguments": {}}',  # no name
        "{ this is not json",
        "",
    ],
)
def test_non_tool_payloads_do_not_register_calls(payload):
    from prompture.agents.agent import _parse_simulated_tool_call

    assert _parse_simulated_tool_call(payload) is None


def test_user_prompt_is_never_mistaken_for_a_tool_result():
    """Only a user message *following* a simulated call is a result."""
    driver = _ScriptedDriver([json.dumps({"type": "final_answer", "content": "hello"})])

    result = _agent(driver).run("this is my prompt")

    assert not any(s.step_type is StepType.tool_result for s in result.steps)
