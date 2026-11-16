"""Conformance: every Anthropic-wire driver (Claude direct, on Azure, on Vertex,
on Bedrock), the same scenarios, the same results."""

from __future__ import annotations

from typing import Any

import pytest
from conformance_harness import (
    ANTHROPIC_WIRE,
    HISTORY,
    ONE_OF_TOOL,
    SIMPLE,
    WEATHER_TOOL,
    Reply,
    Wire,
    call_tools,
    image_messages,
)
from known_gaps import KNOWN_GAPS

PARIS = Reply(tool_calls=[("toolu_1", "get_weather", {"city": "Paris"})])


def targets(test_name: str) -> list[Any]:
    out = []
    for target in ANTHROPIC_WIRE:
        reason = KNOWN_GAPS.get((target.id, test_name))
        marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
        out.append(pytest.param(target, id=target.id, marks=marks))
    return out


def _plain(value: Any) -> Any:
    """Drop prompt-cache markers, which are placement detail, not content."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    return "".join(block.get("text", "") for block in system)


def _blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message["content"]
    return [{"type": "text", "text": content}] if isinstance(content, str) else _plain(content)


@pytest.mark.parametrize("target", targets("test_tool_history_becomes_anthropic_blocks"))
def test_tool_history_becomes_anthropic_blocks(target, monkeypatch):
    wire = Wire(PARIS)
    driver = target.build(monkeypatch, wire)
    call_tools(target, driver, HISTORY, [WEATHER_TOOL])

    payload = wire.payload
    assert _system_text(payload["system"]) == "You are terse."
    assert [m["role"] for m in payload["messages"]] == ["user", "assistant", "user", "user"] or [
        m["role"] for m in payload["messages"]
    ] == ["user", "assistant", "user"]
    assistant = _blocks(payload["messages"][1])
    assert assistant == [{"type": "tool_use", "id": "call_prev", "name": "get_weather", "input": {"city": "Paris"}}]
    result = _blocks(payload["messages"][2])[0]
    assert (result["type"], result["tool_use_id"]) == ("tool_result", "call_prev")
    assert "18C and sunny" in str(result["content"])


@pytest.mark.parametrize("target", targets("test_tools_become_input_schema"))
def test_tools_become_input_schema(target, monkeypatch):
    wire = Wire(PARIS)
    driver = target.build(monkeypatch, wire)
    call_tools(target, driver, SIMPLE, [WEATHER_TOOL])

    fn = WEATHER_TOOL["function"]
    assert _plain(wire.payload["tools"]) == [
        {"name": fn["name"], "description": fn["description"], "input_schema": fn["parameters"]}
    ]


@pytest.mark.parametrize("target", targets("test_top_level_one_of_schema_is_flattened"))
def test_top_level_one_of_schema_is_flattened(target, monkeypatch):
    wire = Wire(Reply(tool_calls=[("toolu_1", "make_video", {"prompt": "a cat"})]))
    driver = target.build(monkeypatch, wire)
    call_tools(target, driver, SIMPLE, [ONE_OF_TOOL])

    schema = wire.payload["tools"][0]["input_schema"]
    assert schema["type"] == "object"
    assert "oneOf" not in schema


@pytest.mark.parametrize("target", targets("test_forced_tool_choice_is_translated"))
def test_forced_tool_choice_is_translated(target, monkeypatch):
    wire = Wire(PARIS)
    driver = target.build(monkeypatch, wire)
    call_tools(target, driver, SIMPLE, [WEATHER_TOOL], {"tool_choice": {"name": "get_weather"}})

    assert wire.payload["tool_choice"] == {"type": "tool", "name": "get_weather"}


@pytest.mark.parametrize("target", targets("test_required_tool_choice_is_translated"))
def test_required_tool_choice_is_translated(target, monkeypatch):
    wire = Wire(PARIS)
    driver = target.build(monkeypatch, wire)
    call_tools(target, driver, SIMPLE, [WEATHER_TOOL], {"tool_choice": "required"})

    assert wire.payload["tool_choice"] == {"type": "any"}


@pytest.mark.parametrize("target", targets("test_image_on_a_tool_turn_is_converted"))
def test_image_on_a_tool_turn_is_converted(target, monkeypatch):
    wire = Wire(PARIS)
    driver = target.build(monkeypatch, wire)
    call_tools(target, driver, image_messages(), [WEATHER_TOOL])

    blocks = _blocks(wire.payload["messages"][0])
    assert blocks[0] == {"type": "text", "text": "What is this?"}
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
    }


@pytest.mark.parametrize("target", targets("test_tool_call_is_normalized"))
def test_tool_call_is_normalized(target, monkeypatch):
    wire = Wire(PARIS)
    driver = target.build(monkeypatch, wire)
    result = call_tools(target, driver, SIMPLE, [WEATHER_TOOL])

    assert result["text"] == ""
    assert result["tool_calls"] == [{"id": "toolu_1", "name": "get_weather", "arguments": {"city": "Paris"}}]
    assert result["stop_reason"] == "tool_use"
    assert result["meta"]["raw_stop_reason"] == "tool_use"
    meta = result["meta"]
    assert (meta["prompt_tokens"], meta["completion_tokens"], meta["total_tokens"]) == (11, 7, 18)
    assert isinstance(meta["cost"], float)


@pytest.mark.parametrize("target", targets("test_stop_reasons_use_the_shared_vocabulary"))
def test_stop_reasons_use_the_shared_vocabulary(target, monkeypatch):
    wire = Wire(Reply(text="Done.", finish="stop_sequence"))
    driver = target.build(monkeypatch, wire)
    result = call_tools(target, driver, SIMPLE, [WEATHER_TOOL])

    assert result["text"] == "Done."
    assert result["stop_reason"] == "end_turn"
    assert result["meta"]["raw_stop_reason"] == "stop_sequence"


@pytest.mark.parametrize("target", targets("test_thinking_is_returned_as_reasoning"))
def test_thinking_is_returned_as_reasoning(target, monkeypatch):
    wire = Wire(Reply(reasoning="Paris is a city.", tool_calls=[("toolu_1", "get_weather", {"city": "Paris"})]))
    driver = target.build(monkeypatch, wire)
    result = call_tools(target, driver, SIMPLE, [WEATHER_TOOL])

    assert result["reasoning_content"] == "Paris is a city."
