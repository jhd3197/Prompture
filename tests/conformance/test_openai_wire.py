"""Conformance: every OpenAI-wire driver, the same scenarios, the same results."""

from __future__ import annotations

from typing import Any

import pytest
from conformance_harness import (
    HISTORY,
    ONE_OF_TOOL,
    OPENAI_WIRE,
    SIMPLE,
    WEATHER_TOOL,
    Reply,
    Target,
    Wire,
    call_tools,
    image_messages,
    run,
    stream_tools,
)
from known_gaps import KNOWN_GAPS

PARIS = Reply(tool_calls=[("call_1", "get_weather", '{"city": "Paris"}')])


def targets(test_name: str, *, only: Any = None) -> list[Any]:
    out = []
    for target in OPENAI_WIRE:
        if only is not None and not only(target):
            continue
        reason = KNOWN_GAPS.get((target.id, test_name))
        marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
        out.append(pytest.param(target, id=target.id, marks=marks))
    return out


def _has(flag: str):
    def check(target: Target) -> bool:
        try:
            return bool(getattr(target.load(), flag, False))
        except BaseException:  # optional SDK missing — the test itself will skip
            return True

    return check


def _strip_stream(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in ("stream", "stream_options")}


# ---------------------------------------------------------------------------
# Request shaping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", targets("test_request_carries_tools_and_history"))
def test_request_carries_tools_and_history(target, monkeypatch):
    wire = Wire(PARIS)
    driver = target.build(monkeypatch, wire)
    call_tools(target, driver, HISTORY, [WEATHER_TOOL])

    payload = wire.payload
    assert payload["tools"] == [WEATHER_TOOL]
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "user"]
    assistant = payload["messages"][2]
    assert assistant["tool_calls"][0]["id"] == "call_prev"
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    assert payload["messages"][3]["tool_call_id"] == "call_prev"


@pytest.mark.parametrize("target", targets("test_image_on_a_tool_turn_is_converted", only=_has("supports_vision")))
def test_image_on_a_tool_turn_is_converted(target, monkeypatch):
    wire = Wire(PARIS)
    driver = target.build(monkeypatch, wire)
    call_tools(target, driver, image_messages(), [WEATHER_TOOL])

    blocks = wire.payload["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "image_url"]
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("target", targets("test_forced_tool_choice_is_translated"))
def test_forced_tool_choice_is_translated(target, monkeypatch):
    wire = Wire(PARIS)
    driver = target.build(monkeypatch, wire)
    call_tools(target, driver, SIMPLE, [WEATHER_TOOL], {"tool_choice": {"name": "get_weather"}})

    assert wire.payload["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


@pytest.mark.parametrize("target", targets("test_required_tool_choice_reaches_the_provider"))
def test_required_tool_choice_reaches_the_provider(target, monkeypatch):
    wire = Wire(PARIS)
    driver = target.build(monkeypatch, wire)
    call_tools(target, driver, SIMPLE, [WEATHER_TOOL], {"tool_choice": "required"})

    assert wire.payload["tool_choice"] == target.required_choice


@pytest.mark.parametrize("target", targets("test_top_level_one_of_schema_is_flattened"))
def test_top_level_one_of_schema_is_flattened(target, monkeypatch):
    wire = Wire(Reply(tool_calls=[("call_1", "make_video", '{"prompt": "a cat"}')]))
    driver = target.build(monkeypatch, wire)
    call_tools(target, driver, SIMPLE, [ONE_OF_TOOL])

    params = wire.payload["tools"][0]["function"]["parameters"]
    assert params["type"] == "object"
    assert "oneOf" not in params
    assert set(params["properties"]) == {"prompt", "image_url"}


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", targets("test_tool_call_is_normalized"))
def test_tool_call_is_normalized(target, monkeypatch):
    wire = Wire(PARIS)
    driver = target.build(monkeypatch, wire)
    result = call_tools(target, driver, SIMPLE, [WEATHER_TOOL])

    assert result["text"] == ""
    assert result["tool_calls"] == [{"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}}]
    assert result["stop_reason"] == "tool_use"
    assert result["meta"]["raw_stop_reason"] == "tool_calls"
    meta = result["meta"]
    assert (meta["prompt_tokens"], meta["completion_tokens"], meta["total_tokens"]) == (11, 7, 18)
    assert isinstance(meta["cost"], float)
    assert "raw_response" in meta


@pytest.mark.parametrize("target", targets("test_plain_answer_ends_the_turn"))
def test_plain_answer_ends_the_turn(target, monkeypatch):
    wire = Wire(Reply(text="18C and sunny"))
    driver = target.build(monkeypatch, wire)
    result = call_tools(target, driver, SIMPLE, [WEATHER_TOOL])

    assert result["text"] == "18C and sunny"
    assert result["tool_calls"] == []
    assert result["stop_reason"] == "end_turn"


@pytest.mark.parametrize("target", targets("test_parallel_tool_calls_keep_their_order"))
def test_parallel_tool_calls_keep_their_order(target, monkeypatch):
    reply = Reply(
        tool_calls=[
            ("call_1", "get_weather", '{"city": "Paris"}'),
            ("call_2", "get_weather", '{"city": "Rome"}'),
        ]
    )
    wire = Wire(reply)
    driver = target.build(monkeypatch, wire)
    result = call_tools(target, driver, SIMPLE, [WEATHER_TOOL])

    assert [(tc["id"], tc["arguments"]["city"]) for tc in result["tool_calls"]] == [
        ("call_1", "Paris"),
        ("call_2", "Rome"),
    ]


@pytest.mark.parametrize("target", targets("test_arguments_already_a_dict_are_accepted"))
def test_arguments_already_a_dict_are_accepted(target, monkeypatch):
    wire = Wire(Reply(tool_calls=[("call_1", "get_weather", {"city": "Paris"})]))
    driver = target.build(monkeypatch, wire)
    result = call_tools(target, driver, SIMPLE, [WEATHER_TOOL])

    assert result["tool_calls"][0]["arguments"] == {"city": "Paris"}
    assert "arguments_error" not in result["tool_calls"][0]


@pytest.mark.parametrize("target", targets("test_malformed_arguments_are_reported"))
def test_malformed_arguments_are_reported(target, monkeypatch):
    wire = Wire(Reply(tool_calls=[("call_1", "get_weather", '{"city": "Par')]))
    driver = target.build(monkeypatch, wire)
    result = call_tools(target, driver, SIMPLE, [WEATHER_TOOL])

    call = result["tool_calls"][0]
    assert call["arguments"] == {}
    assert "get_weather" in call["arguments_error"]


@pytest.mark.parametrize("target", targets("test_truncated_arguments_name_max_tokens"))
def test_truncated_arguments_name_max_tokens(target, monkeypatch):
    wire = Wire(Reply(tool_calls=[("call_1", "get_weather", '{"city": "Par')], finish="length"))
    driver = target.build(monkeypatch, wire)
    result = call_tools(target, driver, SIMPLE, [WEATHER_TOOL])

    assert "max_tokens" in result["tool_calls"][0]["arguments_error"]
    assert result["stop_reason"] == "max_tokens"


@pytest.mark.parametrize("target", targets("test_missing_tool_call_id_gets_one"))
def test_missing_tool_call_id_gets_one(target, monkeypatch):
    wire = Wire(Reply(tool_calls=[(None, "get_weather", '{"city": "Paris"}')]))
    driver = target.build(monkeypatch, wire)
    result = call_tools(target, driver, SIMPLE, [WEATHER_TOOL])

    assert isinstance(result["tool_calls"][0]["id"], str) and result["tool_calls"][0]["id"]


@pytest.mark.parametrize("target", targets("test_reasoning_content_is_returned"))
def test_reasoning_content_is_returned(target, monkeypatch):
    wire = Wire(Reply(reasoning="Paris is a city.", tool_calls=[("call_1", "get_weather", '{"city": "Paris"}')]))
    driver = target.build(monkeypatch, wire)
    result = call_tools(target, driver, SIMPLE, [WEATHER_TOOL])

    assert result["reasoning_content"] == "Paris is a city."


# ---------------------------------------------------------------------------
# Streaming must match the buffered path
# ---------------------------------------------------------------------------

_STREAMS = _has("supports_streaming_tool_use")


@pytest.mark.parametrize("target", targets("test_stream_yields_the_same_tool_call", only=_STREAMS))
def test_stream_yields_the_same_tool_call(target, monkeypatch):
    wire = Wire(PARIS)
    driver = target.build(monkeypatch, wire)
    events = stream_tools(target, driver, SIMPLE, [WEATHER_TOOL])

    stops = [e for e in events if e.event_type == "tool_use_stop"]
    assert [(s.id, s.name, s.input) for s in stops] == [("call_1", "get_weather", {"city": "Paris"})]
    final = events[-1]
    assert final.event_type == "message_stop"
    assert final.stop_reason == "tool_use"
    assert (final.usage["prompt_tokens"], final.usage["completion_tokens"]) == (11, 7)


@pytest.mark.parametrize("target", targets("test_stream_sends_the_same_request", only=_STREAMS))
def test_stream_sends_the_same_request(target, monkeypatch):
    """Whatever shaping the buffered tool call gets, the streamed one gets too."""
    options = {"tool_choice": "required", "temperature": 1.7, "max_tokens": 900, "effort": "low"}

    buffered = Wire(PARIS)
    call_tools(target, target.build(monkeypatch, buffered), image_messages(), [ONE_OF_TOOL], options)
    streamed = Wire(PARIS)
    stream_tools(target, target.build(monkeypatch, streamed), image_messages(), [ONE_OF_TOOL], options)

    assert _strip_stream(streamed.payload) == _strip_stream(buffered.payload)


@pytest.mark.parametrize("target", targets("test_stream_reports_malformed_arguments", only=_STREAMS))
def test_stream_reports_malformed_arguments(target, monkeypatch):
    wire = Wire(Reply(tool_calls=[("call_1", "get_weather", '{"city": "Par')]))
    driver = target.build(monkeypatch, wire)
    events = stream_tools(target, driver, SIMPLE, [WEATHER_TOOL])

    stop = next(e for e in events if e.event_type == "tool_use_stop")
    assert stop.input == {}
    assert stop.arguments_error and "get_weather" in stop.arguments_error


@pytest.mark.parametrize(
    "target",
    targets(
        "test_plain_stream_converts_images",
        only=lambda t: _has("supports_streaming")(t) and _has("supports_vision")(t),
    ),
)
def test_plain_stream_converts_images(target, monkeypatch):
    wire = Wire(Reply(text="A pixel."))
    driver = target.build(monkeypatch, wire)
    chunks = run(driver.generate_messages_stream(image_messages(), dict(target.options)))

    assert [b["type"] for b in wire.payload["messages"][0]["content"]] == ["text", "image_url"]
    assert chunks[-1]["type"] == "done"
    assert chunks[-1]["text"] == "A pixel."
