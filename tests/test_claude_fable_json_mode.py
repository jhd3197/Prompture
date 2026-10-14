"""JSON mode and effort on the models that refuse forced tools and sampling settings.

Claude Fable 5.1, Mythos 5.1, Opus 5.5 and Sonnet 5.5 answer a forced ``tool_choice``
(``{"type": "tool"}``, how JSON mode used to enforce a schema) with a 400, and the
current families 400 on any explicit ``temperature``. Every JSON-mode call to them failed
before the model ran: a chess harness turn, a coach reply, any ``extract_with_model``.

On those models JSON mode now asks for structured output (``output_config.format``),
and ``effort`` reaches ``output_config.effort``. Older models keep the forced tool.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

SCHEMA = {"type": "object", "properties": {"move": {"type": "string", "enum": ["e4", "d4"]}}, "required": ["move"]}


class _StubUsage:
    input_tokens = 10
    output_tokens = 5
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _Block:
    def __init__(self, type_: str, **fields: Any) -> None:
        self.type = type_
        for key, value in fields.items():
            setattr(self, key, value)


class _StubResponse:
    def __init__(self, content: list[_Block]) -> None:
        self.usage = _StubUsage()
        self.content = content
        self.stop_reason = "end_turn"

    def keys(self) -> list[str]:
        return []

    def __getitem__(self, _: str) -> Any:
        raise KeyError


class _Messages:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> _StubResponse:
        self.last_kwargs = kwargs
        if "tool_choice" in kwargs:
            return _StubResponse([_Block("tool_use", input={"move": "d4"})])
        # A thinking model puts its (empty, omitted) thinking block before the answer.
        return _StubResponse([_Block("thinking", thinking=""), _Block("text", text='{"move": "e4"}')])

    @contextmanager
    def stream(self, **kwargs: Any):
        self.last_kwargs = kwargs
        events = [
            SimpleNamespace(type="message_start", message=SimpleNamespace(usage={"input_tokens": 10})),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="thinking_delta", thinking="Consider the center"),
            ),
        ]
        if "tool_choice" in kwargs:
            events += [
                SimpleNamespace(
                    type="content_block_start",
                    index=1,
                    content_block=SimpleNamespace(type="tool_use", name="extract_json"),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    index=1,
                    delta=SimpleNamespace(type="input_json_delta", partial_json='{"move": '),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    index=1,
                    delta=SimpleNamespace(type="input_json_delta", partial_json='"d4"}'),
                ),
            ]
        else:
            events.append(
                SimpleNamespace(
                    type="content_block_delta", index=1, delta=SimpleNamespace(type="text_delta", text='{"move": "e4"}')
                )
            )
        events.append(
            SimpleNamespace(
                type="message_delta", usage={"output_tokens": 5}, delta=SimpleNamespace(stop_reason="end_turn")
            )
        )
        yield iter(events)


@pytest.fixture
def driver_for(monkeypatch: pytest.MonkeyPatch):
    from prompture.drivers import claude_driver as cd

    messages = _Messages()

    class _StubAnthropic:
        def __new__(cls, *_: Any, **__: Any) -> Any:  # type: ignore[misc]
            return type("C", (), {"messages": messages})()

    monkeypatch.setattr(cd, "anthropic", type("M", (), {"Anthropic": _StubAnthropic}))
    return lambda model: (cd.ClaudeDriver(api_key="test-key", model=model), messages)


@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5"])
def test_json_mode_uses_structured_output_where_forced_tools_400(driver_for, model) -> None:
    driver, messages = driver_for(model)
    out = driver.generate("hi", {"json_mode": True, "json_schema": SCHEMA, "effort": "high"})

    kw = messages.last_kwargs
    assert "tool_choice" not in kw and "tools" not in kw
    output_config = kw["extra_body"]["output_config"]
    assert output_config["effort"] == "high"
    assert output_config["format"]["type"] == "json_schema"
    assert output_config["format"]["schema"]["additionalProperties"] is False
    assert out["text"] == '{"move": "e4"}'  # the text block, not the thinking block before it


@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-opus-5", "claude-haiku-5-5"])
def test_no_temperature_reaches_models_that_reject_sampling(driver_for, model) -> None:
    driver, messages = driver_for(model)
    driver.generate("hi", {"temperature": 0.0})
    assert "temperature" not in messages.last_kwargs.get("extra_body", {})


def test_older_models_keep_the_forced_tool_and_temperature(driver_for) -> None:
    driver, messages = driver_for("claude-sonnet-4-6")
    out = driver.generate("hi", {"json_mode": True, "json_schema": SCHEMA})

    kw = messages.last_kwargs
    assert kw["tool_choice"] == {"type": "tool", "name": "extract_json"}
    assert kw["extra_body"]["temperature"] == 0.0
    assert "output_config" not in kw["extra_body"]
    assert out["text"] == '{"move": "d4"}'


@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5"])
def test_stream_keeps_native_schema_effort_timeout_and_usage(driver_for, model) -> None:
    driver, messages = driver_for(model)
    chunks = list(
        driver.generate_messages_stream(
            [{"role": "user", "content": "Pick a move"}],
            {
                "json_mode": True,
                "json_schema": SCHEMA,
                "effort": "high",
                "timeout": 900,
            },
        )
    )
    kw = messages.last_kwargs
    assert "tool_choice" not in kw
    assert kw["timeout"] == 900
    assert kw["extra_body"]["output_config"]["effort"] == "high"
    schema = kw["extra_body"]["output_config"]["format"]["schema"]
    assert schema["properties"]["move"]["enum"] == ["e4", "d4"]
    assert schema["additionalProperties"] is False
    assert chunks[-1]["text"] == '{"move": "e4"}'
    assert chunks[-1]["meta"]["total_tokens"] == 15
    assert chunks[-1]["meta"]["raw_stop_reason"] == "end_turn"
    assert chunks[0]["type"] == "thinking_delta"


def test_older_models_stream_forced_tool_arguments_as_json(driver_for) -> None:
    driver, messages = driver_for("claude-sonnet-4-6")
    chunks = list(
        driver.generate_messages_stream(
            [{"role": "user", "content": "Pick a move"}],
            {
                "json_mode": True,
                "json_schema": SCHEMA,
            },
        )
    )
    assert messages.last_kwargs["tool_choice"] == {"type": "tool", "name": "extract_json"}
    assert messages.last_kwargs["tools"][0]["input_schema"] == SCHEMA
    assert "".join(c["text"] for c in chunks if c["type"] == "delta") == '{"move": "d4"}'
    assert chunks[-1]["text"] == '{"move": "d4"}'
    assert chunks[-1]["meta"]["total_tokens"] == 15
