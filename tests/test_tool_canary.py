"""Live canary: one real tool round-trip per provider.

The conformance suite proves every driver builds the right request and reads
the right reply, against fakes. It cannot notice a provider changing its API.
This does: it sends a real forced tool call, feeds the result back, and checks
the normalized shape, buffered and streamed.

It runs only when asked and only against the models you name::

    TOOL_CANARY_MODELS="openai/gpt-4o-mini,groq/llama-3.3-70b-versatile" \\
        pytest tests/test_tool_canary.py --run-integration

Each model costs two or three small requests. Schedule it weekly rather than
on every push.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from prompture.drivers import get_driver_for_model
from prompture.drivers.base import STOP_REASONS

pytestmark = pytest.mark.integration

MODELS = [m.strip() for m in os.getenv("TOOL_CANARY_MODELS", "").split(",") if m.strip()]

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}
ASK = [
    {"role": "system", "content": "Use the get_weather tool to answer weather questions."},
    {"role": "user", "content": "What is the weather in Paris right now?"},
]
OPTIONS: dict[str, Any] = {"max_tokens": 512, "tool_choice": "required"}


def _check_call(call: dict[str, Any]) -> None:
    assert isinstance(call["id"], str) and call["id"]
    assert call["name"] == "get_weather"
    assert "arguments_error" not in call, call.get("arguments_error")
    assert "paris" in str(call["arguments"].get("city", "")).lower()


@pytest.mark.skipif(not MODELS, reason="set TOOL_CANARY_MODELS to a comma-separated list of provider/model")
@pytest.mark.parametrize("model", MODELS or ["unset"])
def test_tool_round_trip(model: str) -> None:
    driver = get_driver_for_model(model)
    if not getattr(driver, "supports_tool_use", False):
        pytest.skip(f"{model} does not support tool use")

    first = driver.generate_messages_with_tools(ASK, [TOOL], dict(OPTIONS))
    assert first["stop_reason"] in STOP_REASONS, first["stop_reason"]
    assert first["stop_reason"] == "tool_use"
    assert first["tool_calls"], "a forced tool call returned no tool calls"
    call = first["tool_calls"][0]
    _check_call(call)
    assert first["meta"]["prompt_tokens"] > 0

    # Feed the result back in the canonical shape and expect a plain answer.
    history = [
        *ASK,
        {
            "role": "assistant",
            "content": first.get("text") or "",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                }
                for tc in first["tool_calls"]
            ],
            **({"reasoning_content": first["reasoning_content"]} if first.get("reasoning_content") else {}),
            **({"_google_parts": first["_google_parts"]} if first.get("_google_parts") is not None else {}),
        },
        *[
            {"role": "tool", "tool_call_id": tc["id"], "content": "18 degrees Celsius and sunny"}
            for tc in first["tool_calls"]
        ],
    ]
    second = driver.generate_messages_with_tools(history, [TOOL], {"max_tokens": 512})
    assert second["stop_reason"] in STOP_REASONS, second["stop_reason"]
    assert second["text"] or second["tool_calls"], "the model said nothing after the tool result"


@pytest.mark.skipif(not MODELS, reason="set TOOL_CANARY_MODELS to a comma-separated list of provider/model")
@pytest.mark.parametrize("model", MODELS or ["unset"])
def test_streamed_tool_call(model: str) -> None:
    driver = get_driver_for_model(model)
    if not getattr(driver, "supports_tool_use", False):
        pytest.skip(f"{model} does not support tool use")

    events = list(driver.generate_messages_with_tools_stream(ASK, [TOOL], dict(OPTIONS)))
    stops = [e for e in events if e.event_type == "tool_use_stop"]
    assert stops, "a forced tool call streamed no tool call"
    _check_call({"id": stops[0].id, "name": stops[0].name, "arguments": stops[0].input})
    assert stops[0].arguments_error is None
    final = events[-1]
    assert final.event_type == "message_stop"
    assert final.stop_reason == "tool_use"
