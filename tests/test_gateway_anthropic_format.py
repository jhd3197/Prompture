"""Tests for prompture.gateway.anthropic_format (Anthropic Messages API)."""

from __future__ import annotations

import json
from typing import Any

from prompture.agents.live_events import MessageStop, TextDelta, ToolInputDelta, ToolUseStart, ToolUseStop
from prompture.gateway import (
    ChatOutcome,
    anthropic_message,
    anthropic_sse,
    anthropic_to_driver,
    estimate_input_tokens,
    live_events_for,
    stream_anthropic_events,
)

TOOLS = [
    {"name": "read_file", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {}}}},
    {"type": "web_search_20250305", "name": "web_search"},  # server tool: skipped
]


def test_request_conversion_round_trips_tool_turns():
    body = {
        "model": "combo/chat",
        "max_tokens": 1024,
        "temperature": 0.2,
        "stop_sequences": ["END"],
        "system": [{"type": "text", "text": "You are terse."}],
        "tool_choice": {"type": "tool", "name": "read_file"},
        "tools": TOOLS,
        "messages": [
            {"role": "user", "content": "open README"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "x"},
                    {"type": "text", "text": "Reading it."},
                    {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "README.md"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "# Hi"}]},
                    {"type": "text", "text": "summarize"},
                ],
            },
        ],
    }
    messages, tools, options = anthropic_to_driver(body)

    assert messages[0] == {"role": "system", "content": "You are terse."}
    assert messages[1] == {"role": "user", "content": "open README"}
    assistant = messages[2]
    assert assistant["role"] == "assistant" and assistant["content"] == "Reading it."
    call = assistant["tool_calls"][0]
    assert call["id"] == "toolu_1"
    assert json.loads(call["function"]["arguments"]) == {"path": "README.md"}
    assert messages[3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "# Hi"}
    assert messages[4] == {"role": "user", "content": "summarize"}

    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {}}},
            },
        }
    ]
    assert options == {
        "max_tokens": 1024,
        "temperature": 0.2,
        "stop": ["END"],
        "tool_choice": {"type": "function", "function": {"name": "read_file"}},
    }


def test_tool_error_and_image_blocks():
    messages, _, _ = anthropic_to_driver(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t", "content": "boom", "is_error": True}],
                },
            ]
        }
    )
    assert messages[0]["content"][0] == {"type": "text", "text": "what is this"}
    assert messages[0]["content"][1]["type"] == "image"
    assert messages[1] == {"role": "tool", "tool_call_id": "t", "content": "[tool error] boom"}


def test_anthropic_message_shape():
    outcome = ChatOutcome(
        text="Let me check.",
        meta={"prompt_tokens": 10, "completion_tokens": 5, "cached_prompt_tokens": 4},
        tool_calls=[{"id": "toolu_9", "name": "read_file", "arguments": {"path": "a"}}],
        stop_reason="tool_use",
    )
    msg = anthropic_message(outcome, model="combo/chat", message_id="msg_1")
    assert msg["type"] == "message" and msg["role"] == "assistant"
    assert msg["content"] == [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_9", "name": "read_file", "input": {"path": "a"}},
    ]
    assert msg["stop_reason"] == "tool_use"
    assert msg["usage"] == {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 4}
    plain = anthropic_message(ChatOutcome(text="hi", stop_reason="max_tokens"), model="m")
    assert plain["stop_reason"] == "max_tokens"


def _names(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [e for e, _ in events]


def test_stream_text_then_tool_use():
    done: list[ChatOutcome] = []
    live = [
        TextDelta(text="Checking"),
        TextDelta(text="..."),
        ToolUseStart(id="toolu_1", name="read_file"),
        ToolInputDelta(id="toolu_1", fragment='{"path":'),
        ToolInputDelta(id="toolu_1", fragment='"a"}'),
        ToolUseStop(id="toolu_1", name="read_file", input={"path": "a"}),
        MessageStop(stop_reason="tool_use", usage={"prompt_tokens": 7, "completion_tokens": 3}),
    ]
    events = list(stream_anthropic_events(live, model="m", message_id="msg_x", on_complete=done.append))
    assert _names(events) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[0][1]["message"]["id"] == "msg_x"
    assert events[5][1]["content_block"] == {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}}
    assert events[5][1]["index"] == 1
    assert events[6][1]["delta"] == {"type": "input_json_delta", "partial_json": '{"path":'}
    assert events[9][1]["delta"]["stop_reason"] == "tool_use"
    assert events[9][1]["usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert done[0].text == "Checking..." and done[0].tool_calls[0]["arguments"] == {"path": "a"}


def test_stream_tool_without_input_deltas_emits_full_json():
    live = [
        ToolUseStart(id="t", name="f"),
        ToolUseStop(id="t", name="f", input={"x": 1}),
        MessageStop(stop_reason="tool_use"),
    ]
    events = list(stream_anthropic_events(live, model="m"))
    deltas = [d for e, d in events if e == "content_block_delta"]
    assert deltas == [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"x": 1}'}}
    ]


def test_stream_plain_dict_events_and_error():
    def gen():
        yield {"type": "delta", "text": "Hel"}
        raise RuntimeError("cut")

    seen: list[ChatOutcome] = []
    events = list(stream_anthropic_events(gen(), model="m", on_complete=seen.append))
    assert _names(events)[-2:] == ["content_block_stop", "error"]
    assert events[-1][1] == {"type": "error", "error": {"type": "api_error", "message": "cut"}}
    assert isinstance(seen[0].error, RuntimeError)

    ok = list(
        stream_anthropic_events(
            iter([{"type": "delta", "text": "yo"}, {"type": "done", "meta": {"completion_tokens": 1}}]), model="m"
        )
    )
    assert ok[-2][1]["delta"]["stop_reason"] == "end_turn"


def test_live_events_for_falls_back_to_buffered_call():
    class Buffered:
        supports_tool_use = True
        supports_streaming = False

        def generate_messages_with_tools_stream(self, messages, tools, options):
            yield TextDelta(text="streamed")
            yield MessageStop(stop_reason="end_turn")

        def generate_messages(self, messages, options):
            return {"text": "buffered", "meta": {"completion_tokens": 2}}

    drv = Buffered()
    with_tools = list(live_events_for(drv, [], [{"type": "function", "function": {"name": "f"}}], {}))
    assert with_tools[0].text == "streamed"
    no_tools = list(live_events_for(drv, [], [], {}))
    assert no_tools[0].text == "buffered"
    assert no_tools[-1].event_type == "message_stop"


def test_sse_and_estimate():
    assert anthropic_sse("ping", {"type": "ping"}) == 'event: ping\ndata: {"type":"ping"}\n\n'
    assert estimate_input_tokens([{"role": "user", "content": "x" * 400}]) >= 100
