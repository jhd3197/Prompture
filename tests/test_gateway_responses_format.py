"""Tests for prompture.gateway.responses_format (OpenAI Responses API)."""

from __future__ import annotations

import json

from prompture.agents.live_events import MessageStop, TextDelta, ToolInputDelta, ToolUseStart, ToolUseStop
from prompture.gateway import (
    ChatOutcome,
    response_object,
    responses_sse,
    responses_to_driver,
    stream_responses_events,
)


def test_request_conversion():
    messages, tools, options = responses_to_driver(
        {
            "model": "combo/chat",
            "instructions": "be brief",
            "max_output_tokens": 300,
            "reasoning": {"effort": "low"},
            "tool_choice": {"type": "function", "name": "shell"},
            "text": {"format": {"type": "json_schema", "name": "x", "schema": {"type": "object"}}},
            "tools": [
                {"type": "function", "name": "shell", "description": "run", "parameters": {"type": "object"}},
                {"type": "web_search"},
            ],
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "dev note"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]},
                {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": '{"cmd":"ls"}'},
                {"type": "function_call", "call_id": "c2", "name": "shell", "arguments": '{"cmd":"pwd"}'},
                {"type": "function_call_output", "call_id": "c1", "output": "a.py"},
                {"type": "function_call_output", "call_id": "c2", "output": {"cwd": "/x"}},
                {"type": "reasoning", "summary": []},
            ],
        }
    )
    assert messages[0] == {"role": "system", "content": "be brief"}
    assert messages[1] == {"role": "system", "content": "dev note"}
    assert messages[2] == {"role": "user", "content": "list files"}
    assert [c["id"] for c in messages[3]["tool_calls"]] == ["c1", "c2"]  # consecutive calls grouped
    assert messages[4] == {"role": "tool", "tool_call_id": "c1", "content": "a.py"}
    assert json.loads(messages[5]["content"]) == {"cwd": "/x"}
    assert tools == [
        {"type": "function", "function": {"name": "shell", "description": "run", "parameters": {"type": "object"}}}
    ]
    assert options == {
        "max_tokens": 300,
        "reasoning_effort": "low",
        "tool_choice": {"type": "function", "function": {"name": "shell"}},
        "json_mode": True,
        "json_schema": {"type": "object"},
    }
    assert responses_to_driver({"input": "hi"})[0] == [{"role": "user", "content": "hi"}]


def test_response_object():
    body = response_object(
        ChatOutcome(
            text="Running it.",
            meta={"prompt_tokens": 5, "completion_tokens": 2},
            tool_calls=[{"id": "c9", "name": "shell", "arguments": {"cmd": "ls"}}],
        ),
        model="m",
        response_id="resp_1",
    )
    assert body["id"] == "resp_1" and body["object"] == "response" and body["status"] == "completed"
    assert body["output"][0]["content"][0] == {"type": "output_text", "text": "Running it.", "annotations": []}
    call = body["output"][1]
    assert (call["type"], call["call_id"], call["name"]) == ("function_call", "c9", "shell")
    assert json.loads(call["arguments"]) == {"cmd": "ls"}
    assert body["usage"] == {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
    truncated = response_object(ChatOutcome(text="x", stop_reason="max_tokens"), model="m")
    assert truncated["status"] == "incomplete"


def test_streaming_text_and_function_call():
    done: list[ChatOutcome] = []
    live = [
        TextDelta(text="Let me "),
        TextDelta(text="check."),
        ToolUseStart(id="c1", name="shell"),
        ToolInputDelta(id="c1", fragment='{"cmd":'),
        ToolInputDelta(id="c1", fragment='"ls"}'),
        ToolUseStop(id="c1", name="shell", input={"cmd": "ls"}),
        MessageStop(stop_reason="tool_use", usage={"prompt_tokens": 3, "completion_tokens": 4}),
    ]
    events = list(stream_responses_events(live, model="m", response_id="resp_s", on_complete=done.append))
    types = [e["type"] for e in events]
    assert types == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    assert events[0]["response"]["status"] == "in_progress"
    assert events[5]["text"] == "Let me check."
    assert events[11]["arguments"] == '{"cmd":"ls"}'
    final = events[-1]["response"]
    assert [o["type"] for o in final["output"]] == ["message", "function_call"]
    assert final["usage"]["total_tokens"] == 7
    assert done[0].tool_calls == [{"id": "c1", "name": "shell", "arguments": {"cmd": "ls"}}]


def test_streaming_plain_dicts_and_failure():
    def gen():
        yield {"type": "delta", "text": "hi"}
        raise RuntimeError("cut")

    events = list(stream_responses_events(gen(), model="m"))
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"]["message"] == "cut"

    ok = list(stream_responses_events(iter([{"type": "delta", "text": "yo"}, {"type": "done", "meta": {}}]), model="m"))
    assert ok[-1]["type"] == "response.completed"
    assert responses_sse(ok[-1]).startswith("event: response.completed\ndata: ")
