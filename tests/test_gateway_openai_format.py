"""Tests for prompture.gateway.openai_format (OpenAI wire format helpers)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from prompture.gateway import (
    ChatOutcome,
    arun_chat,
    astream_chat_chunks,
    chat_completion,
    driver_options,
    finish_reason,
    models_list,
    run_chat,
    sse,
    stream_chat_chunks,
    to_driver_messages,
    tool_calls_to_openai,
    usage_from_meta,
)

META = {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10, "cost": 0.25, "raw_response": {}}


class TestRequestSide:
    def test_to_driver_messages_flattens_text_and_keeps_tool_fields(self):
        msgs = to_driver_messages(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "42"},
            ]
        )
        assert msgs[0] == {"role": "system", "content": "be terse"}
        assert msgs[1] == {"role": "user", "content": "a\nb"}
        assert msgs[2]["tool_calls"][0]["id"] == "c1"
        assert msgs[2]["content"] == ""
        assert msgs[3] == {"role": "tool", "content": "42", "tool_call_id": "c1"}

    def test_images_become_universal_blocks(self):
        data_uri = "data:image/png;base64,iVBORw0KGgo="
        (msg,) = to_driver_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ]
        )
        assert msg["content"][0] == {"type": "text", "text": "what is this"}
        assert msg["content"][1]["type"] == "image"
        assert msg["content"][1]["source"].media_type == "image/png"

    def test_accepts_pydantic_models(self):
        from pydantic import BaseModel

        class M(BaseModel):
            role: str
            content: str | None = None
            name: str | None = None

        assert to_driver_messages([M(role="user", content="hi")]) == [{"role": "user", "content": "hi"}]

    def test_driver_options(self):
        opts = driver_options(
            {
                "model": "x",
                "temperature": 0.2,
                "top_p": None,
                "max_completion_tokens": 50,
                "stop": ["\n"],
                "seed": 7,
                "user": "someone",
                "response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": {"type": "object"}}},
            }
        )
        assert opts == {
            "temperature": 0.2,
            "max_tokens": 50,
            "stop": ["\n"],
            "seed": 7,
            "json_mode": True,
            "json_schema": {"type": "object"},
        }

    def test_json_object_response_format(self):
        assert driver_options({"response_format": {"type": "json_object"}}) == {"json_mode": True}


class TestResponseSide:
    @pytest.mark.parametrize(
        ("stop", "expected"),
        [("end_turn", "stop"), ("max_tokens", "length"), ("tool_use", "tool_calls"), (None, "stop"), ("weird", "stop")],
    )
    def test_finish_reason(self, stop, expected):
        assert finish_reason(stop) == expected

    def test_usage_from_meta(self):
        assert usage_from_meta(META) == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
        assert usage_from_meta({"prompt_tokens": 1, "completion_tokens": 2})["total_tokens"] == 3
        assert usage_from_meta({"cached_prompt_tokens": 3})["prompt_tokens_details"] == {"cached_tokens": 3}

    def test_tool_calls_to_openai(self):
        (tc,) = tool_calls_to_openai([{"id": "c1", "name": "lookup", "arguments": {"q": "x"}}])
        assert tc == {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "x"}'}}
        already = {"id": "c2", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        assert tool_calls_to_openai([already]) == [already]

    def test_chat_completion_with_tool_calls(self):
        body = chat_completion(
            model="p/m",
            text="",
            meta=META,
            tool_calls=[{"id": "c", "name": "f", "arguments": {}}],
            stop_reason="tool_use",
        )
        choice = body["choices"][0]
        assert body["object"] == "chat.completion"
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "f"
        assert body["usage"]["total_tokens"] == 10

    def test_sse_framing(self):
        assert sse("[DONE]") == "data: [DONE]\n\n"
        assert sse({"a": 1}) == 'data: {"a":1}\n\n'

    def test_models_list(self):
        assert models_list(["a/b"], owned_by="me")["data"] == [{"id": "a/b", "object": "model", "owned_by": "me"}]


class _Driver:
    supports_tool_use = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def generate_messages(self, messages, options):
        self.calls.append(("messages", messages))
        return {"text": "hello", "meta": dict(META)}

    def generate_messages_with_tools(self, messages, tools, options):
        self.calls.append(("tools", tools))
        return {
            "text": "",
            "meta": dict(META),
            "tool_calls": [{"id": "c1", "name": "f", "arguments": {"x": 1}}],
            "stop_reason": "tool_use",
        }


class TestRunChat:
    def test_plain(self):
        drv = _Driver()
        out = run_chat(drv, [{"role": "user", "content": "hi"}], {"temperature": 0})
        assert isinstance(out, ChatOutcome)
        assert (out.text, out.cost, out.usage["total_tokens"]) == ("hello", 0.25, 10)
        assert drv.calls[0][0] == "messages"

    def test_tools(self):
        drv = _Driver()
        out = run_chat(drv, [], {}, tools=[{"type": "function", "function": {"name": "f"}}])
        body = out.to_completion("p/m")
        assert body["choices"][0]["finish_reason"] == "tool_calls"

    def test_tools_on_driver_without_support(self):
        drv = _Driver()
        drv.supports_tool_use = False
        with pytest.raises(NotImplementedError):
            run_chat(drv, [], {}, tools=[{"type": "function"}])

    def test_async(self):
        class A:
            supports_tool_use = False

            async def generate_messages(self, messages, options):
                return {"text": "yo", "meta": dict(META)}

        out = asyncio.run(arun_chat(A(), [], {}))
        assert out.text == "yo"


def _events(fail: bool = False):
    yield {"type": "delta", "text": "Hel"}
    yield {"type": "delta", "text": "lo"}
    if fail:
        raise RuntimeError("upstream cut")
    yield {"type": "done", "text": "Hello", "meta": dict(META)}


class TestStreaming:
    def test_chunks_and_on_complete(self):
        seen: list[ChatOutcome] = []
        chunks = list(stream_chat_chunks(_events(), model="p/m", completion_id="id1", on_complete=seen.append))
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert [c["choices"][0]["delta"].get("content") for c in chunks[1:3]] == ["Hel", "lo"]
        final = chunks[-1]
        assert final["choices"][0]["finish_reason"] == "stop"
        assert final["usage"]["total_tokens"] == 10
        assert all(c["id"] == "id1" for c in chunks)
        assert seen[0].text == "Hello" and seen[0].cost == 0.25 and seen[0].error is None

    def test_error_mid_stream(self):
        seen: list[ChatOutcome] = []
        chunks = list(stream_chat_chunks(_events(fail=True), model="p/m", on_complete=seen.append))
        assert chunks[-1] == {"error": {"message": "upstream cut", "type": "driver_error", "code": None}}
        assert seen[0].text == "Hello"
        assert isinstance(seen[0].error, RuntimeError)

    def test_async_stream_with_async_callback(self):
        seen: list[str] = []

        async def agen():
            for e in _events():
                yield e

        async def on_complete(outcome: ChatOutcome) -> None:
            seen.append(outcome.text)

        async def collect():
            return [c async for c in astream_chat_chunks(agen(), model="p/m", on_complete=on_complete)]

        chunks = asyncio.run(collect())
        assert json.loads(json.dumps(chunks))[-1]["choices"][0]["finish_reason"] == "stop"
        assert seen == ["Hello"]
