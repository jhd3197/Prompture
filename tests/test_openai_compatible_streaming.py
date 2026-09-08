"""Exercise compatible gateways over real loopback HTTP, without a live LLM."""

import asyncio
import inspect
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from prompture.drivers.async_openai_compatible_driver import AsyncOpenAICompatibleDriver
from prompture.drivers.openai_compatible_driver import OpenAICompatibleDriver

MESSAGES = [{"role": "user", "content": "What is the weather?"}]
TOOLS = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
USAGE = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}


@pytest.fixture
def gateway():
    """Serve JSON completions and fragmented SSE on an ephemeral local port."""
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, dict(self.headers), payload))
            if payload["model"] == "error":
                self.send_error(400, "Unsupported model")
                return
            tool_turn = bool(payload.get("tools")) and payload["messages"][-1]["role"] != "tool"
            arguments = '{"city":"Miami"}' if payload["model"] != "malformed" else '{"city":'
            finish = "length" if payload["model"] == "malformed" else "tool_calls" if tool_turn else "stop"
            if payload.get("stream"):
                chunks = [{"choices": [{"delta": {"content": "Hello "}}]}]
                if tool_turn:
                    for fragment in [arguments[:8], arguments[8:]]:
                        chunks.append(
                            {
                                "choices": [
                                    {
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "index": 0,
                                                    "id": "call_weather",
                                                    "type": "function",
                                                    "function": {"name": "weather", "arguments": fragment},
                                                }
                                            ]
                                        }
                                    }
                                ]
                            }
                        )
                chunks.extend(
                    [
                        {"choices": [{"delta": {"content": "world"}, "finish_reason": finish}]},
                        {"choices": [], "usage": USAGE},
                    ]
                )
                body = (
                    ": keepalive\n\ndata: invalid-json\n\n"
                    + "".join("data:" + json.dumps(chunk) + "\n\n" for chunk in chunks)
                    + "data: [DONE]\n\n"
                )
                content_type = "text/event-stream; charset=utf-8"
            else:
                message = {"role": "assistant", "content": None if tool_turn else "Hello world"}
                if tool_turn:
                    message["tool_calls"] = [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": arguments,
                            },
                        }
                    ]
                body = json.dumps({"choices": [{"message": message, "finish_reason": finish}], "usage": USAGE})
                content_type = "application/json"
            body = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture(params=[OpenAICompatibleDriver, AsyncOpenAICompatibleDriver], ids=["sync", "async"])
def driver(request, gateway, monkeypatch):
    instance = request.param(endpoint=gateway[0], model="gateway/model")
    instance.api_key = None
    monkeypatch.setattr(
        instance,
        "_get_model_config",
        lambda *args: {
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
    )
    monkeypatch.setattr(instance, "_calculate_cost", lambda *args, **kwargs: 0.0)
    return instance


def invoke(driver, method, *args):
    result = getattr(driver, method)(*args)
    if inspect.isawaitable(result):
        return asyncio.run(result)
    if hasattr(result, "__aiter__"):

        async def collect():
            return [event async for event in result]

        return asyncio.run(collect())
    return list(result) if "stream" in method else result


class TestCompatibleGateway:
    """Both transports preserve native protocol semantics and gateway options."""

    def test_text_stream_and_overrides(self, driver, gateway):
        endpoint, received = gateway
        driver.endpoint = "http://127.0.0.1:1/unreachable"
        options = {
            "endpoint": endpoint + "/",
            "api_key": "test-key",
            "model": "override/model",
            "json_mode": True,
            "max_tokens": 73,
            "temperature": 0.2,
            "extra_body": {"vendor_option": True},
        }
        chunks = invoke(driver, "generate_messages_stream", MESSAGES, options)
        assert [c["text"] for c in chunks if c["type"] == "delta"] == ["Hello ", "world"]
        assert chunks[-1]["text"] == "Hello world"
        assert chunks[-1]["meta"]["total_tokens"] == 17
        assert chunks[-1]["meta"]["endpoint"] == endpoint
        assert chunks[-1]["meta"]["pricing_unknown"] is True
        path, headers, payload = received[-1]
        assert path == "/v1/chat/completions"
        assert headers["Authorization"] == "Bearer test-key"
        assert payload["model"] == "override/model"
        assert payload["max_tokens"] == 73
        assert payload["temperature"] == 0.2
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["vendor_option"] is True
        assert payload["stream_options"] == {"include_usage": True}
        assert "tools" not in payload

    @pytest.mark.parametrize("stream", [False, True])
    def test_native_tools_and_result_round_trip(self, driver, gateway, stream):
        method = "generate_messages_with_tools" + ("_stream" if stream else "")
        options = {"tool_choice": "required", "parallel_tool_calls": False}
        result = invoke(driver, method, MESSAGES, TOOLS, options)
        if stream:
            stop = next(e for e in result if e.event_type == "tool_use_stop")
            assert stop.input == {"city": "Miami"}
            assert not stop.truncated
            assert result[-1].event_type == "message_stop"
            assert result[-1].usage["total_tokens"] == 17
            assert len([e for e in result if e.event_type == "tool_input_delta"]) == 2
        else:
            assert result["tool_calls"] == [{"id": "call_weather", "name": "weather", "arguments": {"city": "Miami"}}]
            assert result["stop_reason"] == "tool_calls"
            assert result["text"] == ""
        _, headers, payload = gateway[1][-1]
        assert "Authorization" not in headers
        assert payload["model"] == "gateway/model"
        assert payload["tools"] == TOOLS
        assert payload["tool_choice"] == "required"
        assert payload["parallel_tool_calls"] is False
        history = [
            *MESSAGES,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"Miami"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_weather", "content": "Sunny"},
        ]
        final = invoke(driver, "generate_messages_with_tools", history, TOOLS, {})
        assert final["text"] == "Hello world"
        assert final["tool_calls"] == []
        assert gateway[1][-1][2]["messages"] == history

    @pytest.mark.parametrize("stream", [False, True])
    def test_truncated_arguments(self, driver, stream):
        method = "generate_messages_with_tools" + ("_stream" if stream else "")
        result = invoke(driver, method, MESSAGES, TOOLS, {"model": "malformed"})
        if stream:
            stop = next(e for e in result if e.event_type == "tool_use_stop")
            assert stop.truncated
            assert stop.input == {}
        else:
            assert "truncated" in result["tool_calls"][0]["arguments_error"]

    @pytest.mark.parametrize(
        "method", ["generate_messages_stream", "generate_messages_with_tools", "generate_messages_with_tools_stream"]
    )
    def test_http_errors(self, driver, method):
        args = [MESSAGES, TOOLS] if "tools" in method else [MESSAGES]
        with pytest.raises(RuntimeError, match="OpenAI-compatible API request failed"):
            invoke(driver, method, *args, {"model": "error"})

    def test_schema_and_profile_preserved_in_live_stream(self, driver, gateway, monkeypatch):
        seen = []
        driver.profile = "fireworks"

        def config(provider, model):
            seen.append((provider, model))
            return {"tokens_param": "max_completion_tokens", "supports_temperature": False}

        monkeypatch.setattr(driver, "_get_model_config", config)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        invoke(
            driver,
            "generate_messages_with_tools_stream",
            MESSAGES,
            TOOLS,
            {
                "json_mode": True,
                "json_schema": schema,
                "guided_decoding": True,
                "extra_body": {"vendor_option": 7},
            },
        )
        payload = gateway[1][-1][2]
        assert seen == [("fireworks", "gateway/model")]
        assert payload["max_completion_tokens"] == 4096
        assert "temperature" not in payload
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["vendor_option"] == 7
        assert payload["guided_json"] == schema
        assert "additionalProperties" not in schema
