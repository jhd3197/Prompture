"""Responses transport, diagnostics and streaming contracts without live APIs."""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

from prompture.drivers._openai_responses import build_request, responses_input
from prompture.drivers.async_openai_driver import AsyncOpenAIDriver
from prompture.drivers.openai_driver import OpenAIDriver
from prompture.exceptions import DriverError


@pytest.fixture(autouse=True)
def rates(monkeypatch):
    monkeypatch.setattr(
        "prompture.infra.model_rates.get_model_rates", lambda *args: {"input": 5, "output": 30, "cache_read": 0.5}
    )


def make_driver(async_mode=False):
    cls = AsyncOpenAIDriver if async_mode else OpenAIDriver
    driver = object.__new__(cls)
    driver.model = "gpt-5.5"
    driver.api = "responses"
    driver.client = NS(responses=NS(create=AsyncMock() if async_mode else Mock()))
    driver._get_model_config = lambda *args: {"supports_temperature": False}
    return driver


def response(**changes):
    return {
        "id": "resp_1",
        "model": "gpt-5.5",
        "status": "completed",
        "service_tier": "default",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}],
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 800},
            "output_tokens_details": {"reasoning_tokens": 80},
        },
        "prompt_cache_diagnostics": {"type": "cache_miss", "reason": "tools_changed", "cache_missed_tokens": 200},
        **changes,
    }


class TestResponsesRequests:
    def test_converts_history_tools_and_images(self):
        messages = [
            {"role": "system", "content": "Be brief"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        items = responses_input(messages)
        assert items[1]["content"][1]["type"] == "input_image"
        assert items[2] == {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"}
        assert items[3]["output"] == "result"
        assert messages[1]["content"][1]["type"] == "image_url"

    def test_schema_diagnostics_and_tool_choice(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        request = build_request(
            make_driver(),
            [{"role": "user", "content": "hi"}],
            {
                "json_mode": True,
                "json_schema": schema,
                "temperature": 1,
                "prompt_cache_options": {"comparison_response_id": "resp_previous"},
                "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            },
            [{"type": "function", "function": {"name": "lookup", "parameters": schema}}],
        )
        assert request["text"]["format"]["schema"]["additionalProperties"] is False
        assert "additionalProperties" not in schema
        assert request["tools"][0]["name"] == "lookup"
        assert request["tool_choice"] == {"type": "function", "name": "lookup"}
        assert request["prompt_cache_options"]["comparison_response_id"] == "resp_previous"
        assert request["store"] is False
        assert "temperature" not in request

    def test_counting_excludes_generation_settings(self):
        request = build_request(make_driver(), [], {"service_tier": "flex", "max_tokens": 40}, counting=True)
        assert request == {"model": "gpt-5.5", "input": []}


class TestResponsesGeneration:
    def test_metadata_and_reasoning_are_not_double_billed(self):
        driver = make_driver()
        driver.client.responses.create.return_value = response()
        result = driver.generate("hi", {})
        assert result["text"] == "hello"
        assert result["meta"]["cost"] == pytest.approx(0.0044)
        assert result["meta"]["usage_details"]["reasoning_tokens"] == 80
        assert result["meta"]["response_id"] == "resp_1"
        assert result["meta"]["prompt_cache_diagnostics"]["reason"] == "tools_changed"

    def test_function_call_output_and_truncation(self):
        driver = make_driver()
        driver.client.responses.create.return_value = response(
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
            output=[{"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": '{"a":'}],
        )
        result = driver.generate_messages_with_tools([], [], {})
        assert result["stop_reason"] == "max_tokens"
        assert result["tool_calls"][0]["arguments_error"]

    def test_failed_response_raises(self):
        driver = make_driver()
        driver.client.responses.create.return_value = response(status="failed")
        with pytest.raises(DriverError):
            driver.generate("hi", {})

    @pytest.mark.asyncio
    async def test_async_generation_matches_sync(self):
        driver = make_driver(True)
        driver.client.responses.create.return_value = response()
        result = await driver.generate_messages_with_tools([], [], {})
        assert result["meta"]["cost"] == pytest.approx(0.0044)
        driver.client.responses.create.assert_awaited_once()


class TestResponsesStreaming:
    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter", None])
    async def test_malformed_tool_arguments_use_actual_terminal_reason(self, async_mode, reason):
        driver = make_driver(async_mode)
        item = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": '{"x":'}
        terminal = response(
            status="incomplete" if reason else "completed",
            incomplete_details={"reason": reason} if reason else None,
            output=[item],
        )
        source = [
            {"type": "response.output_item.added", "item": item},
            {"type": "response.output_item.done", "item": item},
            {"type": "response.incomplete" if reason else "response.completed", "response": terminal},
        ]

        async def aevents():
            for event in source:
                yield event

        driver.client.responses.create.return_value = aevents() if async_mode else iter(source)
        stream = driver.generate_messages_with_tools_stream([], [], {})
        events = [event async for event in stream] if async_mode else list(stream)
        stops = [event for event in events if event.event_type == "tool_use_stop"]
        assert len(stops) == 1
        assert stops[0].truncated is (reason == "max_output_tokens")
        assert stops[0].raw_stop_reason == (reason or "stop")
        assert events[-1].event_type == "message_stop"

    def test_text_and_terminal_diagnostics(self):
        driver = make_driver()
        driver.client.responses.create.return_value = iter(
            [
                {"type": "response.output_text.delta", "delta": "hello"},
                {"type": "response.completed", "response": response()},
            ]
        )
        events = list(driver.generate_messages_stream([], {}))
        assert events[0] == {"type": "delta", "text": "hello"}
        assert events[-1]["meta"]["usage_complete"]
        assert events[-1]["meta"]["prompt_cache_diagnostics"]["reason"] == "tools_changed"

    def test_missing_terminal_event_is_incomplete(self):
        driver = make_driver()
        driver.client.responses.create.return_value = iter([{"type": "response.output_text.delta", "delta": "partial"}])
        done = list(driver.generate_messages_stream([], {}))[-1]
        assert done["text"] == "partial"
        assert done["meta"]["usage_complete"] is False
        assert done["meta"]["cost_status"] == "partial"

    def test_live_tool_events(self):
        driver = make_driver()
        item = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": '{"x":1}'}
        driver.client.responses.create.return_value = iter(
            [
                {"type": "response.output_item.added", "item": item},
                {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"x":1}'},
                {"type": "response.output_item.done", "item": item},
                {"type": "response.completed", "response": response(output=[item])},
            ]
        )
        events = list(driver.generate_messages_with_tools_stream([], [], {}))
        assert [event.event_type for event in events] == [
            "tool_use_start",
            "tool_input_delta",
            "tool_use_stop",
            "message_stop",
        ]
        assert events[2].input == {"x": 1}
        assert events[-1].stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_async_stream(self):
        driver = make_driver(True)

        async def events():
            yield {"type": "response.output_text.delta", "delta": "hello"}
            yield {"type": "response.completed", "response": response()}

        driver.client.responses.create.return_value = events()
        received = [event async for event in driver.generate_messages_stream([], {})]
        assert received[-1]["meta"]["cost"] == pytest.approx(0.0044)
