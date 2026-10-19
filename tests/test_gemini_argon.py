"""Gemini wire compatibility using real SDK types and mocked API transports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import types

from prompture.agents.async_conversation import AsyncConversation
from prompture.agents.conversation import Conversation
from prompture.agents.tools_schema import ToolRegistry
from prompture.drivers.async_google_driver import AsyncGoogleDriver
from prompture.drivers.async_google_vertexai_driver import AsyncGoogleVertexAIDriver
from prompture.drivers.google_driver import GoogleDriver
from prompture.drivers.google_vertexai_driver import GoogleVertexAIDriver
from prompture.exceptions import DriverError
from prompture.infra import model_rates, pricing

DRIVERS = [GoogleDriver, AsyncGoogleDriver, GoogleVertexAIDriver, AsyncGoogleVertexAIDriver]
MESSAGES = [{"role": "user", "content": "Calculate it"}]


@pytest.fixture(autouse=True)
def offline_rates(monkeypatch):
    monkeypatch.setattr(model_rates, "_loaded", True)
    monkeypatch.setattr(model_rates, "_data", {})
    pricing.clear_pricing_sources()
    pricing.register_pricing_source(pricing.LocalKBPricingSource())
    clock = MagicMock()
    clock.now.return_value = datetime(2026, 11, 30, 23, 59, 59, tzinfo=timezone.utc)
    monkeypatch.setattr(pricing, "datetime", clock)
    yield clock
    pricing.clear_pricing_sources()


def make_driver(cls):
    driver = cls.__new__(cls)
    driver.model = "gemini-4-argon"
    driver.options = {}
    client = MagicMock()
    if "Vertex" in cls.__name__:
        driver._gemini_client = client
    else:
        driver._client = client
    return driver, client.aio.models if cls.__name__.startswith("Async") else client.models


def build(driver, messages=MESSAGES, options=None):
    method = getattr(driver, "_build_generation_args", None) or driver._build_gemini_generation_args
    return method(messages, options or {})


def response(parts=None, usage=True):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model", parts=parts or [types.Part(text="Done", thought_signature=b"\xfffinal")]
                ),
                finish_reason="STOP",
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1000,
            cached_content_token_count=700,
            candidates_token_count=50,
            thoughts_token_count=200,
            total_token_count=1250,
        )
        if usage
        else None,
    )


def signed_tool_response():
    return response(
        [
            types.Part(text="Planning", thought=True, thought_signature=b"\xffthought"),
            types.Part(text="Checking", thought_signature=b"\x80text"),
            types.Part(
                function_call=types.FunctionCall(name="double", args={"x": 2}), thought_signature=b"\xfe\x00call"
            ),
            types.Part(function_call=types.FunctionCall(name="double", args={"x": 3}, id="api-call-2")),
        ]
    )


class TestArgonConfiguration:
    """Argon's effort and sampling rules apply across all four drivers."""

    @pytest.mark.parametrize("cls", DRIVERS)
    @pytest.mark.parametrize(
        "effort,level",
        [("none", "low"), ("minimal", "low"), ("low", "low"), ("medium", "medium"), ("high", "high"), ("max", "high")],
    )
    def test_effort_and_sampling(self, cls, effort, level):
        driver, _ = make_driver(cls)
        _, config = build(
            driver,
            options={
                "effort": effort,
                "temperature": 0,
                "top_p": 0.9,
                "max_tokens": 32768,
                "generation_config": {"temperature": 1, "top_k": 20},
            },
        )
        assert config == {"max_output_tokens": 32768, "thinking_config": {"thinking_level": level}}
        types.GenerateContentConfig(**config)

    @pytest.mark.parametrize("cls", DRIVERS)
    def test_explicit_thinking_and_schema(self, cls):
        driver, _ = make_driver(cls)
        _, config = build(
            driver,
            options={
                "effort": "high",
                "json_mode": True,
                "json_schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
                "generation_config": {"thinking_config": {"thinking_level": "low"}},
            },
        )
        assert config["thinking_config"] == {"thinking_level": "low"}
        assert config["response_mime_type"] == "application/json"
        assert config["response_schema"]["type"] == "object"
        _, budget = build(
            driver, options={"effort": "high", "generation_config": {"thinking_config": {"thinking_budget": 8192}}}
        )
        assert budget["thinking_config"] == {"thinking_budget": 8192}

    @pytest.mark.parametrize("cls", DRIVERS)
    def test_legacy_model_still_uses_low_for_medium(self, cls):
        driver, _ = make_driver(cls)
        driver.model = "gemini-3.1-pro-preview"
        _, config = build(driver, options={"effort": "medium", "temperature": 0.3})
        assert config["thinking_config"]["thinking_level"] == "low"
        assert config["temperature"] == 0.3

    @pytest.mark.parametrize("cls", DRIVERS)
    def test_orphan_tool_result_rejected(self, cls):
        driver, _ = make_driver(cls)
        with pytest.raises(DriverError, match="unknown tool_call_id"):
            build(driver, [{"role": "tool", "tool_call_id": "missing", "content": "value"}])


class TestArgonUsage:
    """Token costs include reasoning once and retain cached-input discounts."""

    @pytest.mark.parametrize("cls", DRIVERS)
    def test_thinking_and_cached_tokens(self, cls):
        driver, _ = make_driver(cls)
        extract = getattr(driver, "_extract_usage_metadata", None) or driver._extract_gemini_usage
        meta = extract(response(), MESSAGES)
        assert meta["prompt_tokens"] == 1000
        assert meta["completion_tokens"] == 250
        assert meta["reasoning_tokens"] == 200
        assert meta["total_tokens"] == 1250
        assert meta["cached_prompt_tokens"] == 700
        assert meta["cost"] == pytest.approx(0.00317)

    @pytest.mark.parametrize("cls", DRIVERS)
    @pytest.mark.asyncio
    async def test_stream_uses_last_usage_chunk(self, cls):
        driver, api = make_driver(cls)
        chunks = [
            response([types.Part(text="A")]),
            response([types.Part(text="B")]),
            types.GenerateContentResponse(usage_metadata=response().usage_metadata),
        ]
        if cls.__name__.startswith("Async"):

            async def stream():
                for chunk in chunks:
                    yield chunk

            api.generate_content_stream = AsyncMock(return_value=stream())
            result = [chunk async for chunk in driver.generate_messages_stream(MESSAGES, {})]
        else:
            api.generate_content_stream.return_value = iter(chunks)
            result = list(driver.generate_messages_stream(MESSAGES, {}))
        assert result[-1]["text"] == "AB"
        assert result[-1]["meta"]["completion_tokens"] == 250
        assert result[-1]["meta"]["cost"] == pytest.approx(0.00317)

    @pytest.mark.parametrize("cls", DRIVERS)
    def test_missing_usage_and_missing_total(self, cls):
        driver, _ = make_driver(cls)
        extract = getattr(driver, "_extract_usage_metadata", None) or driver._extract_gemini_usage
        meta = extract(response(usage=False), MESSAGES)
        assert meta["completion_tokens"] == 1
        assert meta["reasoning_tokens"] == 0
        resp = SimpleNamespace(
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=3, thoughts_token_count=7)
        )
        assert extract(resp, MESSAGES)["total_tokens"] == 20


class TestArgonConversation:
    """Signed tool parts survive agent loops and JSON history serialization."""

    @pytest.mark.parametrize("cls", DRIVERS)
    @pytest.mark.parametrize("events", [False, True, "live"])
    @pytest.mark.asyncio
    async def test_parallel_tool_loop(self, cls, events):
        driver, api = make_driver(cls)
        original = signed_tool_response()
        responses = [original, response()]
        is_async = cls.__name__.startswith("Async")
        api.generate_content = AsyncMock(side_effect=responses) if is_async else MagicMock(side_effect=responses)
        registry = ToolRegistry()

        def double(x: int) -> int:
            """Double x."""
            return 2 * x

        registry.register(double)
        conv = (AsyncConversation if is_async else Conversation)(driver=driver, tools=registry)
        if events == "live":
            if is_async:
                result = [event async for event in conv.ask_live("Calculate")]
            else:
                result = list(conv.ask_live("Calculate"))
            assert result[-1].event_type == "turn_complete"
        elif events:
            if is_async:
                result = [event async for event in conv.ask_with_tool_events("Calculate")]
            else:
                result = list(conv.ask_with_tool_events("Calculate"))
            assert result
        elif is_async:
            assert await conv.ask("Calculate") == "Done"
        else:
            assert conv.ask("Calculate") == "Done"
        assert api.generate_content.call_count == 2
        sent = api.generate_content.call_args.kwargs["contents"]
        model = next(c for c in sent if c["role"] == "model")
        assert model["parts"] == original.candidates[0].content.parts
        tool_results = [p.function_response for p in sent[-1]["parts"]]
        assert len(sent) == 3  # user prompt, all model calls, all tool results
        assert [r.name for r in tool_results] == ["double", "double"]
        assert [r.id for r in tool_results] == [None, "api-call-2"]
        history = json.loads(json.dumps(conv.export()))["messages"]
        replay, _ = build(driver, history)
        assert next(c for c in replay if c["role"] == "model")["parts"] == model["parts"]
        assert history[1]["content"] == "Checking"
        assert types.Part.model_validate(history[-1]["_google_parts"][0]).thought_signature == b"\xfffinal"

    @pytest.mark.parametrize("cls", DRIVERS)
    @pytest.mark.parametrize("streaming", [False, True])
    @pytest.mark.asyncio
    async def test_plain_response_state_and_context_size(self, cls, streaming):
        driver, api = make_driver(cls)
        is_async = cls.__name__.startswith("Async")
        conv = (AsyncConversation if is_async else Conversation)(driver=driver)
        if streaming:
            if is_async:

                async def stream():
                    yield response()

                api.generate_content_stream = AsyncMock(return_value=stream())
                assert "".join([part async for part in conv.ask_stream("Hello")]) == "Done"
            else:
                api.generate_content_stream.return_value = iter([response()])
                assert "".join(conv.ask_stream("Hello")) == "Done"
        else:
            api.generate_content = (
                AsyncMock(return_value=response()) if is_async else MagicMock(return_value=response())
            )
            assert (await conv.ask("Hello") if is_async else conv.ask("Hello")) == "Done"
        restored = json.loads(json.dumps(conv.messages))
        contents, _ = build(driver, restored)
        assert contents[-1]["parts"][0].thought_signature == b"\xfffinal"
        conv._accumulate_usage({"prompt_tokens": 90000, "initial_prompt_tokens": 1000})
        assert conv._last_prompt_tokens == 1000

    def test_only_selected_candidate_executes(self):
        from prompture.drivers._google_genai import tool_response

        resp = signed_tool_response()
        resp.candidates.append(
            types.Candidate(
                content=types.Content(
                    parts=[types.Part(function_call=types.FunctionCall(name="must_not_execute", args={}))]
                )
            )
        )
        result = tool_response(resp, {})
        assert [call["name"] for call in result["tool_calls"]] == ["double", "double"]


class TestArgonPricing:
    """Catalog and introductory prices remain correct across the date boundary."""

    @pytest.mark.parametrize("provider", ["google", "google_vertexai"])
    def test_price_transition_without_reload(self, provider, offline_rates):
        source = pricing.LocalKBPricingSource()
        assert source.get_rates(provider, "gemini-4-argon") == {"input": 2.0, "output": 10.0, "cache_read": 0.1}
        offline_rates.now.return_value = datetime(2026, 12, 1, tzinfo=timezone.utc)
        assert source.get_rates(provider, "gemini-4-argon") == {"input": 4.0, "output": 20.0, "cache_read": 0.2}
        caps = model_rates.get_model_capabilities(provider, "gemini-4-argon")
        assert caps.supports_tool_use and caps.supports_structured_output and caps.supports_vision
        assert not caps.supports_temperature
        assert caps.context_window == 1048576
        assert caps.max_output_tokens == 65536
        assert "gemini-4-argon" in model_rates.get_kb_models_for_provider(provider)
