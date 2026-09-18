"""Provider driver reporting contracts across buffered, streamed and tool calls."""

from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from prompture.agents.live_events import MessageStop, ToolUseStop
from prompture.drivers._usage_reporting import usage_meta
from prompture.drivers.async_claude_driver import AsyncClaudeDriver
from prompture.drivers.async_openai_driver import AsyncOpenAIDriver
from prompture.drivers.claude_driver import ClaudeDriver
from prompture.drivers.openai_driver import OpenAIDriver
from prompture.infra.cost_mixin import CostMixin

MESSAGES = [{"role": "user", "content": "Hello"}]
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]
RATES = {"input": 10.0, "output": 30.0, "cache_read": 1.0, "cache_write": 12.5}


class SDKObject(SimpleNamespace):
    """SDK-shaped object with explicit attributes and recursive serialization."""

    def model_dump(self):
        def convert(item):
            if isinstance(item, SDKObject):
                return item.model_dump()
            if isinstance(item, list):
                return [convert(value) for value in item]
            return item

        return {key: convert(value) for key, value in vars(self).items() if not key.startswith("_")}


async def async_items(items):
    for item in items:
        yield item


@pytest.fixture(autouse=True)
def stable_pricing(monkeypatch):
    monkeypatch.setattr("prompture.infra.model_rates.get_model_rates", lambda *args: dict(RATES))
    monkeypatch.setattr(
        "prompture.infra.pricing.get_model_pricing_rules", lambda *args: {"service_tiers": {"flex": 0.5}}
    )


def make_driver(monkeypatch, provider, asynchronous, *, response=None, chunks=None):
    classes = {
        ("openai", False): OpenAIDriver,
        ("openai", True): AsyncOpenAIDriver,
        ("claude", False): ClaudeDriver,
        ("claude", True): AsyncClaudeDriver,
    }
    cls = classes[provider, asynchronous]
    driver = cls.__new__(cls)
    driver.api_key = "unit-test"
    driver.model = "requested-model"
    driver.api = "chat_completions"
    driver._get_model_config = lambda *args: {"tokens_param": "max_tokens", "supports_temperature": True}
    driver._validate_model_capabilities = lambda *args, **kwargs: None
    if provider == "openai":
        value = (
            async_items(chunks)
            if asynchronous and chunks is not None
            else iter(chunks)
            if chunks is not None
            else response
        )
        create = AsyncMock(return_value=value) if asynchronous else MagicMock(return_value=value)
        driver.client = SDKObject(chat=SDKObject(completions=SDKObject(create=create)))
    else:
        create = AsyncMock(return_value=response) if asynchronous else MagicMock(return_value=response)

        @contextmanager
        def stream(**kwargs):
            yield iter(chunks or [])

        @asynccontextmanager
        async def astream(**kwargs):
            yield async_items(chunks or [])

        driver.client = SDKObject(messages=SDKObject(create=create, stream=astream if asynchronous else stream))
        if not asynchronous:
            monkeypatch.setattr("prompture.drivers.claude_driver.anthropic.Anthropic", lambda **kwargs: driver.client)
    return driver


def openai_usage():
    return SDKObject(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_tokens_details=SDKObject(cached_tokens=800),
        completion_tokens_details=SDKObject(
            reasoning_tokens=200, accepted_prediction_tokens=100, rejected_prediction_tokens=50
        ),
    )


def openai_response():
    return SDKObject(
        id="chat-1",
        _request_id="req-1",
        model="returned-model",
        service_tier="flex",
        usage=openai_usage(),
        choices=[SDKObject(message=SDKObject(content="hello", tool_calls=[]), finish_reason="stop")],
    )


def openai_chunks(*, usage=True, tools=False):
    calls = [SDKObject(index=0, id="call-1", function=SDKObject(name="lookup", arguments="{}"))] if tools else []
    result = [
        SDKObject(
            id="chat-1",
            model="returned-model",
            service_tier="flex",
            usage=None,
            choices=[
                SDKObject(
                    delta=SDKObject(content="hello", tool_calls=calls), finish_reason="tool_calls" if tools else "stop"
                )
            ],
        )
    ]
    if usage:
        result.append(SDKObject(choices=[], usage=openai_usage()))
    return result


def claude_usage():
    return SDKObject(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=200,
        cache_creation_input_tokens=300,
        cache_creation=SDKObject(ephemeral_5m_input_tokens=100, ephemeral_1h_input_tokens=200),
        service_tier="standard",
        inference_geo="global",
        server_tool_use=SDKObject(web_search_requests=2, web_fetch_requests=1),
    )


def claude_response():
    return SDKObject(
        id="msg-1",
        _request_id="req-1",
        model="returned-model",
        usage=claude_usage(),
        content=[SDKObject(type="text", text="hello")],
        stop_reason="end_turn",
    )


def claude_chunks(*, final_usage=True, tools=False):
    initial = SDKObject(
        input_tokens=100,
        output_tokens=1,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        cache_creation=SDKObject(ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=0),
    )
    result = [SDKObject(type="message_start", message=SDKObject(id="msg-1", model="returned-model", usage=initial))]
    if tools:
        result.extend(
            [
                SDKObject(
                    type="content_block_start",
                    index=0,
                    content_block=SDKObject(type="tool_use", id="call-1", name="lookup"),
                ),
                SDKObject(
                    type="content_block_delta", index=0, delta=SDKObject(type="input_json_delta", partial_json="{}")
                ),
                SDKObject(type="content_block_stop", index=0),
            ]
        )
    else:
        result.append(SDKObject(type="content_block_delta", index=0, delta=SDKObject(type="text_delta", text="hello")))
    if final_usage:
        result.append(
            SDKObject(
                type="message_delta",
                usage=claude_usage(),
                delta=SDKObject(stop_reason="tool_use" if tools else "end_turn"),
            )
        )
    return result


class TestOpenAIUsageReporting:
    """OpenAI token subsets and identity survive every driver response path."""

    @pytest.mark.parametrize("asynchronous", [False, True])
    @pytest.mark.parametrize("tools", [False, True])
    async def test_buffered_counts_and_actual_service_tier(self, monkeypatch, asynchronous, tools):
        driver = make_driver(monkeypatch, "openai", asynchronous, response=openai_response())
        options = {"service_tier": "auto", "retry_attempt": 2, "fallback": True, "extraction_success": True}
        result = (
            driver.generate_messages_with_tools(MESSAGES, TOOLS, options)
            if tools
            else driver.generate("Hello", options)
        )
        if asynchronous:
            result = await result
        meta = result["meta"]
        assert meta["total_tokens"] == 1500
        assert meta["completion_tokens"] == 500
        assert meta["usage_details"]["reasoning_tokens"] == 200
        assert meta["usage_details"]["accepted_prediction_tokens"] == 100
        assert meta["usage_details"]["rejected_prediction_tokens"] == 50
        assert meta["cost"] == pytest.approx(0.0178 * 0.5)
        assert meta["service_tier"] == "flex"
        assert meta["requested_model"] == "requested-model"
        assert meta["returned_model"] == meta["model_name"] == "returned-model"
        assert meta["request_id"] == "req-1"
        assert meta["response_id"] == "chat-1"
        assert meta["retry_attempt"] == 2
        assert meta["fallback"] is True
        assert meta["extraction_success"] is True
        assert meta["raw_response"]["usage"]["completion_tokens"] == 500
        assert driver.client.chat.completions.create.call_args.kwargs["service_tier"] == "auto"

    @pytest.mark.parametrize("asynchronous", [False, True])
    @pytest.mark.parametrize("tools", [False, True])
    @pytest.mark.parametrize("has_usage", [False, True])
    async def test_stream_usage_and_missing_final_usage(self, monkeypatch, asynchronous, tools, has_usage):
        driver = make_driver(monkeypatch, "openai", asynchronous, chunks=openai_chunks(usage=has_usage, tools=tools))
        stream = (
            driver.generate_messages_with_tools_stream(MESSAGES, TOOLS, {})
            if tools
            else driver.generate_messages_stream(MESSAGES, {})
        )
        events = [event async for event in stream] if asynchronous else list(stream)
        meta = events[-1].usage if tools else events[-1]["meta"]
        assert meta["model_name"] == "returned-model"
        assert meta["response_id"] == "chat-1"
        assert meta["service_tier"] == "flex"
        assert meta["usage_complete"] is has_usage
        if has_usage:
            assert meta["cost"] == pytest.approx(0.0089)
            assert meta["usage_details"]["reasoning_tokens"] == 200
            assert meta["cost_status"] == "estimated"
        else:
            assert meta["cost_status"] in {"partial", "unknown"}
            assert "incomplete_usage" in meta["pricing"]["unpriced"]
        if tools:
            assert isinstance(events[-1], MessageStop)
            assert any(isinstance(event, ToolUseStop) and event.input == {} for event in events)


class TestClaudeUsageReporting:
    """Returned TTL and late usage counters override caller assumptions."""

    @pytest.mark.parametrize("asynchronous", [False, True])
    @pytest.mark.parametrize("tools", [False, True])
    async def test_buffered_mixed_ttl_and_hosted_search(self, monkeypatch, asynchronous, tools):
        driver = make_driver(monkeypatch, "claude", asynchronous, response=claude_response())
        options = {"cache_ttl": "1h", "cache_prompt": False}
        result = (
            driver.generate_messages_with_tools(MESSAGES, TOOLS, options)
            if tools
            else driver.generate("Hello", options)
        )
        if asynchronous:
            result = await result
        self.assert_complete(result["meta"])
        assert result["meta"]["request_id"] == "req-1"

    @pytest.mark.parametrize("asynchronous", [False, True])
    @pytest.mark.parametrize("tools", [False, True])
    @pytest.mark.parametrize("has_final", [False, True])
    async def test_stream_late_cache_counts_and_partial_usage(self, monkeypatch, asynchronous, tools, has_final):
        driver = make_driver(
            monkeypatch, "claude", asynchronous, chunks=claude_chunks(final_usage=has_final, tools=tools)
        )
        options = {"cache_ttl": "1h", "cache_prompt": False}
        stream = (
            driver.generate_messages_with_tools_stream(MESSAGES, TOOLS, options)
            if tools
            else driver.generate_messages_stream(MESSAGES, options)
        )
        events = [event async for event in stream] if asynchronous else list(stream)
        meta = events[-1].usage if tools else events[-1]["meta"]
        assert meta["usage_complete"] is has_final
        if has_final:
            self.assert_complete(meta)
        else:
            assert meta["prompt_tokens"] == 100
            assert meta["cost_status"] in {"partial", "unknown"}
            assert "incomplete_usage" in meta["pricing"]["unpriced"]
        if tools:
            assert any(isinstance(event, ToolUseStop) and event.input == {} for event in events)

    @staticmethod
    def assert_complete(meta):
        assert meta["prompt_tokens"] == 600
        assert meta["completion_tokens"] == 50
        assert meta["total_tokens"] == 650
        assert meta["model_name"] == meta["returned_model"] == "returned-model"
        assert meta["usage_details"]["cache_creation_5m_tokens"] == 100
        assert meta["usage_details"]["cache_creation_1h_tokens"] == 200
        assert meta["usage_details"]["server_tool_usage"] == {"web_search_requests": 2, "web_fetch_requests": 1}
        assert meta["cost_breakdown"]["cache_write_5m"] == pytest.approx(0.00125)
        assert meta["cost_breakdown"]["cache_write_1h"] == pytest.approx(0.004)
        assert meta["cost_breakdown"]["tools"] == pytest.approx(0.02)
        assert meta["cost"] == pytest.approx(0.02795)
        assert meta["cost_status"] == "estimated"


class TestUsageNormalizer:
    """Normalize SDK objects and dictionaries without counting missing attributes."""

    def test_dictionary_and_sdk_usage_are_equivalent(self):
        response = openai_response()
        obj = usage_meta(CostMixin(), "openai", "requested-model", response.usage, response=response)
        raw = response.model_dump()
        raw["_request_id"] = "req-1"
        assert obj == usage_meta(CostMixin(), "openai", "requested-model", raw["usage"], response=raw)

    @pytest.mark.parametrize("usage", [{}, {"prompt_tokens": 100}, None])
    def test_incomplete_usage_is_not_reported_as_complete(self, usage):
        meta = usage_meta(CostMixin(), "openai", "model", usage)
        assert meta["usage_complete"] is False
        assert meta["cost_status"] in {"unknown", "partial"}

    def test_unspecified_mock_cache_details_do_not_override_ttl(self):
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        usage.cache_creation_input_tokens = 300
        usage.cache_read_input_tokens = 0
        meta = usage_meta(CostMixin(), "claude", "model", usage, options={"cache_ttl": "1h"})
        assert meta["cost_breakdown"]["cache_write_1h"] == pytest.approx(0.006)
        assert "cache_creation_breakdown_mismatch" not in meta["pricing"]["unpriced"]

    def test_unknown_hosted_tools_are_reported_as_unpriced(self):
        meta = usage_meta(
            CostMixin(),
            "openai",
            "model",
            {"input_tokens": 100, "output_tokens": 50},
            response={"output": [{"type": "code_interpreter_call"}, {"type": "image_generation_call"}]},
            responses_api=True,
        )
        assert meta["usage_details"]["server_tool_usage"] == {"code_interpreter_call": 1, "image_generation_call": 1}
        assert meta["cost_status"] == "partial"
        assert set(meta["pricing"]["unpriced"]) >= {"tool:code_interpreter_call", "tool:image_generation_call"}

    def test_audio_usage_preserves_known_subtotal_and_marks_partial(self):
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {"audio_tokens": 10},
            "completion_tokens_details": {"audio_tokens": 20},
        }
        meta = usage_meta(CostMixin(), "openai", "model", usage)
        assert meta["cost"] > 0
        assert meta["cost_status"] == "partial"
        assert "audio_tokens" in meta["pricing"]["unpriced"]

    def test_file_search_call_fee_does_not_repeat_retrieval_tokens(self):
        meta = usage_meta(
            CostMixin(),
            "openai",
            "model",
            {"input_tokens": 100, "output_tokens": 50},
            response={"output": [{"type": "file_search_call"}, {"type": "file_search_call"}]},
            responses_api=True,
        )
        assert meta["cost_breakdown"]["tools"] == pytest.approx(0.005)
        assert meta["cost_breakdown"]["uncached_input"] == pytest.approx(0.001)
        assert meta["cost"] == pytest.approx(0.0075)
        assert meta["cost_status"] == "estimated"

    def test_openai_unknown_cache_write_rate_is_partial(self, monkeypatch):
        monkeypatch.setattr("prompture.infra.model_rates.get_model_rates", lambda *args: {"input": 10, "output": 30})
        meta = usage_meta(
            CostMixin(),
            "openai",
            "model",
            {"prompt_tokens": 100, "completion_tokens": 50, "prompt_tokens_details": {"cache_write_tokens": 20}},
        )
        assert meta["cost_status"] == "partial"
        assert "cache_write_rate" in meta["pricing"]["unpriced"]

    @pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5.5"])
    def test_standard_web_search_prices_call_but_never_adds_content_tokens(self, model):
        meta = usage_meta(
            CostMixin(),
            "openai",
            model,
            {"input_tokens": 100, "output_tokens": 50},
            response={"tools": [{"type": "web_search"}], "output": [{"type": "web_search_call"}]},
            responses_api=True,
        )
        assert meta["usage_details"]["server_tool_usage"] == {"web_search_requests": 1}
        assert meta["cost_breakdown"]["tools"] == pytest.approx(0.01)
        assert meta["cost_breakdown"]["uncached_input"] == pytest.approx(0.001)
        assert meta["cost"] == pytest.approx(0.0125)
        assert meta["cost_status"] == "partial"
        assert "web_search_content_tokens" in meta["pricing"]["unpriced"]
        assert meta["pricing"]["tool_rates_per_call"] == {"web_search_requests": 0.01}

    @pytest.mark.parametrize("reasoning,fee", [(True, 0.01), (False, 0.025), (None, 0)])
    def test_legacy_search_preview_uses_known_model_capability(self, monkeypatch, reasoning, fee):
        monkeypatch.setattr(
            "prompture.infra.model_rates.get_model_capabilities", lambda *args: SDKObject(is_reasoning=reasoning)
        )
        meta = usage_meta(
            CostMixin(),
            "openai",
            "model",
            {"input_tokens": 100, "output_tokens": 50},
            response={"tools": [{"type": "web_search_preview"}], "output": [{"type": "web_search_call"}]},
            responses_api=True,
        )
        assert meta["cost_breakdown"]["tools"] == pytest.approx(fee)
        assert meta["cost_status"] == "partial"

    def test_ambiguous_search_variant_cannot_be_silently_priced(self):
        meta = usage_meta(
            CostMixin(),
            "openai",
            "model",
            {"input_tokens": 100, "output_tokens": 50},
            response={"output": [{"type": "web_search_call"}]},
            responses_api=True,
        )
        assert meta["cost_breakdown"]["tools"] == 0
        assert "tool:web_search_call" in meta["pricing"]["unpriced"]

    @pytest.mark.parametrize("environment", ["container_auto", "container_reference", "local", None])
    def test_hosted_shell_costs_cannot_disappear_from_reporting(self, environment):
        tools = [{"type": "shell", "environment": {"type": environment}}] if environment else []
        meta = usage_meta(
            CostMixin(),
            "openai",
            "model",
            {"input_tokens": 100, "output_tokens": 50},
            response={"tools": tools, "output": [{"type": "shell_call"}]},
            responses_api=True,
        )
        if environment == "local":
            assert meta["cost_status"] == "estimated"
            assert "server_tool_usage" not in meta["usage_details"]
        else:
            assert meta["cost_status"] == "partial"
            assert meta["usage_details"]["server_tool_usage"] == {"shell_call": 1}
            assert "tool:shell_call" in meta["pricing"]["unpriced"]
