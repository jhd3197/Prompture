"""Tests for the CachiBot.ai API with GPT-5.x models.

Validates that the CachiBot proxy correctly handles openai/gpt-5.1 and
openai/gpt-5.2 model strings — driver resolution, model config, basic
generation, streaming, and tool use.

Findings so far:
  - Both models return 502 Bad Gateway from cachibot.ai (Cloudflare page)
  - Both models work fine directly against OpenAI API
  - gpt-5-mini works fine through cachibot.ai
  - gpt-5.1/5.2 need tokens_param="max_completion_tokens" (set in
    rates/openai.json capabilities KB)
  - gpt-5.1/5.2 are NOT reasoning models (reasoning_tokens=0), so they
    DO support temperature (unlike gpt-5-mini)
  - The 502 is server-side: likely the models are missing from the
    CachiBotWebsite's ModelToggle table, or the server crashes during
    model config resolution
"""

import os

import httpx
import pytest
import requests

from prompture.drivers.async_cachibot_driver import AsyncCachiBotDriver
from prompture.drivers.cachibot_driver import CachiBotDriver, _resolve_model

# Models under test
GPT_5_MODELS = ["openai/gpt-5.1", "openai/gpt-5.2"]
# Known working model for comparison
GPT_5_WORKING = "openai/gpt-5-mini"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_cachibot_key() -> str | None:
    return os.getenv("CACHIBOT_API_KEY")


def _get_openai_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


def _skip_no_cachibot_key():
    if not _get_cachibot_key():
        pytest.skip("CACHIBOT_API_KEY not set")


def _skip_no_openai_key():
    if not _get_openai_key():
        pytest.skip("OPENAI_API_KEY not set")


def _make_sync_driver(model: str) -> CachiBotDriver:
    return CachiBotDriver(api_key=_get_cachibot_key(), model=model)


def _make_async_driver(model: str) -> AsyncCachiBotDriver:
    return AsyncCachiBotDriver(api_key=_get_cachibot_key(), model=model)


def _raw_api_call(model: str, token_param: str = "max_tokens") -> requests.Response:
    """Make a raw HTTP call to cachibot.ai/api/v1/chat/completions."""
    key = _get_cachibot_key()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi"}],
        token_param: 50,
    }
    return requests.post(
        "https://cachibot.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# Unit tests — model resolution (no API call needed)
# ---------------------------------------------------------------------------


class TestModelResolution:
    """Verify _resolve_model correctly splits GPT-5.x model strings."""

    @pytest.mark.parametrize("model", GPT_5_MODELS)
    def test_resolve_returns_correct_provider(self, model):
        api_model, provider, name = _resolve_model(model)
        assert provider == "openai"
        assert api_model == model

    @pytest.mark.parametrize("model", GPT_5_MODELS)
    def test_resolve_extracts_model_name(self, model):
        _, _, name = _resolve_model(model)
        expected = model.split("/", 1)[1]
        assert name == expected


class TestModelConfig:
    """Verify model config resolution for GPT-5.x (tokens_param, temperature).

    The capabilities KB (rates/openai.json) must include gpt-5.1 and gpt-5.2
    with tokens_param="max_completion_tokens" because OpenAI rejects
    "max_tokens" for these models.  The CachiBot Website server uses
    the OpenAI driver directly, so this is the critical config.
    """

    @pytest.mark.parametrize("model_id", ["gpt-5.1", "gpt-5.2"])
    def test_openai_driver_has_model_config(self, model_id):
        """gpt-5.1/5.2 must resolve a config via _get_model_config."""
        from prompture.drivers.openai_driver import OpenAIDriver

        driver = OpenAIDriver(model=model_id, api_key="test-key")
        config = driver._get_model_config("openai", model_id)
        assert config is not None, f"{model_id} config is None"
        assert "tokens_param" in config
        print(f"\n  {model_id} config: {config}")

    @pytest.mark.parametrize("model_id", ["gpt-5.1", "gpt-5.2"])
    def test_tokens_param_is_max_completion_tokens(self, model_id):
        """GPT-5 models need max_completion_tokens, not max_tokens."""
        from prompture.drivers.openai_driver import OpenAIDriver

        driver = OpenAIDriver(model=model_id, api_key="test-key")
        config = driver._get_model_config("openai", model_id)
        tokens_param = config["tokens_param"]
        print(f"\n  {model_id}: tokens_param = {tokens_param}")
        assert tokens_param == "max_completion_tokens", (
            f"{model_id} has tokens_param='{tokens_param}', expected 'max_completion_tokens'"
        )

    @pytest.mark.parametrize("model_id", ["gpt-5.1", "gpt-5.2"])
    def test_supports_temperature(self, model_id):
        """Check temperature support resolution from capabilities KB."""
        from prompture.drivers.openai_driver import OpenAIDriver

        driver = OpenAIDriver(model=model_id, api_key="test-key")
        config = driver._get_model_config("openai", model_id)
        print(f"\n  {model_id}: supports_temperature={config['supports_temperature']}")
        # gpt-5.1/5.2 are NOT reasoning models — they support temperature
        assert config["supports_temperature"] is True

    def test_gpt5_mini_has_correct_config(self):
        """gpt-5-mini config is correct via _get_model_config."""
        from prompture.drivers.openai_driver import OpenAIDriver

        driver = OpenAIDriver(model="gpt-5-mini", api_key="test-key")
        config = driver._get_model_config("openai", "gpt-5-mini")
        assert config["tokens_param"] == "max_completion_tokens"
        assert config["supports_temperature"] is False
        print(f"\n  gpt-5-mini config: {config}")

    @pytest.mark.parametrize("model", GPT_5_MODELS)
    def test_cachibot_driver_does_not_crash(self, model):
        """CachiBot proxy driver config lookup should not crash."""
        driver = _make_sync_driver(model)
        _, provider, name = _resolve_model(model)
        config = driver._get_model_config(provider, name)
        assert "tokens_param" in config
        assert "supports_temperature" in config


class TestCostCalculation:
    """Verify cost calculation doesn't crash for GPT-5.x models."""

    @pytest.mark.parametrize("model", GPT_5_MODELS)
    def test_calculate_cost_returns_float(self, model):
        driver = _make_sync_driver(model)
        _, provider, name = _resolve_model(model)
        cost = driver._calculate_cost(provider, name, 100, 50)
        assert isinstance(cost, float)
        print(f"\n  {model}: cost for 100+50 tokens = ${cost}")
        if cost == 0.0:
            pytest.xfail(f"{model} has $0.00 cost — no pricing data.")


# ---------------------------------------------------------------------------
# Integration: raw HTTP calls to cachibot.ai
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRawAPI:
    """Raw HTTP requests to cachibot.ai to diagnose the 502 error."""

    def test_gpt5_mini_works(self):
        """Baseline: gpt-5-mini should return 200."""
        _skip_no_cachibot_key()
        resp = _raw_api_call(GPT_5_WORKING)
        print(f"\n  gpt-5-mini status: {resp.status_code}")
        print(f"  body: {resp.text[:300]}")
        assert resp.status_code == 200

    @pytest.mark.parametrize("model", GPT_5_MODELS)
    def test_returns_502(self, model):
        """Document that gpt-5.1/5.2 currently return 502."""
        _skip_no_cachibot_key()
        resp = _raw_api_call(model)
        print(f"\n  {model} status: {resp.status_code}")
        # Try to get error detail from JSON response
        try:
            body = resp.json()
            print(f"  error detail: {body.get('detail', body)}")
        except Exception:
            # Cloudflare HTML page
            print(f"  body (truncated): {resp.text[:200]}")
        # This documents the current broken state
        assert resp.status_code != 200, "Model is actually working now!"

    @pytest.mark.parametrize(
        "model,param",
        [
            ("openai/gpt-5.1", "max_tokens"),
            ("openai/gpt-5.1", "max_completion_tokens"),
            ("openai/gpt-5.2", "max_tokens"),
            ("openai/gpt-5.2", "max_completion_tokens"),
        ],
    )
    def test_both_token_params_fail(self, model, param):
        """Both max_tokens and max_completion_tokens fail — the issue is server-side."""
        _skip_no_cachibot_key()
        resp = _raw_api_call(model, token_param=param)
        print(f"\n  {model} with {param}: status {resp.status_code}")
        # Both should fail since the server crashes before reaching OpenAI
        assert resp.status_code != 200


# ---------------------------------------------------------------------------
# Integration: direct OpenAI comparison
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDirectOpenAI:
    """Test GPT-5.x directly via OpenAI (proves models work, issue is CachiBot)."""

    @pytest.mark.parametrize("model_id", ["gpt-5.1", "gpt-5.2"])
    def test_direct_openai_works(self, model_id):
        """These models work fine when hitting OpenAI directly."""
        _skip_no_openai_key()
        from prompture.drivers.openai_driver import OpenAIDriver

        driver = OpenAIDriver(api_key=_get_openai_key(), model=model_id)
        result = driver.generate("Say hi in one word.", {"max_tokens": 50})

        assert "text" in result
        assert len(result["text"].strip()) > 0
        print(f"\n  Direct {model_id}: {result['text'][:100]!r}")
        meta = result["meta"]
        print(f"  tokens: {meta['prompt_tokens']}/{meta['completion_tokens']}")

    @pytest.mark.parametrize("model_id", ["gpt-5.1", "gpt-5.2"])
    def test_direct_openai_not_reasoning(self, model_id):
        """Verify gpt-5.1/5.2 are NOT reasoning models (reasoning_tokens=0)."""
        _skip_no_openai_key()
        from openai import OpenAI

        client = OpenAI(api_key=_get_openai_key())
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say hi"}],
            max_completion_tokens=50,
        )
        reasoning = resp.usage.completion_tokens_details.reasoning_tokens
        print(f"\n  {model_id}: reasoning_tokens = {reasoning}")
        assert reasoning == 0, (
            f"{model_id} has reasoning_tokens={reasoning} — "
            f"it IS a reasoning model, supports_temperature should be False"
        )


# ---------------------------------------------------------------------------
# Integration: CachiBot driver tests (will fail until 502 is fixed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSyncGeneration:
    """Test sync CachiBot driver with GPT-5.x models."""

    @pytest.mark.parametrize("model", GPT_5_MODELS)
    def test_basic_generate(self, model):
        _skip_no_cachibot_key()
        driver = _make_sync_driver(model)
        result = driver.generate("Say hello in one word.", {"max_tokens": 50})

        assert "text" in result
        assert len(result["text"].strip()) > 0
        meta = result["meta"]
        print(f"\n  {model}: {result['text'][:100]!r}")
        print(f"  tokens: {meta.get('prompt_tokens')}/{meta.get('completion_tokens')}")

    @pytest.mark.parametrize("model", GPT_5_MODELS)
    def test_generate_messages(self, model):
        _skip_no_cachibot_key()
        driver = _make_sync_driver(model)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2? Answer with just the number."},
        ]
        result = driver.generate_messages(messages, {"max_tokens": 50})
        assert "text" in result
        assert "4" in result["text"]


@pytest.mark.integration
class TestSyncStreaming:
    """Test sync streaming with GPT-5.x via CachiBot proxy."""

    @pytest.mark.parametrize("model", GPT_5_MODELS)
    def test_streaming_produces_chunks(self, model):
        _skip_no_cachibot_key()
        driver = _make_sync_driver(model)
        messages = [{"role": "user", "content": "Count from 1 to 5."}]

        chunks = list(driver.generate_messages_stream(messages, {"max_tokens": 100}))
        deltas = [c for c in chunks if c["type"] == "delta"]
        dones = [c for c in chunks if c["type"] == "done"]
        assert len(deltas) > 0
        assert len(dones) == 1
        print(f"\n  {model} streamed: {dones[0]['text'][:100]!r}")


@pytest.mark.integration
class TestSyncToolUse:
    """Test tool calling with GPT-5.x via CachiBot proxy."""

    @pytest.mark.parametrize("model", GPT_5_MODELS)
    def test_tool_call(self, model):
        _skip_no_cachibot_key()
        driver = _make_sync_driver(model)
        messages = [{"role": "user", "content": "What is the weather in Lima, Peru?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "country": {"type": "string", "description": "Country code"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        result = driver.generate_messages_with_tools(messages, tools, {"max_tokens": 256})
        assert "tool_calls" in result
        if result["tool_calls"]:
            assert result["tool_calls"][0]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# Async integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAsyncGeneration:
    """Test async CachiBot driver with GPT-5.x models."""

    @pytest.mark.parametrize("model", GPT_5_MODELS)
    @pytest.mark.asyncio
    async def test_async_generate(self, model):
        _skip_no_cachibot_key()
        driver = _make_async_driver(model)
        result = await driver.generate("Say hello in one word.", {"max_tokens": 50})
        assert "text" in result
        assert len(result["text"].strip()) > 0

    @pytest.mark.parametrize("model", GPT_5_MODELS)
    @pytest.mark.asyncio
    async def test_async_streaming(self, model):
        _skip_no_cachibot_key()
        driver = _make_async_driver(model)
        messages = [{"role": "user", "content": "Count from 1 to 3."}]

        chunks = []
        async for chunk in driver.generate_messages_stream(messages, {"max_tokens": 100}):
            chunks.append(chunk)

        deltas = [c for c in chunks if c["type"] == "delta"]
        dones = [c for c in chunks if c["type"] == "done"]
        assert len(deltas) > 0
        assert len(dones) == 1

    @pytest.mark.parametrize("model", GPT_5_MODELS)
    @pytest.mark.asyncio
    async def test_async_tool_call(self, model):
        _skip_no_cachibot_key()
        driver = _make_async_driver(model)
        messages = [{"role": "user", "content": "What is the weather in Lima?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        result = await driver.generate_messages_with_tools(messages, tools, {"max_tokens": 256})
        assert "tool_calls" in result
