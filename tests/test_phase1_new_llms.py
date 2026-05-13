"""Unit tests for Phase 1 LLM drivers: Mistral, DeepSeek, OpenAI-compatible.

All HTTP calls are mocked — no live network access required.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from prompture.drivers import get_driver_for_model
from prompture.drivers.async_deepseek_driver import AsyncDeepSeekDriver
from prompture.drivers.async_mistral_driver import AsyncMistralDriver
from prompture.drivers.async_openai_compatible_driver import (
    AsyncOpenAICompatibleDriver,
)
from prompture.drivers.deepseek_driver import DeepSeekDriver
from prompture.drivers.mistral_driver import MistralDriver
from prompture.drivers.openai_compatible_driver import (
    OPENAI_COMPATIBLE_PROFILES,
    OpenAICompatibleDriver,
    _parse_profile_and_model,
)


def _mk_resp(content: str, *, reasoning: str | None = None, prompt: int = 10, completion: int = 5) -> MagicMock:
    """Build a fake requests.Response-like object."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    payload = {
        "id": "cmpl-test",
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status = MagicMock()
    mock_resp.status_code = 200
    return mock_resp


# ───────────────────────── Mistral ──────────────────────────


class TestMistralDriver:
    def test_generate_returns_text_and_usage(self):
        d = MistralDriver(api_key="sk-test", model="mistral-large-latest")
        with patch("prompture.drivers.mistral_driver.requests.post", return_value=_mk_resp("hi")) as mp:
            result = d.generate("hello", {})
        assert result["text"] == "hi"
        meta = result["meta"]
        assert meta["prompt_tokens"] == 10
        assert meta["completion_tokens"] == 5
        assert meta["total_tokens"] == 15
        # Pricing is in rates/mistral.json so cost should be > 0
        assert meta["cost"] > 0
        # Posted to the right URL
        called_url = mp.call_args.args[0]
        assert called_url == "https://api.mistral.ai/v1/chat/completions"

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
            MistralDriver(api_key=None, model="mistral-large-latest")

    def test_async_generate(self):
        d = AsyncMistralDriver(api_key="sk-test", model="mistral-large-latest")

        async def _fake_post(*args, **kwargs):
            return _mk_resp("hi-async")

        with patch("httpx.AsyncClient.post", side_effect=_fake_post):
            result = asyncio.run(d.generate("hello", {}))
        assert result["text"] == "hi-async"
        assert result["meta"]["cost"] > 0


# ───────────────────────── DeepSeek ─────────────────────────


class TestDeepSeekDriver:
    def test_generate_returns_text_and_usage(self):
        d = DeepSeekDriver(api_key="sk-test", model="deepseek-chat")
        with patch("prompture.drivers.deepseek_driver.requests.post", return_value=_mk_resp("hi")) as mp:
            result = d.generate("hello", {})
        assert result["text"] == "hi"
        meta = result["meta"]
        assert meta["prompt_tokens"] == 10
        assert meta["total_tokens"] == 15
        called_url = mp.call_args.args[0]
        assert called_url == "https://api.deepseek.com/v1/chat/completions"

    def test_reasoner_surfaces_reasoning_content(self):
        d = DeepSeekDriver(api_key="sk-test", model="deepseek-reasoner")
        fake = _mk_resp("final answer", reasoning="step-by-step thinking")
        with patch("prompture.drivers.deepseek_driver.requests.post", return_value=fake):
            result = d.generate("solve x", {})
        assert result["text"] == "final answer"
        assert "reasoning_content" in result
        assert result["reasoning_content"] == "step-by-step thinking"

    def test_reasoner_falls_back_to_reasoning_when_content_empty(self):
        d = DeepSeekDriver(api_key="sk-test", model="deepseek-reasoner")
        fake = _mk_resp("", reasoning="only reasoning available")
        with patch("prompture.drivers.deepseek_driver.requests.post", return_value=fake):
            result = d.generate("x", {})
        assert result["text"] == "only reasoning available"
        assert result["reasoning_content"] == "only reasoning available"

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            DeepSeekDriver(api_key=None, model="deepseek-chat")

    def test_async_generate(self):
        d = AsyncDeepSeekDriver(api_key="sk-test", model="deepseek-chat")

        async def _fake_post(*args, **kwargs):
            return _mk_resp("hi-async")

        with patch("httpx.AsyncClient.post", side_effect=_fake_post):
            result = asyncio.run(d.generate("hello", {}))
        assert result["text"] == "hi-async"


# ─────────────────── OpenAI-compatible ──────────────────────


class TestOpenAICompatibleDriver:
    def test_profile_resolves_endpoint(self):
        d = OpenAICompatibleDriver(api_key="sk-test", profile="fireworks", model="some-model")
        assert d.endpoint == "https://api.fireworks.ai/inference/v1"
        assert d.profile == "fireworks"

    def test_explicit_endpoint_overrides_profile(self):
        d = OpenAICompatibleDriver(
            api_key="sk-test",
            profile="fireworks",
            endpoint="https://example.com/v1",
            model="some-model",
        )
        # Explicit endpoint wins
        assert d.endpoint == "https://example.com/v1"

    def test_neither_profile_nor_endpoint_raises(self):
        with pytest.raises(ValueError, match="profile.*endpoint|endpoint.*profile"):
            OpenAICompatibleDriver(api_key="sk-test", model="some-model")

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown openai_compatible profile"):
            OpenAICompatibleDriver(api_key="sk-test", profile="bogus", model="x")

    def test_model_string_auto_detects_profile(self):
        # When called from the registry, model looks like "fireworks/accounts/..."
        # and no profile is passed — the driver should detect it.
        d = OpenAICompatibleDriver(
            api_key="sk-test",
            model="fireworks/accounts/fireworks/models/llama-v3p1-70b-instruct",
        )
        assert d.profile == "fireworks"
        assert d.model == "accounts/fireworks/models/llama-v3p1-70b-instruct"

    def test_generate_calls_profile_endpoint(self):
        d = OpenAICompatibleDriver(
            api_key="sk-test",
            profile="fireworks",
            model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        )
        with patch(
            "prompture.drivers.openai_compatible_driver.requests.post",
            return_value=_mk_resp("ok"),
        ) as mp:
            result = d.generate("hi", {})
        called_url = mp.call_args.args[0]
        assert called_url == "https://api.fireworks.ai/inference/v1/chat/completions"
        # Pricing is in rates/fireworks.json → cost > 0, no pricing_unknown flag
        assert result["meta"]["cost"] > 0
        assert "pricing_unknown" not in result["meta"]

    def test_unknown_endpoint_sets_pricing_unknown(self):
        d = OpenAICompatibleDriver(
            api_key="sk-test",
            endpoint="https://example.com/v1",
            model="some-unknown-model",
        )
        with patch(
            "prompture.drivers.openai_compatible_driver.requests.post",
            return_value=_mk_resp("ok"),
        ) as mp:
            result = d.generate("hi", {})
        called_url = mp.call_args.args[0]
        assert called_url == "https://example.com/v1/chat/completions"
        assert result["meta"]["cost"] == 0
        assert result["meta"].get("pricing_unknown") is True

    def test_per_call_endpoint_override(self):
        d = OpenAICompatibleDriver(api_key="sk-test", profile="fireworks", model="some-model")
        with patch(
            "prompture.drivers.openai_compatible_driver.requests.post",
            return_value=_mk_resp("ok"),
        ) as mp:
            d.generate("hi", {"endpoint": "https://other.example.com/v1"})
        called_url = mp.call_args.args[0]
        assert called_url == "https://other.example.com/v1/chat/completions"

    def test_async_construct_and_generate(self):
        d = AsyncOpenAICompatibleDriver(api_key="sk-test", profile="fireworks", model="some-model")
        assert d.endpoint == "https://api.fireworks.ai/inference/v1"

        async def _fake_post(*args, **kwargs):
            return _mk_resp("hi-async")

        with patch("httpx.AsyncClient.post", side_effect=_fake_post):
            result = asyncio.run(d.generate("hello", {}))
        assert result["text"] == "hi-async"

    def test_profiles_table_has_all_eight(self):
        expected = {
            "fireworks",
            "together",
            "cerebras",
            "sambanova",
            "perplexity",
            "nvidia",
            "deepinfra",
            "siliconflow",
        }
        assert set(OPENAI_COMPATIBLE_PROFILES) == expected

    def test_parse_profile_and_model_helper(self):
        assert _parse_profile_and_model("fireworks/llama-3") == (
            "fireworks",
            "llama-3",
        )
        assert _parse_profile_and_model("unknown/llama-3") == (None, "unknown/llama-3")
        assert _parse_profile_and_model("just-a-model") == (None, "just-a-model")


# ───────────────────────── Registry ─────────────────────────


class TestDriverRegistryRouting:
    def test_mistral_routes_correctly(self):
        drv = get_driver_for_model("mistral/mistral-large-latest", api_key="sk-test")
        assert isinstance(drv, MistralDriver)
        assert drv.model == "mistral-large-latest"

    def test_deepseek_routes_correctly(self):
        drv = get_driver_for_model("deepseek/deepseek-chat", api_key="sk-test")
        assert isinstance(drv, DeepSeekDriver)
        assert drv.model == "deepseek-chat"

    def test_openai_compatible_routes_to_profile(self):
        # The registry strips the leading "openai_compatible/" and passes
        # the rest as model — the driver auto-detects the profile.
        drv = get_driver_for_model("openai_compatible/fireworks/accounts/fireworks/models/llama-v3p1-70b-instruct")
        assert isinstance(drv, OpenAICompatibleDriver)
        assert drv.profile == "fireworks"
        assert drv.model == "accounts/fireworks/models/llama-v3p1-70b-instruct"

    def test_mistralai_alias(self):
        drv = get_driver_for_model("mistralai/mistral-small-latest", api_key="sk-test")
        assert isinstance(drv, MistralDriver)
        assert drv.model == "mistral-small-latest"
