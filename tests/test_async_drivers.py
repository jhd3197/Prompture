"""Tests for async driver implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prompture.drivers.async_base import AsyncDriver
from prompture.infra.cost_mixin import CostMixin

# ---------------------------------------------------------------------------
# AsyncDriver base
# ---------------------------------------------------------------------------


class TestAsyncDriverBase:
    async def test_generate_not_implemented(self):
        driver = AsyncDriver()
        with pytest.raises(NotImplementedError):
            await driver.generate("test", {})


# ---------------------------------------------------------------------------
# CostMixin
# ---------------------------------------------------------------------------


class TestCostMixin:
    def test_calculate_cost_with_live_rates(self):
        mixin = CostMixin()
        with patch("prompture.infra.model_rates.get_model_rates") as mock_rates:
            mock_rates.return_value = {"input": 10.0, "output": 30.0}
            cost = mixin._calculate_cost("openai", "gpt-4", 1000, 500)
            # (1000 / 1_000_000) * 10.0 + (500 / 1_000_000) * 30.0
            expected = round(0.01 + 0.015, 6)
            assert cost == expected

    def test_calculate_cost_fallback_to_model_pricing(self):
        class TestDriver(CostMixin):
            MODEL_PRICING = {"test-model": {"prompt": 0.01, "completion": 0.02}}

        driver = TestDriver()
        with patch("prompture.infra.model_rates.get_model_rates", return_value=None):
            cost = driver._calculate_cost("test", "test-model", 1000, 500)
            # (1000 / 1000) * 0.01 + (500 / 1000) * 0.02
            expected = round(0.01 + 0.01, 6)
            assert cost == expected

    def test_calculate_cost_custom_pricing_unit(self):
        class TestDriver(CostMixin):
            _PRICING_UNIT = 1_000_000
            MODEL_PRICING = {"test-model": {"prompt": 3.0, "completion": 15.0}}

        driver = TestDriver()
        with patch("prompture.infra.model_rates.get_model_rates", return_value=None):
            cost = driver._calculate_cost("test", "test-model", 1000, 500)
            # (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
            expected = round(0.003 + 0.0075, 6)
            assert cost == expected

    def test_calculate_cost_unknown_model_returns_zero(self):
        mixin = CostMixin()
        with patch("prompture.infra.model_rates.get_model_rates", return_value=None):
            cost = mixin._calculate_cost("unknown", "unknown-model", 1000, 500)
            assert cost == 0.0


# ---------------------------------------------------------------------------
# AsyncOpenAIDriver
# ---------------------------------------------------------------------------


class TestAsyncOpenAIDriver:
    @patch("prompture.drivers.async_openai_driver.AsyncOpenAI")
    async def test_generate(self, mock_openai_cls):
        from prompture.drivers.async_openai_driver import AsyncOpenAIDriver

        # Setup mock
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        mock_message = MagicMock()
        mock_message.content = '{"result": "ok"}'

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_resp = MagicMock()
        mock_resp.usage = mock_usage
        mock_resp.choices = [mock_choice]
        mock_resp.model_dump.return_value = {"id": "test"}

        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        driver = AsyncOpenAIDriver(api_key="test-key")
        with patch.object(driver, "_calculate_cost", return_value=0.001):
            result = await driver.generate("test prompt", {})

        assert result["text"] == '{"result": "ok"}'
        assert result["meta"]["prompt_tokens"] == 10
        assert result["meta"]["completion_tokens"] == 5


# ---------------------------------------------------------------------------
# AsyncClaudeDriver
# ---------------------------------------------------------------------------


class TestAsyncClaudeDriver:
    @patch("prompture.drivers.async_claude_driver.anthropic")
    async def test_generate(self, mock_anthropic):
        from prompture.drivers.async_claude_driver import AsyncClaudeDriver

        mock_client = AsyncMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        mock_usage = MagicMock()
        mock_usage.input_tokens = 12
        mock_usage.output_tokens = 8

        mock_content = MagicMock()
        mock_content.text = '{"name": "Alice"}'

        mock_resp = MagicMock()
        mock_resp.usage = mock_usage
        mock_resp.content = [mock_content]
        mock_resp.__iter__ = lambda self: iter([("usage", mock_usage)])

        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        driver = AsyncClaudeDriver(api_key="test-key")
        with patch.object(driver, "_calculate_cost", return_value=0.002):
            result = await driver.generate("test", {"max_tokens": 256})

        assert result["text"] == '{"name": "Alice"}'
        assert result["meta"]["prompt_tokens"] == 12


# ---------------------------------------------------------------------------
# AsyncOllamaDriver
# ---------------------------------------------------------------------------


class TestAsyncOllamaDriver:
    @patch("prompture.drivers.async_ollama_driver.httpx.AsyncClient")
    async def test_generate(self, mock_client_cls):
        from prompture.drivers.async_ollama_driver import AsyncOllamaDriver

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": '{"answer": "yes"}',
            "prompt_eval_count": 20,
            "eval_count": 10,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        driver = AsyncOllamaDriver(endpoint="http://localhost:11434/api/generate")
        result = await driver.generate("test", {})

        assert result["text"] == '{"answer": "yes"}'
        assert result["meta"]["cost"] == 0.0
        assert result["meta"]["prompt_tokens"] == 20


# ---------------------------------------------------------------------------
# AsyncGrokDriver
# ---------------------------------------------------------------------------


class TestAsyncGrokDriver:
    @patch("prompture.drivers.async_grok_driver.httpx.AsyncClient")
    async def test_generate(self, mock_client_cls):
        from prompture.drivers.async_grok_driver import AsyncGrokDriver

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"key": "value"}'}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 7, "total_tokens": 22},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        driver = AsyncGrokDriver(api_key="test-key")
        with patch.object(driver, "_calculate_cost", return_value=0.0001):
            result = await driver.generate("test", {})

        assert result["text"] == '{"key": "value"}'
        assert result["meta"]["prompt_tokens"] == 15


# ---------------------------------------------------------------------------
# AsyncLocalHTTPDriver
# ---------------------------------------------------------------------------


class TestAsyncLocalHTTPDriver:
    @patch("prompture.drivers.async_local_http_driver.httpx.AsyncClient")
    async def test_generate_normalized(self, mock_client_cls):
        from prompture.drivers.async_local_http_driver import AsyncLocalHTTPDriver

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "hello", "prompt_tokens": 5, "completion_tokens": 3}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        driver = AsyncLocalHTTPDriver()
        result = await driver.generate("test", {})

        assert result["text"] == "hello"
        assert result["meta"]["cost"] == 0.0

    @patch("prompture.drivers.async_local_http_driver.httpx.AsyncClient")
    async def test_generate_passthrough(self, mock_client_cls):
        from prompture.drivers.async_local_http_driver import AsyncLocalHTTPDriver

        passthrough = {"text": "direct", "meta": {"prompt_tokens": 1, "completion_tokens": 2}}
        mock_response = MagicMock()
        mock_response.json.return_value = passthrough
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        driver = AsyncLocalHTTPDriver()
        result = await driver.generate("test", {})

        assert result == passthrough


# ---------------------------------------------------------------------------
# Async registry
# ---------------------------------------------------------------------------


class TestAsyncRegistry:
    def test_get_async_driver_for_model_invalid_provider(self):
        from prompture.drivers.async_registry import get_async_driver_for_model

        with pytest.raises(ValueError, match="Unsupported provider"):
            get_async_driver_for_model("nonexistent/model")

    def test_get_async_driver_for_model_empty_string(self):
        from prompture.drivers.async_registry import get_async_driver_for_model

        with pytest.raises(ValueError, match="cannot be empty"):
            get_async_driver_for_model("")
