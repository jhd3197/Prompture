"""Tests for native JSON mode / structured output support."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from prompture.drivers.async_base import AsyncDriver
from prompture.extraction.core import ask_for_json
from prompture.drivers.base import Driver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockDriver(Driver):
    """Mock driver with configurable JSON mode capabilities."""

    supports_json_mode = False
    supports_json_schema = False

    def __init__(self, *, json_mode: bool = False, json_schema: bool = False, response_text: str = '{"name": "Alice"}'):
        self.supports_json_mode = json_mode
        self.supports_json_schema = json_schema
        self._response_text = response_text
        self.last_options: dict[str, Any] = {}
        self.last_prompt: str = ""

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.last_prompt = prompt
        self.last_options = dict(options)
        return {
            "text": self._response_text,
            "meta": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.0,
                "raw_response": {},
                "model_name": "mock",
            },
        }


SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name"],
}


# ---------------------------------------------------------------------------
# 1. Driver capability flag tests
# ---------------------------------------------------------------------------


class TestDriverCapabilityFlags:
    """Verify supports_json_mode and supports_json_schema are set correctly on each driver class."""

    def test_base_driver_defaults(self):
        assert Driver.supports_json_mode is False
        assert Driver.supports_json_schema is False

    def test_base_async_driver_defaults(self):
        assert AsyncDriver.supports_json_mode is False
        assert AsyncDriver.supports_json_schema is False

    def test_openai_driver_flags(self):
        from prompture.drivers.openai_driver import OpenAIDriver

        assert OpenAIDriver.supports_json_mode is True
        assert OpenAIDriver.supports_json_schema is True

    def test_azure_driver_flags(self):
        from prompture.drivers.azure_driver import AzureDriver

        assert AzureDriver.supports_json_mode is True
        assert AzureDriver.supports_json_schema is True

    def test_claude_driver_flags(self):
        from prompture.drivers.claude_driver import ClaudeDriver

        assert ClaudeDriver.supports_json_mode is True
        assert ClaudeDriver.supports_json_schema is True

    def test_google_driver_flags(self):
        from prompture.drivers.google_driver import GoogleDriver

        assert GoogleDriver.supports_json_mode is True
        assert GoogleDriver.supports_json_schema is True

    def test_groq_driver_flags(self):
        from prompture.drivers.groq_driver import GroqDriver

        assert GroqDriver.supports_json_mode is True
        assert GroqDriver.supports_json_schema is False

    def test_grok_driver_flags(self):
        from prompture.drivers.grok_driver import GrokDriver

        assert GrokDriver.supports_json_mode is True
        assert GrokDriver.supports_json_schema is False

    def test_openrouter_driver_flags(self):
        from prompture.drivers.openrouter_driver import OpenRouterDriver

        assert OpenRouterDriver.supports_json_mode is True
        assert OpenRouterDriver.supports_json_schema is True

    def test_ollama_driver_flags(self):
        from prompture.drivers.ollama_driver import OllamaDriver

        assert OllamaDriver.supports_json_mode is True
        assert OllamaDriver.supports_json_schema is True

    def test_lmstudio_driver_flags(self):
        from prompture.drivers.lmstudio_driver import LMStudioDriver

        assert LMStudioDriver.supports_json_mode is True
        assert LMStudioDriver.supports_json_schema is True

    def test_hugging_driver_flags(self):
        from prompture.drivers.hugging_driver import HuggingFaceDriver

        assert HuggingFaceDriver.supports_json_mode is False
        assert HuggingFaceDriver.supports_json_schema is False

    def test_local_http_driver_flags(self):
        from prompture.drivers.local_http_driver import LocalHTTPDriver

        assert LocalHTTPDriver.supports_json_mode is False
        assert LocalHTTPDriver.supports_json_schema is False

    def test_airllm_driver_flags(self):
        from prompture.drivers.airllm_driver import AirLLMDriver

        assert AirLLMDriver.supports_json_mode is False
        assert AirLLMDriver.supports_json_schema is False


class TestAsyncDriverCapabilityFlags:
    """Verify supports_json_mode and supports_json_schema are set correctly on each async driver class."""

    def test_async_openai_driver_flags(self):
        from prompture.drivers.async_openai_driver import AsyncOpenAIDriver

        assert AsyncOpenAIDriver.supports_json_mode is True
        assert AsyncOpenAIDriver.supports_json_schema is True

    def test_async_azure_driver_flags(self):
        from prompture.drivers.async_azure_driver import AsyncAzureDriver

        assert AsyncAzureDriver.supports_json_mode is True
        assert AsyncAzureDriver.supports_json_schema is True

    def test_async_claude_driver_flags(self):
        from prompture.drivers.async_claude_driver import AsyncClaudeDriver

        assert AsyncClaudeDriver.supports_json_mode is True
        assert AsyncClaudeDriver.supports_json_schema is True

    def test_async_google_driver_flags(self):
        from prompture.drivers.async_google_driver import AsyncGoogleDriver

        assert AsyncGoogleDriver.supports_json_mode is True
        assert AsyncGoogleDriver.supports_json_schema is True

    def test_async_groq_driver_flags(self):
        from prompture.drivers.async_groq_driver import AsyncGroqDriver

        assert AsyncGroqDriver.supports_json_mode is True
        assert AsyncGroqDriver.supports_json_schema is False

    def test_async_grok_driver_flags(self):
        from prompture.drivers.async_grok_driver import AsyncGrokDriver

        assert AsyncGrokDriver.supports_json_mode is True
        assert AsyncGrokDriver.supports_json_schema is False

    def test_async_openrouter_driver_flags(self):
        from prompture.drivers.async_openrouter_driver import AsyncOpenRouterDriver

        assert AsyncOpenRouterDriver.supports_json_mode is True
        assert AsyncOpenRouterDriver.supports_json_schema is True

    def test_async_ollama_driver_flags(self):
        from prompture.drivers.async_ollama_driver import AsyncOllamaDriver

        assert AsyncOllamaDriver.supports_json_mode is True
        assert AsyncOllamaDriver.supports_json_schema is True

    def test_async_lmstudio_driver_flags(self):
        from prompture.drivers.async_lmstudio_driver import AsyncLMStudioDriver

        assert AsyncLMStudioDriver.supports_json_mode is True
        assert AsyncLMStudioDriver.supports_json_schema is True


# ---------------------------------------------------------------------------
# 2. ask_for_json auto-detection
# ---------------------------------------------------------------------------


class TestAskForJsonAutoDetection:
    """Verify auto-detection passes json_mode/json_schema to driver when supported."""

    def test_auto_enables_json_mode_for_supported_driver(self):
        driver = MockDriver(json_mode=True)
        ask_for_json(driver, "Extract name", SAMPLE_SCHEMA, json_mode="auto")
        assert driver.last_options.get("json_mode") is True

    def test_auto_enables_json_schema_for_supported_driver(self):
        driver = MockDriver(json_mode=True, json_schema=True)
        ask_for_json(driver, "Extract name", SAMPLE_SCHEMA, json_mode="auto")
        assert driver.last_options.get("json_mode") is True
        assert driver.last_options.get("json_schema") == SAMPLE_SCHEMA

    def test_auto_does_not_enable_for_unsupported_driver(self):
        driver = MockDriver(json_mode=False)
        ask_for_json(driver, "Extract name", SAMPLE_SCHEMA, json_mode="auto")
        assert driver.last_options.get("json_mode") is None

    def test_auto_does_not_pass_schema_when_unsupported(self):
        driver = MockDriver(json_mode=True, json_schema=False)
        ask_for_json(driver, "Extract name", SAMPLE_SCHEMA, json_mode="auto")
        assert driver.last_options.get("json_mode") is True
        assert driver.last_options.get("json_schema") is None


# ---------------------------------------------------------------------------
# 3. ask_for_json explicit modes
# ---------------------------------------------------------------------------


class TestAskForJsonExplicitModes:
    """Verify json_mode='on' and 'off' override auto-detection."""

    def test_on_forces_json_mode_even_for_unsupported(self):
        driver = MockDriver(json_mode=False)
        ask_for_json(driver, "Extract name", SAMPLE_SCHEMA, json_mode="on")
        assert driver.last_options.get("json_mode") is True

    def test_off_disables_json_mode_even_for_supported(self):
        driver = MockDriver(json_mode=True, json_schema=True)
        ask_for_json(driver, "Extract name", SAMPLE_SCHEMA, json_mode="off")
        assert driver.last_options.get("json_mode") is None

    def test_on_does_not_pass_schema_for_unsupported(self):
        """When json_mode='on' but driver doesn't support schema, don't pass json_schema."""
        driver = MockDriver(json_mode=False, json_schema=False)
        ask_for_json(driver, "Extract name", SAMPLE_SCHEMA, json_mode="on")
        assert driver.last_options.get("json_mode") is True
        assert driver.last_options.get("json_schema") is None


# ---------------------------------------------------------------------------
# 4. Prompt simplification based on JSON mode
# ---------------------------------------------------------------------------


class TestPromptSimplification:
    """Verify the instruction prompt is adjusted based on JSON mode capabilities."""

    def test_schema_enforced_prompt_is_minimal(self):
        """When driver supports both json_mode and json_schema, prompt should be minimal."""
        driver = MockDriver(json_mode=True, json_schema=True)
        ask_for_json(driver, "Extract name", SAMPLE_SCHEMA, json_mode="auto")
        assert "Extract data matching the requested schema" in driver.last_prompt
        assert "Return only a single JSON object" not in driver.last_prompt

    def test_json_mode_only_prompt_includes_schema(self):
        """When driver supports json_mode but NOT json_schema, prompt should include schema."""
        driver = MockDriver(json_mode=True, json_schema=False)
        ask_for_json(driver, "Extract name", SAMPLE_SCHEMA, json_mode="auto")
        assert "Return a JSON object that validates against this schema" in driver.last_prompt
        # Should not include the strict formatting instructions
        assert "no markdown, no extra text" not in driver.last_prompt

    def test_no_json_mode_prompt_is_strict(self):
        """When driver doesn't support json_mode, prompt should be the strict original."""
        driver = MockDriver(json_mode=False)
        ask_for_json(driver, "Extract name", SAMPLE_SCHEMA, json_mode="auto")
        assert "Return only a single JSON object (no markdown, no extra text)" in driver.last_prompt

    def test_off_mode_always_uses_strict_prompt(self):
        """Even for a capable driver, json_mode='off' uses the strict prompt."""
        driver = MockDriver(json_mode=True, json_schema=True)
        ask_for_json(driver, "Extract name", SAMPLE_SCHEMA, json_mode="off")
        assert "Return only a single JSON object (no markdown, no extra text)" in driver.last_prompt


# ---------------------------------------------------------------------------
# 5. OpenAI driver response_format
# ---------------------------------------------------------------------------


class TestOpenAIJsonMode:
    """Verify OpenAI driver sets response_format correctly."""

    def test_json_mode_with_schema(self):
        """When json_mode + json_schema are passed, response_format should use json_schema type."""
        from prompture.drivers.openai_driver import OpenAIDriver

        driver = OpenAIDriver.__new__(OpenAIDriver)
        driver.api_key = "test"
        driver.model = "gpt-4o"

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"name": "Alice"}'
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_resp.usage.total_tokens = 15
        mock_resp.model_dump.return_value = {}

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        driver.client = mock_client

        driver.generate("test", {"json_mode": True, "json_schema": SAMPLE_SCHEMA})

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["name"] == "extraction"
        assert call_kwargs["response_format"]["json_schema"]["strict"] is True

    def test_json_mode_without_schema(self):
        """When json_mode without json_schema, response_format should use json_object type."""
        from prompture.drivers.openai_driver import OpenAIDriver

        driver = OpenAIDriver.__new__(OpenAIDriver)
        driver.api_key = "test"
        driver.model = "gpt-4o"

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"name": "Alice"}'
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_resp.usage.total_tokens = 15
        mock_resp.model_dump.return_value = {}

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        driver.client = mock_client

        driver.generate("test", {"json_mode": True})

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_no_json_mode(self):
        """Without json_mode, response_format should not be set."""
        from prompture.drivers.openai_driver import OpenAIDriver

        driver = OpenAIDriver.__new__(OpenAIDriver)
        driver.api_key = "test"
        driver.model = "gpt-4o"

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"name": "Alice"}'
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_resp.usage.total_tokens = 15
        mock_resp.model_dump.return_value = {}

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        driver.client = mock_client

        driver.generate("test", {})

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "response_format" not in call_kwargs


# ---------------------------------------------------------------------------
# 6. Claude driver tool-use
# ---------------------------------------------------------------------------


class TestClaudeJsonMode:
    """Verify Claude driver uses tool-use for structured output."""

    @patch("prompture.drivers.claude_driver.anthropic")
    def test_json_mode_with_schema_uses_tools(self, mock_anthropic):
        from prompture.drivers.claude_driver import ClaudeDriver

        mock_block = MagicMock()
        mock_block.type = "tool_use"
        mock_block.input = {"name": "Alice", "age": 30}

        mock_resp = MagicMock()
        mock_resp.content = [mock_block]
        mock_resp.usage.input_tokens = 10
        mock_resp.usage.output_tokens = 5

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        mock_anthropic.Anthropic.return_value = mock_client

        driver = ClaudeDriver.__new__(ClaudeDriver)
        driver.api_key = "test"
        driver.model = "claude-3-5-haiku-20241022"

        result = driver.generate("test", {"json_mode": True, "json_schema": SAMPLE_SCHEMA})

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["name"] == "extract_json"
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "extract_json"}

        # Response text should be the JSON-dumped tool input
        assert json.loads(result["text"]) == {"name": "Alice", "age": 30}

    @patch("prompture.drivers.claude_driver.anthropic")
    def test_no_json_mode_uses_regular_call(self, mock_anthropic):
        from prompture.drivers.claude_driver import ClaudeDriver

        mock_block = MagicMock()
        mock_block.text = '{"name": "Alice"}'

        mock_resp = MagicMock()
        mock_resp.content = [mock_block]
        mock_resp.usage.input_tokens = 10
        mock_resp.usage.output_tokens = 5

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        mock_anthropic.Anthropic.return_value = mock_client

        driver = ClaudeDriver.__new__(ClaudeDriver)
        driver.api_key = "test"
        driver.model = "claude-3-5-haiku-20241022"

        driver.generate("test", {})

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs


# ---------------------------------------------------------------------------
# 7. Google driver generation_config
# ---------------------------------------------------------------------------


class TestGoogleJsonMode:
    """Verify Google driver adds response_mime_type to config_dict."""

    def test_json_mode_adds_response_mime_type(self):
        from prompture.drivers.google_driver import GoogleDriver

        driver = GoogleDriver.__new__(GoogleDriver)
        driver.api_key = "test"
        driver.model = "gemini-1.5-pro"
        driver.options = {}

        messages = [{"role": "user", "content": "test"}]
        _gen_input, config_dict = driver._build_generation_args(messages, {"json_mode": True})
        assert config_dict["response_mime_type"] == "application/json"

    def test_json_mode_with_schema(self):
        from prompture.drivers.google_driver import GoogleDriver

        driver = GoogleDriver.__new__(GoogleDriver)
        driver.api_key = "test"
        driver.model = "gemini-1.5-pro"
        driver.options = {}

        messages = [{"role": "user", "content": "test"}]
        _gen_input, config_dict = driver._build_generation_args(
            messages, {"json_mode": True, "json_schema": SAMPLE_SCHEMA}
        )
        assert config_dict["response_mime_type"] == "application/json"
        assert config_dict["response_schema"] == SAMPLE_SCHEMA


# ---------------------------------------------------------------------------
# 8. Groq driver response_format
# ---------------------------------------------------------------------------


class TestGroqJsonMode:
    """Verify Groq driver sets response_format for JSON mode."""

    def test_json_mode_sets_response_format(self):
        from prompture.drivers.groq_driver import GroqDriver

        driver = GroqDriver.__new__(GroqDriver)
        driver.api_key = "test"
        driver.model = "llama2-70b-4096"

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"name": "Alice"}'
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_resp.usage.total_tokens = 15
        mock_resp.model_dump.return_value = {}

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        driver.client = mock_client

        driver.generate("test", {"json_mode": True})

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# 9. Fallback for unsupported drivers
# ---------------------------------------------------------------------------


class TestUnsupportedDriverFallback:
    """Verify prompt-based enforcement when driver doesn't support JSON mode."""

    def test_unsupported_driver_uses_prompt_enforcement(self):
        driver = MockDriver(json_mode=False)
        ask_for_json(driver, "Extract name from: Alice is 30", SAMPLE_SCHEMA, json_mode="auto")

        # Should NOT have json_mode in options
        assert "json_mode" not in driver.last_options
        assert "json_schema" not in driver.last_options

        # Should have the full schema enforcement prompt
        assert "Return only a single JSON object" in driver.last_prompt
        assert "no markdown, no extra text" in driver.last_prompt

    def test_result_structure_unchanged(self):
        """JSON mode should not change the result structure."""
        driver = MockDriver(json_mode=True, json_schema=True)
        result = ask_for_json(driver, "Extract name", SAMPLE_SCHEMA, json_mode="auto")

        assert "json_string" in result
        assert "json_object" in result
        assert "usage" in result
        assert result["json_object"]["name"] == "Alice"


# ---------------------------------------------------------------------------
# 10. Cache key includes json_mode
# ---------------------------------------------------------------------------


class TestCacheKeyJsonMode:
    """Verify json_mode is included in cache key generation."""

    def test_json_mode_in_cache_relevant_options(self):
        from prompture.infra.cache import _CACHE_RELEVANT_OPTIONS

        assert "json_mode" in _CACHE_RELEVANT_OPTIONS

    def test_different_json_mode_produces_different_key(self):
        from prompture.infra.cache import make_cache_key

        key_on = make_cache_key(
            prompt="test",
            model_name="openai/gpt-4",
            schema=SAMPLE_SCHEMA,
            options={"json_mode": True},
        )
        key_off = make_cache_key(
            prompt="test",
            model_name="openai/gpt-4",
            schema=SAMPLE_SCHEMA,
            options={},
        )
        assert key_on != key_off
