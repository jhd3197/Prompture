"""Tests for driver generate_messages support."""

from __future__ import annotations

from typing import Any

import pytest

from prompture.async_driver import AsyncDriver
from prompture.driver import Driver


class TestFlattenMessages:
    """Test the _flatten_messages static helper."""

    def test_single_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = Driver._flatten_messages(messages)
        assert result == "[User]: Hello"

    def test_system_and_user(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = Driver._flatten_messages(messages)
        assert "[System]: You are helpful" in result
        assert "[User]: Hi" in result

    def test_full_conversation(self):
        messages = [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Bye"},
        ]
        result = Driver._flatten_messages(messages)
        assert "[System]: Be concise" in result
        assert "[User]: Hello" in result
        assert "[Assistant]: Hi there" in result
        assert "[User]: Bye" in result

    def test_empty_messages(self):
        result = Driver._flatten_messages([])
        assert result == ""

    def test_missing_role_defaults_to_user(self):
        messages = [{"content": "No role"}]
        result = Driver._flatten_messages(messages)
        assert "[User]: No role" in result

    def test_async_driver_flatten_is_same(self):
        messages = [{"role": "user", "content": "test"}]
        sync_result = Driver._flatten_messages(messages)
        async_result = AsyncDriver._flatten_messages(messages)
        assert sync_result == async_result


class TestBaseDriverGenerateMessages:
    """Test the default generate_messages fallback on the base class."""

    def test_default_flattens_and_delegates(self):
        class StubDriver(Driver):
            def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
                return {"text": f"echoed: {prompt}", "meta": {}}

        driver = StubDriver()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        result = driver.generate_messages(messages, {})
        assert "echoed:" in result["text"]
        assert "[System]: sys" in result["text"]
        assert "[User]: hello" in result["text"]

    def test_supports_messages_false_by_default(self):
        assert Driver.supports_messages is False
        assert AsyncDriver.supports_messages is False


class TestBaseAsyncDriverGenerateMessages:
    """Test the default async generate_messages fallback."""

    @pytest.mark.asyncio
    async def test_default_flattens_and_delegates(self):
        class StubAsyncDriver(AsyncDriver):
            async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
                return {"text": f"async-echoed: {prompt}", "meta": {}}

        driver = StubAsyncDriver()
        messages = [
            {"role": "user", "content": "async test"},
        ]
        result = await driver.generate_messages(messages, {})
        assert "async-echoed:" in result["text"]
        assert "[User]: async test" in result["text"]


class TestDriverSupportsMessages:
    """Verify that refactored drivers declare supports_messages = True."""

    def test_openai_driver_supports_messages(self):
        from prompture.drivers.openai_driver import OpenAIDriver

        assert OpenAIDriver.supports_messages is True

    def test_claude_driver_supports_messages(self):
        from prompture.drivers.claude_driver import ClaudeDriver

        assert ClaudeDriver.supports_messages is True

    def test_groq_driver_supports_messages(self):
        from prompture.drivers.groq_driver import GroqDriver

        assert GroqDriver.supports_messages is True

    def test_grok_driver_supports_messages(self):
        from prompture.drivers.grok_driver import GrokDriver

        assert GrokDriver.supports_messages is True

    def test_ollama_driver_supports_messages(self):
        from prompture.drivers.ollama_driver import OllamaDriver

        assert OllamaDriver.supports_messages is True

    def test_lmstudio_driver_supports_messages(self):
        from prompture.drivers.lmstudio_driver import LMStudioDriver

        assert LMStudioDriver.supports_messages is True

    def test_openrouter_driver_supports_messages(self):
        from prompture.drivers.openrouter_driver import OpenRouterDriver

        assert OpenRouterDriver.supports_messages is True

    def test_azure_driver_supports_messages(self):
        from prompture.drivers.azure_driver import AzureDriver

        assert AzureDriver.supports_messages is True


class TestAsyncDriverSupportsMessages:
    """Verify async drivers declare supports_messages = True."""

    def test_async_openai_driver(self):
        from prompture.drivers.async_openai_driver import AsyncOpenAIDriver

        assert AsyncOpenAIDriver.supports_messages is True

    def test_async_claude_driver(self):
        from prompture.drivers.async_claude_driver import AsyncClaudeDriver

        assert AsyncClaudeDriver.supports_messages is True

    def test_async_groq_driver(self):
        from prompture.drivers.async_groq_driver import AsyncGroqDriver

        assert AsyncGroqDriver.supports_messages is True

    def test_async_grok_driver(self):
        from prompture.drivers.async_grok_driver import AsyncGrokDriver

        assert AsyncGrokDriver.supports_messages is True

    def test_async_ollama_driver(self):
        from prompture.drivers.async_ollama_driver import AsyncOllamaDriver

        assert AsyncOllamaDriver.supports_messages is True

    def test_async_lmstudio_driver(self):
        from prompture.drivers.async_lmstudio_driver import AsyncLMStudioDriver

        assert AsyncLMStudioDriver.supports_messages is True

    def test_async_openrouter_driver(self):
        from prompture.drivers.async_openrouter_driver import AsyncOpenRouterDriver

        assert AsyncOpenRouterDriver.supports_messages is True

    def test_async_azure_driver(self):
        from prompture.drivers.async_azure_driver import AsyncAzureDriver

        assert AsyncAzureDriver.supports_messages is True
