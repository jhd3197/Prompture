"""Tests for the AsyncConversation class."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from prompture.async_conversation import AsyncConversation
from prompture.async_driver import AsyncDriver


class MockAsyncDriver(AsyncDriver):
    """A mock async driver for testing conversation functionality."""

    supports_json_mode = True
    supports_json_schema = False
    supports_messages = True

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or ['{"value": "test"}'])
        self._call_count = 0
        self._last_messages: list[dict[str, str]] = []
        self.model = "mock-async-model"

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self._make_response()

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        self._last_messages = messages
        return self._make_response()

    def _make_response(self) -> dict[str, Any]:
        idx = min(self._call_count, len(self.responses) - 1)
        text = self.responses[idx]
        self._call_count += 1
        return {
            "text": text,
            "meta": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.001,
                "raw_response": {},
                "model_name": "mock-async-model",
            },
        }


class TestAsyncConversationInit:
    """Test AsyncConversation initialization."""

    def test_init_with_driver(self):
        driver = MockAsyncDriver()
        conv = AsyncConversation(driver=driver, system_prompt="Be helpful")
        assert conv._driver is driver
        assert conv._system_prompt == "Be helpful"
        assert conv.messages == []

    def test_init_requires_model_or_driver(self):
        with pytest.raises(ValueError, match="Either model_name or driver"):
            AsyncConversation()

    def test_init_with_options(self):
        driver = MockAsyncDriver()
        conv = AsyncConversation(driver=driver, options={"temperature": 0.5})
        assert conv._options == {"temperature": 0.5}


class TestAsyncConversationMessageManagement:
    """Test message history management."""

    def test_messages_property_returns_copy(self):
        driver = MockAsyncDriver()
        conv = AsyncConversation(driver=driver)
        conv.add_context("user", "Hello")
        msgs = conv.messages
        msgs.clear()
        assert len(conv.messages) == 1

    def test_clear_resets_history(self):
        driver = MockAsyncDriver()
        conv = AsyncConversation(driver=driver, system_prompt="sys")
        conv.add_context("user", "msg1")
        conv.clear()
        assert len(conv.messages) == 0
        assert conv._system_prompt == "sys"

    def test_add_context_invalid_role(self):
        driver = MockAsyncDriver()
        conv = AsyncConversation(driver=driver)
        with pytest.raises(ValueError, match="role must be"):
            conv.add_context("system", "should fail")


class TestAsyncConversationAsk:
    """Test the async ask() method."""

    @pytest.mark.asyncio
    async def test_ask_returns_text(self):
        driver = MockAsyncDriver(responses=["Hello back!"])
        conv = AsyncConversation(driver=driver)
        result = await conv.ask("Hello")
        assert result == "Hello back!"

    @pytest.mark.asyncio
    async def test_ask_appends_to_history(self):
        driver = MockAsyncDriver(responses=["response"])
        conv = AsyncConversation(driver=driver)
        await conv.ask("question")
        assert len(conv.messages) == 2
        assert conv.messages[0] == {"role": "user", "content": "question"}
        assert conv.messages[1] == {"role": "assistant", "content": "response"}

    @pytest.mark.asyncio
    async def test_ask_accumulates_usage(self):
        driver = MockAsyncDriver(responses=["r1", "r2"])
        conv = AsyncConversation(driver=driver)
        await conv.ask("q1")
        await conv.ask("q2")
        usage = conv.usage
        assert usage["turns"] == 2
        assert usage["prompt_tokens"] == 20


class TestAsyncConversationAskForJson:
    """Test the async ask_for_json() method."""

    @pytest.mark.asyncio
    async def test_ask_for_json_returns_parsed(self):
        driver = MockAsyncDriver(responses=['{"name": "John", "age": 30}'])
        conv = AsyncConversation(driver=driver)
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        result = await conv.ask_for_json("Extract info", schema)
        assert result["json_object"] == {"name": "John", "age": 30}

    @pytest.mark.asyncio
    async def test_ask_for_json_stores_clean_context(self):
        driver = MockAsyncDriver(responses=['{"name": "John"}'])
        conv = AsyncConversation(driver=driver)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        await conv.ask_for_json("Extract name", schema)
        assert conv.messages[0] == {"role": "user", "content": "Extract name"}

    @pytest.mark.asyncio
    async def test_multi_turn_context(self):
        driver = MockAsyncDriver(responses=['{"name": "John"}', '{"age": 30}'])
        conv = AsyncConversation(driver=driver, system_prompt="Extract data")

        name_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        age_schema = {"type": "object", "properties": {"age": {"type": "integer"}}}

        await conv.ask_for_json("Extract name", name_schema)
        await conv.ask_for_json("Now extract age", age_schema)

        last_msgs = driver._last_messages
        assert len(last_msgs) >= 3


class TestAsyncConversationExtractWithModel:
    """Test extract_with_model on AsyncConversation."""

    @pytest.mark.asyncio
    async def test_extract_with_model_returns_pydantic(self):
        class Person(BaseModel):
            name: str
            age: int = 0

        driver = MockAsyncDriver(responses=['{"name": "Alice", "age": 25}'])
        conv = AsyncConversation(driver=driver)
        result = await conv.extract_with_model(Person, "Alice is 25")
        assert result["model"].name == "Alice"
        assert result["model"].age == 25


class TestAsyncConversationUsage:
    """Test usage tracking."""

    def test_initial_usage_is_zero(self):
        driver = MockAsyncDriver()
        conv = AsyncConversation(driver=driver)
        usage = conv.usage
        assert usage["prompt_tokens"] == 0
        assert usage["turns"] == 0

    def test_usage_returns_copy(self):
        driver = MockAsyncDriver()
        conv = AsyncConversation(driver=driver)
        u1 = conv.usage
        u1["turns"] = 999
        assert conv.usage["turns"] == 0
