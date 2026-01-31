"""Tests for the Conversation class."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from prompture.conversation import Conversation
from prompture.driver import Driver
from prompture.serialization import EXPORT_VERSION


class MockDriver(Driver):
    """A mock driver for testing conversation functionality."""

    supports_json_mode = True
    supports_json_schema = False
    supports_messages = True

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or ['{"value": "test"}'])
        self._call_count = 0
        self._last_messages: list[dict[str, str]] = []
        self.model = "mock-model"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self._make_response()

    def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
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
                "model_name": "mock-model",
            },
        }


class TestConversationInit:
    """Test Conversation initialization."""

    def test_init_with_driver(self):
        driver = MockDriver()
        conv = Conversation(driver=driver, system_prompt="Be helpful")
        assert conv._driver is driver
        assert conv._system_prompt == "Be helpful"
        assert conv.messages == []

    def test_init_with_model_name(self):
        with patch("prompture.conversation.get_driver_for_model") as mock_get:
            mock_get.return_value = MockDriver()
            conv = Conversation(model_name="openai/gpt-4")
            assert conv._model_name == "openai/gpt-4"

    def test_init_requires_model_or_driver(self):
        with pytest.raises(ValueError, match="Either model_name or driver"):
            Conversation()

    def test_init_with_options(self):
        driver = MockDriver()
        conv = Conversation(driver=driver, options={"temperature": 0.5})
        assert conv._options == {"temperature": 0.5}


class TestConversationMessageManagement:
    """Test message history management."""

    def test_messages_property_returns_copy(self):
        driver = MockDriver()
        conv = Conversation(driver=driver)
        conv.add_context("user", "Hello")
        msgs = conv.messages
        msgs.clear()  # Should not affect internal state
        assert len(conv.messages) == 1

    def test_clear_resets_history(self):
        driver = MockDriver()
        conv = Conversation(driver=driver, system_prompt="sys")
        conv.add_context("user", "msg1")
        conv.add_context("assistant", "reply1")
        assert len(conv.messages) == 2

        conv.clear()
        assert len(conv.messages) == 0
        # system_prompt should be preserved
        assert conv._system_prompt == "sys"

    def test_add_context_user(self):
        driver = MockDriver()
        conv = Conversation(driver=driver)
        conv.add_context("user", "seed message")
        assert conv.messages == [{"role": "user", "content": "seed message"}]

    def test_add_context_assistant(self):
        driver = MockDriver()
        conv = Conversation(driver=driver)
        conv.add_context("assistant", "seeded reply")
        assert conv.messages == [{"role": "assistant", "content": "seeded reply"}]

    def test_add_context_invalid_role(self):
        driver = MockDriver()
        conv = Conversation(driver=driver)
        with pytest.raises(ValueError, match="role must be"):
            conv.add_context("system", "should fail")

    def test_build_messages_with_system_prompt(self):
        driver = MockDriver()
        conv = Conversation(driver=driver, system_prompt="You are helpful")
        conv.add_context("user", "prev")
        conv.add_context("assistant", "prev reply")

        msgs = conv._build_messages("new question")
        assert msgs[0] == {"role": "system", "content": "You are helpful"}
        assert msgs[1] == {"role": "user", "content": "prev"}
        assert msgs[2] == {"role": "assistant", "content": "prev reply"}
        assert msgs[3] == {"role": "user", "content": "new question"}

    def test_build_messages_without_system_prompt(self):
        driver = MockDriver()
        conv = Conversation(driver=driver)

        msgs = conv._build_messages("hello")
        assert msgs == [{"role": "user", "content": "hello"}]


class TestConversationAsk:
    """Test the ask() method."""

    def test_ask_returns_text(self):
        driver = MockDriver(responses=["Hello back!"])
        conv = Conversation(driver=driver)
        result = conv.ask("Hello")
        assert result == "Hello back!"

    def test_ask_appends_to_history(self):
        driver = MockDriver(responses=["response"])
        conv = Conversation(driver=driver)
        conv.ask("question")
        assert len(conv.messages) == 2
        assert conv.messages[0] == {"role": "user", "content": "question"}
        assert conv.messages[1] == {"role": "assistant", "content": "response"}

    def test_ask_accumulates_usage(self):
        driver = MockDriver(responses=["r1", "r2"])
        conv = Conversation(driver=driver)
        conv.ask("q1")
        conv.ask("q2")
        usage = conv.usage
        assert usage["turns"] == 2
        assert usage["prompt_tokens"] == 20
        assert usage["completion_tokens"] == 10
        assert usage["total_tokens"] == 30

    def test_ask_passes_messages_to_driver(self):
        driver = MockDriver(responses=["reply"])
        conv = Conversation(driver=driver, system_prompt="sys")
        conv.ask("hello")

        last_msgs = driver._last_messages
        assert last_msgs[0] == {"role": "system", "content": "sys"}
        assert last_msgs[1]["role"] == "user"
        assert "hello" in last_msgs[1]["content"]


class TestConversationAskForJson:
    """Test the ask_for_json() method."""

    def test_ask_for_json_returns_parsed(self):
        driver = MockDriver(responses=['{"name": "John", "age": 30}'])
        conv = Conversation(driver=driver)
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        result = conv.ask_for_json("Extract info", schema)
        assert result["json_object"] == {"name": "John", "age": 30}
        assert isinstance(result["json_string"], str)

    def test_ask_for_json_stores_clean_context(self):
        driver = MockDriver(responses=['{"name": "John"}'])
        conv = Conversation(driver=driver)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        conv.ask_for_json("Extract name from: John Smith", schema)

        # History should store original content (not schema-augmented version)
        assert conv.messages[0] == {"role": "user", "content": "Extract name from: John Smith"}

    def test_ask_for_json_accumulates_usage(self):
        driver = MockDriver(responses=['{"v": 1}', '{"v": 2}'])
        conv = Conversation(driver=driver)
        schema = {"type": "object", "properties": {"v": {"type": "integer"}}}
        conv.ask_for_json("q1", schema)
        conv.ask_for_json("q2", schema)
        assert conv.usage["turns"] == 2

    def test_ask_for_json_includes_usage_metadata(self):
        driver = MockDriver(responses=['{"x": 1}'])
        conv = Conversation(driver=driver)
        result = conv.ask_for_json("q", {"type": "object", "properties": {"x": {"type": "integer"}}})
        assert "usage" in result
        assert "prompt_tokens" in result["usage"]

    def test_multi_turn_context(self):
        driver = MockDriver(responses=['{"name": "John"}', '{"age": 30}'])
        conv = Conversation(driver=driver, system_prompt="Extract data")

        name_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        age_schema = {"type": "object", "properties": {"age": {"type": "integer"}}}

        conv.ask_for_json("Extract name from: John, age 30", name_schema)
        conv.ask_for_json("Now extract age", age_schema)

        # Second call should include first turn in messages
        last_msgs = driver._last_messages
        assert len(last_msgs) >= 3  # system + first user+assistant + second user


class TestConversationExtractWithModel:
    """Test extract_with_model on Conversation."""

    def test_extract_with_model_returns_pydantic(self):
        class Person(BaseModel):
            name: str
            age: int = 0

        driver = MockDriver(responses=['{"name": "Alice", "age": 25}'])
        conv = Conversation(driver=driver)
        result = conv.extract_with_model(Person, "Alice is 25")
        assert result["model"].name == "Alice"
        assert result["model"].age == 25


class TestConversationUsageProperty:
    """Test the usage accumulation property."""

    def test_usage_returns_copy(self):
        driver = MockDriver()
        conv = Conversation(driver=driver)
        u1 = conv.usage
        u1["turns"] = 999
        assert conv.usage["turns"] == 0  # Original unaffected

    def test_initial_usage_is_zero(self):
        driver = MockDriver()
        conv = Conversation(driver=driver)
        usage = conv.usage
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0
        assert usage["cost"] == 0.0
        assert usage["turns"] == 0


class TestConversationPersistence:
    """Test export/import, save/load, auto-save, id, and tags."""

    def test_export_returns_correct_structure(self):
        driver = MockDriver(responses=["response"])
        conv = Conversation(driver=driver, system_prompt="sys", options={"temperature": 0.5})
        conv.ask("hello")

        data = conv.export()
        assert data["version"] == EXPORT_VERSION
        assert data["conversation_id"] == conv.conversation_id
        assert data["system_prompt"] == "sys"
        assert data["options"] == {"temperature": 0.5}
        assert len(data["messages"]) == 2

    def test_conversation_id_property(self):
        driver = MockDriver()
        conv = Conversation(driver=driver, conversation_id="my-id")
        assert conv.conversation_id == "my-id"

    def test_conversation_id_auto_generated(self):
        driver = MockDriver()
        conv = Conversation(driver=driver)
        assert conv.conversation_id  # not empty
        assert len(conv.conversation_id) > 10  # UUID-like

    def test_tags_property(self):
        driver = MockDriver()
        conv = Conversation(driver=driver, tags=["demo", "test"])
        assert conv.tags == ["demo", "test"]
        conv.tags = ["new-tag"]
        assert conv.tags == ["new-tag"]

    def test_from_export_round_trip(self):
        driver = MockDriver(responses=["response"])
        conv = Conversation(
            model_name="mock/model", driver=driver, system_prompt="sys",
            options={"temperature": 0.5}, tags=["demo"],
        )
        conv.ask("hello")
        data = conv.export()

        with patch("prompture.conversation.get_driver_for_model") as mock_get:
            mock_get.return_value = MockDriver()
            restored = Conversation.from_export(data)
            assert restored.conversation_id == conv.conversation_id
            assert restored._system_prompt == "sys"
            assert restored._options == {"temperature": 0.5}
            assert len(restored.messages) == 2
            assert restored.usage["turns"] == 1
            assert restored.tags == ["demo"]

    def test_save_and_load(self, tmp_path: Path):
        driver = MockDriver(responses=["response"])
        conv = Conversation(model_name="mock/model", driver=driver, system_prompt="sys")
        conv.ask("hello")
        path = tmp_path / "conv.json"
        conv.save(path)

        assert path.exists()

        with patch("prompture.conversation.get_driver_for_model") as mock_get:
            mock_get.return_value = MockDriver()
            loaded = Conversation.load(path)
            assert loaded.conversation_id == conv.conversation_id
            assert len(loaded.messages) == 2

    def test_auto_save_triggers(self, tmp_path: Path):
        auto_path = tmp_path / "auto.json"
        driver = MockDriver(responses=["response"])
        conv = Conversation(model_name="mock/model", driver=driver, auto_save=auto_path)
        conv.ask("hello")
        assert auto_path.exists()

    def test_callbacks_not_in_export(self):
        driver = MockDriver(responses=["response"])
        conv = Conversation(driver=driver)
        conv.ask("hello")
        data = conv.export()
        assert "callbacks" not in data

    def test_tool_functions_not_in_export(self):
        driver = MockDriver(responses=["response"])
        conv = Conversation(driver=driver)
        conv.ask("hello")
        data = conv.export()
        # tools key should not be present if no tools registered
        assert "tools" not in data or all("function" not in t for t in data.get("tools", []))
