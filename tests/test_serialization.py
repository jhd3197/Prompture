"""Tests for the serialization module."""

from __future__ import annotations

from typing import Any

import pytest

from prompture.image import ImageContent
from prompture.serialization import (
    EXPORT_VERSION,
    _deserialize_message_content,
    _serialize_message_content,
    export_conversation,
    export_usage_session,
    import_conversation,
    import_usage_session,
)
from prompture.session import UsageSession

# ------------------------------------------------------------------
# Message content round-trips
# ------------------------------------------------------------------


class TestSerializeMessageContent:
    """Test _serialize_message_content / _deserialize_message_content."""

    def test_string_passthrough(self):
        assert _serialize_message_content("hello") == "hello"
        assert _deserialize_message_content("hello") == "hello"

    def test_text_only_list(self):
        blocks = [{"type": "text", "text": "hello"}]
        serialized = _serialize_message_content(blocks)
        assert serialized == blocks
        deserialized = _deserialize_message_content(serialized)
        assert deserialized == blocks

    def test_image_content_round_trip(self):
        ic = ImageContent(data="abc123", media_type="image/png", source_type="base64", url=None)
        blocks = [
            {"type": "text", "text": "Describe this"},
            {"type": "image", "source": ic},
        ]
        serialized = _serialize_message_content(blocks)

        # ImageContent should be converted to a plain dict
        assert isinstance(serialized[1]["source"], dict)
        assert serialized[1]["source"]["data"] == "abc123"
        assert serialized[1]["source"]["media_type"] == "image/png"

        # Round-trip back
        deserialized = _deserialize_message_content(serialized)
        assert isinstance(deserialized[1]["source"], ImageContent)
        assert deserialized[1]["source"].data == "abc123"
        assert deserialized[1]["source"].media_type == "image/png"

    def test_image_dict_source_passthrough(self):
        """When source is already a dict (e.g. previously serialized), it passes through."""
        blocks = [
            {
                "type": "image",
                "source": {"data": "x", "media_type": "image/jpeg", "source_type": "base64", "url": None},
            },
        ]
        serialized = _serialize_message_content(blocks)
        assert serialized == blocks

    def test_non_content_passthrough(self):
        assert _serialize_message_content(42) == 42
        assert _deserialize_message_content(None) is None


# ------------------------------------------------------------------
# UsageSession round-trips
# ------------------------------------------------------------------


class TestUsageSessionSerialization:
    """Test export_usage_session / import_usage_session."""

    def test_round_trip(self):
        session = UsageSession(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost=0.005,
            call_count=3,
            errors=1,
        )
        session._per_model["openai/gpt-4"] = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost": 0.005,
            "calls": 3,
        }

        exported = export_usage_session(session)
        assert exported["prompt_tokens"] == 100
        assert exported["cost"] == 0.005
        assert exported["total_cost"] == 0.005  # Deprecated alias still present
        assert "openai/gpt-4" in exported["per_model"]

        restored = import_usage_session(exported)
        assert restored.prompt_tokens == 100
        assert restored.completion_tokens == 50
        assert restored.cost == 0.005
        assert restored.call_count == 3
        assert restored.errors == 1
        assert "openai/gpt-4" in restored._per_model

    def test_import_from_old_format(self):
        """Import should work with old format using only total_cost (no cost key)."""
        old_format = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "total_cost": 0.005,  # Old format only has total_cost
            "call_count": 1,
            "errors": 0,
            "per_model": {},
        }
        restored = import_usage_session(old_format)
        assert restored.cost == 0.005

    def test_empty_session(self):
        session = UsageSession()
        exported = export_usage_session(session)
        restored = import_usage_session(exported)
        assert restored.total_tokens == 0
        assert restored.call_count == 0


# ------------------------------------------------------------------
# Conversation export / import
# ------------------------------------------------------------------


class TestConversationExportImport:
    """Test export_conversation / import_conversation."""

    def _make_export(self, **overrides: Any) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "model_name": "openai/gpt-4",
            "system_prompt": "You are helpful",
            "options": {"temperature": 0.7},
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001, "turns": 1},
            "max_tool_rounds": 10,
            "conversation_id": "test-id-123",
        }
        defaults.update(overrides)
        return export_conversation(**defaults)

    def test_basic_round_trip(self):
        exported = self._make_export()
        assert exported["version"] == EXPORT_VERSION
        assert exported["conversation_id"] == "test-id-123"
        assert exported["model_name"] == "openai/gpt-4"
        assert len(exported["messages"]) == 2

        imported = import_conversation(exported)
        assert imported["model_name"] == "openai/gpt-4"
        assert imported["system_prompt"] == "You are helpful"
        assert len(imported["messages"]) == 2

    def test_version_validation(self):
        exported = self._make_export()
        exported["version"] = 99
        with pytest.raises(ValueError, match="Unsupported export version"):
            import_conversation(exported)

    def test_version_1_accepted(self):
        exported = self._make_export()
        assert exported["version"] == 1
        imported = import_conversation(exported)
        assert imported["version"] == 1

    def test_image_content_preserved(self):
        ic = ImageContent(data="imgdata", media_type="image/png")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image", "source": ic},
                ],
            },
        ]
        exported = self._make_export(messages=messages)
        # Serialized: source should be dict
        assert isinstance(exported["messages"][0]["content"][1]["source"], dict)

        imported = import_conversation(exported)
        # Deserialized: source should be ImageContent
        assert isinstance(imported["messages"][0]["content"][1]["source"], ImageContent)
        assert imported["messages"][0]["content"][1]["source"].data == "imgdata"

    def test_strip_images(self):
        ic = ImageContent(data="bigbase64data", media_type="image/png")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image", "source": ic},
                ],
            },
        ]
        exported = self._make_export(messages=messages, strip_images=True)
        # With strip_images, only text remains and should be collapsed to string
        assert exported["messages"][0]["content"] == "Describe this"

    def test_tool_metadata_serialized(self):
        tools_meta = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        ]
        exported = self._make_export(tools_metadata=tools_meta)
        assert exported["tools"] == tools_meta
        assert "function" not in str(exported["tools"])

    def test_tool_calls_in_messages(self):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": '{"temp": 72}'},
            {"role": "assistant", "content": "It's 72°F in NYC."},
        ]
        exported = self._make_export(messages=messages)
        imported = import_conversation(exported)
        assert len(imported["messages"]) == 4
        assert imported["messages"][1]["tool_calls"][0]["id"] == "tc1"
        assert imported["messages"][2]["role"] == "tool"

    def test_usage_session_included(self):
        session = UsageSession(prompt_tokens=500, total_tokens=700, cost=0.015, call_count=5)
        exported = self._make_export(usage_session=session)
        assert "usage_session" in exported
        assert exported["usage_session"]["prompt_tokens"] == 500

        imported = import_conversation(exported)
        assert isinstance(imported["usage_session"], UsageSession)
        assert imported["usage_session"].prompt_tokens == 500

    def test_metadata_includes_timestamps(self):
        exported = self._make_export()
        assert "metadata" in exported
        assert "created_at" in exported["metadata"]
        assert "last_active" in exported["metadata"]
        assert exported["metadata"]["turn_count"] == 1

    def test_no_tools_key_when_none(self):
        exported = self._make_export()
        assert "tools" not in exported

    def test_no_usage_session_key_when_none(self):
        exported = self._make_export()
        assert "usage_session" not in exported
