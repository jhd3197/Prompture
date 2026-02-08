"""Tests for vision/image support across the Prompture stack."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from prompture.media.image import (
    ImageContent,
    image_from_base64,
    image_from_bytes,
    image_from_file,
    image_from_url,
    make_image,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal 1x1 red PNG (67 bytes)
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8BQDwAEgAF/pooBPQAAAABJRU5ErkJggg=="
)

_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")


# ---------------------------------------------------------------------------
# TestImageContent — constructors and auto-detection
# ---------------------------------------------------------------------------


class TestImageContent:
    def test_from_bytes(self):
        ic = image_from_bytes(_PNG_BYTES)
        assert ic.data == _PNG_B64
        assert ic.media_type == "image/png"
        assert ic.source_type == "base64"
        assert ic.url is None

    def test_from_bytes_with_media_type(self):
        ic = image_from_bytes(_PNG_BYTES, media_type="image/webp")
        assert ic.media_type == "image/webp"

    def test_from_bytes_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            image_from_bytes(b"")

    def test_from_base64_raw(self):
        ic = image_from_base64(_PNG_B64)
        assert ic.data == _PNG_B64
        assert ic.media_type == "image/png"  # default

    def test_from_base64_data_uri(self):
        uri = f"data:image/jpeg;base64,{_PNG_B64}"
        ic = image_from_base64(uri)
        assert ic.data == _PNG_B64
        assert ic.media_type == "image/jpeg"

    def test_from_file(self, tmp_path: Path):
        p = tmp_path / "test.png"
        p.write_bytes(_PNG_BYTES)
        ic = image_from_file(p)
        assert ic.data == _PNG_B64
        assert ic.media_type == "image/png"

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            image_from_file("/nonexistent/file.png")

    def test_from_url(self):
        ic = image_from_url("https://example.com/photo.jpg")
        assert ic.source_type == "url"
        assert ic.url == "https://example.com/photo.jpg"
        assert ic.media_type == "image/jpeg"

    def test_from_url_custom_media_type(self):
        ic = image_from_url("https://example.com/img", media_type="image/webp")
        assert ic.media_type == "image/webp"

    def test_frozen(self):
        ic = image_from_bytes(_PNG_BYTES)
        with pytest.raises(AttributeError):
            ic.data = "changed"  # type: ignore[misc]


class TestMakeImage:
    def test_passthrough_image_content(self):
        ic = ImageContent(data="abc", media_type="image/png")
        assert make_image(ic) is ic

    def test_bytes(self):
        ic = make_image(_PNG_BYTES)
        assert ic.data == _PNG_B64

    def test_path(self, tmp_path: Path):
        p = tmp_path / "pic.png"
        p.write_bytes(_PNG_BYTES)
        ic = make_image(p)
        assert ic.data == _PNG_B64

    def test_string_data_uri(self):
        uri = f"data:image/gif;base64,{_PNG_B64}"
        ic = make_image(uri)
        assert ic.media_type == "image/gif"

    def test_string_url(self):
        ic = make_image("https://example.com/img.png")
        assert ic.source_type == "url"

    def test_string_file_path(self, tmp_path: Path):
        p = tmp_path / "x.jpg"
        p.write_bytes(_PNG_BYTES)
        ic = make_image(str(p))
        assert ic.media_type == "image/jpeg"

    def test_string_raw_base64(self):
        ic = make_image(_PNG_B64)
        assert ic.data == _PNG_B64

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            make_image(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestBuildContentWithImages
# ---------------------------------------------------------------------------


class TestBuildContentWithImages:
    def test_no_images_returns_string(self):
        from prompture.agents.conversation import Conversation

        result = Conversation._build_content_with_images("hello")
        assert result == "hello"

    def test_no_images_none_returns_string(self):
        from prompture.agents.conversation import Conversation

        result = Conversation._build_content_with_images("hello", None)
        assert result == "hello"

    def test_empty_list_returns_string(self):
        from prompture.agents.conversation import Conversation

        result = Conversation._build_content_with_images("hello", [])
        assert result == "hello"

    def test_with_images_returns_blocks(self):
        from prompture.agents.conversation import Conversation

        result = Conversation._build_content_with_images("hello", [_PNG_BYTES])
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "hello"}
        assert result[1]["type"] == "image"
        assert isinstance(result[1]["source"], ImageContent)


# ---------------------------------------------------------------------------
# TestConversationAskWithImages — mock driver captures correct messages
# ---------------------------------------------------------------------------


class TestConversationAskWithImages:
    def _make_conv(self) -> tuple:
        """Create a Conversation with a mock driver."""
        from prompture.agents.conversation import Conversation

        mock_driver = MagicMock()
        mock_driver.supports_tool_use = False
        mock_driver.supports_streaming = False
        mock_driver.supports_vision = True
        mock_driver.generate_messages_with_hooks.return_value = {
            "text": "response",
            "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0},
        }
        conv = Conversation(driver=mock_driver)
        return conv, mock_driver

    def test_ask_without_images_sends_string_content(self):
        conv, mock = self._make_conv()
        conv.ask("hello")
        msgs = mock.generate_messages_with_hooks.call_args[0][0]
        assert msgs[-1]["content"] == "hello"

    def test_ask_with_images_sends_list_content(self):
        conv, mock = self._make_conv()
        conv.ask("describe this", images=[_PNG_BYTES])
        msgs = mock.generate_messages_with_hooks.call_args[0][0]
        content = msgs[-1]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "describe this"}
        assert content[1]["type"] == "image"

    def test_ask_records_images_in_history(self):
        conv, _ = self._make_conv()
        conv.ask("describe this", images=[_PNG_BYTES])
        # History should have user + assistant
        assert len(conv.messages) == 2
        user_msg = conv.messages[0]
        assert isinstance(user_msg["content"], list)

    def test_ask_stream_with_images(self):
        conv, _mock = self._make_conv()
        # Fall back to non-streaming
        result = list(conv.ask_stream("describe", images=[_PNG_BYTES]))
        assert result == ["response"]


# ---------------------------------------------------------------------------
# TestPrepareMessages — provider-specific wire formats
# ---------------------------------------------------------------------------


def _universal_msgs(text: str = "describe") -> list[dict[str, Any]]:
    """Build a universal messages list with one image block."""
    ic = image_from_bytes(_PNG_BYTES)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image", "source": ic},
            ],
        }
    ]


class TestPrepareMessagesOpenAI:
    def test_converts_to_image_url(self):
        from prompture.drivers.vision_helpers import _prepare_openai_vision_messages

        result = _prepare_openai_vision_messages(_universal_msgs())
        content = result[0]["content"]
        assert content[0] == {"type": "text", "text": "describe"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_url_source_passes_through(self):
        from prompture.drivers.vision_helpers import _prepare_openai_vision_messages

        ic = image_from_url("https://example.com/img.png")
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image", "source": ic}]}]
        result = _prepare_openai_vision_messages(msgs)
        assert result[0]["content"][1]["image_url"]["url"] == "https://example.com/img.png"

    def test_string_content_unchanged(self):
        from prompture.drivers.vision_helpers import _prepare_openai_vision_messages

        msgs = [{"role": "user", "content": "plain text"}]
        result = _prepare_openai_vision_messages(msgs)
        assert result[0]["content"] == "plain text"


class TestPrepareMessagesClaude:
    def test_converts_to_claude_format(self):
        from prompture.drivers.vision_helpers import _prepare_claude_vision_messages

        result = _prepare_claude_vision_messages(_universal_msgs())
        content = result[0]["content"]
        assert content[0] == {"type": "text", "text": "describe"}
        img_block = content[1]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/png"
        assert img_block["source"]["data"] == _PNG_B64

    def test_url_source_uses_url_type(self):
        from prompture.drivers.vision_helpers import _prepare_claude_vision_messages

        ic = image_from_url("https://example.com/img.jpg")
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image", "source": ic}]}]
        result = _prepare_claude_vision_messages(msgs)
        assert result[0]["content"][1]["source"]["type"] == "url"
        assert result[0]["content"][1]["source"]["url"] == "https://example.com/img.jpg"


class TestPrepareMessagesGoogle:
    def test_converts_to_gemini_parts(self):
        from prompture.drivers.vision_helpers import _prepare_google_vision_messages

        result = _prepare_google_vision_messages(_universal_msgs())
        content = result[0]["content"]
        assert content[0] == "describe"  # text becomes plain string
        assert content[1]["inline_data"]["mime_type"] == "image/png"
        assert content[1]["inline_data"]["data"] == _PNG_B64
        assert result[0]["_vision_parts"] is True


class TestPrepareMessagesOllama:
    def test_separates_images_field(self):
        from prompture.drivers.vision_helpers import _prepare_ollama_vision_messages

        result = _prepare_ollama_vision_messages(_universal_msgs())
        assert result[0]["content"] == "describe"
        assert result[0]["images"] == [_PNG_B64]

    def test_no_images_field_for_text_only(self):
        from prompture.drivers.vision_helpers import _prepare_ollama_vision_messages

        msgs = [{"role": "user", "content": "plain text"}]
        result = _prepare_ollama_vision_messages(msgs)
        assert result[0]["content"] == "plain text"
        assert "images" not in result[0]


# ---------------------------------------------------------------------------
# TestFlattenWithImages — _flatten_messages handles image blocks
# ---------------------------------------------------------------------------


class TestFlattenWithImages:
    def test_flatten_with_image_blocks(self):
        from prompture.drivers.base import Driver

        msgs = _universal_msgs("hello")
        result = Driver._flatten_messages(msgs)
        assert "[image]" in result
        assert "hello" in result

    def test_flatten_string_content_unchanged(self):
        from prompture.drivers.base import Driver

        msgs = [{"role": "user", "content": "plain text"}]
        result = Driver._flatten_messages(msgs)
        assert result == "[User]: plain text"


# ---------------------------------------------------------------------------
# TestNonVisionDriverRaises
# ---------------------------------------------------------------------------


class TestNonVisionDriverRaises:
    def test_check_raises_on_image_input(self):
        from prompture.drivers.base import Driver

        d = Driver()
        assert d.supports_vision is False
        with pytest.raises(NotImplementedError, match="does not support vision"):
            d._check_vision_support(_universal_msgs())

    def test_check_passes_for_text_only(self):
        from prompture.drivers.base import Driver

        d = Driver()
        d._check_vision_support([{"role": "user", "content": "hello"}])

    def test_prepare_messages_raises_on_non_vision_driver(self):
        from prompture.drivers.base import Driver

        d = Driver()
        with pytest.raises(NotImplementedError, match="does not support vision"):
            d._prepare_messages(_universal_msgs())


# ---------------------------------------------------------------------------
# TestBackwardCompatibility — string-only messages unchanged
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_conversation_ask_without_images(self):
        """String-only ask still works identically."""
        from prompture.agents.conversation import Conversation

        mock_driver = MagicMock()
        mock_driver.supports_tool_use = False
        mock_driver.supports_vision = True
        mock_driver.generate_messages_with_hooks.return_value = {
            "text": "ok",
            "meta": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0},
        }
        conv = Conversation(driver=mock_driver)
        result = conv.ask("hello")
        assert result == "ok"
        msgs = mock_driver.generate_messages_with_hooks.call_args[0][0]
        # Content should be a plain string
        assert msgs[-1]["content"] == "hello"

    def test_build_messages_no_images(self):
        from prompture.agents.conversation import Conversation

        mock_driver = MagicMock()
        mock_driver.supports_tool_use = False
        conv = Conversation(driver=mock_driver, system_prompt="sys")
        msgs = conv._build_messages("hello")
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1] == {"role": "user", "content": "hello"}

    def test_add_context_without_images(self):
        from prompture.agents.conversation import Conversation

        mock_driver = MagicMock()
        mock_driver.supports_tool_use = False
        conv = Conversation(driver=mock_driver)
        conv.add_context("user", "old message")
        assert conv.messages[0]["content"] == "old message"
