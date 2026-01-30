"""Shared helpers for converting universal vision message blocks to provider-specific formats."""

from __future__ import annotations

from typing import Any


def _prepare_openai_vision_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert universal image blocks to OpenAI-compatible vision format.

    Works for OpenAI, Azure, Groq, Grok, LM Studio, and OpenRouter.

    Universal format::

        {"type": "image", "source": ImageContent(...)}

    OpenAI format::

        {"type": "image_url", "image_url": {"url": "data:mime;base64,..."}}
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        new_blocks: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                source = block["source"]
                if source.source_type == "url" and source.url:
                    url = source.url
                else:
                    url = f"data:{source.media_type};base64,{source.data}"
                new_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    }
                )
            else:
                new_blocks.append(block)
        out.append({**msg, "content": new_blocks})
    return out


def _prepare_claude_vision_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert universal image blocks to Anthropic Claude format.

    Claude format::

        {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        new_blocks: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                source = block["source"]
                if source.source_type == "url" and source.url:
                    new_blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": source.url,
                            },
                        }
                    )
                else:
                    new_blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": source.media_type,
                                "data": source.data,
                            },
                        }
                    )
            else:
                new_blocks.append(block)
        out.append({**msg, "content": new_blocks})
    return out


def _prepare_google_vision_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert universal image blocks to Google Gemini format.

    Gemini expects ``parts`` arrays containing text and inline_data dicts::

        {"role": "user", "parts": [
            "text prompt",
            {"inline_data": {"mime_type": "image/png", "data": "base64..."}},
        ]}
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        # Convert content blocks to Gemini parts
        parts: list[Any] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
            elif isinstance(block, dict) and block.get("type") == "image":
                source = block["source"]
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": source.media_type,
                            "data": source.data,
                        }
                    }
                )
            else:
                parts.append(block)
        out.append({**msg, "content": parts, "_vision_parts": True})
    return out


def _prepare_ollama_vision_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert universal image blocks to Ollama format.

    Ollama expects images as a separate field::

        {"role": "user", "content": "text", "images": ["base64..."]}
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        text_parts: list[str] = []
        images: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block["text"])
            elif isinstance(block, dict) and block.get("type") == "image":
                source = block["source"]
                images.append(source.data)
        new_msg = {**msg, "content": " ".join(text_parts)}
        if images:
            new_msg["images"] = images
        out.append(new_msg)
    return out
