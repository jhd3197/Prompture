"""Conversation serialization — pure data transforms for export/import.

Handles converting Conversation state to/from plain dicts suitable for
JSON serialization.  No I/O is performed here; see :mod:`persistence`
for file and database storage.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from typing import Any

from ..infra.session import UsageSession
from ..media.image import ImageContent

logger = logging.getLogger(__name__)

EXPORT_VERSION = 1

_SENSITIVE_SUBSTRINGS = frozenset(
    {
        "api_key",
        "secret",
        "token",
        "password",
        "auth",
        "credential",
        "secret_key",
        "access_key",
        "private_key",
    }
)


def _scrub_sensitive(options: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *options* with keys containing sensitive substrings removed."""
    return {k: v for k, v in options.items() if not any(sub in k.lower() for sub in _SENSITIVE_SUBSTRINGS)}


# ------------------------------------------------------------------
# Message content helpers
# ------------------------------------------------------------------


def _serialize_message_content(content: Any) -> Any:
    """Convert ``ImageContent`` objects inside message content to plain dicts."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        out: list[Any] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                source = block.get("source")
                if isinstance(source, ImageContent):
                    out.append(
                        {
                            "type": "image",
                            "source": {
                                "data": source.data,
                                "media_type": source.media_type,
                                "source_type": source.source_type,
                                "url": source.url,
                            },
                        }
                    )
                elif isinstance(source, dict):
                    out.append(block)
                else:
                    out.append(block)
            else:
                out.append(block)
        return out

    return content


def _deserialize_message_content(content: Any) -> Any:
    """Reconstruct ``ImageContent`` objects from plain dicts in message content."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        out: list[Any] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                source = block.get("source")
                if isinstance(source, dict) and "media_type" in source:
                    out.append(
                        {
                            "type": "image",
                            "source": ImageContent(
                                data=source.get("data", ""),
                                media_type=source["media_type"],
                                source_type=source.get("source_type", "base64"),
                                url=source.get("url"),
                            ),
                        }
                    )
                else:
                    out.append(block)
            else:
                out.append(block)
        return out

    return content


# ------------------------------------------------------------------
# UsageSession export/import
# ------------------------------------------------------------------


def export_usage_session(session: UsageSession) -> dict[str, Any]:
    """Serialize a :class:`UsageSession` to a plain dict."""
    return {
        "prompt_tokens": session.prompt_tokens,
        "completion_tokens": session.completion_tokens,
        "total_tokens": session.total_tokens,
        "cost": session.cost,
        "total_cost": session.cost,  # Deprecated alias for backwards compatibility
        "call_count": session.call_count,
        "errors": session.errors,
        "per_model": dict(session._per_model),
    }


def import_usage_session(data: dict[str, Any]) -> UsageSession:
    """Reconstruct a :class:`UsageSession` from an exported dict."""
    # Support both 'cost' (new) and 'total_cost' (deprecated) for backwards compatibility
    cost_value = data.get("cost", data.get("total_cost", 0.0))
    session = UsageSession(
        prompt_tokens=data.get("prompt_tokens", 0),
        completion_tokens=data.get("completion_tokens", 0),
        total_tokens=data.get("total_tokens", 0),
        cost=cost_value,
        call_count=data.get("call_count", 0),
        errors=data.get("errors", 0),
    )
    per_model = data.get("per_model", {})
    for model, stats in per_model.items():
        session._per_model[model] = dict(stats)
    return session


# ------------------------------------------------------------------
# Conversation export/import
# ------------------------------------------------------------------


def export_conversation(
    *,
    model_name: str,
    system_prompt: str | None,
    options: dict[str, Any],
    messages: list[dict[str, Any]],
    usage: dict[str, Any],
    max_tool_rounds: int,
    tools_metadata: list[dict[str, Any]] | None = None,
    usage_session: UsageSession | None = None,
    metadata: dict[str, Any] | None = None,
    conversation_id: str,
    strip_images: bool = False,
) -> dict[str, Any]:
    """Export conversation state to a JSON-serializable dict.

    Args:
        strip_images: When *True*, image blocks are removed from messages
            and list-of-blocks content that becomes text-only is collapsed
            to a plain string.
    """
    serialized_messages: list[dict[str, Any]] = []
    for msg in messages:
        msg_copy = dict(msg)
        content = msg_copy.get("content")

        if strip_images and isinstance(content, list):
            filtered = [b for b in content if not (isinstance(b, dict) and b.get("type") == "image")]
            if len(filtered) == 1 and isinstance(filtered[0], dict) and filtered[0].get("type") == "text":
                msg_copy["content"] = filtered[0]["text"]
            elif filtered:
                msg_copy["content"] = _serialize_message_content(filtered)
            else:
                msg_copy["content"] = ""
        else:
            msg_copy["content"] = _serialize_message_content(content)

        serialized_messages.append(msg_copy)

    now = datetime.now(timezone.utc).isoformat()
    meta = dict(metadata) if metadata else {}
    meta.setdefault("created_at", now)
    meta["last_active"] = now
    meta["turn_count"] = usage.get("turns", 0)

    export: dict[str, Any] = {
        "version": EXPORT_VERSION,
        "conversation_id": conversation_id,
        "model_name": model_name,
        "system_prompt": system_prompt,
        "options": _scrub_sensitive(dict(options)),
        "messages": serialized_messages,
        "usage": dict(usage),
        "max_tool_rounds": max_tool_rounds,
        "metadata": meta,
    }

    if tools_metadata:
        export["tools"] = tools_metadata

    if usage_session is not None:
        export["usage_session"] = export_usage_session(usage_session)

    return export


def import_conversation(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and deserialize an exported conversation dict.

    Returns a dict with deserialized messages (``ImageContent`` objects
    reconstructed).

    Raises:
        ValueError: If the export version is unsupported.
    """
    version = data.get("version")
    if version != EXPORT_VERSION:
        raise ValueError(f"Unsupported export version: {version}. Expected {EXPORT_VERSION}.")

    result = copy.deepcopy(data)

    # Defense-in-depth: scrub sensitive keys from imported options
    if isinstance(result.get("options"), dict):
        result["options"] = _scrub_sensitive(result["options"])

    # Validate and filter messages
    _VALID_ROLES = {"user", "assistant", "system", "tool"}
    valid_messages: list[dict[str, Any]] = []
    for i, msg in enumerate(result.get("messages", [])):
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or role not in _VALID_ROLES:
            logger.warning("import_conversation: dropping message %d with invalid role: %r", i, role)
            continue
        if not isinstance(content, (str, list)):
            logger.warning(
                "import_conversation: dropping message %d with invalid content type: %s", i, type(content).__name__
            )
            continue
        valid_messages.append(msg)
    result["messages"] = valid_messages

    # Deserialize message content
    for msg in result["messages"]:
        if "content" in msg:
            msg["content"] = _deserialize_message_content(msg["content"])

    # Deserialize usage_session if present
    if "usage_session" in result and isinstance(result["usage_session"], dict):
        result["usage_session"] = import_usage_session(result["usage_session"])

    return result
