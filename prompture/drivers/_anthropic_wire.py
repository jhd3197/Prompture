"""One implementation of tool calling for every Anthropic-wire endpoint.

Claude is served by Anthropic directly, by Azure, by Vertex AI and by Bedrock.
All four take the Anthropic Messages format, which differs from the canonical
(OpenAI chat) shape conversations are stored in: the system prompt is a
top-level field, an assistant's tool calls are ``tool_use`` content blocks, a
tool's answer is a ``tool_result`` block inside a *user* message, and
``tool_choice`` has its own vocabulary.

This module does that translation once. A driver adds only what is its own:
the client, prompt caching, pricing.
"""

from __future__ import annotations

import json
from typing import Any

from ..infra.tool_schema import flatten_tool_schema
from .base import _normalize_stop_reason, _translate_tool_choice

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


def _image_block(source: dict[str, Any]) -> dict[str, Any]:
    return {"type": "image", "source": source}


def _image_from_url(url: str) -> dict[str, Any]:
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header.split(";")[0].split(":", 1)[-1] or "image/jpeg"
        return _image_block({"type": "base64", "media_type": media_type, "data": data})
    return _image_block({"type": "url", "url": url})


def normalize_content(content: Any) -> Any:
    """Content blocks as Anthropic takes them.

    Accepts the universal image block (``{"type": "image", "source":
    ImageContent}``), an OpenAI ``image_url`` block, or a block already in
    Anthropic's shape, so it is safe to call on messages another helper has
    prepared. Strings and unknown blocks pass through.
    """
    if not isinstance(content, list):
        return content
    out: list[Any] = []
    for block in content:
        if not isinstance(block, dict):
            out.append(block)
            continue
        kind = block.get("type")
        if kind == "image":
            source = block.get("source")
            if isinstance(source, dict):
                out.append(block)
            elif source is not None and getattr(source, "source_type", None) == "url" and source.url:
                out.append(_image_block({"type": "url", "url": source.url}))
            elif source is not None:
                out.append(_image_block({"type": "base64", "media_type": source.media_type, "data": source.data}))
            continue
        if kind == "image_url":
            image = block.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if isinstance(url, str) and url:
                out.append(_image_from_url(url))
            continue
        out.append(block)
    return out


def _tool_input(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_raw": raw}
    return {}


def _is_tool_results(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, list)
        and bool(content)
        and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    )


def to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Canonical messages → ``(system, messages)`` for the Messages API.

    - the system message is lifted out;
    - an assistant message with ``tool_calls`` becomes ``tool_use`` blocks;
    - each ``role: "tool"`` message becomes a ``tool_result`` block, and
      consecutive ones share a single user turn, as Anthropic expects after an
      assistant turn with several tool calls;
    - anything else keeps only ``role`` and ``content``, because the API
      rejects keys it does not know (``reasoning_content``, ``name`` …).
    """
    system: str | None = None
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system = msg.get("content", "")
            continue

        if role == "assistant" and msg.get("tool_calls"):
            blocks: list[dict[str, Any]] = []
            content = msg.get("content")
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                blocks.extend(normalize_content(content))
            for call in msg["tool_calls"]:
                fn = call.get("function", call)
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id", ""),
                        "name": fn.get("name", call.get("name", "")),
                        "input": _tool_input(fn.get("arguments", call.get("arguments", {}))),
                    }
                )
            out.append({"role": "assistant", "content": blocks})
            continue

        if role == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": normalize_content(msg.get("content", "")),
            }
            if out and _is_tool_results(out[-1]):
                out[-1]["content"].append(result)
            else:
                out.append({"role": "user", "content": [result]})
            continue

        out.append({"role": role, "content": normalize_content(msg.get("content"))})
    return system, out


def to_anthropic_tools(tools: list[dict[str, Any]], *, strict: bool = False) -> list[dict[str, Any]]:
    """OpenAI- or Anthropic-shaped tool dicts → Anthropic tools, schemas flattened.

    With *strict*, client tools are marked ``"strict": true`` so the model's
    arguments are guaranteed to match the schema.
    """
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            converted: dict[str, Any] = {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": flatten_tool_schema(fn.get("parameters", {"type": "object", "properties": {}})),
            }
        elif "input_schema" in tool:
            converted = {**tool, "input_schema": flatten_tool_schema(tool["input_schema"])}
        else:
            # A server tool (web search, code execution …): pass through untouched.
            out.append(tool)
            continue
        if strict and "strict" not in converted:
            converted["strict"] = True
        out.append(converted)
    return out


def build_tool_request(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    options: dict[str, Any],
    *,
    model: str,
    max_tokens: int = 4096,
    temperature: float | None = 0.0,
) -> dict[str, Any]:
    """The Messages API arguments for one tool-call turn.

    ``options`` wins over *max_tokens* / *temperature*. The result has
    ``model``, ``messages``, ``max_tokens`` and ``tools``, plus ``system``,
    ``temperature`` and ``tool_choice`` when they apply.
    """
    system, api_messages = to_anthropic_messages(messages)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": api_messages,
        "max_tokens": int(options.get("max_tokens") or max_tokens),
        "tools": to_anthropic_tools(tools, strict=bool(options.get("strict_tools"))),
    }
    temperature = options.get("temperature", temperature)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if system:
        kwargs["system"] = system
    tool_choice = _translate_tool_choice(options.get("tool_choice"), "anthropic")
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    return kwargs


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    value = getattr(obj, name, default)
    return default if value is None else value


def parse_message(resp: Any) -> dict[str, Any]:
    """An Anthropic message (SDK object or JSON dict) → the normalized pieces.

    Returns ``text``, ``tool_calls``, ``stop_reason`` (shared vocabulary),
    ``raw_stop_reason``, ``prompt_tokens``, ``completion_tokens`` and, when the
    model thought out loud, ``reasoning_content``.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in _field(resp, "content", []) or []:
        kind = _field(block, "type")
        if kind == "text":
            text_parts.append(_field(block, "text", "") or "")
        elif kind == "thinking":
            thought = _field(block, "thinking", "")
            if thought:
                thinking_parts.append(thought)
        elif kind == "tool_use":
            arguments = _field(block, "input", {})
            tool_calls.append(
                {
                    "id": _field(block, "id", "") or "",
                    "name": _field(block, "name", "") or "",
                    "arguments": arguments if isinstance(arguments, dict) else {},
                }
            )
    usage = _field(resp, "usage", {}) or {}
    raw_stop_reason = _field(resp, "stop_reason")
    parsed: dict[str, Any] = {
        "text": "".join(text_parts),
        "tool_calls": tool_calls,
        "stop_reason": _normalize_stop_reason(raw_stop_reason, tool_calls_present=bool(tool_calls)),
        "raw_stop_reason": raw_stop_reason,
        "prompt_tokens": int(_field(usage, "input_tokens", 0) or 0),
        "completion_tokens": int(_field(usage, "output_tokens", 0) or 0),
    }
    if thinking_parts:
        parsed["reasoning_content"] = "\n".join(thinking_parts)
    return parsed


def tool_result(parsed: dict[str, Any], *, model: str, cost: float, raw_response: Any) -> dict[str, Any]:
    """Wrap :func:`parse_message` output as a ``generate_messages_with_tools`` result."""
    prompt_tokens = parsed["prompt_tokens"]
    completion_tokens = parsed["completion_tokens"]
    result: dict[str, Any] = {
        "text": parsed["text"],
        "meta": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": round(cost, 6),
            "raw_response": raw_response,
            "model_name": model,
            "raw_stop_reason": parsed["raw_stop_reason"],
        },
        "tool_calls": parsed["tool_calls"],
        "stop_reason": parsed["stop_reason"],
    }
    if "reasoning_content" in parsed:
        result["reasoning_content"] = parsed["reasoning_content"]
    return result


__all__ = [
    "build_tool_request",
    "normalize_content",
    "parse_message",
    "to_anthropic_messages",
    "to_anthropic_tools",
    "tool_result",
]
