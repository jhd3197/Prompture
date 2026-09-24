"""Anthropic Messages API wire format ↔ Prompture drivers.

Lets any Prompture model (including combos and ``auto/`` routes) sit behind
an Anthropic-compatible ``/v1/messages`` endpoint — which is what Claude
Code and the Anthropic SDKs speak.

* request side — :func:`anthropic_to_driver` → ``(messages, tools, options)``
  in Prompture's universal (OpenAI-shaped) format
* one-shot — :func:`anthropic_message` builds a ``message`` object from a
  :class:`~.openai_format.ChatOutcome`
* streaming — :func:`live_events_for` picks the best streaming method the
  driver has; :func:`stream_anthropic_events` turns its events into the
  ``message_start`` … ``message_stop`` sequence; :func:`anthropic_sse`
  frames one event.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from .openai_format import ChatOutcome, run_chat

# ---------------------------------------------------------------------------
# Request side
# ---------------------------------------------------------------------------


def _block_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(p for p in parts if p)


def _image_block(block: dict[str, Any]) -> dict[str, Any] | None:
    from ..media.image import make_image

    src = block.get("source") or {}
    if src.get("type") == "base64" and src.get("data"):
        uri = f"data:{src.get('media_type', 'image/png')};base64,{src['data']}"
        return {"type": "image", "source": make_image(uri)}
    if src.get("type") == "url" and src.get("url"):
        return {"type": "image", "source": make_image(src["url"])}
    return None


def _convert_message(msg: dict[str, Any]) -> list[dict[str, Any]]:
    role = msg.get("role", "user")
    content = msg.get("content")
    if isinstance(content, str) or content is None:
        return [{"role": role, "content": content or ""}]

    out: list[dict[str, Any]] = []
    text_parts: list[str] = []
    images: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text_parts.append(block.get("text", ""))
        elif kind == "image":
            img = _image_block(block)
            if img:
                images.append(img)
        elif kind == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {"name": block.get("name", ""), "arguments": json.dumps(block.get("input") or {})},
                }
            )
        elif kind == "tool_result":
            result = _block_text(block.get("content"))
            if block.get("is_error"):
                result = f"[tool error] {result}"
            out.append({"role": "tool", "tool_call_id": block.get("tool_use_id", ""), "content": result})
        # thinking / redacted_thinking / server-tool blocks are provider-specific; drop them.

    text = "\n".join(t for t in text_parts if t)
    if tool_calls:
        out.append({"role": "assistant", "content": text, "tool_calls": tool_calls})
    elif images:
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        out.append({"role": role, "content": blocks + images})
    elif text or not out:
        out.append({"role": role, "content": text})
    return out


def anthropic_tools_to_openai(tools: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Custom tools → OpenAI function tools. Anthropic server tools are skipped."""
    out = []
    for t in tools or []:
        if t.get("type") not in (None, "custom") or "input_schema" not in t:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def _tool_choice(choice: Any) -> Any:
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    if kind == "any":
        return "required"
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    if kind in ("auto", "none"):
        return kind
    return None


def anthropic_to_driver(body: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Anthropic ``/v1/messages`` request → ``(messages, tools, options)``."""
    data = body if isinstance(body, dict) else body.model_dump(exclude_none=True)
    messages: list[dict[str, Any]] = []
    system = _block_text(data.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for msg in data.get("messages") or []:
        messages.extend(_convert_message(msg if isinstance(msg, dict) else dict(msg)))

    options: dict[str, Any] = {}
    for src, dst in (("max_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p")):
        if data.get(src) is not None:
            options[dst] = data[src]
    if data.get("stop_sequences"):
        options["stop"] = data["stop_sequences"]
    choice = _tool_choice(data.get("tool_choice"))
    if choice is not None:
        options["tool_choice"] = choice
    return messages, anthropic_tools_to_openai(data.get("tools")), options


# ---------------------------------------------------------------------------
# Response side
# ---------------------------------------------------------------------------

_STOP_REASONS = {
    "end_turn": "end_turn",
    "stop": "end_turn",
    "tool_use": "tool_use",
    "tool_calls": "tool_use",
    "max_tokens": "max_tokens",
    "length": "max_tokens",
    "stop_sequence": "stop_sequence",
    "content_filter": "refusal",
    "refusal": "refusal",
}


def anthropic_stop_reason(stop_reason: str | None, *, has_tool_calls: bool = False) -> str:
    if has_tool_calls:
        return "tool_use"
    return _STOP_REASONS.get((stop_reason or "end_turn").lower(), "end_turn")


def anthropic_usage(meta: dict[str, Any] | None) -> dict[str, Any]:
    meta = meta or {}
    usage = {
        "input_tokens": int(meta.get("prompt_tokens", 0) or 0),
        "output_tokens": int(meta.get("completion_tokens", 0) or 0),
    }
    if meta.get("cached_prompt_tokens"):
        usage["cache_read_input_tokens"] = int(meta["cached_prompt_tokens"])
    if meta.get("cache_creation_tokens"):
        usage["cache_creation_input_tokens"] = int(meta["cache_creation_tokens"])
    return usage


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _tool_input(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args or "{}")
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"_raw": args}
    return {}


def anthropic_message(outcome: ChatOutcome, *, model: str, message_id: str | None = None) -> dict[str, Any]:
    """A complete Anthropic ``message`` object from a chat outcome."""
    content: list[dict[str, Any]] = []
    if outcome.text:
        content.append({"type": "text", "text": outcome.text})
    for tc in outcome.tool_calls:
        fn = tc.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": tc.get("name") or fn.get("name", ""),
                "input": _tool_input(tc.get("arguments", fn.get("arguments"))),
            }
        )
    return {
        "id": message_id or new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": anthropic_stop_reason(outcome.stop_reason, has_tool_calls=bool(outcome.tool_calls)),
        "stop_sequence": None,
        "usage": anthropic_usage(outcome.meta),
    }


def anthropic_error(message: str, *, type_: str = "api_error") -> dict[str, Any]:
    return {"type": "error", "error": {"type": type_, "message": message}}


def anthropic_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'), default=str)}\n\n"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def live_events_for(
    driver: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    options: dict[str, Any],
) -> Iterator[Any]:
    """Stream one assistant turn using the best method *driver* offers.

    Yields either :mod:`~prompture.agents.live_events` objects (tool-capable
    path) or ``{"type": "delta"|"done"}`` dicts (plain streaming); when the
    driver can't stream at all, the buffered answer is replayed as events.
    """
    if tools and getattr(driver, "supports_tool_use", False):
        yield from driver.generate_messages_with_tools_stream(messages, tools, dict(options))
        return
    if not tools and getattr(driver, "supports_streaming", False):
        yield from driver.generate_messages_stream(messages, dict(options))
        return
    from ..agents.live_events import MessageStop, TextDelta, ToolUseStart, ToolUseStop

    outcome = run_chat(driver, messages, options, tools=tools or None)
    if outcome.text:
        yield TextDelta(text=outcome.text)
    for tc in outcome.tool_calls:
        tc_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"
        yield ToolUseStart(id=tc_id, name=tc.get("name", ""))
        yield ToolUseStop(id=tc_id, name=tc.get("name", ""), input=_tool_input(tc.get("arguments")))
    yield MessageStop(stop_reason=outcome.stop_reason or "end_turn", usage=outcome.meta)


def stream_anthropic_events(
    events: Iterable[Any],
    *,
    model: str,
    message_id: str | None = None,
    on_complete: Callable[[ChatOutcome], None] | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Driver stream events → Anthropic ``(event_name, data)`` pairs.

    Thinking deltas are dropped: clients replay thinking blocks with a
    provider signature we can't produce for non-Anthropic models.
    """
    mid = message_id or new_message_id()
    outcome = ChatOutcome()
    index = -1
    open_kind: str | None = None
    tool_ids: dict[str, int] = {}
    tool_has_delta: set[str] = set()

    def close() -> Iterator[tuple[str, dict[str, Any]]]:
        nonlocal open_kind
        if open_kind is not None:
            yield "content_block_stop", {"type": "content_block_stop", "index": index}
            open_kind = None

    def open_block(kind: str, block: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
        nonlocal index, open_kind
        yield from close()
        index += 1
        open_kind = kind
        yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": block}

    def text(t: str) -> Iterator[tuple[str, dict[str, Any]]]:
        if not t:
            return
        outcome.text += t
        if open_kind != "text":
            yield from open_block("text", {"type": "text", "text": ""})
        yield (
            "content_block_delta",
            {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": t}},
        )

    yield (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": mid,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    try:
        for ev in events:
            if isinstance(ev, dict):
                if ev.get("type") == "delta":
                    yield from text(ev.get("text", "") or "")
                elif ev.get("type") == "done":
                    outcome.meta = ev.get("meta") or {}
                    outcome.stop_reason = ev.get("stop_reason") or outcome.meta.get("stop_reason")
                continue
            kind = getattr(ev, "event_type", None)
            if kind == "text_delta":
                yield from text(ev.text)
            elif kind == "tool_use_start":
                yield from open_block("tool_use", {"type": "tool_use", "id": ev.id, "name": ev.name, "input": {}})
                tool_ids[ev.id] = index
            elif kind == "tool_input_delta":
                tool_has_delta.add(ev.id)
                yield (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": tool_ids.get(ev.id, index),
                        "delta": {"type": "input_json_delta", "partial_json": ev.fragment},
                    },
                )
            elif kind == "tool_use_stop":
                if ev.id not in tool_has_delta:
                    yield (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": tool_ids.get(ev.id, index),
                            "delta": {"type": "input_json_delta", "partial_json": json.dumps(ev.input or {})},
                        },
                    )
                outcome.tool_calls.append({"id": ev.id, "name": ev.name, "arguments": ev.input or {}})
                yield from close()
            elif kind == "message_stop":
                outcome.stop_reason = ev.stop_reason
                outcome.meta = dict(ev.usage or {})
    except Exception as exc:
        outcome.error = exc
        yield from close()
        if on_complete:
            on_complete(outcome)
        yield "error", anthropic_error(str(exc))
        return

    yield from close()
    yield (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": anthropic_stop_reason(outcome.stop_reason, has_tool_calls=bool(outcome.tool_calls)),
                "stop_sequence": None,
            },
            "usage": anthropic_usage(outcome.meta),
        },
    )
    yield "message_stop", {"type": "message_stop"}
    if on_complete:
        on_complete(outcome)


def estimate_input_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    """Cheap ~4-chars-per-token estimate for ``count_tokens`` when no provider counter exists."""
    chars = sum(len(json.dumps(m.get("content"), default=str)) for m in messages)
    chars += len(json.dumps(tools or [], default=str))
    return max(1, chars // 4)
