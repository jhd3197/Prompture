"""OpenAI Responses API (``/v1/responses``) wire format ↔ Prompture drivers.

Stateless subset: ``input`` (string or item list with messages,
``function_call`` and ``function_call_output`` items), ``instructions``,
function ``tools``, sampling options, and streaming. ``previous_response_id``
needs server-side storage and is left to the caller to reject or resolve.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from .anthropic_format import _tool_input
from .openai_format import ChatOutcome

# ---------------------------------------------------------------------------
# Request side
# ---------------------------------------------------------------------------


def _content_text_and_images(content: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            texts.append(part.get("text", ""))
        elif kind == "input_image" and (part.get("image_url") or part.get("url")):
            from ..media.image import make_image

            images.append({"type": "image", "source": make_image(part.get("image_url") or part.get("url"))})
    return "\n".join(t for t in texts if t), images


def responses_tools_to_openai(tools: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Flat Responses function tools → chat-completions function tools. Built-in tools are skipped."""
    out = []
    for t in tools or []:
        if t.get("type") != "function":
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def responses_to_driver(body: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Responses API request → ``(messages, tools, options)``."""
    data = body if isinstance(body, dict) else body.model_dump(exclude_none=True)
    messages: list[dict[str, Any]] = []
    if data.get("instructions"):
        messages.append({"role": "system", "content": data["instructions"]})

    raw_input = data.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    else:
        pending_calls: list[dict[str, Any]] = []

        def flush_calls() -> None:
            if pending_calls:
                messages.append({"role": "assistant", "content": "", "tool_calls": list(pending_calls)})
                pending_calls.clear()

        for item in raw_input or []:
            kind = item.get("type", "message")
            if kind == "function_call":
                pending_calls.append(
                    {
                        "id": item.get("call_id") or item.get("id", ""),
                        "type": "function",
                        "function": {"name": item.get("name", ""), "arguments": item.get("arguments") or "{}"},
                    }
                )
                continue
            flush_calls()
            if kind == "function_call_output":
                output = item.get("output")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": output if isinstance(output, str) else json.dumps(output),
                    }
                )
            elif kind == "message":
                role = item.get("role", "user")
                role = "system" if role == "developer" else role
                text, images = _content_text_and_images(item.get("content"))
                if images:
                    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
                    messages.append({"role": role, "content": blocks + images})
                else:
                    messages.append({"role": role, "content": text})
            # reasoning / built-in tool items are provider-specific; skip.
        flush_calls()

    options: dict[str, Any] = {}
    for src, dst in (("max_output_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p")):
        if data.get(src) is not None:
            options[dst] = data[src]
    effort = (data.get("reasoning") or {}).get("effort")
    if effort:
        options["reasoning_effort"] = effort
    choice = data.get("tool_choice")
    if isinstance(choice, str):
        options["tool_choice"] = choice
    elif isinstance(choice, dict) and choice.get("type") == "function" and choice.get("name"):
        options["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    fmt = ((data.get("text") or {}).get("format")) or {}
    if fmt.get("type") == "json_object":
        options["json_mode"] = True
    elif fmt.get("type") == "json_schema":
        options["json_mode"] = True
        if fmt.get("schema"):
            options["json_schema"] = fmt["schema"]
    return messages, responses_tools_to_openai(data.get("tools")), options


# ---------------------------------------------------------------------------
# Response side
# ---------------------------------------------------------------------------


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def _item_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def responses_usage(meta: dict[str, Any] | None) -> dict[str, Any]:
    meta = meta or {}
    inp = int(meta.get("prompt_tokens", 0) or 0)
    out = int(meta.get("completion_tokens", 0) or 0)
    usage: dict[str, Any] = {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}
    if meta.get("cached_prompt_tokens"):
        usage["input_tokens_details"] = {"cached_tokens": int(meta["cached_prompt_tokens"])}
    return usage


def _message_item(text: str, item_id: str | None = None) -> dict[str, Any]:
    return {
        "type": "message",
        "id": item_id or _item_id("msg"),
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _call_item(call_id: str, name: str, arguments: str, item_id: str | None = None) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": item_id or _item_id("fc"),
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def response_object(
    outcome: ChatOutcome,
    *,
    model: str,
    response_id: str | None = None,
    output: list[dict[str, Any]] | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    """A complete ``response`` object from a chat outcome."""
    if output is None:
        output = []
        if outcome.text:
            output.append(_message_item(outcome.text))
        for tc in outcome.tool_calls:
            args = tc.get("arguments", {})
            output.append(
                _call_item(
                    tc.get("id") or _item_id("call"),
                    tc.get("name", ""),
                    args if isinstance(args, str) else json.dumps(args or {}),
                )
            )
    body: dict[str, Any] = {
        "id": response_id or new_response_id(),
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "output": output,
        "usage": responses_usage(outcome.meta),
    }
    if outcome.stop_reason in ("max_tokens", "length"):
        body["status"] = "incomplete"
        body["incomplete_details"] = {"reason": "max_output_tokens"}
    return body


def responses_sse(data: dict[str, Any]) -> str:
    return f"event: {data['type']}\ndata: {json.dumps(data, separators=(',', ':'), default=str)}\n\n"


def stream_responses_events(
    events: Iterable[Any],
    *,
    model: str,
    response_id: str | None = None,
    on_complete: Callable[[ChatOutcome], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Driver stream events → Responses API streaming events (each has ``type`` + ``sequence_number``)."""
    rid = response_id or new_response_id()
    outcome = ChatOutcome()
    seq = 0
    output: list[dict[str, Any]] = []
    text_item: dict[str, Any] | None = None
    calls: dict[str, dict[str, Any]] = {}

    def ev(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal seq
        payload["sequence_number"] = seq
        seq += 1
        return payload

    def shell(status: str) -> dict[str, Any]:
        return response_object(outcome, model=model, response_id=rid, output=list(output), status=status)

    def close_text() -> Iterator[dict[str, Any]]:
        nonlocal text_item
        if text_item is None:
            return
        idx = output.index(text_item)
        text = text_item["content"][0]["text"]
        yield ev(
            {
                "type": "response.output_text.done",
                "item_id": text_item["id"],
                "output_index": idx,
                "content_index": 0,
                "text": text,
            }
        )
        yield ev(
            {
                "type": "response.content_part.done",
                "item_id": text_item["id"],
                "output_index": idx,
                "content_index": 0,
                "part": text_item["content"][0],
            }
        )
        text_item["status"] = "completed"
        yield ev({"type": "response.output_item.done", "output_index": idx, "item": text_item})
        text_item = None

    yield ev({"type": "response.created", "response": shell("in_progress")})
    try:
        for e in events:
            kind = e.get("type") if isinstance(e, dict) else getattr(e, "event_type", None)
            if kind in ("delta", "text_delta"):
                delta = (e.get("text") if isinstance(e, dict) else e.text) or ""
                if not delta:
                    continue
                outcome.text += delta
                if text_item is None:
                    text_item = _message_item("")
                    text_item["status"] = "in_progress"
                    output.append(text_item)
                    idx = len(output) - 1
                    yield ev(
                        {
                            "type": "response.output_item.added",
                            "output_index": idx,
                            "item": {**text_item, "content": []},
                        }
                    )
                    yield ev(
                        {
                            "type": "response.content_part.added",
                            "item_id": text_item["id"],
                            "output_index": idx,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []},
                        }
                    )
                text_item["content"][0]["text"] += delta
                yield ev(
                    {
                        "type": "response.output_text.delta",
                        "item_id": text_item["id"],
                        "output_index": output.index(text_item),
                        "content_index": 0,
                        "delta": delta,
                    }
                )
            elif kind == "done":
                outcome.meta = e.get("meta") or {}
                outcome.stop_reason = e.get("stop_reason") or outcome.meta.get("stop_reason")
            elif kind == "tool_use_start":
                yield from close_text()
                item = _call_item(e.id, e.name, "")
                item["status"] = "in_progress"
                output.append(item)
                calls[e.id] = item
                yield ev({"type": "response.output_item.added", "output_index": len(output) - 1, "item": dict(item)})
            elif kind == "tool_input_delta":
                item = calls.get(e.id)
                if item is not None:
                    item["arguments"] += e.fragment
                    yield ev(
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": item["id"],
                            "output_index": output.index(item),
                            "delta": e.fragment,
                        }
                    )
            elif kind == "tool_use_stop":
                item = calls.get(e.id)
                if item is None:
                    continue
                if not item["arguments"]:
                    item["arguments"] = json.dumps(e.input or {})
                    yield ev(
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": item["id"],
                            "output_index": output.index(item),
                            "delta": item["arguments"],
                        }
                    )
                item["status"] = "completed"
                idx = output.index(item)
                yield ev(
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": item["id"],
                        "output_index": idx,
                        "arguments": item["arguments"],
                    }
                )
                yield ev({"type": "response.output_item.done", "output_index": idx, "item": item})
                outcome.tool_calls.append({"id": e.id, "name": e.name, "arguments": _tool_input(item["arguments"])})
            elif kind == "message_stop":
                outcome.stop_reason = e.stop_reason
                outcome.meta = dict(e.usage or {})
    except Exception as exc:
        outcome.error = exc
        if on_complete:
            on_complete(outcome)
        failed = shell("failed")
        failed["error"] = {"code": "server_error", "message": str(exc)}
        yield ev({"type": "response.failed", "response": failed})
        return

    yield from close_text()
    final = shell("completed")
    yield ev(
        {"type": "response.incomplete" if final["status"] == "incomplete" else "response.completed", "response": final}
    )
    if on_complete:
        on_complete(outcome)
