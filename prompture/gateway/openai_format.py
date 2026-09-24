"""OpenAI wire format ↔ Prompture drivers, with no web-framework dependency.

Everything an OpenAI-compatible HTTP surface needs lives here so every
server built on Prompture (``prompture serve``, a gateway, your own app)
speaks the same dialect:

* request side — :func:`to_driver_messages`, :func:`driver_options`
* one-shot calls — :func:`run_chat` / :func:`arun_chat` pick the right driver
  method (tools, messages, or flat prompt) and return a :class:`ChatOutcome`
* response side — :func:`chat_completion`, :func:`chat_chunk`,
  :func:`tool_calls_to_openai`, :func:`finish_reason`, :func:`usage_from_meta`
* streaming — :func:`stream_chat_chunks` / :func:`astream_chat_chunks` turn a
  driver stream into OpenAI chunk dicts; :func:`sse` frames them
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Request side
# ---------------------------------------------------------------------------

#: Chat-completion request fields forwarded verbatim as driver options.
#: ``response_format`` is translated separately (see :func:`driver_options`);
#: ``tools`` / ``tool_choice`` travel through :func:`run_chat`.
OPTION_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "reasoning_effort",
)


def flatten_content(content: Any) -> str:
    """Reduce OpenAI content (string or multipart list) to its text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("text", "input_text") and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def extract_images(content: Any) -> list[str]:
    """URLs / data URIs of ``image_url`` parts in an OpenAI multipart message."""
    if not isinstance(content, list):
        return []
    images: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        url_field = part.get("image_url")
        url = url_field.get("url") if isinstance(url_field, dict) else url_field
        if isinstance(url, str) and url:
            images.append(url)
    return images


def _as_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    return dict(vars(message))


def to_driver_messages(messages: Iterable[Any], *, keep_images: bool = True) -> list[dict[str, Any]]:
    """Convert OpenAI chat messages (dicts or pydantic models) to Prompture's universal format.

    Text parts are joined; ``image_url`` parts become universal
    ``{"type": "image", "source": ImageContent}`` blocks (drivers translate
    those to their own vision format). ``tool_calls``, ``tool_call_id`` and
    ``name`` pass through — drivers already accept OpenAI-shaped tool turns.
    """
    out: list[dict[str, Any]] = []
    for raw in messages:
        msg = _as_dict(raw)
        content = msg.get("content")
        entry: dict[str, Any] = {"role": msg.get("role", "user")}
        images = extract_images(content) if keep_images else []
        if images:
            from ..media.image import make_image

            blocks: list[dict[str, Any]] = []
            text = flatten_content(content)
            if text:
                blocks.append({"type": "text", "text": text})
            blocks.extend({"type": "image", "source": make_image(url)} for url in images)
            entry["content"] = blocks
        else:
            entry["content"] = flatten_content(content)
        for key in ("tool_calls", "tool_call_id", "name"):
            if msg.get(key):
                entry[key] = msg[key]
        out.append(entry)
    return out


def driver_options(request: Any, fields: Iterable[str] = OPTION_FIELDS) -> dict[str, Any]:
    """Driver options from an OpenAI chat request (dict or pydantic model).

    Copies non-null sampling fields, maps ``max_completion_tokens`` onto
    ``max_tokens``, and translates ``response_format`` into Prompture's
    ``json_mode`` / ``json_schema`` options.
    """
    data = _as_dict(request)
    opts = {f: data[f] for f in fields if data.get(f) is not None}
    if "max_tokens" not in opts and data.get("max_completion_tokens") is not None:
        opts["max_tokens"] = data["max_completion_tokens"]
    fmt = data.get("response_format")
    if isinstance(fmt, dict):
        kind = fmt.get("type")
        if kind == "json_object":
            opts["json_mode"] = True
        elif kind == "json_schema":
            opts["json_mode"] = True
            spec = fmt.get("json_schema") or {}
            schema = spec.get("schema") if isinstance(spec, dict) else None
            if schema:
                opts["json_schema"] = schema
    return opts


# ---------------------------------------------------------------------------
# Response side
# ---------------------------------------------------------------------------

_FINISH_REASONS = {
    "end_turn": "stop",
    "stop": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "length": "length",
    "tool_use": "tool_calls",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
    "error": "stop",
}


def finish_reason(stop_reason: str | None, *, has_tool_calls: bool = False) -> str:
    """Map Prompture's canonical stop reason to OpenAI's ``finish_reason``."""
    if has_tool_calls:
        return "tool_calls"
    return _FINISH_REASONS.get((stop_reason or "end_turn").lower(), "stop")


def usage_from_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    """OpenAI ``usage`` block from a driver ``meta`` dict."""
    meta = meta or {}
    prompt = int(meta.get("prompt_tokens", 0) or 0)
    completion = int(meta.get("completion_tokens", 0) or 0)
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(meta.get("total_tokens", 0) or 0) or prompt + completion,
    }
    cached = meta.get("cached_prompt_tokens")
    if cached:
        usage["prompt_tokens_details"] = {"cached_tokens": int(cached)}
    return usage


def tool_calls_to_openai(tool_calls: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Prompture ``{"id","name","arguments": dict}`` → OpenAI function tool calls."""
    out: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        if "function" in tc:  # already OpenAI-shaped
            out.append(tc)
            continue
        args = tc.get("arguments", {})
        out.append(
            {
                "id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": args if isinstance(args, str) else json.dumps(args or {}),
                },
            }
        )
    return out


def new_completion_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def chat_completion(
    *,
    model: str,
    text: str,
    meta: dict[str, Any] | None = None,
    tool_calls: Iterable[dict[str, Any]] | None = None,
    stop_reason: str | None = None,
    completion_id: str | None = None,
    created: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A complete ``chat.completion`` object."""
    calls = tool_calls_to_openai(tool_calls)
    message: dict[str, Any] = {"role": "assistant", "content": text if (text or not calls) else None}
    if calls:
        message["tool_calls"] = calls
    body: dict[str, Any] = {
        "id": completion_id or new_completion_id(),
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason(stop_reason, has_tool_calls=bool(calls))}
        ],
        "usage": usage_from_meta(meta),
    }
    if extra:
        body.update(extra)
    return body


def chat_chunk(
    *,
    completion_id: str,
    model: str,
    delta: dict[str, Any],
    finish: str | None = None,
    created: int | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One ``chat.completion.chunk`` object."""
    chunk: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def error_body(message: str, *, type_: str = "driver_error", code: str | None = None) -> dict[str, Any]:
    """OpenAI-style error envelope (also used as an in-stream error event)."""
    return {"error": {"message": message, "type": type_, "code": code}}


def sse(payload: dict[str, Any] | str) -> str:
    """Frame one Server-Sent Event line. Strings (``"[DONE]"``) pass through."""
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, separators=(',', ':'), default=str)}\n\n"


SSE_DONE = sse("[DONE]")

#: Headers that keep proxies from buffering an SSE response.
SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def models_list(model_ids: Iterable[str], *, owned_by: str = "prompture") -> dict[str, Any]:
    """``GET /v1/models`` body."""
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": owned_by} for m in model_ids]}


# ---------------------------------------------------------------------------
# Calling drivers
# ---------------------------------------------------------------------------


@dataclass
class ChatOutcome:
    """Normalized result of one chat call (streamed or not)."""

    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    error: BaseException | None = None

    @property
    def usage(self) -> dict[str, Any]:
        return usage_from_meta(self.meta)

    @property
    def cost(self) -> float:
        return float(self.meta.get("cost", 0.0) or 0.0)

    def to_completion(self, model: str, **kwargs: Any) -> dict[str, Any]:
        return chat_completion(
            model=model,
            text=self.text,
            meta=self.meta,
            tool_calls=self.tool_calls,
            stop_reason=self.stop_reason,
            **kwargs,
        )


def _outcome(resp: dict[str, Any]) -> ChatOutcome:
    return ChatOutcome(
        text=resp.get("text", "") or "",
        meta=resp.get("meta") or resp.get("usage") or {},
        tool_calls=list(resp.get("tool_calls") or []),
        stop_reason=resp.get("stop_reason"),
    )


def run_chat(
    driver: Any,
    messages: list[dict[str, Any]],
    options: dict[str, Any] | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> ChatOutcome:
    """One non-streaming chat turn through the best method the driver offers."""
    opts = dict(options or {})
    if tools:
        if not getattr(driver, "supports_tool_use", False):
            raise NotImplementedError(f"{type(driver).__name__} does not support tool use")
        return _outcome(driver.generate_messages_with_tools(messages, tools, opts))
    return _outcome(driver.generate_messages(messages, opts))


async def arun_chat(
    driver: Any,
    messages: list[dict[str, Any]],
    options: dict[str, Any] | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> ChatOutcome:
    """Async :func:`run_chat` for :class:`~prompture.drivers.async_base.AsyncDriver`."""
    opts = dict(options or {})
    if tools:
        if not getattr(driver, "supports_tool_use", False):
            raise NotImplementedError(f"{type(driver).__name__} does not support tool use")
        return _outcome(await driver.generate_messages_with_tools(messages, tools, opts))
    return _outcome(await driver.generate_messages(messages, opts))


def _apply_stream_event(event: dict[str, Any], outcome: ChatOutcome) -> str | None:
    """Fold a driver stream event into *outcome*; return delta text to emit."""
    kind = event.get("type")
    if kind == "delta":
        text = event.get("text", "") or ""
        outcome.text += text
        return text or None
    if kind == "done":
        outcome.text = event.get("text", outcome.text) or outcome.text
        outcome.meta = event.get("meta") or {}
        outcome.stop_reason = event.get("stop_reason") or outcome.meta.get("stop_reason")
    return None


def stream_chat_chunks(
    events: Iterable[dict[str, Any]],
    *,
    model: str,
    completion_id: str | None = None,
    include_usage: bool = True,
    on_complete: Callable[[ChatOutcome], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Turn a driver's ``generate_messages_stream`` events into OpenAI chunk dicts.

    Emits the role chunk first, content deltas, then a final chunk with
    ``finish_reason`` (and ``usage`` when *include_usage*). If the driver
    raises mid-stream, yields an :func:`error_body` dict instead of a final
    chunk. *on_complete* receives the accumulated :class:`ChatOutcome`
    (with ``error`` set on failure) once the stream ends — use it for metering.
    """
    cid = completion_id or new_completion_id()
    created = int(time.time())
    outcome = ChatOutcome()
    yield chat_chunk(completion_id=cid, model=model, delta={"role": "assistant"}, created=created)
    try:
        for event in events:
            text = _apply_stream_event(event, outcome)
            if text:
                yield chat_chunk(completion_id=cid, model=model, delta={"content": text}, created=created)
    except Exception as exc:
        outcome.error = exc
        if on_complete:
            on_complete(outcome)
        yield error_body(str(exc))
        return
    yield chat_chunk(
        completion_id=cid,
        model=model,
        delta={},
        finish=finish_reason(outcome.stop_reason),
        created=created,
        usage=outcome.usage if include_usage else None,
    )
    if on_complete:
        on_complete(outcome)


async def astream_chat_chunks(
    events: AsyncIterator[dict[str, Any]],
    *,
    model: str,
    completion_id: str | None = None,
    include_usage: bool = True,
    on_complete: Callable[[ChatOutcome], Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Async :func:`stream_chat_chunks`. *on_complete* may be sync or async."""
    import inspect

    async def _done(outcome: ChatOutcome) -> None:
        if on_complete:
            result = on_complete(outcome)
            if inspect.isawaitable(result):
                await result

    cid = completion_id or new_completion_id()
    created = int(time.time())
    outcome = ChatOutcome()
    yield chat_chunk(completion_id=cid, model=model, delta={"role": "assistant"}, created=created)
    try:
        async for event in events:
            text = _apply_stream_event(event, outcome)
            if text:
                yield chat_chunk(completion_id=cid, model=model, delta={"content": text}, created=created)
    except Exception as exc:
        outcome.error = exc
        await _done(outcome)
        yield error_body(str(exc))
        return
    yield chat_chunk(
        completion_id=cid,
        model=model,
        delta={},
        finish=finish_reason(outcome.stop_reason),
        created=created,
        usage=outcome.usage if include_usage else None,
    )
    await _done(outcome)
