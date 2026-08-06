"""Shared streaming-tool helpers for OpenAI-compatible drivers.

Any provider that speaks the OpenAI ``chat.completions`` wire format
(Groq, Grok, DeepSeek, Moonshot, Mistral, OpenRouter, Azure, ModelScope,
ZAI, MiniMax, LMStudio, Cachibot, and the generic
``OpenAICompatibleDriver``) can opt into native interleaved streaming-tool
support with a 3-line override::

    # sync driver
    class GroqDriver(...):
        supports_streaming_tool_use = True

        def generate_messages_with_tools_stream(self, messages, tools, options):
            from ._openai_compat_stream import stream_openai_compat_tool_call
            yield from stream_openai_compat_tool_call(
                self, messages, tools, options, provider="groq"
            )

    # async driver
    class AsyncGroqDriver(...):
        supports_streaming_tool_use = True

        async def generate_messages_with_tools_stream(self, messages, tools, options):
            from ._openai_compat_stream import astream_openai_compat_tool_call
            async for ev in astream_openai_compat_tool_call(
                self, messages, tools, options, provider="groq"
            ):
                yield ev

The high-level helpers expect the driver to have:

- ``self.client`` — an OpenAI (or AsyncOpenAI) SDK client
- ``self.model`` — default model string
- ``self._get_model_config(provider, model)`` — for ``tokens_param`` and
  ``supports_temperature`` (provided by ``CostMixin``)
- ``self._calculate_cost(provider, model, prompt_tokens, completion_tokens, cached_tokens=...)``
  — provided by ``CostMixin``

If a driver builds request kwargs differently (vision blocks, custom
headers, etc.), use the lower-level
:func:`iter_openai_compat_live_events` / :func:`aiter_openai_compat_live_events`
directly and pass your own stream object.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)

CostFn = Callable[[int, int, int], float]
"""``(prompt_tokens, completion_tokens, cached_prompt_tokens) -> cost_usd``"""

AsyncCostFn = Callable[[int, int, int], Awaitable[float] | float]
"""Sync OR async variant of :data:`CostFn`. The async helper awaits when needed."""


def _extract_cached_tokens(usage: Any) -> int:
    """Pull cached prompt-token count from an OpenAI-shaped usage object."""
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    try:
        return int(getattr(details, "cached_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _process_chunk(
    chunk: Any,
    state: dict[str, Any],
) -> Iterator[Any]:
    """Process one OpenAI-compat stream chunk into zero or more LiveEvents.

    ``state`` is a mutable bag the caller threads across chunks; this keeps
    the sync and async helpers aligned without duplicating event logic.
    Mutates: ``state["prompt_tokens"]``, ``state["completion_tokens"]``,
    ``state["cached_prompt_tokens"]``, ``state["finish_reason"]``,
    ``state["tool_calls"]`` (dict indexed by tool-call slot).
    """
    from ..agents.live_events import TextDelta, ThinkingDelta, ToolInputDelta, ToolUseStart

    if getattr(chunk, "usage", None):
        state["prompt_tokens"] = chunk.usage.prompt_tokens or 0
        state["completion_tokens"] = chunk.usage.completion_tokens or 0
        state["cached_prompt_tokens"] = _extract_cached_tokens(chunk.usage)

    if not getattr(chunk, "choices", None):
        return

    choice = chunk.choices[0]
    delta = getattr(choice, "delta", None)
    if delta is None:
        fr = getattr(choice, "finish_reason", None)
        if fr:
            state["finish_reason"] = fr
        return

    # Reasoning/thinking deltas (Grok, DeepSeek, Moonshot reasoning models)
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning:
        yield ThinkingDelta(text=reasoning)

    content = getattr(delta, "content", None) or ""
    if content:
        yield TextDelta(text=content)

    delta_tool_calls = getattr(delta, "tool_calls", None) or []
    for tc in delta_tool_calls:
        idx = getattr(tc, "index", 0) or 0
        bucket: dict[str, Any] = state["tool_calls"].setdefault(
            idx,
            {"id": "", "name": "", "args_fragments": [], "start_emitted": False},
        )
        tc_id = getattr(tc, "id", None)
        if tc_id and not bucket["id"]:
            bucket["id"] = tc_id
        fn = getattr(tc, "function", None)
        if fn is not None:
            fn_name = getattr(fn, "name", None)
            if fn_name and not bucket["name"]:
                bucket["name"] = fn_name
            fn_args = getattr(fn, "arguments", None)
            if fn_args:
                bucket["args_fragments"].append(fn_args)
        if not bucket["start_emitted"] and bucket["id"] and bucket["name"]:
            bucket["start_emitted"] = True
            yield ToolUseStart(id=bucket["id"], name=bucket["name"])
        if bucket["start_emitted"] and fn is not None:
            fn_args_now = getattr(fn, "arguments", None)
            if fn_args_now:
                yield ToolInputDelta(id=bucket["id"], fragment=fn_args_now)

    fr = getattr(choice, "finish_reason", None)
    if fr:
        state["finish_reason"] = fr


def _finalize_tool_use_stops(state: dict[str, Any]) -> Iterator[Any]:
    """After the stream ends, emit ``ToolUseStop`` per accumulated tool.

    Guarantees a ``ToolUseStart`` always precedes its ``ToolUseStop``
    (synthesizing one here when the provider's chunks never completed the
    start conditions) and generates a ``call_<uuid4>`` fallback id when the
    provider omitted tool-call ids.  When the finish reason was
    ``length``/``max_tokens`` and the assembled arguments fail to parse,
    the stop is flagged ``truncated=True`` (contract C3).
    """
    import uuid

    from ..agents.live_events import ToolUseStart, ToolUseStop

    finish_reason = state.get("finish_reason")
    for idx in sorted(state["tool_calls"].keys()):
        bucket = state["tool_calls"][idx]
        if not bucket["id"]:
            bucket["id"] = f"call_{uuid.uuid4().hex[:24]}"
        if not bucket["start_emitted"]:
            yield ToolUseStart(id=bucket["id"], name=bucket["name"])
            bucket["start_emitted"] = True
        args_str = "".join(bucket["args_fragments"])
        parse_failed = False
        try:
            parsed = json.loads(args_str) if args_str else {}
            if not isinstance(parsed, dict):
                parse_failed = True
                logger.warning(
                    "Streamed tool input for %s parsed to non-object JSON: %r",
                    bucket["name"],
                    args_str[:200],
                )
                parsed = {}
        except json.JSONDecodeError:
            parse_failed = True
            logger.warning(
                "Failed to parse streamed tool input for %s: %r",
                bucket["name"],
                args_str[:200],
            )
            parsed = {}
        truncated = parse_failed and finish_reason in ("length", "max_tokens")
        yield ToolUseStop(
            id=bucket["id"],
            name=bucket["name"],
            input=parsed,
            truncated=truncated,
            raw_stop_reason=finish_reason if parse_failed else None,
        )


def _build_meta(state: dict[str, Any], model: str, cost: float) -> dict[str, Any]:
    return {
        "prompt_tokens": state["prompt_tokens"],
        "completion_tokens": state["completion_tokens"],
        "total_tokens": state["prompt_tokens"] + state["completion_tokens"],
        "cached_prompt_tokens": state["cached_prompt_tokens"],
        "cost": round(cost, 6),
        "model_name": model,
    }


def _build_message_stop(state: dict[str, Any], model: str, cost: float) -> Any:
    """Build the terminal ``MessageStop`` with a normalized ``stop_reason``
    (shared vocabulary, contract M3); the provider's raw finish reason is
    preserved in ``usage["raw_stop_reason"]``."""
    from ..agents.live_events import MessageStop
    from .base import _normalize_stop_reason

    meta = _build_meta(state, model, cost)
    raw_finish_reason = state.get("finish_reason")
    meta["raw_stop_reason"] = raw_finish_reason
    return MessageStop(
        stop_reason=_normalize_stop_reason(raw_finish_reason, tool_calls_present=bool(state["tool_calls"])),
        usage=meta,
    )


def _fresh_state() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_prompt_tokens": 0,
        "finish_reason": "stop",
        "tool_calls": {},
    }


def iter_openai_compat_live_events(
    stream: Iterator[Any],
    *,
    model: str,
    cost_fn: CostFn,
) -> Iterator[Any]:
    """Consume a sync OpenAI-compat ``chat.completions`` stream with tools
    and yield :class:`~prompture.agents.live_events.LiveEvent`.

    Yields ``TextDelta`` for content deltas, ``ToolUseStart`` /
    ``ToolInputDelta`` / ``ToolUseStop`` for tool calls (input JSON is
    assembled from per-chunk fragments), and a final ``MessageStop`` with
    usage and cost via *cost_fn*.

    The caller is responsible for creating the stream
    (``client.chat.completions.create(stream=True, stream_options={"include_usage": True}, ...)``)
    so the helper stays driver-agnostic. Set
    ``stream_options={"include_usage": True}`` to populate usage in the
    final chunk; without it ``cost_fn`` will receive zeros.
    """
    state = _fresh_state()
    for chunk in stream:
        yield from _process_chunk(chunk, state)
    yield from _finalize_tool_use_stops(state)
    cost = cost_fn(
        state["prompt_tokens"],
        state["completion_tokens"],
        state["cached_prompt_tokens"],
    )
    yield _build_message_stop(state, model, cost)


async def aiter_openai_compat_live_events(
    stream: AsyncIterator[Any],
    *,
    model: str,
    cost_fn: AsyncCostFn,
) -> AsyncIterator[Any]:
    """Async sibling of :func:`iter_openai_compat_live_events`.

    *cost_fn* may be sync or async; the helper awaits when needed.
    """
    import inspect

    state = _fresh_state()
    async for chunk in stream:
        for ev in _process_chunk(chunk, state):
            yield ev
    for ev in _finalize_tool_use_stops(state):
        yield ev
    raw_cost = cost_fn(
        state["prompt_tokens"],
        state["completion_tokens"],
        state["cached_prompt_tokens"],
    )
    cost = await raw_cost if inspect.isawaitable(raw_cost) else raw_cost
    yield _build_message_stop(state, model, cost)


# ----------------------------------------------------------------------
# High-level one-call helpers for OpenAI-compat drivers
# ----------------------------------------------------------------------


def stream_openai_compat_tool_call(
    driver: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    options: dict[str, Any],
    *,
    provider: str,
    default_max_tokens: int = 4096,
    default_temperature: float = 1.0,
) -> Iterator[Any]:
    """Generic sync streaming-tool implementation for any OpenAI-compat driver.

    Looks up the driver's model config, builds the chat-completions request
    with ``stream=True``, opens the stream, and yields ``LiveEvent`` via
    :func:`iter_openai_compat_live_events`.

    The driver must satisfy the contract described in this module's
    docstring. *provider* is the pricing-table key (e.g. ``"groq"``).
    """
    from .base import _apply_openai_tool_options
    from .openai_driver import _build_openai_base_kwargs

    model = options.get("model", driver.model)
    model_config = driver._get_model_config(provider, model)
    tokens_param = model_config["tokens_param"]
    supports_temperature = model_config["supports_temperature"]

    opts = {"temperature": default_temperature, "max_tokens": default_max_tokens, **options}

    # Only first-party OpenAI gets prompt_cache_key — third-party
    # OpenAI-compatible endpoints reject unknown fields (see
    # _build_openai_base_kwargs). Mirrors the buffered path.
    prompt_cache_key = None
    if provider == "openai":
        from .openai_driver import _openai_prompt_cache_key

        prompt_cache_key = _openai_prompt_cache_key(messages, opts, tools)

    kwargs = _build_openai_base_kwargs(
        model,
        messages,
        opts,
        tokens_param,
        supports_temperature,
        default_max_tokens,
        extra={
            "tools": tools,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        prompt_cache_key=prompt_cache_key,
    )
    _apply_openai_tool_options(kwargs, options)

    stream = driver.client.chat.completions.create(**kwargs)

    def cost_fn(prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> float:
        return driver._calculate_cost(provider, model, prompt_tokens, completion_tokens, cached_tokens=cached_tokens)

    yield from iter_openai_compat_live_events(stream, model=model, cost_fn=cost_fn)


async def astream_openai_compat_tool_call(
    driver: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    options: dict[str, Any],
    *,
    provider: str,
    default_max_tokens: int = 4096,
    default_temperature: float = 1.0,
) -> AsyncIterator[Any]:
    """Async sibling of :func:`stream_openai_compat_tool_call`."""
    from .base import _apply_openai_tool_options
    from .openai_driver import _build_openai_base_kwargs

    model = options.get("model", driver.model)
    model_config = driver._get_model_config(provider, model)
    tokens_param = model_config["tokens_param"]
    supports_temperature = model_config["supports_temperature"]

    opts = {"temperature": default_temperature, "max_tokens": default_max_tokens, **options}

    prompt_cache_key = None
    if provider == "openai":
        from .openai_driver import _openai_prompt_cache_key

        prompt_cache_key = _openai_prompt_cache_key(messages, opts, tools)

    kwargs = _build_openai_base_kwargs(
        model,
        messages,
        opts,
        tokens_param,
        supports_temperature,
        default_max_tokens,
        extra={
            "tools": tools,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        prompt_cache_key=prompt_cache_key,
    )
    _apply_openai_tool_options(kwargs, options)

    stream = await driver.client.chat.completions.create(**kwargs)

    def cost_fn(prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> float:
        return driver._calculate_cost(provider, model, prompt_tokens, completion_tokens, cached_tokens=cached_tokens)

    async for ev in aiter_openai_compat_live_events(stream, model=model, cost_fn=cost_fn):
        yield ev


# ----------------------------------------------------------------------
# Raw-HTTP variants for drivers that POST directly instead of using
# the OpenAI SDK (grok, openrouter, moonshot, mistral, modelscope, zai,
# cachibot, ...). Same event output, parses SSE lines from a streaming
# requests / httpx response.
# ----------------------------------------------------------------------


def _dict_to_chunk(d: Any) -> Any:
    """Convert a JSON-decoded dict (and nested dicts/lists) to a
    namespace-like object so ``_process_chunk`` (which uses attribute
    access throughout) works on raw HTTP payloads without changes."""
    from types import SimpleNamespace

    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_chunk(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_chunk(item) for item in d]
    return d


def _build_raw_http_payload(
    driver: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    options: dict[str, Any],
    *,
    provider: str,
    default_max_tokens: int,
    default_temperature: float,
) -> tuple[str, dict[str, Any]]:
    """Build (model, payload) for a raw-HTTP OpenAI-compat call.

    Pulls ``tokens_param`` / ``supports_temperature`` from the driver's
    ``_get_model_config(provider, model)``. Adds ``stream`` and
    ``stream_options.include_usage`` so the final chunk carries usage.
    """
    model = options.get("model", driver.model)
    model_config = driver._get_model_config(provider, model)
    tokens_param = model_config["tokens_param"]
    supports_temperature = model_config["supports_temperature"]

    opts = {"temperature": default_temperature, "max_tokens": default_max_tokens, **options}

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    payload[tokens_param] = opts.get("max_tokens", default_max_tokens)
    if supports_temperature and "temperature" in opts:
        payload["temperature"] = opts["temperature"]

    from .base import _apply_openai_tool_options

    _apply_openai_tool_options(payload, options)

    return model, payload


def _bearer_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def stream_raw_http_compat_tool_call(
    driver: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    options: dict[str, Any],
    *,
    provider: str,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    default_max_tokens: int = 4096,
    default_temperature: float = 1.0,
    timeout: int = 120,
) -> Iterator[Any]:
    """Generic sync streaming-tool implementation for raw-HTTP OpenAI-compat
    drivers (those that call ``requests.post(/chat/completions)`` instead
    of using the OpenAI SDK).

    Defaults assume the driver exposes:
    - ``self.api_base`` — base URL (no trailing ``/chat/completions``)
    - ``self.api_key`` — used as ``Authorization: Bearer <key>``

    Override *url* or *headers* if the driver uses a different shape.
    """
    import requests

    model, payload = _build_raw_http_payload(
        driver,
        messages,
        tools,
        options,
        provider=provider,
        default_max_tokens=default_max_tokens,
        default_temperature=default_temperature,
    )
    resolved_url = url or f"{driver.api_base}/chat/completions"
    resolved_headers = headers or _bearer_headers(driver.api_key)

    state = _fresh_state()
    with requests.post(
        resolved_url,
        headers=resolved_headers,
        json=payload,
        stream=True,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk_dict = json.loads(data)
            except json.JSONDecodeError:
                continue
            chunk = _dict_to_chunk(chunk_dict)
            yield from _process_chunk(chunk, state)

    yield from _finalize_tool_use_stops(state)
    cost = driver._calculate_cost(
        provider,
        model,
        state["prompt_tokens"],
        state["completion_tokens"],
        cached_tokens=state["cached_prompt_tokens"],
    )
    yield _build_message_stop(state, model, cost)


async def astream_raw_http_compat_tool_call(
    driver: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    options: dict[str, Any],
    *,
    provider: str,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    default_max_tokens: int = 4096,
    default_temperature: float = 1.0,
    timeout: float = 120.0,
) -> AsyncIterator[Any]:
    """Async sibling of :func:`stream_raw_http_compat_tool_call` using
    ``httpx.AsyncClient.stream``."""
    import httpx

    model, payload = _build_raw_http_payload(
        driver,
        messages,
        tools,
        options,
        provider=provider,
        default_max_tokens=default_max_tokens,
        default_temperature=default_temperature,
    )
    resolved_url = url or f"{driver.api_base}/chat/completions"
    resolved_headers = headers or _bearer_headers(driver.api_key)

    state = _fresh_state()
    async with (
        httpx.AsyncClient(timeout=timeout) as client,
        client.stream(
            "POST",
            resolved_url,
            headers=resolved_headers,
            json=payload,
        ) as resp,
    ):
        resp.raise_for_status()
        async for raw_line in resp.aiter_lines():
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk_dict = json.loads(data)
            except json.JSONDecodeError:
                continue
            chunk = _dict_to_chunk(chunk_dict)
            for ev in _process_chunk(chunk, state):
                yield ev

    for ev in _finalize_tool_use_stops(state):
        yield ev
    cost = driver._calculate_cost(
        provider,
        model,
        state["prompt_tokens"],
        state["completion_tokens"],
        cached_tokens=state["cached_prompt_tokens"],
    )
    yield _build_message_stop(state, model, cost)


__all__ = [
    "AsyncCostFn",
    "CostFn",
    "aiter_openai_compat_live_events",
    "astream_openai_compat_tool_call",
    "astream_raw_http_compat_tool_call",
    "iter_openai_compat_live_events",
    "stream_openai_compat_tool_call",
    "stream_raw_http_compat_tool_call",
]
