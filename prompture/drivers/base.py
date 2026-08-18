"""Driver base class for LLM adapters."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

try:
    import requests
except ImportError:
    requests = None

import contextlib

from ..exceptions import DriverError as _DriverError
from ..infra.callbacks import DriverCallbacks

logger = logging.getLogger("prompture.driver")


# ------------------------------------------------------------------
# Shared driver error with HTTP context
# ------------------------------------------------------------------


class DriverHTTPError(_DriverError):
    """A driver request failed with HTTP/provider context attached.

    Subclasses :class:`prompture.exceptions.DriverError` so existing
    ``except DriverError`` callers keep working, while adding structured
    fields for retry logic and observability:

    - ``status_code`` — HTTP status when the provider returned one.
    - ``provider`` — pricing-table/provider key (``"grok"``, ``"ollama"`` …).
    - ``retryable`` — heuristic: timeouts/5xx/429 are worth retrying,
      4xx auth/validation errors are not.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider
        if retryable is None:
            retryable = status_code is None or status_code in (408, 409, 425, 429) or status_code >= 500
        self.retryable = retryable


# ------------------------------------------------------------------
# Shared stop-reason normalization (driver boundary)
# ------------------------------------------------------------------

#: Canonical stop-reason vocabulary shared by all drivers.
STOP_REASONS: frozenset[str] = frozenset({"end_turn", "tool_use", "max_tokens", "content_filter", "error"})

_STOP_REASON_MAP: dict[str, str] = {
    # OpenAI-compatible finish_reason values
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "content_filter",
    # Anthropic (already canonical, listed for completeness)
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "end_turn",
    "refusal": "content_filter",
    # Cohere v2 finish_reason values
    "complete": "end_turn",
    "tool_call": "tool_use",
    "error": "error",
    "error_limit": "error",
    # Ollama done_reason values ("stop"/"length" covered above)
    "load": "end_turn",
    "unload": "end_turn",
    # Google Gemini FinishReason enum values (string form)
    "safety": "content_filter",
    "recitation": "content_filter",
    "blocklist": "content_filter",
    "prohibited_content": "content_filter",
    "image_safety": "content_filter",
    "malformed_function_call": "error",
    "finish_reason_unspecified": "end_turn",
    "other": "end_turn",
}

#: Google Gemini numeric FinishReason enum values.
_GOOGLE_STOP_REASON_MAP: dict[int, str] = {
    1: "end_turn",  # STOP
    2: "max_tokens",  # MAX_TOKENS
    3: "content_filter",  # SAFETY
    4: "content_filter",  # RECITATION
    5: "end_turn",  # OTHER
}


def _normalize_stop_reason(raw: Any, *, tool_calls_present: bool = False) -> str:
    """Normalize a provider stop/finish reason to the shared vocabulary.

    Returns one of ``end_turn``, ``tool_use``, ``max_tokens``,
    ``content_filter``, ``error``.  Unknown strings pass through unchanged
    so no information is lost.  When the response contains tool calls but
    the provider reported a plain end-of-turn (Ollama does this), the
    reason is upgraded to ``tool_use``.
    """
    import enum

    if isinstance(raw, enum.Enum):
        raw = raw.value
    if isinstance(raw, bool):
        normalized = "end_turn"
    elif isinstance(raw, int):
        normalized = _GOOGLE_STOP_REASON_MAP.get(raw, "end_turn")
    elif isinstance(raw, str):
        normalized = _STOP_REASON_MAP.get(raw.lower(), raw)
    else:
        normalized = "end_turn"
    if tool_calls_present and normalized == "end_turn":
        return "tool_use"
    return normalized


# ------------------------------------------------------------------
# Shared tool_choice translation
# ------------------------------------------------------------------


def _translate_tool_choice(tool_choice: Any, api: str) -> Any:
    """Translate a normalized ``tool_choice`` option into wire format.

    Accepted normalized input: ``"auto"``, ``"none"``, ``"required"``, or
    ``{"name": "<tool>"}`` to force a specific tool.  Provider-shaped
    values (already carrying ``type``/``mode`` keys) pass through
    unchanged so advanced users keep an escape hatch.

    *api* is ``"openai"`` (OpenAI-compatible chat completions),
    ``"anthropic"``, or ``"google"`` (returns a ``function_calling_config``
    dict the caller wraps in ``types.ToolConfig``).

    Returns ``None`` when *tool_choice* is ``None`` or unusable (a
    warning is logged in the latter case).
    """
    if tool_choice is None:
        return None

    if api == "anthropic":
        if isinstance(tool_choice, str):
            mapping = {"auto": "auto", "none": "none", "required": "any"}
            t = mapping.get(tool_choice)
            if t is None:
                logger.warning("Unsupported tool_choice %r for anthropic; ignoring", tool_choice)
                return None
            return {"type": t}
        if isinstance(tool_choice, dict):
            if "type" in tool_choice:
                return tool_choice
            if "name" in tool_choice:
                return {"type": "tool", "name": tool_choice["name"]}
        logger.warning("Unsupported tool_choice %r for anthropic; ignoring", tool_choice)
        return None

    if api == "google":
        if isinstance(tool_choice, str):
            mapping = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
            m = mapping.get(tool_choice)
            if m is None:
                logger.warning("Unsupported tool_choice %r for google; ignoring", tool_choice)
                return None
            return {"mode": m}
        if isinstance(tool_choice, dict):
            if "mode" in tool_choice:
                return tool_choice
            if "name" in tool_choice:
                return {"mode": "ANY", "allowed_function_names": [tool_choice["name"]]}
        logger.warning("Unsupported tool_choice %r for google; ignoring", tool_choice)
        return None

    # OpenAI-compatible wire format
    if isinstance(tool_choice, str):
        if tool_choice in ("auto", "none", "required"):
            return tool_choice
        logger.warning("Unsupported tool_choice %r for openai-compatible API; ignoring", tool_choice)
        return None
    if isinstance(tool_choice, dict):
        if "type" in tool_choice:
            return tool_choice
        if "name" in tool_choice:
            return {"type": "function", "function": {"name": tool_choice["name"]}}
    logger.warning("Unsupported tool_choice %r for openai-compatible API; ignoring", tool_choice)
    return None


def _apply_openai_tool_options(kwargs: dict[str, Any], options: dict[str, Any]) -> None:
    """Pass ``tool_choice`` / ``parallel_tool_calls`` through to an
    OpenAI-compatible request payload, translating ``tool_choice`` from
    the normalized form (see :func:`_translate_tool_choice`)."""
    if "tool_choice" in options:
        translated = _translate_tool_choice(options["tool_choice"], "openai")
        if translated is not None:
            kwargs["tool_choice"] = translated
    if "parallel_tool_calls" in options:
        kwargs["parallel_tool_calls"] = options["parallel_tool_calls"]


# ------------------------------------------------------------------
# Shared tool-argument parser for OpenAI-compatible drivers
# ------------------------------------------------------------------


def _parse_tool_arguments_with_error(
    raw_args: Any, tool_name: str, stop_reason: str | None = None
) -> tuple[dict[str, Any], str | None]:
    """Parse tool call arguments, returning ``(arguments, error)``.

    Same resilience contract as :func:`_parse_tool_arguments` (never
    raises, falls back to ``{}``) but also returns a human-readable
    error message when parsing failed or truncation is detected, so
    callers can attach ``tc["arguments_error"]`` and the conversation
    layer can ask the model to retry instead of executing garbage.
    """
    if isinstance(raw_args, dict):
        return raw_args, None
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed, None
            msg = f"Tool arguments for {tool_name} parsed to non-object JSON: {raw_args[:200]!r}"
            logger.warning("Tool arguments for %s parsed to non-object JSON: %r", tool_name, raw_args[:200])
            return {}, msg
        except json.JSONDecodeError:
            if stop_reason in ("length", "max_tokens"):
                msg = (
                    f"Tool arguments for {tool_name} were truncated due to the max_tokens limit. "
                    "Increase max_tokens in options to allow longer tool outputs."
                )
                logger.warning(
                    "Tool arguments for %s were truncated due to max_tokens limit. "
                    "Increase max_tokens in options to allow longer tool outputs. "
                    "Truncated arguments: %r",
                    tool_name,
                    raw_args[:200] if raw_args else raw_args,
                )
            else:
                msg = f"Failed to parse tool arguments for {tool_name} as JSON: {raw_args[:200]!r}"
                logger.warning(
                    "Failed to parse tool arguments for %s: %r",
                    tool_name,
                    raw_args[:200] if raw_args else raw_args,
                )
            return {}, msg
    if raw_args is None:
        return {}, None
    msg = f"Unexpected argument type {type(raw_args).__name__} for tool {tool_name}"
    logger.warning(
        "Unexpected argument type %s for tool %s: %r",
        type(raw_args).__name__,
        tool_name,
        raw_args,
    )
    return {}, msg


def _parse_tool_arguments(raw_args: Any, tool_name: str, stop_reason: str | None = None) -> dict[str, Any]:
    """Parse tool call arguments, handling both string and dict formats.

    Some providers return ``arguments`` as a JSON string, others as an
    already-parsed dict.  Calling ``json.loads()`` on a dict raises
    ``TypeError`` which previously caused a silent fallback to ``{}``.
    """
    args, _error = _parse_tool_arguments_with_error(raw_args, tool_name, stop_reason)
    return args


def _tool_call_dict(
    tool_id: Any,
    name: str,
    raw_args: Any,
    stop_reason: str | None = None,
    *,
    generate_id: bool = True,
) -> dict[str, Any]:
    """Build a normalized tool-call dict for ``generate_messages_with_tools``.

    Generates a ``call_<uuid4>`` fallback id when the provider omits one
    (OpenAI-compat streaming and some raw-HTTP providers do) and attaches
    ``arguments_error`` when argument parsing failed or truncation is
    detected (contract C1).
    """
    import uuid as _uuid

    args, args_error = _parse_tool_arguments_with_error(raw_args, name, stop_reason)
    tc: dict[str, Any] = {
        "id": tool_id or (f"call_{_uuid.uuid4().hex[:24]}" if generate_id else ""),
        "name": name,
        "arguments": args,
    }
    if args_error:
        tc["arguments_error"] = args_error
    return tc


# ------------------------------------------------------------------
# Shared helper for OpenAI-compatible /v1/models endpoints
# ------------------------------------------------------------------


def _fetch_openai_compatible_models(
    base_url: str,
    *,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 10,
) -> list[str] | None:
    """Fetch model IDs from an OpenAI-compatible ``/v1/models`` endpoint.

    Args:
        base_url: The base URL **without** a trailing ``/models`` path.
                  E.g. ``"https://api.openai.com/v1"`` or
                  ``"http://127.0.0.1:1234/v1"``.
        api_key: Optional Bearer token.
        headers: Optional extra headers (merged with auth header).
        timeout: HTTP timeout in seconds.

    Returns:
        A list of model ID strings on success, or ``None`` on any failure.
    """
    try:
        url = f"{base_url.rstrip('/')}/models"
        hdrs: dict[str, str] = {}
        if api_key:
            hdrs["Authorization"] = f"Bearer {api_key}"
        if headers:
            hdrs.update(headers)

        if requests is None:
            import httpx

            resp = httpx.get(url, headers=hdrs, timeout=timeout)
            if resp.status_code != 200:
                detail = ""
                with contextlib.suppress(Exception):
                    detail = f" — {resp.json().get('error', {}).get('message', resp.text[:200])}"
                logger.warning("Model discovery: %s returned HTTP %s%s", url, resp.status_code, detail)
                return None
            data = resp.json()
            models = data.get("data", [])
            return [m["id"] for m in models if m.get("id")]

        resp = requests.get(url, headers=hdrs, timeout=timeout)
        if resp.status_code != 200:
            detail = ""
            with contextlib.suppress(Exception):
                detail = f" — {resp.json().get('error', {}).get('message', resp.text[:200])}"
            logger.warning("Model discovery: %s returned HTTP %s%s", url, resp.status_code, detail)
            return None

        data = resp.json()
        models = data.get("data", [])
        return [m["id"] for m in models if m.get("id")]
    except Exception:
        logger.warning("Model discovery: failed to reach %s", base_url, exc_info=True)
        return None


class Driver(ABC):
    """Adapter base. Implement ``generate(prompt, options)`` returning
    ``{"text": ... , "meta": {...}}``.

    The 'meta' object in the response should have a standardized structure:

    {
        "prompt_tokens": int,     # Number of tokens in the prompt
        "completion_tokens": int, # Number of tokens in the completion
        "total_tokens": int,      # Total tokens used (prompt + completion)
        "cost": float,            # Cost in USD (0.0 for free models)
        "raw_response": dict      # Raw response from LLM provider
    }

    All drivers must populate these fields. The 'raw_response' field can contain
    additional provider-specific metadata while the core fields provide
    standardized access to token usage and cost information.
    """

    supports_json_mode: bool = False
    supports_json_schema: bool = False
    supports_messages: bool = False
    supports_tool_use: bool = False
    supports_streaming: bool = False
    supports_streaming_tool_use: bool = False
    supports_vision: bool = False

    callbacks: DriverCallbacks | None = None

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    @classmethod
    def list_models(cls, **kwargs: Any) -> list[str] | None:
        """Return model IDs available from this provider's API.

        Subclasses should override this with a real implementation.
        Returns ``None`` when the provider has no listing API or
        credentials are missing.
        """
        return None

    @abstractmethod
    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]: ...

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        """Generate a response from a list of conversation messages.

        Each message is a dict with ``"role"`` (``"system"``, ``"user"``, or
        ``"assistant"``) and ``"content"`` keys.

        The default implementation flattens the messages into a single prompt
        string and delegates to :meth:`generate`.  Drivers that natively
        support message arrays should override this method and set
        ``supports_messages = True``.
        """
        self._check_vision_support(messages)
        prompt = self._flatten_messages(messages)
        return self.generate(prompt, options)

    # ------------------------------------------------------------------
    # Tool use
    # ------------------------------------------------------------------

    def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a response that may include tool calls.

        Returns a dict with keys: ``text``, ``meta``, ``tool_calls``, ``stop_reason``.
        ``tool_calls`` is a list of ``{"id": str, "name": str, "arguments": dict}``.

        Drivers that support tool use should override this method and set
        ``supports_tool_use = True``.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support tool use")

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        """Yield response chunks incrementally.

        Each chunk is a dict:
        - ``{"type": "delta", "text": str}`` for content fragments
        - ``{"type": "done", "text": str, "meta": dict}`` for the final summary

        Drivers that support streaming should override this method and set
        ``supports_streaming = True``.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support streaming")

    # ------------------------------------------------------------------
    # Live streaming with interleaved tool calls
    # ------------------------------------------------------------------

    def generate_messages_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[Any]:
        """Yield :class:`~prompture.agents.live_events.LiveEvent` for one
        assistant turn that may interleave text and tool calls.

        The driver is responsible only for one turn — from the model deciding
        what to say/call to ``MessageStop``. The conversation layer drives the
        outer loop, executes tools, and re-invokes this method for the next
        turn.

        Default implementation: wraps the buffered
        :meth:`generate_messages_with_tools` into a synthetic event sequence
        so callers always get a uniform interface, even on drivers that don't
        natively stream tool use. Drivers that DO support native interleaved
        streaming should override this and set
        ``supports_streaming_tool_use = True`` — they'll feel like Claude Code
        (narration between tool calls) instead of "buffered then dumped".
        """
        from ..agents.live_events import (
            MessageStop,
            TextDelta,
            ThinkingDelta,
            ToolInputDelta,
            ToolUseStart,
            ToolUseStop,
        )

        resp = self.generate_messages_with_tools(messages, tools, options)
        text = resp.get("text", "") or ""
        reasoning = resp.get("reasoning_content")
        tool_calls = resp.get("tool_calls", []) or []
        stop_reason = resp.get("stop_reason", "end_turn") or "end_turn"
        meta = resp.get("meta", {}) or {}

        if reasoning:
            yield ThinkingDelta(text=reasoning)
        if text:
            yield TextDelta(text=text)

        import json as _json
        import uuid as _uuid

        for tc in tool_calls:
            tc_id = tc.get("id") or f"call_{_uuid.uuid4().hex[:24]}"
            tc_name = tc.get("name", "") or ""
            tc_args = tc.get("arguments", {}) or {}
            yield ToolUseStart(id=tc_id, name=tc_name)
            try:
                fragment = _json.dumps(tc_args)
            except (TypeError, ValueError):
                fragment = "{}"
            if fragment and fragment != "{}":
                yield ToolInputDelta(id=tc_id, fragment=fragment)
            yield ToolUseStop(
                id=tc_id,
                name=tc_name,
                input=tc_args,
                truncated=bool(tc.get("arguments_error")) and stop_reason in ("length", "max_tokens"),
                raw_stop_reason=meta.get("raw_stop_reason") if tc.get("arguments_error") else None,
            )

        yield MessageStop(stop_reason=stop_reason, usage=meta)

    # ------------------------------------------------------------------
    # Hook-aware wrappers
    # ------------------------------------------------------------------

    def generate_with_hooks(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Wrap :meth:`generate` with on_request / on_response / on_error callbacks."""
        driver_name = getattr(self, "model", self.__class__.__name__)
        self._fire_callback(
            "on_request",
            {"prompt": prompt, "messages": None, "options": options, "driver": driver_name},
        )
        t0 = time.perf_counter()
        try:
            resp = self.generate(prompt, options)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._fire_callback(
                "on_error",
                {"error": exc, "prompt": prompt, "messages": None, "options": options, "driver": driver_name},
            )
            self._auto_record_usage({}, elapsed_ms, status="error", error=exc)
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        meta = resp.get("meta", {})
        logger.debug(
            "[driver] generate driver=%s tokens=%d (prompt=%d completion=%d) cost=%.6f elapsed=%.0fms",
            driver_name,
            meta.get("total_tokens", 0),
            meta.get("prompt_tokens", 0),
            meta.get("completion_tokens", 0),
            meta.get("cost", 0.0),
            elapsed_ms,
        )
        self._fire_callback(
            "on_response",
            {
                "text": resp.get("text", ""),
                "meta": meta,
                "driver": driver_name,
                "elapsed_ms": elapsed_ms,
            },
        )
        self._auto_record_usage(resp, elapsed_ms)
        return resp

    def generate_messages_with_hooks(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        """Wrap :meth:`generate_messages` with callbacks."""
        driver_name = getattr(self, "model", self.__class__.__name__)
        self._fire_callback(
            "on_request",
            {"prompt": None, "messages": messages, "options": options, "driver": driver_name},
        )
        t0 = time.perf_counter()
        try:
            resp = self.generate_messages(messages, options)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._fire_callback(
                "on_error",
                {"error": exc, "prompt": None, "messages": messages, "options": options, "driver": driver_name},
            )
            self._auto_record_usage({}, elapsed_ms, status="error", error=exc)
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        meta = resp.get("meta", {})
        logger.debug(
            "[driver] generate_messages driver=%s tokens=%d (prompt=%d completion=%d) cost=%.6f elapsed=%.0fms",
            driver_name,
            meta.get("total_tokens", 0),
            meta.get("prompt_tokens", 0),
            meta.get("completion_tokens", 0),
            meta.get("cost", 0.0),
            elapsed_ms,
        )
        self._fire_callback(
            "on_response",
            {
                "text": resp.get("text", ""),
                "meta": meta,
                "driver": driver_name,
                "elapsed_ms": elapsed_ms,
            },
        )
        self._auto_record_usage(resp, elapsed_ms)
        return resp

    def generate_messages_with_tools_with_hooks(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Wrap :meth:`generate_messages_with_tools` with callbacks."""
        driver_name = getattr(self, "model", self.__class__.__name__)
        self._fire_callback(
            "on_request",
            {"prompt": None, "messages": messages, "options": options, "driver": driver_name},
        )
        t0 = time.perf_counter()
        try:
            resp = self.generate_messages_with_tools(messages, tools, options)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._fire_callback(
                "on_error",
                {"error": exc, "prompt": None, "messages": messages, "options": options, "driver": driver_name},
            )
            self._auto_record_usage({}, elapsed_ms, status="error", error=exc)
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        meta = resp.get("meta", {})
        logger.debug(
            "[driver] generate_messages_with_tools driver=%s tokens=%d "
            "(prompt=%d completion=%d) cost=%.6f elapsed=%.0fms",
            driver_name,
            meta.get("total_tokens", 0),
            meta.get("prompt_tokens", 0),
            meta.get("completion_tokens", 0),
            meta.get("cost", 0.0),
            elapsed_ms,
        )
        self._fire_callback(
            "on_response",
            {
                "text": resp.get("text", ""),
                "meta": meta,
                "driver": driver_name,
                "elapsed_ms": elapsed_ms,
            },
        )
        self._auto_record_usage(resp, elapsed_ms)
        return resp

    # ------------------------------------------------------------------
    # Auto-recording to usage tracker
    # ------------------------------------------------------------------

    def _auto_record_usage(
        self,
        resp: dict[str, Any],
        elapsed_ms: float,
        *,
        status: str = "success",
        error: Exception | None = None,
    ) -> None:
        """Record a usage event to the global tracker.  Fire-and-forget."""
        try:
            from ..infra.ledger import _resolve_api_key_hash
            from ..infra.tracker import UsageEvent, get_tracker

            tracker = get_tracker()
            if not tracker._enabled:
                return

            meta = resp.get("meta", {}) if resp else {}
            driver_name = getattr(self, "model", self.__class__.__name__)

            # Parse provider/model
            if "/" in driver_name:
                provider, model = driver_name.split("/", 1)
            else:
                # Class-name fallback. Strip the Async prefix first: the async
                # twin of a driver is the same provider, and "asyncclaude" vs
                # "claude" would split one provider's spend into two rows.
                cls_name = self.__class__.__name__.removeprefix("Async")
                provider = cls_name.removesuffix("Driver").lower()
                model = driver_name

            model_name = f"{provider}/{model}" if provider else model

            event = UsageEvent(
                model_name=model_name,
                provider=provider,
                api_key_hash=_resolve_api_key_hash(model_name),
                prompt_tokens=meta.get("prompt_tokens", 0),
                completion_tokens=meta.get("completion_tokens", 0),
                total_tokens=meta.get("total_tokens", 0),
                cached_prompt_tokens=meta.get("cached_prompt_tokens", 0),
                cache_creation_tokens=meta.get("cache_creation_tokens", 0),
                cost=meta.get("cost", 0.0),
                elapsed_ms=elapsed_ms,
                status=status,
                error_type=type(error).__name__ if error else None,
            )
            tracker.record(event)
        except Exception:
            pass  # fire-and-forget

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire_callback(self, event: str, payload: dict[str, Any]) -> None:
        """Invoke a single callback, swallowing and logging any exception."""
        if self.callbacks is None:
            return
        cb = getattr(self.callbacks, event, None)
        if cb is None:
            return
        try:
            cb(payload)
        except Exception:
            logger.exception("Callback %s raised an exception", event)

    def _should_use_json_schema(self, provider: str, model: str) -> bool:
        """Check whether *model* supports structured output (``json_schema``).

        Uses models.dev capability metadata.  Returns ``True`` (optimistic)
        when the model is unknown so that we try the richer mode first.
        """
        from ..infra.model_rates import get_model_capabilities

        caps = get_model_capabilities(provider, model)
        if caps is None:
            return True  # unknown model — optimistically try
        return caps.supports_structured_output is not False

    @staticmethod
    def _inject_schema_into_messages(
        messages: list[dict[str, Any]], json_schema: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Append schema instructions to the last user message.

        Used when falling back from ``json_schema`` mode to plain
        ``json_object`` mode so the model still knows the target structure.
        """
        import json as _json

        messages = [dict(m) for m in messages]  # shallow copy
        schema_str = _json.dumps(json_schema, indent=2)
        note = (
            "\n\nReturn a JSON object that validates against this schema:\n"
            f"{schema_str}\n"
            "If a value is unknown use null."
        )
        for msg in reversed(messages):
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    msg["content"] += note
                break
        return messages

    def _validate_model_capabilities(
        self,
        provider: str,
        model: str,
        *,
        using_tool_use: bool = False,
        using_json_schema: bool = False,
        using_vision: bool = False,
    ) -> None:
        """Log warnings when the model may not support a requested feature.

        Uses models.dev metadata as a secondary signal.  Warnings only — the
        API is the final authority and models.dev data may be stale.
        """
        from ..infra.model_rates import get_model_capabilities

        caps = get_model_capabilities(provider, model)
        if caps is None:
            return

        if using_tool_use and caps.supports_tool_use is False:
            logger.warning(
                "Model %s/%s may not support tool use according to models.dev metadata",
                provider,
                model,
            )
        if using_json_schema and caps.supports_structured_output is False:
            logger.warning(
                "Model %s/%s may not support structured output / JSON schema according to models.dev metadata",
                provider,
                model,
            )
        if using_vision and caps.supports_vision is False:
            logger.warning(
                "Model %s/%s may not support vision/image inputs according to models.dev metadata",
                provider,
                model,
            )

    def _check_vision_support(self, messages: list[dict[str, Any]]) -> None:
        """Raise if messages contain image blocks and the driver lacks vision support."""
        if self.supports_vision:
            return
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image":
                        raise NotImplementedError(
                            f"{self.__class__.__name__} does not support vision/image inputs. "
                            "Use a vision-capable model."
                        )

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform universal message format into provider-specific wire format.

        Vision-capable drivers override this to convert the universal image
        blocks into their provider-specific format.  The base implementation
        validates vision support and returns messages unchanged.
        """
        self._check_vision_support(messages)
        return messages

    @staticmethod
    def _flatten_messages(messages: list[dict[str, Any]]) -> str:
        """Join messages into a single prompt string with role prefixes."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Handle content that is a list of blocks (vision messages)
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "image":
                            text_parts.append("[image]")
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = " ".join(text_parts)
            if role == "system":
                parts.append(f"[System]: {content}")
            elif role == "assistant":
                parts.append(f"[Assistant]: {content}")
            else:
                parts.append(f"[User]: {content}")
        return "\n\n".join(parts)
