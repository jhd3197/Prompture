"""Driver base class for LLM adapters."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

import requests

from ..infra.callbacks import DriverCallbacks

logger = logging.getLogger("prompture.driver")


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

        resp = requests.get(url, headers=hdrs, timeout=timeout)
        if resp.status_code != 200:
            logger.debug("_fetch_openai_compatible_models %s returned %s", url, resp.status_code)
            return None

        data = resp.json()
        models = data.get("data", [])
        return [m["id"] for m in models if m.get("id")]
    except Exception:
        logger.debug("_fetch_openai_compatible_models failed for %s", base_url, exc_info=True)
        return None


class Driver:
    """Adapter base. Implementar generate(prompt, options) -> {"text": ... , "meta": {...}}

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

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

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
                provider = self.__class__.__name__.replace("Driver", "").lower()
                model = driver_name

            model_name = f"{provider}/{model}" if provider else model

            event = UsageEvent(
                model_name=model_name,
                provider=provider,
                api_key_hash=_resolve_api_key_hash(model_name),
                prompt_tokens=meta.get("prompt_tokens", 0),
                completion_tokens=meta.get("completion_tokens", 0),
                total_tokens=meta.get("total_tokens", 0),
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
        if caps.supports_structured_output is False:
            return False
        return True

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
