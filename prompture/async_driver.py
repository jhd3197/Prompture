"""Async driver base class for LLM adapters."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from .callbacks import DriverCallbacks
from .driver import Driver

logger = logging.getLogger("prompture.async_driver")


class AsyncDriver:
    """Async adapter base. Implement ``async generate(prompt, options)``
    returning ``{"text": ..., "meta": {...}}``.

    The ``meta`` dict follows the same contract as :class:`Driver`:

    .. code-block:: python

        {
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
            "cost": float,
            "raw_response": dict,
        }
    """

    supports_json_mode: bool = False
    supports_json_schema: bool = False
    supports_messages: bool = False
    supports_tool_use: bool = False
    supports_streaming: bool = False
    supports_vision: bool = False

    callbacks: DriverCallbacks | None = None

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        """Generate a response from a list of conversation messages (async).

        Default implementation flattens the messages into a single prompt
        and delegates to :meth:`generate`.  Drivers that natively support
        message arrays should override this and set
        ``supports_messages = True``.
        """
        prompt = Driver._flatten_messages(messages)
        return await self.generate(prompt, options)

    # ------------------------------------------------------------------
    # Tool use
    # ------------------------------------------------------------------

    async def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a response that may include tool calls (async).

        Returns a dict with keys: ``text``, ``meta``, ``tool_calls``, ``stop_reason``.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support tool use")

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield response chunks incrementally (async).

        Each chunk is a dict:
        - ``{"type": "delta", "text": str}``
        - ``{"type": "done", "text": str, "meta": dict}``
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support streaming")
        # yield is needed to make this an async generator
        yield  # pragma: no cover

    # ------------------------------------------------------------------
    # Hook-aware wrappers
    # ------------------------------------------------------------------

    async def generate_with_hooks(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Wrap :meth:`generate` with on_request / on_response / on_error callbacks."""
        driver_name = getattr(self, "model", self.__class__.__name__)
        self._fire_callback(
            "on_request",
            {"prompt": prompt, "messages": None, "options": options, "driver": driver_name},
        )
        t0 = time.perf_counter()
        try:
            resp = await self.generate(prompt, options)
        except Exception as exc:
            self._fire_callback(
                "on_error",
                {"error": exc, "prompt": prompt, "messages": None, "options": options, "driver": driver_name},
            )
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._fire_callback(
            "on_response",
            {
                "text": resp.get("text", ""),
                "meta": resp.get("meta", {}),
                "driver": driver_name,
                "elapsed_ms": elapsed_ms,
            },
        )
        return resp

    async def generate_messages_with_hooks(
        self, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> dict[str, Any]:
        """Wrap :meth:`generate_messages` with callbacks."""
        driver_name = getattr(self, "model", self.__class__.__name__)
        self._fire_callback(
            "on_request",
            {"prompt": None, "messages": messages, "options": options, "driver": driver_name},
        )
        t0 = time.perf_counter()
        try:
            resp = await self.generate_messages(messages, options)
        except Exception as exc:
            self._fire_callback(
                "on_error",
                {"error": exc, "prompt": None, "messages": messages, "options": options, "driver": driver_name},
            )
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._fire_callback(
            "on_response",
            {
                "text": resp.get("text", ""),
                "meta": resp.get("meta", {}),
                "driver": driver_name,
                "elapsed_ms": elapsed_ms,
            },
        )
        return resp

    async def generate_messages_with_tools_with_hooks(
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
            resp = await self.generate_messages_with_tools(messages, tools, options)
        except Exception as exc:
            self._fire_callback(
                "on_error",
                {"error": exc, "prompt": None, "messages": messages, "options": options, "driver": driver_name},
            )
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._fire_callback(
            "on_response",
            {
                "text": resp.get("text", ""),
                "meta": resp.get("meta", {}),
                "driver": driver_name,
                "elapsed_ms": elapsed_ms,
            },
        )
        return resp

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
        from .model_rates import get_model_capabilities

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

        Vision-capable async drivers override this to convert the universal
        image blocks into their provider-specific format.
        """
        self._check_vision_support(messages)
        return messages

    # Re-export the static helper for convenience
    _flatten_messages = staticmethod(Driver._flatten_messages)
