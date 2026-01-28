"""Async driver base class for LLM adapters."""

from __future__ import annotations

import logging
import time
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

    callbacks: DriverCallbacks | None = None

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        """Generate a response from a list of conversation messages (async).

        Default implementation flattens the messages into a single prompt
        and delegates to :meth:`generate`.  Drivers that natively support
        message arrays should override this and set
        ``supports_messages = True``.
        """
        prompt = Driver._flatten_messages(messages)
        return await self.generate(prompt, options)

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
        self, messages: list[dict[str, str]], options: dict[str, Any]
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

    # Re-export the static helper for convenience
    _flatten_messages = staticmethod(Driver._flatten_messages)
