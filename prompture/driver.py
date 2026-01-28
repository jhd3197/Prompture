"""Driver base class for LLM adapters."""

from __future__ import annotations

import logging
import time
from typing import Any

from .callbacks import DriverCallbacks

logger = logging.getLogger("prompture.driver")


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

    callbacks: DriverCallbacks | None = None

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        """Generate a response from a list of conversation messages.

        Each message is a dict with ``"role"`` (``"system"``, ``"user"``, or
        ``"assistant"``) and ``"content"`` keys.

        The default implementation flattens the messages into a single prompt
        string and delegates to :meth:`generate`.  Drivers that natively
        support message arrays should override this method and set
        ``supports_messages = True``.
        """
        prompt = self._flatten_messages(messages)
        return self.generate(prompt, options)

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

    def generate_messages_with_hooks(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
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

    @staticmethod
    def _flatten_messages(messages: list[dict[str, str]]) -> str:
        """Join messages into a single prompt string with role prefixes."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System]: {content}")
            elif role == "assistant":
                parts.append(f"[Assistant]: {content}")
            else:
                parts.append(f"[User]: {content}")
        return "\n\n".join(parts)
