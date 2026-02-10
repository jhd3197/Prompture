"""Base class for text-to-speech (TTS) drivers."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

from ..infra.callbacks import DriverCallbacks

logger = logging.getLogger("prompture.tts_driver")


class TTSDriver:
    """Adapter base for text-to-speech. Implement ``synthesize(text, options)``.

    Response contract::

        {
            "audio": bytes,        # Raw audio bytes
            "media_type": str,     # MIME type (e.g. "audio/mpeg")
            "meta": {
                "characters": int,
                "cost": float,
                "model_name": str,
                "raw_response": dict,
            },
        }

    Stream contract (each chunk)::

        {"type": "delta", "audio": bytes, "media_type": str}
        {"type": "done", "audio": bytes, "media_type": str, "meta": dict}
    """

    supports_streaming: bool = False
    supports_ssml: bool = False
    available_voices: list[str] = []

    callbacks: DriverCallbacks | None = None

    def synthesize(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        """Synthesize text to audio.

        Args:
            text: Text to convert to speech.
            options: Provider-specific options (voice, format, speed, etc.).

        Returns:
            Dict with ``audio``, ``media_type``, and ``meta`` keys.
        """
        raise NotImplementedError

    def synthesize_stream(self, text: str, options: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield audio chunks incrementally.

        Each chunk is a dict:
        - ``{"type": "delta", "audio": bytes, "media_type": str}``
        - ``{"type": "done", "audio": bytes, "media_type": str, "meta": dict}``
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support streaming")

    # ------------------------------------------------------------------
    # Hook-aware wrapper
    # ------------------------------------------------------------------

    def synthesize_with_hooks(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        """Wrap :meth:`synthesize` with on_request / on_response / on_error callbacks."""
        driver_name = getattr(self, "model", self.__class__.__name__)
        self._fire_callback(
            "on_request",
            {"text_length": len(text), "options": options, "driver": driver_name},
        )
        t0 = time.perf_counter()
        try:
            resp = self.synthesize(text, options)
        except Exception as exc:
            self._fire_callback(
                "on_error",
                {"error": exc, "text_length": len(text), "options": options, "driver": driver_name},
            )
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        meta = resp.get("meta", {})
        logger.debug(
            "[tts] synthesize driver=%s chars=%d cost=%.6f elapsed=%.0fms",
            driver_name,
            meta.get("characters", 0),
            meta.get("cost", 0.0),
            elapsed_ms,
        )
        self._fire_callback(
            "on_response",
            {
                "audio_size": len(resp.get("audio", b"")),
                "meta": meta,
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
