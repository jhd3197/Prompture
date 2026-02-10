"""Base class for speech-to-text (STT) drivers."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..infra.callbacks import DriverCallbacks

logger = logging.getLogger("prompture.stt_driver")


class STTDriver:
    """Adapter base for speech-to-text. Implement ``transcribe(audio, options)``.

    Response contract::

        {
            "text": str,           # Transcribed text
            "segments": list,      # Timed segments (if supported)
            "language": str|None,  # Detected language code
            "meta": {
                "duration_seconds": float,
                "cost": float,
                "model_name": str,
                "raw_response": dict,
            },
        }
    """

    supports_timestamps: bool = False
    supports_language_detection: bool = False

    callbacks: DriverCallbacks | None = None

    def transcribe(self, audio: bytes, options: dict[str, Any]) -> dict[str, Any]:
        """Transcribe audio bytes to text.

        Args:
            audio: Raw audio bytes (MP3, WAV, etc.).
            options: Provider-specific options (language, response_format, etc.).

        Returns:
            Dict with ``text``, ``segments``, ``language``, and ``meta`` keys.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Hook-aware wrapper
    # ------------------------------------------------------------------

    def transcribe_with_hooks(self, audio: bytes, options: dict[str, Any]) -> dict[str, Any]:
        """Wrap :meth:`transcribe` with on_request / on_response / on_error callbacks."""
        driver_name = getattr(self, "model", self.__class__.__name__)
        self._fire_callback(
            "on_request",
            {"audio_size": len(audio), "options": options, "driver": driver_name},
        )
        t0 = time.perf_counter()
        try:
            resp = self.transcribe(audio, options)
        except Exception as exc:
            self._fire_callback(
                "on_error",
                {"error": exc, "audio_size": len(audio), "options": options, "driver": driver_name},
            )
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        meta = resp.get("meta", {})
        logger.debug(
            "[stt] transcribe driver=%s duration=%.1fs cost=%.6f elapsed=%.0fms",
            driver_name,
            meta.get("duration_seconds", 0),
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
