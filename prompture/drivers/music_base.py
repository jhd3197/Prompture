"""Base class for music generation drivers."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..infra.callbacks import DriverCallbacks

logger = logging.getLogger("prompture.music_driver")


class MusicGenDriver:
    """Adapter base for music generation. Implement ``generate_music``.

    A music model turns a text prompt (and optional style / lyrics / source
    audio) into one or more audio tracks. The operation — create / remix /
    extend / mashup / add-vocals — is selected by the model slug and/or an
    ``operation`` option; conditioned operations take a source ``audio_url``.

    Response contract::

        {
            "audio": list[AudioContent],
            "meta": {
                "audio_count": int,
                "operation": str | None,
                "cost": float,
                "model_name": str,
                "request_id": str | None,
                "raw_response": dict,
            },
        }
    """

    supports_instrumental: bool = True
    supports_vocals: bool = True
    supports_polling: bool = True

    callbacks: DriverCallbacks | None = None

    def generate_music(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate music track(s) from a prompt."""
        raise NotImplementedError

    def generate_music_with_hooks(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Wrap :meth:`generate_music` with on_request / on_response / on_error callbacks."""
        driver_name = getattr(self, "model", self.__class__.__name__)
        self._fire_callback(
            "on_request", {"prompt_length": len(prompt or ""), "options": options, "driver": driver_name}
        )
        t0 = time.perf_counter()
        try:
            resp = self.generate_music(prompt, options)
        except Exception as exc:
            self._fire_callback("on_error", {"error": exc, "options": options, "driver": driver_name})
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        meta = resp.get("meta", {})
        logger.debug(
            "[music] generate driver=%s tracks=%d cost=%.6f elapsed=%.0fms",
            driver_name,
            meta.get("audio_count", 0),
            meta.get("cost", 0.0),
            elapsed_ms,
        )
        self._fire_callback(
            "on_response",
            {"audio_count": meta.get("audio_count", 0), "meta": meta, "driver": driver_name, "elapsed_ms": elapsed_ms},
        )
        return resp

    def _fire_callback(self, event: str, payload: dict[str, Any]) -> None:
        if self.callbacks is None:
            return
        cb = getattr(self.callbacks, event, None)
        if cb is None:
            return
        try:
            cb(payload)
        except Exception:
            logger.exception("Callback %s raised an exception", event)


class AsyncMusicGenDriver(MusicGenDriver):
    """Async base for music drivers. Implement async ``generate_music``."""

    async def generate_music(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        raise NotImplementedError
