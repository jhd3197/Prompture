"""Async base class for video generation drivers."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..infra.callbacks import DriverCallbacks
from ._media_usage import record_media_usage

logger = logging.getLogger("prompture.async_video_gen_driver")


class AsyncVideoGenDriver:
    """Async adapter base for video generation. Implement ``async generate_video(prompt, options)``."""

    supports_image_input: bool = False
    supports_reference_images: bool = False
    supports_video_input: bool = False
    supports_audio: bool = False
    supports_polling: bool = True
    supported_aspect_ratios: list[str] = []
    supported_resolutions: list[str] = []
    max_seconds: int | None = None

    callbacks: DriverCallbacks | None = None

    async def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate video(s) from a text prompt (async)."""
        raise NotImplementedError

    async def generate_video_with_hooks(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Wrap :meth:`generate_video` with on_request / on_response / on_error callbacks."""
        driver_name = getattr(self, "model", self.__class__.__name__)
        self._fire_callback(
            "on_request",
            {"prompt_length": len(prompt), "options": options, "driver": driver_name},
        )
        t0 = time.perf_counter()
        try:
            resp = await self.generate_video(prompt, options)
        except Exception as exc:
            self._fire_callback(
                "on_error",
                {"error": exc, "prompt_length": len(prompt), "options": options, "driver": driver_name},
            )
            record_media_usage(
                self, {}, (time.perf_counter() - t0) * 1000,
                modality="video", count_key="video_count", status="error", error=exc,
            )
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        meta = resp.get("meta", {})
        logger.debug(
            "[video_gen] async generate driver=%s videos=%d cost=%.6f elapsed=%.0fms",
            driver_name,
            meta.get("video_count", 0),
            meta.get("cost", 0.0),
            elapsed_ms,
        )
        self._fire_callback(
            "on_response",
            {
                "video_count": meta.get("video_count", 0),
                "meta": meta,
                "driver": driver_name,
                "elapsed_ms": elapsed_ms,
            },
        )
        record_media_usage(self, meta, elapsed_ms, modality="video", count_key="video_count")
        return resp

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
