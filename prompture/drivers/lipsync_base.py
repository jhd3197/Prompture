"""Base class for lipsync / talking-head generation drivers."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..infra.callbacks import DriverCallbacks

logger = logging.getLogger("prompture.lipsync_driver")


class LipsyncDriver:
    """Adapter base for lipsync generation. Implement ``generate_lipsync``.

    A lipsync model takes an **audio** track plus either a portrait **image**
    (image + audio → talking-head video) or a source **video** (video + audio →
    relipsynced video) and produces a video.

    Response contract (mirrors video generation)::

        {
            "videos": list[VideoContent],
            "meta": {
                "video_count": int,
                "category": "image" | "video",
                "cost": float,
                "model_name": str,
                "request_id": str | None,
                "raw_response": dict,
            },
        }
    """

    supports_image: bool = True  # image + audio → video
    supports_video: bool = True  # video + audio → video
    supports_polling: bool = True

    callbacks: DriverCallbacks | None = None

    def generate_lipsync(
        self,
        audio: Any,
        options: dict[str, Any],
        *,
        image: Any | None = None,
        video: Any | None = None,
    ) -> dict[str, Any]:
        """Generate a lipsynced video from ``audio`` plus an ``image`` or ``video``.

        Inputs may be URL strings, raw bytes, file paths, or media ``*Content``
        objects; conditioned inputs are uploaded/hosted as needed. Exactly one of
        ``image`` / ``video`` is normally supplied.
        """
        raise NotImplementedError

    def generate_lipsync_with_hooks(
        self,
        audio: Any,
        options: dict[str, Any],
        *,
        image: Any | None = None,
        video: Any | None = None,
    ) -> dict[str, Any]:
        """Wrap :meth:`generate_lipsync` with on_request / on_response / on_error callbacks."""
        driver_name = getattr(self, "model", self.__class__.__name__)
        self._fire_callback("on_request", {"options": options, "driver": driver_name})
        t0 = time.perf_counter()
        try:
            resp = self.generate_lipsync(audio, options, image=image, video=video)
        except Exception as exc:
            self._fire_callback("on_error", {"error": exc, "options": options, "driver": driver_name})
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        meta = resp.get("meta", {})
        logger.debug(
            "[lipsync] generate driver=%s videos=%d cost=%.6f elapsed=%.0fms",
            driver_name,
            meta.get("video_count", 0),
            meta.get("cost", 0.0),
            elapsed_ms,
        )
        self._fire_callback(
            "on_response",
            {"video_count": meta.get("video_count", 0), "meta": meta, "driver": driver_name, "elapsed_ms": elapsed_ms},
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


class AsyncLipsyncDriver(LipsyncDriver):
    """Async base for lipsync drivers. Implement async ``generate_lipsync``."""

    async def generate_lipsync(  # type: ignore[override]
        self,
        audio: Any,
        options: dict[str, Any],
        *,
        image: Any | None = None,
        video: Any | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError
