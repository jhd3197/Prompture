"""Async base class for embedding drivers."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..infra.callbacks import DriverCallbacks

logger = logging.getLogger("prompture.async_embedding_driver")


class AsyncEmbeddingDriver:
    """Async adapter base for text embedding. Implement ``async embed(texts, options)``.

    Response and contracts are identical to :class:`EmbeddingDriver`.
    """

    default_dimensions: int = 0
    max_batch_size: int = 2048
    supports_truncation: bool = False

    callbacks: DriverCallbacks | None = None

    async def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        """Generate embeddings for a list of texts (async).

        Args:
            texts: List of text strings to embed.
            options: Provider-specific options (dimensions, encoding_format, etc.).

        Returns:
            Dict with ``embeddings`` and ``meta`` keys.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Hook-aware wrapper
    # ------------------------------------------------------------------

    async def embed_with_hooks(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        """Wrap :meth:`embed` with on_request / on_response / on_error callbacks."""
        driver_name = getattr(self, "model", self.__class__.__name__)
        self._fire_callback(
            "on_request",
            {"text_count": len(texts), "options": options, "driver": driver_name},
        )
        t0 = time.perf_counter()
        try:
            resp = await self.embed(texts, options)
        except Exception as exc:
            self._fire_callback(
                "on_error",
                {"error": exc, "text_count": len(texts), "options": options, "driver": driver_name},
            )
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        meta = resp.get("meta", {})
        logger.debug(
            "[embedding] async embed driver=%s texts=%d dims=%d cost=%.6f elapsed=%.0fms",
            driver_name,
            len(texts),
            meta.get("dimensions", 0),
            meta.get("cost", 0.0),
            elapsed_ms,
        )
        self._fire_callback(
            "on_response",
            {
                "text_count": len(texts),
                "dimensions": meta.get("dimensions", 0),
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
