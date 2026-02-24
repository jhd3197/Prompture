"""Base class for embedding drivers."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..infra.callbacks import DriverCallbacks

logger = logging.getLogger("prompture.embedding_driver")

# Known model dimensions — maps model IDs to their default embedding dimensions.
# Used by downstream consumers (e.g. vector stores) to pre-allocate storage.
EMBEDDING_MODEL_DIMENSIONS: dict[str, int] = {
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Ollama / local
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "mxbai-embed-large": 1024,
    "snowflake-arctic-embed": 1024,
    "bge-m3": 1024,
    # Fastembed / BAAI
    "bge-small-en-v1.5": 384,
    "bge-base-en-v1.5": 768,
    "bge-large-en-v1.5": 1024,
}


class EmbeddingDriver:
    """Adapter base for text embedding. Implement ``embed(texts, options)``.

    Response contract::

        {
            "embeddings": list[list[float]],
            "meta": {
                "model_name": str,
                "dimensions": int,
                "total_tokens": int,
                "cost": float,
                "raw_response": dict,
            },
        }
    """

    default_dimensions: int = 0
    max_batch_size: int = 2048
    supports_truncation: bool = False

    callbacks: DriverCallbacks | None = None

    def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        """Generate embeddings for a list of texts.

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

    def embed_with_hooks(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        """Wrap :meth:`embed` with on_request / on_response / on_error callbacks."""
        driver_name = getattr(self, "model", self.__class__.__name__)
        self._fire_callback(
            "on_request",
            {"text_count": len(texts), "options": options, "driver": driver_name},
        )
        t0 = time.perf_counter()
        try:
            resp = self.embed(texts, options)
        except Exception as exc:
            self._fire_callback(
                "on_error",
                {"error": exc, "text_count": len(texts), "options": options, "driver": driver_name},
            )
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        meta = resp.get("meta", {})
        logger.debug(
            "[embedding] embed driver=%s texts=%d dims=%d cost=%.6f elapsed=%.0fms",
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
