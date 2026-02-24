"""Ollama text embedding driver. Uses httpx to call Ollama's /api/embed endpoint."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from .embedding_base import EmbeddingDriver

logger = logging.getLogger(__name__)

# Known Ollama embedding model dimensions
_OLLAMA_MODEL_DIMENSIONS: dict[str, int] = {
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "mxbai-embed-large": 1024,
    "snowflake-arctic-embed": 1024,
    "bge-m3": 1024,
}


class OllamaEmbeddingDriver(EmbeddingDriver):
    """Text embedding via Ollama's /api/embed endpoint (free/local)."""

    default_dimensions = 768
    max_batch_size = 512
    supports_truncation = False

    def __init__(
        self,
        endpoint: str | None = None,
        model: str = "nomic-embed-text",
    ):
        ep = endpoint or os.getenv("OLLAMA_ENDPOINT") or "http://localhost:11434/api/generate"
        self.endpoint = ep
        self.model = model
        # Resolve the base URL from the endpoint (strip /api/*)
        self._base_url = ep.split("/api/")[0] if "/api/" in ep else ep
        self.default_dimensions = _OLLAMA_MODEL_DIMENSIONS.get(model, 768)

    def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        """Generate embeddings using Ollama /api/embed.

        Args:
            texts: List of text strings to embed.
            options: Supports ``model`` (override).
        """
        model = options.get("model", self.model)
        embed_url = f"{self._base_url}/api/embed"

        all_embeddings: list[list[float]] = []

        # Batch if needed
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]

            resp = httpx.post(
                embed_url,
                json={"model": model, "input": batch},
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()

            embeddings = data.get("embeddings", [])
            all_embeddings.extend(embeddings)

        actual_dims = len(all_embeddings[0]) if all_embeddings else self.default_dimensions

        return {
            "embeddings": all_embeddings,
            "meta": {
                "model_name": f"ollama/{model}",
                "dimensions": actual_dims,
                "total_tokens": 0,
                "cost": 0.0,
                "raw_response": {},
            },
        }
