"""OpenAI text embedding driver. Requires the ``openai`` package (>=1.0.0)."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore[misc, assignment]

from ..infra.cost_mixin import EmbeddingCostMixin
from .embedding_base import EmbeddingDriver

logger = logging.getLogger(__name__)


class OpenAIEmbeddingDriver(EmbeddingCostMixin, EmbeddingDriver):
    """Text embedding via OpenAI Embeddings API."""

    default_dimensions = 1536
    max_batch_size = 2048
    supports_truncation = True

    EMBEDDING_PRICING: dict[str, dict[str, float]] = {
        "text-embedding-3-small": {"per_million_tokens": 0.02},
        "text-embedding-3-large": {"per_million_tokens": 0.13},
        "text-embedding-ada-002": {"per_million_tokens": 0.10},
    }

    MODEL_DIMENSIONS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.default_dimensions = self.MODEL_DIMENSIONS.get(model, 1536)
        if OpenAI is not None:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        """Generate embeddings using OpenAI Embeddings API.

        Args:
            texts: List of text strings to embed.
            options: Supports ``dimensions`` (int, for embedding-3 models),
                     ``model`` (override), ``encoding_format`` ("float"/"base64").
        """
        if self.client is None:
            raise RuntimeError("openai package (>=1.0.0) is not installed")

        model = options.get("model", self.model)
        dimensions = options.get("dimensions")

        all_embeddings: list[list[float]] = []
        total_tokens = 0

        # Batch if needed
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]

            kwargs: dict[str, Any] = {
                "model": model,
                "input": batch,
            }
            # Only embedding-3 models support the dimensions parameter
            if dimensions and "embedding-3" in model:
                kwargs["dimensions"] = dimensions

            resp = self.client.embeddings.create(**kwargs)

            for item in resp.data:
                all_embeddings.append(item.embedding)

            total_tokens += resp.usage.total_tokens

        actual_dims = len(all_embeddings[0]) if all_embeddings else self.default_dimensions
        cost = self._calculate_embedding_cost("openai", model, total_tokens=total_tokens)

        return {
            "embeddings": all_embeddings,
            "meta": {
                "model_name": f"openai/{model}",
                "dimensions": actual_dims,
                "total_tokens": total_tokens,
                "cost": round(cost, 6),
                "raw_response": {},
            },
        }
