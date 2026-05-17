"""Async Cohere v2 embedding driver (httpx)."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import httpx

from ..infra.cost_mixin import EmbeddingCostMixin
from .async_embedding_base import AsyncEmbeddingDriver
from .cohere_embedding_driver import CohereEmbeddingDriver

logger = logging.getLogger(__name__)


class AsyncCohereEmbeddingDriver(EmbeddingCostMixin, AsyncEmbeddingDriver):
    """Async text embedding via Cohere v2 ``/embed`` API."""

    default_dimensions = 1024
    max_batch_size = 96
    supports_truncation = True

    EMBEDDING_PRICING = CohereEmbeddingDriver.EMBEDDING_PRICING
    MODEL_DIMENSIONS = CohereEmbeddingDriver.MODEL_DIMENSIONS
    BASE_URL = CohereEmbeddingDriver.BASE_URL

    def __init__(self, api_key: str | None = None, model: str = "embed-v4.0"):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key not found. Set COHERE_API_KEY env var.")
        self.model = model
        self.default_dimensions = self.MODEL_DIMENSIONS.get(model, 1024)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)
        input_type = options.get("input_type", "search_document")
        embedding_types = options.get("embedding_types", ["float"])

        all_embeddings: list[list[float]] = []
        total_tokens = 0
        last_resp: dict[str, Any] = {}

        async with httpx.AsyncClient(timeout=120) as client:
            for i in range(0, len(texts), self.max_batch_size):
                batch = texts[i : i + self.max_batch_size]
                payload: dict[str, Any] = {
                    "model": model,
                    "texts": batch,
                    "input_type": input_type,
                    "embedding_types": embedding_types,
                }
                for k in ("truncate", "output_dimension"):
                    if k in options:
                        payload[k] = options[k]

                try:
                    response = await client.post(self.BASE_URL, headers=self.headers, json=payload)
                    response.raise_for_status()
                    resp = response.json()
                except httpx.HTTPStatusError as e:
                    body = ""
                    with contextlib.suppress(Exception):
                        body = e.response.text
                    error_msg = f"Cohere embed API request failed: {e!s}"
                    if body:
                        error_msg += f"\nResponse: {body}"
                    raise RuntimeError(error_msg) from e
                except httpx.HTTPError as e:
                    raise RuntimeError(f"Cohere embed API request failed: {e!s}") from e

                last_resp = resp
                embeddings_block = resp.get("embeddings") or {}
                if isinstance(embeddings_block, dict):
                    vectors = embeddings_block.get("float") or []
                elif isinstance(embeddings_block, list):
                    vectors = embeddings_block
                else:
                    vectors = []
                for vec in vectors:
                    all_embeddings.append(list(vec))

                meta_info = resp.get("meta", {}) or {}
                billed = meta_info.get("billed_units") if isinstance(meta_info, dict) else None
                if isinstance(billed, dict):
                    total_tokens += int(billed.get("input_tokens", 0) or 0)

        actual_dims = len(all_embeddings[0]) if all_embeddings else self.default_dimensions
        cost = self._calculate_embedding_cost("cohere", model, total_tokens=total_tokens)

        return {
            "embeddings": all_embeddings,
            "meta": {
                "model_name": f"cohere/{model}",
                "dimensions": actual_dims,
                "total_tokens": total_tokens,
                "input_tokens": total_tokens,
                "cost": round(cost, 6),
                "raw_response": last_resp,
            },
        }
