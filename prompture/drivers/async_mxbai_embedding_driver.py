"""Async Mixedbread embedding driver (httpx)."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import httpx

from ..infra.cost_mixin import EmbeddingCostMixin
from .async_embedding_base import AsyncEmbeddingDriver
from .mxbai_embedding_driver import MxbaiEmbeddingDriver

logger = logging.getLogger(__name__)


class AsyncMxbaiEmbeddingDriver(EmbeddingCostMixin, AsyncEmbeddingDriver):
    """Async text embedding via Mixedbread ``/v1/embeddings`` (OpenAI shape)."""

    default_dimensions = 1024
    max_batch_size = 128
    supports_truncation = True

    EMBEDDING_PRICING = MxbaiEmbeddingDriver.EMBEDDING_PRICING
    MODEL_DIMENSIONS = MxbaiEmbeddingDriver.MODEL_DIMENSIONS
    BASE_URL = MxbaiEmbeddingDriver.BASE_URL

    def __init__(self, api_key: str | None = None, model: str = "mxbai-embed-large-v1"):
        self.api_key = api_key or os.getenv("MIXEDBREAD_API_KEY")
        if not self.api_key:
            raise ValueError("Mixedbread API key not found. Set MIXEDBREAD_API_KEY env var.")
        self.model = model
        self.default_dimensions = self.MODEL_DIMENSIONS.get(model, 1024)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)
        encoding_format = options.get("encoding_format", "float")

        all_embeddings: list[list[float]] = []
        total_tokens = 0
        last_resp: dict[str, Any] = {}

        async with httpx.AsyncClient(timeout=120) as client:
            for i in range(0, len(texts), self.max_batch_size):
                batch = texts[i : i + self.max_batch_size]
                payload: dict[str, Any] = {
                    "model": model,
                    "input": batch,
                    "encoding_format": encoding_format,
                }
                for k in ("dimensions", "normalized", "truncation_strategy"):
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
                    error_msg = f"Mixedbread embed API request failed: {e!s}"
                    if body:
                        error_msg += f"\nResponse: {body}"
                    raise RuntimeError(error_msg) from e
                except httpx.HTTPError as e:
                    raise RuntimeError(f"Mixedbread embed API request failed: {e!s}") from e

                last_resp = resp
                for item in resp.get("data", []) or []:
                    vec = item.get("embedding") or []
                    all_embeddings.append(list(vec))

                usage = resp.get("usage", {}) or {}
                total_tokens += int(usage.get("total_tokens", usage.get("prompt_tokens", 0)) or 0)

        actual_dims = len(all_embeddings[0]) if all_embeddings else self.default_dimensions
        cost = self._calculate_embedding_cost("mixedbread", model, total_tokens=total_tokens)

        return {
            "embeddings": all_embeddings,
            "meta": {
                "model_name": f"mixedbread/{model}",
                "dimensions": actual_dims,
                "total_tokens": total_tokens,
                "input_tokens": total_tokens,
                "cost": round(cost, 6),
                "raw_response": last_resp,
            },
        }
