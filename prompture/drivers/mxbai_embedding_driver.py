"""Mixedbread (Mxbai) text embedding driver.

Uses Mixedbread's OpenAI-compatible ``/v1/embeddings`` endpoint.
Requires ``MIXEDBREAD_API_KEY``.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import requests

from ..infra.cost_mixin import EmbeddingCostMixin
from .embedding_base import EmbeddingDriver

logger = logging.getLogger(__name__)


class MxbaiEmbeddingDriver(EmbeddingCostMixin, EmbeddingDriver):
    """Text embedding via Mixedbread ``/v1/embeddings`` (OpenAI shape).

    Default model: ``mxbai-embed-large-v1``.
    """

    default_dimensions = 1024
    max_batch_size = 128
    supports_truncation = True

    EMBEDDING_PRICING: dict[str, dict[str, float]] = {
        "mxbai-embed-large-v1": {"per_million_tokens": 0.02},
        "mxbai-embed-2d-large-v1": {"per_million_tokens": 0.02},
    }

    MODEL_DIMENSIONS: dict[str, int] = {
        "mxbai-embed-large-v1": 1024,
        "mxbai-embed-2d-large-v1": 1024,
    }

    BASE_URL = "https://api.mixedbread.com/v1/embeddings"

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

    def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)
        encoding_format = options.get("encoding_format", "float")

        all_embeddings: list[list[float]] = []
        total_tokens = 0
        last_resp: dict[str, Any] = {}

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
                response = requests.post(self.BASE_URL, headers=self.headers, json=payload, timeout=120)
                response.raise_for_status()
                resp = response.json()
            except requests.exceptions.HTTPError as e:
                body = ""
                if e.response is not None:
                    with contextlib.suppress(Exception):
                        body = e.response.text
                error_msg = f"Mixedbread embed API request failed: {e!s}"
                if body:
                    error_msg += f"\nResponse: {body}"
                raise RuntimeError(error_msg) from e
            except requests.exceptions.RequestException as e:
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
