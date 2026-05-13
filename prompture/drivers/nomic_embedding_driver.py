"""Nomic Atlas text embedding driver.

Uses Nomic's ``/v1/embedding/text`` endpoint.  Requires ``NOMIC_API_KEY``.
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


class NomicEmbeddingDriver(EmbeddingCostMixin, EmbeddingDriver):
    """Text embedding via Nomic Atlas ``/v1/embedding/text``.

    Default model: ``nomic-embed-text-v1.5``.  Nomic accepts a ``task_type``
    field — we default to ``"search_document"`` but it can be overridden
    via options.
    """

    default_dimensions = 768
    max_batch_size = 128
    supports_truncation = True

    EMBEDDING_PRICING: dict[str, dict[str, float]] = {
        "nomic-embed-text-v1.5": {"per_million_tokens": 0.10},
        "nomic-embed-text-v1": {"per_million_tokens": 0.10},
        "nomic-embed-vision-v1.5": {"per_million_tokens": 0.10},
    }

    MODEL_DIMENSIONS: dict[str, int] = {
        "nomic-embed-text-v1.5": 768,
        "nomic-embed-text-v1": 768,
        "nomic-embed-vision-v1.5": 768,
    }

    BASE_URL = "https://api-atlas.nomic.ai/v1/embedding/text"

    def __init__(self, api_key: str | None = None, model: str = "nomic-embed-text-v1.5"):
        self.api_key = api_key or os.getenv("NOMIC_API_KEY")
        if not self.api_key:
            raise ValueError("Nomic API key not found. Set NOMIC_API_KEY env var.")
        self.model = model
        self.default_dimensions = self.MODEL_DIMENSIONS.get(model, 768)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)
        task_type = options.get("task_type", "search_document")

        all_embeddings: list[list[float]] = []
        total_tokens = 0
        last_resp: dict[str, Any] = {}

        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            payload: dict[str, Any] = {
                "model": model,
                "texts": batch,
                "task_type": task_type,
            }
            for k in ("dimensionality", "long_text_mode"):
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
                error_msg = f"Nomic embed API request failed: {e!s}"
                if body:
                    error_msg += f"\nResponse: {body}"
                raise RuntimeError(error_msg) from e
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Nomic embed API request failed: {e!s}") from e

            last_resp = resp
            for vec in resp.get("embeddings", []) or []:
                all_embeddings.append(list(vec))

            usage = resp.get("usage", {}) or {}
            total_tokens += int(usage.get("total_tokens", usage.get("prompt_tokens", 0)) or 0)

        actual_dims = len(all_embeddings[0]) if all_embeddings else self.default_dimensions
        cost = self._calculate_embedding_cost("nomic", model, total_tokens=total_tokens)

        return {
            "embeddings": all_embeddings,
            "meta": {
                "model_name": f"nomic/{model}",
                "dimensions": actual_dims,
                "total_tokens": total_tokens,
                "input_tokens": total_tokens,
                "cost": round(cost, 6),
                "raw_response": last_resp,
            },
        }
