"""Voyage AI text embedding driver."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import requests

from ..infra.cost_mixin import EmbeddingCostMixin
from .embedding_base import EmbeddingDriver

logger = logging.getLogger(__name__)


class VoyageEmbeddingDriver(EmbeddingCostMixin, EmbeddingDriver):
    """Text embedding via Voyage AI ``/v1/embeddings``.

    Default model: ``voyage-3.5``.  Optional ``input_type`` (``"query"`` or
    ``"document"``) is passed through when supplied via options.
    """

    default_dimensions = 1024
    max_batch_size = 128
    supports_truncation = True

    EMBEDDING_PRICING: dict[str, dict[str, float]] = {
        "voyage-3.5": {"per_million_tokens": 0.06},
        "voyage-3.5-lite": {"per_million_tokens": 0.02},
        "voyage-3-large": {"per_million_tokens": 0.18},
        "voyage-code-3": {"per_million_tokens": 0.18},
        "voyage-finance-2": {"per_million_tokens": 0.12},
        "voyage-law-2": {"per_million_tokens": 0.12},
        "voyage-multilingual-2": {"per_million_tokens": 0.12},
    }

    MODEL_DIMENSIONS: dict[str, int] = {
        "voyage-3.5": 1024,
        "voyage-3.5-lite": 1024,
        "voyage-3-large": 1024,
        "voyage-code-3": 1024,
        "voyage-finance-2": 1024,
        "voyage-law-2": 1024,
        "voyage-multilingual-2": 1024,
    }

    BASE_URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str | None = None, model: str = "voyage-3.5"):
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Voyage API key not found. Set VOYAGE_API_KEY env var.")
        self.model = model
        self.default_dimensions = self.MODEL_DIMENSIONS.get(model, 1024)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)
        input_type = options.get("input_type", "document")

        all_embeddings: list[list[float]] = []
        total_tokens = 0
        last_resp: dict[str, Any] = {}

        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            payload: dict[str, Any] = {
                "model": model,
                "input": batch,
            }
            if input_type:
                payload["input_type"] = input_type
            for k in ("truncation", "output_dimension", "output_dtype"):
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
                error_msg = f"Voyage embed API request failed: {e!s}"
                if body:
                    error_msg += f"\nResponse: {body}"
                raise RuntimeError(error_msg) from e
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Voyage embed API request failed: {e!s}") from e

            last_resp = resp
            for item in resp.get("data", []) or []:
                vec = item.get("embedding") or []
                all_embeddings.append(list(vec))

            usage = resp.get("usage", {}) or {}
            total_tokens += int(usage.get("total_tokens", 0) or 0)

        actual_dims = len(all_embeddings[0]) if all_embeddings else self.default_dimensions
        cost = self._calculate_embedding_cost("voyage", model, total_tokens=total_tokens)

        return {
            "embeddings": all_embeddings,
            "meta": {
                "model_name": f"voyage/{model}",
                "dimensions": actual_dims,
                "total_tokens": total_tokens,
                "input_tokens": total_tokens,
                "cost": round(cost, 12),
                "raw_response": last_resp,
            },
        }
