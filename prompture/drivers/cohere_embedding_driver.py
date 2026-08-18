"""Cohere text embedding driver (v2 ``/embed`` endpoint)."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import requests

from ..infra.cost_mixin import EmbeddingCostMixin
from .embedding_base import EmbeddingDriver

logger = logging.getLogger(__name__)


class CohereEmbeddingDriver(EmbeddingCostMixin, EmbeddingDriver):
    """Text embedding via Cohere v2 ``/embed`` API.

    Default model: ``embed-v4.0``.  Cohere requires an ``input_type`` field;
    we default to ``"search_document"`` but it can be overridden via options.
    """

    default_dimensions = 1024
    max_batch_size = 96  # Cohere v2 limit per request
    supports_truncation = True

    EMBEDDING_PRICING: dict[str, dict[str, float]] = {
        "embed-v4.0": {"per_million_tokens": 0.12},
        "embed-english-v3.0": {"per_million_tokens": 0.10},
        "embed-multilingual-v3.0": {"per_million_tokens": 0.10},
        "embed-english-light-v3.0": {"per_million_tokens": 0.10},
    }

    MODEL_DIMENSIONS: dict[str, int] = {
        "embed-v4.0": 1536,
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
    }

    BASE_URL = "https://api.cohere.com/v2/embed"

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

    def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)
        input_type = options.get("input_type", "search_document")
        embedding_types = options.get("embedding_types", ["float"])

        all_embeddings: list[list[float]] = []
        total_tokens = 0
        last_resp: dict[str, Any] = {}

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
                response = requests.post(self.BASE_URL, headers=self.headers, json=payload, timeout=120)
                response.raise_for_status()
                resp = response.json()
            except requests.exceptions.HTTPError as e:
                body = ""
                if e.response is not None:
                    with contextlib.suppress(Exception):
                        body = e.response.text
                error_msg = f"Cohere embed API request failed: {e!s}"
                if body:
                    error_msg += f"\nResponse: {body}"
                raise RuntimeError(error_msg) from e
            except requests.exceptions.RequestException as e:
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
                "cost": round(cost, 12),
                "raw_response": last_resp,
            },
        }
