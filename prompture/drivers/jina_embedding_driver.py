"""Jina AI text embedding driver."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import requests

from ..infra.cost_mixin import EmbeddingCostMixin
from .embedding_base import EmbeddingDriver

logger = logging.getLogger(__name__)


class JinaEmbeddingDriver(EmbeddingCostMixin, EmbeddingDriver):
    """Text embedding via Jina AI ``/v1/embeddings``.

    Default model: ``jina-embeddings-v3``.
    """

    default_dimensions = 1024
    max_batch_size = 128
    supports_truncation = True

    EMBEDDING_PRICING: dict[str, dict[str, float]] = {
        "jina-embeddings-v3": {"per_million_tokens": 0.02},
        "jina-embeddings-v2-base-en": {"per_million_tokens": 0.018},
        "jina-clip-v2": {"per_million_tokens": 0.02},
        "jina-colbert-v2": {"per_million_tokens": 0.02},
    }

    MODEL_DIMENSIONS: dict[str, int] = {
        "jina-embeddings-v3": 1024,
        "jina-embeddings-v2-base-en": 768,
        "jina-clip-v2": 1024,
        "jina-colbert-v2": 128,
    }

    BASE_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(self, api_key: str | None = None, model: str = "jina-embeddings-v3"):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Jina API key not found. Set JINA_API_KEY env var.")
        self.model = model
        self.default_dimensions = self.MODEL_DIMENSIONS.get(model, 1024)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)
        task = options.get("task", "retrieval.passage")

        all_embeddings: list[list[float]] = []
        total_tokens = 0
        last_resp: dict[str, Any] = {}

        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            payload: dict[str, Any] = {"model": model, "input": batch}
            if task:
                payload["task"] = task
            for k in ("dimensions", "late_chunking", "embedding_type", "truncate"):
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
                error_msg = f"Jina embed API request failed: {e!s}"
                if body:
                    error_msg += f"\nResponse: {body}"
                raise RuntimeError(error_msg) from e
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Jina embed API request failed: {e!s}") from e

            last_resp = resp
            for item in resp.get("data", []) or []:
                vec = item.get("embedding") or []
                all_embeddings.append(list(vec))

            usage = resp.get("usage", {}) or {}
            total_tokens += int(usage.get("total_tokens", usage.get("prompt_tokens", 0)) or 0)

        actual_dims = len(all_embeddings[0]) if all_embeddings else self.default_dimensions
        cost = self._calculate_embedding_cost("jina", model, total_tokens=total_tokens)

        return {
            "embeddings": all_embeddings,
            "meta": {
                "model_name": f"jina/{model}",
                "dimensions": actual_dims,
                "total_tokens": total_tokens,
                "input_tokens": total_tokens,
                "cost": round(cost, 6),
                "raw_response": last_resp,
            },
        }
