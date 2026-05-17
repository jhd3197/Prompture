"""Voyage AI rerank driver.

Uses Voyage's ``/v1/rerank`` endpoint.  Requires ``VOYAGE_API_KEY``.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import requests

from .rerank_base import RerankDriver, RerankResult, calculate_rerank_cost

logger = logging.getLogger(__name__)


class VoyageRerankDriver(RerankDriver):
    """Voyage AI rerank driver.

    Default model: ``rerank-2.5``.
    """

    supports_async = False

    DEFAULT_MODEL = "rerank-2.5"
    BASE_URL = "https://api.voyageai.com/v1/rerank"

    KNOWN_MODELS: tuple[str, ...] = (
        "rerank-2.5",
        "rerank-2.5-lite",
        "rerank-2",
    )

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Voyage API key not found. Set VOYAGE_API_KEY env var.")
        self.model = model or self.DEFAULT_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        return_documents: bool = False,
        **options: Any,
    ) -> list[RerankResult]:
        model = options.pop("model", self.model)
        payload: dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": documents,
            "return_documents": bool(return_documents),
        }
        if top_n is not None:
            payload["top_k"] = top_n
        for k, v in options.items():
            payload[k] = v

        try:
            response = requests.post(self.BASE_URL, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            resp = response.json()
        except requests.exceptions.HTTPError as e:
            body = ""
            if e.response is not None:
                with contextlib.suppress(Exception):
                    body = e.response.text
            error_msg = f"Voyage rerank API request failed: {e!s}"
            if body:
                error_msg += f"\nResponse: {body}"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Voyage rerank API request failed: {e!s}") from e

        raw_results = resp.get("data", []) or []
        out: list[RerankResult] = []
        for item in raw_results:
            idx = int(item.get("index", 0))
            score = float(item.get("relevance_score", 0.0))
            doc_text: str | None = None
            if return_documents:
                doc_field = item.get("document")
                if isinstance(doc_field, str):
                    doc_text = doc_field
                elif isinstance(doc_field, dict):
                    doc_text = doc_field.get("text")
                else:
                    if 0 <= idx < len(documents):
                        doc_text = documents[idx]
            out.append(RerankResult(index=idx, relevance_score=score, document=doc_text))

        out.sort(key=lambda r: r.relevance_score, reverse=True)

        usage = resp.get("usage", {}) or {}
        total_tokens = int(usage.get("total_tokens", 0) or 0)

        cost, pricing_unknown = calculate_rerank_cost("voyage", model, search_units=0, total_tokens=total_tokens)
        self.last_usage = {
            "model_name": f"voyage/{model}",
            "search_units": 1,
            "total_tokens": total_tokens,
            "cost": cost,
            "pricing_unknown": pricing_unknown,
            "raw_response": resp,
        }
        return out
