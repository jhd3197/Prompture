"""Cohere rerank driver.

Uses Cohere's v2 ``/rerank`` endpoint.  Requires ``COHERE_API_KEY``.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import requests

from .rerank_base import RerankDriver, RerankResult

logger = logging.getLogger(__name__)


class CohereRerankDriver(RerankDriver):
    """Cohere rerank driver (v2 API).

    Default model: ``rerank-v3.5``.
    """

    supports_async = False

    DEFAULT_MODEL = "rerank-v3.5"
    BASE_URL = "https://api.cohere.com/v2/rerank"

    KNOWN_MODELS: tuple[str, ...] = (
        "rerank-v3.5",
        "rerank-english-v3.0",
        "rerank-multilingual-v3.0",
    )

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key not found. Set COHERE_API_KEY env var.")
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
            payload["top_n"] = top_n
        # Forward remaining provider-specific options
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
            error_msg = f"Cohere rerank API request failed: {e!s}"
            if body:
                error_msg += f"\nResponse: {body}"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Cohere rerank API request failed: {e!s}") from e

        raw_results = resp.get("results", []) or []
        out: list[RerankResult] = []
        for item in raw_results:
            idx = int(item.get("index", 0))
            score = float(item.get("relevance_score", 0.0))
            doc_text: str | None = None
            if return_documents:
                doc_field = item.get("document")
                if isinstance(doc_field, dict):
                    doc_text = doc_field.get("text")
                elif isinstance(doc_field, str):
                    doc_text = doc_field
                else:
                    # Fall back to original input
                    if 0 <= idx < len(documents):
                        doc_text = documents[idx]
            out.append(RerankResult(index=idx, relevance_score=score, document=doc_text))

        # Cohere returns results sorted by score, but be defensive.
        out.sort(key=lambda r: r.relevance_score, reverse=True)

        meta_info = resp.get("meta", {}) or {}
        billed = (meta_info.get("billed_units") or {}) if isinstance(meta_info, dict) else {}
        search_units = int(billed.get("search_units", 1) or 1)

        # Phase 2: no rate JSON for rerank providers yet — flag as unknown.
        cost = 0.0
        pricing_unknown = True

        self.last_usage = {
            "model_name": f"cohere/{model}",
            "search_units": search_units,
            "total_tokens": 0,
            "cost": cost,
            "pricing_unknown": pricing_unknown,
            "raw_response": resp,
        }
        return out
