"""Async Mixedbread rerank driver (httpx)."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import httpx

from .rerank_base import AsyncRerankDriver, RerankResult, calculate_rerank_cost

logger = logging.getLogger(__name__)


class AsyncMxbaiRerankDriver(AsyncRerankDriver):
    """Async Mixedbread rerank driver."""

    DEFAULT_MODEL = "mxbai-rerank-large-v1"
    BASE_URL = "https://api.mixedbread.com/v1/rerank"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("MIXEDBREAD_API_KEY")
        if not self.api_key:
            raise ValueError("Mixedbread API key not found. Set MIXEDBREAD_API_KEY env var.")
        self.model = model or self.DEFAULT_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def rerank(
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
            "input": documents,
            "return_input": bool(return_documents),
        }
        if top_n is not None:
            payload["top_k"] = top_n
        for k, v in options.items():
            payload[k] = v

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(self.BASE_URL, headers=self.headers, json=payload)
                response.raise_for_status()
                resp = response.json()
        except httpx.HTTPStatusError as e:
            body = ""
            with contextlib.suppress(Exception):
                body = e.response.text
            error_msg = f"Mixedbread rerank API request failed: {e!s}"
            if body:
                error_msg += f"\nResponse: {body}"
            raise RuntimeError(error_msg) from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"Mixedbread rerank API request failed: {e!s}") from e

        raw_results = resp.get("data", []) or resp.get("results", []) or []
        out: list[RerankResult] = []
        for item in raw_results:
            idx = int(item.get("index", 0))
            score = float(item.get("score", item.get("relevance_score", 0.0)) or 0.0)
            doc_text: str | None = None
            if return_documents:
                doc_field = item.get("input") if "input" in item else item.get("document")
                if isinstance(doc_field, dict):
                    doc_text = doc_field.get("text")
                elif isinstance(doc_field, str):
                    doc_text = doc_field
                elif 0 <= idx < len(documents):
                    doc_text = documents[idx]
            out.append(RerankResult(index=idx, relevance_score=score, document=doc_text))

        out.sort(key=lambda r: r.relevance_score, reverse=True)

        usage = resp.get("usage", {}) or {}
        total_tokens = int(usage.get("total_tokens", 0) or 0)

        cost, pricing_unknown = calculate_rerank_cost("mixedbread", model, search_units=0, total_tokens=total_tokens)

        self.last_usage = {
            "model_name": f"mixedbread/{model}",
            "search_units": 0,
            "total_tokens": total_tokens,
            "cost": cost,
            "pricing_unknown": pricing_unknown,
            "raw_response": resp,
        }
        return out
