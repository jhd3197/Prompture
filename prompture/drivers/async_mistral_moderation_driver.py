"""Async Mistral moderation driver."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import httpx

from .moderation_base import (
    AsyncModerationDriver,
    ModerationResult,
    calculate_moderation_cost,
)

logger = logging.getLogger(__name__)


class AsyncMistralModerationDriver(AsyncModerationDriver):
    """Async Mistral moderation driver."""

    DEFAULT_MODEL = "mistral-moderation-latest"
    BASE_URL = "https://api.mistral.ai/v1/moderations"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY env var.")
        self.model = model or self.DEFAULT_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def moderate(
        self,
        input: str | list[str],
        **options: Any,
    ) -> ModerationResult | list[ModerationResult]:
        is_batch = isinstance(input, list)
        model = options.pop("model", self.model)
        api_input = input if is_batch else [input]
        payload: dict[str, Any] = {"model": model, "input": api_input}
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
            error_msg = f"Mistral moderation API request failed: {e!s}"
            if body:
                error_msg += f"\nResponse: {body}"
            raise RuntimeError(error_msg) from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"Mistral moderation API request failed: {e!s}") from e

        raw_results = resp.get("results", []) or []
        out: list[ModerationResult] = []
        for item in raw_results:
            categories = {k: bool(v) for k, v in (item.get("categories") or {}).items()}
            scores = {k: float(v) for k, v in (item.get("category_scores") or {}).items()}
            flagged = any(categories.values())
            out.append(
                ModerationResult(
                    flagged=flagged,
                    categories=categories,
                    category_scores=scores,
                    raw_response=item,
                )
            )

        input_count = len(api_input)
        cost, pricing_unknown = calculate_moderation_cost("mistral", model, requests=input_count, total_tokens=0)

        self.last_usage = {
            "model_name": f"mistral/{model}",
            "input_count": input_count,
            "total_tokens": 0,
            "cost": cost,
            "pricing_unknown": pricing_unknown,
            "raw_response": resp,
        }

        if is_batch:
            return out
        return out[0] if out else ModerationResult(flagged=False)
