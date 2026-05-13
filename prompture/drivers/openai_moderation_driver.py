"""OpenAI moderation driver.

Uses OpenAI's ``/v1/moderations`` endpoint.  Requires ``OPENAI_API_KEY``.

OpenAI moderation is offered free of charge by OpenAI; the rate entries in
``rates/openai.json`` record this fact (``cost.per_request = 0``) so that
callers can distinguish "known free" from "unknown pricing".
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import requests

from .moderation_base import (
    ModerationDriver,
    ModerationResult,
    calculate_moderation_cost,
)

logger = logging.getLogger(__name__)


class OpenAIModerationDriver(ModerationDriver):
    """OpenAI moderation driver.

    Default model: ``omni-moderation-latest`` (multimodal, recommended).
    Legacy text-only models (``text-moderation-latest`` / ``-stable``) are
    also supported.
    """

    supports_async = False

    DEFAULT_MODEL = "omni-moderation-latest"
    BASE_URL = "https://api.openai.com/v1/moderations"

    KNOWN_MODELS: tuple[str, ...] = (
        "omni-moderation-latest",
        "omni-moderation-2024-09-26",
        "text-moderation-latest",
        "text-moderation-stable",
    )

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY env var.")
        self.model = model or self.DEFAULT_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def moderate(
        self,
        input: str | list[str],
        **options: Any,
    ) -> ModerationResult | list[ModerationResult]:
        is_batch = isinstance(input, list)
        model = options.pop("model", self.model)
        payload: dict[str, Any] = {"model": model, "input": input}
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
            error_msg = f"OpenAI moderation API request failed: {e!s}"
            if body:
                error_msg += f"\nResponse: {body}"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenAI moderation API request failed: {e!s}") from e

        raw_results = resp.get("results", []) or []
        out: list[ModerationResult] = []
        for item in raw_results:
            categories = {k: bool(v) for k, v in (item.get("categories") or {}).items()}
            scores = {k: float(v) for k, v in (item.get("category_scores") or {}).items()}
            flagged_field = item.get("flagged")
            if flagged_field is None:
                flagged = any(categories.values())
            else:
                flagged = bool(flagged_field)
            out.append(
                ModerationResult(
                    flagged=flagged,
                    categories=categories,
                    category_scores=scores,
                    raw_response=item,
                )
            )

        input_count = len(input) if is_batch else 1
        cost, pricing_unknown = calculate_moderation_cost("openai", model, requests=input_count, total_tokens=0)

        self.last_usage = {
            "model_name": f"openai/{model}",
            "input_count": input_count,
            "total_tokens": 0,
            "cost": cost,
            "pricing_unknown": pricing_unknown,
            "raw_response": resp,
        }

        if is_batch:
            return out
        # Single string input → return a single result (first one).
        return out[0] if out else ModerationResult(flagged=False)
