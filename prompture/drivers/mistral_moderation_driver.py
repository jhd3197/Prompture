"""Mistral moderation driver.

Uses Mistral's ``/v1/moderations`` endpoint.  Requires ``MISTRAL_API_KEY``.

Unlike OpenAI, Mistral's response does not include a top-level ``flagged``
field.  We compute it as ``any(categories.values())``.
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


class MistralModerationDriver(ModerationDriver):
    """Mistral moderation driver.

    Default model: ``mistral-moderation-latest``.
    """

    supports_async = False

    DEFAULT_MODEL = "mistral-moderation-latest"
    BASE_URL = "https://api.mistral.ai/v1/moderations"

    KNOWN_MODELS: tuple[str, ...] = (
        "mistral-moderation-latest",
        "mistral-moderation-2411",
    )

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

    def moderate(
        self,
        input: str | list[str],
        **options: Any,
    ) -> ModerationResult | list[ModerationResult]:
        is_batch = isinstance(input, list)
        model = options.pop("model", self.model)
        # Mistral's API expects a list of strings — wrap a single string.
        api_input = input if is_batch else [input]
        payload: dict[str, Any] = {"model": model, "input": api_input}
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
            error_msg = f"Mistral moderation API request failed: {e!s}"
            if body:
                error_msg += f"\nResponse: {body}"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Mistral moderation API request failed: {e!s}") from e

        raw_results = resp.get("results", []) or []
        out: list[ModerationResult] = []
        for item in raw_results:
            categories = {k: bool(v) for k, v in (item.get("categories") or {}).items()}
            scores = {k: float(v) for k, v in (item.get("category_scores") or {}).items()}
            # Mistral does NOT return a top-level ``flagged`` — compute it.
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
