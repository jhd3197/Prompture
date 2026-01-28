"""Async Google Generative AI (Gemini) driver."""

from __future__ import annotations

import logging
import os
from typing import Any

import google.generativeai as genai

from ..async_driver import AsyncDriver
from ..cost_mixin import CostMixin
from .google_driver import GoogleDriver

logger = logging.getLogger(__name__)


class AsyncGoogleDriver(CostMixin, AsyncDriver):
    """Async driver for Google's Generative AI API (Gemini)."""

    supports_json_mode = True
    supports_json_schema = True

    MODEL_PRICING = GoogleDriver.MODEL_PRICING
    _PRICING_UNIT = 1_000_000

    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-pro"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY env var or pass api_key to constructor")
        self.model = model
        genai.configure(api_key=self.api_key)
        self.options: dict[str, Any] = {}

    def _calculate_cost_chars(self, prompt_chars: int, completion_chars: int) -> float:
        """Calculate cost from character counts (same logic as sync GoogleDriver)."""
        from ..model_rates import get_model_rates

        live_rates = get_model_rates("google", self.model)
        if live_rates:
            est_prompt_tokens = prompt_chars / 4
            est_completion_tokens = completion_chars / 4
            prompt_cost = (est_prompt_tokens / 1_000_000) * live_rates["input"]
            completion_cost = (est_completion_tokens / 1_000_000) * live_rates["output"]
        else:
            model_pricing = self.MODEL_PRICING.get(self.model, {"prompt": 0, "completion": 0})
            prompt_cost = (prompt_chars / 1_000_000) * model_pricing["prompt"]
            completion_cost = (completion_chars / 1_000_000) * model_pricing["completion"]
        return round(prompt_cost + completion_cost, 6)

    async def generate(self, prompt: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        generation_config = merged_options.get("generation_config", {})
        safety_settings = merged_options.get("safety_settings", {})

        if "temperature" in merged_options and "temperature" not in generation_config:
            generation_config["temperature"] = merged_options["temperature"]
        if "max_tokens" in merged_options and "max_output_tokens" not in generation_config:
            generation_config["max_output_tokens"] = merged_options["max_tokens"]
        if "top_p" in merged_options and "top_p" not in generation_config:
            generation_config["top_p"] = merged_options["top_p"]
        if "top_k" in merged_options and "top_k" not in generation_config:
            generation_config["top_k"] = merged_options["top_k"]

        # Native JSON mode support
        if merged_options.get("json_mode"):
            generation_config["response_mime_type"] = "application/json"
            json_schema = merged_options.get("json_schema")
            if json_schema:
                generation_config["response_schema"] = json_schema

        try:
            model = genai.GenerativeModel(self.model)

            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config if generation_config else None,
                safety_settings=safety_settings if safety_settings else None,
            )

            if not response.text:
                raise ValueError("Empty response from model")

            prompt_chars = len(prompt)
            completion_chars = len(response.text)

            total_cost = self._calculate_cost_chars(prompt_chars, completion_chars)

            meta = {
                "prompt_chars": prompt_chars,
                "completion_chars": completion_chars,
                "total_chars": prompt_chars + completion_chars,
                "cost": total_cost,
                "raw_response": response.prompt_feedback if hasattr(response, "prompt_feedback") else None,
                "model_name": self.model,
            }

            return {"text": response.text, "meta": meta}

        except Exception as e:
            logger.error(f"Google API request failed: {e}")
            raise RuntimeError(f"Google API request failed: {e}") from e
