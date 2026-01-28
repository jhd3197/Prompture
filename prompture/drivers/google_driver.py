import logging
import os
from typing import Any, Optional

import google.generativeai as genai

from ..cost_mixin import CostMixin
from ..driver import Driver

logger = logging.getLogger(__name__)


class GoogleDriver(CostMixin, Driver):
    """Driver for Google's Generative AI API (Gemini)."""

    # Based on current Gemini pricing (as of 2025)
    # Source: https://cloud.google.com/vertex-ai/pricing#gemini_models
    _PRICING_UNIT = 1_000_000
    MODEL_PRICING = {
        "gemini-1.5-pro": {
            "prompt": 0.00025,  # $0.25/1M chars input
            "completion": 0.0005,  # $0.50/1M chars output
        },
        "gemini-1.5-pro-vision": {
            "prompt": 0.00025,  # $0.25/1M chars input
            "completion": 0.0005,  # $0.50/1M chars output
        },
        "gemini-2.5-pro": {
            "prompt": 0.0004,  # $0.40/1M chars input
            "completion": 0.0008,  # $0.80/1M chars output
        },
        "gemini-2.5-flash": {
            "prompt": 0.0004,  # $0.40/1M chars input
            "completion": 0.0008,  # $0.80/1M chars output
        },
        "gemini-2.5-flash-lite": {
            "prompt": 0.0002,  # $0.20/1M chars input
            "completion": 0.0004,  # $0.40/1M chars output
        },
        "gemini-2.0-flash": {
            "prompt": 0.0004,  # $0.40/1M chars input
            "completion": 0.0008,  # $0.80/1M chars output
        },
        "gemini-2.0-flash-lite": {
            "prompt": 0.0002,  # $0.20/1M chars input
            "completion": 0.0004,  # $0.40/1M chars output
        },
        "gemini-1.5-flash": {"prompt": 0.00001875, "completion": 0.000075},
        "gemini-1.5-flash-8b": {"prompt": 0.00001, "completion": 0.00004},
    }

    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-pro"):
        """Initialize the Google Driver.

        Args:
            api_key: Google API key. If not provided, will look for GOOGLE_API_KEY env var
            model: Model to use. Defaults to "gemini-1.5-pro"
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY env var or pass api_key to constructor")

        self.model = model
        # Warn if model is not in pricing table but allow it (might be new)
        if model not in self.MODEL_PRICING:
            logger.warning(f"Model {model} not found in pricing table. Cost calculations will be 0.")

        # Configure google.generativeai
        genai.configure(api_key=self.api_key)
        self.options: dict[str, Any] = {}

        # Validate connection and model availability
        self._validate_connection()

    def _validate_connection(self):
        """Validate connection to Google's API and model availability."""
        try:
            # List models to validate API key and connectivity
            genai.list_models()
            logger.debug("Connection to Google API validated successfully")
        except Exception as e:
            logger.warning(f"Could not validate connection to Google API: {e}")
            raise

    def _calculate_cost_chars(self, prompt_chars: int, completion_chars: int) -> float:
        """Calculate cost from character counts.

        Live rates use token-based pricing (estimate ~4 chars/token).
        Hardcoded MODEL_PRICING uses per-1M-character rates.
        """
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

    def generate(self, prompt: str, options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Generate text using Google's Generative AI.

        Args:
            prompt: The input prompt
            options: Additional options to pass to the model

        Returns:
            Dict containing generated text and metadata
        """
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        # Extract specific options for Google's API
        generation_config = merged_options.get("generation_config", {})
        safety_settings = merged_options.get("safety_settings", {})

        # Map common options to generation_config if not present
        if "temperature" in merged_options and "temperature" not in generation_config:
            generation_config["temperature"] = merged_options["temperature"]
        if "max_tokens" in merged_options and "max_output_tokens" not in generation_config:
            generation_config["max_output_tokens"] = merged_options["max_tokens"]
        if "top_p" in merged_options and "top_p" not in generation_config:
            generation_config["top_p"] = merged_options["top_p"]
        if "top_k" in merged_options and "top_k" not in generation_config:
            generation_config["top_k"] = merged_options["top_k"]

        try:
            logger.debug(f"Initializing {self.model} for generation")
            model = genai.GenerativeModel(self.model)

            # Generate response
            logger.debug(f"Generating with prompt: {prompt}")
            response = model.generate_content(
                prompt,
                generation_config=generation_config if generation_config else None,
                safety_settings=safety_settings if safety_settings else None,
            )

            if not response.text:
                raise ValueError("Empty response from model")

            # Calculate token usage and cost
            prompt_chars = len(prompt)
            completion_chars = len(response.text)

            # Google uses character-based cost estimation
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
