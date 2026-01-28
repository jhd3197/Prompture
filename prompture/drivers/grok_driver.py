"""xAI Grok driver.
Requires the `requests` package. Uses GROK_API_KEY env var.
"""

import os
from typing import Any

import requests

from ..cost_mixin import CostMixin
from ..driver import Driver


class GrokDriver(CostMixin, Driver):
    supports_json_mode = True

    # Pricing per 1M tokens based on xAI's documentation
    _PRICING_UNIT = 1_000_000
    MODEL_PRICING = {
        "grok-code-fast-1": {
            "prompt": 0.20,
            "completion": 1.50,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "grok-4-fast-reasoning": {
            "prompt": 0.20,
            "completion": 0.50,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "grok-4-fast-non-reasoning": {
            "prompt": 0.20,
            "completion": 0.50,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "grok-4-0709": {
            "prompt": 3.00,
            "completion": 15.00,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "grok-3-mini": {
            "prompt": 0.30,
            "completion": 0.50,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "grok-3": {
            "prompt": 3.00,
            "completion": 15.00,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "grok-2-vision-1212us-east-1": {
            "prompt": 2.00,
            "completion": 10.00,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "grok-2-vision-1212eu-west-1": {
            "prompt": 2.00,
            "completion": 10.00,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
    }

    def __init__(self, api_key: str | None = None, model: str = "grok-4-fast-reasoning"):
        """Initialize Grok driver.

        Args:
            api_key: xAI API key. If not provided, reads from GROK_API_KEY env var
            model: Model to use. Defaults to grok-4-fast-reasoning
        """
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        self.model = model
        self.api_base = "https://api.x.ai/v1"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate completion using Grok API.

        Args:
            prompt: Input prompt
            options: Generation options

        Returns:
            Dict containing generated text and metadata

        Raises:
            RuntimeError: If API key is missing or request fails
        """
        if not self.api_key:
            raise RuntimeError("GROK_API_KEY environment variable is required")

        model = options.get("model", self.model)

        # Lookup model-specific config
        model_info = self.MODEL_PRICING.get(model, {})
        tokens_param = model_info.get("tokens_param", "max_tokens")
        supports_temperature = model_info.get("supports_temperature", True)

        # Defaults
        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        # Base request payload
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Add token limit with correct parameter name
        payload[tokens_param] = opts.get("max_tokens", 512)

        # Add temperature if supported
        if supports_temperature and "temperature" in opts:
            payload["temperature"] = opts["temperature"]

        # Native JSON mode support
        if options.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            resp = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Grok API request failed: {e!s}") from e

        # Extract usage info
        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # Calculate cost via shared mixin
        total_cost = self._calculate_cost("grok", model, prompt_tokens, completion_tokens)

        # Standardized meta object
        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp,
            "model_name": model,
        }

        text = resp["choices"][0]["message"]["content"]
        return {"text": text, "meta": meta}
