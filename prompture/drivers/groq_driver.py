"""Groq driver for prompture.
Requires the `groq` package. Uses GROQ_API_KEY env var.
"""

import os
from typing import Any

try:
    import groq
except Exception:
    groq = None

from ..driver import Driver


class GroqDriver(Driver):
    # Approximate pricing per 1K tokens (to be updated with official pricing)
    # Each model entry defines token parameters and temperature support
    MODEL_PRICING = {
        "llama2-70b-4096": {
            "prompt": 0.0007,  # Estimated pricing
            "completion": 0.0007,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "mixtral-8x7b-32768": {
            "prompt": 0.0004,  # Estimated pricing
            "completion": 0.0004,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
    }

    def __init__(self, api_key: str | None = None, model: str = "llama2-70b-4096"):
        """Initialize Groq driver.

        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Model to use (defaults to llama2-70b-4096)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        if groq:
            self.client = groq.Client(api_key=self.api_key)
        else:
            self.client = None

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate completion using Groq API.

        Args:
            prompt: Input prompt
            options: Generation options

        Returns:
            Dict containing generated text and metadata

        Raises:
            RuntimeError: If groq package is not installed
            groq.error.*: Various Groq API errors
        """
        if self.client is None:
            raise RuntimeError("groq package is not installed")

        model = options.get("model", self.model)

        # Lookup model-specific config
        model_info = self.MODEL_PRICING.get(model, {})
        tokens_param = model_info.get("tokens_param", "max_tokens")
        supports_temperature = model_info.get("supports_temperature", True)

        # Base configuration
        opts = {"temperature": 0.7, "max_tokens": 512, **options}

        # Base kwargs for API call
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Set token limit with correct parameter name
        kwargs[tokens_param] = opts.get("max_tokens", 512)

        # Only include temperature if model supports it
        if supports_temperature and "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]

        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception:
            # Re-raise any Groq API errors
            raise

        # Extract usage statistics
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        # Calculate costs — try live rates first (per 1M tokens), fall back to hardcoded (per 1K tokens)
        from ..model_rates import get_model_rates

        live_rates = get_model_rates("groq", model)
        if live_rates:
            prompt_cost = (prompt_tokens / 1_000_000) * live_rates["input"]
            completion_cost = (completion_tokens / 1_000_000) * live_rates["output"]
        else:
            model_pricing = self.MODEL_PRICING.get(model, {"prompt": 0, "completion": 0})
            prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
            completion_cost = (completion_tokens / 1000) * model_pricing["completion"]
        total_cost = prompt_cost + completion_cost

        # Standard metadata object
        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp.model_dump(),
            "model_name": model,
        }

        # Extract generated text
        text = resp.choices[0].message.content
        return {"text": text, "meta": meta}
