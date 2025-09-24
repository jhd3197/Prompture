"""Minimal OpenAI driver (migrated to openai>=1.0.0).
Requires the `openai` package. Uses OPENAI_API_KEY env var.
"""
import os
from typing import Any, Dict
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from ..driver import Driver


class OpenAIDriver(Driver):
    # Approximate pricing per 1K tokens (keep updated with OpenAI's official pricing)
    # Each model entry also defines which token parameter it supports and
    # whether it accepts temperature.
    MODEL_PRICING = {
        "gpt-5-mini": {
            "prompt": 0.0003,
            "completion": 0.0006,
            "tokens_param": "max_completion_tokens",
            "supports_temperature": False,
        },
        "gpt-4o": {
            "prompt": 0.005,
            "completion": 0.015,
            "tokens_param": "max_completion_tokens",
            "supports_temperature": True,
        },
        "gpt-4o-mini": {
            "prompt": 0.00015,
            "completion": 0.0006,
            "tokens_param": "max_completion_tokens",
            "supports_temperature": True,
        },
        "gpt-4": {
            "prompt": 0.03,
            "completion": 0.06,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "gpt-4-turbo": {
            "prompt": 0.01,
            "completion": 0.03,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "gpt-3.5-turbo": {
            "prompt": 0.0015,
            "completion": 0.002,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
    }

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if OpenAI:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def generate(self, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError("openai package (>=1.0.0) is not installed")

        model = options.get("model", self.model)

        # Lookup model-specific config
        model_info = self.MODEL_PRICING.get(model, {})
        tokens_param = model_info.get("tokens_param", "max_tokens")
        supports_temperature = model_info.get("supports_temperature", True)

        # Defaults
        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        # Base kwargs
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Assign token limit with the correct parameter name
        kwargs[tokens_param] = opts.get("max_tokens", 512)

        # Only include temperature if the model supports it
        if supports_temperature and "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]

        resp = self.client.chat.completions.create(**kwargs)

        # Extract usage info
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        # Calculate cost
        model_pricing = self.MODEL_PRICING.get(model, {"prompt": 0, "completion": 0})
        prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * model_pricing["completion"]
        total_cost = prompt_cost + completion_cost

        # Standardized meta object
        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp.model_dump(),
            "model_name": model,
        }

        text = resp.choices[0].message.content
        return {"text": text, "meta": meta}
