"""Driver mínimo para OpenAI. Requiere la librería `openai` o puede adaptarse a requests.
USAR con API key en OPENAI_API_KEY env var. Comentarios en español.
"""
import os
from typing import Any, Dict
try:
    import openai
except Exception:
    openai = None

from ..core import Driver

class OpenAIDriver(Driver):
    # OpenAI pricing per 1000 tokens (prices should be kept current with OpenAI's pricing)
    MODEL_PRICING = {
        "gpt-5-mini": {
            "prompt": 0.0003,     # $0.0003 per 1K prompt tokens
            "completion": 0.0006,  # $0.0006 per 1K completion tokens
        },
        "gpt-4o": {
            "prompt": 0.005,      # $0.005 per 1K prompt tokens
            "completion": 0.015,   # $0.015 per 1K completion tokens
        },
        "gpt-4o-mini": {
            "prompt": 0.00015,    # $0.00015 per 1K prompt tokens
            "completion": 0.0006,  # $0.0006 per 1K completion tokens
        },
        "gpt-4": {
            "prompt": 0.03,       # $0.03 per 1K prompt tokens
            "completion": 0.06,    # $0.06 per 1K completion tokens
        },
        "gpt-4-turbo": {
            "prompt": 0.01,       # $0.01 per 1K prompt tokens
            "completion": 0.03,    # $0.03 per 1K completion tokens
        },
        "gpt-3.5-turbo": {
            "prompt": 0.0015,     # $0.0015 per
            "completion": 0.002,   # $0.002 per 1K completion tokens
        },
    }

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if openai:
            openai.api_key = self.api_key

    def generate(self, prompt: str, options: Dict[str,Any]) -> Dict[str,Any]:
        if openai is None:
            raise RuntimeError("openai package not installed")
        model = options.get("model", self.model)

        # Build opts dict, excluding temperature for gpt-5-mini
        if model == "gpt-5-mini":
            opts = {**{"max_tokens": 512}, **options}
            if "max_completion_tokens" not in opts:
                raise ValueError("max_completion_tokens must be provided in options for gpt-5-mini")
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role":"user","content": prompt}],
                max_completion_tokens=opts["max_completion_tokens"]
            )
        else:
            opts = {**{"temperature": 0.0, "max_tokens": 512}, **options}
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role":"user","content": prompt}],
                temperature=opts["temperature"],
                max_tokens=opts["max_tokens"]
            )

        # Extract token usage from OpenAI response
        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # Calculate cost based on model pricing
        model_pricing = self.MODEL_PRICING.get(model, {"prompt": 0, "completion": 0})
        prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * model_pricing["completion"]
        total_cost = prompt_cost + completion_cost

        # Create standardized meta object
        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),  # Round to 6 decimal places
            "raw_response": dict(resp),
            "model_name": model
        }

        text = resp["choices"][0]["message"]["content"]
        return {"text": text, "meta": meta}