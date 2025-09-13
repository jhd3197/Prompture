"""Driver for Anthropic's Claude models. Requires the `anthropic` library.
Use with API key in CLAUDE_API_KEY env var or provide directly.
"""
import os
from typing import Any, Dict
try:
    import anthropic
except Exception:
    anthropic = None

from ..core import Driver

class ClaudeDriver(Driver):
    # Claude pricing per 1000 tokens (prices should be kept current with Anthropic's pricing)
    MODEL_PRICING = {
        "claude-3-opus-20240229": {
            "prompt": 0.015,      # $0.015 per 1K prompt tokens
            "completion": 0.075,   # $0.075 per 1K completion tokens
        },
        "claude-3-sonnet-20240229": {
            "prompt": 0.003,      # $0.003 per 1K prompt tokens
            "completion": 0.015,   # $0.015 per 1K completion tokens
        },
        "claude-3-haiku-20240307": {
            "prompt": 0.00025,    # $0.00025 per 1K prompt tokens
            "completion": 0.00125, # $0.00125 per 1K completion tokens
        },
    }

    def __init__(self, api_key: str | None = None, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        self.model = model

    def generate(self, prompt: str, options: Dict[str,Any]) -> Dict[str,Any]:
        if anthropic is None:
            raise RuntimeError("anthropic package not installed")
        
        opts = {**{"temperature": 0.0, "max_tokens": 512}, **options}
        model = options.get("model", self.model)
        
        client = anthropic.Anthropic(api_key=self.api_key)
        resp = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=opts["temperature"],
            max_tokens=opts["max_tokens"]
        )
        
        # Extract token usage from Claude response
        usage = resp.usage
        prompt_tokens = usage.input_tokens
        completion_tokens = usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens
        
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
        
        text = resp.content[0].text
        return {"text": text, "meta": meta}