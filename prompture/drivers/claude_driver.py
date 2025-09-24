"""Driver for Anthropic's Claude models. Requires the `anthropic` library.
Use with API key in CLAUDE_API_KEY env var or provide directly.
"""
import os
from typing import Any, Dict
try:
    import anthropic
except Exception:
    anthropic = None

from ..driver import Driver

class ClaudeDriver(Driver):
    # Claude pricing per 1000 tokens (prices should be kept current with Anthropic's pricing)
    MODEL_PRICING = {
        # Claude Opus 4.1
        "claude-opus-4-1-20250805": {
            "prompt": 0.015,      # $15 per 1M prompt tokens
            "completion": 0.075,   # $75 per 1M completion tokens
        },
        # Claude Opus 4.0
        "claude-opus-4-20250514": {
            "prompt": 0.015,      # $15 per 1M prompt tokens
            "completion": 0.075,   # $75 per 1M completion tokens
        },
        # Claude Sonnet 4.0
        "claude-sonnet-4-20250514": {
            "prompt": 0.003,      # $3 per 1M prompt tokens
            "completion": 0.015,   # $15 per 1M completion tokens
        },
        # Claude Sonnet 3.7
        "claude-3-7-sonnet-20250219": {
            "prompt": 0.003,      # $3 per 1M prompt tokens
            "completion": 0.015,   # $15 per 1M completion tokens
        },
        # Claude Haiku 3.5
        "claude-3-5-haiku-20241022": {
            "prompt": 0.0008,     # $0.80 per 1M prompt tokens
            "completion": 0.004,   # $4 per 1M completion tokens
        }
    }

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-haiku-20241022"):
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        self.model = model or os.getenv("CLAUDE_MODEL_NAME", "claude-3-5-haiku-20241022")

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
        prompt_tokens = resp.usage.input_tokens
        completion_tokens = resp.usage.output_tokens
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