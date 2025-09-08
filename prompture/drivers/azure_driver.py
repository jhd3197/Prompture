"""Driver for Azure OpenAI Service. Requires the `openai` library.
"""
import os
from typing import Any, Dict
try:
    import openai
except Exception:
    openai = None

from ..core import Driver

class AzureDriver(Driver):
    # Reusing OpenAI pricing model - can be adjusted for Azure-specific pricing if needed
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
    }

    def __init__(
        self, 
        api_key: str | None = None,
        endpoint: str | None = None, 
        deployment_id: str | None = None,
        model: str = "gpt-4o-mini"
    ):
        self.api_key = api_key or os.getenv("AZURE_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_API_ENDPOINT")
        self.deployment_id = deployment_id or os.getenv("AZURE_DEPLOYMENT_ID")
        self.model = model
        
        if openai:
            openai.api_key = self.api_key
            openai.api_type = "azure"
            openai.api_base = self.endpoint
            openai.api_version = "2023-07-01-preview"

    def generate(self, prompt: str, options: Dict[str,Any]) -> Dict[str,Any]:
        if not openai:
            raise RuntimeError("openai package not installed")
            
        opts = {**{"temperature": 0.0, "max_tokens": 512}, **options}
        model = options.get("model", self.model)

        resp = openai.ChatCompletion.create(
            engine=self.deployment_id,
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
            "raw_response": dict(resp)
        }

        text = resp["choices"][0]["message"]["content"]
        return {"text": text, "meta": meta}