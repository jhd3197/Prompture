"""Driver for Azure OpenAI Service (migrated to openai>=1.0.0).
Requires the `openai` package.
"""
import os
from typing import Any, Dict
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

from ..driver import Driver


class AzureDriver(Driver):
    # Pricing per 1K tokens (adjust if your Azure pricing differs from OpenAI defaults)
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
        "gpt-4.1": {
            "prompt": 0.03,
            "completion": 0.06,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
    }

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        deployment_id: str | None = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.getenv("AZURE_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_API_ENDPOINT")
        self.deployment_id = deployment_id or os.getenv("AZURE_DEPLOYMENT_ID")
        self.api_version = os.getenv("AZURE_API_VERSION", "2023-07-01-preview")
        self.model = model

        # Validate required configuration
        if not self.api_key:
            raise ValueError("Missing Azure API key (AZURE_API_KEY).")
        if not self.endpoint:
            raise ValueError("Missing Azure API endpoint (AZURE_API_ENDPOINT).")
        if not self.deployment_id:
            raise ValueError("Missing Azure deployment ID (AZURE_DEPLOYMENT_ID).")

        if AzureOpenAI:
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
            )
        else:
            self.client = None

    def generate(self, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError("openai package (>=1.0.0) with AzureOpenAI not installed")

        model = options.get("model", self.model)
        model_info = self.MODEL_PRICING.get(model, {})
        tokens_param = model_info.get("tokens_param", "max_tokens")
        supports_temperature = model_info.get("supports_temperature", True)

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        # Build request kwargs
        kwargs = {
            "model": self.deployment_id,  # for Azure, use deployment name
            "messages": [{"role": "user", "content": prompt}],
        }
        kwargs[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]

        resp = self.client.chat.completions.create(**kwargs)

        # Extract usage
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
            "deployment_id": self.deployment_id,
        }

        text = resp.choices[0].message.content
        return {"text": text, "meta": meta}
