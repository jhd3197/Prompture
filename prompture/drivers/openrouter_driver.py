"""OpenRouter driver implementation.
Requires the `requests` package. Uses OPENROUTER_API_KEY env var.
"""
import os
from typing import Any, Dict
import requests

from ..driver import Driver


class OpenRouterDriver(Driver):
    # Approximate pricing per 1K tokens based on OpenRouter's pricing
    # https://openrouter.ai/docs#pricing
    MODEL_PRICING = {
        "openai/gpt-3.5-turbo": {
            "prompt": 0.0015,
            "completion": 0.002,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "anthropic/claude-2": {
            "prompt": 0.008,
            "completion": 0.024,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "google/palm-2-chat-bison": {
            "prompt": 0.0005,
            "completion": 0.0005,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "meta-llama/llama-2-70b-chat": {
            "prompt": 0.0007,
            "completion": 0.0007,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
    }

    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-3.5-turbo"):
        """Initialize OpenRouter driver.
        
        Args:
            api_key: OpenRouter API key. If not provided, will look for OPENROUTER_API_KEY env var
            model: Model to use. Defaults to openai/gpt-3.5-turbo
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY env var.")
        
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Required headers for OpenRouter
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/jhd3197/prompture",  # Required by OpenRouter
            "Content-Type": "application/json",
        }

    def generate(self, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Generate completion using OpenRouter API.
        
        Args:
            prompt: The prompt text
            options: Generation options
            
        Returns:
            Dict containing generated text and metadata
        """
        if not self.api_key:
            raise RuntimeError("OpenRouter API key not found")

        model = options.get("model", self.model)
        
        # Lookup model-specific config
        model_info = self.MODEL_PRICING.get(model, {})
        tokens_param = model_info.get("tokens_param", "max_tokens")
        supports_temperature = model_info.get("supports_temperature", True)

        # Defaults
        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        # Base request data
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Add token limit with correct parameter name
        data[tokens_param] = opts.get("max_tokens", 512)

        # Only include temperature if model supports it
        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
            )
            response.raise_for_status()
            resp = response.json()

            # Extract usage info
            usage = resp.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Calculate cost — try live rates first (per 1M tokens), fall back to hardcoded (per 1K tokens)
            from ..model_rates import get_model_rates
            live_rates = get_model_rates("openrouter", model)
            if live_rates:
                prompt_cost = (prompt_tokens / 1_000_000) * live_rates["input"]
                completion_cost = (completion_tokens / 1_000_000) * live_rates["output"]
            else:
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
                "raw_response": resp,
                "model_name": model,
            }

            text = resp["choices"][0]["message"]["content"]
            return {"text": text, "meta": meta}

        except requests.exceptions.RequestException as e:
            error_msg = f"OpenRouter API request failed: {str(e)}"
            if hasattr(e.response, 'json'):
                try:
                    error_details = e.response.json()
                    error_msg = f"{error_msg} - {error_details.get('error', {}).get('message', '')}"
                except Exception:
                    pass
            raise RuntimeError(error_msg) from e