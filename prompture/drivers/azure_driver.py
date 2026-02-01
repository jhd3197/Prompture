"""Driver for Azure OpenAI Service (migrated to openai>=1.0.0).
Requires the `openai` package.
"""

import os
from typing import Any

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

from ..cost_mixin import CostMixin, prepare_strict_schema
from ..driver import Driver


class AzureDriver(CostMixin, Driver):
    supports_json_mode = True
    supports_json_schema = True
    supports_vision = True

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

    supports_messages = True

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_openai_vision_messages

        return _prepare_openai_vision_messages(messages)

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return self._do_generate(messages, options)

    def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return self._do_generate(self._prepare_messages(messages), options)

    def _do_generate(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("openai package (>=1.0.0) with AzureOpenAI not installed")

        model = options.get("model", self.model)
        model_config = self._get_model_config("azure", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        # Build request kwargs
        kwargs = {
            "model": self.deployment_id,  # for Azure, use deployment name
            "messages": messages,
        }
        kwargs[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]

        # Native JSON mode support
        if options.get("json_mode"):
            json_schema = options.get("json_schema")
            if json_schema:
                schema_copy = prepare_strict_schema(json_schema)
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "strict": True,
                        "schema": schema_copy,
                    },
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)

        # Extract usage
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        # Calculate cost via shared mixin
        total_cost = self._calculate_cost("azure", model, prompt_tokens, completion_tokens)

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
