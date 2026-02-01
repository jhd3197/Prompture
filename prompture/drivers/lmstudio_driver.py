import json
import logging
import os
from typing import Any, Optional

import requests

from ..driver import Driver

logger = logging.getLogger(__name__)


class LMStudioDriver(Driver):
    supports_json_mode = True
    supports_json_schema = True
    supports_vision = True

    # LM Studio is local – costs are always zero.
    MODEL_PRICING = {"default": {"prompt": 0.0, "completion": 0.0}}

    def __init__(
        self,
        endpoint: str | None = None,
        model: str = "deepseek/deepseek-r1-0528-qwen3-8b",
        api_key: str | None = None,
    ):
        # Allow override via env var
        self.endpoint = endpoint or os.getenv("LMSTUDIO_ENDPOINT", "http://127.0.0.1:1234/v1/chat/completions")
        self.model = model
        self.options: dict[str, Any] = {}

        # Derive base_url once for reuse across management endpoints
        self.base_url = self.endpoint.split("/v1/")[0]

        # API key for LM Studio 0.4.0+ authentication
        self.api_key = api_key or os.getenv("LMSTUDIO_API_KEY")
        self._headers = self._build_headers()

        # Validate connection to LM Studio server
        self._validate_connection()

    def _build_headers(self) -> dict[str, str]:
        """Build request headers, including auth if an API key is configured."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _validate_connection(self):
        """Validate connection to the LM Studio server."""
        try:
            health_url = f"{self.base_url}/v1/models"

            logger.debug(f"Validating connection to LM Studio server at: {health_url}")
            response = requests.get(health_url, headers=self._headers, timeout=5)
            response.raise_for_status()
            logger.debug("Connection to LM Studio server validated successfully")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not validate connection to LM Studio server: {e}")

    supports_messages = True

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_openai_vision_messages

        return _prepare_openai_vision_messages(messages)

    def generate(self, prompt: str, options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return self._do_generate(messages, options)

    def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return self._do_generate(self._prepare_messages(messages), options)

    def _do_generate(self, messages: list[dict[str, str]], options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        payload = {
            "model": merged_options.get("model", self.model),
            "messages": messages,
            "temperature": merged_options.get("temperature", 0.7),
        }

        # Native JSON mode support (LM Studio requires json_schema, not json_object)
        if merged_options.get("json_mode"):
            json_schema = merged_options.get("json_schema")
            if json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "schema": json_schema,
                    },
                }
            else:
                # No schema provided — omit response_format entirely;
                # LM Studio rejects "json_object" type.
                pass

        try:
            logger.debug(f"Sending request to LM Studio endpoint: {self.endpoint}")
            logger.debug(f"Request payload: {payload}")

            r = requests.post(self.endpoint, json=payload, headers=self._headers, timeout=120)
            r.raise_for_status()

            response_data = r.json()
            logger.debug(f"Parsed response data: {response_data}")

            if "choices" not in response_data or not response_data["choices"]:
                raise ValueError(f"Unexpected response format: {response_data}")

        except requests.exceptions.RequestException as e:
            logger.error(f"LM Studio request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from LM Studio: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LM Studio request: {e}")
            raise RuntimeError(f"LM Studio request failed: {e}") from e

        # Extract text
        message = response_data["choices"][0]["message"]
        text = message.get("content") or ""
        reasoning_content = message.get("reasoning_content")

        # Reasoning models (e.g. DeepSeek R1) may return content in reasoning_content
        if not text and reasoning_content:
            text = reasoning_content

        # Meta info
        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": 0.0,
            "raw_response": response_data,
            "model_name": merged_options.get("model", self.model),
        }

        result: dict[str, Any] = {"text": text, "meta": meta}
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result

    # -- Model management (LM Studio 0.4.0+) ----------------------------------

    def list_models(self) -> list[dict[str, Any]]:
        """List currently loaded models via GET /v1/models (OpenAI-compatible)."""
        url = f"{self.base_url}/v1/models"
        r = requests.get(url, headers=self._headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("data", [])

    def load_model(self, model: str, context_length: int | None = None) -> dict[str, Any]:
        """Load a model into LM Studio via POST /api/v1/models/load."""
        url = f"{self.base_url}/api/v1/models/load"
        payload: dict[str, Any] = {"model": model}
        if context_length is not None:
            payload["context_length"] = context_length
        r = requests.post(url, json=payload, headers=self._headers, timeout=120)
        r.raise_for_status()
        return r.json()

    def unload_model(self, model: str) -> dict[str, Any]:
        """Unload a model from LM Studio via POST /api/v1/models/unload."""
        url = f"{self.base_url}/api/v1/models/unload"
        payload = {"instance_id": model}
        r = requests.post(url, json=payload, headers=self._headers, timeout=30)
        r.raise_for_status()
        return r.json()
