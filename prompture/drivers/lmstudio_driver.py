import os
import json
import requests
import logging
from ..driver import Driver
from typing import Any, Dict

logger = logging.getLogger(__name__)


class LMStudioDriver(Driver):
    # LM Studio is local – costs are always zero.
    MODEL_PRICING = {
        "default": {"prompt": 0.0, "completion": 0.0}
    }

    def __init__(self, endpoint: str | None = None, model: str = "deepseek/deepseek-r1-0528-qwen3-8b"):
        # Allow override via env var
        self.endpoint = endpoint or os.getenv(
            "LMSTUDIO_ENDPOINT", "http://127.0.0.1:1234/v1/chat/completions"
        )
        self.model = model
        self.options: Dict[str, Any] = {}

        # Validate connection to LM Studio server
        self._validate_connection()

    def _validate_connection(self):
        """Validate connection to the LM Studio server."""
        try:
            base_url = self.endpoint.split("/v1/")[0]
            health_url = f"{base_url}/v1/models"

            logger.debug(f"Validating connection to LM Studio server at: {health_url}")
            response = requests.get(health_url, timeout=5)
            response.raise_for_status()
            logger.debug("Connection to LM Studio server validated successfully")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not validate connection to LM Studio server: {e}")

    def generate(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        payload = {
            "model": merged_options.get("model", self.model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": merged_options.get("temperature", 0.7),
        }

        try:
            logger.debug(f"Sending request to LM Studio endpoint: {self.endpoint}")
            logger.debug(f"Request payload: {payload}")

            r = requests.post(self.endpoint, json=payload, timeout=60)
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
            raise RuntimeError(f"LM Studio request failed: {e}")

        # Extract text
        text = response_data["choices"][0]["message"]["content"]

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

        return {"text": text, "meta": meta}
