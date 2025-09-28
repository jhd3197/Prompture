import os
import json
import requests
import logging
from ..driver import Driver
from typing import Any, Dict

logger = logging.getLogger(__name__)


class OllamaDriver(Driver):
    # Ollama is free – costs are always zero.
    MODEL_PRICING = {
        "default": {"prompt": 0.0, "completion": 0.0}
    }

    def __init__(self, endpoint: str | None = None, model: str = "llama3"):
        # Allow override via env var
        self.endpoint = endpoint or os.getenv(
            "OLLAMA_ENDPOINT", "http://localhost:11434/api/generate"
        )
        self.model = model
        self.options = {}  # Initialize empty options dict
        
        # Validate connection to Ollama server
        self._validate_connection()
        
    def _validate_connection(self):
        """Validate connection to the Ollama server."""
        try:
            # Send a simple HEAD request to check if server is accessible
            # Use the base API endpoint without the specific path
            base_url = self.endpoint.split('/api/')[0]
            health_url = f"{base_url}/api/version"
            
            logger.debug(f"Validating connection to Ollama server at: {health_url}")
            response = requests.head(health_url, timeout=5)
            response.raise_for_status()
            logger.debug("Connection to Ollama server validated successfully")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not validate connection to Ollama server: {e}")
            # We don't raise an error here to allow for delayed server startup
            # The actual error will be raised when generate() is called

    def generate(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        # Merge instance options with call-specific options
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        payload = {
            "prompt": prompt,
            "model": merged_options.get("model", self.model),
            "stream": False,
        }

        # Add any Ollama-specific options from merged_options
        if "temperature" in merged_options:
            payload["temperature"] = merged_options["temperature"]
        if "top_p" in merged_options:
            payload["top_p"] = merged_options["top_p"]
        if "top_k" in merged_options:
            payload["top_k"] = merged_options["top_k"]

        try:
            logger.debug(f"Sending request to Ollama endpoint: {self.endpoint}")
            logger.debug(f"Request payload: {payload}")
            
            r = requests.post(self.endpoint, json=payload, timeout=120)
            logger.debug(f"Response status code: {r.status_code}")
            
            r.raise_for_status()
            
            response_text = r.text
            logger.debug(f"Raw response text: {response_text}")
            
            response_data = r.json()
            logger.debug(f"Parsed response data: {response_data}")
            
            if not isinstance(response_data, dict):
                raise ValueError(f"Expected dict response, got {type(response_data)}")
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to Ollama endpoint: {e}")
            # Preserve original exception
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from Ollama endpoint: {e}")
            # Preserve original exception
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            # Re-raise JSONDecodeError with more context
            raise json.JSONDecodeError(f"Invalid JSON response from Ollama: {e.msg}", e.doc, e.pos)
        except Exception as e:
            logger.error(f"Unexpected error in Ollama request: {e}")
            # Only wrap unknown exceptions in RuntimeError
            raise RuntimeError(f"Ollama request failed: {e}")

        # Extract token counts
        prompt_tokens = response_data.get("prompt_eval_count", 0)
        completion_tokens = response_data.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        # Build meta info
        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": 0.0,
            "raw_response": response_data,
            "model_name": merged_options.get("model", self.model),
        }

        # Ollama returns text in "response"
        return {"text": response_data.get("response", ""), "meta": meta}