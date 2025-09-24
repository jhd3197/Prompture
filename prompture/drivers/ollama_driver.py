import os
import requests
from ..driver import Driver
from typing import Any, Dict


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

    def generate(self, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "prompt": prompt,
            "model": options.get("model", self.model),
            "options": options,
            "stream": False,
        }
        try:
            r = requests.post(self.endpoint, json=payload, timeout=60)
            r.raise_for_status()
            response_data = r.json()
        except Exception as e:
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
            "model_name": options.get("model", self.model),
        }

        # Ollama returns text in "response"
        return {"text": response_data.get("response", ""), "meta": meta}
