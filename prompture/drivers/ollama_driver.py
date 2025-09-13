import requests
from ..core import Driver
from typing import Any, Dict

class OllamaDriver(Driver):
    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint
        self.model = model

    def generate(self, prompt: str, options: Dict[str,Any]) -> Dict[str,Any]:
        payload = {
            "prompt": prompt,
            "model": options.get("model", self.model),
            "options": options,
            "stream": False
        }
        r = requests.post(self.endpoint, json=payload, timeout=60)
        r.raise_for_status()
        response_data = r.json()

        # Extract token counts from Ollama API response
        prompt_tokens = response_data.get("prompt_eval_count", 0)
        completion_tokens = response_data.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        # Ollama is free to use, so cost is always 0.0
        cost = 0.0

        # Create standardized meta object
        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "raw_response": response_data,
            "model_name": options.get("model", self.model)
        }

        # Ollama returns the text in the "response" key
        return {"text": response_data.get("response", ""), "meta": meta}