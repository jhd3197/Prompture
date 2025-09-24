import os
import requests
from ..driver import Driver
from typing import Any, Dict


class LocalHTTPDriver(Driver):
    # Default: no cost; extend if your local service has pricing logic
    MODEL_PRICING = {
        "default": {"prompt": 0.0, "completion": 0.0}
    }

    def __init__(self, endpoint: str | None = None, model: str = "local-model"):
        self.endpoint = endpoint or os.getenv("LOCAL_HTTP_ENDPOINT", "http://localhost:8000/generate")
        self.model = model

    def generate(self, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"prompt": prompt, "options": options}
        try:
            r = requests.post(self.endpoint, json=payload, timeout=options.get("timeout", 30))
            r.raise_for_status()
            response_data = r.json()
        except Exception as e:
            raise RuntimeError(f"LocalHTTPDriver request failed: {e}")

        # If the local API already provides {"text": "...", "meta": {...}}, just return it
        if "text" in response_data and "meta" in response_data:
            return response_data

        # Otherwise, normalize the response
        meta = {
            "prompt_tokens": response_data.get("prompt_tokens", 0),
            "completion_tokens": response_data.get("completion_tokens", 0),
            "total_tokens": response_data.get("total_tokens", 0),
            "cost": 0.0,  # Local service assumed free
            "raw_response": response_data,
            "model_name": options.get("model", self.model),
        }

        text = response_data.get("text") or response_data.get("response") or str(response_data)
        return {"text": text, "meta": meta}
