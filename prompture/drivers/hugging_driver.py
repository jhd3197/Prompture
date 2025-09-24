import os
import requests
from ..driver import Driver
from typing import Any, Dict


class HuggingFaceDriver(Driver):
    # Hugging Face is usage-based (credits/subscription), but we set costs to 0 for now.
    MODEL_PRICING = {
        "default": {"prompt": 0.0, "completion": 0.0}
    }

    def __init__(self, endpoint: str | None = None, token: str | None = None, model: str = "bert-base-uncased"):
        self.endpoint = endpoint or os.getenv("HF_ENDPOINT")
        self.token = token or os.getenv("HF_TOKEN")
        self.model = model

        if not self.endpoint:
            raise ValueError("Hugging Face endpoint is not configured. Set HF_ENDPOINT or pass explicitly.")
        if not self.token:
            raise ValueError("Hugging Face token is not configured. Set HF_TOKEN or pass explicitly.")

        self.headers = {"Authorization": f"Bearer {self.token}"}

    def generate(self, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "inputs": prompt,
            "parameters": options,  # HF allows temperature, max_new_tokens, etc. here
        }
        try:
            r = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=options.get("timeout", 60))
            r.raise_for_status()
            response_data = r.json()
        except Exception as e:
            raise RuntimeError(f"HuggingFaceDriver request failed: {e}")

        # Different HF models return slightly different response formats
        # Text-generation models usually return [{"generated_text": "..."}]
        text = None
        if isinstance(response_data, list) and "generated_text" in response_data[0]:
            text = response_data[0]["generated_text"]
        elif isinstance(response_data, dict) and "generated_text" in response_data:
            text = response_data["generated_text"]
        else:
            # fallback to raw JSON
            text = str(response_data)

        # HF API does not return token counts unless self-hosted with TGI
        meta = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,  # assume free / covered by subscription
            "raw_response": response_data,
            "model_name": options.get("model", self.model),
        }

        return {"text": text, "meta": meta}
