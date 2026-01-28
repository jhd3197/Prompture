"""Async Hugging Face driver using httpx."""

from __future__ import annotations

import os
from typing import Any

import httpx

from ..async_driver import AsyncDriver


class AsyncHuggingFaceDriver(AsyncDriver):
    MODEL_PRICING = {"default": {"prompt": 0.0, "completion": 0.0}}

    def __init__(self, endpoint: str | None = None, token: str | None = None, model: str = "bert-base-uncased"):
        self.endpoint = endpoint or os.getenv("HF_ENDPOINT")
        self.token = token or os.getenv("HF_TOKEN")
        self.model = model

        if not self.endpoint:
            raise ValueError("Hugging Face endpoint is not configured. Set HF_ENDPOINT or pass explicitly.")
        if not self.token:
            raise ValueError("Hugging Face token is not configured. Set HF_TOKEN or pass explicitly.")

        self.headers = {"Authorization": f"Bearer {self.token}"}

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "inputs": prompt,
            "parameters": options,
        }

        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(
                    self.endpoint, headers=self.headers, json=payload, timeout=options.get("timeout", 60)
                )
                r.raise_for_status()
                response_data = r.json()
            except Exception as e:
                raise RuntimeError(f"AsyncHuggingFaceDriver request failed: {e}") from e

        text = None
        if isinstance(response_data, list) and response_data and "generated_text" in response_data[0]:
            text = response_data[0]["generated_text"]
        elif isinstance(response_data, dict) and "generated_text" in response_data:
            text = response_data["generated_text"]
        else:
            text = str(response_data)

        meta = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "raw_response": response_data,
            "model_name": options.get("model", self.model),
        }

        return {"text": text, "meta": meta}
