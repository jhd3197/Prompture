"""Async Deepgram TTS driver."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import httpx
except Exception:
    httpx = None  # type: ignore[assignment]

from .async_tts_base import AsyncTTSDriver
from .deepgram_tts_driver import _media_type_for

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.deepgram.com"


class AsyncDeepgramTTSDriver(AsyncTTSDriver):
    """Async text-to-speech via Deepgram REST API (Aura family)."""

    supports_streaming = False
    supports_ssml = False
    available_voices = []

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "aura-2-thalia-en",
        endpoint: str = _DEFAULT_ENDPOINT,
    ):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        self.model = model
        self.endpoint = endpoint.rstrip("/")

    async def synthesize(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        if httpx is None:
            raise RuntimeError("httpx package is not installed")
        if not self.api_key:
            raise RuntimeError("Deepgram API key is required (set DEEPGRAM_API_KEY).")

        model_id = options.get("model_id", self.model)
        encoding = options.get("encoding", "mp3")

        params: dict[str, str] = {"model": model_id, "encoding": encoding}
        if "sample_rate" in options:
            params["sample_rate"] = str(options["sample_rate"])
        if "container" in options:
            params["container"] = options["container"]

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.endpoint}/v1/speak"
        body = {"text": text}

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=body, headers=headers, params=params, timeout=60)
        if resp.status_code >= 400:
            raise RuntimeError(f"Deepgram TTS request failed [{resp.status_code}]: {resp.text}")

        audio_bytes = resp.content
        media_type = _media_type_for(encoding)

        return {
            "audio": audio_bytes,
            "media_type": media_type,
            "meta": {
                "characters": len(text),
                "cost": 0.0,
                "pricing_unknown": True,
                "model_name": f"deepgram/{model_id}",
                "raw_response": {},
            },
        }
