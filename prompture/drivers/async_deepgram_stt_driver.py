"""Async Deepgram STT driver."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import httpx
except Exception:
    httpx = None  # type: ignore[assignment]

from .async_stt_base import AsyncSTTDriver

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.deepgram.com"


class AsyncDeepgramSTTDriver(AsyncSTTDriver):
    """Async speech-to-text via Deepgram REST API."""

    supports_timestamps = True
    supports_language_detection = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "nova-3",
        endpoint: str = _DEFAULT_ENDPOINT,
    ):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        self.model = model
        self.endpoint = endpoint.rstrip("/")

    async def transcribe(self, audio: bytes, options: dict[str, Any]) -> dict[str, Any]:
        if httpx is None:
            raise RuntimeError("httpx package is not installed")
        if not self.api_key:
            raise RuntimeError("Deepgram API key is required (set DEEPGRAM_API_KEY).")

        model_id = options.get("model_id", self.model)
        params: dict[str, str] = {"model": model_id}
        if options.get("smart_format", True):
            params["smart_format"] = "true"
        if "language" in options:
            params["language"] = options["language"]
        if options.get("diarize"):
            params["diarize"] = "true"
        if options.get("punctuate"):
            params["punctuate"] = "true"

        content_type = options.get("content_type", "audio/mpeg")
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": content_type,
        }
        url = f"{self.endpoint}/v1/listen"

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, content=audio, headers=headers, params=params, timeout=120)
        if resp.status_code >= 400:
            raise RuntimeError(f"Deepgram STT request failed [{resp.status_code}]: {resp.text}")

        result = resp.json()
        try:
            alt = result["results"]["channels"][0]["alternatives"][0]
            text = alt.get("transcript", "")
        except (KeyError, IndexError, TypeError):
            text = ""

        duration = 0.0
        try:
            duration = float(result.get("metadata", {}).get("duration", 0.0))
        except (TypeError, ValueError):
            duration = 0.0

        return {
            "text": text,
            "segments": [],
            "language": options.get("language"),
            "meta": {
                "duration_seconds": duration,
                "cost": 0.0,
                "pricing_unknown": True,
                "model_name": f"deepgram/{model_id}",
                "raw_response": result,
            },
        }
