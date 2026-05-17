"""Async Cartesia TTS driver. Uses httpx.AsyncClient."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import httpx
except Exception:
    httpx = None  # type: ignore[assignment]

from .async_tts_base import AsyncTTSDriver
from .cartesia_tts_driver import (
    _CARTESIA_DEFAULT_ENDPOINT,
    _CARTESIA_VERSION,
    _media_type_for,
)

logger = logging.getLogger(__name__)


class AsyncCartesiaTTSDriver(AsyncTTSDriver):
    """Async text-to-speech via Cartesia REST API."""

    supports_streaming = False
    supports_ssml = False
    available_voices = []

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "sonic-2",
        endpoint: str = _CARTESIA_DEFAULT_ENDPOINT,
        voice_id: str | None = None,
    ):
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY")
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.voice_id = voice_id

    async def synthesize(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        """Synthesize text to audio (async) via Cartesia ``POST /tts/bytes``."""
        if httpx is None:
            raise RuntimeError("httpx package is not installed")
        if not self.api_key:
            raise RuntimeError("Cartesia API key is required (set CARTESIA_API_KEY).")

        voice_id = options.get("voice_id") or self.voice_id
        if not voice_id:
            raise RuntimeError("Cartesia requires a voice_id. Pass options['voice_id'] or set it on the driver.")

        model_id = options.get("model_id", self.model)
        language = options.get("language", "en")
        output_format = options.get(
            "output_format",
            {"container": "wav", "encoding": "pcm_f32le", "sample_rate": 44100},
        )

        body: dict[str, Any] = {
            "model_id": model_id,
            "transcript": text,
            "voice": {"mode": "id", "id": voice_id},
            "output_format": output_format,
            "language": language,
        }
        for extra_key in ("speed", "emotion", "duration", "experimental_controls"):
            if extra_key in options:
                body[extra_key] = options[extra_key]

        headers = {
            "X-API-Key": self.api_key,
            "Cartesia-Version": _CARTESIA_VERSION,
            "Content-Type": "application/json",
        }
        url = f"{self.endpoint}/tts/bytes"

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=body, headers=headers, timeout=60)
        if resp.status_code >= 400:
            raise RuntimeError(f"Cartesia TTS request failed [{resp.status_code}]: {resp.text}")

        audio_bytes = resp.content
        container = output_format.get("container", "wav") if isinstance(output_format, dict) else "wav"
        media_type = _media_type_for(container)

        return {
            "audio": audio_bytes,
            "media_type": media_type,
            "meta": {
                "characters": len(text),
                "cost": 0.0,
                "pricing_unknown": True,
                "model_name": f"cartesia/{model_id}",
                "voice_id": voice_id,
                "raw_response": {},
            },
        }
