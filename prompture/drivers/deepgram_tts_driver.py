"""Deepgram TTS driver (Aura family). Uses httpx for the REST API."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from .tts_base import TTSDriver

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.deepgram.com"


def _media_type_for(encoding: str) -> str:
    encoding = (encoding or "").lower()
    if encoding == "mp3":
        return "audio/mpeg"
    if encoding == "wav":
        return "audio/wav"
    if encoding.startswith("linear") or encoding.startswith("pcm"):
        return "audio/pcm"
    if encoding == "opus":
        return "audio/opus"
    if encoding == "flac":
        return "audio/flac"
    return "application/octet-stream"


class DeepgramTTSDriver(TTSDriver):
    """Text-to-speech via Deepgram REST API (Aura family)."""

    supports_streaming = False
    supports_ssml = False
    available_voices = []

    KNOWN_MODELS = [
        "aura-2-thalia-en",
        "aura-2-arcas-en",
        "aura-2-perseus-en",
        "aura-asteria-en",
        "aura-luna-en",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "aura-2-thalia-en",
        endpoint: str = _DEFAULT_ENDPOINT,
    ):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        self.model = model
        self.endpoint = endpoint.rstrip("/")

    def synthesize(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        """Synthesize text via Deepgram ``POST /v1/speak``.

        Args:
            text: Text to convert to speech.
            options: Supports ``model_id``, ``encoding`` (default ``"mp3"``),
                ``sample_rate``, ``container``.
        """
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

        resp = httpx.post(url, json=body, headers=headers, params=params, timeout=60)
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
