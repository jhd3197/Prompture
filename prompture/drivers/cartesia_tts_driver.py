"""Cartesia TTS driver. Uses httpx for the Sonic REST API.

Cartesia (https://cartesia.ai/) provides ultra-low-latency text-to-speech
via the Sonic family of models.  This driver targets the synchronous
``POST /tts/bytes`` endpoint which returns raw audio bytes.

A voice ID is required for every request — Cartesia does not have a
default voice.  Pass ``voice_id`` in ``options`` or set it via the
constructor.
"""

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

# Cartesia API version pin (per docs, recommended to send explicit version).
_CARTESIA_VERSION = "2025-04-16"
_CARTESIA_DEFAULT_ENDPOINT = "https://api.cartesia.ai"


def _media_type_for(container: str) -> str:
    container = (container or "").lower()
    if container == "mp3":
        return "audio/mpeg"
    if container == "wav":
        return "audio/wav"
    if container == "raw":
        return "audio/pcm"
    return "application/octet-stream"


class CartesiaTTSDriver(TTSDriver):
    """Text-to-speech via Cartesia REST API (Sonic family)."""

    supports_streaming = False
    supports_ssml = False
    available_voices = []  # Cartesia exposes a voice library via API; not enumerated here.

    # Known Sonic-family models (used by discovery; not authoritative).
    KNOWN_MODELS = [
        "sonic-2",
        "sonic-2-2025-03-07",
        "sonic-turbo",
        "sonic-english",
        "sonic-multilingual",
    ]

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

    def _build_request(self, text: str, options: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, Any]]:
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

        # Allow callers to pass arbitrary extra Cartesia fields (e.g. ``speed``).
        for extra_key in ("speed", "emotion", "duration", "experimental_controls"):
            if extra_key in options:
                body[extra_key] = options[extra_key]

        headers = {
            "X-API-Key": self.api_key,
            "Cartesia-Version": _CARTESIA_VERSION,
            "Content-Type": "application/json",
        }

        url = f"{self.endpoint}/tts/bytes"
        return url, headers, body

    def synthesize(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        """Synthesize text to audio via Cartesia ``POST /tts/bytes``.

        Args:
            text: Text to convert to speech.
            options: Supports ``voice_id`` (required), ``model_id``,
                ``language`` (default ``"en"``), ``output_format`` dict
                (``container``, ``encoding``, ``sample_rate``).
        """
        if httpx is None:
            raise RuntimeError("httpx package is not installed")

        url, headers, body = self._build_request(text, options)
        resp = httpx.post(url, json=body, headers=headers, timeout=60)
        if resp.status_code >= 400:
            raise RuntimeError(f"Cartesia TTS request failed [{resp.status_code}]: {resp.text}")

        audio_bytes = resp.content
        container = body["output_format"].get("container", "wav") if isinstance(body["output_format"], dict) else "wav"
        media_type = _media_type_for(container)

        return {
            "audio": audio_bytes,
            "media_type": media_type,
            "meta": {
                "characters": len(text),
                "cost": 0.0,
                "pricing_unknown": True,
                "model_name": f"cartesia/{body['model_id']}",
                "voice_id": body["voice"]["id"],
                "raw_response": {},
            },
        }
