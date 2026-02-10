"""ElevenLabs STT driver. Uses httpx for the REST API."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import httpx
except Exception:
    httpx = None

from ..infra.cost_mixin import AudioCostMixin
from .stt_base import STTDriver

logger = logging.getLogger(__name__)


class ElevenLabsSTTDriver(AudioCostMixin, STTDriver):
    """Speech-to-text via ElevenLabs REST API."""

    supports_timestamps = False
    supports_language_detection = True

    AUDIO_PRICING: dict[str, dict[str, float]] = {}  # ElevenLabs STT pricing TBD

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "scribe_v1",
        endpoint: str = "https://api.elevenlabs.io/v1",
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.model = model
        self.endpoint = endpoint.rstrip("/")

    def transcribe(self, audio: bytes, options: dict[str, Any]) -> dict[str, Any]:
        """Transcribe audio using ElevenLabs Speech-to-Text API.

        Args:
            audio: Raw audio bytes.
            options: Supports ``language_code``, ``tag_audio_events`` (bool),
                     ``filename`` (default ``"audio.mp3"``).
        """
        if httpx is None:
            raise RuntimeError("httpx package is not installed")
        if not self.api_key:
            raise RuntimeError("ElevenLabs API key is required")

        model_id = options.get("model_id", self.model)
        filename = options.get("filename", "audio.mp3")

        url = f"{self.endpoint}/speech-to-text"

        headers = {
            "xi-api-key": self.api_key,
        }

        files = {
            "audio": (filename, audio, "audio/mpeg"),
        }
        data: dict[str, str] = {
            "model_id": model_id,
        }
        if "language_code" in options:
            data["language_code"] = options["language_code"]
        if options.get("tag_audio_events"):
            data["tag_audio_events"] = "true"

        resp = httpx.post(url, headers=headers, files=files, data=data, timeout=120)
        resp.raise_for_status()

        result = resp.json()

        text = result.get("text", "")
        language = result.get("language_code")

        return {
            "text": text,
            "segments": [],
            "language": language,
            "meta": {
                "duration_seconds": 0,
                "cost": 0.0,
                "model_name": f"elevenlabs/{model_id}",
                "raw_response": result,
            },
        }
