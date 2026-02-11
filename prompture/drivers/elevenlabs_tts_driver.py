"""ElevenLabs TTS driver. Uses httpx for the REST API."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

try:
    import httpx
except Exception:
    httpx = None

from ..infra.cost_mixin import AudioCostMixin
from .tts_base import TTSDriver

logger = logging.getLogger(__name__)


class ElevenLabsTTSDriver(AudioCostMixin, TTSDriver):
    """Text-to-speech via ElevenLabs REST API."""

    supports_streaming = True
    supports_ssml = False
    available_voices = []  # Populated dynamically from API

    # Pricing varies by plan; these are pay-as-you-go estimates per character
    AUDIO_PRICING = {
        "eleven_multilingual_v2": {"per_character": 0.000030},
        "eleven_turbo_v2_5": {"per_character": 0.000030},
        "eleven_flash_v2_5": {"per_character": 0.000015},
    }

    @classmethod
    def list_models(
        cls,
        *,
        api_key: str | None = None,
        endpoint: str = "https://api.elevenlabs.io/v1",
        timeout: int = 10,
        **kw: object,
    ) -> list[str] | None:
        """List TTS models available from the ElevenLabs API.

        Calls ``GET /v1/models`` and returns model IDs where
        ``can_do_text_to_speech`` is ``True``.
        """
        if httpx is None:
            logger.debug("httpx not installed — cannot list ElevenLabs models")
            return None

        key = api_key or os.getenv("ELEVENLABS_API_KEY")
        headers: dict[str, str] = {}
        if key:
            headers["xi-api-key"] = key

        try:
            url = f"{endpoint.rstrip('/')}/models"
            resp = httpx.get(url, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                logger.debug("ElevenLabs /models returned %s", resp.status_code)
                return None

            models = resp.json()
            return [
                m["model_id"]
                for m in models
                if isinstance(m, dict) and m.get("can_do_text_to_speech")
            ]
        except Exception:
            logger.debug("ElevenLabs list_models failed", exc_info=True)
            return None

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "eleven_multilingual_v2",
        endpoint: str = "https://api.elevenlabs.io/v1",
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.model = model
        self.endpoint = endpoint.rstrip("/")

    def synthesize(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        """Synthesize text to audio using ElevenLabs API.

        Args:
            text: Text to convert to speech.
            options: Supports ``voice_id`` (required), ``model_id``,
                     ``output_format`` (mp3_44100_128, pcm_16000, etc.),
                     ``voice_settings`` dict (stability, similarity_boost).
        """
        if httpx is None:
            raise RuntimeError("httpx package is not installed")
        if not self.api_key:
            raise RuntimeError("ElevenLabs API key is required")

        voice_id = options.get("voice_id", "21m00Tcm4TlvDq8ikWAM")  # Rachel default
        model_id = options.get("model_id", self.model)
        output_format = options.get("output_format", "mp3_44100_128")

        url = f"{self.endpoint}/text-to-speech/{voice_id}"

        body: dict[str, Any] = {
            "text": text,
            "model_id": model_id,
        }
        if "voice_settings" in options:
            body["voice_settings"] = options["voice_settings"]

        params = {}
        if output_format:
            params["output_format"] = output_format

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        resp = httpx.post(url, json=body, headers=headers, params=params, timeout=60)
        resp.raise_for_status()

        audio_bytes = resp.content
        media_type = "audio/mpeg"
        if "pcm" in output_format:
            media_type = "audio/pcm"
        elif "wav" in output_format:
            media_type = "audio/wav"

        characters = len(text)
        cost = self._calculate_audio_cost("elevenlabs", model_id, characters=characters)

        return {
            "audio": audio_bytes,
            "media_type": media_type,
            "meta": {
                "characters": characters,
                "cost": round(cost, 6),
                "model_name": f"elevenlabs/{model_id}",
                "voice_id": voice_id,
                "raw_response": {},
            },
        }

    def synthesize_stream(self, text: str, options: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Stream audio chunks from ElevenLabs API."""
        if httpx is None:
            raise RuntimeError("httpx package is not installed")
        if not self.api_key:
            raise RuntimeError("ElevenLabs API key is required")

        voice_id = options.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        model_id = options.get("model_id", self.model)
        output_format = options.get("output_format", "mp3_44100_128")

        url = f"{self.endpoint}/text-to-speech/{voice_id}/stream"

        body: dict[str, Any] = {
            "text": text,
            "model_id": model_id,
        }
        if "voice_settings" in options:
            body["voice_settings"] = options["voice_settings"]

        params = {}
        if output_format:
            params["output_format"] = output_format

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        media_type = "audio/mpeg"
        if "pcm" in output_format:
            media_type = "audio/pcm"
        elif "wav" in output_format:
            media_type = "audio/wav"

        full_audio = b""
        with httpx.stream("POST", url, json=body, headers=headers, params=params, timeout=60) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes(chunk_size=4096):
                full_audio += chunk
                yield {"type": "delta", "audio": chunk, "media_type": media_type}

        characters = len(text)
        cost = self._calculate_audio_cost("elevenlabs", model_id, characters=characters)

        yield {
            "type": "done",
            "audio": full_audio,
            "media_type": media_type,
            "meta": {
                "characters": characters,
                "cost": round(cost, 6),
                "model_name": f"elevenlabs/{model_id}",
                "voice_id": voice_id,
                "raw_response": {},
            },
        }
