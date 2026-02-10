"""Async ElevenLabs TTS driver. Uses httpx for the REST API."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

try:
    import httpx
except Exception:
    httpx = None

from ..infra.cost_mixin import AudioCostMixin
from .async_tts_base import AsyncTTSDriver
from .elevenlabs_tts_driver import ElevenLabsTTSDriver

logger = logging.getLogger(__name__)


class AsyncElevenLabsTTSDriver(AudioCostMixin, AsyncTTSDriver):
    """Async text-to-speech via ElevenLabs REST API."""

    supports_streaming = True
    supports_ssml = False
    available_voices = []

    AUDIO_PRICING = ElevenLabsTTSDriver.AUDIO_PRICING

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "eleven_multilingual_v2",
        endpoint: str = "https://api.elevenlabs.io/v1",
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.model = model
        self.endpoint = endpoint.rstrip("/")

    async def synthesize(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        """Synthesize text to audio using ElevenLabs API (async).

        Args:
            text: Text to convert to speech.
            options: Supports ``voice_id`` (required), ``model_id``,
                     ``output_format``, ``voice_settings``.
        """
        if httpx is None:
            raise RuntimeError("httpx package is not installed")
        if not self.api_key:
            raise RuntimeError("ElevenLabs API key is required")

        voice_id = options.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
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

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=body, headers=headers, params=params, timeout=60)
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

    async def synthesize_stream(
        self, text: str, options: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream audio chunks from ElevenLabs API (async)."""
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
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", url, json=body, headers=headers, params=params, timeout=60
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(chunk_size=4096):
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
