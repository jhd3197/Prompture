"""OpenAI TTS driver. Requires the ``openai`` package (>=1.0.0)."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from ..infra.cost_mixin import AudioCostMixin
from .tts_base import TTSDriver

logger = logging.getLogger(__name__)

# MIME types for TTS output formats
_FORMAT_MIME: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


class OpenAITTSDriver(AudioCostMixin, TTSDriver):
    """Text-to-speech via OpenAI TTS API."""

    supports_streaming = True
    supports_ssml = False
    available_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    # Pricing: tts-1 $15/1M chars, tts-1-hd $30/1M chars
    AUDIO_PRICING = {
        "tts-1": {"per_character": 0.000015},
        "tts-1-hd": {"per_character": 0.000030},
    }

    def __init__(self, api_key: str | None = None, model: str = "tts-1"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if OpenAI:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def synthesize(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        """Synthesize text to audio using OpenAI TTS API.

        Args:
            text: Text to convert to speech.
            options: Supports ``voice`` (alloy/echo/fable/onyx/nova/shimmer),
                     ``format`` (mp3/opus/aac/flac/wav/pcm), ``speed`` (0.25-4.0),
                     ``model``.
        """
        if self.client is None:
            raise RuntimeError("openai package (>=1.0.0) is not installed")

        model = options.get("model", self.model)
        voice = options.get("voice", "alloy")
        output_format = options.get("format", "mp3")
        speed = options.get("speed", 1.0)

        kwargs: dict[str, Any] = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": output_format,
            "speed": speed,
        }

        resp = self.client.audio.speech.create(**kwargs)
        audio_bytes = resp.content

        media_type = _FORMAT_MIME.get(output_format, "audio/mpeg")
        characters = len(text)
        cost = self._calculate_audio_cost("openai", model, characters=characters)

        return {
            "audio": audio_bytes,
            "media_type": media_type,
            "meta": {
                "characters": characters,
                "cost": round(cost, 6),
                "model_name": f"openai/{model}",
                "voice": voice,
                "raw_response": {},
            },
        }

    def synthesize_stream(self, text: str, options: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Stream audio chunks from OpenAI TTS API."""
        if self.client is None:
            raise RuntimeError("openai package (>=1.0.0) is not installed")

        model = options.get("model", self.model)
        voice = options.get("voice", "alloy")
        output_format = options.get("format", "mp3")
        speed = options.get("speed", 1.0)

        kwargs: dict[str, Any] = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": output_format,
            "speed": speed,
        }

        media_type = _FORMAT_MIME.get(output_format, "audio/mpeg")

        with self.client.audio.speech.with_streaming_response.create(**kwargs) as resp:
            full_audio = b""
            for chunk in resp.iter_bytes(chunk_size=4096):
                full_audio += chunk
                yield {"type": "delta", "audio": chunk, "media_type": media_type}

        characters = len(text)
        cost = self._calculate_audio_cost("openai", model, characters=characters)

        yield {
            "type": "done",
            "audio": full_audio,
            "media_type": media_type,
            "meta": {
                "characters": characters,
                "cost": round(cost, 6),
                "model_name": f"openai/{model}",
                "voice": voice,
                "raw_response": {},
            },
        }
