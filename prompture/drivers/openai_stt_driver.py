"""OpenAI Whisper STT driver. Requires the ``openai`` package (>=1.0.0)."""

from __future__ import annotations

import io
import logging
import os
from typing import Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from ..infra.cost_mixin import AudioCostMixin
from .stt_base import STTDriver

logger = logging.getLogger(__name__)


class OpenAISTTDriver(AudioCostMixin, STTDriver):
    """Speech-to-text via OpenAI Whisper API."""

    supports_timestamps = True
    supports_language_detection = True

    # Pricing: $0.006 per minute = $0.0001 per second
    AUDIO_PRICING = {
        "whisper-1": {"per_second": 0.0001},
    }

    def __init__(self, api_key: str | None = None, model: str = "whisper-1"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if OpenAI:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def transcribe(self, audio: bytes, options: dict[str, Any]) -> dict[str, Any]:
        """Transcribe audio using OpenAI Whisper API.

        Args:
            audio: Raw audio bytes.
            options: Supports ``language``, ``response_format``
                     (json/text/verbose_json/srt/vtt), ``temperature``,
                     ``filename`` (default ``"audio.mp3"``).
        """
        if self.client is None:
            raise RuntimeError("openai package (>=1.0.0) is not installed")

        model = options.get("model", self.model)
        filename = options.get("filename", "audio.mp3")

        kwargs: dict[str, Any] = {
            "model": model,
            "file": (filename, io.BytesIO(audio)),
        }

        if "language" in options:
            kwargs["language"] = options["language"]
        if "temperature" in options:
            kwargs["temperature"] = options["temperature"]

        # Use verbose_json to get segments and duration
        response_format = options.get("response_format", "verbose_json")
        kwargs["response_format"] = response_format

        resp = self.client.audio.transcriptions.create(**kwargs)

        # Parse response based on format
        text = ""
        segments: list[dict[str, Any]] = []
        language: str | None = None
        duration_seconds: float = 0
        raw_response: dict[str, Any] = {}

        if response_format == "verbose_json":
            if hasattr(resp, "text"):
                text = resp.text
            if hasattr(resp, "segments") and resp.segments:
                segments = [
                    {
                        "start": getattr(s, "start", 0),
                        "end": getattr(s, "end", 0),
                        "text": getattr(s, "text", ""),
                    }
                    for s in resp.segments
                ]
            if hasattr(resp, "language"):
                language = resp.language
            if hasattr(resp, "duration"):
                duration_seconds = float(resp.duration)
            if hasattr(resp, "model_dump"):
                raw_response = resp.model_dump()
        elif response_format == "json":
            if hasattr(resp, "text"):
                text = resp.text
            if hasattr(resp, "model_dump"):
                raw_response = resp.model_dump()
        else:
            # text, srt, vtt formats return plain strings
            text = str(resp)

        cost = self._calculate_audio_cost(
            "openai", model, duration_seconds=duration_seconds
        )

        return {
            "text": text,
            "segments": segments,
            "language": language,
            "meta": {
                "duration_seconds": duration_seconds,
                "cost": round(cost, 6),
                "model_name": f"openai/{model}",
                "raw_response": raw_response,
            },
        }
