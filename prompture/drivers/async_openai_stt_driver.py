"""Async OpenAI Whisper STT driver. Requires the ``openai`` package (>=1.0.0)."""

from __future__ import annotations

import io
import logging
import os
from typing import Any

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None  # type: ignore[misc, assignment]

from ..infra.cost_mixin import AudioCostMixin
from .async_stt_base import AsyncSTTDriver
from .openai_stt_driver import OpenAISTTDriver

logger = logging.getLogger(__name__)


class AsyncOpenAISTTDriver(AudioCostMixin, AsyncSTTDriver):
    """Async speech-to-text via OpenAI Whisper API."""

    supports_timestamps = True
    supports_language_detection = True

    AUDIO_PRICING = OpenAISTTDriver.AUDIO_PRICING

    def __init__(self, api_key: str | None = None, model: str = "whisper-1"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if AsyncOpenAI is not None:
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None

    async def transcribe(self, audio: bytes, options: dict[str, Any]) -> dict[str, Any]:
        """Transcribe audio using OpenAI Whisper API (async).

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

        response_format = options.get("response_format", "verbose_json")
        kwargs["response_format"] = response_format

        resp = await self.client.audio.transcriptions.create(**kwargs)

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
            text = str(resp)

        cost = self._calculate_audio_cost("openai", model, duration_seconds=duration_seconds)

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
