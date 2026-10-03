"""Groq Whisper speech-to-text driver.

Groq serves Whisper through an OpenAI-compatible ``/audio/transcriptions``
endpoint, so this reuses :class:`~prompture.drivers.openai_stt_driver.OpenAISTTDriver`
with Groq's base URL, model ids and pricing. :func:`ensure_groq_stt_registered`
adds it to the STT registry as ``groq`` (without overwriting a driver someone
else registered), so ``get_stt_driver_for_model("groq/whisper-large-v3-turbo")``
works once media understanding is imported.
"""

from __future__ import annotations

import os
from typing import Any

from ...drivers.openai_stt_driver import OpenAI, OpenAISTTDriver

GROQ_STT_ENDPOINT = "https://api.groq.com/openai/v1"
GROQ_STT_DEFAULT_MODEL = "whisper-large-v3-turbo"


class GroqSTTDriver(OpenAISTTDriver):
    """Speech-to-text via Groq-hosted Whisper (OpenAI-compatible API)."""

    # USD per audio second (Groq lists per-hour prices: $0.04, $0.111, $0.02).
    AUDIO_PRICING = {
        "whisper-large-v3-turbo": {"per_second": 0.04 / 3600},
        "whisper-large-v3": {"per_second": 0.111 / 3600},
        "distil-whisper-large-v3-en": {"per_second": 0.02 / 3600},
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = GROQ_STT_DEFAULT_MODEL,
        endpoint: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or GROQ_STT_DEFAULT_MODEL
        self.endpoint = (endpoint or os.getenv("GROQ_STT_ENDPOINT") or GROQ_STT_ENDPOINT).rstrip("/")
        if OpenAI is not None:
            self.client = OpenAI(api_key=self.api_key, base_url=self.endpoint)
        else:
            self.client = None

    def transcribe(self, audio: bytes, options: dict[str, Any]) -> dict[str, Any]:
        resp = super().transcribe(audio, options)
        meta = resp.setdefault("meta", {})
        model = options.get("model", self.model)
        meta["model_name"] = f"groq/{model}"
        meta["cost"] = round(
            self._calculate_audio_cost("groq", model, duration_seconds=float(meta.get("duration_seconds") or 0)),
            6,
        )
        return resp


def _groq_factory(model: str | None = None) -> GroqSTTDriver:
    from ...infra.settings import settings

    return GroqSTTDriver(api_key=getattr(settings, "groq_api_key", None), model=model or GROQ_STT_DEFAULT_MODEL)


def ensure_groq_stt_registered() -> None:
    """Register ``groq`` in the STT driver registry unless something already is."""
    from ...drivers.registry import is_stt_driver_registered, register_stt_driver

    try:
        if not is_stt_driver_registered("groq"):
            register_stt_driver("groq", _groq_factory)
    except Exception:  # pragma: no cover - registry races are harmless here
        pass
