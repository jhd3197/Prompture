"""Runway TTS driver — text-to-speech and sound-effect generation.

Two endpoints share this driver:

- ``POST /v1/text_to_speech`` with ``eleven_multilingual_v2`` — turn text into speech.
- ``POST /v1/sound_effect``    with ``eleven_text_to_sound_v2`` — turn a prompt into a sound effect.

Both submit a task and return audio bytes once the task completes.
The endpoint is picked from the model: any ``eleven_text_to_sound_v2`` model
goes to ``/v1/sound_effect``; anything else goes to ``/v1/text_to_speech``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from ..infra.cost_mixin import AudioCostMixin
from .runway_img_gen_driver import (
    _API_VERSION,
    _DEFAULT_ENDPOINT,
    _TERMINAL_FAIL,
    _TERMINAL_OK,
    _format_error,
    _get_runway_api_key,
)
from .tts_base import TTSDriver

logger = logging.getLogger(__name__)

SOUND_EFFECT_MODEL = "eleven_text_to_sound_v2"
TTS_MODEL = "eleven_multilingual_v2"

PRESET_VOICES: list[str] = [
    "Maya", "Arjun", "Serene", "Bernard", "Billy", "Mark", "Clint", "Mabel",
    "Chad", "Leslie", "Eleanor", "Elias", "Elliot", "Grungle", "Brodie",
    "Sandra", "Kirk", "Kylie", "Lara", "Lisa", "Malachi", "Marlene", "Martin",
    "Miriam", "Monster", "Paula", "Pip", "Rusty", "Ragnar", "Xylar", "Maggie",
    "Jack", "Katie", "Noah", "James", "Rina", "Ella", "Mariah", "Frank",
    "Claudia", "Niki", "Vincent", "Kendrick", "Myrna", "Tom", "Wanda",
    "Benjamin", "Kiana", "Rachel",
]


def _endpoint_for_model(model: str) -> str:
    return "/v1/sound_effect" if model == SOUND_EFFECT_MODEL else "/v1/text_to_speech"


def _fetch_audio(client: httpx.Client, url: str) -> tuple[bytes, str]:
    r = client.get(url, timeout=120.0)
    if r.status_code >= 400:
        raise RuntimeError(f"Failed to download audio ({r.status_code}): {url}")
    return r.content, r.headers.get("content-type", "audio/mpeg")


class RunwayTTSDriver(AudioCostMixin, TTSDriver):
    """Runway TTS + sound-effect synthesis."""

    supports_streaming = False
    supports_ssml = False
    available_voices = list(PRESET_VOICES)

    KNOWN_MODELS = [TTS_MODEL, SOUND_EFFECT_MODEL]

    # USD per character / per second. ElevenLabs-style pricing applies.
    AUDIO_PRICING = {
        TTS_MODEL: {"per_character": 0.000030},
        SOUND_EFFECT_MODEL: {"per_second": 0.04},
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = TTS_MODEL,
        endpoint: str | None = None,
    ):
        self.api_key = _get_runway_api_key(api_key)
        self.model = model
        self.endpoint = (
            endpoint or os.getenv("RUNWAY_ENDPOINT") or _DEFAULT_ENDPOINT
        ).rstrip("/")

    @classmethod
    def list_models(cls, *, api_key: str | None = None, **kw: object) -> list[str] | None:
        key = _get_runway_api_key(api_key)
        if not key:
            return None
        return list(cls.KNOWN_MODELS)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Runway-Version": _API_VERSION,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _build(self, text: str, options: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        model = options.get("model", self.model)
        body: dict[str, Any] = {"model": model, "promptText": text}

        if model == SOUND_EFFECT_MODEL:
            if "duration" in options:
                duration = float(options["duration"])
                if not 0.5 <= duration <= 30:
                    raise ValueError("sound_effect duration must be in [0.5, 30]")
                body["duration"] = duration
            if "loop" in options:
                body["loop"] = bool(options["loop"])
        else:
            # text_to_speech requires a preset voice.
            voice = options.get("voice")
            if voice is None:
                preset_id = options.get("preset_id") or options.get("presetId") or "Maya"
                voice = {"type": "runway-preset", "presetId": preset_id}
            elif isinstance(voice, str):
                voice = {"type": "runway-preset", "presetId": voice}
            body["voice"] = voice

        return model, _endpoint_for_model(model), body

    def synthesize(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        """Synthesize audio.

        Options:
            model: ``eleven_multilingual_v2`` (TTS) or ``eleven_text_to_sound_v2``
                (sound effect). Defaults to instance model.
            voice: For TTS — preset voice name (string) or full ``{type,presetId}``
                dict. Defaults to ``Maya``.
            duration / loop: Sound-effect only (duration 0.5–30 s).
            poll, poll_interval, timeout: Control task polling.
        """
        if not self.api_key:
            raise RuntimeError("RUNWAY_API_KEY (or RUNWAYML_API_SECRET) is not configured")
        if not text:
            raise ValueError("text cannot be empty")

        model, path, body = self._build(text, options)
        headers = self._headers()

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{self.endpoint}{path}", headers=headers, json=body)
            if resp.status_code >= 400:
                raise RuntimeError(f"Runway {path} failed: {_format_error(resp.status_code, resp)}")
            task = resp.json()
            task_id = task.get("id")
            if not task_id:
                raise RuntimeError(f"Runway response missing task id: {task}")

            if not options.get("poll", True):
                return {
                    "audio": b"",
                    "media_type": "audio/mpeg",
                    "meta": {
                        "characters": len(text),
                        "cost": 0.0,
                        "model_name": f"runway/{model}",
                        "task_id": task_id,
                        "status": task.get("status", "PENDING"),
                        "raw_response": task,
                    },
                }

            final = self._poll_until_done(
                client,
                task_id,
                headers=headers,
                poll_interval=float(options.get("poll_interval", 5)),
                timeout_seconds=float(options.get("timeout", 600)),
            )
            outputs = final.get("output") or []
            if not outputs:
                raise RuntimeError(f"Runway {path} produced no output URLs: {final}")
            audio, media_type = _fetch_audio(client, outputs[0])

        characters = len(text)
        duration_s = float(body.get("duration") or 0.0)
        cost = self._calculate_audio_cost(
            "runway",
            model,
            duration_seconds=duration_s,
            characters=characters if model == TTS_MODEL else 0,
        )

        return {
            "audio": audio,
            "media_type": media_type,
            "meta": {
                "characters": characters,
                "cost": cost,
                "model_name": f"runway/{model}",
                "task_id": final.get("id"),
                "status": final.get("status"),
                "raw_response": final,
            },
        }

    def _poll_until_done(
        self,
        client: httpx.Client,
        task_id: str,
        *,
        headers: dict[str, str],
        poll_interval: float,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            r = client.get(f"{self.endpoint}/v1/tasks/{task_id}", headers=headers)
            if r.status_code >= 400:
                raise RuntimeError(f"Runway task poll failed: {_format_error(r.status_code, r)}")
            data = r.json()
            status = data.get("status")
            if status in _TERMINAL_OK:
                return data
            if status in _TERMINAL_FAIL:
                failure = data.get("failure") or data.get("failureCode") or status
                raise RuntimeError(f"Runway task {task_id} {status}: {failure}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Runway task {task_id} timed out (status={status})")
            time.sleep(poll_interval)
