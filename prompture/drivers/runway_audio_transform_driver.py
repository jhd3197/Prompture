"""Runway audio-transform driver — voice dubbing, voice isolation, and speech-to-speech.

These three endpoints take audio (or video) as input and emit audio as output,
which doesn't fit Prompture's STT (audio → text) or TTS (text → audio) modality
contracts. The driver lives outside the modality registry; import it directly:

    from prompture.drivers import RunwayAudioTransformDriver
    driver = RunwayAudioTransformDriver()
    result = driver.dub("https://.../audio.mp3", target_lang="es")

All three methods submit a task, poll until terminal, download the resulting
audio, and return ``{"audio": bytes, "media_type": str, "meta": {...}}``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from .runway_img_gen_driver import (
    _API_VERSION,
    _DEFAULT_ENDPOINT,
    _TERMINAL_FAIL,
    _TERMINAL_OK,
    _format_error,
    _get_runway_api_key,
)

logger = logging.getLogger(__name__)

VOICE_DUBBING_MODEL = "eleven_voice_dubbing"
VOICE_ISOLATION_MODEL = "eleven_voice_isolation"
SPEECH_TO_SPEECH_MODEL = "eleven_multilingual_sts_v2"

DUBBING_LANGUAGES: set[str] = {
    "en",
    "hi",
    "pt",
    "zh",
    "es",
    "fr",
    "de",
    "ja",
    "ar",
    "ru",
    "ko",
    "id",
    "it",
    "nl",
    "tr",
    "pl",
    "sv",
    "fil",
    "ms",
    "ro",
    "uk",
    "el",
    "cs",
    "da",
    "fi",
    "bg",
    "hr",
    "sk",
    "ta",
}


def _normalize_media_input(media: Any, *, kind: str = "audio") -> dict[str, str]:
    """Return a ``{type, uri}`` block for a media argument."""
    if isinstance(media, str):
        return {"type": kind, "uri": media}
    if isinstance(media, dict):
        uri = media.get("uri") or media.get("url")
        if not uri:
            raise ValueError(f"{kind} dict must include 'uri'")
        return {"type": media.get("type", kind), "uri": str(uri)}
    raise ValueError(f"unsupported {kind} input type: {type(media).__name__}")


def _fetch_audio(client: httpx.Client, url: str) -> tuple[bytes, str]:
    r = client.get(url, timeout=120.0)
    if r.status_code >= 400:
        raise RuntimeError(f"Failed to download audio ({r.status_code}): {url}")
    return r.content, r.headers.get("content-type", "audio/mpeg")


class RunwayAudioTransformDriver:
    """Runway audio-transform operations (dub / isolate / convert voice)."""

    KNOWN_MODELS = [
        VOICE_DUBBING_MODEL,
        VOICE_ISOLATION_MODEL,
        SPEECH_TO_SPEECH_MODEL,
    ]

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
    ):
        self.api_key = _get_runway_api_key(api_key)
        self.endpoint = (endpoint or os.getenv("RUNWAY_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

    # ── public API ──────────────────────────────────────────────────────

    def dub(
        self,
        audio: str | dict[str, Any],
        target_lang: str,
        *,
        disable_voice_cloning: bool | None = None,
        drop_background_audio: bool | None = None,
        num_speakers: int | None = None,
        poll: bool = True,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        """Dub the input audio to ``target_lang`` (ISO 639-1 code)."""
        if target_lang not in DUBBING_LANGUAGES:
            raise ValueError(f"Unsupported targetLang {target_lang!r}. Accepted: {sorted(DUBBING_LANGUAGES)}")
        audio_norm = _normalize_media_input(audio, kind="audio")
        body: dict[str, Any] = {
            "model": VOICE_DUBBING_MODEL,
            "audioUri": audio_norm["uri"],
            "targetLang": target_lang,
        }
        if disable_voice_cloning is not None:
            body["disableVoiceCloning"] = bool(disable_voice_cloning)
        if drop_background_audio is not None:
            body["dropBackgroundAudio"] = bool(drop_background_audio)
        if num_speakers is not None:
            body["numSpeakers"] = int(num_speakers)
        return self._submit_and_collect(
            "/v1/voice_dubbing", body, poll=poll, poll_interval=poll_interval, timeout=timeout
        )

    def isolate(
        self,
        audio: str | dict[str, Any],
        *,
        poll: bool = True,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        """Isolate voice from background audio (4.6 s ≤ duration < 3600 s)."""
        audio_norm = _normalize_media_input(audio, kind="audio")
        body = {
            "model": VOICE_ISOLATION_MODEL,
            "audioUri": audio_norm["uri"],
        }
        return self._submit_and_collect(
            "/v1/voice_isolation", body, poll=poll, poll_interval=poll_interval, timeout=timeout
        )

    def convert_voice(
        self,
        media: str | dict[str, Any],
        voice_preset_id: str,
        *,
        media_kind: str = "audio",
        remove_background_noise: bool | None = None,
        poll: bool = True,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        """Re-voice ``media`` (audio or video) with the given Runway voice preset."""
        if media_kind not in {"audio", "video"}:
            raise ValueError("media_kind must be 'audio' or 'video'")
        media_norm = _normalize_media_input(media, kind=media_kind)
        body: dict[str, Any] = {
            "model": SPEECH_TO_SPEECH_MODEL,
            "media": media_norm,
            "voice": {"type": "runway-preset", "presetId": voice_preset_id},
        }
        if remove_background_noise is not None:
            body["removeBackgroundNoise"] = bool(remove_background_noise)
        return self._submit_and_collect(
            "/v1/speech_to_speech", body, poll=poll, poll_interval=poll_interval, timeout=timeout
        )

    # ── internals ───────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Runway-Version": _API_VERSION,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _submit_and_collect(
        self,
        path: str,
        body: dict[str, Any],
        *,
        poll: bool,
        poll_interval: float,
        timeout: float,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("RUNWAY_API_KEY (or RUNWAYML_API_SECRET) is not configured")
        headers = self._headers()

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{self.endpoint}{path}", headers=headers, json=body)
            if resp.status_code >= 400:
                raise RuntimeError(f"Runway {path} failed: {_format_error(resp.status_code, resp)}")
            task = resp.json()
            task_id = task.get("id")
            if not task_id:
                raise RuntimeError(f"Runway response missing task id: {task}")

            if not poll:
                return {
                    "audio": b"",
                    "media_type": "audio/mpeg",
                    "meta": {
                        "model_name": f"runway/{body['model']}",
                        "endpoint": path,
                        "task_id": task_id,
                        "status": task.get("status", "PENDING"),
                        "raw_response": task,
                    },
                }

            final = self._poll_until_done(
                client, task_id, headers=headers, poll_interval=poll_interval, timeout_seconds=timeout
            )
            outputs = final.get("output") or []
            if not outputs:
                raise RuntimeError(f"Runway {path} produced no output URLs: {final}")
            audio, media_type = _fetch_audio(client, outputs[0])

        return {
            "audio": audio,
            "media_type": media_type,
            "meta": {
                "model_name": f"runway/{body['model']}",
                "endpoint": path,
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
