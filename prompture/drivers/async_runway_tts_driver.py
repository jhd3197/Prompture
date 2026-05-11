"""Async Runway TTS driver — mirrors :mod:`runway_tts_driver`."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from ..infra.cost_mixin import AudioCostMixin
from .async_tts_base import AsyncTTSDriver
from .runway_img_gen_driver import (
    _API_VERSION,
    _DEFAULT_ENDPOINT,
    _TERMINAL_FAIL,
    _TERMINAL_OK,
    _format_error,
    _get_runway_api_key,
)
from .runway_tts_driver import (
    PRESET_VOICES,
    TTS_MODEL,
    RunwayTTSDriver,
)

logger = logging.getLogger(__name__)


async def _fetch_audio(client: httpx.AsyncClient, url: str) -> tuple[bytes, str]:
    r = await client.get(url, timeout=120.0)
    if r.status_code >= 400:
        raise RuntimeError(f"Failed to download audio ({r.status_code}): {url}")
    return r.content, r.headers.get("content-type", "audio/mpeg")


class AsyncRunwayTTSDriver(AudioCostMixin, AsyncTTSDriver):
    """Async Runway TTS + sound-effect synthesis."""

    supports_streaming = False
    supports_ssml = False
    available_voices = list(PRESET_VOICES)

    KNOWN_MODELS = RunwayTTSDriver.KNOWN_MODELS
    AUDIO_PRICING = RunwayTTSDriver.AUDIO_PRICING

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

    async def synthesize(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("RUNWAY_API_KEY (or RUNWAYML_API_SECRET) is not configured")
        if not text:
            raise ValueError("text cannot be empty")

        builder = RunwayTTSDriver.__new__(RunwayTTSDriver)
        builder.model = self.model
        model, path, body = builder._build(text, options)
        headers = self._headers()

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self.endpoint}{path}", headers=headers, json=body)
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

            final = await self._poll_until_done(
                client,
                task_id,
                headers=headers,
                poll_interval=float(options.get("poll_interval", 5)),
                timeout_seconds=float(options.get("timeout", 600)),
            )
            outputs = final.get("output") or []
            if not outputs:
                raise RuntimeError(f"Runway {path} produced no output URLs: {final}")
            audio, media_type = await _fetch_audio(client, outputs[0])

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

    async def _poll_until_done(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        *,
        headers: dict[str, str],
        poll_interval: float,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            r = await client.get(f"{self.endpoint}/v1/tasks/{task_id}", headers=headers)
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
            await asyncio.sleep(poll_interval)
