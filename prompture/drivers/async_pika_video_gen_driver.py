"""Async Pika Labs video generation driver."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from ..infra.cost_mixin import VideoCostMixin
from .async_video_gen_base import AsyncVideoGenDriver
from .pika_video_gen_driver import (
    _DEFAULT_ENDPOINT,
    _DEFAULT_POLL_INTERVAL,
    _KNOWN_MODELS,
    _MAX_RETRIES,
    _TERMINAL_FAIL,
    _TERMINAL_OK,
    PikaVideoGenDriver,
    _build_body,
    _format_error,
    _get_pika_api_key,
    _make_video_response,
    _model_version,
    _pending_response,
)

logger = logging.getLogger(__name__)


class AsyncPikaVideoGenDriver(VideoCostMixin, AsyncVideoGenDriver):
    """Async video generation via Pika Labs."""

    supports_image_input = PikaVideoGenDriver.supports_image_input
    supports_reference_images = PikaVideoGenDriver.supports_reference_images
    supports_video_input = PikaVideoGenDriver.supports_video_input
    supports_audio = PikaVideoGenDriver.supports_audio
    supports_polling = PikaVideoGenDriver.supports_polling
    supported_aspect_ratios = PikaVideoGenDriver.supported_aspect_ratios
    supported_resolutions = PikaVideoGenDriver.supported_resolutions
    max_seconds = PikaVideoGenDriver.max_seconds

    KNOWN_MODELS = PikaVideoGenDriver.KNOWN_MODELS
    VIDEO_PRICING = PikaVideoGenDriver.VIDEO_PRICING

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "pika-2.2",
        endpoint: str | None = None,
    ):
        self.api_key = _get_pika_api_key(api_key)
        self.model = model
        self.endpoint = (endpoint or os.getenv("PIKA_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("PIKA_API_KEY is not configured")
        model = options.get("model", self.model)
        if model not in _KNOWN_MODELS:
            logger.warning("Pika model %r not in known list %s", model, sorted(_KNOWN_MODELS))

        op, body = _build_body(prompt, options)
        version = _model_version(model)
        submit_url = f"{self.endpoint}/generate/{version}/{op}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(submit_url, headers=self._headers(), json=body)
            if resp.status_code >= 400:
                raise RuntimeError(f"Pika submit failed: {_format_error(resp.status_code, resp)}")
            submitted = resp.json()
            task_id = submitted.get("video_id") or submitted.get("id")
            if not task_id:
                raise RuntimeError(f"Pika response missing video_id: {submitted}")

            if not options.get("poll", True):
                return _pending_response(task_id, op, model, body, submitted)

            final = await self._poll(
                client,
                task_id,
                poll_interval=float(options.get("poll_interval", _DEFAULT_POLL_INTERVAL)),
                max_retries=int(options.get("max_retries", _MAX_RETRIES)),
            )

        duration = float(body.get("options", {}).get("duration", 0) or 0)
        cost = self._calculate_video_cost("pika", model, duration_seconds=duration, n=1)
        return _make_video_response(final, op, model, body, cost)

    async def _poll(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        *,
        poll_interval: float,
        max_retries: int,
    ) -> dict[str, Any]:
        url = f"{self.endpoint}/videos/{task_id}"
        for _ in range(max_retries):
            r = await client.get(url, headers=self._headers())
            if r.status_code >= 400:
                raise RuntimeError(f"Pika poll failed: {_format_error(r.status_code, r)}")
            data = r.json()
            status = (data.get("status") or "").lower()
            if status in _TERMINAL_OK:
                return data
            if status in _TERMINAL_FAIL:
                reason = data.get("error") or data.get("message") or status
                raise RuntimeError(f"Pika task {task_id} {status}: {reason}")
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"Pika task {task_id} timed out after {max_retries} retries")
