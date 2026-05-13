"""Async Luma AI (Dream Machine) video generation driver."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from ..infra.cost_mixin import VideoCostMixin
from .async_video_gen_base import AsyncVideoGenDriver
from .luma_video_gen_driver import (
    _DEFAULT_ENDPOINT,
    _DEFAULT_POLL_INTERVAL,
    _GENERATIONS_PATH,
    _MAX_RETRIES,
    _TERMINAL_FAIL,
    _TERMINAL_OK,
    LumaVideoGenDriver,
    _build_body,
    _format_error,
    _get_luma_api_key,
    _make_video_response,
    _pending_response,
)


class AsyncLumaVideoGenDriver(VideoCostMixin, AsyncVideoGenDriver):
    """Async video generation via Luma AI Dream Machine."""

    supports_image_input = LumaVideoGenDriver.supports_image_input
    supports_reference_images = LumaVideoGenDriver.supports_reference_images
    supports_video_input = LumaVideoGenDriver.supports_video_input
    supports_audio = LumaVideoGenDriver.supports_audio
    supports_polling = LumaVideoGenDriver.supports_polling
    supported_aspect_ratios = LumaVideoGenDriver.supported_aspect_ratios
    supported_resolutions = LumaVideoGenDriver.supported_resolutions
    max_seconds = LumaVideoGenDriver.max_seconds

    KNOWN_MODELS = LumaVideoGenDriver.KNOWN_MODELS
    VIDEO_PRICING = LumaVideoGenDriver.VIDEO_PRICING

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "ray-2",
        endpoint: str | None = None,
    ):
        self.api_key = _get_luma_api_key(api_key)
        self.model = model
        self.endpoint = (endpoint or os.getenv("LUMA_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("LUMA_API_KEY is not configured")

        body = _build_body(prompt, options, self.model)
        url = f"{self.endpoint}{_GENERATIONS_PATH}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, headers=self._headers(), json=body)
            if resp.status_code >= 400:
                raise RuntimeError(f"Luma submit failed: {_format_error(resp.status_code, resp)}")
            submitted = resp.json()
            task_id = submitted.get("id")
            if not task_id:
                raise RuntimeError(f"Luma response missing id: {submitted}")

            if not options.get("poll", True):
                return _pending_response(task_id, body, submitted)

            final = await self._poll(
                client,
                task_id,
                poll_interval=float(options.get("poll_interval", _DEFAULT_POLL_INTERVAL)),
                max_retries=int(options.get("max_retries", _MAX_RETRIES)),
            )

        duration = float(options.get("duration", 0) or 0)
        cost = self._calculate_video_cost("luma", body["model"], duration_seconds=duration, n=1)
        return _make_video_response(final, body, cost)

    async def _poll(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        *,
        poll_interval: float,
        max_retries: int,
    ) -> dict[str, Any]:
        url = f"{self.endpoint}{_GENERATIONS_PATH}/{task_id}"
        for _ in range(max_retries):
            r = await client.get(url, headers=self._headers())
            if r.status_code >= 400:
                raise RuntimeError(f"Luma poll failed: {_format_error(r.status_code, r)}")
            data = r.json()
            state = data.get("state")
            if state in _TERMINAL_OK:
                return data
            if state in _TERMINAL_FAIL:
                reason = data.get("failure_reason") or state
                raise RuntimeError(f"Luma task {task_id} {state}: {reason}")
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"Luma task {task_id} timed out after {max_retries} retries")
