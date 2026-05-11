"""Async Runway video generation driver. Mirrors :mod:`runway_video_gen_driver`."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from ..infra.cost_mixin import VideoCostMixin
from .async_video_gen_base import AsyncVideoGenDriver
from .runway_img_gen_driver import (
    _API_VERSION,
    _DEFAULT_ENDPOINT,
    _TERMINAL_FAIL,
    _TERMINAL_OK,
    _format_error,
    _get_runway_api_key,
)
from .runway_video_gen_driver import RunwayVideoGenDriver

logger = logging.getLogger(__name__)


class AsyncRunwayVideoGenDriver(VideoCostMixin, AsyncVideoGenDriver):
    """Async video generation via Runway text/image/video → video endpoints."""

    supports_image_input = RunwayVideoGenDriver.supports_image_input
    supports_reference_images = RunwayVideoGenDriver.supports_reference_images
    supports_video_input = RunwayVideoGenDriver.supports_video_input
    supports_audio = RunwayVideoGenDriver.supports_audio
    supports_polling = RunwayVideoGenDriver.supports_polling
    supported_aspect_ratios = RunwayVideoGenDriver.supported_aspect_ratios
    supported_resolutions = RunwayVideoGenDriver.supported_resolutions
    max_seconds = RunwayVideoGenDriver.max_seconds

    KNOWN_MODELS = RunwayVideoGenDriver.KNOWN_MODELS
    VIDEO_PRICING = RunwayVideoGenDriver.VIDEO_PRICING

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gen4.5",
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

    async def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(
                "RUNWAY_API_KEY (or RUNWAYML_API_SECRET) is not configured"
            )
        if not prompt:
            raise ValueError("prompt cannot be empty")

        builder = RunwayVideoGenDriver.__new__(RunwayVideoGenDriver)
        builder.model = self.model
        mode, path, body = builder._build(prompt, options)
        headers = self._headers()

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self.endpoint}{path}", headers=headers, json=body)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Runway {mode} failed: {_format_error(resp.status_code, resp)}"
                )
            task = resp.json()
            task_id = task.get("id")
            if not task_id:
                raise RuntimeError(f"Runway response missing task id: {task}")

            if not options.get("poll", True):
                return RunwayVideoGenDriver._pending_response(builder, task_id, mode, body, task)

            final = await self._poll_until_done(
                client,
                task_id,
                headers=headers,
                poll_interval=float(options.get("poll_interval", 5)),
                timeout_seconds=float(options.get("timeout", 600)),
            )

        return RunwayVideoGenDriver._success_response(builder, final, mode, body, options)

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
                raise RuntimeError(
                    f"Runway task poll failed: {_format_error(r.status_code, r)}"
                )
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
