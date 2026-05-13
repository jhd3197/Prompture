"""Fal.ai video generation driver.

Fal is a serverless inference aggregator that hosts a large catalog of
image/video models under a uniform queue-based REST API.  This driver
targets the public queue endpoints directly so we don't pull in the
``fal-client`` SDK as a hard dependency:

- ``POST https://queue.fal.run/{model_id}`` — submit a job, returns a request_id
- ``GET https://queue.fal.run/{model_id}/requests/{request_id}/status`` — poll
- ``GET https://queue.fal.run/{model_id}/requests/{request_id}`` — fetch result

The Prompture ``model`` string is the Fal model id, e.g.
``fal-ai/kling-video/v2.6/pro/image-to-video``.  Any options are forwarded to
the model's input schema.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from ..infra.cost_mixin import VideoCostMixin
from ..media.video import video_from_url
from .async_video_gen_base import AsyncVideoGenDriver
from .video_gen_base import VideoGenDriver

logger = logging.getLogger(__name__)

_QUEUE_BASE = "https://queue.fal.run"
_DEFAULT_MODEL = "fal-ai/kling-video/v2.6/pro/image-to-video"


def _get_fal_api_key(api_key: str | None = None) -> str | None:
    return api_key or os.getenv("FAL_API_KEY") or os.getenv("FAL_KEY")


def _build_video_input(prompt: str, options: dict[str, Any]) -> dict[str, Any]:
    """Pass-through builder. Caller is responsible for shaping options
    to match the target Fal model's input schema."""
    input_payload: dict[str, Any] = {}
    if prompt:
        input_payload["prompt"] = prompt
    # Common Fal video keys we map from generic options.
    if "image" in options or "image_url" in options:
        input_payload["image_url"] = options.get("image_url") or options.get("image")
    if "video" in options or "video_url" in options:
        input_payload["video_url"] = options.get("video_url") or options.get("video")
    if "duration" in options:
        input_payload["duration"] = str(options["duration"])
    if "negative_prompt" in options:
        input_payload["negative_prompt"] = options["negative_prompt"]
    if options.get("generate_audio") is not None:
        input_payload["generate_audio"] = bool(options["generate_audio"])
    # Allow callers to pass-through anything else via ``extra``.
    extra = options.get("extra")
    if isinstance(extra, dict):
        input_payload.update(extra)
    return input_payload


class FalVideoGenDriver(VideoCostMixin, VideoGenDriver):
    """Video generation via Fal.ai queue API."""

    supports_image_input = True
    supports_reference_images = True
    supports_video_input = True
    supports_audio = True
    supports_polling = True
    supported_aspect_ratios = []
    supported_resolutions = []
    max_seconds = None

    KNOWN_MODELS = [
        "fal-ai/kling-video/v2.6/pro/image-to-video",
        "fal-ai/kling-video/v2.6/pro/motion-control",
        "fal-ai/kling-video/v2/master/image-to-video",
        "fal-ai/luma-dream-machine",
        "fal-ai/runway-gen3/turbo/image-to-video",
        "fal-ai/minimax-video",
        "fal-ai/veo3",
    ]

    # Per-second placeholder pricing. Real Fal pricing varies per model
    # (see fal.ai/pricing) — refine via the local KB.
    VIDEO_PRICING: dict[str, dict[str, Any]] = {}

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        endpoint: str | None = None,
    ):
        self.api_key = _get_fal_api_key(api_key)
        self.model = model
        self.endpoint = (endpoint or os.getenv("FAL_ENDPOINT") or _QUEUE_BASE).rstrip("/")

    @classmethod
    def list_models(cls, **kw: object) -> list[str] | None:
        return list(cls.KNOWN_MODELS)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("FAL_API_KEY is not configured")
        model_id = options.get("model", self.model)
        input_payload = _build_video_input(prompt, options)

        with httpx.Client(timeout=120.0) as client:
            submit = client.post(f"{self.endpoint}/{model_id}", headers=self._headers(), json=input_payload)
            if submit.status_code >= 400:
                raise RuntimeError(f"Fal submit failed {submit.status_code}: {submit.text}")
            submitted = submit.json()
            request_id = submitted.get("request_id")
            if not request_id:
                raise RuntimeError(f"Fal response missing request_id: {submitted}")

            if not options.get("poll", True):
                return self._pending_response(model_id, request_id, submitted)

            self._poll_status(
                client,
                model_id,
                request_id,
                poll_interval=float(options.get("poll_interval", 5)),
                timeout_seconds=float(options.get("timeout", 900)),
            )
            result = self._fetch_result(client, model_id, request_id)

        url = self._extract_video_url(result)
        videos = [video_from_url(url)] if url else []
        duration = float(options.get("duration", 0) or 0)
        cost = self._calculate_video_cost("fal", model_id, duration_seconds=duration, n=1)
        return {
            "videos": videos,
            "meta": {
                "video_count": len(videos),
                "duration_seconds": duration or None,
                "aspect_ratio": options.get("aspect_ratio"),
                "resolution": options.get("resolution"),
                "cost": cost,
                "model_name": f"fal/{model_id}",
                "request_id": request_id,
                "raw_response": result,
            },
        }

    def _poll_status(
        self,
        client: httpx.Client,
        model_id: str,
        request_id: str,
        *,
        poll_interval: float,
        timeout_seconds: float,
    ) -> None:
        url = f"{self.endpoint}/{model_id}/requests/{request_id}/status"
        deadline = time.monotonic() + timeout_seconds
        while True:
            r = client.get(url, headers=self._headers())
            if r.status_code >= 400:
                raise RuntimeError(f"Fal status failed: {r.status_code} {r.text}")
            data = r.json()
            status = data.get("status")
            if status == "COMPLETED":
                return
            if status in {"FAILED", "ERROR", "CANCELED"}:
                raise RuntimeError(f"Fal job {request_id} {status}: {data}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Fal job {request_id} timed out (status={status})")
            time.sleep(poll_interval)

    def _fetch_result(self, client: httpx.Client, model_id: str, request_id: str) -> dict[str, Any]:
        url = f"{self.endpoint}/{model_id}/requests/{request_id}"
        r = client.get(url, headers=self._headers())
        if r.status_code >= 400:
            raise RuntimeError(f"Fal result fetch failed: {r.status_code} {r.text}")
        return r.json()

    @staticmethod
    def _extract_video_url(result: dict[str, Any]) -> str | None:
        # Fal result shapes vary: {"video": {"url": ...}}, {"videos": [{"url": ...}]}, etc.
        video = result.get("video")
        if isinstance(video, dict) and video.get("url"):
            return video["url"]
        videos = result.get("videos")
        if isinstance(videos, list) and videos:
            first = videos[0]
            if isinstance(first, dict) and first.get("url"):
                return first["url"]
        return None

    def _pending_response(self, model_id: str, request_id: str, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "videos": [],
            "meta": {
                "video_count": 0,
                "duration_seconds": None,
                "aspect_ratio": None,
                "resolution": None,
                "cost": 0.0,
                "model_name": f"fal/{model_id}",
                "request_id": request_id,
                "status": "pending",
                "raw_response": raw,
            },
        }


class AsyncFalVideoGenDriver(VideoCostMixin, AsyncVideoGenDriver):
    """Async video generation via Fal.ai."""

    supports_image_input = FalVideoGenDriver.supports_image_input
    supports_reference_images = FalVideoGenDriver.supports_reference_images
    supports_video_input = FalVideoGenDriver.supports_video_input
    supports_audio = FalVideoGenDriver.supports_audio
    supports_polling = FalVideoGenDriver.supports_polling
    supported_aspect_ratios = FalVideoGenDriver.supported_aspect_ratios
    supported_resolutions = FalVideoGenDriver.supported_resolutions
    max_seconds = FalVideoGenDriver.max_seconds

    KNOWN_MODELS = FalVideoGenDriver.KNOWN_MODELS
    VIDEO_PRICING = FalVideoGenDriver.VIDEO_PRICING

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        endpoint: str | None = None,
    ):
        self.api_key = _get_fal_api_key(api_key)
        self.model = model
        self.endpoint = (endpoint or os.getenv("FAL_ENDPOINT") or _QUEUE_BASE).rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("FAL_API_KEY is not configured")
        model_id = options.get("model", self.model)
        input_payload = _build_video_input(prompt, options)

        async with httpx.AsyncClient(timeout=120.0) as client:
            submit = await client.post(f"{self.endpoint}/{model_id}", headers=self._headers(), json=input_payload)
            if submit.status_code >= 400:
                raise RuntimeError(f"Fal submit failed {submit.status_code}: {submit.text}")
            submitted = submit.json()
            request_id = submitted.get("request_id")
            if not request_id:
                raise RuntimeError(f"Fal response missing request_id: {submitted}")

            if not options.get("poll", True):
                helper = FalVideoGenDriver(api_key=self.api_key)
                return helper._pending_response(model_id, request_id, submitted)

            await self._poll_status(
                client,
                model_id,
                request_id,
                poll_interval=float(options.get("poll_interval", 5)),
                timeout_seconds=float(options.get("timeout", 900)),
            )
            result = await self._fetch_result(client, model_id, request_id)

        url = FalVideoGenDriver._extract_video_url(result)
        videos = [video_from_url(url)] if url else []
        duration = float(options.get("duration", 0) or 0)
        cost = self._calculate_video_cost("fal", model_id, duration_seconds=duration, n=1)
        return {
            "videos": videos,
            "meta": {
                "video_count": len(videos),
                "duration_seconds": duration or None,
                "aspect_ratio": options.get("aspect_ratio"),
                "resolution": options.get("resolution"),
                "cost": cost,
                "model_name": f"fal/{model_id}",
                "request_id": request_id,
                "raw_response": result,
            },
        }

    async def _poll_status(
        self,
        client: httpx.AsyncClient,
        model_id: str,
        request_id: str,
        *,
        poll_interval: float,
        timeout_seconds: float,
    ) -> None:
        url = f"{self.endpoint}/{model_id}/requests/{request_id}/status"
        deadline = time.monotonic() + timeout_seconds
        while True:
            r = await client.get(url, headers=self._headers())
            if r.status_code >= 400:
                raise RuntimeError(f"Fal status failed: {r.status_code} {r.text}")
            data = r.json()
            status = data.get("status")
            if status == "COMPLETED":
                return
            if status in {"FAILED", "ERROR", "CANCELED"}:
                raise RuntimeError(f"Fal job {request_id} {status}: {data}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Fal job {request_id} timed out (status={status})")
            await asyncio.sleep(poll_interval)

    async def _fetch_result(self, client: httpx.AsyncClient, model_id: str, request_id: str) -> dict[str, Any]:
        url = f"{self.endpoint}/{model_id}/requests/{request_id}"
        r = await client.get(url, headers=self._headers())
        if r.status_code >= 400:
            raise RuntimeError(f"Fal result fetch failed: {r.status_code} {r.text}")
        return r.json()
