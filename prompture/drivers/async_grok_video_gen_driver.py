"""Async xAI Grok Imagine video generation driver."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from ..infra.cost_mixin import VideoCostMixin
from .async_video_gen_base import AsyncVideoGenDriver
from .grok_video_gen_driver import (
    _DEFAULT_ENDPOINT,
    _duration_from_options,
    _error_text,
    _extract_video_content,
    _get_xai_api_key,
    _image_payload,
)


class AsyncGrokVideoGenDriver(VideoCostMixin, AsyncVideoGenDriver):
    """Async video generation via xAI Grok Imagine Video REST API."""

    supports_image_input = True
    supports_reference_images = True
    supports_video_input = False
    supports_audio = True
    supports_polling = True
    supported_aspect_ratios = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
    supported_resolutions = ["480p", "720p"]
    max_seconds = 15

    KNOWN_MODELS = ["grok-imagine-video"]
    VIDEO_PRICING: dict[str, dict[str, Any]] = {
        "grok-imagine-video": {
            "per_second": 0.05,
            "per_second_by_resolution": {
                "480p": 0.05,
                "720p": 0.07,
            },
        }
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "grok-imagine-video",
        endpoint: str | None = None,
    ):
        self.api_key = _get_xai_api_key(api_key)
        self.model = model
        self.endpoint = (endpoint or os.getenv("GROK_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

    @classmethod
    def list_models(cls, *, api_key: str | None = None, **kw: object) -> list[str] | None:
        """Return known xAI video models when an API key is configured."""
        key = _get_xai_api_key(api_key)
        if not key:
            return None
        return list(cls.KNOWN_MODELS)

    async def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate a video using xAI Grok Imagine Video (async)."""
        if not self.api_key:
            raise RuntimeError("GROK_API_KEY or XAI_API_KEY environment variable is required")
        if not prompt:
            raise ValueError("prompt cannot be empty")

        model = options.get("model", self.model)
        duration = _duration_from_options(options)
        aspect_ratio = options.get("aspect_ratio", "16:9")
        resolution = options.get("resolution", "480p")

        if aspect_ratio not in self.supported_aspect_ratios:
            raise ValueError(f"Unsupported aspect_ratio '{aspect_ratio}'")
        if resolution not in self.supported_resolutions:
            raise ValueError(f"Unsupported resolution '{resolution}'")

        image = options.get("image")
        reference_images = options.get("reference_images")
        if image is not None and reference_images:
            raise ValueError("image and reference_images cannot be used together")
        if reference_images:
            if len(reference_images) > 7:
                raise ValueError("reference_images supports at most 7 images")
            if duration > 10:
                raise ValueError("duration must be 10 seconds or less when using reference_images")

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
        }
        if image is not None:
            payload["image"] = _image_payload(image)
        if reference_images:
            payload["reference_images"] = [_image_payload(img) for img in reference_images]

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoint}/videos/generations",
                headers=headers,
                json=payload,
                timeout=120.0,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Grok video generation failed: HTTP {response.status_code}: {_error_text(response)}"
                )

            start_data = response.json()
            request_id = start_data.get("request_id")
            if not request_id:
                raise RuntimeError("Grok video generation response did not include request_id")

            if not options.get("poll", True):
                return {
                    "videos": [],
                    "meta": {
                        "video_count": 0,
                        "request_id": request_id,
                        "status": start_data.get("status", "pending"),
                        "duration_seconds": duration,
                        "aspect_ratio": aspect_ratio,
                        "resolution": resolution,
                        "cost": 0.0,
                        "model_name": f"grok/{model}",
                        "raw_response": start_data,
                    },
                }

            final_data = await self._poll_until_done(
                client,
                request_id,
                headers=headers,
                poll_interval=float(options.get("poll_interval", 5)),
                timeout_seconds=float(options.get("timeout", 600)),
            )

        video = _extract_video_content(final_data)
        duration_seconds = video.duration_seconds or float(duration)
        cost = self._calculate_video_cost(
            "grok",
            model,
            duration_seconds=duration_seconds,
            n=1,
            resolution=resolution,
        )

        return {
            "videos": [video],
            "meta": {
                "video_count": 1,
                "request_id": request_id,
                "status": final_data.get("status"),
                "duration_seconds": duration_seconds,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "cost": cost,
                "model_name": f"grok/{model}",
                "raw_response": {"start": start_data, "final": final_data},
            },
        }

    async def _poll_until_done(
        self,
        client: httpx.AsyncClient,
        request_id: str,
        *,
        headers: dict[str, str],
        poll_interval: float,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            response = await client.get(f"{self.endpoint}/videos/{request_id}", headers=headers, timeout=30.0)
            if response.status_code >= 400:
                raise RuntimeError(f"Grok video status failed: HTTP {response.status_code}: {_error_text(response)}")

            data = response.json()
            status = data.get("status")
            if status == "done":
                return data
            if status in {"failed", "expired"}:
                raise RuntimeError(f"Grok video generation {status}: {data}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Grok video generation timed out for request_id={request_id}")
            await asyncio.sleep(poll_interval)
