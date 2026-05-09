"""xAI Grok Imagine video generation driver."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from ..infra.cost_mixin import VideoCostMixin
from ..media.image import ImageContent, ImageInput, make_image
from ..media.video import VideoContent, video_from_url
from .video_gen_base import VideoGenDriver

_DEFAULT_ENDPOINT = "https://api.x.ai/v1"


def _get_xai_api_key(api_key: str | None = None) -> str | None:
    return api_key or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")


def _duration_from_options(options: dict[str, Any], default: int = 8) -> int:
    duration = options.get("duration", options.get("seconds", default))
    try:
        value = int(duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration must be an integer number of seconds") from exc
    if value < 1 or value > 15:
        raise ValueError("duration must be between 1 and 15 seconds")
    return value


def _image_payload(image: ImageInput | dict[str, Any]) -> dict[str, str]:
    if isinstance(image, dict):
        if "url" in image:
            return {"url": str(image["url"])}
        if "data" in image:
            return {"url": str(image["data"])}
        raise ValueError("image dictionaries must include 'url' or 'data'")

    content: ImageContent = make_image(image)
    if content.source_type == "url":
        if not content.url:
            raise ValueError("URL image input is missing a URL")
        return {"url": content.url}
    return {"url": f"data:{content.media_type};base64,{content.data}"}


def _extract_video_content(data: dict[str, Any]) -> VideoContent:
    video_info = data.get("video") or {}
    url = video_info.get("url")
    if not url:
        raise RuntimeError("Grok video generation completed without a video URL")
    duration = video_info.get("duration")
    try:
        duration_seconds = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration_seconds = None
    return video_from_url(url, duration_seconds=duration_seconds)


def _error_text(response: httpx.Response) -> str:
    try:
        return response.text
    except Exception:
        return "<unavailable>"


class GrokVideoGenDriver(VideoCostMixin, VideoGenDriver):
    """Video generation via xAI Grok Imagine Video REST API."""

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

    def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate a video using xAI Grok Imagine Video.

        Options:
            model: Model override. Defaults to ``grok-imagine-video``.
            duration/seconds: Length in seconds, 1-15. Defaults to 8.
            aspect_ratio: One of the supported aspect ratios. Defaults to ``16:9``.
            resolution: ``480p`` or ``720p``. Defaults to ``480p``.
            image: Optional image URL, data URI, bytes, path, or ImageContent for image-to-video.
            reference_images: Optional list of image inputs for reference-to-video.
            poll: Whether to poll until complete. Defaults to True.
            poll_interval: Seconds between polls. Defaults to 5.
            timeout: Maximum polling seconds. Defaults to 600.
        """
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
        response = httpx.post(f"{self.endpoint}/videos/generations", headers=headers, json=payload, timeout=120.0)
        if response.status_code >= 400:
            raise RuntimeError(f"Grok video generation failed: HTTP {response.status_code}: {_error_text(response)}")

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

        final_data = self._poll_until_done(
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

    def _poll_until_done(
        self,
        request_id: str,
        *,
        headers: dict[str, str],
        poll_interval: float,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            response = httpx.get(f"{self.endpoint}/videos/{request_id}", headers=headers, timeout=30.0)
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
            time.sleep(poll_interval)
