"""Luma AI (Dream Machine) video generation driver.

Wraps the Luma Dream Machine generations API:

- ``POST https://api.lumalabs.ai/dream-machine/v1/generations`` — submit a job
- ``GET  https://api.lumalabs.ai/dream-machine/v1/generations/<id>`` — poll

The endpoint returns ``{"id": ..., "state": "queued", "assets": null, ...}``.
The driver polls until ``state == "completed"``, then returns the video URL
from ``assets.video`` as a :class:`VideoContent`.

For image-to-video, pass ``image`` (or ``image_url``) in options; it is
materialized as ``keyframes.frame0 = {"type": "image", "url": ...}``. Pass
``end_image`` to also set ``keyframes.frame1``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from ..infra.cost_mixin import VideoCostMixin
from ..media.video import video_from_url
from .video_gen_base import VideoGenDriver

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.lumalabs.ai"
_GENERATIONS_PATH = "/dream-machine/v1/generations"

_KNOWN_MODELS = {"ray-2", "ray-flash-2", "ray-1-6"}
_LUMA_RATIOS = {"1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"}

_TERMINAL_OK = {"completed"}
_TERMINAL_FAIL = {"failed"}

_MAX_RETRIES = 60
_DEFAULT_POLL_INTERVAL = 5.0


def _get_luma_api_key(api_key: str | None = None) -> str | None:
    return api_key or os.getenv("LUMA_API_KEY")


def _build_keyframes(options: dict[str, Any]) -> dict[str, Any] | None:
    """Build the ``keyframes`` payload from start/end image options."""
    start = options.get("image") or options.get("image_url") or options.get("start_image")
    end = options.get("end_image") or options.get("last_frame")

    if not start and not end:
        return None

    keyframes: dict[str, Any] = {}
    if start:
        keyframes["frame0"] = {"type": "image", "url": start}
    if end:
        keyframes["frame1"] = {"type": "image", "url": end}
    return keyframes


def _build_body(prompt: str, options: dict[str, Any], default_model: str) -> dict[str, Any]:
    """Build the request body for a Luma generation."""
    model = options.get("model", default_model)
    if model not in _KNOWN_MODELS:
        logger.warning("Luma model %r is not in the known list %s", model, sorted(_KNOWN_MODELS))

    aspect_ratio = options.get("aspect_ratio", options.get("ratio", "16:9"))
    if aspect_ratio not in _LUMA_RATIOS:
        raise ValueError(f"Unsupported aspect_ratio {aspect_ratio!r}; expected one of {sorted(_LUMA_RATIOS)}")

    body: dict[str, Any] = {
        "prompt": prompt or "",
        "aspect_ratio": aspect_ratio,
        "loop": bool(options.get("loop", False)),
        "model": model,
    }

    keyframes = _build_keyframes(options)
    if keyframes is not None:
        body["keyframes"] = keyframes

    if "duration" in options:
        body["duration"] = str(options["duration"])
    if "resolution" in options:
        body["resolution"] = options["resolution"]

    return body


def _format_error(status_code: int, resp: httpx.Response) -> str:
    try:
        return f"{status_code}: {resp.json()}"
    except Exception:
        return f"{status_code}: {resp.text}"


def _make_video_response(
    final: dict[str, Any],
    body: dict[str, Any],
    cost: float,
) -> dict[str, Any]:
    assets = final.get("assets") or {}
    url = assets.get("video")
    videos = [video_from_url(url)] if url else []
    return {
        "videos": videos,
        "meta": {
            "video_count": len(videos),
            "duration_seconds": None,
            "aspect_ratio": body.get("aspect_ratio"),
            "resolution": body.get("resolution"),
            "cost": cost,
            "model_name": f"luma/{body['model']}",
            "task_id": final.get("id"),
            "status": final.get("state"),
            "raw_response": final,
        },
    }


def _pending_response(task_id: str, body: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "videos": [],
        "meta": {
            "video_count": 0,
            "duration_seconds": None,
            "aspect_ratio": body.get("aspect_ratio"),
            "resolution": body.get("resolution"),
            "cost": 0.0,
            "model_name": f"luma/{body['model']}",
            "task_id": task_id,
            "status": raw.get("state", "queued"),
            "raw_response": raw,
        },
    }


class LumaVideoGenDriver(VideoCostMixin, VideoGenDriver):
    """Video generation via Luma AI Dream Machine."""

    supports_image_input = True
    supports_reference_images = False
    supports_video_input = False
    supports_audio = False
    supports_polling = True
    supported_aspect_ratios = sorted(_LUMA_RATIOS)
    supported_resolutions = ["540p", "720p", "1080p", "4k"]
    max_seconds = 9

    KNOWN_MODELS = sorted(_KNOWN_MODELS)

    # USD per second placeholders; refine via local KB.
    VIDEO_PRICING: dict[str, dict[str, Any]] = {
        "ray-2": {"per_second": 0.16},
        "ray-flash-2": {"per_second": 0.08},
        "ray-1-6": {"per_second": 0.10},
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "ray-2",
        endpoint: str | None = None,
    ):
        self.api_key = _get_luma_api_key(api_key)
        self.model = model
        self.endpoint = (endpoint or os.getenv("LUMA_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

    @classmethod
    def list_models(cls, *, api_key: str | None = None, **kw: object) -> list[str] | None:
        key = _get_luma_api_key(api_key)
        if not key:
            return None
        return list(cls.KNOWN_MODELS)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("LUMA_API_KEY is not configured")

        body = _build_body(prompt, options, self.model)
        url = f"{self.endpoint}{_GENERATIONS_PATH}"

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, headers=self._headers(), json=body)
            if resp.status_code >= 400:
                raise RuntimeError(f"Luma submit failed: {_format_error(resp.status_code, resp)}")
            submitted = resp.json()
            task_id = submitted.get("id")
            if not task_id:
                raise RuntimeError(f"Luma response missing id: {submitted}")

            if not options.get("poll", True):
                return _pending_response(task_id, body, submitted)

            final = self._poll(
                client,
                task_id,
                poll_interval=float(options.get("poll_interval", _DEFAULT_POLL_INTERVAL)),
                max_retries=int(options.get("max_retries", _MAX_RETRIES)),
            )

        duration = float(options.get("duration", 0) or 0)
        cost = self._calculate_video_cost("luma", body["model"], duration_seconds=duration, n=1)
        return _make_video_response(final, body, cost)

    def _poll(
        self,
        client: httpx.Client,
        task_id: str,
        *,
        poll_interval: float,
        max_retries: int,
    ) -> dict[str, Any]:
        url = f"{self.endpoint}{_GENERATIONS_PATH}/{task_id}"
        for _ in range(max_retries):
            r = client.get(url, headers=self._headers())
            if r.status_code >= 400:
                raise RuntimeError(f"Luma poll failed: {_format_error(r.status_code, r)}")
            data = r.json()
            state = data.get("state")
            if state in _TERMINAL_OK:
                return data
            if state in _TERMINAL_FAIL:
                reason = data.get("failure_reason") or state
                raise RuntimeError(f"Luma task {task_id} {state}: {reason}")
            time.sleep(poll_interval)
        raise TimeoutError(f"Luma task {task_id} timed out after {max_retries} retries")
