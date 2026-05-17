"""MiniMax (Hailuo) AI video generation driver.

Wraps the four ``POST /v1/video_generation`` modes exposed by MiniMax:

- **T2V** — Text-to-Video (default)
- **I2V** — Image-to-Video (first frame supplied)
- **FL2V** — First-and-Last-Frame Video (first + last frame; forces ``MiniMax-Hailuo-02``)
- **S2V** — Subject-Reference Video (uses ``S2V-01`` with a character reference image)

The endpoint returns a ``task_id``; the driver then polls
``GET /v1/query/video_generation`` and resolves the resulting ``file_id`` to
a download URL via ``GET /v1/files/retrieve``.
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

_DEFAULT_ENDPOINT = "https://api.minimax.io/v1"

_KNOWN_MODELS = {
    "MiniMax-Hailuo-2.3",
    "MiniMax-Hailuo-2.3-Fast",
    "MiniMax-Hailuo-02",
    "S2V-01",
}
_RESOLUTIONS = {"768P", "1080P"}


def _get_minimax_api_key(api_key: str | None = None) -> str | None:
    return api_key or os.getenv("MINIMAX_API_KEY") or os.getenv("HAILUO_API_KEY")


def _normalize_image(value: str, mime: str = "image/jpeg") -> str:
    """Return either an http(s) URL or a base64 data URL."""
    if value.startswith(("http://", "https://", "data:")):
        return value
    return f"data:{mime};base64,{value}"


def _duration(options: dict[str, Any]) -> int:
    """MiniMax supports only 6 and 10 second outputs."""
    raw = int(options.get("duration", options.get("seconds", 6)))
    return 10 if raw > 8 else 6


def _resolution(options: dict[str, Any]) -> str:
    r = str(options.get("resolution", "768P")).upper()
    return r if r in _RESOLUTIONS else "768P"


def _select_model(model: str, has_first: bool, has_last: bool) -> str:
    # FL2V requires Hailuo-02
    if has_first and has_last:
        return "MiniMax-Hailuo-02"
    return model


def _build_body(prompt: str, options: dict[str, Any], default_model: str) -> dict[str, Any]:
    image = options.get("image")
    last_frame = options.get("last_frame")
    subject_image = options.get("subject_image")

    if subject_image:
        # S2V mode
        return {
            "model": "S2V-01",
            "prompt": prompt or "",
            "subject_reference": [{"type": "character", "image": [_normalize_image(subject_image)]}],
        }

    model = _select_model(options.get("model", default_model), bool(image), bool(last_frame))

    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt or "",
        "duration": _duration(options),
        "resolution": _resolution(options),
        "aspect_ratio": options.get("aspect_ratio", "16:9"),
    }
    if image:
        body["first_frame_image"] = _normalize_image(image)
    if last_frame:
        body["last_frame_image"] = _normalize_image(last_frame)
    return body


def _check_base_resp(payload: dict[str, Any]) -> None:
    base = payload.get("base_resp")
    if isinstance(base, dict) and base.get("status_code", 0) != 0:
        raise RuntimeError(f"MiniMax API error: {base.get('status_msg')}")


class MiniMaxVideoGenDriver(VideoCostMixin, VideoGenDriver):
    """Video generation via MiniMax / Hailuo."""

    supports_image_input = True
    supports_reference_images = True
    supports_video_input = False
    supports_audio = False
    supports_polling = True
    supported_aspect_ratios = ["16:9", "9:16"]
    supported_resolutions = ["768P", "1080P"]
    max_seconds = 10

    KNOWN_MODELS = sorted(_KNOWN_MODELS)

    # USD per second placeholders — refine via local KB.
    VIDEO_PRICING: dict[str, dict[str, Any]] = {
        "MiniMax-Hailuo-2.3": {"per_second": 0.07},
        "MiniMax-Hailuo-2.3-Fast": {"per_second": 0.04},
        "MiniMax-Hailuo-02": {"per_second": 0.09},
        "S2V-01": {"per_second": 0.08},
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "MiniMax-Hailuo-2.3",
        endpoint: str | None = None,
    ):
        self.api_key = _get_minimax_api_key(api_key)
        self.model = model
        self.endpoint = (endpoint or os.getenv("MINIMAX_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

    @classmethod
    def list_models(cls, **kw: object) -> list[str] | None:
        return list(cls.KNOWN_MODELS)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY is not configured")
        body = _build_body(prompt, options, self.model)

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{self.endpoint}/video_generation", headers=self._headers(), json=body)
            if resp.status_code >= 400:
                raise RuntimeError(f"MiniMax API error {resp.status_code}: {resp.text}")
            result = resp.json()
            _check_base_resp(result)
            task_id = result.get("task_id")
            if not task_id:
                raise RuntimeError(f"MiniMax response missing task_id: {result}")

            if not options.get("poll", True):
                return self._pending_response(task_id, body, result)

            file_id, status_data = self._poll_task(
                client,
                task_id,
                poll_interval=float(options.get("poll_interval", 5)),
                timeout_seconds=float(options.get("timeout", 600)),
            )
            video_url = self._resolve_file(client, file_id)

        video = video_from_url(video_url)
        duration = float(body.get("duration", 0))
        cost = self._calculate_video_cost("minimax", body["model"], duration_seconds=duration, n=1)
        return {
            "videos": [video],
            "meta": {
                "video_count": 1,
                "duration_seconds": duration,
                "aspect_ratio": body.get("aspect_ratio"),
                "resolution": body.get("resolution"),
                "cost": cost,
                "model_name": f"minimax/{body['model']}",
                "task_id": task_id,
                "raw_response": {"task": status_data, "url": video_url},
            },
        }

    def _poll_task(
        self,
        client: httpx.Client,
        task_id: str,
        *,
        poll_interval: float,
        timeout_seconds: float,
    ) -> tuple[str, dict[str, Any]]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            r = client.get(
                f"{self.endpoint}/query/video_generation",
                params={"task_id": task_id},
                headers=self._headers(),
            )
            if r.status_code >= 400:
                raise RuntimeError(f"MiniMax poll failed {r.status_code}: {r.text}")
            data = r.json()
            _check_base_resp(data)
            status = data.get("status")
            if status == "Success":
                file_id = data.get("file_id")
                if not file_id:
                    raise RuntimeError("MiniMax success but no file_id")
                return file_id, data
            if status == "Fail":
                raise RuntimeError(f"MiniMax generation failed: {data}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"MiniMax task {task_id} timed out (status={status})")
            time.sleep(poll_interval)

    def _resolve_file(self, client: httpx.Client, file_id: str) -> str:
        r = client.get(
            f"{self.endpoint}/files/retrieve",
            params={"file_id": file_id},
            headers=self._headers(),
        )
        if r.status_code >= 400:
            raise RuntimeError(f"MiniMax file retrieve failed: {r.text}")
        data = r.json()
        _check_base_resp(data)
        url = (data.get("file") or {}).get("download_url")
        if not url:
            raise RuntimeError(f"MiniMax file response missing download_url: {data}")
        return url

    def _pending_response(self, task_id: str, body: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "videos": [],
            "meta": {
                "video_count": 0,
                "duration_seconds": float(body.get("duration", 0) or 0),
                "aspect_ratio": body.get("aspect_ratio"),
                "resolution": body.get("resolution"),
                "cost": 0.0,
                "model_name": f"minimax/{body['model']}",
                "task_id": task_id,
                "status": "pending",
                "raw_response": raw,
            },
        }


class AsyncMiniMaxVideoGenDriver(VideoCostMixin, AsyncVideoGenDriver):
    """Async video generation via MiniMax / Hailuo."""

    supports_image_input = MiniMaxVideoGenDriver.supports_image_input
    supports_reference_images = MiniMaxVideoGenDriver.supports_reference_images
    supports_video_input = MiniMaxVideoGenDriver.supports_video_input
    supports_audio = MiniMaxVideoGenDriver.supports_audio
    supports_polling = MiniMaxVideoGenDriver.supports_polling
    supported_aspect_ratios = MiniMaxVideoGenDriver.supported_aspect_ratios
    supported_resolutions = MiniMaxVideoGenDriver.supported_resolutions
    max_seconds = MiniMaxVideoGenDriver.max_seconds

    KNOWN_MODELS = MiniMaxVideoGenDriver.KNOWN_MODELS
    VIDEO_PRICING = MiniMaxVideoGenDriver.VIDEO_PRICING

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "MiniMax-Hailuo-2.3",
        endpoint: str | None = None,
    ):
        self.api_key = _get_minimax_api_key(api_key)
        self.model = model
        self.endpoint = (endpoint or os.getenv("MINIMAX_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY is not configured")
        body = _build_body(prompt, options, self.model)

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self.endpoint}/video_generation", headers=self._headers(), json=body)
            if resp.status_code >= 400:
                raise RuntimeError(f"MiniMax API error {resp.status_code}: {resp.text}")
            result = resp.json()
            _check_base_resp(result)
            task_id = result.get("task_id")
            if not task_id:
                raise RuntimeError(f"MiniMax response missing task_id: {result}")

            if not options.get("poll", True):
                helper = MiniMaxVideoGenDriver(api_key=self.api_key)
                return helper._pending_response(task_id, body, result)

            file_id, status_data = await self._poll_task(
                client,
                task_id,
                poll_interval=float(options.get("poll_interval", 5)),
                timeout_seconds=float(options.get("timeout", 600)),
            )
            video_url = await self._resolve_file(client, file_id)

        video = video_from_url(video_url)
        duration = float(body.get("duration", 0))
        cost = self._calculate_video_cost("minimax", body["model"], duration_seconds=duration, n=1)
        return {
            "videos": [video],
            "meta": {
                "video_count": 1,
                "duration_seconds": duration,
                "aspect_ratio": body.get("aspect_ratio"),
                "resolution": body.get("resolution"),
                "cost": cost,
                "model_name": f"minimax/{body['model']}",
                "task_id": task_id,
                "raw_response": {"task": status_data, "url": video_url},
            },
        }

    async def _poll_task(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        *,
        poll_interval: float,
        timeout_seconds: float,
    ) -> tuple[str, dict[str, Any]]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            r = await client.get(
                f"{self.endpoint}/query/video_generation",
                params={"task_id": task_id},
                headers=self._headers(),
            )
            if r.status_code >= 400:
                raise RuntimeError(f"MiniMax poll failed {r.status_code}: {r.text}")
            data = r.json()
            _check_base_resp(data)
            status = data.get("status")
            if status == "Success":
                file_id = data.get("file_id")
                if not file_id:
                    raise RuntimeError("MiniMax success but no file_id")
                return file_id, data
            if status == "Fail":
                raise RuntimeError(f"MiniMax generation failed: {data}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"MiniMax task {task_id} timed out (status={status})")
            await asyncio.sleep(poll_interval)

    async def _resolve_file(self, client: httpx.AsyncClient, file_id: str) -> str:
        r = await client.get(
            f"{self.endpoint}/files/retrieve",
            params={"file_id": file_id},
            headers=self._headers(),
        )
        if r.status_code >= 400:
            raise RuntimeError(f"MiniMax file retrieve failed: {r.text}")
        data = r.json()
        _check_base_resp(data)
        url = (data.get("file") or {}).get("download_url")
        if not url:
            raise RuntimeError(f"MiniMax file response missing download_url: {data}")
        return url
