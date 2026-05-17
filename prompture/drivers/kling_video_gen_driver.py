"""Kling AI video generation driver.

Routes between the upstream image-to-video and multi-image-to-video endpoints
based on the inputs in ``options``:

- ``image_list`` (>= 2) → ``POST /v1/videos/multi-image2video``
- ``image`` (one) or pure text → ``POST /v1/videos/image2video``

For motion-control workflows (extract motion from a reference video, apply to
a character image), pass ``character_image`` + ``motion_video`` — the driver
runs the two-step upload + create flow against ``/v1/videos/motion``.
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
from .kling_img_gen_driver import (
    _DEFAULT_ENDPOINT,
    _get_kling_credentials,
    _normalize_image,
    generate_kling_jwt,
)
from .video_gen_base import VideoGenDriver

logger = logging.getLogger(__name__)

_KLING_VIDEO_MODELS = {
    "kling-v1-6",
    "kling-v2-1",
    "kling-v2-1-master",
    "kling-v2-5-turbo",
    "kling-v2-6",
    "kling-v2-master",
}
_KLING_RATIOS = {"16:9", "9:16", "1:1"}
_PRO_ONLY_MODELS = {"kling-v2-6", "kling-v2-master"}


def _build_video_body(prompt: str, options: dict[str, Any], default_model: str) -> tuple[str, dict[str, Any]]:
    """Return (path, body) for a Kling video request."""
    model = options.get("model", default_model)
    if model not in _KLING_VIDEO_MODELS:
        logger.warning("Kling model %r not in known list", model)

    duration = int(options.get("duration", options.get("seconds", 5)))
    if duration not in (5, 10):
        raise ValueError("Kling duration must be 5 or 10")

    ratio = options.get("aspect_ratio", options.get("ratio", "16:9"))
    if ratio not in _KLING_RATIOS:
        raise ValueError(f"Unsupported aspect_ratio '{ratio}'")

    image_list = options.get("image_list")
    if image_list and len(image_list) > 1:
        body = {
            "model_name": "kling-v1-6",  # multi-image only supports v1-6 upstream
            "mode": "std",
            "duration": str(duration),
            "prompt": prompt or "",
            "aspect_ratio": ratio,
            "image_list": [{"image": _normalize_image(i)} for i in image_list],
        }
        return "/v1/videos/multi-image2video", body

    image = options.get("image")
    last_frame = options.get("last_frame") or options.get("image_tail")
    motion_video = options.get("motion_video")
    use_pro = bool(last_frame or motion_video) or model in _PRO_ONLY_MODELS

    body: dict[str, Any] = {
        "model_name": model,
        "mode": "pro" if use_pro else "std",
        "duration": str(duration),
        "aspect_ratio": ratio,
        "prompt": prompt or "",
    }
    if image is not None:
        body["image"] = _normalize_image(image)
    if last_frame is not None:
        body["image_tail"] = _normalize_image(last_frame)
    if motion_video is not None:
        body["motion_video"] = _normalize_image(motion_video)
    return "/v1/videos/image2video", body


class KlingVideoGenDriver(VideoCostMixin, VideoGenDriver):
    """Video generation via Kling AI."""

    supports_image_input = True
    supports_reference_images = True
    supports_video_input = False
    supports_audio = False
    supports_polling = True
    supported_aspect_ratios = sorted(_KLING_RATIOS)
    supported_resolutions = ["720p", "1080p"]
    max_seconds = 10

    KNOWN_MODELS = sorted(_KLING_VIDEO_MODELS)

    # USD per second placeholders — tune via local KB.
    VIDEO_PRICING: dict[str, dict[str, Any]] = {
        "kling-v1-6": {"per_second": 0.07},
        "kling-v2-1": {"per_second": 0.10},
        "kling-v2-1-master": {"per_second": 0.14},
        "kling-v2-5-turbo": {"per_second": 0.08},
        "kling-v2-6": {"per_second": 0.14},
        "kling-v2-master": {"per_second": 0.18},
    }

    def __init__(
        self,
        access_key: str | None = None,
        secret_key: str | None = None,
        model: str = "kling-v2-1",
        endpoint: str | None = None,
    ):
        ak, sk = _get_kling_credentials(access_key, secret_key)
        self.access_key = ak
        self.secret_key = sk
        self.model = model
        self.endpoint = (endpoint or os.getenv("KLING_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

    @classmethod
    def list_models(cls, **kw: object) -> list[str] | None:
        return list(cls.KNOWN_MODELS)

    def _token(self) -> str:
        if not self.access_key or not self.secret_key:
            raise RuntimeError("KLING_ACCESS_KEY and KLING_SECRET_KEY must be configured")
        return generate_kling_jwt(self.access_key, self.secret_key)

    def _headers(self, token: str) -> dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        token = self._token()

        # Motion-control workflow has its own two-step flow.
        if options.get("character_image") and options.get("motion_video_ref"):
            return self._motion_control(prompt, options, token)

        path, body = _build_video_body(prompt, options, self.model)
        is_multi = path.endswith("multi-image2video")
        poll_endpoint = "multi-image2video" if is_multi else "image2video"

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{self.endpoint}{path}", headers=self._headers(token), json=body)
            if resp.status_code >= 400:
                raise RuntimeError(f"Kling video API error {resp.status_code}: {resp.text}")
            result = resp.json()
            if result.get("code") != 0:
                raise RuntimeError(f"Kling video API error: {result.get('message')}")
            task_id = result.get("data", {}).get("task_id")
            if not task_id:
                raise RuntimeError(f"Kling video response missing task_id: {result}")

            if not options.get("poll", True):
                return self._pending_response(task_id, body, result)

            final = self._poll_video(
                client,
                task_id,
                token,
                endpoint=poll_endpoint,
                poll_interval=float(options.get("poll_interval", 5)),
                timeout_seconds=float(options.get("timeout", 600)),
            )

        return self._success_response(final, body, options)

    def _poll_video(
        self,
        client: httpx.Client,
        task_id: str,
        token: str,
        *,
        endpoint: str,
        poll_interval: float,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        path = f"/v1/videos/{endpoint}/{task_id}"
        deadline = time.monotonic() + timeout_seconds
        while True:
            r = client.get(f"{self.endpoint}{path}", headers=self._headers(token))
            if r.status_code >= 400:
                raise RuntimeError(f"Kling poll failed {r.status_code}: {r.text}")
            data = r.json()
            if data.get("code") != 0:
                raise RuntimeError(f"Kling poll error: {data.get('message')}")
            status = data.get("data", {}).get("task_status")
            if status == "succeed":
                return data
            if status == "failed":
                msg = data.get("data", {}).get("task_status_msg", "unknown")
                raise RuntimeError(f"Kling video task failed: {msg}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Kling video task {task_id} timed out (status={status})")
            time.sleep(poll_interval)

    def _motion_control(self, prompt: str, options: dict[str, Any], token: str) -> dict[str, Any]:
        """Run two-step motion-control workflow."""
        character = _normalize_image(options["character_image"])
        motion = _normalize_image(options["motion_video_ref"])
        duration = int(options.get("duration", 5))

        with httpx.Client(timeout=120.0) as client:
            up = client.post(
                f"{self.endpoint}/v1/videos/motion/upload",
                headers=self._headers(token),
                json={"video": motion},
            )
            if up.status_code >= 400:
                raise RuntimeError(f"Kling motion upload failed {up.status_code}: {up.text}")
            up_data = up.json()
            if up_data.get("code") != 0:
                raise RuntimeError(f"Kling motion upload error: {up_data.get('message')}")
            upload_task_id = up_data.get("data", {}).get("task_id")

            # Poll upload until succeeded → get work_id
            work_id = self._poll_motion_upload(client, upload_task_id, token, options)

            cr = client.post(
                f"{self.endpoint}/v1/videos/motion",
                headers=self._headers(token),
                json={
                    "work_id": work_id,
                    "image": character,
                    "prompt": prompt or "",
                    "duration": str(duration),
                    "mode": "pro",
                },
            )
            if cr.status_code >= 400:
                raise RuntimeError(f"Kling motion-create failed {cr.status_code}: {cr.text}")
            cr_data = cr.json()
            if cr_data.get("code") != 0:
                raise RuntimeError(f"Kling motion-create error: {cr_data.get('message')}")
            create_task_id = cr_data.get("data", {}).get("task_id")

            final = self._poll_motion_create(client, create_task_id, token, options)

        body = {"model_name": "kling-motion", "aspect_ratio": None, "duration": str(duration)}
        return self._success_response(final, body, options, mode="motion_control")

    def _poll_motion_upload(self, client: httpx.Client, task_id: str, token: str, options: dict[str, Any]) -> str:
        deadline = time.monotonic() + float(options.get("timeout", 600))
        interval = float(options.get("poll_interval", 3))
        while True:
            r = client.get(
                f"{self.endpoint}/v1/videos/motion/upload/{task_id}",
                headers=self._headers(token),
            )
            if r.status_code >= 400:
                raise RuntimeError(f"Motion upload poll failed: {r.text}")
            d = r.json()
            if d.get("code") != 0:
                raise RuntimeError(f"Motion upload poll error: {d.get('message')}")
            status = d.get("data", {}).get("task_status")
            if status == "succeed":
                work_id = d.get("data", {}).get("task_result", {}).get("work_id")
                if not work_id:
                    raise RuntimeError("Motion upload succeeded but no work_id")
                return work_id
            if status == "failed":
                raise RuntimeError(f"Motion extraction failed: {d}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Motion extraction timed out")
            time.sleep(interval)

    def _poll_motion_create(
        self, client: httpx.Client, task_id: str, token: str, options: dict[str, Any]
    ) -> dict[str, Any]:
        deadline = time.monotonic() + float(options.get("timeout", 600))
        interval = float(options.get("poll_interval", 5))
        while True:
            r = client.get(f"{self.endpoint}/v1/videos/motion/{task_id}", headers=self._headers(token))
            if r.status_code >= 400:
                raise RuntimeError(f"Motion-create poll failed: {r.text}")
            d = r.json()
            if d.get("code") != 0:
                raise RuntimeError(f"Motion-create poll error: {d.get('message')}")
            status = d.get("data", {}).get("task_status")
            if status == "succeed":
                return d
            if status == "failed":
                raise RuntimeError(f"Motion-create failed: {d}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Motion-create timed out")
            time.sleep(interval)

    def _pending_response(self, task_id: str, body: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "videos": [],
            "meta": {
                "video_count": 0,
                "duration_seconds": float(body.get("duration", 0) or 0),
                "aspect_ratio": body.get("aspect_ratio"),
                "resolution": None,
                "cost": 0.0,
                "model_name": f"kling/{body['model_name']}",
                "task_id": task_id,
                "status": "pending",
                "raw_response": raw,
            },
        }

    def _success_response(
        self,
        final: dict[str, Any],
        body: dict[str, Any],
        options: dict[str, Any],
        mode: str | None = None,
    ) -> dict[str, Any]:
        videos = final.get("data", {}).get("task_result", {}).get("videos") or []
        urls = [v.get("url") for v in videos if v.get("url")]
        video_contents = [video_from_url(u) for u in urls]
        duration = float(body.get("duration", 0) or 0)
        cost = self._calculate_video_cost(
            "kling",
            body["model_name"],
            duration_seconds=duration,
            n=max(len(video_contents), 1),
        )
        return {
            "videos": video_contents,
            "meta": {
                "video_count": len(video_contents),
                "duration_seconds": duration,
                "aspect_ratio": body.get("aspect_ratio"),
                "resolution": None,
                "cost": cost,
                "model_name": f"kling/{body['model_name']}",
                "mode": mode,
                "task_id": final.get("data", {}).get("task_id"),
                "raw_response": final,
            },
        }


class AsyncKlingVideoGenDriver(VideoCostMixin, AsyncVideoGenDriver):
    """Async video generation via Kling AI."""

    supports_image_input = KlingVideoGenDriver.supports_image_input
    supports_reference_images = KlingVideoGenDriver.supports_reference_images
    supports_video_input = KlingVideoGenDriver.supports_video_input
    supports_audio = KlingVideoGenDriver.supports_audio
    supports_polling = KlingVideoGenDriver.supports_polling
    supported_aspect_ratios = KlingVideoGenDriver.supported_aspect_ratios
    supported_resolutions = KlingVideoGenDriver.supported_resolutions
    max_seconds = KlingVideoGenDriver.max_seconds

    KNOWN_MODELS = KlingVideoGenDriver.KNOWN_MODELS
    VIDEO_PRICING = KlingVideoGenDriver.VIDEO_PRICING

    def __init__(
        self,
        access_key: str | None = None,
        secret_key: str | None = None,
        model: str = "kling-v2-1",
        endpoint: str | None = None,
    ):
        ak, sk = _get_kling_credentials(access_key, secret_key)
        self.access_key = ak
        self.secret_key = sk
        self.model = model
        self.endpoint = (endpoint or os.getenv("KLING_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

    def _token(self) -> str:
        if not self.access_key or not self.secret_key:
            raise RuntimeError("KLING_ACCESS_KEY and KLING_SECRET_KEY must be configured")
        return generate_kling_jwt(self.access_key, self.secret_key)

    def _headers(self, token: str) -> dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    async def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        token = self._token()
        path, body = _build_video_body(prompt, options, self.model)
        poll_endpoint = "multi-image2video" if path.endswith("multi-image2video") else "image2video"

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self.endpoint}{path}", headers=self._headers(token), json=body)
            if resp.status_code >= 400:
                raise RuntimeError(f"Kling video API error {resp.status_code}: {resp.text}")
            result = resp.json()
            if result.get("code") != 0:
                raise RuntimeError(f"Kling video API error: {result.get('message')}")
            task_id = result.get("data", {}).get("task_id")
            if not task_id:
                raise RuntimeError(f"Kling video response missing task_id: {result}")

            if not options.get("poll", True):
                # mirror sync pending shape
                helper = KlingVideoGenDriver(access_key=self.access_key, secret_key=self.secret_key)
                return helper._pending_response(task_id, body, result)

            final = await self._poll_video(
                client,
                task_id,
                token,
                endpoint=poll_endpoint,
                poll_interval=float(options.get("poll_interval", 5)),
                timeout_seconds=float(options.get("timeout", 600)),
            )

        helper = KlingVideoGenDriver(access_key=self.access_key, secret_key=self.secret_key)
        return helper._success_response(final, body, options)

    async def _poll_video(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        token: str,
        *,
        endpoint: str,
        poll_interval: float,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        path = f"/v1/videos/{endpoint}/{task_id}"
        deadline = time.monotonic() + timeout_seconds
        while True:
            r = await client.get(f"{self.endpoint}{path}", headers=self._headers(token))
            if r.status_code >= 400:
                raise RuntimeError(f"Kling poll failed {r.status_code}: {r.text}")
            data = r.json()
            if data.get("code") != 0:
                raise RuntimeError(f"Kling poll error: {data.get('message')}")
            status = data.get("data", {}).get("task_status")
            if status == "succeed":
                return data
            if status == "failed":
                msg = data.get("data", {}).get("task_status_msg", "unknown")
                raise RuntimeError(f"Kling video task failed: {msg}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Kling video task {task_id} timed out (status={status})")
            await asyncio.sleep(poll_interval)
