"""Runway video generation driver.

A single driver covers the three Runway video endpoints:

- ``POST /v1/text_to_video``   — text-only generation
- ``POST /v1/image_to_video``  — text + image(s) with optional ``first``/``last`` keyframes
- ``POST /v1/video_to_video``  — video + text (gen4_aleph)

The endpoint is chosen from the ``mode`` option, or auto-detected from the
inputs: ``video`` ⇒ video_to_video, ``image`` ⇒ image_to_video, else
text_to_video. Each call returns a Runway task ID; the driver polls
``GET /v1/tasks/{id}`` until terminal and returns the first output video URL
as a :class:`VideoContent`.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from ..infra.cost_mixin import VideoCostMixin
from ..media.video import VideoContent, video_from_url
from .runway_img_gen_driver import (
    _API_VERSION,
    _DEFAULT_ENDPOINT,
    _TERMINAL_FAIL,
    _TERMINAL_OK,
    _format_error,
    _get_runway_api_key,
)
from .video_gen_base import VideoGenDriver

logger = logging.getLogger(__name__)

# Ratios accepted per endpoint, per the official docs.
_T2V_RATIOS: set[str] = {"1280:720", "720:1280"}
_I2V_RATIOS: set[str] = {
    "1280:720",
    "720:1280",
    "1104:832",
    "960:960",
    "832:1104",
    "1584:672",
}

_MODE_ENDPOINTS = {
    "text_to_video": "/v1/text_to_video",
    "image_to_video": "/v1/image_to_video",
    "video_to_video": "/v1/video_to_video",
}

_KNOWN_MODELS_BY_MODE: dict[str, set[str]] = {
    "text_to_video": {"gen4.5", "veo3", "veo3.1", "veo3.1_fast"},
    "image_to_video": {
        "gen4.5",
        "gen4_turbo",
        "gen3a_turbo",
        "veo3",
        "veo3.1",
        "veo3.1_fast",
    },
    "video_to_video": {"gen4_aleph"},
}


def _detect_mode(options: dict[str, Any]) -> str:
    if options.get("mode"):
        mode = options["mode"]
        if mode not in _MODE_ENDPOINTS:
            raise ValueError(f"Unknown mode {mode!r}. Expected one of {sorted(_MODE_ENDPOINTS)}.")
        return mode
    if options.get("video") or options.get("videoUri") or options.get("video_uri"):
        return "video_to_video"
    if options.get("image") or options.get("promptImage") or options.get("prompt_image"):
        return "image_to_video"
    return "text_to_video"


def _normalize_prompt_image(image: Any) -> Any:
    """Return a value valid for ``promptImage``: a string URI or a list of keyframe dicts."""
    if isinstance(image, str):
        return image
    if isinstance(image, dict):
        uri = image.get("uri") or image.get("url")
        if not uri:
            raise ValueError("image dict must include 'uri'")
        position = image.get("position", "first")
        return [{"uri": uri, "position": position}]
    if isinstance(image, list):
        out: list[dict[str, str]] = []
        for item in image:
            if isinstance(item, str):
                out.append({"uri": item, "position": "first"})
            elif isinstance(item, dict):
                uri = item.get("uri") or item.get("url")
                if not uri:
                    raise ValueError("image entries must include 'uri'")
                out.append({"uri": uri, "position": item.get("position", "first")})
            else:
                raise ValueError(f"unsupported image entry type: {type(item).__name__}")
        return out
    raise ValueError(f"unsupported image input type: {type(image).__name__}")


def _normalize_references(refs: Any) -> list[dict[str, str]]:
    if not refs:
        return []
    if not isinstance(refs, list):
        raise ValueError("references must be a list of {type, uri} dicts")
    out: list[dict[str, str]] = []
    for ref in refs:
        if not isinstance(ref, dict):
            raise ValueError("each reference must be a dict")
        uri = ref.get("uri") or ref.get("url")
        if not uri:
            raise ValueError("reference entries require 'uri'")
        out.append({"type": ref.get("type", "image"), "uri": str(uri)})
    return out


class RunwayVideoGenDriver(VideoCostMixin, VideoGenDriver):
    """Video generation via Runway text/image/video → video endpoints."""

    supports_image_input = True
    supports_reference_images = True
    supports_video_input = True
    supports_audio = False
    supports_polling = True
    supported_aspect_ratios = sorted(_T2V_RATIOS | _I2V_RATIOS)
    supported_resolutions = ["720p", "1080p"]
    max_seconds = 10

    KNOWN_MODELS = sorted({m for s in _KNOWN_MODELS_BY_MODE.values() for m in s})

    # USD per second of output. Credits → USD at $0.01/credit per Runway docs.
    VIDEO_PRICING: dict[str, dict[str, Any]] = {
        "gen4.5": {"per_second": 0.12},
        "gen4_turbo": {"per_second": 0.05},
        "gen3a_turbo": {"per_second": 0.05},
        "gen4_aleph": {"per_second": 0.15},
        "veo3": {"per_second": 0.40},
        "veo3.1": {"per_second": 0.30},
        "veo3.1_fast": {"per_second": 0.12},
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gen4.5",
        endpoint: str | None = None,
    ):
        self.api_key = _get_runway_api_key(api_key)
        self.model = model
        self.endpoint = (endpoint or os.getenv("RUNWAY_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

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

    def _build(self, prompt: str, options: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        """Return (mode, path, body) for the request."""
        mode = _detect_mode(options)
        model = options.get("model", self.model)

        if model not in _KNOWN_MODELS_BY_MODE[mode]:
            logger.warning(
                "model %r is not in the known list for %s: %s",
                model,
                mode,
                sorted(_KNOWN_MODELS_BY_MODE[mode]),
            )

        body: dict[str, Any] = {"model": model, "promptText": prompt}

        # ── ratio ───────────────────────────────────────────────────
        if mode != "video_to_video":
            ratio = options.get("ratio") or options.get("aspect_ratio")
            if mode == "text_to_video":
                ratio = ratio or "1280:720"
                if ratio not in _T2V_RATIOS:
                    raise ValueError(f"text_to_video ratio must be one of {sorted(_T2V_RATIOS)}, got {ratio!r}")
            else:  # image_to_video
                ratio = ratio or "1280:720"
                if ratio not in _I2V_RATIOS:
                    raise ValueError(f"image_to_video ratio must be one of {sorted(_I2V_RATIOS)}, got {ratio!r}")
            body["ratio"] = ratio

        # ── duration ────────────────────────────────────────────────
        if mode in {"text_to_video", "image_to_video"}:
            duration = int(options.get("duration", options.get("seconds", 5)))
            if not 2 <= duration <= 10:
                raise ValueError("duration must be an integer in [2, 10]")
            body["duration"] = duration

        # ── inputs per mode ─────────────────────────────────────────
        if mode == "image_to_video":
            image = options.get("image") or options.get("promptImage") or options.get("prompt_image")
            if not image:
                raise ValueError("image_to_video requires 'image' (URL/runway://uri/data URI or list of keyframes)")
            body["promptImage"] = _normalize_prompt_image(image)

        elif mode == "video_to_video":
            video_uri = options.get("video") or options.get("videoUri") or options.get("video_uri")
            if not video_uri or not isinstance(video_uri, str):
                raise ValueError("video_to_video requires 'video' as a URI string")
            body["videoUri"] = video_uri
            refs = _normalize_references(options.get("references") or options.get("reference_images"))
            if refs:
                if len(refs) > 1:
                    raise ValueError("video_to_video supports at most 1 reference image")
                body["references"] = refs

        # ── shared optional fields ──────────────────────────────────
        seed = options.get("seed")
        if seed is not None:
            body["seed"] = int(seed)

        content_mod = options.get("content_moderation") or options.get("contentModeration")
        if content_mod is not None:
            body["contentModeration"] = content_mod

        return mode, _MODE_ENDPOINTS[mode], body

    def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate a video. See module docstring for endpoint selection rules.

        Options:
            mode: ``text_to_video`` | ``image_to_video`` | ``video_to_video``.
                  Auto-detected if omitted.
            model: Model override (default: instance ``model``, e.g. ``gen4.5``).
            ratio / aspect_ratio: Endpoint-specific ratio string.
            duration / seconds: 2–10 (T2V, I2V). Ignored for V2V.
            image / promptImage: Required for I2V. String URI, dict with
                ``uri``/``position``, or list of keyframe dicts.
            video / videoUri: Required for V2V. URI string.
            references: Optional V2V references — ``[{"type":"image","uri":...}]``.
            seed, content_moderation: passthrough.
            poll, poll_interval, timeout: Control task polling.
        """
        if not self.api_key:
            raise RuntimeError("RUNWAY_API_KEY (or RUNWAYML_API_SECRET) is not configured")
        if not prompt:
            raise ValueError("prompt cannot be empty")

        mode, path, body = self._build(prompt, options)
        headers = self._headers()

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{self.endpoint}{path}", headers=headers, json=body)
            if resp.status_code >= 400:
                raise RuntimeError(f"Runway {mode} failed: {_format_error(resp.status_code, resp)}")
            task = resp.json()
            task_id = task.get("id")
            if not task_id:
                raise RuntimeError(f"Runway response missing task id: {task}")

            if not options.get("poll", True):
                return self._pending_response(task_id, mode, body, task)

            final = self._poll_until_done(
                client,
                task_id,
                headers=headers,
                poll_interval=float(options.get("poll_interval", 5)),
                timeout_seconds=float(options.get("timeout", 600)),
            )

        return self._success_response(final, mode, body, options)

    # ------------------------------------------------------------------

    def _poll_until_done(
        self,
        client: httpx.Client,
        task_id: str,
        *,
        headers: dict[str, str],
        poll_interval: float,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            r = client.get(f"{self.endpoint}/v1/tasks/{task_id}", headers=headers)
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
            time.sleep(poll_interval)

    def _pending_response(
        self,
        task_id: str,
        mode: str,
        body: dict[str, Any],
        task: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "videos": [],
            "meta": {
                "video_count": 0,
                "duration_seconds": body.get("duration"),
                "aspect_ratio": body.get("ratio"),
                "resolution": None,
                "cost": 0.0,
                "model_name": f"runway/{body['model']}",
                "mode": mode,
                "task_id": task_id,
                "status": task.get("status", "PENDING"),
                "raw_response": task,
            },
        }

    def _success_response(
        self,
        final: dict[str, Any],
        mode: str,
        body: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        outputs = final.get("output") or []
        videos: list[VideoContent] = [video_from_url(url) for url in outputs]

        duration = body.get("duration") or 0
        model = body["model"]
        cost = self._calculate_video_cost(
            "runway",
            model,
            duration_seconds=float(duration) if duration else 0.0,
            n=max(len(videos), 1),
        )

        return {
            "videos": videos,
            "meta": {
                "video_count": len(videos),
                "duration_seconds": float(duration) if duration else None,
                "aspect_ratio": body.get("ratio"),
                "resolution": None,
                "cost": cost,
                "model_name": f"runway/{model}",
                "mode": mode,
                "task_id": final.get("id"),
                "status": final.get("status"),
                "raw_response": final,
            },
        }
