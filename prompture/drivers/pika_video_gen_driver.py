"""Pika Labs video generation driver.

.. note::
   The Pika public API is in flux and not well-documented externally at the
   time of writing. The shape used here reflects the **assumed** developer
   API surface, modelled after the public-facing job submit + poll pattern:

   - ``POST {endpoint}/generate/{version}/t2v`` — text-to-video submit
   - ``POST {endpoint}/generate/{version}/i2v`` — image-to-video submit
   - ``GET  {endpoint}/videos/<video_id>``       — poll status

   The submit response is assumed to look like
   ``{"video_id": "...", "status": "pending"}`` and the poll response like
   ``{"video_id": "...", "status": "finished", "result_url": "https://..."}``.

   The driver is structurally correct and testable with mocks. The exact
   endpoint URLs / payload field names should be reconciled against the
   real Pika developer documentation when it stabilizes.
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

_DEFAULT_ENDPOINT = "https://api.dev.pikalabs.org"
_KNOWN_MODELS = {"pika-2.2", "pika-2.1", "pika-1.5"}
_PIKA_RATIOS = {"16:9", "9:16", "1:1", "4:5", "5:4", "4:3", "3:4"}

_TERMINAL_OK = {"finished", "completed", "succeeded"}
_TERMINAL_FAIL = {"failed", "error", "cancelled", "canceled"}

_MAX_RETRIES = 60
_DEFAULT_POLL_INTERVAL = 5.0


def _get_pika_api_key(api_key: str | None = None) -> str | None:
    return api_key or os.getenv("PIKA_API_KEY")


def _model_version(model: str) -> str:
    """Map a Prompture model name to the URL version segment."""
    # "pika-2.2" → "2.2"
    if "-" in model:
        return model.split("-", 1)[1]
    return model


def _build_body(prompt: str, options: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return (op, body) where op is ``"t2v"`` or ``"i2v"``."""
    image = options.get("image") or options.get("image_url")

    aspect_ratio = options.get("aspect_ratio", options.get("ratio", "16:9"))
    if aspect_ratio not in _PIKA_RATIOS:
        raise ValueError(f"Unsupported aspect_ratio {aspect_ratio!r}; expected one of {sorted(_PIKA_RATIOS)}")

    api_options: dict[str, Any] = {
        "aspectRatio": aspect_ratio,
        "frameRate": int(options.get("frame_rate", options.get("frameRate", 24))),
        "duration": int(options.get("duration", options.get("seconds", 5))),
    }
    if "seed" in options:
        api_options["seed"] = int(options["seed"])
    if "negative_prompt" in options:
        api_options["negativePrompt"] = options["negative_prompt"]
    if "motion" in options:
        api_options["motion"] = options["motion"]

    body: dict[str, Any] = {"promptText": prompt or "", "options": api_options}
    if image:
        body["image"] = image
        return "i2v", body
    return "t2v", body


def _format_error(status_code: int, resp: httpx.Response) -> str:
    try:
        return f"{status_code}: {resp.json()}"
    except Exception:
        return f"{status_code}: {resp.text}"


def _make_video_response(
    final: dict[str, Any],
    op: str,
    model: str,
    body: dict[str, Any],
    cost: float,
) -> dict[str, Any]:
    url = final.get("result_url") or final.get("video_url") or final.get("url")
    videos = [video_from_url(url)] if url else []
    api_options = body.get("options", {})
    duration = api_options.get("duration")
    return {
        "videos": videos,
        "meta": {
            "video_count": len(videos),
            "duration_seconds": float(duration) if duration else None,
            "aspect_ratio": api_options.get("aspectRatio"),
            "resolution": api_options.get("resolution"),
            "cost": cost,
            "model_name": f"pika/{model}",
            "mode": op,
            "task_id": final.get("video_id") or final.get("id"),
            "status": final.get("status"),
            "raw_response": final,
        },
    }


def _pending_response(
    task_id: str,
    op: str,
    model: str,
    body: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    api_options = body.get("options", {})
    duration = api_options.get("duration")
    return {
        "videos": [],
        "meta": {
            "video_count": 0,
            "duration_seconds": float(duration) if duration else None,
            "aspect_ratio": api_options.get("aspectRatio"),
            "resolution": None,
            "cost": 0.0,
            "model_name": f"pika/{model}",
            "mode": op,
            "task_id": task_id,
            "status": raw.get("status", "pending"),
            "raw_response": raw,
        },
    }


class PikaVideoGenDriver(VideoCostMixin, VideoGenDriver):
    """Video generation via Pika Labs."""

    supports_image_input = True
    supports_reference_images = False
    supports_video_input = False
    supports_audio = False
    supports_polling = True
    supported_aspect_ratios = sorted(_PIKA_RATIOS)
    supported_resolutions = ["720p", "1080p"]
    max_seconds = 10

    KNOWN_MODELS = sorted(_KNOWN_MODELS)

    # USD per second placeholders; refine via local KB once Pika
    # publishes stable pricing.
    VIDEO_PRICING: dict[str, dict[str, Any]] = {
        "pika-2.2": {"per_second": 0.09},
        "pika-2.1": {"per_second": 0.08},
        "pika-1.5": {"per_second": 0.06},
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "pika-2.2",
        endpoint: str | None = None,
    ):
        self.api_key = _get_pika_api_key(api_key)
        self.model = model
        self.endpoint = (endpoint or os.getenv("PIKA_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

    @classmethod
    def list_models(cls, *, api_key: str | None = None, **kw: object) -> list[str] | None:
        if not _get_pika_api_key(api_key):
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
            raise RuntimeError("PIKA_API_KEY is not configured")
        model = options.get("model", self.model)
        if model not in _KNOWN_MODELS:
            logger.warning("Pika model %r not in known list %s", model, sorted(_KNOWN_MODELS))

        op, body = _build_body(prompt, options)
        version = _model_version(model)
        submit_url = f"{self.endpoint}/generate/{version}/{op}"

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(submit_url, headers=self._headers(), json=body)
            if resp.status_code >= 400:
                raise RuntimeError(f"Pika submit failed: {_format_error(resp.status_code, resp)}")
            submitted = resp.json()
            task_id = submitted.get("video_id") or submitted.get("id")
            if not task_id:
                raise RuntimeError(f"Pika response missing video_id: {submitted}")

            if not options.get("poll", True):
                return _pending_response(task_id, op, model, body, submitted)

            final = self._poll(
                client,
                task_id,
                poll_interval=float(options.get("poll_interval", _DEFAULT_POLL_INTERVAL)),
                max_retries=int(options.get("max_retries", _MAX_RETRIES)),
            )

        duration = float(body.get("options", {}).get("duration", 0) or 0)
        cost = self._calculate_video_cost("pika", model, duration_seconds=duration, n=1)
        return _make_video_response(final, op, model, body, cost)

    def _poll(
        self,
        client: httpx.Client,
        task_id: str,
        *,
        poll_interval: float,
        max_retries: int,
    ) -> dict[str, Any]:
        url = f"{self.endpoint}/videos/{task_id}"
        for _ in range(max_retries):
            r = client.get(url, headers=self._headers())
            if r.status_code >= 400:
                raise RuntimeError(f"Pika poll failed: {_format_error(r.status_code, r)}")
            data = r.json()
            status = (data.get("status") or "").lower()
            if status in _TERMINAL_OK:
                return data
            if status in _TERMINAL_FAIL:
                reason = data.get("error") or data.get("message") or status
                raise RuntimeError(f"Pika task {task_id} {status}: {reason}")
            time.sleep(poll_interval)
        raise TimeoutError(f"Pika task {task_id} timed out after {max_retries} retries")
