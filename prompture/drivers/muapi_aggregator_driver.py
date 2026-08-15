"""Muapi.ai aggregator driver (sync) — reference subclass of the aggregator base.

Muapi fronts a large catalog of image/video/edit models behind a single
``x-api-key`` and a uniform submit→poll API:

- ``POST {base}/api/v1/{slug}`` — submit; returns ``{"request_id": ...}``.
- ``GET  {base}/api/v1/predictions/{request_id}/result`` — poll; payload carries
  ``status`` + ``outputs`` when finished.
- ``POST {base}/api/v1/upload_file`` — multipart upload → hosted URL.

The Prompture ``model`` string is the Muapi endpoint slug, e.g.
``"muapi/nano-banana-pro"``. All the HTTP/job machinery lives in
:mod:`aggregator_base`; this module supplies only Muapi's URL shapes, auth, and
payload builders — the template to clone for Replicate / Novita / PiAPI.
"""

from __future__ import annotations

import os
from typing import Any

from ..infra.cost_mixin import ImageCostMixin, VideoCostMixin
from .aggregator_base import (
    AggregatorImageDriver,
    AggregatorLipsyncDriver,
    AggregatorMusicDriver,
    AggregatorVideoDriver,
    extract_media_urls,
)

_DEFAULT_BASE = "https://api.muapi.ai"
_DEFAULT_IMAGE_MODEL = "nano-banana"
_DEFAULT_VIDEO_MODEL = "kling-video-v2-1"
_DEFAULT_LIPSYNC_MODEL = "infinite-talk"
_DEFAULT_MUSIC_MODEL = "suno-create-music"

_IMAGE_KEYS = (
    "aspect_ratio",
    "resolution",
    "quality",
    "seed",
    "negative_prompt",
    "num_images",
    "image_url",
    "images_list",
    "strength",
)
_VIDEO_KEYS = (
    "aspect_ratio",
    "resolution",
    "quality",
    "duration",
    "mode",
    "image_url",
    "video_url",
    "last_image",
    "end_image_url",
    "request_id",
    "negative_prompt",
    "seed",
)

# Backwards-compatible module-level helpers (also used by the async sibling).
_extract_urls = extract_media_urls


def _get_muapi_key(api_key: str | None = None) -> str | None:
    return api_key or os.getenv("MUAPI_API_KEY") or os.getenv("MUAPI_KEY")


def _build_payload(prompt: str, options: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if prompt:
        payload["prompt"] = prompt
    for k in keys:
        v = options.get(k)
        if v is not None:
            payload[k] = v
    extra = options.get("extra")
    if isinstance(extra, dict):
        payload.update(extra)
    return payload


class _MuapiMixin:
    """Muapi-specific overrides shared by the sync and async drivers."""

    PROVIDER = "muapi"
    DEFAULT_BASE = _DEFAULT_BASE
    API_KEY_ENVS = ("MUAPI_API_KEY", "MUAPI_KEY")
    ENDPOINT_ENV = "MUAPI_ENDPOINT"
    AUTH_STYLE = "x-api-key"
    DEFAULT_IMAGE_MODEL = _DEFAULT_IMAGE_MODEL
    DEFAULT_VIDEO_MODEL = _DEFAULT_VIDEO_MODEL
    DEFAULT_LIPSYNC_MODEL = _DEFAULT_LIPSYNC_MODEL
    DEFAULT_MUSIC_MODEL = _DEFAULT_MUSIC_MODEL

    def submit_url(self, slug: str) -> str:
        return f"{self.endpoint}/api/v1/{slug}"

    def result_url(self, request_id: str) -> str:
        return f"{self.endpoint}/api/v1/predictions/{request_id}/result"

    def upload_url(self) -> str:
        return f"{self.endpoint}/api/v1/upload_file"

    def build_image_payload(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return _build_payload(prompt, options, _IMAGE_KEYS)

    def build_video_payload(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return _build_payload(prompt, options, _VIDEO_KEYS)


class MuapiImageGenDriver(ImageCostMixin, _MuapiMixin, AggregatorImageDriver):
    """Image generation + i2i editing via the Muapi aggregator."""

    IMAGE_PRICING: dict[str, dict[str, float]] = {}


class MuapiVideoGenDriver(VideoCostMixin, _MuapiMixin, AggregatorVideoDriver):
    """Video generation (t2v / i2v / v2v) via the Muapi aggregator."""

    VIDEO_PRICING: dict[str, dict[str, Any]] = {}


class MuapiLipsyncDriver(_MuapiMixin, AggregatorLipsyncDriver):
    """Lipsync (image|video + audio → video) via the Muapi aggregator."""


class MuapiMusicGenDriver(_MuapiMixin, AggregatorMusicDriver):
    """Music generation (Suno: create / remix / extend / mashup) via Muapi."""
