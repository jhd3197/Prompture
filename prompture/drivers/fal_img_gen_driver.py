"""Fal.ai image generation driver.

Same queue API as :mod:`fal_video_gen_driver`.  Model strings are arbitrary
Fal model ids (e.g. ``fal-ai/flux/dev``, ``fal-ai/stable-diffusion-v35-medium``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_url
from .async_img_gen_base import AsyncImageGenDriver
from .fal_video_gen_driver import _QUEUE_BASE, _get_fal_api_key
from .img_gen_base import ImageGenDriver

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "fal-ai/flux/dev"


def _build_img_input(prompt: str, options: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if prompt:
        payload["prompt"] = prompt
    for key in ("image_url", "image_size", "num_images", "seed", "negative_prompt"):
        if key in options:
            payload[key] = options[key]
    if "image" in options and "image_url" not in options:
        payload["image_url"] = options["image"]
    if "size" in options and "image_size" not in options:
        payload["image_size"] = options["size"]
    extra = options.get("extra")
    if isinstance(extra, dict):
        payload.update(extra)
    return payload


def _extract_image_urls(result: dict[str, Any]) -> list[str]:
    images = result.get("images")
    if isinstance(images, list):
        return [img["url"] for img in images if isinstance(img, dict) and img.get("url")]
    image = result.get("image")
    if isinstance(image, dict) and image.get("url"):
        return [image["url"]]
    return []


class FalImageGenDriver(ImageCostMixin, ImageGenDriver):
    """Image generation via Fal.ai queue API."""

    supports_multiple = True
    supports_size_variants = True
    supported_sizes = []
    max_images = 4

    KNOWN_MODELS = [
        "fal-ai/flux/dev",
        "fal-ai/flux/schnell",
        "fal-ai/stable-diffusion-v35-medium",
        "fal-ai/stable-diffusion-v35-large",
        "fal-ai/kling-v1-5/text-to-image",
    ]

    IMAGE_PRICING: dict[str, dict[str, float]] = {}

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

    def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("FAL_API_KEY is not configured")
        model_id = options.get("model", self.model)
        input_payload = _build_img_input(prompt, options)

        with httpx.Client(timeout=120.0) as client:
            submit = client.post(
                f"{self.endpoint}/{model_id}", headers=self._headers(), json=input_payload
            )
            if submit.status_code >= 400:
                raise RuntimeError(f"Fal submit failed {submit.status_code}: {submit.text}")
            submitted = submit.json()
            request_id = submitted.get("request_id")
            if not request_id:
                raise RuntimeError(f"Fal response missing request_id: {submitted}")

            if not options.get("poll", True):
                return {
                    "images": [],
                    "meta": {
                        "image_count": 0,
                        "size": options.get("image_size") or options.get("size"),
                        "revised_prompt": None,
                        "cost": 0.0,
                        "model_name": f"fal/{model_id}",
                        "request_id": request_id,
                        "status": "pending",
                        "raw_response": submitted,
                    },
                }

            self._poll_status(
                client,
                model_id,
                request_id,
                poll_interval=float(options.get("poll_interval", 3)),
                timeout_seconds=float(options.get("timeout", 300)),
            )
            result_url = f"{self.endpoint}/{model_id}/requests/{request_id}"
            r = client.get(result_url, headers=self._headers())
            if r.status_code >= 400:
                raise RuntimeError(f"Fal result fetch failed: {r.text}")
            result = r.json()

        urls = _extract_image_urls(result)
        images = [image_from_url(u) for u in urls]
        cost = self._calculate_image_cost("fal", model_id, n=max(len(images), 1))
        return {
            "images": images,
            "meta": {
                "image_count": len(images),
                "size": options.get("image_size") or options.get("size"),
                "revised_prompt": None,
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


class AsyncFalImageGenDriver(ImageCostMixin, AsyncImageGenDriver):
    """Async image generation via Fal.ai."""

    supports_multiple = FalImageGenDriver.supports_multiple
    supports_size_variants = FalImageGenDriver.supports_size_variants
    supported_sizes = FalImageGenDriver.supported_sizes
    max_images = FalImageGenDriver.max_images

    KNOWN_MODELS = FalImageGenDriver.KNOWN_MODELS
    IMAGE_PRICING = FalImageGenDriver.IMAGE_PRICING

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

    async def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("FAL_API_KEY is not configured")
        model_id = options.get("model", self.model)
        input_payload = _build_img_input(prompt, options)

        async with httpx.AsyncClient(timeout=120.0) as client:
            submit = await client.post(
                f"{self.endpoint}/{model_id}", headers=self._headers(), json=input_payload
            )
            if submit.status_code >= 400:
                raise RuntimeError(f"Fal submit failed {submit.status_code}: {submit.text}")
            submitted = submit.json()
            request_id = submitted.get("request_id")
            if not request_id:
                raise RuntimeError(f"Fal response missing request_id: {submitted}")

            if not options.get("poll", True):
                return {
                    "images": [],
                    "meta": {
                        "image_count": 0,
                        "size": options.get("image_size") or options.get("size"),
                        "revised_prompt": None,
                        "cost": 0.0,
                        "model_name": f"fal/{model_id}",
                        "request_id": request_id,
                        "status": "pending",
                        "raw_response": submitted,
                    },
                }

            await self._poll_status(
                client,
                model_id,
                request_id,
                poll_interval=float(options.get("poll_interval", 3)),
                timeout_seconds=float(options.get("timeout", 300)),
            )
            url = f"{self.endpoint}/{model_id}/requests/{request_id}"
            r = await client.get(url, headers=self._headers())
            if r.status_code >= 400:
                raise RuntimeError(f"Fal result fetch failed: {r.text}")
            result = r.json()

        urls = _extract_image_urls(result)
        images = [image_from_url(u) for u in urls]
        cost = self._calculate_image_cost("fal", model_id, n=max(len(images), 1))
        return {
            "images": images,
            "meta": {
                "image_count": len(images),
                "size": options.get("image_size") or options.get("size"),
                "revised_prompt": None,
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
