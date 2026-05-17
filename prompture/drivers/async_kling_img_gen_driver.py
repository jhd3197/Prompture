"""Async Kling AI image generation driver."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_url
from .async_img_gen_base import AsyncImageGenDriver
from .kling_img_gen_driver import (
    _DEFAULT_ENDPOINT,
    KlingImageGenDriver,
    _get_kling_credentials,
    generate_kling_jwt,
)

logger = logging.getLogger(__name__)


class AsyncKlingImageGenDriver(ImageCostMixin, AsyncImageGenDriver):
    """Async image generation via Kling AI."""

    supports_multiple = KlingImageGenDriver.supports_multiple
    supports_size_variants = KlingImageGenDriver.supports_size_variants
    supported_sizes = KlingImageGenDriver.supported_sizes
    max_images = KlingImageGenDriver.max_images

    KNOWN_MODELS = KlingImageGenDriver.KNOWN_MODELS
    IMAGE_PRICING = KlingImageGenDriver.IMAGE_PRICING

    def __init__(
        self,
        access_key: str | None = None,
        secret_key: str | None = None,
        model: str = "kling-v2-1",
        endpoint: str | None = None,
    ):
        import os

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

    async def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not prompt:
            raise ValueError("prompt cannot be empty")
        token = self._token()
        # Reuse the sync build helper — it's pure.
        sync = KlingImageGenDriver(
            access_key=self.access_key,
            secret_key=self.secret_key,
            model=self.model,
            endpoint=self.endpoint,
        )
        path, body = sync._build_body(prompt, options)
        is_multi = path.endswith("multi-image2image")

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self.endpoint}{path}", headers=self._headers(token), json=body)
            if resp.status_code >= 400:
                raise RuntimeError(f"Kling API error {resp.status_code}: {resp.text}")
            result = resp.json()
            if result.get("code") != 0:
                raise RuntimeError(f"Kling API error: {result.get('message')}")
            task_id = result.get("data", {}).get("task_id")
            if not task_id:
                raise RuntimeError(f"Kling response missing task_id: {result}")

            if not options.get("poll", True):
                return {
                    "images": [],
                    "meta": {
                        "image_count": 0,
                        "size": body.get("aspect_ratio"),
                        "revised_prompt": None,
                        "cost": 0.0,
                        "model_name": f"kling/{body['model_name']}",
                        "task_id": task_id,
                        "status": "pending",
                        "raw_response": result,
                    },
                }

            final = await self._poll_image(
                client,
                task_id,
                token,
                multi=is_multi,
                poll_interval=float(options.get("poll_interval", 3)),
                timeout_seconds=float(options.get("timeout", 120)),
            )

        urls = [
            img["url"] for img in (final.get("data", {}).get("task_result", {}).get("images") or []) if img.get("url")
        ]
        images = [image_from_url(u) for u in urls]
        cost = self._calculate_image_cost(
            "kling", body["model_name"], size=str(body["image_size"]), n=max(len(images), 1)
        )
        return {
            "images": images,
            "meta": {
                "image_count": len(images),
                "size": body["aspect_ratio"],
                "revised_prompt": None,
                "cost": cost,
                "model_name": f"kling/{body['model_name']}",
                "task_id": task_id,
                "raw_response": final,
            },
        }

    async def _poll_image(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        token: str,
        *,
        multi: bool,
        poll_interval: float,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        base = "multi-image2image" if multi else "generations"
        path = f"/v1/images/{base}/{task_id}"
        import time as _t

        deadline = _t.monotonic() + timeout_seconds
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
                raise RuntimeError(f"Kling image task failed: {msg}")
            if _t.monotonic() >= deadline:
                raise TimeoutError(f"Kling image task {task_id} timed out (status={status})")
            await asyncio.sleep(poll_interval)
