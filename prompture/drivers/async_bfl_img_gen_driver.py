"""Async Black Forest Labs (BFL) image generation driver."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_url
from .async_img_gen_base import AsyncImageGenDriver
from .bfl_img_gen_driver import (
    _DEFAULT_ENDPOINT,
    _DEFAULT_MODEL,
    _TERMINAL_ERR,
    _TERMINAL_OK,
    BFLImageGenDriver,
    _build_body,
)

logger = logging.getLogger(__name__)


class AsyncBFLImageGenDriver(ImageCostMixin, AsyncImageGenDriver):
    """Async image generation via the Black Forest Labs (BFL) API."""

    supports_multiple = BFLImageGenDriver.supports_multiple
    supports_size_variants = BFLImageGenDriver.supports_size_variants
    supported_sizes = BFLImageGenDriver.supported_sizes
    max_images = BFLImageGenDriver.max_images

    KNOWN_MODELS = BFLImageGenDriver.KNOWN_MODELS
    IMAGE_PRICING = BFLImageGenDriver.IMAGE_PRICING

    POLL_INTERVAL_SECONDS: float = BFLImageGenDriver.POLL_INTERVAL_SECONDS
    POLL_MAX_RETRIES: int = BFLImageGenDriver.POLL_MAX_RETRIES

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        endpoint: str | None = None,
    ):
        self.api_key = api_key or os.getenv("BFL_API_KEY")
        self.model = model
        self.endpoint = (endpoint or os.getenv("BFL_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

    @classmethod
    def list_models(cls, **kw: object) -> list[str] | None:
        return list(cls.KNOWN_MODELS)

    def _headers(self) -> dict[str, str]:
        return {"x-key": self.api_key or "", "Content-Type": "application/json"}

    async def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("BFL_API_KEY is not configured")
        if not prompt:
            raise ValueError("prompt cannot be empty")

        model = options.get("model", self.model)
        body = _build_body(prompt, options, model)
        submit_url = f"{self.endpoint}/v1/{model}"

        poll_interval = float(options.get("poll_interval", self.POLL_INTERVAL_SECONDS))
        max_retries = int(options.get("max_retries", self.POLL_MAX_RETRIES))

        async with httpx.AsyncClient(timeout=120.0) as client:
            submit = await client.post(submit_url, headers=self._headers(), json=body)
            if submit.status_code >= 400:
                raise RuntimeError(f"BFL submit failed {submit.status_code}: {submit.text}")
            submitted = submit.json()
            request_id = submitted.get("id")
            polling_url = submitted.get("polling_url")
            if not request_id or not polling_url:
                raise RuntimeError(f"BFL response missing id/polling_url: {submitted}")

            if not options.get("poll", True):
                return {
                    "images": [],
                    "meta": {
                        "image_count": 0,
                        "size": f"{body.get('width', '?')}x{body.get('height', '?')}",
                        "revised_prompt": None,
                        "cost": 0.0,
                        "model_name": f"bfl/{model}",
                        "request_id": request_id,
                        "polling_url": polling_url,
                        "status": "pending",
                        "raw_response": submitted,
                    },
                }

            final = await self._poll(client, polling_url, max_retries=max_retries, poll_interval=poll_interval)

        result_payload = final.get("result") or {}
        sample = result_payload.get("sample")
        images = [image_from_url(sample)] if isinstance(sample, str) and sample else []
        cost = self._calculate_image_cost("bfl", model, n=max(len(images), 1))

        return {
            "images": images,
            "meta": {
                "image_count": len(images),
                "size": f"{body.get('width', '?')}x{body.get('height', '?')}",
                "revised_prompt": None,
                "cost": cost,
                "model_name": f"bfl/{model}",
                "request_id": request_id,
                "polling_url": polling_url,
                "raw_response": final,
            },
        }

    async def _poll(
        self,
        client: httpx.AsyncClient,
        polling_url: str,
        *,
        max_retries: int,
        poll_interval: float,
    ) -> dict[str, Any]:
        status: str | None = None
        for _attempt in range(max_retries):
            r = await client.get(polling_url, headers=self._headers())
            if r.status_code >= 400:
                raise RuntimeError(f"BFL poll failed {r.status_code}: {r.text}")
            data = r.json()
            status = data.get("status")
            if status in _TERMINAL_OK:
                return data
            if status in _TERMINAL_ERR:
                raise RuntimeError(f"BFL job {status}: {data}")
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"BFL job timed out after {max_retries} retries (last status={status!r})")
