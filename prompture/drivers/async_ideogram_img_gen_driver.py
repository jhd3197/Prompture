"""Async Ideogram image generation driver."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_url
from .async_img_gen_base import AsyncImageGenDriver
from .ideogram_img_gen_driver import (
    _DEFAULT_ENDPOINT,
    IdeogramImageGenDriver,
)

logger = logging.getLogger(__name__)


class AsyncIdeogramImageGenDriver(ImageCostMixin, AsyncImageGenDriver):
    """Async image generation via the Ideogram REST API (v3)."""

    supports_multiple = IdeogramImageGenDriver.supports_multiple
    supports_size_variants = IdeogramImageGenDriver.supports_size_variants
    supported_sizes = IdeogramImageGenDriver.supported_sizes
    max_images = IdeogramImageGenDriver.max_images

    KNOWN_MODELS = IdeogramImageGenDriver.KNOWN_MODELS
    IMAGE_PRICING = IdeogramImageGenDriver.IMAGE_PRICING

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "ideogram-v3",
        endpoint: str | None = None,
    ):
        self.api_key = api_key or os.getenv("IDEOGRAM_API_KEY")
        self.model = model
        self.endpoint = (endpoint or os.getenv("IDEOGRAM_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

    @classmethod
    def list_models(cls, **kw: object) -> list[str] | None:
        return list(cls.KNOWN_MODELS)

    def _headers(self) -> dict[str, str]:
        return {"Api-Key": self.api_key or ""}

    async def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("IDEOGRAM_API_KEY is not configured")
        if not prompt:
            raise ValueError("prompt cannot be empty")

        model = options.get("model", self.model)
        # Reuse the sync build helper — it's pure.
        sync = IdeogramImageGenDriver(api_key=self.api_key, model=self.model, endpoint=self.endpoint)
        form, path = sync._build_form(prompt, options, model)

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.endpoint}{path}",
                headers=self._headers(),
                data=form,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"Ideogram API error {resp.status_code}: {resp.text}")
            result = resp.json()

        data = result.get("data") or []
        urls = [item.get("url") for item in data if isinstance(item, dict) and item.get("url")]
        images = [image_from_url(u) for u in urls]
        revised_prompt = None
        if data and isinstance(data[0], dict):
            revised_prompt = data[0].get("prompt")

        cost = self._calculate_image_cost("ideogram", model, size=form["rendering_speed"], n=max(len(images), 1))

        return {
            "images": images,
            "meta": {
                "image_count": len(images),
                "size": form["aspect_ratio"],
                "revised_prompt": revised_prompt,
                "cost": cost,
                "model_name": f"ideogram/{model}",
                "raw_response": result,
            },
        }
