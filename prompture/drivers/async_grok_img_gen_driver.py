"""Async Grok/xAI Aurora image generation driver. Uses OpenAI SDK with xAI base URL."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_base64
from .async_img_gen_base import AsyncImageGenDriver
from .grok_img_gen_driver import _XAI_BASE_URL, GrokImageGenDriver

logger = logging.getLogger(__name__)


class AsyncGrokImageGenDriver(ImageCostMixin, AsyncImageGenDriver):
    """Async image generation via Grok/xAI Aurora API (OpenAI-compatible)."""

    supports_multiple = GrokImageGenDriver.supports_multiple
    supports_size_variants = GrokImageGenDriver.supports_size_variants
    supported_sizes = GrokImageGenDriver.supported_sizes
    max_images = GrokImageGenDriver.max_images

    IMAGE_PRICING = GrokImageGenDriver.IMAGE_PRICING

    def __init__(self, api_key: str | None = None, model: str = "grok-2-image"):
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        self.model = model
        if AsyncOpenAI:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=_XAI_BASE_URL)
        else:
            self.client = None

    async def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate image(s) using Grok/xAI Aurora API (async).

        Args:
            prompt: Text description of the desired image.
            options: Supports ``n`` (number of images), ``model``.
        """
        if self.client is None:
            raise RuntimeError("openai package (>=1.0.0) is not installed")

        model = options.get("model", self.model)
        n = options.get("n", 1)

        kwargs: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "response_format": "b64_json",
        }

        resp = await self.client.images.generate(**kwargs)

        images = []
        revised_prompt = None
        for img_data in resp.data:
            b64 = img_data.b64_json
            images.append(image_from_base64(b64, media_type="image/png"))
            if img_data.revised_prompt and revised_prompt is None:
                revised_prompt = img_data.revised_prompt

        cost = self._calculate_image_cost("grok", model, n=len(images))

        return {
            "images": images,
            "meta": {
                "image_count": len(images),
                "size": "1024x1024",
                "revised_prompt": revised_prompt,
                "cost": round(cost, 6),
                "model_name": f"grok/{model}",
                "raw_response": {},
            },
        }
