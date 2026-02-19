"""Async OpenAI DALL-E image generation driver. Requires the ``openai`` package (>=1.0.0)."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None  # type: ignore[misc, assignment]

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_base64
from .async_img_gen_base import AsyncImageGenDriver
from .openai_img_gen_driver import OpenAIImageGenDriver

logger = logging.getLogger(__name__)


class AsyncOpenAIImageGenDriver(ImageCostMixin, AsyncImageGenDriver):
    """Async image generation via OpenAI DALL-E 2/3 API."""

    supports_multiple = OpenAIImageGenDriver.supports_multiple
    supports_size_variants = OpenAIImageGenDriver.supports_size_variants
    supported_sizes = OpenAIImageGenDriver.supported_sizes
    max_images = OpenAIImageGenDriver.max_images

    IMAGE_PRICING = OpenAIImageGenDriver.IMAGE_PRICING

    def __init__(self, api_key: str | None = None, model: str = "dall-e-3"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if AsyncOpenAI is not None:
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None

    async def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate image(s) using OpenAI DALL-E API (async).

        Args:
            prompt: Text description of the desired image.
            options: Supports ``size``, ``quality``, ``style``, ``n``, ``model``.
        """
        if self.client is None:
            raise RuntimeError("openai package (>=1.0.0) is not installed")

        model = options.get("model", self.model)
        size = options.get("size", "1024x1024")
        quality = options.get("quality", "standard")
        n = options.get("n", 1)
        style = options.get("style", "vivid")

        images = []
        revised_prompt = None

        is_dalle3 = "dall-e-3" in model
        batch_size = 1 if is_dalle3 else n

        remaining = n
        while remaining > 0:
            batch_n = min(batch_size, remaining) if is_dalle3 else remaining

            kwargs: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "n": batch_n,
                "size": size,
                "response_format": "b64_json",
            }
            if is_dalle3:
                kwargs["quality"] = quality
                kwargs["style"] = style

            resp = await self.client.images.generate(**kwargs)

            for img_data in resp.data:
                b64 = img_data.b64_json
                images.append(image_from_base64(b64, media_type="image/png"))
                if img_data.revised_prompt and revised_prompt is None:
                    revised_prompt = img_data.revised_prompt

            remaining -= batch_n

        cost = self._calculate_image_cost("openai", model, size=size, quality=quality, n=n)

        return {
            "images": images,
            "meta": {
                "image_count": len(images),
                "size": size,
                "revised_prompt": revised_prompt,
                "cost": round(cost, 6),
                "model_name": f"openai/{model}",
                "raw_response": {},
            },
        }
