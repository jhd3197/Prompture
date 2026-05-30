"""Async OpenAI DALL-E image generation driver. Requires the ``openai`` package (>=1.0.0)."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[misc, assignment]

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_base64, image_from_bytes
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

    @staticmethod
    async def _download_image(url: str) -> bytes:
        """Fetch image bytes from a result URL (returned when response_format
        isn't requested — which current models reject anyway)."""
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    async def _collect_images(self, data: Any) -> tuple[list, str | None]:
        """Normalize an images response, whether it carries ``b64_json`` or a ``url``."""
        images: list = []
        revised_prompt: str | None = None
        for item in data:
            b64 = getattr(item, "b64_json", None)
            if b64:
                images.append(image_from_base64(b64, media_type="image/png"))
            else:
                url = getattr(item, "url", None)
                if not url:
                    raise RuntimeError(
                        "OpenAI image response contained neither b64_json nor url"
                    )
                images.append(image_from_bytes(await self._download_image(url), media_type="image/png"))
            revised = getattr(item, "revised_prompt", None)
            if revised and revised_prompt is None:
                revised_prompt = revised
        return images, revised_prompt

    async def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate image(s) using OpenAI DALL-E API (async).

        Args:
            prompt: Text description of the desired image.
            options: Supports ``size``, ``quality``, ``style``, ``n``, ``model``.
        """
        if self.client is None:
            raise RuntimeError("openai package (>=1.0.0) is not installed")

        model = options.get("model", self.model)
        is_gpt_image = model.startswith("gpt-image")
        is_dalle3 = "dall-e-3" in model

        size = options.get("size", "1024x1024")
        default_quality = "high" if is_gpt_image else "standard"
        quality = options.get("quality", default_quality)
        n = options.get("n", 1)
        style = options.get("style", "vivid")

        images = []
        revised_prompt = None

        batch_size = 1 if is_dalle3 else n

        remaining = n
        while remaining > 0:
            batch_n = min(batch_size, remaining) if is_dalle3 else remaining

            kwargs: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "n": batch_n,
                "size": size,
            }
            if is_gpt_image:
                kwargs["quality"] = quality
            elif is_dalle3:
                # DALL·E 3 no longer accepts response_format (rejected like
                # gpt-image-*); omit it and download any returned URL.
                kwargs["quality"] = quality
                kwargs["style"] = style
            # dall-e-2: no extra kwargs; response_format intentionally omitted.

            resp = await self.client.images.generate(**kwargs)

            batch_images, batch_revised = await self._collect_images(resp.data)
            images.extend(batch_images)
            if batch_revised and revised_prompt is None:
                revised_prompt = batch_revised

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
