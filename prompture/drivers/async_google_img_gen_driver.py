"""Async Google Imagen image generation driver. Requires the ``google-genai`` package."""

from __future__ import annotations

import logging
import os
from typing import Any

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_bytes
from .async_img_gen_base import AsyncImageGenDriver
from .google_img_gen_driver import GoogleImageGenDriver

logger = logging.getLogger(__name__)


class AsyncGoogleImageGenDriver(ImageCostMixin, AsyncImageGenDriver):
    """Async image generation via Google Imagen API."""

    supports_multiple = GoogleImageGenDriver.supports_multiple
    supports_size_variants = GoogleImageGenDriver.supports_size_variants
    supported_sizes = GoogleImageGenDriver.supported_sizes
    max_images = GoogleImageGenDriver.max_images

    IMAGE_PRICING = GoogleImageGenDriver.IMAGE_PRICING

    def __init__(self, api_key: str | None = None, model: str = "imagen-3.0-generate-002"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except ImportError as exc:
                raise RuntimeError("google-genai package is not installed") from exc
        return self._client

    async def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate image(s) using Google Imagen API (async).

        Args:
            prompt: Text description of the desired image.
            options: Supports ``n`` (1-4), ``model``.
        """
        client = self._get_client()
        from google.genai import types

        model = options.get("model", self.model)
        n = min(options.get("n", 1), self.max_images)

        config = types.GenerateImagesConfig(number_of_images=n)

        try:
            resp = await client.aio.models.generate_images(
                model=model,
                prompt=prompt,
                config=config,
            )
        except Exception as exc:
            exc_name = type(exc).__name__
            if "Blocked" in exc_name or "blocked" in str(exc).lower():
                raise ValueError(f"Prompt blocked by safety filter: {exc}") from exc
            raise

        images = []
        if resp.generated_images:
            for img in resp.generated_images:
                images.append(image_from_bytes(img.image.image_bytes, media_type="image/png"))

        cost = self._calculate_image_cost("google", model, n=len(images))

        return {
            "images": images,
            "meta": {
                "image_count": len(images),
                "size": "1024x1024",
                "revised_prompt": None,
                "cost": round(cost, 6),
                "model_name": f"google/{model}",
                "raw_response": {},
            },
        }
