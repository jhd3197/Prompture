"""Google Imagen image generation driver. Requires the ``google-genai`` package."""

from __future__ import annotations

import logging
import os
from typing import Any

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_bytes
from .img_gen_base import ImageGenDriver

logger = logging.getLogger(__name__)


class GoogleImageGenDriver(ImageCostMixin, ImageGenDriver):
    """Image generation via Google Imagen API."""

    supports_multiple = True
    supports_size_variants = False
    supported_sizes = ["1024x1024"]
    max_images = 4

    IMAGE_PRICING = {
        "imagen-3.0-generate-002": {"default": 0.04},
        "imagen-3.0-fast-generate-001": {"default": 0.02},
    }

    def __init__(self, api_key: str | None = None, model: str = "imagen-3.0-generate-002"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except ImportError as exc:
                raise RuntimeError("google-genai package is not installed") from exc
        return self._client

    def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate image(s) using Google Imagen API.

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
            resp = client.models.generate_images(
                model=model,
                prompt=prompt,
                config=config,
            )
        except Exception as exc:
            # Handle safety-blocked prompts
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
