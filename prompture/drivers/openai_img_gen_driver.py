"""OpenAI DALL-E image generation driver. Requires the ``openai`` package (>=1.0.0)."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_base64
from .img_gen_base import ImageGenDriver

logger = logging.getLogger(__name__)


class OpenAIImageGenDriver(ImageCostMixin, ImageGenDriver):
    """Image generation via OpenAI DALL-E 2/3 API."""

    supports_multiple = True
    supports_size_variants = True
    supported_sizes = ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
    max_images = 10

    IMAGE_PRICING = {
        "dall-e-3": {
            "1024x1024/standard": 0.04,
            "1024x1024/hd": 0.08,
            "1792x1024/standard": 0.08,
            "1792x1024/hd": 0.12,
            "1024x1792/standard": 0.08,
            "1024x1792/hd": 0.12,
        },
        "dall-e-2": {
            "256x256": 0.016,
            "512x512": 0.018,
            "1024x1024": 0.020,
        },
    }

    def __init__(self, api_key: str | None = None, model: str = "dall-e-3"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if OpenAI:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate image(s) using OpenAI DALL-E API.

        Args:
            prompt: Text description of the desired image.
            options: Supports ``size`` (e.g. "1024x1024"), ``quality`` ("standard"/"hd"),
                     ``style`` ("vivid"/"natural"), ``n`` (number of images), ``model``.
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

        # DALL-E 3 only supports n=1, so we loop for multiple images
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

            resp = self.client.images.generate(**kwargs)

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
