"""Grok/xAI Aurora image generation driver. Uses OpenAI SDK with xAI base URL."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore[misc, assignment]

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_base64
from .img_gen_base import ImageGenDriver

logger = logging.getLogger(__name__)

_XAI_BASE_URL = "https://api.x.ai/v1"


class GrokImageGenDriver(ImageCostMixin, ImageGenDriver):
    """Image generation via Grok/xAI Aurora API (OpenAI-compatible)."""

    supports_multiple = True
    supports_size_variants = False
    supported_sizes = ["1024x1024"]
    max_images = 10

    IMAGE_PRICING = {
        "grok-2-image": {"default": 0.07},
    }

    def __init__(self, api_key: str | None = None, model: str = "grok-2-image"):
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        self.model = model
        if OpenAI is not None:
            self.client = OpenAI(api_key=self.api_key, base_url=_XAI_BASE_URL)
        else:
            self.client = None

    def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate image(s) using Grok/xAI Aurora API.

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

        resp = self.client.images.generate(**kwargs)

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
