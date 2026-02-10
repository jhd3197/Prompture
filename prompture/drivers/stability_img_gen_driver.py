"""Stability AI image generation driver. Uses ``httpx`` for REST API calls."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_bytes
from .img_gen_base import ImageGenDriver

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.stability.ai"


class StabilityImageGenDriver(ImageCostMixin, ImageGenDriver):
    """Image generation via Stability AI REST API."""

    supports_multiple = False
    supports_size_variants = True
    supported_sizes = ["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"]
    max_images = 1

    IMAGE_PRICING = {
        "stable-image-core": {"default": 0.03},
        "sd3.5-large": {"default": 0.065},
        "sd3.5-large-turbo": {"default": 0.04},
        "sd3.5-medium": {"default": 0.035},
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "stable-image-core",
        endpoint: str | None = None,
    ):
        self.api_key = api_key or os.getenv("STABILITY_API_KEY")
        self.model = model
        self.endpoint = (endpoint or os.getenv("STABILITY_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

    def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate an image using Stability AI REST API.

        Args:
            prompt: Text description of the desired image.
            options: Supports ``aspect_ratio`` (e.g. "1:1", "16:9"),
                     ``output_format`` ("png"/"jpeg"/"webp"), ``model``,
                     ``negative_prompt``, ``seed``.
        """
        if not self.api_key:
            raise RuntimeError("STABILITY_API_KEY is not configured")

        model = options.get("model", self.model)
        aspect_ratio = options.get("aspect_ratio", "1:1")
        output_format = options.get("output_format", "png")

        url = f"{self.endpoint}/v2beta/stable-image/generate/core"

        data: dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
        }

        negative_prompt = options.get("negative_prompt")
        if negative_prompt:
            data["negative_prompt"] = negative_prompt

        seed = options.get("seed")
        if seed is not None:
            data["seed"] = str(seed)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "image/*",
        }

        resp = httpx.post(url, data=data, headers=headers, timeout=120.0)

        if resp.status_code != 200:
            raise RuntimeError(f"Stability API error {resp.status_code}: {resp.text}")

        media_type = f"image/{output_format}" if output_format != "jpeg" else "image/jpeg"
        image = image_from_bytes(resp.content, media_type=media_type)

        cost = self._calculate_image_cost("stability", model, n=1)

        return {
            "images": [image],
            "meta": {
                "image_count": 1,
                "size": aspect_ratio,
                "revised_prompt": None,
                "cost": round(cost, 6),
                "model_name": f"stability/{model}",
                "raw_response": {},
            },
        }
