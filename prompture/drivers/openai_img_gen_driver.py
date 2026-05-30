"""OpenAI DALL-E image generation driver. Requires the ``openai`` package (>=1.0.0)."""

from __future__ import annotations

import io
import logging
import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_base64, image_from_bytes
from .img_gen_base import ImageGenDriver

logger = logging.getLogger(__name__)


class OpenAIImageGenDriver(ImageCostMixin, ImageGenDriver):
    """Image generation via OpenAI DALL-E 2/3 API."""

    supports_multiple = True
    supports_size_variants = True
    supports_edit = True
    supports_variation = True
    supported_sizes = [
        "256x256",
        "512x512",
        "1024x1024",
        "1536x1024",
        "1024x1536",
        "1792x1024",
        "1024x1792",
    ]
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
        # gpt-image-1 (and gpt-image-2 when OpenAI ships it) — token-priced in reality;
        # values below are per-image USD approximations published by OpenAI.
        "gpt-image-1": {
            "1024x1024/low": 0.011,
            "1024x1024/medium": 0.042,
            "1024x1024/high": 0.167,
            "1536x1024/low": 0.016,
            "1536x1024/medium": 0.063,
            "1536x1024/high": 0.25,
            "1024x1536/low": 0.016,
            "1024x1536/medium": 0.063,
            "1024x1536/high": 0.25,
            "default": 0.042,
        },
        "gpt-image-2": {
            "1024x1024/low": 0.011,
            "1024x1024/medium": 0.042,
            "1024x1024/high": 0.167,
            "default": 0.042,
        },
    }

    def __init__(self, api_key: str | None = None, model: str = "dall-e-3"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if OpenAI is not None:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    @staticmethod
    def _download_image(url: str) -> bytes:
        """Fetch image bytes from a result URL.

        The images API returns a ``url`` (not base64) unless
        ``response_format='b64_json'`` is requested — and current models
        reject that parameter outright. So URL results are downloaded here to
        keep a single ``{images, meta}`` contract for callers.
        """
        import httpx

        resp = httpx.get(url, timeout=60.0)
        resp.raise_for_status()
        return resp.content

    def _collect_images(self, data: Any) -> tuple[list, str | None]:
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
                images.append(image_from_bytes(self._download_image(url), media_type="image/png"))
            revised = getattr(item, "revised_prompt", None)
            if revised and revised_prompt is None:
                revised_prompt = revised
        return images, revised_prompt

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
        is_gpt_image = model.startswith("gpt-image")
        is_dalle3 = "dall-e-3" in model

        size = options.get("size", "1024x1024")
        # gpt-image-* uses low/medium/high/auto; DALL-E 3 uses standard/hd; DALL-E 2 ignores quality.
        default_quality = "high" if is_gpt_image else "standard"
        quality = options.get("quality", default_quality)
        n = options.get("n", 1)
        style = options.get("style", "vivid")

        images = []
        revised_prompt = None

        # DALL-E 3 only supports n=1, so we loop for multiple images
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
                # gpt-image-* always return b64; response_format is rejected.
                kwargs["quality"] = quality
            elif is_dalle3:
                # DALL·E 3 takes quality/style. It no longer accepts
                # response_format (rejected like gpt-image-*), so it's omitted
                # and any returned URL is downloaded in _collect_images.
                kwargs["quality"] = quality
                kwargs["style"] = style
            # dall-e-2: no extra kwargs; response_format intentionally omitted.

            resp = self.client.images.generate(**kwargs)

            batch_images, batch_revised = self._collect_images(resp.data)
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

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    _VARIATION_SIZES = {"256x256", "512x512", "1024x1024"}

    def _safe_cost(self, model: str, size: str, n: int, quality: str = "standard") -> float:
        try:
            return round(self._calculate_image_cost("openai", model, size=size, quality=quality, n=n), 6)
        except Exception:  # pragma: no cover - pricing table gaps shouldn't break a call
            return 0.0

    def _result(self, data: Any, model: str, size: str, n: int) -> dict[str, Any]:
        images, _ = self._collect_images(data)
        return {
            "images": images,
            "meta": {
                "image_count": len(images),
                "size": size,
                "revised_prompt": None,
                "cost": self._safe_cost(model, size, n),
                "model_name": f"openai/{model}",
                "raw_response": {},
            },
        }

    def edit_image(
        self,
        image: bytes,
        prompt: str,
        options: dict[str, Any],
        *,
        mask: bytes | None = None,
    ) -> dict[str, Any]:
        """Edit an image with an instruction. Uses gpt-image-1 by default."""
        if self.client is None:
            raise RuntimeError("openai package (>=1.0.0) is not installed")

        model = options.get("model", "gpt-image-1")
        size = options.get("size", "1024x1024")
        n = options.get("n", 1)
        kwargs: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "image": ("image.png", io.BytesIO(image), "image/png"),
            "n": n,
            "size": size,
        }
        if mask is not None:
            kwargs["mask"] = ("mask.png", io.BytesIO(mask), "image/png")

        resp = self.client.images.edit(**kwargs)
        return self._result(resp.data, model, size, n)

    def create_variation(self, image: bytes, options: dict[str, Any]) -> dict[str, Any]:
        """Produce variation(s) of an image. Only DALL·E 2 supports this."""
        if self.client is None:
            raise RuntimeError("openai package (>=1.0.0) is not installed")

        model = "dall-e-2"  # the only OpenAI model with a variations endpoint
        size = options.get("size", "1024x1024")
        if size not in self._VARIATION_SIZES:
            size = "1024x1024"
        n = options.get("n", 1)

        resp = self.client.images.create_variation(
            model=model,
            image=("image.png", io.BytesIO(image), "image/png"),
            n=n,
            size=size,
        )
        return self._result(resp.data, model, size, n)
