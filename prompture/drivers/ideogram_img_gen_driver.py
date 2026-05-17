"""Ideogram image generation driver.

Uses the Ideogram v3 ``/v1/ideogram-v3/generate`` endpoint with multipart form
data. Authentication is via the ``Api-Key`` header.

Reference: https://developer.ideogram.ai/api-reference/api-reference/generate-v3
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_url
from .img_gen_base import ImageGenDriver

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.ideogram.ai"

_V3_MODELS = {"ideogram-v3"}
_LEGACY_MODELS = {"ideogram-v2", "ideogram-v2-turbo", "ideogram-v1"}

_ALLOWED_RATIOS = {
    "1x1",
    "16x9",
    "9x16",
    "4x3",
    "3x4",
    "3x2",
    "2x3",
    "16x10",
    "10x16",
    "5x4",
    "4x5",
    "1x3",
    "3x1",
}
_ALLOWED_SPEEDS = {"TURBO", "DEFAULT", "QUALITY"}


def _normalize_aspect_ratio(ratio: str) -> str:
    """Accept ``"16:9"`` and ``"16x9"`` forms; return Ideogram's ``"16x9"`` form."""
    if isinstance(ratio, str) and ":" in ratio:
        ratio = ratio.replace(":", "x")
    return ratio


class IdeogramImageGenDriver(ImageCostMixin, ImageGenDriver):
    """Image generation via the Ideogram REST API (v3)."""

    supports_multiple = True
    supports_size_variants = True
    supported_sizes = sorted(_ALLOWED_RATIOS)
    max_images = 8

    KNOWN_MODELS = sorted(_V3_MODELS | _LEGACY_MODELS)

    # Rough public per-image pricing in USD (updated 2026). Treat as
    # best-effort fallbacks; canonical pricing lives in the local KB JSONs.
    IMAGE_PRICING: dict[str, dict[str, float]] = {
        "ideogram-v3": {
            "TURBO": 0.03,
            "DEFAULT": 0.06,
            "QUALITY": 0.09,
            "default": 0.06,
        },
        "ideogram-v2": {"default": 0.08},
        "ideogram-v2-turbo": {"default": 0.05},
        "ideogram-v1": {"default": 0.08},
    }

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

    def _build_form(self, prompt: str, options: dict[str, Any], model: str) -> tuple[dict[str, str], str]:
        """Build the multipart form-data dict for v3 + return the URL path."""
        aspect_ratio = _normalize_aspect_ratio(options.get("aspect_ratio") or options.get("ratio") or "1x1")
        if aspect_ratio not in _ALLOWED_RATIOS:
            raise ValueError(f"Unsupported aspect_ratio '{aspect_ratio}'. Allowed: {sorted(_ALLOWED_RATIOS)}")

        rendering_speed = str(options.get("rendering_speed", "DEFAULT")).upper()
        if rendering_speed not in _ALLOWED_SPEEDS:
            raise ValueError(f"Unsupported rendering_speed '{rendering_speed}'. Allowed: {sorted(_ALLOWED_SPEEDS)}")

        n = int(options.get("num_images") or options.get("n") or 1)
        n = max(1, min(n, self.max_images))

        form: dict[str, str] = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "rendering_speed": rendering_speed,
            "num_images": str(n),
        }

        # Optional pass-throughs.
        for src_key, dst_key in (
            ("seed", "seed"),
            ("negative_prompt", "negative_prompt"),
            ("style_type", "style_type"),
            ("magic_prompt", "magic_prompt"),
        ):
            val = options.get(src_key)
            if val is not None:
                form[dst_key] = str(val)

        if model in _V3_MODELS:
            path = "/v1/ideogram-v3/generate"
        elif model in _LEGACY_MODELS:
            # Legacy v1/v2 endpoint uses a JSON body — surface a clear error
            # so callers can route via a different driver/version.
            raise NotImplementedError(
                f"Ideogram model '{model}' uses the legacy /generate JSON endpoint "
                "which is not yet supported. Use 'ideogram-v3' or open an issue."
            )
        else:
            # Default to v3 for any future/unknown id.
            path = "/v1/ideogram-v3/generate"

        return form, path

    def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("IDEOGRAM_API_KEY is not configured")
        if not prompt:
            raise ValueError("prompt cannot be empty")

        model = options.get("model", self.model)
        form, path = self._build_form(prompt, options, model)

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
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
