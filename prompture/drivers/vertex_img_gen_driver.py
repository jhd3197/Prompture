"""Google Vertex AI image generation via Gemini image models.

Unlike the Imagen driver (``generate_images``), the ``gemini-*-flash-image``
models emit images through ``generate_content`` with ``response_modalities``
including ``IMAGE`` — so this is a distinct driver. Uses the ``google-genai``
SDK with ``vertexai=True``.

Auth (same as the Vertex LLM driver): a Vertex Express API key
(``GOOGLE_VERTEX_API_KEY``) for the express path, or project + ADC
(``GOOGLE_VERTEX_PROJECT_ID`` / ``GOOGLE_VERTEX_LOCATION``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_bytes
from .img_gen_base import ImageGenDriver

logger = logging.getLogger(__name__)


class VertexImageGenDriver(ImageCostMixin, ImageGenDriver):
    """Image generation via Vertex AI Gemini image models (generate_content)."""

    supports_multiple = False  # generate_content returns a single image
    supports_size_variants = True
    # Aspect ratios (Gemini image gen is aspect-ratio based, not pixel sizes).
    supported_sizes = ["1:1", "16:9", "9:16", "4:3", "3:4"]
    max_images = 1

    # Per-image USD approximations; refine as Google publishes rates.
    IMAGE_PRICING = {
        "gemini-2.5-flash-image": {"default": 0.04},
        "gemini-3.1-flash-image": {"default": 0.04},
    }

    KNOWN_MODELS = ["gemini-2.5-flash-image", "gemini-3.1-flash-image"]

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
        model: str = "gemini-2.5-flash-image",
    ):
        self.api_key = api_key or os.getenv("GOOGLE_VERTEX_API_KEY")
        self.project_id = project_id or os.getenv("GOOGLE_VERTEX_PROJECT_ID")
        self.location = location or os.getenv("GOOGLE_VERTEX_LOCATION", "global")
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError("google-genai package is not installed") from exc

            kwargs: dict[str, Any] = {"vertexai": True}
            if self.api_key:
                # Express mode: api key is mutually exclusive with project/location.
                kwargs["api_key"] = self.api_key
            elif self.project_id:
                kwargs["project"] = self.project_id
                kwargs["location"] = self.location
            else:
                raise RuntimeError(
                    "Vertex image generation requires GOOGLE_VERTEX_API_KEY "
                    "or GOOGLE_VERTEX_PROJECT_ID"
                )
            self._client = genai.Client(**kwargs)
        return self._client

    def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate an image via Vertex Gemini ``generate_content``.

        Options: ``model``, ``aspect_ratio`` (or ``size``), ``image_size``
        (e.g. "1K"/"2K"). Gemini image gen produces one image per call.
        """
        client = self._get_client()
        from google.genai import types

        model = options.get("model", self.model)
        aspect = options.get("aspect_ratio") or options.get("size") or "1:1"
        # If a pixel size slipped through (e.g. "1024x1024"), don't send it as an
        # aspect ratio — fall back to square.
        if "x" in str(aspect) and ":" not in str(aspect):
            aspect = "1:1"
        image_size = options.get("image_size", "1K")

        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect, image_size=image_size),
        )

        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
        except Exception as exc:
            name = type(exc).__name__
            if "Blocked" in name or "blocked" in str(exc).lower() or "safety" in str(exc).lower():
                raise ValueError(f"Prompt blocked by safety filter: {exc}") from exc
            raise

        images = []
        for cand in (getattr(resp, "candidates", None) or []):
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", None) or []):
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None) if inline is not None else None
                if data:
                    mime = getattr(inline, "mime_type", None) or "image/png"
                    images.append(image_from_bytes(data, media_type=mime))

        if not images:
            raise ValueError(
                "Vertex returned no image — the model may have refused or replied with text only."
            )

        cost = self._calculate_image_cost("vertex_ai", model, n=len(images))

        return {
            "images": images,
            "meta": {
                "image_count": len(images),
                "size": aspect,
                "revised_prompt": None,
                "cost": round(cost, 6),
                "model_name": f"vertex_ai/{model}",
                "raw_response": {},
            },
        }
