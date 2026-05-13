"""Black Forest Labs (BFL) image generation driver.

Provides direct access to the Flux model family via the BFL API at
``https://api.bfl.ai``. BFL uses an async job pattern:

1. ``POST /v1/<model>`` with an ``x-key`` auth header — returns
   ``{"id": "...", "polling_url": "..."}``.
2. Poll the ``polling_url`` until the response contains
   ``{"status": "Ready", "result": {"sample": "<image_url>"}}``.

Supported models include ``flux-pro-1.1``, ``flux-pro-1.1-ultra``,
``flux-pro``, ``flux-dev``, ``flux-schnell``, ``flux-kontext-pro``
(image editing), and ``flux-kontext-max``.

Reference: https://docs.bfl.ai/
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_url
from .img_gen_base import ImageGenDriver

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.bfl.ai"
_DEFAULT_MODEL = "flux-pro-1.1"

_BFL_MODELS = {
    "flux-pro-1.1",
    "flux-pro-1.1-ultra",
    "flux-pro",
    "flux-dev",
    "flux-schnell",
    "flux-kontext-pro",
    "flux-kontext-max",
}

# Models that accept an ``input_image`` (base64) for editing.
_KONTEXT_MODELS = {"flux-kontext-pro", "flux-kontext-max"}

_TERMINAL_OK = {"Ready"}
_TERMINAL_ERR = {"Error", "Failed", "Request Moderated", "Content Moderated", "Task not found"}


def _build_body(prompt: str, options: dict[str, Any], model: str) -> dict[str, Any]:
    """Build the JSON body for a BFL ``POST /v1/<model>`` request.

    Splits behaviour for kontext (editing) vs. plain text-to-image models.
    """
    body: dict[str, Any] = {"prompt": prompt}

    if model in _KONTEXT_MODELS:
        # Image-editing models require an ``input_image`` (base64 or URL).
        input_image = options.get("input_image") or options.get("image")
        if input_image is not None:
            body["input_image"] = input_image
        for k in ("aspect_ratio", "output_format", "safety_tolerance", "seed", "prompt_upsampling"):
            if k in options:
                body[k] = options[k]
        return body

    # Standard Flux models — width/height + tuning knobs.
    body["width"] = int(options.get("width", 1024))
    body["height"] = int(options.get("height", 1024))
    body["prompt_upsampling"] = bool(options.get("prompt_upsampling", False))
    if "seed" in options:
        body["seed"] = options["seed"]
    body["safety_tolerance"] = int(options.get("safety_tolerance", 2))
    body["output_format"] = str(options.get("output_format", "jpeg"))

    # Ultra-only knobs.
    if model == "flux-pro-1.1-ultra":
        if "aspect_ratio" in options:
            body["aspect_ratio"] = options["aspect_ratio"]
        if "raw" in options:
            body["raw"] = bool(options["raw"])

    return body


class BFLImageGenDriver(ImageCostMixin, ImageGenDriver):
    """Image generation via the Black Forest Labs (BFL) API."""

    supports_multiple = False
    supports_size_variants = True
    supported_sizes = ["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024", "2048x2048"]
    max_images = 1

    KNOWN_MODELS = sorted(_BFL_MODELS)

    # Per-image USD pricing (approximate; canonical pricing lives in the local KB).
    IMAGE_PRICING: dict[str, dict[str, float]] = {
        "flux-pro-1.1": {"default": 0.04},
        "flux-pro-1.1-ultra": {"default": 0.06},
        "flux-pro": {"default": 0.05},
        "flux-dev": {"default": 0.025},
        "flux-schnell": {"default": 0.003},
        "flux-kontext-pro": {"default": 0.04},
        "flux-kontext-max": {"default": 0.08},
    }

    # Polling tuning — kept as class attributes so tests can override.
    POLL_INTERVAL_SECONDS: float = 2.0
    POLL_MAX_RETRIES: int = 60

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        endpoint: str | None = None,
    ):
        self.api_key = api_key or os.getenv("BFL_API_KEY")
        self.model = model
        self.endpoint = (endpoint or os.getenv("BFL_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

    @classmethod
    def list_models(cls, **kw: object) -> list[str] | None:
        return list(cls.KNOWN_MODELS)

    def _headers(self) -> dict[str, str]:
        return {"x-key": self.api_key or "", "Content-Type": "application/json"}

    def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("BFL_API_KEY is not configured")
        if not prompt:
            raise ValueError("prompt cannot be empty")

        model = options.get("model", self.model)
        body = _build_body(prompt, options, model)
        submit_url = f"{self.endpoint}/v1/{model}"

        poll_interval = float(options.get("poll_interval", self.POLL_INTERVAL_SECONDS))
        max_retries = int(options.get("max_retries", self.POLL_MAX_RETRIES))

        with httpx.Client(timeout=120.0) as client:
            submit = client.post(submit_url, headers=self._headers(), json=body)
            if submit.status_code >= 400:
                raise RuntimeError(f"BFL submit failed {submit.status_code}: {submit.text}")
            submitted = submit.json()
            request_id = submitted.get("id")
            polling_url = submitted.get("polling_url")
            if not request_id or not polling_url:
                raise RuntimeError(f"BFL response missing id/polling_url: {submitted}")

            if not options.get("poll", True):
                return {
                    "images": [],
                    "meta": {
                        "image_count": 0,
                        "size": f"{body.get('width', '?')}x{body.get('height', '?')}",
                        "revised_prompt": None,
                        "cost": 0.0,
                        "model_name": f"bfl/{model}",
                        "request_id": request_id,
                        "polling_url": polling_url,
                        "status": "pending",
                        "raw_response": submitted,
                    },
                }

            final = self._poll(client, polling_url, max_retries=max_retries, poll_interval=poll_interval)

        result_payload = final.get("result") or {}
        sample = result_payload.get("sample")
        images = [image_from_url(sample)] if isinstance(sample, str) and sample else []
        cost = self._calculate_image_cost("bfl", model, n=max(len(images), 1))

        return {
            "images": images,
            "meta": {
                "image_count": len(images),
                "size": f"{body.get('width', '?')}x{body.get('height', '?')}",
                "revised_prompt": None,
                "cost": cost,
                "model_name": f"bfl/{model}",
                "request_id": request_id,
                "polling_url": polling_url,
                "raw_response": final,
            },
        }

    def _poll(
        self,
        client: httpx.Client,
        polling_url: str,
        *,
        max_retries: int,
        poll_interval: float,
    ) -> dict[str, Any]:
        for _attempt in range(max_retries):
            r = client.get(polling_url, headers=self._headers())
            if r.status_code >= 400:
                raise RuntimeError(f"BFL poll failed {r.status_code}: {r.text}")
            data = r.json()
            status = data.get("status")
            if status in _TERMINAL_OK:
                return data
            if status in _TERMINAL_ERR:
                raise RuntimeError(f"BFL job {status}: {data}")
            time.sleep(poll_interval)
        raise TimeoutError(f"BFL job timed out after {max_retries} retries (last status={status!r})")
