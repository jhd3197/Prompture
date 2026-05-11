"""Runway image generation driver (``POST /v1/text_to_image``).

Submits a generation task to the Runway public API, polls
``GET /v1/tasks/{id}`` until terminal, and returns the resulting image as an
:class:`ImageContent`. Reference images are passed through as
``{"uri": ..., "tag": ...}`` records.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import ImageContent, image_from_url
from .img_gen_base import ImageGenDriver

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.dev.runwayml.com"
_API_VERSION = "2024-11-06"

_TERMINAL_OK = {"SUCCEEDED"}
_TERMINAL_FAIL = {"FAILED", "CANCELLED"}

# Default ratios per model — used when caller does not supply one.
_DEFAULT_RATIOS: dict[str, str] = {
    "gen4_image": "1280:720",
    "gen4_image_turbo": "1280:720",
    "gemini_2.5_flash": "1344:768",
    "gpt_image_2": "1920:1088",
}


def _get_runway_api_key(api_key: str | None = None) -> str | None:
    return (
        api_key
        or os.getenv("RUNWAY_API_KEY")
        or os.getenv("RUNWAYML_API_SECRET")
    )


def _normalize_reference_images(refs: Any) -> list[dict[str, str]]:
    if not refs:
        return []
    if not isinstance(refs, list):
        raise ValueError("reference_images must be a list of {uri, tag} dicts")
    out: list[dict[str, str]] = []
    for item in refs:
        if not isinstance(item, dict):
            raise ValueError("each reference image must be a dict with 'uri' and 'tag'")
        uri = item.get("uri") or item.get("url")
        tag = item.get("tag")
        if not uri or not tag:
            raise ValueError("reference image entries require both 'uri' and 'tag'")
        out.append({"uri": str(uri), "tag": str(tag)})
    return out


def _format_error(status_code: int, response: httpx.Response) -> str:
    try:
        body = response.json()
    except Exception:
        return f"HTTP {status_code}: {response.text[:500]}"
    detail = body.get("error") or body.get("message") or ""
    issues = body.get("issues") or []
    if issues:
        parts = [
            f"{(iss.get('path') or ['?'])[-1]}: {iss.get('message', '')}"
            for iss in issues
        ]
        detail = f"{detail} [{'; '.join(parts)}]"
    return f"HTTP {status_code}: {detail}".strip()


class RunwayImageGenDriver(ImageCostMixin, ImageGenDriver):
    """Image generation via Runway ``POST /v1/text_to_image``."""

    supports_multiple = False
    supports_size_variants = True
    # Ratio strings accepted by Runway; this list is informational, not exhaustive.
    supported_sizes = [
        "1920:1080",
        "1080:1920",
        "1024:1024",
        "1360:768",
        "1080:1080",
        "1168:880",
        "1440:1080",
        "1080:1440",
        "1808:768",
        "2112:912",
        "1280:720",
        "720:1280",
        "1344:768",
        "768:1344",
        "1920:1088",
        "1088:1920",
    ]
    max_images = 1

    KNOWN_MODELS = [
        "gen4_image",
        "gen4_image_turbo",
        "gemini_2.5_flash",
        "gpt_image_2",
    ]

    # USD per image. Credits → USD assumed at $0.01/credit; refine when Runway
    # publishes authoritative dollar pricing.
    IMAGE_PRICING: dict[str, dict[str, float]] = {
        "gen4_image": {"720p": 0.05, "1080p": 0.08, "default": 0.05},
        "gen4_image_turbo": {"default": 0.02},
        "gemini_2.5_flash": {"default": 0.05},
        "gpt_image_2": {
            "low": 0.04,
            "medium": 0.06,
            "high": 0.08,
            "default": 0.08,
        },
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gen4_image",
        endpoint: str | None = None,
    ):
        self.api_key = _get_runway_api_key(api_key)
        self.model = model
        self.endpoint = (
            endpoint or os.getenv("RUNWAY_ENDPOINT") or _DEFAULT_ENDPOINT
        ).rstrip("/")

    @classmethod
    def list_models(cls, *, api_key: str | None = None, **kw: object) -> list[str] | None:
        key = _get_runway_api_key(api_key)
        if not key:
            return None
        return list(cls.KNOWN_MODELS)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Runway-Version": _API_VERSION,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _build_body(self, prompt: str, options: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if not prompt:
            raise ValueError("prompt cannot be empty")

        model = options.get("model", self.model)
        ratio = (
            options.get("ratio")
            or options.get("aspect_ratio")
            or _DEFAULT_RATIOS.get(model, "1280:720")
        )

        body: dict[str, Any] = {
            "model": model,
            "promptText": prompt,
            "ratio": ratio,
        }

        if model == "gpt_image_2":
            body["quality"] = options.get("quality", "high")
        elif options.get("quality") is not None:
            raise ValueError(
                f"'quality' is only supported by gpt_image_2 (got model={model!r})"
            )

        refs = _normalize_reference_images(options.get("reference_images"))
        if refs:
            body["referenceImages"] = refs
        elif model == "gen4_image_turbo":
            raise ValueError("gen4_image_turbo requires reference_images")

        seed = options.get("seed")
        if seed is not None:
            body["seed"] = int(seed)

        return model, body

    def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate an image via Runway ``POST /v1/text_to_image``.

        Options:
            model: Override the default model.
            ratio / aspect_ratio: Ratio string like ``"1920:1088"``.
            quality: ``low`` | ``medium`` | ``high`` — gpt_image_2 only.
            reference_images: List of ``{"uri": ..., "tag": ...}`` records.
            seed: Optional integer seed.
            poll: Whether to poll the task to completion. Defaults to True.
            poll_interval: Seconds between polls. Defaults to 5.
            timeout: Max polling seconds. Defaults to 600.
        """
        if not self.api_key:
            raise RuntimeError(
                "RUNWAY_API_KEY (or RUNWAYML_API_SECRET) is not configured"
            )

        model, body = self._build_body(prompt, options)
        headers = self._headers()

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{self.endpoint}/v1/text_to_image", headers=headers, json=body)
            if resp.status_code >= 400:
                raise RuntimeError(f"Runway text_to_image failed: {_format_error(resp.status_code, resp)}")
            task = resp.json()
            task_id = task.get("id")
            if not task_id:
                raise RuntimeError(f"Runway response missing task id: {task}")

            if not options.get("poll", True):
                return self._pending_response(task_id, model, body, task)

            final = self._poll_until_done(
                client,
                task_id,
                headers=headers,
                poll_interval=float(options.get("poll_interval", 5)),
                timeout_seconds=float(options.get("timeout", 600)),
            )

        return self._build_success_response(final, model, body, options)

    # ------------------------------------------------------------------

    def _poll_until_done(
        self,
        client: httpx.Client,
        task_id: str,
        *,
        headers: dict[str, str],
        poll_interval: float,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            r = client.get(f"{self.endpoint}/v1/tasks/{task_id}", headers=headers)
            if r.status_code >= 400:
                raise RuntimeError(f"Runway task poll failed: {_format_error(r.status_code, r)}")
            data = r.json()
            status = data.get("status")
            if status in _TERMINAL_OK:
                return data
            if status in _TERMINAL_FAIL:
                failure = data.get("failure") or data.get("failureCode") or status
                raise RuntimeError(f"Runway task {task_id} {status}: {failure}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Runway task {task_id} timed out (status={status})")
            time.sleep(poll_interval)

    def _pending_response(
        self,
        task_id: str,
        model: str,
        body: dict[str, Any],
        task: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "images": [],
            "meta": {
                "image_count": 0,
                "size": body.get("ratio"),
                "revised_prompt": None,
                "cost": 0.0,
                "model_name": f"runway/{model}",
                "task_id": task_id,
                "status": task.get("status", "PENDING"),
                "raw_response": task,
            },
        }

    def _build_success_response(
        self,
        final: dict[str, Any],
        model: str,
        body: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        outputs = final.get("output") or []
        images: list[ImageContent] = [image_from_url(url, media_type="image/png") for url in outputs]

        ratio = body.get("ratio", "")
        size_key = self._cost_size_key(model, ratio, body.get("quality"))
        cost = self._calculate_image_cost(
            "runway",
            model,
            size=size_key,
            quality=body.get("quality", "standard"),
            n=max(len(images), 1),
        )

        return {
            "images": images,
            "meta": {
                "image_count": len(images),
                "size": ratio,
                "revised_prompt": None,
                "cost": cost,
                "model_name": f"runway/{model}",
                "task_id": final.get("id"),
                "status": final.get("status"),
                "raw_response": final,
            },
        }

    @staticmethod
    def _cost_size_key(model: str, ratio: str, quality: str | None) -> str:
        """Map a Runway ratio to a key in IMAGE_PRICING.

        gen4_image charges 5 credits at 720p tier and 8 at 1080p tier; we
        approximate the tier by the larger dimension in the ratio.
        """
        if model == "gpt_image_2":
            return quality or "high"
        try:
            w, h = (int(part) for part in ratio.split(":"))
        except (ValueError, AttributeError):
            return "default"
        long_side = max(w, h)
        if long_side >= 1600:
            return "1080p"
        return "720p"
