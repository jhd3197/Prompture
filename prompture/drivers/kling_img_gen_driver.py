"""Kling AI image generation driver.

Kling exposes a polled task-based REST API at ``api-singapore.klingai.com``.
Authentication uses a short-lived (30-minute) HS256 JWT signed from a pair of
access/secret keys, mirroring the official upstream pattern.

This driver covers ``POST /v1/images/generations`` (text-to-image and
single-reference image-to-image). For multi-image composition, see
``generate_image`` with ``subject_images`` in ``options``, which routes to
``POST /v1/images/multi-image2image``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import time
from typing import Any

import httpx

from ..infra.cost_mixin import ImageCostMixin
from ..media.image import image_from_url
from .img_gen_base import ImageGenDriver

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api-singapore.klingai.com"
_DATA_URL_RE = re.compile(r"^data:[^;]+;base64,")

_KLING_IMAGE_MODELS = {"kling-v1-5", "kling-v2-1", "kling-v2"}
_KLING_RATIOS = {"1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"}
_KLING_RESOLUTIONS = {"1k", "2k"}


def _get_kling_credentials(
    access_key: str | None = None, secret_key: str | None = None
) -> tuple[str | None, str | None]:
    return (
        access_key or os.getenv("KLING_ACCESS_KEY"),
        secret_key or os.getenv("KLING_SECRET_KEY"),
    )


def _b64url(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def generate_kling_jwt(access_key: str, secret_key: str, ttl_seconds: int = 1800) -> str:
    """Generate an HS256 JWT for the Kling API. Valid for ``ttl_seconds``."""
    if not access_key or not secret_key:
        raise RuntimeError("Kling API credentials not configured")
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {"iss": access_key, "exp": now + ttl_seconds, "nbf": now - 5}
    h_enc = _b64url(json.dumps(header, separators=(",", ":")).encode())
    p_enc = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{h_enc}.{p_enc}".encode()
    sig = hmac.new(secret_key.encode(), signing_input, hashlib.sha256).digest()
    return f"{h_enc}.{p_enc}.{_b64url(sig)}"


def _strip_data_url(value: str) -> str:
    """Strip a ``data:image/...;base64,`` prefix from a base64 string."""
    return _DATA_URL_RE.sub("", value) if isinstance(value, str) else value


def _normalize_image(value: Any) -> str:
    """Normalize a base64/data-URL/http-URL image input into a Kling-acceptable form.

    Kling accepts either a publicly reachable URL or a raw base64 string.
    """
    if not isinstance(value, str):
        raise ValueError("image input must be a string (URL or base64)")
    if value.startswith(("http://", "https://")):
        return value
    return _strip_data_url(value)


class KlingImageGenDriver(ImageCostMixin, ImageGenDriver):
    """Image generation via Kling AI."""

    supports_multiple = False
    supports_size_variants = True
    supported_sizes = sorted(_KLING_RATIOS)
    max_images = 1

    KNOWN_MODELS = sorted(_KLING_IMAGE_MODELS)

    # USD per image. Public Kling pricing is in credits and varies by mode;
    # these are conservative placeholder rates — update via the local KB.
    IMAGE_PRICING: dict[str, dict[str, float]] = {
        "kling-v1-5": {"1k": 0.014, "2k": 0.028, "default": 0.014},
        "kling-v2-1": {"1k": 0.028, "2k": 0.056, "default": 0.028},
        "kling-v2": {"1k": 0.028, "2k": 0.056, "default": 0.028},
    }

    def __init__(
        self,
        access_key: str | None = None,
        secret_key: str | None = None,
        model: str = "kling-v2-1",
        endpoint: str | None = None,
    ):
        ak, sk = _get_kling_credentials(access_key, secret_key)
        self.access_key = ak
        self.secret_key = sk
        self.model = model
        self.endpoint = (endpoint or os.getenv("KLING_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")

    @classmethod
    def list_models(cls, **kw: object) -> list[str] | None:
        return list(cls.KNOWN_MODELS)

    def _token(self) -> str:
        if not self.access_key or not self.secret_key:
            raise RuntimeError("KLING_ACCESS_KEY and KLING_SECRET_KEY must be configured")
        return generate_kling_jwt(self.access_key, self.secret_key)

    def _headers(self, token: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def _build_body(self, prompt: str, options: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Return (path, body) for the request. Routes multi-image when ``subject_images`` is given."""
        model = options.get("model", self.model)
        ratio = options.get("aspect_ratio", options.get("ratio", "16:9"))
        resolution = str(options.get("resolution", "1k")).lower()
        if ratio not in _KLING_RATIOS:
            raise ValueError(f"Unsupported aspect_ratio '{ratio}'")
        if resolution not in _KLING_RESOLUTIONS:
            raise ValueError(f"Unsupported resolution '{resolution}'")

        subject_images = options.get("subject_images")
        if subject_images:
            return self._build_multi_image_body(prompt, model, ratio, resolution, subject_images, options)

        body: dict[str, Any] = {
            "model_name": model,
            "prompt": prompt,
            "aspect_ratio": ratio,
            "image_size": resolution,
            "n": int(options.get("n", 1)),
        }

        image = options.get("image")
        if image is not None:
            first = image[0] if isinstance(image, list) else image
            body["image"] = _normalize_image(first)
            if model == "kling-v1-5":
                ref_mode = options.get("reference_mode", "subject")
                body["image_reference"] = ref_mode
                if ref_mode == "subject":
                    body["face_reference_intensity"] = float(options.get("face_intensity", 65)) / 100
                    body["subject_reference_intensity"] = float(options.get("subject_intensity", 50)) / 100
                elif ref_mode == "face":
                    body["face_reference_intensity"] = float(options.get("face_intensity", 42)) / 100
        return "/v1/images/generations", body

    def _build_multi_image_body(
        self,
        prompt: str,
        model: str,
        ratio: str,
        resolution: str,
        subject_images: list[Any],
        options: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if model not in {"kling-v2", "kling-v2-1"}:
            model = "kling-v2-1"
        body: dict[str, Any] = {
            "model_name": model,
            "prompt": prompt,
            "aspect_ratio": ratio,
            "image_size": resolution,
            "n": int(options.get("n", 1)),
            "subject_image_list": [{"subject_image": _normalize_image(img)} for img in subject_images[:4]],
        }
        if options.get("scene_image"):
            body["scene_image"] = _normalize_image(options["scene_image"])
        if options.get("style_image"):
            body["style_image"] = _normalize_image(options["style_image"])
        return "/v1/images/multi-image2image", body

    def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not prompt:
            raise ValueError("prompt cannot be empty")
        token = self._token()
        path, body = self._build_body(prompt, options)
        is_multi = path.endswith("multi-image2image")

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{self.endpoint}{path}", headers=self._headers(token), json=body)
            if resp.status_code >= 400:
                raise RuntimeError(f"Kling API error {resp.status_code}: {resp.text}")
            result = resp.json()
            if result.get("code") != 0:
                raise RuntimeError(f"Kling API error: {result.get('message')}")
            task_id = result.get("data", {}).get("task_id")
            if not task_id:
                raise RuntimeError(f"Kling response missing task_id: {result}")

            if not options.get("poll", True):
                return self._pending_response(task_id, body, result)

            final = self._poll_image(
                client,
                task_id,
                token,
                multi=is_multi,
                poll_interval=float(options.get("poll_interval", 3)),
                timeout_seconds=float(options.get("timeout", 120)),
            )

        urls = [
            img["url"] for img in (final.get("data", {}).get("task_result", {}).get("images") or []) if img.get("url")
        ]
        images = [image_from_url(u) for u in urls]
        cost = self._calculate_image_cost(
            "kling", body["model_name"], size=str(body["image_size"]), n=max(len(images), 1)
        )
        return {
            "images": images,
            "meta": {
                "image_count": len(images),
                "size": body["aspect_ratio"],
                "revised_prompt": None,
                "cost": cost,
                "model_name": f"kling/{body['model_name']}",
                "task_id": task_id,
                "raw_response": final,
            },
        }

    def _poll_image(
        self,
        client: httpx.Client,
        task_id: str,
        token: str,
        *,
        multi: bool,
        poll_interval: float,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        base = "multi-image2image" if multi else "generations"
        path = f"/v1/images/{base}/{task_id}"
        deadline = time.monotonic() + timeout_seconds
        while True:
            r = client.get(f"{self.endpoint}{path}", headers=self._headers(token))
            if r.status_code >= 400:
                raise RuntimeError(f"Kling poll failed {r.status_code}: {r.text}")
            data = r.json()
            if data.get("code") != 0:
                raise RuntimeError(f"Kling poll error: {data.get('message')}")
            status = data.get("data", {}).get("task_status")
            if status == "succeed":
                return data
            if status == "failed":
                msg = data.get("data", {}).get("task_status_msg", "unknown")
                raise RuntimeError(f"Kling image task failed: {msg}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Kling image task {task_id} timed out (status={status})")
            time.sleep(poll_interval)

    def _pending_response(self, task_id: str, body: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "images": [],
            "meta": {
                "image_count": 0,
                "size": body.get("aspect_ratio"),
                "revised_prompt": None,
                "cost": 0.0,
                "model_name": f"kling/{body['model_name']}",
                "task_id": task_id,
                "status": "pending",
                "raw_response": raw,
            },
        }
