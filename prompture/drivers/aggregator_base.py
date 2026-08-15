"""Generic submit→poll media aggregator base (sync).

A Muapi / Replicate / Novita / PiAPI-style aggregator exposes many model
families behind a single key and a uniform lifecycle: ``POST`` to submit a job,
get back a request id, then ``GET`` a result endpoint until it's done. This
module captures that lifecycle once; a concrete aggregator subclasses it and
supplies only the small provider-specific bits:

- ``AUTH_STYLE`` / ``API_KEY_ENVS`` / ``DEFAULT_BASE`` / ``ENDPOINT_ENV``
- ``submit_url(slug)`` / ``result_url(request_id)`` / ``upload_url()``
- ``parse_request_id(data)`` / ``extract_urls(result)`` / ``status_of(raw)``
- ``build_image_payload(...)`` / ``build_video_payload(...)``

:class:`AggregatorImageDriver` and :class:`AggregatorVideoDriver` then provide
``generate_image`` / ``edit_image`` / ``generate_video`` / ``resume_job`` for
free. See :mod:`muapi_aggregator_driver` for the reference subclass — a new
aggregator is ~40 lines of overrides.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from ..jobs import JobHandle, JobResult, JobStatus, MediaAsset
from ..media.audio import audio_from_url
from ..media.image import image_from_url
from ..media.video import video_from_url
from .img_gen_base import ImageGenDriver
from .job_runner import normalize_status, poll_job
from .lipsync_base import LipsyncDriver
from .music_base import MusicGenDriver
from .video_gen_base import VideoGenDriver

logger = logging.getLogger(__name__)


def extract_media_urls(result: dict[str, Any]) -> list[str]:
    """Normalize an aggregator result payload to a list of output URLs.

    Handles the common shapes: ``outputs`` (list of str or ``{"url"}`` dicts),
    a top-level ``url``, or ``output: {"url"}``.
    """
    outputs = result.get("outputs")
    if isinstance(outputs, list):
        urls: list[str] = []
        for o in outputs:
            if isinstance(o, str):
                urls.append(o)
            elif isinstance(o, dict) and o.get("url"):
                urls.append(o["url"])
        if urls:
            return urls
    if isinstance(result.get("url"), str):
        return [result["url"]]
    output = result.get("output")
    if isinstance(output, dict) and output.get("url"):
        return [output["url"]]
    return []


class AggregatorClient:
    """Shared HTTP + job plumbing for submit→poll media aggregators."""

    # ── Provider config (override in subclasses) ──
    PROVIDER: str = "aggregator"
    DEFAULT_BASE: str = ""
    API_KEY_ENVS: tuple[str, ...] = ()
    ENDPOINT_ENV: str = ""
    AUTH_STYLE: str = "x-api-key"  # x-api-key | bearer | key | token
    DEFAULT_IMAGE_MODEL: str = ""
    DEFAULT_VIDEO_MODEL: str = ""
    DEFAULT_LIPSYNC_MODEL: str = ""
    DEFAULT_MUSIC_MODEL: str = ""

    api_key: str | None
    endpoint: str
    model: str

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.api_key = self._get_key(api_key)
        self.model = model or self._default_model()
        env_endpoint = os.getenv(self.ENDPOINT_ENV) if self.ENDPOINT_ENV else None
        self.endpoint = (endpoint or env_endpoint or self.DEFAULT_BASE).rstrip("/")

    # ── Provider hooks (override) ──
    def _default_model(self) -> str:
        return self.DEFAULT_IMAGE_MODEL

    def submit_url(self, slug: str) -> str:
        raise NotImplementedError

    def result_url(self, request_id: str) -> str:
        raise NotImplementedError

    def upload_url(self) -> str:
        raise NotImplementedError(f"{self.PROVIDER} does not support uploads")

    def parse_request_id(self, data: dict[str, Any]) -> str | None:
        return data.get("request_id") or data.get("id")

    def extract_urls(self, result: dict[str, Any]) -> list[str]:
        return extract_media_urls(result)

    def status_of(self, raw: dict[str, Any]) -> Any:
        return raw.get("status")

    def build_image_payload(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {"prompt": prompt} if prompt else {}

    def build_video_payload(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {"prompt": prompt} if prompt else {}

    # ── Auth / HTTP ──
    def _get_key(self, api_key: str | None) -> str | None:
        if api_key:
            return api_key
        for env in self.API_KEY_ENVS:
            val = os.getenv(env)
            if val:
                return val
        return None

    def _auth_pair(self) -> tuple[str, str]:
        key = self.api_key or ""
        style = self.AUTH_STYLE
        if style == "bearer":
            return ("Authorization", f"Bearer {key}")
        if style == "key":
            return ("Authorization", f"Key {key}")
        if style == "token":
            return ("Authorization", f"Token {key}")
        return ("x-api-key", key)

    def _headers(self) -> dict[str, str]:
        name, val = self._auth_pair()
        return {name: val, "Content-Type": "application/json"}

    def _upload_headers(self) -> dict[str, str]:
        name, val = self._auth_pair()
        return {name: val}

    def _require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(f"{self.PROVIDER.upper()}_API_KEY is not configured")

    def _submit(self, slug: str, payload: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        resp = httpx.post(self.submit_url(slug), headers=self._headers(), json=payload, timeout=120.0)
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.PROVIDER} submit failed {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return self.parse_request_id(data), data

    def _fetch_once(self, request_id: str) -> dict[str, Any]:
        resp = httpx.get(self.result_url(request_id), headers=self._headers(), timeout=120.0)
        if resp.status_code >= 500:
            return {"status": "processing"}  # transient: keep polling
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.PROVIDER} result fetch failed {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def _poll(self, request_id: str, *, interval: float, timeout: float) -> dict[str, Any]:
        return poll_job(
            lambda: self._fetch_once(request_id),
            interval=interval,
            timeout=timeout,
            status_of=self.status_of,
        )

    def upload_file(self, data: bytes, *, filename: str = "input.png") -> str:
        """Upload bytes to the aggregator's host and return the public URL."""
        self._require_key()
        resp = httpx.post(
            self.upload_url(),
            headers=self._upload_headers(),
            files={"file": (filename, data)},
            timeout=120.0,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.PROVIDER} upload failed {resp.status_code}: {resp.text[:200]}")
        dj = resp.json()
        file_url = dj.get("url") or dj.get("file_url") or (dj.get("data") or {}).get("url")
        if not file_url:
            raise RuntimeError(f"{self.PROVIDER} upload returned no URL: {dj}")
        return file_url

    # ── Job handle / resume ──
    def _make_handle(self, slug: str, request_id: str, modality: str) -> JobHandle:
        return JobHandle(
            task_id=request_id,
            provider=self.PROVIDER,
            model=slug,
            modality=modality,
            status=JobStatus.PENDING.value,
            endpoint=self.endpoint,
            polling_url=self.result_url(request_id),
            submitted_at=time.time(),
        )

    def get_job_status(self, handle: JobHandle) -> str:
        self._require_key()
        return normalize_status(self.status_of(self._fetch_once(handle.task_id)))

    def _resume(
        self,
        handle: JobHandle,
        kind: str,
        *,
        poll: bool,
        poll_interval: float | None,
        timeout: float | None,
    ) -> JobResult:
        self._require_key()
        if poll:
            raw = self._poll(handle.task_id, interval=poll_interval or 2.0, timeout=timeout or 900.0)
        else:
            raw = self._fetch_once(handle.task_id)
        status = normalize_status(self.status_of(raw))
        assets = [
            MediaAsset(kind=kind, url=u, model=handle.model, provenance={"request_id": handle.task_id})
            for u in self.extract_urls(raw)
        ]
        return JobResult(
            handle=handle.with_status(status),
            status=status,
            assets=assets,
            meta={"request_id": handle.task_id, "model_name": f"{self.PROVIDER}/{handle.model}"},
            raw=raw,
        )

    # ── Cost (G16 quote) ──
    def _media_cost(
        self, slug: str, *, n: int = 1, duration_seconds: float = 0.0, resolution: str | None = None
    ) -> float:
        from ..infra.media_pricing import estimate_media_cost

        return estimate_media_cost(
            f"{self.PROVIDER}/{slug}",
            n=n,
            duration_seconds=duration_seconds,
            resolution=resolution,
        )

    # ── Media input resolution (for conditioned ops: i2i / i2v / lipsync) ──
    def _coerce_media_url(self, source: Any, *, explicit: str | None = None, filename: str = "input.bin") -> str | None:
        """Resolve a media input to a hosted URL.

        Accepts a URL string (used as-is), raw bytes / file path / ``*Content``
        (uploaded via :meth:`upload_file`), or ``None``. ``explicit`` short-circuits
        with an already-hosted URL from options.
        """
        if explicit:
            return explicit
        if source is None:
            return None
        url = getattr(source, "url", None)  # *Content with a URL source
        if url:
            return url
        if isinstance(source, str):
            if source.startswith(("http://", "https://")):
                return source
            from ..media.hosting import resolve_to_bytes

            return self.upload_file(resolve_to_bytes(source), filename=filename)
        if isinstance(source, (bytes, bytearray)):
            return self.upload_file(bytes(source), filename=filename)
        data = getattr(source, "data", None)  # base64-bearing *Content
        if data:
            import base64

            return self.upload_file(base64.b64decode(data), filename=filename)
        from pathlib import Path

        if isinstance(source, Path):
            return self.upload_file(source.read_bytes(), filename=filename)
        raise TypeError(f"Unsupported media source: {type(source).__name__}")


class AggregatorImageDriver(AggregatorClient, ImageGenDriver):
    """Generic image generation + i2i editing over an aggregator."""

    supports_multiple = True
    supports_size_variants = True
    supports_edit = True
    max_images = 4

    def _default_model(self) -> str:
        return self.DEFAULT_IMAGE_MODEL

    @classmethod
    def list_models(cls, **kw: object) -> list[str] | None:
        return None  # dynamic catalog — see media_capabilities / discovery

    def _finalize_image(
        self, slug: str, request_id: str | None, result: dict[str, Any], options: dict[str, Any]
    ) -> dict[str, Any]:
        images = [image_from_url(u) for u in self.extract_urls(result)]
        cost = self._media_cost(slug, n=max(len(images), 1))
        return {
            "images": images,
            "meta": {
                "image_count": len(images),
                "size": options.get("resolution") or options.get("aspect_ratio"),
                "revised_prompt": None,
                "cost": cost,
                "model_name": f"{self.PROVIDER}/{slug}",
                "request_id": request_id,
                "raw_response": result,
            },
        }

    def _pending_image(
        self, slug: str, request_id: str, submitted: dict[str, Any], options: dict[str, Any]
    ) -> dict[str, Any]:
        handle = self._make_handle(slug, request_id, "image")
        return {
            "images": [],
            "meta": {
                "image_count": 0,
                "size": options.get("resolution") or options.get("aspect_ratio"),
                "revised_prompt": None,
                "cost": 0.0,
                "model_name": f"{self.PROVIDER}/{slug}",
                "request_id": request_id,
                "status": "pending",
                "job_handle": handle.to_dict(),
                "raw_response": submitted,
            },
        }

    def _run_image(self, slug: str, payload: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        request_id, submitted = self._submit(slug, payload)
        if request_id is None:
            return self._finalize_image(slug, None, submitted, options)
        if not options.get("poll", True):
            return self._pending_image(slug, request_id, submitted, options)
        result = self._poll(
            request_id,
            interval=float(options.get("poll_interval", 2)),
            timeout=float(options.get("timeout", 300)),
        )
        return self._finalize_image(slug, request_id, result, options)

    def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self._require_key()
        slug = options.get("model", self.model)
        return self._run_image(slug, self.build_image_payload(prompt, options), options)

    def edit_image(
        self, image: bytes, prompt: str, options: dict[str, Any], *, mask: bytes | None = None
    ) -> dict[str, Any]:
        self._require_key()
        slug = options.get("model", self.model)
        payload = self.build_image_payload(prompt, options)
        if "image_url" not in payload:
            payload["image_url"] = self.upload_file(image, filename="input.png")
        if mask is not None:
            payload["mask_url"] = self.upload_file(mask, filename="mask.png")
        return self._run_image(slug, payload, options)

    def resume_job(
        self,
        handle: JobHandle,
        *,
        poll: bool = True,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> JobResult:
        return self._resume(handle, "image", poll=poll, poll_interval=poll_interval, timeout=timeout)


class AggregatorVideoDriver(AggregatorClient, VideoGenDriver):
    """Generic video generation (t2v / i2v / v2v) over an aggregator."""

    supports_image_input = True
    supports_reference_images = True
    supports_video_input = True
    supports_audio = True
    supports_polling = True

    def _default_model(self) -> str:
        return self.DEFAULT_VIDEO_MODEL

    @classmethod
    def list_models(cls, **kw: object) -> list[str] | None:
        return None

    def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self._require_key()
        slug = options.get("model", self.model)
        payload = self.build_video_payload(prompt, options)
        request_id, submitted = self._submit(slug, payload)

        if request_id is not None and not options.get("poll", True):
            handle = self._make_handle(slug, request_id, "video")
            return {
                "videos": [],
                "meta": {
                    "video_count": 0,
                    "duration_seconds": None,
                    "aspect_ratio": options.get("aspect_ratio"),
                    "resolution": options.get("resolution"),
                    "cost": 0.0,
                    "model_name": f"{self.PROVIDER}/{slug}",
                    "request_id": request_id,
                    "status": "pending",
                    "job_handle": handle.to_dict(),
                    "raw_response": submitted,
                },
            }

        result = (
            submitted
            if request_id is None
            else self._poll(
                request_id,
                interval=float(options.get("poll_interval", 3)),
                timeout=float(options.get("timeout", 900)),
            )
        )
        videos = [video_from_url(u) for u in self.extract_urls(result)]
        duration = float(options.get("duration", 0) or 0)
        cost = self._media_cost(slug, n=1, duration_seconds=duration, resolution=options.get("resolution"))
        return {
            "videos": videos,
            "meta": {
                "video_count": len(videos),
                "duration_seconds": duration or None,
                "aspect_ratio": options.get("aspect_ratio"),
                "resolution": options.get("resolution"),
                "cost": cost,
                "model_name": f"{self.PROVIDER}/{slug}",
                "request_id": request_id,
                "raw_response": result,
            },
        }

    def resume_job(
        self,
        handle: JobHandle,
        *,
        poll: bool = True,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> JobResult:
        return self._resume(handle, "video", poll=poll, poll_interval=poll_interval, timeout=timeout)


class AggregatorLipsyncDriver(AggregatorClient, LipsyncDriver):
    """Generic lipsync (image|video + audio → video) over an aggregator."""

    supports_image = True
    supports_video = True

    def _default_model(self) -> str:
        return self.DEFAULT_LIPSYNC_MODEL

    @classmethod
    def list_models(cls, **kw: object) -> list[str] | None:
        return None

    def build_lipsync_payload(
        self,
        options: dict[str, Any],
        *,
        audio_url: str | None,
        image_url: str | None = None,
        video_url: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if audio_url:
            payload["audio_url"] = audio_url
        if image_url:
            payload["image_url"] = image_url
        if video_url:
            payload["video_url"] = video_url
        for k in ("prompt", "resolution", "seed"):
            v = options.get(k)
            if v is not None:
                payload[k] = v
        extra = options.get("extra")
        if isinstance(extra, dict):
            payload.update(extra)
        return payload

    def generate_lipsync(
        self,
        audio: Any,
        options: dict[str, Any],
        *,
        image: Any | None = None,
        video: Any | None = None,
    ) -> dict[str, Any]:
        self._require_key()
        slug = options.get("model", self.model)
        audio_url = self._coerce_media_url(audio, explicit=options.get("audio_url"), filename="audio.mp3")
        image_url = self._coerce_media_url(image, explicit=options.get("image_url"), filename="image.png")
        video_url = self._coerce_media_url(video, explicit=options.get("video_url"), filename="video.mp4")
        category = "video" if video_url else "image"
        payload = self.build_lipsync_payload(options, audio_url=audio_url, image_url=image_url, video_url=video_url)

        request_id, submitted = self._submit(slug, payload)
        if request_id is not None and not options.get("poll", True):
            handle = self._make_handle(slug, request_id, "lipsync")
            return {
                "videos": [],
                "meta": {
                    "video_count": 0,
                    "category": category,
                    "cost": 0.0,
                    "model_name": f"{self.PROVIDER}/{slug}",
                    "request_id": request_id,
                    "status": "pending",
                    "job_handle": handle.to_dict(),
                    "raw_response": submitted,
                },
            }

        result = (
            submitted
            if request_id is None
            else self._poll(
                request_id,
                interval=float(options.get("poll_interval", 3)),
                timeout=float(options.get("timeout", 900)),
            )
        )
        videos = [video_from_url(u) for u in self.extract_urls(result)]
        duration = float(options.get("duration", 0) or 0)
        cost = self._media_cost(slug, n=1, duration_seconds=duration)
        return {
            "videos": videos,
            "meta": {
                "video_count": len(videos),
                "category": category,
                "cost": cost,
                "model_name": f"{self.PROVIDER}/{slug}",
                "request_id": request_id,
                "raw_response": result,
            },
        }

    def resume_job(
        self,
        handle: JobHandle,
        *,
        poll: bool = True,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> JobResult:
        return self._resume(handle, "video", poll=poll, poll_interval=poll_interval, timeout=timeout)


class AggregatorMusicDriver(AggregatorClient, MusicGenDriver):
    """Generic music generation (create / remix / extend / mashup) over an aggregator."""

    _MUSIC_KEYS = (
        "instrumental",
        "style",
        "duration",
        "operation",
        "continue_at",
        "audio_url",
        "lyrics",
        "voice_id",
        "model_version",
        "negative_prompt",
        "seed",
    )

    def _default_model(self) -> str:
        return self.DEFAULT_MUSIC_MODEL

    @classmethod
    def list_models(cls, **kw: object) -> list[str] | None:
        return None

    def build_music_payload(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if prompt:
            payload["prompt"] = prompt
        for k in self._MUSIC_KEYS:
            v = options.get(k)
            if v is not None:
                payload[k] = v
        extra = options.get("extra")
        if isinstance(extra, dict):
            payload.update(extra)
        return payload

    def generate_music(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self._require_key()
        slug = options.get("model", self.model)
        payload = self.build_music_payload(prompt, options)
        request_id, submitted = self._submit(slug, payload)

        if request_id is not None and not options.get("poll", True):
            handle = self._make_handle(slug, request_id, "music")
            return {
                "audio": [],
                "meta": {
                    "audio_count": 0,
                    "operation": options.get("operation", "create"),
                    "cost": 0.0,
                    "model_name": f"{self.PROVIDER}/{slug}",
                    "request_id": request_id,
                    "status": "pending",
                    "job_handle": handle.to_dict(),
                    "raw_response": submitted,
                },
            }

        result = (
            submitted
            if request_id is None
            else self._poll(
                request_id,
                interval=float(options.get("poll_interval", 3)),
                timeout=float(options.get("timeout", 900)),
            )
        )
        audio = [audio_from_url(u) for u in self.extract_urls(result)]
        cost = self._media_cost(slug, n=max(len(audio), 1))
        return {
            "audio": audio,
            "meta": {
                "audio_count": len(audio),
                "operation": options.get("operation", "create"),
                "cost": cost,
                "model_name": f"{self.PROVIDER}/{slug}",
                "request_id": request_id,
                "raw_response": result,
            },
        }

    def resume_job(
        self,
        handle: JobHandle,
        *,
        poll: bool = True,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> JobResult:
        return self._resume(handle, "audio", poll=poll, poll_interval=poll_interval, timeout=timeout)
