"""Generic submit→poll media aggregator base (async). Mirrors :mod:`aggregator_base`."""

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
from .aggregator_base import extract_media_urls
from .async_img_gen_base import AsyncImageGenDriver
from .async_video_gen_base import AsyncVideoGenDriver
from .job_runner import async_poll_job, normalize_status
from .lipsync_base import AsyncLipsyncDriver
from .music_base import AsyncMusicGenDriver

logger = logging.getLogger(__name__)


class AsyncAggregatorClient:
    """Shared async HTTP + job plumbing for submit→poll aggregators."""

    PROVIDER: str = "aggregator"
    DEFAULT_BASE: str = ""
    API_KEY_ENVS: tuple[str, ...] = ()
    ENDPOINT_ENV: str = ""
    AUTH_STYLE: str = "x-api-key"
    DEFAULT_IMAGE_MODEL: str = ""
    DEFAULT_VIDEO_MODEL: str = ""
    DEFAULT_LIPSYNC_MODEL: str = ""
    DEFAULT_MUSIC_MODEL: str = ""

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

    # ── Hooks (override) ──
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

    async def _submit(
        self, client: httpx.AsyncClient, slug: str, payload: dict[str, Any]
    ) -> tuple[str | None, dict[str, Any]]:
        resp = await client.post(self.submit_url(slug), headers=self._headers(), json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.PROVIDER} submit failed {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return self.parse_request_id(data), data

    async def _fetch_once(self, client: httpx.AsyncClient, request_id: str) -> dict[str, Any]:
        resp = await client.get(self.result_url(request_id), headers=self._headers())
        if resp.status_code >= 500:
            return {"status": "processing"}
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.PROVIDER} result fetch failed {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    async def _poll(
        self, client: httpx.AsyncClient, request_id: str, *, interval: float, timeout: float
    ) -> dict[str, Any]:
        return await async_poll_job(
            lambda: self._fetch_once(client, request_id),
            interval=interval,
            timeout=timeout,
            status_of=self.status_of,
        )

    async def upload_file(self, data: bytes, *, filename: str = "input.png") -> str:
        self._require_key()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                self.upload_url(), headers=self._upload_headers(), files={"file": (filename, data)}
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.PROVIDER} upload failed {resp.status_code}: {resp.text[:200]}")
        dj = resp.json()
        file_url = dj.get("url") or dj.get("file_url") or (dj.get("data") or {}).get("url")
        if not file_url:
            raise RuntimeError(f"{self.PROVIDER} upload returned no URL: {dj}")
        return file_url

    def _make_handle(
        self, slug: str, request_id: str, modality: str, *, pricing: dict[str, Any] | None = None
    ) -> JobHandle:
        # `pricing` captures the quote inputs (n / duration / resolution) at
        # submit time, so a resume from another process can still price the
        # job — the provider result carries none of them.
        return JobHandle(
            task_id=request_id,
            provider=self.PROVIDER,
            model=slug,
            modality=modality,
            status=JobStatus.PENDING.value,
            endpoint=self.endpoint,
            polling_url=self.result_url(request_id),
            submitted_at=time.time(),
            extra={"pricing": pricing} if pricing else {},
        )

    async def get_job_status(self, handle: JobHandle) -> str:
        self._require_key()
        async with httpx.AsyncClient(timeout=120.0) as client:
            raw = await self._fetch_once(client, handle.task_id)
        return normalize_status(self.status_of(raw))

    async def _resume(
        self,
        handle: JobHandle,
        kind: str,
        *,
        poll: bool,
        poll_interval: float | None,
        timeout: float | None,
    ) -> JobResult:
        self._require_key()
        async with httpx.AsyncClient(timeout=120.0) as client:
            if poll:
                raw = await self._poll(client, handle.task_id, interval=poll_interval or 2.0, timeout=timeout or 900.0)
            else:
                raw = await self._fetch_once(client, handle.task_id)
        status = normalize_status(self.status_of(raw))
        assets = [
            MediaAsset(kind=kind, url=u, model=handle.model, provenance={"request_id": handle.task_id})
            for u in self.extract_urls(raw)
        ]
        meta: dict[str, Any] = {"request_id": handle.task_id, "model_name": f"{self.PROVIDER}/{handle.model}"}
        if status == JobStatus.COMPLETED.value:
            # Price from the quote inputs captured at submit time (handle.extra)
            # — without this, every async job reported zero cost at every
            # stage of its lifecycle.
            pricing = (handle.extra or {}).get("pricing") or {}
            meta["cost"] = self._media_cost(
                handle.model,
                n=int(pricing.get("n") or max(len(assets), 1)),
                duration_seconds=float(pricing.get("duration_seconds") or 0.0),
                resolution=pricing.get("resolution"),
            )
        return JobResult(
            handle=handle.with_status(status),
            status=status,
            assets=assets,
            meta=meta,
            raw=raw,
        )

    def _media_cost(
        self, slug: str, *, n: int = 1, duration_seconds: float = 0.0, resolution: str | None = None
    ) -> float:
        from ..infra.media_pricing import estimate_media_cost

        return estimate_media_cost(f"{self.PROVIDER}/{slug}", n=n, duration_seconds=duration_seconds, resolution=resolution)

    def _effective_duration(self, slug: str, options: dict[str, Any]) -> float:
        """The duration the provider will actually bill for.

        An explicit ``duration`` option wins; otherwise the model's declared
        capability default — the provider runs its default clip length and
        bills for it whether or not the caller passed the option, so pricing
        with 0 was quoting $0 for real spend.
        """
        raw = options.get("duration")
        if raw:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
        try:
            from .media_capabilities import get_model_schema

            info = get_model_schema(f"{self.PROVIDER}/{slug}")
            spec = (info.inputs or {}).get("duration") if info else None
            default = getattr(spec, "default", None)
            if default:
                return float(default)
        except Exception:
            pass
        return 0.0

    async def _coerce_media_url(
        self, source: Any, *, explicit: str | None = None, filename: str = "input.bin"
    ) -> str | None:
        """Async resolve a media input to a hosted URL (see sync counterpart)."""
        if explicit:
            return explicit
        if source is None:
            return None
        url = getattr(source, "url", None)
        if url:
            return url
        if isinstance(source, str):
            if source.startswith(("http://", "https://")):
                return source
            from ..media.hosting import resolve_to_bytes

            return await self.upload_file(resolve_to_bytes(source), filename=filename)
        if isinstance(source, (bytes, bytearray)):
            return await self.upload_file(bytes(source), filename=filename)
        data = getattr(source, "data", None)
        if data:
            import base64

            return await self.upload_file(base64.b64decode(data), filename=filename)
        from pathlib import Path

        if isinstance(source, Path):
            return await self.upload_file(source.read_bytes(), filename=filename)
        raise TypeError(f"Unsupported media source: {type(source).__name__}")


class AsyncAggregatorImageDriver(AsyncAggregatorClient, AsyncImageGenDriver):
    """Generic async image generation + i2i editing over an aggregator."""

    supports_multiple = True
    supports_size_variants = True
    supports_edit = True
    max_images = 4

    def _default_model(self) -> str:
        return self.DEFAULT_IMAGE_MODEL

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

    async def _run_image(self, slug: str, payload: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            request_id, submitted = await self._submit(client, slug, payload)
            if request_id is None:
                return self._finalize_image(slug, None, submitted, options)
            if not options.get("poll", True):
                handle = self._make_handle(
                    slug, request_id, "image", pricing={"n": int(options.get("n") or 1)}
                )
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
            result = await self._poll(
                client,
                request_id,
                interval=float(options.get("poll_interval", 2)),
                timeout=float(options.get("timeout", 300)),
            )
        return self._finalize_image(slug, request_id, result, options)

    async def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self._require_key()
        slug = options.get("model", self.model)
        return await self._run_image(slug, self.build_image_payload(prompt, options), options)

    async def edit_image(
        self, image: bytes, prompt: str, options: dict[str, Any], *, mask: bytes | None = None
    ) -> dict[str, Any]:
        self._require_key()
        slug = options.get("model", self.model)
        payload = self.build_image_payload(prompt, options)
        if "image_url" not in payload:
            payload["image_url"] = await self.upload_file(image, filename="input.png")
        if mask is not None:
            payload["mask_url"] = await self.upload_file(mask, filename="mask.png")
        return await self._run_image(slug, payload, options)

    async def resume_job(
        self,
        handle: JobHandle,
        *,
        poll: bool = True,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> JobResult:
        return await self._resume(handle, "image", poll=poll, poll_interval=poll_interval, timeout=timeout)


class AsyncAggregatorVideoDriver(AsyncAggregatorClient, AsyncVideoGenDriver):
    """Generic async video generation over an aggregator."""

    supports_image_input = True
    supports_reference_images = True
    supports_video_input = True
    supports_audio = True
    supports_polling = True

    def _default_model(self) -> str:
        return self.DEFAULT_VIDEO_MODEL

    async def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self._require_key()
        slug = options.get("model", self.model)
        payload = self.build_video_payload(prompt, options)
        duration = self._effective_duration(slug, options)
        async with httpx.AsyncClient(timeout=120.0) as client:
            request_id, submitted = await self._submit(client, slug, payload)
            if request_id is not None and not options.get("poll", True):
                handle = self._make_handle(
                    slug, request_id, "video",
                    pricing={"n": 1, "duration_seconds": duration, "resolution": options.get("resolution")},
                )
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
                else await self._poll(
                    client,
                    request_id,
                    interval=float(options.get("poll_interval", 3)),
                    timeout=float(options.get("timeout", 900)),
                )
            )
        videos = [video_from_url(u) for u in self.extract_urls(result)]
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

    async def resume_job(
        self,
        handle: JobHandle,
        *,
        poll: bool = True,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> JobResult:
        return await self._resume(handle, "video", poll=poll, poll_interval=poll_interval, timeout=timeout)


class AsyncAggregatorLipsyncDriver(AsyncAggregatorClient, AsyncLipsyncDriver):
    """Generic async lipsync (image|video + audio → video) over an aggregator."""

    supports_image = True
    supports_video = True

    def _default_model(self) -> str:
        return self.DEFAULT_LIPSYNC_MODEL

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

    async def generate_lipsync(
        self,
        audio: Any,
        options: dict[str, Any],
        *,
        image: Any | None = None,
        video: Any | None = None,
    ) -> dict[str, Any]:
        self._require_key()
        slug = options.get("model", self.model)
        audio_url = await self._coerce_media_url(audio, explicit=options.get("audio_url"), filename="audio.mp3")
        image_url = await self._coerce_media_url(image, explicit=options.get("image_url"), filename="image.png")
        video_url = await self._coerce_media_url(video, explicit=options.get("video_url"), filename="video.mp4")
        category = "video" if video_url else "image"
        payload = self.build_lipsync_payload(options, audio_url=audio_url, image_url=image_url, video_url=video_url)

        duration = self._effective_duration(slug, options)
        async with httpx.AsyncClient(timeout=120.0) as client:
            request_id, submitted = await self._submit(client, slug, payload)
            if request_id is not None and not options.get("poll", True):
                handle = self._make_handle(
                    slug, request_id, "lipsync",
                    pricing={"n": 1, "duration_seconds": duration},
                )
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
                else await self._poll(
                    client,
                    request_id,
                    interval=float(options.get("poll_interval", 3)),
                    timeout=float(options.get("timeout", 900)),
                )
            )
        videos = [video_from_url(u) for u in self.extract_urls(result)]
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

    async def resume_job(
        self,
        handle: JobHandle,
        *,
        poll: bool = True,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> JobResult:
        return await self._resume(handle, "video", poll=poll, poll_interval=poll_interval, timeout=timeout)


class AsyncAggregatorMusicDriver(AsyncAggregatorClient, AsyncMusicGenDriver):
    """Generic async music generation over an aggregator."""

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

    async def generate_music(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self._require_key()
        slug = options.get("model", self.model)
        payload = self.build_music_payload(prompt, options)
        async with httpx.AsyncClient(timeout=120.0) as client:
            request_id, submitted = await self._submit(client, slug, payload)
            if request_id is not None and not options.get("poll", True):
                handle = self._make_handle(slug, request_id, "music", pricing={"n": 1})
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
                else await self._poll(
                    client,
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

    async def resume_job(
        self,
        handle: JobHandle,
        *,
        poll: bool = True,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> JobResult:
        return await self._resume(handle, "audio", poll=poll, poll_interval=poll_interval, timeout=timeout)
