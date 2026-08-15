"""Long-running media job abstraction: serializable handles, results, and a
public *resume* API so callers can re-attach to in-flight generation jobs by id.

Generating images / videos / audio via aggregators (and many native providers)
is a *submit → poll → fetch* lifecycle. The drivers already submit and, when
``poll=True``, block until the job finishes. But a studio or job-queue needs to
persist the job id and reconnect later — possibly from a *different process*.

This module provides the transport-neutral data types
(:class:`JobHandle`, :class:`JobResult`, :class:`MediaAsset`) plus
:func:`resume` / :func:`get_status` / :func:`fetch_result`, which rehydrate a
stateless driver from settings (via the modality registries) and continue the
job.

A driver participates in this API by implementing::

    def resume_job(self, handle: JobHandle, *, poll: bool = True,
                   poll_interval: float | None = None,
                   timeout: float | None = None) -> JobResult: ...

    def get_job_status(self, handle: JobHandle) -> str: ...  # optional

Example::

    res = driver.generate_image("a cat", {"poll": False})
    handle = JobHandle.from_meta(res["meta"])      # persist handle.to_json()
    ...                                            # later, maybe new process
    result = prompture.jobs.resume(handle)         # blocks until done
    url = result.first_url
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "JobError",
    "JobHandle",
    "JobNotResumableError",
    "JobResult",
    "JobStatus",
    "MediaAsset",
    "fetch_result",
    "get_status",
    "resume",
]


class JobStatus(str, Enum):
    """Normalized lifecycle states shared across every provider's vocabulary."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    TIMED_OUT = "timed_out"


class JobError(RuntimeError):
    """Base error for the job API."""


class JobNotResumableError(JobError):
    """Raised when a handle's driver cannot reconnect to the job."""


# ── Assets ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MediaAsset:
    """A single generated artifact, addressable across modalities.

    Exactly one of ``url`` / ``b64`` is normally populated. ``provenance`` carries
    free-form lineage (source tool, request id, seed, parent asset, …) so the
    workflow engine and agents can reference outputs by id and trace history.
    """

    kind: str  # "image" | "video" | "audio"
    url: str | None = None
    b64: str | None = None
    media_type: str | None = None
    model: str | None = None
    prompt: str | None = None
    id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MediaAsset:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_content(self) -> Any:
        """Convert to the matching media ``*Content`` (lazy import)."""
        if self.kind == "image":
            from .media.image import image_from_base64, image_from_url

            if self.url:
                return image_from_url(self.url)
            return image_from_base64(self.b64 or "", self.media_type or "image/png")
        if self.kind == "video":
            from .media.video import video_from_base64, video_from_url

            if self.url:
                return video_from_url(self.url)
            return video_from_base64(self.b64 or "", self.media_type or "video/mp4")
        if self.kind == "audio":
            from .media.audio import audio_from_base64, audio_from_url

            if self.url:
                return audio_from_url(self.url)
            return audio_from_base64(self.b64 or "", self.media_type or "audio/mpeg")
        raise ValueError(f"Unknown asset kind: {self.kind!r}")


# ── Handles ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JobHandle:
    """A serializable pointer to an in-flight (or finished) media job.

    Everything needed to reconnect from a cold start: which ``provider`` /
    ``model`` ran it, the ``modality`` (so we pick the right registry), the
    provider's ``task_id``, and optionally the ``endpoint`` / ``polling_url``.
    """

    task_id: str
    provider: str
    model: str
    modality: str  # "image" | "video" | "audio" | "lipsync"
    status: str = JobStatus.PENDING.value
    endpoint: str | None = None
    polling_url: str | None = None
    submitted_at: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobHandle:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> JobHandle:
        return cls.from_dict(json.loads(raw))

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> JobHandle:
        """Reconstruct from a driver response ``meta`` block.

        Accepts either a nested ``meta["job_handle"]`` dict or the flat fields
        (``request_id``/``model_name``) drivers already emit.
        """
        jh = meta.get("job_handle")
        if isinstance(jh, dict):
            return cls.from_dict(jh)
        raise JobError("meta does not contain a serialized job_handle")

    def with_status(self, status: str) -> JobHandle:
        return replace(self, status=status)


@dataclass(frozen=True)
class JobResult:
    """Terminal (or current) state of a job plus any produced assets."""

    handle: JobHandle
    status: str
    assets: list[MediaAsset] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None

    @property
    def done(self) -> bool:
        return self.status == JobStatus.COMPLETED.value

    @property
    def first_url(self) -> str | None:
        for a in self.assets:
            if a.url:
                return a.url
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle": self.handle.to_dict(),
            "status": self.status,
            "assets": [a.to_dict() for a in self.assets],
            "meta": self.meta,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobResult:
        return cls(
            handle=JobHandle.from_dict(data["handle"]),
            status=data["status"],
            assets=[MediaAsset.from_dict(a) for a in data.get("assets", [])],
            meta=data.get("meta", {}),
            raw=data.get("raw"),
        )


# ── Resume dispatch ────────────────────────────────────────────────────────


@runtime_checkable
class SupportsJobResume(Protocol):
    def resume_job(
        self,
        handle: JobHandle,
        *,
        poll: bool = True,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> JobResult: ...


def _coerce_handle(handle: JobHandle | dict[str, Any] | str) -> JobHandle:
    if isinstance(handle, JobHandle):
        return handle
    if isinstance(handle, str):
        return JobHandle.from_json(handle)
    if isinstance(handle, dict):
        return JobHandle.from_dict(handle)
    raise TypeError(f"Unsupported handle type: {type(handle).__name__}")


def _resolve_driver(handle: JobHandle) -> Any:
    """Rehydrate a stateless driver for *handle* from the modality registries."""
    model_str = f"{handle.provider}/{handle.model}"
    modality = handle.modality
    if modality == "image":
        from .drivers.img_gen_registry import get_img_gen_driver_for_model

        return get_img_gen_driver_for_model(model_str)
    if modality in {"video", "lipsync"}:
        from .drivers.video_gen_registry import get_video_gen_driver_for_model

        return get_video_gen_driver_for_model(model_str)
    if modality in {"music", "audio"}:
        from .drivers.music_registry import get_music_driver_for_model

        return get_music_driver_for_model(model_str)
    raise JobNotResumableError(f"Resuming '{modality}' jobs is not supported yet (handle for {model_str})")


def resume(
    handle: JobHandle | dict[str, Any] | str,
    *,
    poll: bool = True,
    poll_interval: float | None = None,
    timeout: float | None = None,
) -> JobResult:
    """Reconnect to a media job by handle and return its :class:`JobResult`.

    ``poll=True`` blocks until the job reaches a terminal state; ``poll=False``
    does a single status/result fetch and returns whatever is available now.
    """
    h = _coerce_handle(handle)
    driver = _resolve_driver(h)
    resume_job = getattr(driver, "resume_job", None)
    if not callable(resume_job):
        raise JobNotResumableError(f"Driver for {h.provider}/{h.model} ({h.modality}) does not implement resume_job()")
    return resume_job(h, poll=poll, poll_interval=poll_interval, timeout=timeout)


def fetch_result(handle: JobHandle | dict[str, Any] | str) -> JobResult:
    """One-shot fetch of the current job state (no waiting)."""
    return resume(handle, poll=False)


def get_status(handle: JobHandle | dict[str, Any] | str) -> str:
    """Return the normalized :class:`JobStatus` value for *handle*."""
    h = _coerce_handle(handle)
    driver = _resolve_driver(h)
    get_job_status = getattr(driver, "get_job_status", None)
    if callable(get_job_status):
        return get_job_status(h)
    return fetch_result(h).status
