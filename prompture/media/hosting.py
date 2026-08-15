"""File hosting & resolution for media inputs and outputs.

Conditioned generation (image-to-image, image-to-video, video-to-video,
lipsync, speech-to-text) requires inputs delivered as a public **URL** or raw
**bytes**, and outputs frequently need to be persisted. Prompture's media
helpers (:func:`make_image` / :func:`make_video` / :func:`make_audio`) only
*ingest* references — they never upload or save. This module fills that gap:

- :class:`FileHost` — a tiny protocol: ``put(bytes) -> url``.
- :class:`LocalDiskHost` / :class:`InMemoryHost` / :class:`S3PresignedHost` —
  built-in hosts (disk for desktop, memory for tests, S3 for servers).
- :func:`save_media` / :func:`host_media` — convenience wrappers.
- :func:`resolve_to_bytes` — collapse any media reference (bytes / data-URI /
  http(s) URL / ``memory://`` / file path / base64) down to raw bytes, e.g. to
  feed byte-only providers.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import re
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "FileHost",
    "InMemoryHost",
    "LocalDiskHost",
    "S3PresignedHost",
    "content_hash",
    "default_host",
    "host_media",
    "resolve_to_bytes",
    "save_media",
]

_DATA_URI_RE = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.DOTALL)

_EXT_FROM_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/mp4": ".m4a",
}


def content_hash(data: bytes) -> str:
    """Stable short content id (sha256, first 16 hex chars)."""
    return hashlib.sha256(data).hexdigest()[:16]


def _ext_for(media_type: str | None, filename: str | None) -> str:
    if filename and Path(filename).suffix:
        return Path(filename).suffix
    if media_type:
        ext = _EXT_FROM_MIME.get(media_type) or mimetypes.guess_extension(media_type)
        if ext:
            return ext
    return ".bin"


@runtime_checkable
class FileHost(Protocol):
    """Anything that can take bytes and return a retrievable URL."""

    def put(self, data: bytes, *, media_type: str | None = None, filename: str | None = None) -> str: ...


class LocalDiskHost:
    """Write bytes to a directory; return a ``file://`` (or ``base_url``) URL.

    Suitable for the desktop app, where a local path is enough, or when paired
    with a static file server exposing ``root`` at ``base_url``.
    """

    def __init__(self, root: str | Path, base_url: str | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/") if base_url else None

    def put(self, data: bytes, *, media_type: str | None = None, filename: str | None = None) -> str:
        name = filename or (content_hash(data) + _ext_for(media_type, filename))
        path = self.root / name
        path.write_bytes(data)
        if self.base_url:
            return f"{self.base_url}/{name}"
        return path.resolve().as_uri()


class InMemoryHost:
    """Keep bytes in-process behind ``memory://<id>`` URLs. Ideal for tests."""

    _SCHEME = "memory://"

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def put(self, data: bytes, *, media_type: str | None = None, filename: str | None = None) -> str:
        key = content_hash(data)
        with self._lock:
            self._store[key] = data
        return f"{self._SCHEME}{key}"

    def get(self, url: str) -> bytes:
        key = url[len(self._SCHEME) :] if url.startswith(self._SCHEME) else url
        with self._lock:
            if key not in self._store:
                raise KeyError(f"No in-memory object for {url!r}")
            return self._store[key]


class S3PresignedHost:
    """Upload to S3 (or any S3-compatible store) and return a presigned URL.

    ``boto3`` is imported lazily so it stays an optional dependency.
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "prompture/",
        expires_in: int = 3600,
        client: object | None = None,
        **boto3_kwargs: object,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.expires_in = expires_in
        self._client = client
        self._boto3_kwargs = boto3_kwargs

    def _ensure_client(self) -> object:
        if self._client is None:
            try:
                import boto3  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dep
                raise RuntimeError("S3PresignedHost requires boto3 (pip install boto3)") from exc
            self._client = boto3.client("s3", **self._boto3_kwargs)
        return self._client

    def put(self, data: bytes, *, media_type: str | None = None, filename: str | None = None) -> str:
        client = self._ensure_client()
        key = self.prefix + (filename or (content_hash(data) + _ext_for(media_type, filename)))
        extra = {"ContentType": media_type} if media_type else {}
        client.put_object(Bucket=self.bucket, Key=key, Body=data, **extra)  # type: ignore[attr-defined]
        return client.generate_presigned_url(  # type: ignore[attr-defined]
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.expires_in,
        )


_DEFAULT_HOST: FileHost | None = None
_DEFAULT_LOCK = threading.Lock()


def default_host() -> FileHost:
    """Process-wide default host (lazy ``LocalDiskHost`` under the temp dir)."""
    global _DEFAULT_HOST
    if _DEFAULT_HOST is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_HOST is None:
                import tempfile

                _DEFAULT_HOST = LocalDiskHost(Path(tempfile.gettempdir()) / "prompture_media")
    return _DEFAULT_HOST


def save_media(data: bytes, path: str | Path, *, makedirs: bool = True) -> Path:
    """Write *data* to *path* and return the resolved path."""
    p = Path(path)
    if makedirs:
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p.resolve()


def host_media(
    data: bytes,
    *,
    host: FileHost | None = None,
    media_type: str | None = None,
    filename: str | None = None,
) -> str:
    """Upload *data* via *host* (or the default host) and return its URL."""
    return (host or default_host()).put(data, media_type=media_type, filename=filename)


def resolve_to_bytes(
    source: bytes | str | Path,
    *,
    host: FileHost | None = None,
    timeout: float = 30.0,
) -> bytes:
    """Collapse any media reference down to raw bytes.

    Handles: ``bytes`` · ``data:`` URIs · ``http(s)`` URLs (downloaded) ·
    ``memory://`` (resolved against *host* when it is an :class:`InMemoryHost`) ·
    ``file://`` and local paths · raw base64 strings.
    """
    if isinstance(source, bytes):
        return source
    if isinstance(source, Path):
        return source.read_bytes()
    if not isinstance(source, str):
        raise TypeError(f"Unsupported source type: {type(source).__name__}")

    if source.startswith("data:"):
        m = _DATA_URI_RE.match(source)
        if not m:
            raise ValueError("Malformed data URI")
        payload = m.group(3)
        return base64.b64decode(payload) if m.group(2) else payload.encode("utf-8")

    if source.startswith("memory://"):
        if isinstance(host, InMemoryHost):
            return host.get(source)
        raise ValueError("memory:// reference requires the originating InMemoryHost")

    if source.startswith(("http://", "https://")):
        import httpx

        resp = httpx.get(source, timeout=timeout, follow_redirects=True)
        if resp.status_code >= 400:
            raise RuntimeError(f"Failed to download {source}: {resp.status_code}")
        return resp.content

    if source.startswith("file://"):
        from urllib.parse import urlparse
        from urllib.request import url2pathname

        return Path(url2pathname(urlparse(source).path)).read_bytes()

    p = Path(source)
    if p.exists():
        return p.read_bytes()

    # Last resort: treat as raw base64.
    try:
        return base64.b64decode(source, validate=True)
    except Exception as exc:
        raise ValueError(f"Could not resolve media source to bytes: {source[:64]!r}") from exc
