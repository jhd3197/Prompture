"""Video handling utilities for video generation drivers."""

from __future__ import annotations

import base64
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class VideoContent:
    """Normalized video representation for video generation drivers.

    Attributes:
        data: Base64-encoded video data.
        media_type: MIME type (e.g. ``"video/mp4"``, ``"video/webm"``).
        source_type: How the video is delivered, either ``"base64"`` or ``"url"``.
        url: Original URL when ``source_type`` is ``"url"``.
        duration_seconds: Duration in seconds, if known.
        width: Video width in pixels, if known.
        height: Video height in pixels, if known.
    """

    data: str
    media_type: str
    source_type: str = "base64"
    url: str | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None


# Public type alias accepted by video-aware APIs.
VideoInput = Union[bytes, str, Path, VideoContent]

_DATA_URI_RE = re.compile(r"^data:(video/[a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)

_MIME_FROM_EXT: dict[str, str] = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
}

_MAGIC_BYTES: list[tuple[bytes, str]] = [
    (b"\x1aE\xdf\xa3", "video/webm"),  # EBML: WebM / Matroska
]


def _guess_media_type_from_bytes(data: bytes) -> str:
    """Guess MIME type from the first few bytes of video data."""
    for magic, mime in _MAGIC_BYTES:
        if data[: len(magic)] == magic:
            return mime

    # MP4/MOV: ftyp box at byte offset 4.
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in {b"qt  "}:
            return "video/quicktime"
        return "video/mp4"

    return "video/mp4"


def _guess_media_type(path: str) -> str:
    """Guess MIME type from a file path or URL."""
    clean = path.split("?")[0].split("#")[0]
    ext = Path(clean).suffix.lower()
    if ext in _MIME_FROM_EXT:
        return _MIME_FROM_EXT[ext]
    guessed = mimetypes.guess_type(clean)[0]
    if guessed and guessed.startswith("video/"):
        return guessed
    return "video/mp4"


def video_from_bytes(
    data: bytes,
    media_type: str | None = None,
    *,
    duration_seconds: float | None = None,
    width: int | None = None,
    height: int | None = None,
) -> VideoContent:
    """Create a :class:`VideoContent` from raw bytes."""
    if not data:
        raise ValueError("Video data cannot be empty")
    b64 = base64.b64encode(data).decode("ascii")
    mt = media_type or _guess_media_type_from_bytes(data)
    return VideoContent(data=b64, media_type=mt, duration_seconds=duration_seconds, width=width, height=height)


def video_from_base64(
    b64: str,
    media_type: str = "video/mp4",
    *,
    duration_seconds: float | None = None,
    width: int | None = None,
    height: int | None = None,
) -> VideoContent:
    """Create a :class:`VideoContent` from a base64-encoded string or data URI."""
    m = _DATA_URI_RE.match(b64)
    if m:
        return VideoContent(
            data=m.group(2),
            media_type=m.group(1),
            duration_seconds=duration_seconds,
            width=width,
            height=height,
        )
    return VideoContent(data=b64, media_type=media_type, duration_seconds=duration_seconds, width=width, height=height)


def video_from_file(
    path: str | Path,
    media_type: str | None = None,
    *,
    duration_seconds: float | None = None,
    width: int | None = None,
    height: int | None = None,
) -> VideoContent:
    """Create a :class:`VideoContent` by reading a local file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {p}")
    raw = p.read_bytes()
    mt = media_type or _guess_media_type(str(p))
    return video_from_bytes(raw, mt, duration_seconds=duration_seconds, width=width, height=height)


def video_from_url(
    url: str,
    media_type: str | None = None,
    *,
    duration_seconds: float | None = None,
    width: int | None = None,
    height: int | None = None,
) -> VideoContent:
    """Create a :class:`VideoContent` referencing a remote URL."""
    mt = media_type or _guess_media_type(url)
    return VideoContent(
        data="",
        media_type=mt,
        source_type="url",
        url=url,
        duration_seconds=duration_seconds,
        width=width,
        height=height,
    )


def make_video(source: VideoInput) -> VideoContent:
    """Auto-detect the source type and return a :class:`VideoContent`.

    Accepts:
    - ``VideoContent``: returned as-is.
    - ``bytes``: base64-encoded with auto-detected MIME.
    - ``str``: tries (in order): data URI, URL, file path, raw base64.
    - ``pathlib.Path``: read from disk.
    """
    if isinstance(source, VideoContent):
        return source

    if isinstance(source, bytes):
        return video_from_bytes(source)

    if isinstance(source, Path):
        return video_from_file(source)

    if isinstance(source, str):
        if source.startswith("data:"):
            return video_from_base64(source)

        if source.startswith(("http://", "https://")):
            return video_from_url(source)

        p = Path(source)
        if p.exists():
            return video_from_file(p)

        return video_from_base64(source)

    raise TypeError(f"Unsupported video source type: {type(source).__name__}")
