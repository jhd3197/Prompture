"""Image handling utilities for vision-capable LLM drivers."""

from __future__ import annotations

import base64
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class ImageContent:
    """Normalized image representation for vision-capable drivers.

    Attributes:
        data: Base64-encoded image data.
        media_type: MIME type (e.g. ``"image/png"``, ``"image/jpeg"``).
        source_type: How the image is delivered — ``"base64"`` or ``"url"``.
        url: Original URL when ``source_type`` is ``"url"``.
    """

    data: str
    media_type: str
    source_type: str = "base64"
    url: str | None = None


# Public type alias accepted by all image-aware APIs.
ImageInput = Union[bytes, str, Path, ImageContent]

# Known data-URI prefix pattern
_DATA_URI_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)

# Base64 detection heuristic — must look like pure base64 of reasonable length
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/\n\r]+=*$")

_MIME_FROM_EXT: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}

_MAGIC_BYTES: list[tuple[bytes, str]] = [
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # WebP starts with RIFF...WEBP
    (b"BM", "image/bmp"),
]


def _guess_media_type_from_bytes(data: bytes) -> str:
    """Guess MIME type from the first few bytes of image data."""
    for magic, mime in _MAGIC_BYTES:
        if data[: len(magic)] == magic:
            return mime
    return "image/png"  # safe fallback


def _guess_media_type(path: str) -> str:
    """Guess MIME type from a file path or URL."""
    # Strip query strings for URLs
    clean = path.split("?")[0].split("#")[0]
    ext = Path(clean).suffix.lower()
    if ext in _MIME_FROM_EXT:
        return _MIME_FROM_EXT[ext]
    guessed = mimetypes.guess_type(clean)[0]
    return guessed or "image/png"


# ------------------------------------------------------------------
# Constructor functions
# ------------------------------------------------------------------


def image_from_bytes(data: bytes, media_type: str | None = None) -> ImageContent:
    """Create an :class:`ImageContent` from raw bytes.

    Args:
        data: Raw image bytes.
        media_type: MIME type.  Auto-detected from magic bytes when *None*.
    """
    if not data:
        raise ValueError("Image data cannot be empty")
    b64 = base64.b64encode(data).decode("ascii")
    mt = media_type or _guess_media_type_from_bytes(data)
    return ImageContent(data=b64, media_type=mt)


def image_from_base64(b64: str, media_type: str = "image/png") -> ImageContent:
    """Create an :class:`ImageContent` from a base64-encoded string.

    Accepts both raw base64 and ``data:`` URIs.
    """
    m = _DATA_URI_RE.match(b64)
    if m:
        return ImageContent(data=m.group(2), media_type=m.group(1))
    return ImageContent(data=b64, media_type=media_type)


def image_from_file(path: str | Path, media_type: str | None = None) -> ImageContent:
    """Create an :class:`ImageContent` by reading a local file.

    Args:
        path: Path to an image file.
        media_type: MIME type.  Guessed from extension when *None*.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")
    raw = p.read_bytes()
    mt = media_type or _guess_media_type(str(p))
    return image_from_bytes(raw, mt)


def image_from_url(url: str, media_type: str | None = None) -> ImageContent:
    """Create an :class:`ImageContent` referencing a remote URL.

    The image is **not** downloaded — the URL is stored directly so
    drivers that accept URL-based images can pass it through.  For
    drivers that require base64, the URL is embedded as a data URI.

    Args:
        url: Publicly-accessible image URL.
        media_type: MIME type.  Guessed from the URL when *None*.
    """
    mt = media_type or _guess_media_type(url)
    return ImageContent(data="", media_type=mt, source_type="url", url=url)


# ------------------------------------------------------------------
# Smart constructor
# ------------------------------------------------------------------


def make_image(source: ImageInput) -> ImageContent:
    """Auto-detect the source type and return an :class:`ImageContent`.

    Accepts:
    - ``ImageContent`` — returned as-is.
    - ``bytes`` — base64-encoded with auto-detected MIME.
    - ``str`` — tries (in order): data URI, URL, file path, raw base64.
    - ``pathlib.Path`` — read from disk.
    """
    if isinstance(source, ImageContent):
        return source

    if isinstance(source, bytes):
        return image_from_bytes(source)

    if isinstance(source, Path):
        return image_from_file(source)

    if isinstance(source, str):
        # 1. data URI
        if source.startswith("data:"):
            return image_from_base64(source)

        # 2. URL
        if source.startswith(("http://", "https://")):
            return image_from_url(source)

        # 3. File path (if exists on disk)
        p = Path(source)
        if p.exists():
            return image_from_file(p)

        # 4. Assume raw base64
        return image_from_base64(source)

    raise TypeError(f"Unsupported image source type: {type(source).__name__}")
