"""Audio handling utilities for STT/TTS-capable drivers."""

from __future__ import annotations

import base64
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class AudioContent:
    """Normalized audio representation for STT/TTS drivers.

    Attributes:
        data: Base64-encoded audio data.
        media_type: MIME type (e.g. ``"audio/mpeg"``, ``"audio/wav"``).
        source_type: How the audio is delivered — ``"base64"`` or ``"url"``.
        url: Original URL when ``source_type`` is ``"url"``.
        duration_seconds: Duration in seconds, if known.
    """

    data: str
    media_type: str
    source_type: str = "base64"
    url: str | None = None
    duration_seconds: float | None = None


# Public type alias accepted by all audio-aware APIs.
AudioInput = Union[bytes, str, Path, AudioContent]

# Known data-URI prefix pattern
_DATA_URI_RE = re.compile(r"^data:(audio/[a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)

# Base64 detection heuristic — must look like pure base64 of reasonable length
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/\n\r]+=*$")

_MIME_FROM_EXT: dict[str, str] = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".opus": "audio/opus",
    ".webm": "audio/webm",
    ".wma": "audio/x-ms-wma",
    ".pcm": "audio/pcm",
}

_MAGIC_BYTES: list[tuple[bytes, str]] = [
    (b"\xff\xfb", "audio/mpeg"),  # MP3 frame sync (MPEG1 Layer3)
    (b"\xff\xf3", "audio/mpeg"),  # MP3 frame sync (MPEG2 Layer3)
    (b"\xff\xf2", "audio/mpeg"),  # MP3 frame sync (MPEG2.5 Layer3)
    (b"ID3", "audio/mpeg"),  # MP3 with ID3v2 tag
    (b"OggS", "audio/ogg"),  # OGG container
    (b"fLaC", "audio/flac"),  # FLAC
]

# WAV and M4A need special handling (multi-offset magic)


def _guess_media_type_from_bytes(data: bytes) -> str:
    """Guess MIME type from the first few bytes of audio data."""
    for magic, mime in _MAGIC_BYTES:
        if data[: len(magic)] == magic:
            return mime

    # WAV: starts with RIFF....WAVE
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"

    # M4A/MP4: ftyp box at offset 4
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "audio/mp4"

    return "audio/mpeg"  # safe fallback


def _guess_media_type(path: str) -> str:
    """Guess MIME type from a file path or URL."""
    # Strip query strings for URLs
    clean = path.split("?")[0].split("#")[0]
    ext = Path(clean).suffix.lower()
    if ext in _MIME_FROM_EXT:
        return _MIME_FROM_EXT[ext]
    guessed = mimetypes.guess_type(clean)[0]
    if guessed and guessed.startswith("audio/"):
        return guessed
    return "audio/mpeg"


# ------------------------------------------------------------------
# Constructor functions
# ------------------------------------------------------------------


def audio_from_bytes(data: bytes, media_type: str | None = None) -> AudioContent:
    """Create an :class:`AudioContent` from raw bytes.

    Args:
        data: Raw audio bytes.
        media_type: MIME type.  Auto-detected from magic bytes when *None*.
    """
    if not data:
        raise ValueError("Audio data cannot be empty")
    b64 = base64.b64encode(data).decode("ascii")
    mt = media_type or _guess_media_type_from_bytes(data)
    return AudioContent(data=b64, media_type=mt)


def audio_from_base64(b64: str, media_type: str = "audio/mpeg") -> AudioContent:
    """Create an :class:`AudioContent` from a base64-encoded string.

    Accepts both raw base64 and ``data:`` URIs.
    """
    m = _DATA_URI_RE.match(b64)
    if m:
        return AudioContent(data=m.group(2), media_type=m.group(1))
    return AudioContent(data=b64, media_type=media_type)


def audio_from_file(path: str | Path, media_type: str | None = None) -> AudioContent:
    """Create an :class:`AudioContent` by reading a local file.

    Args:
        path: Path to an audio file.
        media_type: MIME type.  Guessed from extension when *None*.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    raw = p.read_bytes()
    mt = media_type or _guess_media_type(str(p))
    return audio_from_bytes(raw, mt)


def audio_from_url(url: str, media_type: str | None = None) -> AudioContent:
    """Create an :class:`AudioContent` referencing a remote URL.

    The audio is **not** downloaded — the URL is stored directly so
    drivers that accept URL-based audio can pass it through.

    Args:
        url: Publicly-accessible audio URL.
        media_type: MIME type.  Guessed from the URL when *None*.
    """
    mt = media_type or _guess_media_type(url)
    return AudioContent(data="", media_type=mt, source_type="url", url=url)


# ------------------------------------------------------------------
# Smart constructor
# ------------------------------------------------------------------


def make_audio(source: AudioInput) -> AudioContent:
    """Auto-detect the source type and return an :class:`AudioContent`.

    Accepts:
    - ``AudioContent`` — returned as-is.
    - ``bytes`` — base64-encoded with auto-detected MIME.
    - ``str`` — tries (in order): data URI, URL, file path, raw base64.
    - ``pathlib.Path`` — read from disk.
    """
    if isinstance(source, AudioContent):
        return source

    if isinstance(source, bytes):
        return audio_from_bytes(source)

    if isinstance(source, Path):
        return audio_from_file(source)

    if isinstance(source, str):
        # 1. data URI
        if source.startswith("data:"):
            return audio_from_base64(source)

        # 2. URL
        if source.startswith(("http://", "https://")):
            return audio_from_url(source)

        # 3. File path (if exists on disk)
        p = Path(source)
        if p.exists():
            return audio_from_file(p)

        # 4. Assume raw base64
        return audio_from_base64(source)

    raise TypeError(f"Unsupported audio source type: {type(source).__name__}")
