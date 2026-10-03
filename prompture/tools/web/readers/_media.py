"""Optional bridge to :mod:`prompture.media.understand` transcription.

Imported lazily: when the media-understanding package (or its STT
configuration) is missing, readers simply skip the transcription step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def load_transcriber() -> Callable[..., Any] | None:
    """Return ``transcribe`` when transcription is installed *and* configured, else ``None``.

    The availability check is offline (key presence, binaries), never a
    network call.
    """
    try:
        from ....media.understand import transcribe, transcription_available
    except ImportError:
        return None
    try:
        if not transcription_available():
            return None
    except Exception:
        return None
    return transcribe


def transcription_ready() -> bool:
    return load_transcriber() is not None


def transcript_markdown(transcript: Any) -> str:
    """Markdown for a ``Transcript`` (``to_markdown()`` / ``segments`` / ``text``)."""
    to_md = getattr(transcript, "to_markdown", None)
    if callable(to_md):
        try:
            text = to_md()
            if text:
                return str(text)
        except Exception:
            pass
    segments = getattr(transcript, "segments", None) or []
    if segments:
        from .base import fmt_timestamp

        return "\n".join(
            f"[{fmt_timestamp(getattr(s, 'start', 0) or 0)}] {getattr(s, 'text', '').strip()}" for s in segments
        )
    return str(getattr(transcript, "text", "") or "")


TRANSCRIPTION_HINT = "Configure speech-to-text (e.g. GROQ_API_KEY or OPENAI_API_KEY) and install ffmpeg / yt-dlp"
