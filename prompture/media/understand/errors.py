"""Typed errors for media understanding (download, ffmpeg pipeline, transcription).

All messages go through :func:`prompture.security.scrub_secrets` via
:class:`~prompture.capabilities.errors.CapabilityError`, so they are safe to
show to a model or a log.
"""

from __future__ import annotations

from ...capabilities.errors import BackendUnavailableError, CapabilityError


class MediaUnderstandingError(CapabilityError):
    """Base class for media-understanding failures."""


class MediaToolMissingError(MediaUnderstandingError):
    """A required binary (``ffmpeg``, ``ffprobe``, ``yt-dlp``) is missing or broken.

    ``status`` is the probe status (``missing``/``broken``/``timeout``/``error``)
    and ``hint`` the exact fix.
    """

    def __init__(self, tool: str, status: str, hint: str | None = None) -> None:
        msg = f"{tool} is {status}"
        if hint:
            msg += f": {hint}"
        super().__init__(msg)
        self.tool = tool
        self.status = status
        self.hint = hint


class MediaTooLargeError(MediaUnderstandingError):
    """A hard cap (source bytes, duration, chunk count, chunk bytes) was exceeded."""

    def __init__(self, message: str, *, limit: int | float | None = None) -> None:
        super().__init__(f"media_too_large: {message}")
        self.limit = limit


class MediaProcessingError(MediaUnderstandingError):
    """An external command (ffmpeg, ffprobe, yt-dlp) failed or timed out."""


class MediaSourceError(MediaUnderstandingError):
    """The source could not be resolved to audio (missing file, unsupported page, ...)."""


class TranscriptionUnavailableError(BackendUnavailableError):
    """No speech-to-text provider is configured."""


class TranscriptionError(MediaUnderstandingError):
    """Every allowed STT provider failed.

    ``attempts`` has the shape of :attr:`prompture.capabilities.ChainResult.route`
    attempts.
    """

    def __init__(self, message: str, *, attempts: list[dict] | None = None) -> None:
        super().__init__(message)
        self.attempts = list(attempts or [])
