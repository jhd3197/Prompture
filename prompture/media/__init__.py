"""Media handling: images, audio, and video, plus media understanding (transcripts, summaries).

The media-understanding names (``transcribe``, ``summarize_media``, ...) load
lazily from :mod:`prompture.media.understand`: the driver package imports
``prompture.media.image`` while it is still initializing, and understanding
depends on the drivers.
"""

from typing import Any

from .audio import (
    AudioContent,
    AudioInput,
    audio_from_base64,
    audio_from_bytes,
    audio_from_file,
    audio_from_url,
    make_audio,
)
from .hosting import (
    FileHost,
    InMemoryHost,
    LocalDiskHost,
    S3PresignedHost,
    content_hash,
    default_host,
    host_media,
    resolve_to_bytes,
    save_media,
)
from .image import (
    ImageContent,
    ImageInput,
    image_from_base64,
    image_from_bytes,
    image_from_file,
    image_from_url,
    make_image,
)
from .video import (
    VideoContent,
    VideoInput,
    make_video,
    video_from_base64,
    video_from_bytes,
    video_from_file,
    video_from_url,
)

__all__ = [
    "AudioContent",
    "AudioInput",
    "FileHost",
    "ImageContent",
    "ImageInput",
    "InMemoryHost",
    "KeyPoint",
    "LocalDiskHost",
    "MediaSummary",
    "S3PresignedHost",
    "Transcript",
    "TranscriptSegment",
    "VideoContent",
    "VideoInput",
    "audio_from_base64",
    "audio_from_bytes",
    "audio_from_file",
    "audio_from_url",
    "content_hash",
    "default_host",
    "host_media",
    "image_from_base64",
    "image_from_bytes",
    "image_from_file",
    "image_from_url",
    "make_audio",
    "make_image",
    "make_video",
    "resolve_to_bytes",
    "save_media",
    "summarize_media",
    "transcribe",
    "transcription_available",
    "video_from_base64",
    "video_from_bytes",
    "video_from_file",
    "video_from_url",
]

_UNDERSTAND_EXPORTS = frozenset(
    {
        "KeyPoint",
        "MediaSummary",
        "Transcript",
        "TranscriptSegment",
        "summarize_media",
        "transcribe",
        "transcription_available",
    }
)


def __getattr__(name: str) -> Any:
    if name in _UNDERSTAND_EXPORTS:
        from . import understand

        return getattr(understand, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
