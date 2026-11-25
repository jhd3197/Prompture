"""Media understanding: URL / file → transcript → summary.

Quick start::

    from prompture.media.understand import transcribe, summarize_media

    t = transcribe("https://example.com/episode.mp3")
    print(t.to_markdown())

    s = summarize_media(t, model="openai/gpt-4o-mini")
    for kp in s.key_points:
        print(kp.timestamp, kp.text)

* :func:`transcribe` — local files, direct media URLs, and video/podcast pages
  (via a healthy ``yt-dlp``); ffmpeg downmix + 10-minute chunking; stitched
  timestamps; hard size / duration caps; provider ``auto`` = Groq → OpenAI →
  ElevenLabs, with a second provider only on ``allow_provider_fallback=True``.
* :func:`contact_sheet` / :func:`detect_shots` / :func:`video_clip` — let a
  vision model "watch" a video: labelled frame grids, camera cuts, small clips
  (see :mod:`.frames`).
* :func:`summarize_media` — chunked map-reduce summary with timestamped key points.
* :func:`transcribe_media_tool` / :func:`summarize_media_tool` — agent tools.
* ``prompture.media.understand.health`` registers ``transcription`` and
  ``media_download`` rows for ``prompture doctor``.
"""

from __future__ import annotations

from .errors import (
    MediaProcessingError,
    MediaSourceError,
    MediaTooLargeError,
    MediaToolMissingError,
    MediaUnderstandingError,
    TranscriptionError,
    TranscriptionUnavailableError,
)
from .frames import (
    ContactSheet,
    Shot,
    VideoInfo,
    contact_sheet,
    detect_shots,
    extract_frame,
    extract_frames,
    probe_video,
    sample_times,
    snap_to_shots,
    video_clip,
)
from .groq_stt import GroqSTTDriver, ensure_groq_stt_registered
from .summarize import KeyPoint, MediaSummary, summarize_media, summarize_transcript
from .tools import (
    media_understanding_tools,
    summarize_media_tool,
    transcribe_media_tool,
    video_understanding_tools,
)
from .transcribe import (
    STT_PROVIDERS,
    STTBackend,
    Transcript,
    TranscriptSegment,
    active_stt_provider,
    format_timestamp,
    plan_stt_backends,
    transcribe,
    transcription_available,
)

__all__ = [
    "STT_PROVIDERS",
    "ContactSheet",
    "GroqSTTDriver",
    "KeyPoint",
    "MediaProcessingError",
    "MediaSourceError",
    "MediaSummary",
    "MediaTooLargeError",
    "MediaToolMissingError",
    "MediaUnderstandingError",
    "STTBackend",
    "Shot",
    "Transcript",
    "TranscriptSegment",
    "TranscriptionError",
    "TranscriptionUnavailableError",
    "VideoInfo",
    "active_stt_provider",
    "contact_sheet",
    "detect_shots",
    "ensure_groq_stt_registered",
    "extract_frame",
    "extract_frames",
    "format_timestamp",
    "media_understanding_tools",
    "plan_stt_backends",
    "probe_video",
    "sample_times",
    "snap_to_shots",
    "summarize_media",
    "summarize_media_tool",
    "summarize_transcript",
    "transcribe",
    "transcribe_media_tool",
    "transcription_available",
    "video_clip",
    "video_understanding_tools",
]
