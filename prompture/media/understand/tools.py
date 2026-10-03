"""Agent tools for media understanding: ``transcribe_media`` and ``summarize_media``.

Tool functions always return a string and never raise: failures come back as
``"Error: ..."`` text (credentials scrubbed) so the model can react, e.g. by
asking the user to set an API key or install ffmpeg.
"""

from __future__ import annotations

from ...agents.tools_schema import ToolDefinition, tool_from_function
from ...security.redaction import scrub_secrets

#: Characters of transcript returned to the model before truncating.
DEFAULT_TOOL_MAX_CHARS = 20000


def _error(exc: BaseException) -> str:
    hint = getattr(exc, "hint", None)
    msg = f"Error: {type(exc).__name__}: {exc}"
    if hint and str(hint) not in msg:
        msg += f" (fix: {hint})"
    return scrub_secrets(msg)[:2000]


def _truncate(text: str, max_chars: int) -> str:
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip() + f"\n\n[transcript truncated at {max_chars} characters of {len(text)}]"
    return text


def transcribe_media(
    source: str,
    language: str = "",
    timestamps: bool = True,
    max_chars: int = DEFAULT_TOOL_MAX_CHARS,
) -> str:
    """Transcribe a video, podcast or audio file (URL or local path) into timestamped text.

    Works with direct audio/video links, local files, and video/podcast pages
    such as YouTube when yt-dlp is installed. Long media is split and stitched
    automatically.

    Args:
        source: URL or local file path of the media.
        language: Optional ISO language code hint such as "en" or "es".
        timestamps: Prefix each line with its [HH:MM:SS] timestamp.
        max_chars: Maximum characters of transcript to return.

    Returns:
        The transcript as Markdown, or a line starting with "Error:".
    """
    from .transcribe import transcribe

    try:
        transcript = transcribe(source, language=language or None)
        return _truncate(transcript.to_markdown(timestamps=timestamps), max_chars)
    except Exception as exc:
        return _error(exc)


def _summarize(source: str, focus: str, model: str | None) -> str:
    from .summarize import DEFAULT_SUMMARY_MODEL, summarize_media

    try:
        result = summarize_media(source, model=model or DEFAULT_SUMMARY_MODEL, focus=focus or None)
        return result.to_markdown()
    except Exception as exc:
        return _error(exc)


def summarize_media(source: str, focus: str = "") -> str:
    """Summarize a video, podcast or audio file (URL or local path) with timestamped key points.

    Transcribes the media first, then summarizes the transcript.

    Args:
        source: URL or local file path of the media.
        focus: Optional topic to focus the summary on.

    Returns:
        A Markdown summary with timestamped key points, or a line starting with "Error:".
    """
    return _summarize(source, focus, None)


def transcribe_media_tool() -> ToolDefinition:
    """``transcribe_media`` as a :class:`ToolDefinition`."""
    return tool_from_function(transcribe_media, name="transcribe_media", metadata={"category": "media"})


def summarize_media_tool(model: str | None = None) -> ToolDefinition:
    """``summarize_media`` as a :class:`ToolDefinition`, optionally pinned to an LLM *model*."""

    def summarize_media(source: str, focus: str = "") -> str:
        return _summarize(source, focus, model)

    summarize_media.__doc__ = globals()["summarize_media"].__doc__
    return tool_from_function(summarize_media, name="summarize_media", metadata={"category": "media"})


def media_understanding_tools(model: str | None = None) -> list[ToolDefinition]:
    """Both tools, for ``ToolRegistry`` / ``tools=[...]``."""
    return [transcribe_media_tool(), summarize_media_tool(model)]


__all__ = [
    "media_understanding_tools",
    "summarize_media",
    "summarize_media_tool",
    "transcribe_media",
    "transcribe_media_tool",
]
