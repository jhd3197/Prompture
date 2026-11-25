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


def contact_sheet_media(
    path: str,
    start: float,
    end: float,
    cols: int = 3,
    rows: int = 2,
    out_dir: str = "",
) -> str:
    """Make a contact sheet: frames from a local video tiled into one labelled JPEG.

    Each tile carries its number and timestamp, so you can look at the image
    and answer with times. Use it to "watch" a stretch of video cheaply.

    Args:
        path: Local video file.
        start: Start of the range in seconds.
        end: End of the range in seconds.
        cols: Tiles per row.
        rows: Rows of tiles.
        out_dir: Folder for the JPEG (default: the system temp folder).

    Returns:
        The saved JPEG path and the time of each tile, or a line starting with "Error:".
    """
    import tempfile
    from pathlib import Path

    from .frames import contact_sheet

    try:
        sheet = contact_sheet(path, start=float(start), end=float(end), cols=int(cols), rows=int(rows))
        folder = Path(out_dir or tempfile.gettempdir())
        folder.mkdir(parents=True, exist_ok=True)
        out = sheet.save(folder / f"contact-{int(start * 1000)}-{int(end * 1000)}.jpg")
        lines = [f"Contact sheet saved: {out}", f"Tiles: {sheet.legend()}"]
        lines += [f"Note: {n}" for n in sheet.notes]
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


def detect_shots_media(path: str, start: float = 0.0, end: float = 0.0) -> str:
    """List the camera cuts (shots) of a local video, optionally inside a range.

    Args:
        path: Local video file.
        start: Start of the range in seconds (0 = beginning).
        end: End of the range in seconds (0 = the end of the video).

    Returns:
        One shot per line as "start-end" in seconds, or a line starting with "Error:".
    """
    from .frames import detect_shots

    try:
        shots = detect_shots(path, start=float(start) or None, end=float(end) or None)
        return "\n".join(f"{s.start:.2f}-{s.end:.2f}" for s in shots) or "No shots found."
    except Exception as exc:
        return _error(exc)


def video_understanding_tools() -> list[ToolDefinition]:
    """``contact_sheet`` and ``detect_shots`` as tools, for agents that watch video."""
    return [
        tool_from_function(contact_sheet_media, name="contact_sheet", metadata={"category": "media"}),
        tool_from_function(detect_shots_media, name="detect_shots", metadata={"category": "media"}),
    ]


def media_understanding_tools(model: str | None = None, *, video: bool = False) -> list[ToolDefinition]:
    """The transcription tools (plus the video tools when *video* is set)."""
    tools = [transcribe_media_tool(), summarize_media_tool(model)]
    return tools + video_understanding_tools() if video else tools


__all__ = [
    "contact_sheet_media",
    "detect_shots_media",
    "media_understanding_tools",
    "summarize_media",
    "summarize_media_tool",
    "transcribe_media",
    "transcribe_media_tool",
    "video_understanding_tools",
]
