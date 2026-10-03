"""Transcript → summary with timestamped key points (chunked map-reduce).

:func:`summarize_media` accepts a :class:`~.transcribe.Transcript` or anything
:func:`~.transcribe.transcribe` accepts. The transcript is rendered as
``[HH:MM:SS] text`` lines and split into windows that fit comfortably in a
small model's context. Each window is summarized with
:func:`prompture.extraction.core.ask_for_json` against a fixed schema (map);
the partial results are then merged — repeatedly, if they are still too long —
into one summary and a ranked list of key points (reduce). A short transcript
takes a single call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from .errors import MediaTooLargeError
from .transcribe import Transcript, format_timestamp, parse_timestamp, transcribe

DEFAULT_SUMMARY_MODEL = "openai/gpt-4o-mini"
DEFAULT_MAX_CHUNK_CHARS = 12000
DEFAULT_MAX_MAP_CHUNKS = 40
DEFAULT_MAX_KEY_POINTS = 8


class _KeyPointOut(BaseModel):
    timestamp: str = Field(description="HH:MM:SS timestamp copied from the transcript line where the point is made")
    point: str = Field(description="One-sentence key point")


class _SummaryOut(BaseModel):
    summary: str = Field(description="Concise summary of the content")
    key_points: list[_KeyPointOut] = Field(default_factory=list, description="Most important points, in order")


_SCHEMA = _SummaryOut.model_json_schema()


@dataclass
class KeyPoint:
    """A key point; ``timestamp`` is seconds from the start (``None`` when unknown)."""

    timestamp: float | None
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "time": format_timestamp(self.timestamp) if self.timestamp is not None else None,
            "text": self.text,
        }


@dataclass
class MediaSummary:
    """Summary of a media source.

    Attributes:
        summary: Summary text.
        key_points: Ordered key points with timestamps.
        transcript: The transcript that was summarized.
        cost: Total USD cost (transcription + LLM calls).
        model: LLM used for summarization.
        usage: Token counts, call count, and the cost split.
    """

    summary: str
    key_points: list[KeyPoint] = field(default_factory=list)
    transcript: Transcript | None = None
    cost: float = 0.0
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self, *, include_transcript: bool = False) -> str:
        title = (self.transcript.title or self.transcript.source) if self.transcript else None
        lines = [f"# Summary: {title or 'media'}", "", self.summary.strip(), ""]
        if self.key_points:
            lines.append("## Key points")
            lines.append("")
            for kp in self.key_points:
                stamp = f"[{format_timestamp(kp.timestamp)}] " if kp.timestamp is not None else ""
                lines.append(f"- {stamp}{kp.text}")
            lines.append("")
        if include_transcript and self.transcript is not None:
            lines.extend(["## Transcript", "", self.transcript.to_markdown(header=False)])
        return "\n".join(lines).rstrip() + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "key_points": [k.to_dict() for k in self.key_points],
            "transcript": self.transcript.to_dict() if self.transcript else None,
            "cost": self.cost,
            "model": self.model,
            "usage": self.usage,
        }


def transcript_windows(transcript: Transcript, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
    """Split a transcript into ``[HH:MM:SS] text`` windows of at most ~*max_chars*."""
    if transcript.segments:
        lines = [f"[{format_timestamp(s.start)}] {s.text.strip()}" for s in transcript.segments if s.text.strip()]
    else:
        lines = [p.strip() for p in transcript.text.splitlines() if p.strip()]
    windows: list[str] = []
    buf: list[str] = []
    size = 0
    for line in lines:
        while len(line) > max_chars:
            if buf:
                windows.append("\n".join(buf))
                buf, size = [], 0
            windows.append(line[:max_chars])
            line = line[max_chars:]
        if size + len(line) + 1 > max_chars and buf:
            windows.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        windows.append("\n".join(buf))
    return windows


class _Summarizer:
    def __init__(self, model: str, driver: Any, options: dict[str, Any] | None) -> None:
        self.model = model
        self.options = dict(options or {})
        if driver is None:
            from ...drivers import get_driver_for_model

            driver = get_driver_for_model(model)
        self.driver = driver
        self.calls = 0
        self.cost = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def ask(self, prompt: str) -> _SummaryOut:
        from ...extraction.core import ask_for_json

        result = ask_for_json(
            self.driver,
            prompt,
            _SCHEMA,
            model_name=self.model,
            options=self.options,
            cache=False,
        )
        self.calls += 1
        usage = result.get("usage") or {}
        self.cost += float(usage.get("cost") or 0.0)
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)
        data = result.get("json_object") or {}
        if not isinstance(data, dict):
            data = {}
        points = []
        for kp in data.get("key_points") or []:
            if isinstance(kp, dict) and str(kp.get("point") or "").strip():
                points.append(_KeyPointOut(timestamp=str(kp.get("timestamp") or ""), point=str(kp["point"]).strip()))
        return _SummaryOut(summary=str(data.get("summary") or "").strip(), key_points=points)


def _context(transcript: Transcript, focus: str | None) -> str:
    about = transcript.title or transcript.source or "a media file"
    text = f"The transcript is from: {about}."
    if focus:
        text += f" Focus on: {focus}."
    return text


def _map_prompt(window: str, index: int, total: int, ctx: str, max_points: int) -> str:
    part = f"part {index + 1} of {total} of a transcript" if total > 1 else "a transcript"
    return (
        f"Summarize {part}. {ctx}\n"
        "Each line starts with a [HH:MM:SS] timestamp.\n"
        f"Return a concise summary (2-6 sentences) and up to {max_points} key points. "
        "For each key point copy the timestamp of the line where it is made. "
        "Use only what the transcript says.\n\n"
        f"Transcript:\n{window}"
    )


def _render_partial(index: int, part: _SummaryOut) -> str:
    lines = [f"Part {index + 1} summary: {part.summary}"]
    for kp in part.key_points:
        lines.append(f"- [{kp.timestamp}] {kp.point}")
    return "\n".join(lines)


def _reduce_prompt(block: str, ctx: str, max_points: int) -> str:
    return (
        "Below are summaries of consecutive parts of one transcript, each with timestamped key points. "
        f"{ctx}\nMerge them into a single summary (one or two paragraphs) and the {max_points} most important "
        "key points overall, in chronological order. Keep each key point's original timestamp.\n\n"
        f"{block}"
    )


def _to_key_points(points: list[_KeyPointOut], duration: float | None, limit: int) -> list[KeyPoint]:
    out: list[KeyPoint] = []
    seen: set[str] = set()
    for kp in points:
        key = kp.point.lower()
        if key in seen:
            continue
        seen.add(key)
        ts = parse_timestamp(kp.timestamp)
        if ts is not None and duration and ts > duration + 1:
            ts = None
        out.append(KeyPoint(ts, kp.point))
    out = out[:limit]
    # Chronological, with untimed points last (sorted() is stable).
    return sorted(out, key=lambda k: (k.timestamp is None, k.timestamp or 0.0))


def summarize_transcript(
    transcript: Transcript,
    *,
    model: str = DEFAULT_SUMMARY_MODEL,
    focus: str | None = None,
    max_key_points: int = DEFAULT_MAX_KEY_POINTS,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    max_map_chunks: int = DEFAULT_MAX_MAP_CHUNKS,
    driver: Any = None,
    options: dict[str, Any] | None = None,
) -> MediaSummary:
    """Map-reduce summary of an existing :class:`Transcript`."""
    windows = transcript_windows(transcript, max_chunk_chars)
    if not windows:
        return MediaSummary(
            summary="No speech was detected in this media.",
            transcript=transcript,
            cost=transcript.cost,
            model=model,
            usage={"llm_calls": 0, "llm_cost": 0.0, "transcription_cost": transcript.cost},
        )
    if len(windows) > max_map_chunks:
        raise MediaTooLargeError(
            f"transcript needs {len(windows)} summary chunks; the cap is {max_map_chunks}", limit=max_map_chunks
        )

    llm = _Summarizer(model, driver, options)
    ctx = _context(transcript, focus)
    partials = [llm.ask(_map_prompt(w, i, len(windows), ctx, max_key_points)) for i, w in enumerate(windows)]

    while len(partials) > 1:
        rendered = [_render_partial(i, p) for i, p in enumerate(partials)]
        groups: list[list[str]] = []
        size = 0
        for block in rendered:
            if not groups or size + len(block) > max_chunk_chars:
                groups.append([])
                size = 0
            groups[-1].append(block)
            size += len(block) + 2
        if len(groups) >= len(partials):
            # Partials are individually too long to pair up; merge in pairs anyway.
            groups = [rendered[i : i + 2] for i in range(0, len(rendered), 2)]
        partials = [llm.ask(_reduce_prompt("\n\n".join(g), ctx, max_key_points)) for g in groups]

    final = partials[0]
    llm_cost = round(llm.cost, 6)
    return MediaSummary(
        summary=final.summary,
        key_points=_to_key_points(final.key_points, transcript.duration_s, max_key_points),
        transcript=transcript,
        cost=round(transcript.cost + llm_cost, 6),
        model=model,
        usage={
            "llm_calls": llm.calls,
            "map_chunks": len(windows),
            "prompt_tokens": llm.prompt_tokens,
            "completion_tokens": llm.completion_tokens,
            "total_tokens": llm.total_tokens,
            "llm_cost": llm_cost,
            "transcription_cost": transcript.cost,
        },
    )


def summarize_media(
    source_or_transcript: Transcript | str | os.PathLike[str],
    *,
    model: str = DEFAULT_SUMMARY_MODEL,
    focus: str | None = None,
    transcribe_model: str = "auto",
    allow_provider_fallback: bool = False,
    language: str | None = None,
    max_key_points: int = DEFAULT_MAX_KEY_POINTS,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    max_map_chunks: int = DEFAULT_MAX_MAP_CHUNKS,
    driver: Any = None,
    options: dict[str, Any] | None = None,
    **transcribe_kwargs: Any,
) -> MediaSummary:
    """Summarize a media source (or an existing transcript) with timestamped key points.

    Args:
        source_or_transcript: A :class:`Transcript`, or a path / URL passed to
            :func:`transcribe`.
        model: ``provider/model`` LLM used for the map and reduce steps.
        focus: Optional angle for the summary ("pricing discussion").
        transcribe_model: STT model when a source is given (``"auto"``).
        allow_provider_fallback: Forwarded to :func:`transcribe`.
        language: Forwarded to :func:`transcribe`.
        max_key_points: Key points in the final result.
        max_chunk_chars: Transcript characters per map call.
        max_map_chunks: Cap on map calls (≈ cost cap).
        driver: Pre-built LLM driver (skips ``get_driver_for_model``).
        options: Extra driver options (temperature, ...).
        **transcribe_kwargs: Other :func:`transcribe` caps / timeouts.
    """
    if isinstance(source_or_transcript, Transcript):
        transcript = source_or_transcript
    else:
        transcript = transcribe(
            source_or_transcript,
            model=transcribe_model,
            allow_provider_fallback=allow_provider_fallback,
            language=language,
            **transcribe_kwargs,
        )
    return summarize_transcript(
        transcript,
        model=model,
        focus=focus,
        max_key_points=max_key_points,
        max_chunk_chars=max_chunk_chars,
        max_map_chunks=max_map_chunks,
        driver=driver,
        options=options,
    )


__all__ = [
    "DEFAULT_SUMMARY_MODEL",
    "KeyPoint",
    "MediaSummary",
    "summarize_media",
    "summarize_transcript",
    "transcript_windows",
]
