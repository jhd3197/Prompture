"""YouTube reader: video URL → timestamped transcript.

Chain (``PROMPTURE_READER_YOUTUBE_BACKENDS`` reorders):

1. ``transcript_api`` — ``youtube-transcript-api`` (``pip install prompture[web]``).
2. ``yt_dlp`` — caption tracks listed by ``yt-dlp -J`` (manual first, then
   automatic), downloaded and parsed (json3 / WebVTT).
3. ``transcription`` — speech-to-text through :mod:`prompture.media.understand`
   when it is configured.
4. ``web_fetch`` — the watch page itself (title + description only).
"""

from __future__ import annotations

import importlib
import json
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ....capabilities.errors import BackendUnavailableError
from .. import _common
from .._common import RequestRejectedError, host_of, run_command
from ..fetch import web_fetch
from . import _media
from ._media import TRANSCRIPTION_HINT, transcript_markdown
from .base import BaseReader, ReadResult, StepBackend, fmt_timestamp, http_get, http_json

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = ("youtube.com", "youtube-nocookie.com")
DEFAULT_LANGUAGES = ("en", "en-US", "en-GB")


def youtube_video_id(url: str) -> str | None:
    """Extract the 11-character video id from watch / youtu.be / shorts / embed / live URLs."""
    host = host_of(url)
    for prefix in ("www.", "m.", "music."):
        host = host.removeprefix(prefix)
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    segments = [s for s in parts.path.split("/") if s]
    candidate: str | None = None
    if host == "youtu.be" and segments:
        candidate = segments[0]
    elif host in _YT_HOSTS:
        if segments[:1] == ["watch"] or not segments:
            candidate = (parse_qs(parts.query).get("v") or [None])[0]
        elif len(segments) >= 2 and segments[0] in ("shorts", "embed", "live", "v", "e"):
            candidate = segments[1]
    if candidate and _VIDEO_ID_RE.match(candidate):
        return candidate
    return None


def group_segments(segments: list[tuple[float, str]], *, window: float = 30.0, timestamps: bool = True) -> str:
    """Join ``(start_seconds, text)`` caption segments into timestamped paragraphs."""
    paragraphs: list[str] = []
    current: list[str] = []
    para_start: float | None = None
    last = ""
    for start, text in segments:
        text = " ".join(text.replace("\n", " ").split())
        if not text or text == last:
            continue
        last = text
        if para_start is None:
            para_start = start
        elif start - para_start >= window:
            paragraphs.append(_para(para_start, current, timestamps))
            current, para_start = [], start
        current.append(text)
    if current and para_start is not None:
        paragraphs.append(_para(para_start, current, timestamps))
    return "\n\n".join(paragraphs)


def _para(start: float, texts: list[str], timestamps: bool) -> str:
    body = " ".join(texts)
    return f"[{fmt_timestamp(start)}] {body}" if timestamps else body


# ---------------------------------------------------------------------------
# Caption formats
# ---------------------------------------------------------------------------

_VTT_TIME_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})\s+-->")
_TAG_RE = re.compile(r"<[^>]+>")


def parse_vtt(text: str) -> list[tuple[float, str]]:
    """Parse WebVTT / SRT cues into ``(start_seconds, text)``; rolling duplicates removed."""
    out: list[tuple[float, str]] = []
    start: float | None = None
    buf: list[str] = []
    seen_lines: set[str] = set()

    def flush() -> None:
        nonlocal buf
        if start is not None and buf:
            fresh = []
            for line in buf:
                if line not in seen_lines:
                    fresh.append(line)
                    seen_lines.add(line)
            if fresh:
                out.append((start, " ".join(fresh)))
        buf = []

    for raw in text.splitlines():
        line = raw.strip()
        m = _VTT_TIME_RE.match(line)
        if m:
            flush()
            h, mnt, sec, ms = m.groups()
            start = int(h or 0) * 3600 + int(mnt) * 60 + int(sec) + int(ms) / 1000
            continue
        if not line:
            flush()
            continue
        if start is None or line.isdigit() or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        cleaned = _TAG_RE.sub("", line).strip()
        if cleaned:
            buf.append(cleaned)
    flush()
    return out


def parse_json3(data: dict[str, Any]) -> list[tuple[float, str]]:
    """Parse YouTube's ``json3`` caption format."""
    out: list[tuple[float, str]] = []
    for event in data.get("events") or []:
        segs = event.get("segs") or []
        text = "".join(str(s.get("utf8", "")) for s in segs).strip()
        if text:
            out.append((float(event.get("tStartMs", 0)) / 1000.0, text))
    return out


def pick_caption_track(
    info: dict[str, Any], languages: tuple[str, ...] = DEFAULT_LANGUAGES
) -> tuple[str, str, str] | None:
    """Choose ``(url, ext, language)`` from yt-dlp ``subtitles`` / ``automatic_captions``."""
    for key in ("subtitles", "automatic_captions"):
        tracks: dict[str, list[dict[str, Any]]] = info.get(key) or {}
        if not tracks:
            continue
        langs = list(tracks)
        ordered = [lang for lang in languages if lang in tracks]
        ordered += [
            lang for lang in langs if any(lang.startswith(p.split("-")[0]) for p in languages) and lang not in ordered
        ]
        original = info.get("language")
        if original and original in tracks and original not in ordered:
            ordered.append(original)
        if key == "subtitles":
            ordered += [lang for lang in langs if lang not in ordered and lang != "live_chat"]
        for lang in ordered:
            formats = tracks.get(lang) or []
            for ext in ("json3", "vtt", "srv1", "ttml"):
                for fmt in formats:
                    if fmt.get("ext") == ext and fmt.get("url") and ext in ("json3", "vtt"):
                        return str(fmt["url"]), ext, lang
    return None


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _oembed_title(url: str, session: Any = None) -> tuple[str, str]:
    """Title and channel via the keyless oEmbed endpoint (best effort)."""
    try:
        data = http_json(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            session=session,
            backend="youtube",
            timeout=10,
        )
        return str(data.get("title") or ""), str(data.get("author_name") or "")
    except Exception:
        return "", ""


class YouTubeReader(BaseReader):
    """Transcripts for YouTube watch / youtu.be / shorts / embed / live URLs."""

    name = "youtube"
    description = "YouTube videos → timestamped transcript"

    def can_handle(self, url: str) -> bool:
        return youtube_video_id(url) is not None

    def steps(self) -> list[StepBackend]:
        return [
            StepBackend(
                "transcript_api",
                self._via_transcript_api,
                available=lambda: _common.has_module("youtube_transcript_api"),
                requires=("youtube-transcript-api",),
                hint="pip install 'prompture[web]' (youtube-transcript-api)",
            ),
            StepBackend(
                "yt_dlp",
                self._via_yt_dlp,
                available=lambda: _common.binary_ok("yt-dlp"),
                requires=("yt-dlp",),
                hint=lambda: _common.binary_hint("yt-dlp"),
            ),
            StepBackend(
                "transcription",
                self._via_transcription,
                available=lambda: _media.load_transcriber() is not None,
                requires=("speech-to-text provider",),
                hint=TRANSCRIPTION_HINT,
            ),
            StepBackend("web_fetch", self._via_web_fetch),
        ]

    # -- steps -----------------------------------------------------------

    def _result(self, url: str, vid: str, title: str, content: str, meta: dict[str, Any]) -> ReadResult:
        return ReadResult(
            url=url,
            title=title or f"YouTube video {vid}",
            content=content,
            reader=self.name,
            kind="video",
            meta={"video_id": vid, **meta},
        )

    def _via_transcript_api(
        self,
        url: str,
        *,
        languages: tuple[str, ...] | list[str] = DEFAULT_LANGUAGES,
        timestamps: bool = True,
        session: Any = None,
        **_: Any,
    ) -> ReadResult:
        vid = youtube_video_id(url) or ""
        mod = importlib.import_module("youtube_transcript_api")
        api_cls = mod.YouTubeTranscriptApi
        langs = list(languages)
        segments: list[tuple[float, str]]
        language = None
        api = api_cls()
        if hasattr(api, "fetch"):  # 1.x API
            try:
                fetched = api.fetch(vid, languages=langs)
            except Exception:
                transcripts = list(api.list(vid))
                if not transcripts:
                    raise
                fetched = transcripts[0].fetch()
            language = getattr(fetched, "language_code", None)
            segments = [(float(getattr(s, "start", 0.0)), str(getattr(s, "text", ""))) for s in fetched]
        else:  # 0.x API
            try:
                raw = api_cls.get_transcript(vid, languages=langs)
            except Exception:
                transcripts = list(api_cls.list_transcripts(vid))
                if not transcripts:
                    raise
                raw = transcripts[0].fetch()
            segments = [(float(s.get("start", 0.0)), str(s.get("text", ""))) for s in raw]
        if not segments:
            raise RequestRejectedError("transcript_api", "empty transcript")
        title, channel = _oembed_title(url, session)
        return self._result(
            url,
            vid,
            title,
            group_segments(segments, timestamps=timestamps),
            {"channel": channel or None, "language": language, "transcript_source": "captions"},
        )

    def _via_yt_dlp(
        self,
        url: str,
        *,
        languages: tuple[str, ...] | list[str] = DEFAULT_LANGUAGES,
        timestamps: bool = True,
        session: Any = None,
        **_: Any,
    ) -> ReadResult:
        vid = youtube_video_id(url) or ""
        out = run_command(
            [
                "yt-dlp",
                "-J",
                "--skip-download",
                "--no-warnings",
                "--no-playlist",
                f"https://www.youtube.com/watch?v={vid}",
            ],
            timeout=90,
        )
        try:
            info = json.loads(out)
        except ValueError as exc:
            raise RequestRejectedError("yt_dlp", "unparseable metadata") from exc
        track = pick_caption_track(info, tuple(languages))
        meta = {
            "channel": info.get("channel") or info.get("uploader"),
            "duration": info.get("duration"),
            "upload_date": info.get("upload_date"),
            "view_count": info.get("view_count"),
            "description": (info.get("description") or "")[:2000],
            "transcript_source": "captions",
        }
        if track is None:
            raise RequestRejectedError("yt_dlp", "video has no caption track")
        cap_url, ext, lang = track
        resp = http_get(cap_url, session=session, backend="yt_dlp", timeout=30)
        segments = parse_json3(resp.json()) if ext == "json3" else parse_vtt(resp.text)
        if not segments:
            raise RequestRejectedError("yt_dlp", "empty caption track")
        meta["language"] = lang
        return self._result(
            url, vid, str(info.get("title") or ""), group_segments(segments, timestamps=timestamps), meta
        )

    def _via_transcription(self, url: str, *, session: Any = None, **_: Any) -> ReadResult:
        transcribe = _media.load_transcriber()
        if transcribe is None:
            raise BackendUnavailableError(f"transcription is not configured — {TRANSCRIPTION_HINT}")
        vid = youtube_video_id(url) or ""
        transcript = transcribe(f"https://www.youtube.com/watch?v={vid}")
        title, channel = _oembed_title(url, session)
        return self._result(
            url,
            vid,
            title,
            transcript_markdown(transcript),
            {"channel": channel or None, "transcript_source": "speech_to_text"},
        )

    def _via_web_fetch(self, url: str, *, session: Any = None, **_: Any) -> ReadResult:
        vid = youtube_video_id(url) or ""
        fr = web_fetch(f"https://www.youtube.com/watch?v={vid}", max_chars=0, session=session)
        note = (
            "_No transcript was available (install `prompture[web]` or yt-dlp, or configure "
            "speech-to-text). Page content follows._\n\n"
        )
        return self._result(
            url, vid, fr.title, note + fr.content, {"transcript_source": None, "page_served_by": fr.served_by}
        )
