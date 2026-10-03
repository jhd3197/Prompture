"""Podcast reader: episode pages / audio enclosures → transcript.

Episodes are resolved through public data only: Apple Podcasts links via the
keyless iTunes lookup API, other pages via their advertised RSS feed or
``og:audio`` tag, feed URLs via ``episode=<title words>`` (latest otherwise),
and direct audio URLs as-is.

Chain: ``feed_transcript`` (a transcript the publisher ships with
``<podcast:transcript>``) ▸ ``transcription`` (speech-to-text of the
enclosure through :mod:`prompture.media.understand`, when configured) ▸
``show_notes`` (episode description, marked as not a transcript).
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from ....capabilities.errors import BackendUnavailableError
from .._common import RequestRejectedError, TTLCache, host_of
from ..html2md import find_feed_links, html_to_markdown, html_to_text, scan_head
from . import _media
from ._media import TRANSCRIPTION_HINT, transcript_markdown
from .base import BaseReader, ReadResult, StepBackend, clip, fmt_timestamp, http_get, http_json
from .feeds import load_feed, looks_like_feed_url, parse_feed
from .youtube import parse_vtt

AUDIO_EXTENSIONS = (".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wav", ".flac")
ITUNES_LOOKUP = "https://itunes.apple.com/lookup"
_APPLE_ID_RE = re.compile(r"/id(\d+)")

_episodes = TTLCache(ttl=300.0, maxsize=32)


def is_audio_url(url: str) -> bool:
    try:
        return urlsplit(url).path.lower().endswith(AUDIO_EXTENSIONS)
    except ValueError:
        return False


def apple_ids(url: str) -> tuple[str, str | None] | None:
    """``(podcast_id, episode_id)`` from a podcasts.apple.com URL."""
    if host_of(url) != "podcasts.apple.com":
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    m = _APPLE_ID_RE.search(parts.path)
    if not m:
        return None
    episode = (parse_qs(parts.query).get("i") or [None])[0]
    return m.group(1), episode


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _match_entry(
    entries: list[dict[str, Any]], *, link: str | None = None, title: str | None = None, audio: str | None = None
) -> dict[str, Any] | None:
    for e in entries:
        if link and e.get("link") and e["link"].rstrip("/") == link.rstrip("/"):
            return e
        if audio and any(enc.get("url", "").split("?")[0] == audio.split("?")[0] for enc in e.get("enclosures", [])):
            return e
    if title:
        wanted = _norm(title)
        for e in entries:
            have = _norm(e.get("title", ""))
            if have and (have == wanted or wanted in have or have in wanted):
                return e
    return None


def _episode_from_entry(
    entry: dict[str, Any], feed: dict[str, Any], feed_url: str | None, page_url: str
) -> dict[str, Any]:
    audio = next((enc["url"] for enc in entry.get("enclosures", []) if enc.get("url")), None)
    return {
        "title": entry.get("title") or "",
        "show": feed.get("title") or "",
        "published": entry.get("published") or "",
        "summary": entry.get("content") or entry.get("summary") or "",
        "audio_url": audio,
        "transcripts": entry.get("transcripts") or [],
        "page_url": entry.get("link") or page_url,
        "feed_url": feed_url,
        "duration": entry.get("duration"),
    }


def resolve_episode(url: str, *, session: Any = None, episode: str | None = None) -> dict[str, Any]:
    """Find the episode behind *url*: title, show, audio URL, published transcripts."""
    key = (url, episode)
    hit = _episodes.get(key)
    if hit is not None:
        return hit
    result = _resolve(url, session=session, episode=episode)
    _episodes.set(key, result)
    return result


def _resolve(url: str, *, session: Any, episode: str | None) -> dict[str, Any]:
    if is_audio_url(url):
        name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
        return {
            "title": name,
            "show": "",
            "published": "",
            "summary": "",
            "audio_url": url,
            "transcripts": [],
            "page_url": url,
            "feed_url": None,
        }

    ids = apple_ids(url)
    if ids:
        podcast_id, episode_id = ids
        data = http_json(
            ITUNES_LOOKUP,
            params={"id": podcast_id, "entity": "podcastEpisode", "limit": 200},
            session=session,
            backend="podcasts",
        )
        results = data.get("results") or []
        show = next((r for r in results if r.get("wrapperType") == "track" and r.get("kind") == "podcast"), {})
        episodes = [r for r in results if r.get("wrapperType") == "podcastEpisode"]
        chosen = None
        if episode_id:
            chosen = next((r for r in episodes if str(r.get("trackId")) == str(episode_id)), None)
        elif episode:
            wanted = _norm(episode)
            chosen = next((r for r in episodes if wanted in _norm(r.get("trackName", ""))), None)
        chosen = chosen or (episodes[0] if episodes else None)
        if chosen is None:
            raise RequestRejectedError("podcasts", "episode not found in the iTunes lookup")
        ep = {
            "title": chosen.get("trackName") or "",
            "show": chosen.get("collectionName") or show.get("collectionName") or "",
            "published": (chosen.get("releaseDate") or "")[:10],
            "summary": chosen.get("description") or chosen.get("shortDescription") or "",
            "audio_url": chosen.get("episodeUrl"),
            "transcripts": [],
            "page_url": url,
            "feed_url": show.get("feedUrl") or chosen.get("feedUrl"),
            "duration": fmt_timestamp((chosen.get("trackTimeMillis") or 0) / 1000)
            if chosen.get("trackTimeMillis")
            else None,
        }
        if ep["feed_url"]:  # the feed may carry a publisher transcript
            try:
                feed_url, data_bytes = load_feed(ep["feed_url"], session=session)
                feed = parse_feed(data_bytes)
                entry = _match_entry(feed.get("entries", []), audio=ep["audio_url"], title=ep["title"])
                if entry:
                    ep["transcripts"] = entry.get("transcripts") or []
                    ep["audio_url"] = ep["audio_url"] or next((e["url"] for e in entry.get("enclosures", [])), None)
            except Exception:
                pass
        return ep

    if looks_like_feed_url(url):
        feed_url, data_bytes = load_feed(url, session=session)
        feed = parse_feed(data_bytes)
        entries = feed.get("entries", [])
        entry = _match_entry(entries, title=episode) if episode else (entries[0] if entries else None)
        if entry is None:
            raise RequestRejectedError("podcasts", "no matching episode in the feed")
        return _episode_from_entry(entry, feed, feed_url, url)

    # Generic episode page: advertised feed, else og:audio / <audio>.
    resp = http_get(url, session=session, backend="podcasts", check_challenge=True)
    info = scan_head(resp.text, resp.url)
    page_title = info.meta.get("og:title") or info.title
    audio = (
        info.meta.get("og:audio")
        or info.meta.get("og:audio:url")
        or next((m for m in info.media if is_audio_url(m)), None)
    )
    for feed_link in find_feed_links(resp.text, resp.url):
        try:
            feed_url, data_bytes = load_feed(feed_link, session=session)
            feed = parse_feed(data_bytes)
        except Exception:
            continue
        entry = _match_entry(feed.get("entries", []), link=resp.url, audio=audio, title=episode or page_title)
        if entry:
            return _episode_from_entry(entry, feed, feed_url, url)
    if audio:
        return {
            "title": page_title,
            "show": info.meta.get("og:site_name", ""),
            "published": "",
            "summary": info.meta.get("og:description") or info.meta.get("description") or "",
            "audio_url": audio,
            "transcripts": [],
            "page_url": url,
            "feed_url": None,
        }
    raise RequestRejectedError("podcasts", "could not find an episode feed or audio on this page")


def _transcript_text(data: Any, ctype: str, url: str) -> str:
    """Convert a published transcript (VTT/SRT/JSON/HTML/text) to timestamped text."""
    path = url.lower().split("?", 1)[0]
    text = data.content.decode(data.encoding or "utf-8", errors="replace")
    if "vtt" in ctype or "srt" in ctype or path.endswith((".vtt", ".srt")):
        segments = parse_vtt(text)
        return "\n".join(f"[{fmt_timestamp(s)}] {t}" for s, t in segments)
    if "json" in ctype or path.endswith(".json"):
        obj = json.loads(text)
        segs = obj.get("segments") if isinstance(obj, dict) else obj
        lines = []
        for s in segs or []:
            body = (s.get("body") or s.get("text") or "").strip()
            if body:
                speaker = f"{s['speaker']}: " if s.get("speaker") else ""
                lines.append(f"[{fmt_timestamp(float(s.get('startTime', s.get('start', 0)) or 0))}] {speaker}{body}")
        return "\n".join(lines)
    if "html" in ctype or path.endswith((".html", ".htm")):
        return html_to_markdown(text, base_url=url, use_trafilatura=False)
    return text.strip()


class PodcastReader(BaseReader):
    """Podcast episodes (Apple Podcasts links, audio files, or forced on any episode page)."""

    name = "podcasts"
    description = "Podcast episodes → published transcript or speech-to-text"

    def can_handle(self, url: str) -> bool:
        return is_audio_url(url) or apple_ids(url) is not None

    def steps(self) -> list[StepBackend]:
        return [
            StepBackend("feed_transcript", self._via_feed_transcript),
            StepBackend(
                "transcription",
                self._via_transcription,
                available=lambda: _media.load_transcriber() is not None,
                requires=("speech-to-text provider",),
                hint=TRANSCRIPTION_HINT,
            ),
            StepBackend("show_notes", self._via_show_notes),
        ]

    def _result(self, url: str, ep: dict[str, Any], content: str, source: str | None) -> ReadResult:
        header = []
        if ep.get("show"):
            header.append(f"**Show:** {ep['show']}")
        if ep.get("published"):
            header.append(f"**Published:** {ep['published']}")
        if ep.get("duration"):
            header.append(f"**Duration:** {ep['duration']}")
        if ep.get("audio_url"):
            header.append(f"**Audio:** <{ep['audio_url']}>")
        body = ("\n".join(header) + "\n\n" if header else "") + content
        meta = {k: ep.get(k) for k in ("show", "published", "audio_url", "feed_url", "page_url", "duration")}
        meta["transcript_source"] = source
        return ReadResult(url, ep.get("title") or "Podcast episode", body, self.name, "podcast_episode", meta)

    def _via_feed_transcript(
        self, url: str, *, session: Any = None, episode: str | None = None, **_: Any
    ) -> ReadResult:
        ep = resolve_episode(url, session=session, episode=episode)
        transcripts = ep.get("transcripts") or []
        if not transcripts:
            raise RequestRejectedError("feed_transcript", "the publisher ships no transcript")
        order = ("text/vtt", "application/x-subrip", "application/srt", "application/json", "text/html", "text/plain")
        transcripts = sorted(
            transcripts, key=lambda t: order.index(t.get("type")) if t.get("type") in order else len(order)
        )
        last_error: Exception | None = None
        for t in transcripts:
            try:
                resp = http_get(t["url"], session=session, backend="podcasts")
                text = _transcript_text(resp, (t.get("type") or resp.content_type or "").lower(), t["url"])
            except Exception as exc:
                last_error = exc
                continue
            if text.strip():
                return self._result(url, ep, "## Transcript\n\n" + text, "publisher")
        raise RequestRejectedError("feed_transcript", f"transcript unreadable ({last_error})")

    def _via_transcription(self, url: str, *, session: Any = None, episode: str | None = None, **_: Any) -> ReadResult:
        transcribe = _media.load_transcriber()
        if transcribe is None:
            raise BackendUnavailableError(f"transcription is not configured — {TRANSCRIPTION_HINT}")
        ep = resolve_episode(url, session=session, episode=episode)
        if not ep.get("audio_url"):
            raise RequestRejectedError("transcription", "no audio enclosure found")
        transcript = transcribe(ep["audio_url"])
        return self._result(url, ep, "## Transcript\n\n" + transcript_markdown(transcript), "speech_to_text")

    def _via_show_notes(self, url: str, *, session: Any = None, episode: str | None = None, **_: Any) -> ReadResult:
        ep = resolve_episode(url, session=session, episode=episode)
        notes = html_to_text(ep.get("summary") or "") or "_No show notes._"
        content = (
            "_No transcript available (configure speech-to-text to transcribe the audio). Show notes follow._\n\n"
            "## Show notes\n\n" + clip(notes, 20000)
        )
        return self._result(url, ep, content, None)
