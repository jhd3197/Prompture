"""RSS / Atom feed reader.

Handles feed URLs (``.rss``, ``.atom``, ``.xml``, ``/feed``, ``/rss``, ...)
and, when called explicitly (``read_url(url, reader="feeds")``), HTML pages
that advertise a feed with ``<link rel="alternate" type="application/rss+xml">``.

Chain: ``feedparser`` (``pip install prompture[web]``) ▸ ``stdlib``
(:mod:`xml.etree.ElementTree`; documents declaring entities are refused).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET  # nosec B405 - entity declarations are rejected before parsing
from typing import Any
from urllib.parse import urlsplit

from .. import _common
from .._common import RequestRejectedError
from ..html2md import find_feed_links, html_to_text
from .base import BaseReader, ReadResult, StepBackend, clip, http_get

_FEED_PATH_RE = re.compile(r"(\.rss|\.atom|\.rdf|\.xml|/feed|/feeds|/rss|/atom|/rss2|/feed\.json)/?$", re.IGNORECASE)
_FEED_QUERY_RE = re.compile(r"(^|&)(format|feed|type)=(rss2?|atom|feed)(&|$)", re.IGNORECASE)
_ENTITY_RE = re.compile(rb"<!ENTITY", re.IGNORECASE)


def looks_like_feed_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.path.lower().endswith("sitemap.xml"):
        return False
    return bool(_FEED_PATH_RE.search(parts.path) or _FEED_QUERY_RE.search(parts.query))


def _looks_like_html(data: bytes, content_type: str) -> bool:
    if "html" in content_type:
        return True
    head = data[:1024].lstrip().lower()
    return head.startswith((b"<!doctype html", b"<html"))


def load_feed(url: str, *, session: Any = None, timeout: float = 20.0) -> tuple[str, bytes]:
    """Fetch *url*; if it is an HTML page, follow its advertised feed. Returns ``(feed_url, bytes)``."""
    resp = http_get(url, session=session, timeout=timeout, backend="feeds", check_challenge=True)
    if _looks_like_html(resp.content, resp.content_type):
        feeds = find_feed_links(resp.text, resp.url)
        if not feeds:
            raise RequestRejectedError("feeds", "page does not advertise an RSS/Atom feed")
        resp = http_get(feeds[0], session=session, timeout=timeout, backend="feeds", check_challenge=True)
    return resp.url, resp.content


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _local(tag: Any) -> str:
    return str(tag).rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def _child(el: ET.Element, *names: str) -> ET.Element | None:
    for c in el:
        if _local(c.tag) in names:
            return c
    return None


def _text(el: ET.Element, *names: str) -> str:
    c = _child(el, *names)
    if c is None:
        return ""
    if len(c) and _local(c.tag) in ("content", "summary") and c.get("type") == "xhtml":
        return ET.tostring(c, encoding="unicode", method="html")
    return (c.text or "").strip()


def _entry(el: ET.Element, atom: bool) -> dict[str, Any]:
    link = ""
    enclosures: list[dict[str, Any]] = []
    transcripts: list[dict[str, Any]] = []
    for c in el:
        name = _local(c.tag)
        if name == "link":
            if atom:
                rel = c.get("rel", "alternate")
                if rel == "alternate" and not link:
                    link = c.get("href", "")
                elif rel == "enclosure" and c.get("href"):
                    enclosures.append({"url": c.get("href"), "type": c.get("type", ""), "length": c.get("length")})
            elif not link:
                link = (c.text or "").strip()
        elif name == "enclosure" and c.get("url"):
            enclosures.append({"url": c.get("url"), "type": c.get("type", ""), "length": c.get("length")})
        elif name == "transcript" and c.get("url"):
            transcripts.append({"url": c.get("url"), "type": c.get("type", ""), "language": c.get("language")})
    summary = _text(el, "summary", "description") or _text(el, "encoded", "content")
    content = _text(el, "encoded", "content")
    return {
        "title": html_to_text(_text(el, "title")),
        "link": link or _text(el, "guid", "id"),
        "published": _text(el, "pubdate", "published", "updated", "date"),
        "summary": html_to_text(summary) if summary else "",
        "content": html_to_text(content) if content and content != summary else "",
        "guid": _text(el, "guid", "id"),
        "duration": _text(el, "duration"),
        "enclosures": enclosures,
        "transcripts": transcripts,
    }


def parse_feed_stdlib(data: bytes) -> dict[str, Any]:
    """Parse RSS 2.0 / RSS 1.0 (RDF) / Atom with the standard library."""
    if _ENTITY_RE.search(data[:65536]):
        raise RequestRejectedError("stdlib", "feed declares XML entities")
    try:
        root = ET.fromstring(data)  # nosec B314 - entity declarations rejected above
    except ET.ParseError as exc:
        raise RequestRejectedError("stdlib", f"not a valid XML feed: {exc}") from exc
    kind = _local(root.tag)
    if kind == "feed":
        entries = [_entry(e, True) for e in root if _local(e.tag) == "entry"]
        link_el = next((c for c in root if _local(c.tag) == "link" and c.get("rel", "alternate") == "alternate"), None)
        return {
            "title": html_to_text(_text(root, "title")),
            "link": link_el.get("href", "") if link_el is not None else "",
            "description": html_to_text(_text(root, "subtitle")),
            "format": "atom",
            "entries": entries,
        }
    if kind in ("rss", "rdf"):
        channel = _child(root, "channel")
        if channel is None:
            raise RequestRejectedError("stdlib", "RSS document has no channel")
        items = [e for e in channel if _local(e.tag) == "item"] or [e for e in root if _local(e.tag) == "item"]
        return {
            "title": html_to_text(_text(channel, "title")),
            "link": _text(channel, "link"),
            "description": html_to_text(_text(channel, "description")),
            "format": "rss" if kind == "rss" else "rdf",
            "entries": [_entry(e, False) for e in items],
        }
    raise RequestRejectedError("stdlib", f"unrecognized feed root <{kind}>")


def parse_feed_feedparser(data: bytes) -> dict[str, Any]:
    """Parse with ``feedparser`` (handles malformed feeds and more dialects)."""
    import feedparser  # type: ignore[import-not-found]

    parsed = feedparser.parse(data)
    if not parsed.get("entries") and parsed.get("bozo") and not (parsed.get("feed") or {}).get("title"):
        raise RequestRejectedError("feedparser", f"not a feed ({parsed.get('bozo_exception')})")
    feed = parsed.get("feed") or {}
    entries = []
    for e in parsed.get("entries", []):
        enclosures = [
            {"url": enc.get("href") or enc.get("url"), "type": enc.get("type", ""), "length": enc.get("length")}
            for enc in e.get("enclosures", []) or []
            if enc.get("href") or enc.get("url")
        ]
        transcripts = []
        raw_t = e.get("podcast_transcript")
        for t in raw_t if isinstance(raw_t, list) else [raw_t] if raw_t else []:
            if isinstance(t, dict) and t.get("url"):
                transcripts.append({"url": t["url"], "type": t.get("type", ""), "language": t.get("language")})
        content_list = e.get("content") or []
        content = content_list[0].get("value", "") if content_list else ""
        summary = e.get("summary", "")
        entries.append(
            {
                "title": html_to_text(e.get("title", "")),
                "link": e.get("link", ""),
                "published": e.get("published") or e.get("updated") or "",
                "summary": html_to_text(summary) if summary else "",
                "content": html_to_text(content) if content and content != summary else "",
                "guid": e.get("id", ""),
                "duration": e.get("itunes_duration", ""),
                "enclosures": enclosures,
                "transcripts": transcripts,
            }
        )
    return {
        "title": html_to_text(feed.get("title", "")),
        "link": feed.get("link", ""),
        "description": html_to_text(feed.get("subtitle", "") or feed.get("description", "")),
        "format": parsed.get("version") or "feed",
        "entries": entries,
    }


def parse_feed(data: bytes) -> dict[str, Any]:
    """Parse feed bytes with ``feedparser`` when installed, else the stdlib parser."""
    if _common.has_module("feedparser"):
        try:
            return parse_feed_feedparser(data)
        except RequestRejectedError:
            raise
        except Exception:
            pass
    return parse_feed_stdlib(data)


def render_feed(feed: dict[str, Any], *, max_items: int = 20, summary_chars: int = 500) -> str:
    lines: list[str] = []
    if feed.get("description"):
        lines.append(clip(feed["description"], 600))
        lines.append("")
    entries = feed.get("entries", [])
    for e in entries[:max_items]:
        title = e.get("title") or "(untitled)"
        lines.append(f"## [{title}]({e['link']})" if e.get("link") else f"## {title}")
        details = [d for d in (e.get("published"), e.get("duration") and f"duration {e['duration']}") if d]
        if details:
            lines.append("_" + " · ".join(details) + "_")
        if e.get("summary"):
            lines.append("")
            lines.append(clip(e["summary"], summary_chars))
        for enc in e.get("enclosures", [])[:1]:
            lines.append(f"\nEnclosure: <{enc['url']}> {enc.get('type') or ''}".rstrip())
        lines.append("")
    if len(entries) > max_items:
        lines.append(f"_{len(entries) - max_items} more entries not shown._")
    return "\n".join(lines).strip()


class FeedReader(BaseReader):
    """RSS / Atom / RDF feeds, or pages advertising one."""

    name = "feeds"
    description = "RSS/Atom feeds → latest entries"

    def can_handle(self, url: str) -> bool:
        return looks_like_feed_url(url)

    def steps(self) -> list[StepBackend]:
        return [
            StepBackend(
                "feedparser",
                lambda url, **kw: self._read(url, parse_feed_feedparser, **kw),
                available=lambda: _common.has_module("feedparser"),
                requires=("feedparser",),
                hint="pip install 'prompture[web]' (feedparser)",
            ),
            StepBackend("stdlib", lambda url, **kw: self._read(url, parse_feed_stdlib, **kw)),
        ]

    def _read(self, url: str, parser: Any, *, session: Any = None, max_items: int = 20, **_: Any) -> ReadResult:
        feed_url, data = load_feed(url, session=session)
        feed = parser(data)
        entries = feed.get("entries", [])
        return ReadResult(
            url=url,
            title=feed.get("title") or feed_url,
            content=render_feed(feed, max_items=max_items),
            reader=self.name,
            kind="feed",
            meta={
                "feed_url": feed_url,
                "format": feed.get("format"),
                "link": feed.get("link"),
                "entry_count": len(entries),
                "entries": [
                    {k: e.get(k) for k in ("title", "link", "published", "enclosures", "transcripts")}
                    for e in entries[:max_items]
                ],
            },
        )
