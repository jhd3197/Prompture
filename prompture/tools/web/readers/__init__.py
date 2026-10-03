"""URL-routed readers behind :func:`read_url`.

Readers are tried in registry order; the first one whose ``can_handle``
matches and that has a usable backend reads the URL. When none matches (or
every matching reader fails), :func:`read_url` falls back to
:func:`~prompture.tools.web.fetch.web_fetch`.

Built-in order: YouTube, GitHub, Hacker News, arXiv, Wikipedia, AniList, Podcasts,
Feeds. Add your own with :func:`register_reader` (``first=True`` to take
precedence over the built-ins).
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Any

import requests

from ....capabilities.errors import UnsafeURLError
from ....capabilities.url_safety import normalize_public_http_url
from ....resilience.errors import classify_error
from .. import cache as web_cache
from .._common import error_text, page
from ..fetch import web_fetch
from .anilist import AniListReader, search_anilist
from .arxiv import ArxivReader
from .base import BaseReader, Reader, ReadResult, StepBackend
from .feeds import FeedReader
from .github import GitHubReader
from .hackernews import HackerNewsReader
from .podcasts import PodcastReader
from .wikipedia import WikipediaReader
from .youtube import YouTubeReader

_readers: list[Any] = []
_lock = threading.Lock()


def register_reader(reader: Any, *, first: bool = False) -> Any:
    """Add *reader* to the registry (replacing one with the same name).

    Args:
        reader: Object implementing the :class:`Reader` protocol.
        first: Insert at the front so it wins over existing readers.
    """
    if not isinstance(reader, Reader):
        raise TypeError("reader must implement name, can_handle(url), read(url, **kw) and check(live)")
    with _lock:
        _readers[:] = [r for r in _readers if r.name != reader.name]
        if first:
            _readers.insert(0, reader)
        else:
            _readers.append(reader)
    return reader


def unregister_reader(name: str) -> None:
    with _lock:
        _readers[:] = [r for r in _readers if r.name != name]


def list_readers() -> list[Any]:
    with _lock:
        return list(_readers)


def get_reader(name: str) -> Any | None:
    with _lock:
        return next((r for r in _readers if r.name == name), None)


def _healthy(reader: Any) -> bool:
    try:
        avail = getattr(reader, "available", None)
        if callable(avail):
            return bool(avail())
        return bool(reader.check(False).ok)
    except Exception:
        return False


def matching_readers(url: str) -> list[Any]:
    """Readers whose ``can_handle`` accepts *url*, in registry order."""
    out = []
    for r in list_readers():
        try:
            if r.can_handle(url):
                out.append(r)
        except Exception:
            continue
    return out


def _looks_like_feed(fr: Any) -> bool:
    ctype = (fr.content_type or "").lower()
    if ctype in ("application/rss+xml", "application/atom+xml", "application/rdf+xml"):
        return True
    head = fr.content.lstrip()[:300].lower()
    return head.startswith(("<rss", "<feed", "<rdf:rdf")) or (
        head.startswith("<?xml") and ("<rss" in head or "<feed" in head)
    )


def read_url(
    url: str,
    *,
    reader: str | None = None,
    max_chars: int = 20000,
    start: int = 0,
    fallback: bool = True,
    session: requests.Session | None = None,
    use_cache: bool = True,
    cache_ttl: float | None = None,
    **kwargs: Any,
) -> ReadResult:
    """Read *url* with the best matching reader, falling back to :func:`web_fetch`.

    Args:
        url: Public http(s) URL (private targets are refused up front).
        reader: Force a reader by name (``"feeds"``, ``"podcasts"``, ...).
        max_chars: Characters per slice (``0`` = everything).
        start: Offset for paging through long content.
        fallback: Fall back to :func:`web_fetch` when no reader succeeds.
        session: Optional ``requests.Session``.
        use_cache: Serve a recent read of the same URL from the local web cache.
            Lifetime depends on the reader: days for YouTube transcripts and
            papers, minutes for issues, HN threads and feeds.
        cache_ttl: Override the cache lifetime in seconds (``0`` = don't store).
        **kwargs: Reader options (``languages``, ``timestamps``, ``max_comments``,
            ``full_text``, ``full``, ``episode``, ``max_items``, ...).

    Raises:
        UnsafeURLError: The URL is not a public http(s) address.
        ValueError: Unknown reader name.
    """
    safe_url = normalize_public_http_url(url)
    key = web_cache.make_key("read", safe_url, reader, fallback, kwargs)
    stored = web_cache.get(key) if use_cache else None
    if stored is not None:
        value, age = stored
        full = ReadResult(**value)
        full.route = web_cache.mark_cached(full.route, age)
    else:
        full = _read_url_live(safe_url, reader=reader, fallback=fallback, session=session, **kwargs)
        if use_cache and full.content:
            ttl = cache_ttl if cache_ttl is not None else web_cache.read_ttl(safe_url, full.reader, full.kind)
            web_cache.put(key, asdict(full), ttl)
    piece, truncated, next_start, total = page(full.content, start=start, max_chars=max_chars)
    full.content, full.truncated, full.next_start, full.total_chars = piece, truncated, next_start, total
    return full


def _read_url_live(
    safe_url: str,
    *,
    reader: str | None,
    fallback: bool,
    session: requests.Session | None,
    **kwargs: Any,
) -> ReadResult:
    """Run the readers for *safe_url* and return the full (unpaged) result."""
    max_chars, start = 0, 0
    if reader:
        forced = get_reader(reader)
        if forced is None:
            raise ValueError(f"Unknown reader {reader!r}. Known: {', '.join(r.name for r in list_readers())}")
        candidates = [forced]
    else:
        candidates = matching_readers(safe_url)

    attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for r in candidates:
        if not reader and not _healthy(r):
            attempts.append({"backend": r.name, "status": "skipped", "reason": "unavailable"})
            continue
        started = time.monotonic()
        try:
            result = r.read(safe_url, session=session, **kwargs)
        except UnsafeURLError:
            raise
        except Exception as exc:
            last_error = exc
            info = classify_error(exc)
            attempts.append(
                {
                    "backend": r.name,
                    "status": "error",
                    "error": error_text(exc),
                    "category": info.category,
                    "attempts": list(getattr(exc, "attempts", []) or []),
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                }
            )
            continue
        inner = dict(result.route or {})
        inner_served = inner.get("served_by")
        result.route = {
            "reader": r.name,
            "served_by": f"{r.name}/{inner_served}" if inner_served and inner_served != r.name else r.name,
            "fallback": any(a["status"] == "error" for a in attempts) or bool(inner.get("fallback")),
            "attempts": [*attempts, {"backend": r.name, "status": "ok", "attempts": inner.get("attempts", [])}],
        }
        piece, truncated, next_start, total = page(result.content, start=start, max_chars=max_chars)
        result.content, result.truncated, result.next_start, result.total_chars = piece, truncated, next_start, total
        return result

    if not fallback:
        if last_error is not None:
            raise last_error
        raise ValueError(f"No reader handles {safe_url}")

    fr = web_fetch(safe_url, max_chars=max_chars, start=start, session=session, use_cache=False)
    feeds = get_reader("feeds")
    if feeds is not None and not reader and _looks_like_feed(fr):
        # No URL pattern matched, but the body is a feed — let the feed reader render it.
        try:
            result = feeds.read(safe_url, session=session, **kwargs)
        except Exception:
            result = None
        if result is not None:
            inner = dict(result.route or {})
            result.route = {
                "reader": feeds.name,
                "served_by": f"{feeds.name}/{inner.get('served_by', feeds.name)}",
                "fallback": bool(attempts and any(a["status"] == "error" for a in attempts)),
                "attempts": [*attempts, {"backend": feeds.name, "status": "ok", "attempts": inner.get("attempts", [])}],
            }
            piece, truncated, next_start, total = page(result.content, start=start, max_chars=max_chars)
            result.content, result.truncated, result.next_start, result.total_chars = (
                piece,
                truncated,
                next_start,
                total,
            )
            return result
    return ReadResult(
        url=safe_url,
        title=fr.title,
        content=fr.content,
        reader="web_fetch",
        kind="page",
        meta={"final_url": fr.final_url, "content_type": fr.content_type},
        route={
            "reader": "web_fetch",
            "served_by": f"web_fetch/{fr.served_by}",
            "fallback": bool(attempts and any(a["status"] == "error" for a in attempts))
            or bool(fr.route.get("fallback")),
            "attempts": [*attempts, {"backend": "web_fetch", "status": "ok", "attempts": fr.route.get("attempts", [])}],
        },
        truncated=fr.truncated,
        next_start=fr.next_start,
        total_chars=fr.total_chars,
    )


async def aread_url(url: str, **kwargs: Any) -> ReadResult:
    """Async :func:`read_url` (runs in a worker thread)."""
    import asyncio

    return await asyncio.to_thread(read_url, url, **kwargs)


def _register_builtins() -> None:
    for r in (
        YouTubeReader(),
        GitHubReader(),
        HackerNewsReader(),
        ArxivReader(),
        WikipediaReader(),
        AniListReader(),
        PodcastReader(),
        FeedReader(),
    ):
        register_reader(r)


_register_builtins()

__all__ = [
    "AniListReader",
    "ArxivReader",
    "BaseReader",
    "FeedReader",
    "GitHubReader",
    "HackerNewsReader",
    "PodcastReader",
    "ReadResult",
    "Reader",
    "StepBackend",
    "WikipediaReader",
    "YouTubeReader",
    "aread_url",
    "get_reader",
    "list_readers",
    "matching_readers",
    "read_url",
    "register_reader",
    "search_anilist",
    "unregister_reader",
]
