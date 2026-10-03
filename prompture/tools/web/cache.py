"""Local cache for web search, fetch, URL readers and platform search.

Repeated questions shouldn't cost a network round trip, but a weather or
price query must not be served from yesterday. Each entry gets a lifetime
from :func:`ttl_for`, based on how fast that kind of content changes:

=====================================================  ==========
Content                                                Lifetime
=====================================================  ==========
Volatile searches (weather, prices, scores, news,      10 min
"today", ``recency_days <= 1``)
Recent-window searches (``recency_days <= 7``)         1 h
Other searches                                         6 h
Platform search (GitHub, HN, arXiv, YouTube)           30 min
Fast-moving pages (HN, Reddit, X, status pages)        10 min
Ordinary pages                                         1 h
Wikipedia                                              1 day
Papers, PDFs, DOIs, commit-pinned GitHub files,        7 days
YouTube transcripts
GitHub issues / pull requests, HN items                10 min
Feeds                                                  15 min
Podcast episodes                                       1 day
=====================================================  ==========

``PROMPTURE_WEB_CACHE`` picks the store: ``disk`` (default, SQLite at
``~/.prompture/cache/web_cache.db`` — shared across processes and CLI runs),
``memory`` (this process only) or ``off``. ``PROMPTURE_WEB_CACHE_PATH``
moves the database. Every function also takes ``use_cache=False`` and
``cache_ttl=<seconds>``. Errors and challenge pages are never cached, and a
cached result says so: ``route["cached"] is True`` with ``cache_age_s``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ...infra.cache import CacheBackend, MemoryCacheBackend, SQLiteCacheBackend

logger = logging.getLogger("prompture.tools.web.cache")

CACHE_ENV = "PROMPTURE_WEB_CACHE"
CACHE_PATH_ENV = "PROMPTURE_WEB_CACHE_PATH"
DEFAULT_PATH = Path.home() / ".prompture" / "cache" / "web_cache.db"
MAX_ENTRIES = 5000
MEMORY_ENTRIES = 512
# Pages above this size stay out of the disk cache (still served, just not stored).
MAX_VALUE_BYTES = 2 * 1024 * 1024

MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR

# Queries whose answers change by the hour.
_VOLATILE_QUERY = re.compile(
    r"\b(weather|forecast|temperature|rain|snow|today|tonight|tomorrow|yesterday|right now|currently|"
    r"live|breaking|latest|news|headlines?|price|prices|stock|stocks|shares?|quote|exchange rate|crypto|"
    r"bitcoin|score|scores|standings|results|traffic|outage|status|election|polls?|this (week|month))\b",
    re.IGNORECASE,
)
_VOLATILE_HOSTS = (
    "news.ycombinator.com",
    "reddit.com",
    "x.com",
    "twitter.com",
    "status.",
    "weather.",
    "finance.yahoo.com",
)
_STATIC_HOSTS = ("arxiv.org", "doi.org", "export.arxiv.org", "openreview.net", "aclanthology.org")
_SHA_PATH = re.compile(r"/[0-9a-f]{40}(/|$)")

_lock = threading.Lock()
_store: tuple[tuple[str, str], CacheBackend | None] | None = None


# ---------------------------------------------------------------------------
# Lifetimes
# ---------------------------------------------------------------------------


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def search_ttl(query: str, recency_days: int | None = None) -> int:
    """Lifetime for a web search: short for volatile topics, hours otherwise."""
    if recency_days is not None and recency_days <= 1:
        return 10 * MINUTE
    if _VOLATILE_QUERY.search(query or ""):
        return 10 * MINUTE
    if recency_days is not None and recency_days <= 7:
        return HOUR
    return 6 * HOUR


def page_ttl(url: str) -> int:
    """Lifetime for a fetched page, from how static its URL looks."""
    host = _host(url)
    path = urlsplit(url).path.lower() if url else ""
    if any(host == h or host.endswith("." + h) or (h.endswith(".") and host.startswith(h)) for h in _VOLATILE_HOSTS):
        return 10 * MINUTE
    if any(host == h or host.endswith("." + h) for h in _STATIC_HOSTS) or path.endswith(".pdf"):
        return 7 * DAY
    if host in ("github.com", "raw.githubusercontent.com") and _SHA_PATH.search(path):
        return 7 * DAY
    if host.endswith("wikipedia.org"):
        return DAY
    return HOUR


def read_ttl(url: str, reader: str, kind: str = "page") -> int:
    """Lifetime for a :func:`read_url` result, per reader and content kind."""
    reader = (reader or "").lower()
    if reader == "youtube":
        return 7 * DAY  # a published video's transcript doesn't change
    if reader == "arxiv":
        return 7 * DAY
    if reader in ("wikipedia", "anilist"):
        return DAY
    if reader == "podcasts":
        return DAY
    if reader == "feeds":
        return 15 * MINUTE
    if reader == "hackernews":
        return 10 * MINUTE
    if reader == "github":
        if kind in ("issue", "pull_request", "pr", "discussion") or re.search(r"/(issues|pull|discussions)/", url):
            return 10 * MINUTE
        return page_ttl(url)
    return page_ttl(url)


PLATFORM_TTL = 30 * MINUTE


def ttl_for(kind: str, **context: Any) -> int:
    """Lifetime in seconds for an entry of *kind* (``search``, ``fetch``, ``read``, ``platform``)."""
    if kind == "search":
        return search_ttl(context.get("query", ""), context.get("recency_days"))
    if kind == "fetch":
        return page_ttl(context.get("url", ""))
    if kind == "read":
        return read_ttl(context.get("url", ""), context.get("reader", ""), context.get("result_kind", "page"))
    if kind == "platform":
        return PLATFORM_TTL
    return HOUR


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _setting(name: str) -> str | None:
    try:
        from ...infra.credentials import get_config_value

        value = get_config_value(name)
    except Exception:
        import os

        value = os.environ.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def cache_mode() -> str:
    """``disk`` (default), ``memory`` or ``off``."""
    mode = (_setting(CACHE_ENV) or "disk").lower()
    if mode in ("0", "false", "no", "none", "disabled"):
        return "off"
    return mode if mode in ("disk", "memory", "off") else "disk"


def cache_path() -> Path:
    custom = _setting(CACHE_PATH_ENV)
    return Path(custom).expanduser() if custom else DEFAULT_PATH


def _backend() -> CacheBackend | None:
    """The active store, rebuilt when the mode or path changes."""
    global _store
    signature = (cache_mode(), str(cache_path()))
    with _lock:
        if _store is not None and _store[0] == signature:
            return _store[1]
        mode, path = signature
        backend: CacheBackend | None
        if mode == "off":
            backend = None
        elif mode == "memory":
            backend = MemoryCacheBackend(maxsize=MEMORY_ENTRIES)
        else:
            try:
                backend = SQLiteCacheBackend(db_path=path, max_entries=MAX_ENTRIES)
            except Exception as exc:  # read-only home, locked file, ...
                logger.warning("web cache: disk store unavailable (%s); using memory", exc)
                backend = MemoryCacheBackend(maxsize=MEMORY_ENTRIES)
        _store = (signature, backend)
        return backend


def make_key(kind: str, *parts: Any) -> str:
    """Stable key for *kind* and its arguments."""
    raw = json.dumps([kind, *parts], sort_keys=True, default=str, ensure_ascii=False)
    return f"web:{kind}:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get(key: str) -> tuple[Any, float] | None:
    """Return ``(value, age_seconds)`` for a live entry, else ``None``."""
    backend = _backend()
    if backend is None:
        return None
    try:
        entry = backend.get(key)
    except Exception as exc:
        logger.debug("web cache read failed: %s", exc)
        return None
    if not isinstance(entry, dict) or "value" not in entry:
        return None
    return entry["value"], max(0.0, time.time() - float(entry.get("stored_at", time.time())))


def put(key: str, value: Any, ttl: float) -> None:
    """Store a JSON-serializable *value* for *ttl* seconds (``<= 0`` skips)."""
    backend = _backend()
    if backend is None or ttl <= 0:
        return
    entry = {"stored_at": time.time(), "value": value}
    if isinstance(backend, SQLiteCacheBackend):
        try:
            if len(json.dumps(entry, default=str)) > MAX_VALUE_BYTES:
                return
        except (TypeError, ValueError):
            return
    try:
        backend.set(key, entry, ttl=int(ttl))
    except Exception as exc:
        logger.debug("web cache write failed: %s", exc)


def mark_cached(route: dict[str, Any] | None, age: float) -> dict[str, Any]:
    """Copy of *route* flagged as served from the cache."""
    return {**(route or {}), "cached": True, "cache_age_s": round(age, 1)}


def clear_web_cache() -> None:
    """Drop every cached search, page, reader result and platform search."""
    backend = _backend()
    if backend is not None:
        backend.clear()


def cache_info() -> dict[str, Any]:
    """Mode, location and entry count (for doctor / debugging).

    Read-only: it never creates the cache database.
    """
    mode = cache_mode()
    path = cache_path()
    info: dict[str, Any] = {"mode": mode, "path": str(path) if mode == "disk" else None, "entries": None}
    if mode == "disk":
        if not path.exists():
            info["entries"] = 0
            return info
        try:
            import sqlite3

            conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=2)
            try:
                info["entries"] = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            finally:
                conn.close()
        except Exception:
            info["entries"] = None
    elif mode == "memory":
        with _lock:
            backend = _store[1] if _store is not None and _store[0] == (mode, str(path)) else None
        info["entries"] = len(backend._data) if isinstance(backend, MemoryCacheBackend) else 0
    return info


def _reset_for_tests() -> None:
    global _store
    with _lock:
        _store = None
