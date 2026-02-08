"""Response caching layer for prompture.

Provides pluggable cache backends (memory, SQLite, Redis) so repeated
identical LLM calls can be served from cache.  Disabled by default.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Cache key generation
# ---------------------------------------------------------------------------

_CACHE_RELEVANT_OPTIONS = frozenset(
    {
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
        "json_mode",
    }
)


def make_cache_key(
    prompt: str,
    model_name: str,
    schema: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    output_format: str = "json",
    pydantic_qualname: str | None = None,
) -> str:
    """Return a deterministic SHA-256 hex key for the given call parameters.

    Only cache-relevant options (temperature, max_tokens, etc.) are included
    so that unrelated option changes don't bust the cache.
    """
    filtered_opts: dict[str, Any] = {}
    if options:
        filtered_opts = {k: v for k, v in sorted(options.items()) if k in _CACHE_RELEVANT_OPTIONS}

    parts: dict[str, Any] = {
        "prompt": prompt,
        "model_name": model_name,
        "schema": schema,
        "options": filtered_opts,
        "output_format": output_format,
    }
    if pydantic_qualname is not None:
        parts["pydantic_qualname"] = pydantic_qualname

    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------


class CacheBackend(ABC):
    """Abstract base class for cache storage backends."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Return the cached value or ``None`` on miss."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store *value* under *key* with optional TTL in seconds."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a single key."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries."""

    @abstractmethod
    def has(self, key: str) -> bool:
        """Return whether *key* exists and is not expired."""


# ---------------------------------------------------------------------------
# Memory backend
# ---------------------------------------------------------------------------


class MemoryCacheBackend(CacheBackend):
    """In-process LRU cache backed by an ``OrderedDict``.

    Parameters
    ----------
    maxsize:
        Maximum number of entries before the least-recently-used item is
        evicted.  Defaults to 256.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()
        self._lock = threading.Lock()

    # -- helpers --
    def _is_expired(self, entry: tuple[Any, float | None]) -> bool:
        _value, expires_at = entry
        if expires_at is None:
            return False
        return time.time() > expires_at

    # -- public API --
    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if self._is_expired(entry):
                del self._data[key]
                return None
            # Move to end (most-recently used)
            self._data.move_to_end(key)
            return entry[0]

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        expires_at = (time.time() + ttl) if ttl else None
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (value, expires_at)
            # Evict LRU entries
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def has(self, key: str) -> bool:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            if self._is_expired(entry):
                del self._data[key]
                return False
            return True


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

_DEFAULT_SQLITE_PATH = Path.home() / ".prompture" / "cache" / "response_cache.db"


class SQLiteCacheBackend(CacheBackend):
    """Persistent cache using a local SQLite database.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Defaults to
        ``~/.prompture/cache/response_cache.db``.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_SQLITE_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path), timeout=5)

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache (
                        key        TEXT PRIMARY KEY,
                        value      TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        ttl        REAL
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def get(self, key: str) -> Any | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT value, created_at, ttl FROM cache WHERE key = ?", (key,)).fetchone()
                if row is None:
                    return None
                value_json, created_at, ttl = row
                if ttl is not None and time.time() > created_at + ttl:
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
                    return None
                return json.loads(value_json)
            finally:
                conn.close()

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        value_json = json.dumps(value, default=str)
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, created_at, ttl) VALUES (?, ?, ?, ?)",
                    (key, value_json, now, ttl),
                )
                conn.commit()
            finally:
                conn.close()

    def delete(self, key: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
            finally:
                conn.close()

    def clear(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM cache")
                conn.commit()
            finally:
                conn.close()

    def has(self, key: str) -> bool:
        return self.get(key) is not None


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------


class RedisCacheBackend(CacheBackend):
    """Cache backend using Redis with native TTL support.

    Requires the ``redis`` package (``pip install redis`` or
    ``pip install prompture[redis]``).

    Parameters
    ----------
    redis_url:
        Redis connection URL (e.g. ``redis://localhost:6379/0``).
    prefix:
        Key prefix.  Defaults to ``"prompture:cache:"``.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", prefix: str = "prompture:cache:") -> None:
        try:
            import redis as _redis
        except ImportError:
            raise RuntimeError(
                "Redis cache backend requires the 'redis' package. "
                "Install it with: pip install redis  (or: pip install prompture[redis])"
            ) from None

        self._client = _redis.from_url(redis_url, decode_responses=True)
        self._prefix = prefix

    def _prefixed(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Any | None:
        raw = self._client.get(self._prefixed(key))
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        value_json = json.dumps(value, default=str)
        if ttl:
            self._client.setex(self._prefixed(key), ttl, value_json)
        else:
            self._client.set(self._prefixed(key), value_json)

    def delete(self, key: str) -> None:
        self._client.delete(self._prefixed(key))

    def clear(self) -> None:
        # Scan for keys with our prefix and delete them
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor, match=f"{self._prefix}*", count=100)
            if keys:
                self._client.delete(*keys)
            if cursor == 0:
                break

    def has(self, key: str) -> bool:
        return bool(self._client.exists(self._prefixed(key)))


# ---------------------------------------------------------------------------
# ResponseCache orchestrator
# ---------------------------------------------------------------------------


class ResponseCache:
    """Orchestrator that wraps a :class:`CacheBackend` with hit/miss stats
    and an ``enabled`` toggle.

    Parameters
    ----------
    backend:
        The storage backend to use.
    enabled:
        Whether caching is active.  When ``False``, all lookups return
        ``None`` and stores are no-ops.
    default_ttl:
        Default time-to-live in seconds for cached entries.
    """

    def __init__(
        self,
        backend: CacheBackend,
        enabled: bool = True,
        default_ttl: int = 3600,
    ) -> None:
        self.backend = backend
        self.enabled = enabled
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._lock = threading.Lock()

    def get(self, key: str, *, force: bool = False) -> Any | None:
        if not self.enabled and not force:
            with self._lock:
                self._misses += 1
            return None
        value = self.backend.get(key)
        with self._lock:
            if value is not None:
                self._hits += 1
            else:
                self._misses += 1
        return value

    def set(self, key: str, value: Any, ttl: int | None = None, *, force: bool = False) -> None:
        if not self.enabled and not force:
            return
        self.backend.set(key, value, ttl or self.default_ttl)
        with self._lock:
            self._sets += 1

    def invalidate(self, key: str) -> None:
        self.backend.delete(key)

    def clear(self) -> None:
        self.backend.clear()
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._sets = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses, "sets": self._sets}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_cache_instance: ResponseCache | None = None
_cache_lock = threading.Lock()


def get_cache() -> ResponseCache:
    """Return the module-level :class:`ResponseCache` singleton.

    If :func:`configure_cache` has not been called, returns a disabled
    cache backed by :class:`MemoryCacheBackend`.
    """
    global _cache_instance
    with _cache_lock:
        if _cache_instance is None:
            _cache_instance = ResponseCache(
                backend=MemoryCacheBackend(),
                enabled=False,
            )
        return _cache_instance


def configure_cache(
    backend: str = "memory",
    enabled: bool = True,
    ttl: int = 3600,
    maxsize: int = 256,
    db_path: str | None = None,
    redis_url: str | None = None,
) -> ResponseCache:
    """Create (or replace) the module-level cache singleton.

    Parameters
    ----------
    backend:
        ``"memory"``, ``"sqlite"``, or ``"redis"``.
    enabled:
        Whether the cache is active.
    ttl:
        Default TTL in seconds.
    maxsize:
        Maximum entries for the memory backend.
    db_path:
        SQLite database path (only for ``"sqlite"`` backend).
    redis_url:
        Redis connection URL (only for ``"redis"`` backend).

    Returns
    -------
    The newly configured :class:`ResponseCache`.
    """
    global _cache_instance

    if backend == "memory":
        be = MemoryCacheBackend(maxsize=maxsize)
    elif backend == "sqlite":
        be = SQLiteCacheBackend(db_path=db_path)
    elif backend == "redis":
        be = RedisCacheBackend(redis_url=redis_url or "redis://localhost:6379/0")
    else:
        raise ValueError(f"Unknown cache backend '{backend}'. Choose 'memory', 'sqlite', or 'redis'.")

    with _cache_lock:
        _cache_instance = ResponseCache(backend=be, enabled=enabled, default_ttl=ttl)
        return _cache_instance


def _reset_cache() -> None:
    """Reset the singleton to ``None``.  **For testing only.**"""
    global _cache_instance
    with _cache_lock:
        _cache_instance = None
