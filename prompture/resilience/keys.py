"""Multiple credentials per provider, rotated round-robin.

Configure a pool with ``{PROVIDER}_API_KEYS=key1,key2,...`` (for example
``OPENAI_API_KEYS`` or ``CLAUDE_API_KEYS`` / ``ANTHROPIC_API_KEYS``), or
register one in code with :func:`register_key_pool`. Keys are never logged:
they are identified by the first 8 hex chars of their SHA-256.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import threading
from collections.abc import Iterable


def key_id(key: str) -> str:
    """Stable, non-reversible short id for a credential (matches the usage ledger)."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


class KeyPool:
    """An ordered set of credentials for one provider."""

    def __init__(self, provider: str, keys: Iterable[str]) -> None:
        unique: list[str] = []
        for k in keys:
            k = k.strip()
            if k and k not in unique:
                unique.append(k)
        if not unique:
            raise ValueError(f"KeyPool for '{provider}' needs at least one key")
        self.provider = provider
        self._keys = unique
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def key_ids(self) -> list[str]:
        return [key_id(k) for k in self._keys]

    def rotation(self) -> list[str]:
        """All keys, starting from the next one in round-robin order."""
        with self._lock:
            start = next(self._counter) % len(self._keys)
        return self._keys[start:] + self._keys[:start]


_pools: dict[str, KeyPool] = {}
_pools_lock = threading.Lock()


def register_key_pool(provider: str, keys: Iterable[str]) -> KeyPool:
    """Register (or replace) the key pool used for *provider*."""
    pool = KeyPool(provider, keys)
    with _pools_lock:
        _pools[provider] = pool
    return pool


def clear_key_pools() -> None:
    with _pools_lock:
        _pools.clear()


def _env_names(provider: str) -> list[str]:
    names = [f"{provider.upper()}_API_KEYS"]
    try:
        from ..drivers import PROVIDER_DRIVER_MAP, PROVIDER_NAME_MAP

        info = PROVIDER_DRIVER_MAP.get(provider)
        if info is not None:
            attr = info[1].get("api_key")
            if attr:
                names.append(f"{attr.upper()}S")
        canonical = PROVIDER_NAME_MAP.get(provider)
        if canonical:
            names.append(f"{canonical.upper()}_API_KEYS")
    except Exception:  # pragma: no cover - registry import problems shouldn't break routing
        pass
    return list(dict.fromkeys(names))


def get_key_pool(provider: str) -> KeyPool | None:
    """Pool for *provider*: an explicitly registered one, else one built from env."""
    with _pools_lock:
        pool = _pools.get(provider)
    if pool is not None:
        return pool
    for name in _env_names(provider):
        raw = os.environ.get(name, "")
        keys = [k for k in raw.split(",") if k.strip()]
        if len(keys) >= 1:
            with _pools_lock:
                return _pools.setdefault(provider, KeyPool(provider, keys))
    return None
