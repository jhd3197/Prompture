"""Live model rates from models.dev API with local caching.

Fetches pricing and metadata for LLM models from https://models.dev/api.json,
caches locally with TTL-based auto-refresh, and provides lookup functions
used by drivers for cost calculations.
"""

import contextlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Maps prompture provider names to models.dev provider names
PROVIDER_MAP: dict[str, str] = {
    "openai": "openai",
    "claude": "anthropic",
    "google": "google",
    "groq": "groq",
    "grok": "xai",
    "azure": "azure",
    "openrouter": "openrouter",
}

_API_URL = "https://models.dev/api.json"
_CACHE_DIR = Path.home() / ".prompture" / "cache"
_CACHE_FILE = _CACHE_DIR / "models_dev.json"
_META_FILE = _CACHE_DIR / "models_dev_meta.json"

_lock = threading.Lock()
_data: Optional[dict[str, Any]] = None
_loaded = False


def _get_ttl_days() -> int:
    """Get TTL from settings if available, otherwise default to 7."""
    try:
        from .settings import settings

        return getattr(settings, "model_rates_ttl_days", 7)
    except Exception:
        return 7


def _cache_is_valid() -> bool:
    """Check whether the local cache exists and is within TTL."""
    if not _CACHE_FILE.exists() or not _META_FILE.exists():
        return False
    try:
        meta = json.loads(_META_FILE.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(meta["fetched_at"])
        ttl_days = meta.get("ttl_days", _get_ttl_days())
        age = datetime.now(timezone.utc) - fetched_at
        return age.total_seconds() < ttl_days * 86400
    except Exception:
        return False


def _write_cache(data: dict[str, Any]) -> None:
    """Write API data and metadata to local cache."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(data), encoding="utf-8")
        meta = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "ttl_days": _get_ttl_days(),
        }
        _META_FILE.write_text(json.dumps(meta), encoding="utf-8")
    except Exception as exc:
        logger.debug("Failed to write model rates cache: %s", exc)


def _read_cache() -> Optional[dict[str, Any]]:
    """Read cached API data from disk."""
    try:
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fetch_from_api() -> Optional[dict[str, Any]]:
    """Fetch fresh data from models.dev API."""
    try:
        import requests

        resp = requests.get(_API_URL, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("Failed to fetch model rates from %s: %s", _API_URL, exc)
        return None


def _ensure_loaded() -> Optional[dict[str, Any]]:
    """Lazy-load data: use cache if valid, otherwise fetch from API."""
    global _data, _loaded
    if _loaded:
        return _data

    with _lock:
        # Double-check after acquiring lock
        if _loaded:
            return _data

        if _cache_is_valid():
            _data = _read_cache()
            if _data is not None:
                _loaded = True
                return _data

        # Cache missing or expired — fetch fresh
        fresh = _fetch_from_api()
        if fresh is not None:
            _data = fresh
            _write_cache(fresh)
        else:
            # Fetch failed — try stale cache as last resort
            _data = _read_cache()

        _loaded = True
        return _data


def _lookup_model(provider: str, model_id: str) -> Optional[dict[str, Any]]:
    """Find a model entry in the cached data.

    The API structure is ``{provider: {model_id: {...}, ...}, ...}``.
    """
    data = _ensure_loaded()
    if data is None:
        return None

    api_provider = PROVIDER_MAP.get(provider, provider)
    provider_data = data.get(api_provider)
    if not isinstance(provider_data, dict):
        return None

    return provider_data.get(model_id)


# ── Public API ──────────────────────────────────────────────────────────────


def get_model_rates(provider: str, model_id: str) -> Optional[dict[str, float]]:
    """Return pricing dict for a model, or ``None`` if unavailable.

    Returned keys mirror models.dev cost fields (per 1M tokens):
    ``input``, ``output``, and optionally ``cache_read``, ``cache_write``,
    ``reasoning``.
    """
    entry = _lookup_model(provider, model_id)
    if entry is None:
        return None

    cost = entry.get("cost")
    if not isinstance(cost, dict):
        return None

    rates: dict[str, float] = {}
    for key in ("input", "output", "cache_read", "cache_write", "reasoning"):
        val = cost.get(key)
        if val is not None:
            with contextlib.suppress(TypeError, ValueError):
                rates[key] = float(val)

    # Must have at least input and output to be useful
    if "input" in rates and "output" in rates:
        return rates
    return None


def get_model_info(provider: str, model_id: str) -> Optional[dict[str, Any]]:
    """Return full model metadata (cost, limits, capabilities), or ``None``."""
    return _lookup_model(provider, model_id)


def get_all_provider_models(provider: str) -> list[str]:
    """Return list of model IDs available for a provider."""
    data = _ensure_loaded()
    if data is None:
        return []

    api_provider = PROVIDER_MAP.get(provider, provider)
    provider_data = data.get(api_provider)
    if not isinstance(provider_data, dict):
        return []

    return list(provider_data.keys())


def refresh_rates_cache(force: bool = False) -> bool:
    """Fetch fresh data from models.dev.

    Args:
        force: If ``True``, fetch even when the cache is still within TTL.

    Returns:
        ``True`` if fresh data was fetched and cached successfully.
    """
    global _data, _loaded

    with _lock:
        if not force and _cache_is_valid():
            return False

        fresh = _fetch_from_api()
        if fresh is not None:
            _data = fresh
            _write_cache(fresh)
            _loaded = True
            return True

        return False
