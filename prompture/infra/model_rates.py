"""Live model rates from models.dev API with local caching.

Fetches pricing and metadata for LLM models from https://models.dev/api.json,
caches locally with TTL-based auto-refresh, and provides lookup functions
used by drivers for cost calculations.
"""

import contextlib
import json
import logging
import threading
from dataclasses import dataclass
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
    "moonshot": "moonshotai",
    "zai": "zai",
    "elevenlabs": "elevenlabs",
}

# Proxy providers that re-expose models from other providers.  Used by
# _lookup_model() to trigger a cross-provider search when the proxy's own
# name isn't present in models.dev.
_PROXY_PROVIDERS: frozenset[str] = frozenset({"cachibot"})

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
        return bool(age.total_seconds() < ttl_days * 86400)
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
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except Exception:
        return None


def _fetch_from_api() -> Optional[dict[str, Any]]:
    """Fetch fresh data from models.dev API."""
    try:
        import requests

        resp = requests.get(_API_URL, timeout=15)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]
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


def _strip_to_base_model(model_id: str) -> Optional[str]:
    """Try to derive the base model name from a versioned or fine-tuned model ID.

    Handles common patterns:
    - Date-versioned: ``gpt-4o-2024-08-06`` → ``gpt-4o``
    - Fine-tuned: ``ft:gpt-4o-mini:org:name:id`` → ``gpt-4o-mini``
    - Snapshot suffixes: ``claude-3-opus-20240229`` → ``claude-3-opus``

    Returns ``None`` if no base model can be derived or if the result
    would be identical to the input.
    """
    import re

    candidate: Optional[str] = None

    # Fine-tuned models: ft:base-model:rest
    if model_id.startswith("ft:"):
        parts = model_id.split(":", 2)
        if len(parts) >= 2 and parts[1]:
            candidate = parts[1]

    # Date-versioned: strip trailing -YYYY-MM-DD or -YYYYMMDD
    if candidate is None:
        stripped = re.sub(r"-\d{4}-?\d{2}-?\d{2}$", "", model_id)
        if stripped != model_id:
            candidate = stripped

    return candidate if candidate and candidate != model_id else None


def _lookup_in_provider(
    data: dict[str, Any],
    api_provider: str,
    model_id: str,
) -> Optional[dict[str, Any]]:
    """Lookup a model in a specific provider's data, with base-model fallback."""
    provider_data = data.get(api_provider)
    if not isinstance(provider_data, dict):
        return None

    models = provider_data.get("models", provider_data)
    if not isinstance(models, dict):
        return None

    # Exact match first
    entry = models.get(model_id)
    if entry is not None:
        return entry

    # Fallback: try base model name (date-stripped / fine-tune prefix)
    base = _strip_to_base_model(model_id)
    if base is not None:
        return models.get(base)

    return None


def _lookup_model(provider: str, model_id: str) -> Optional[dict[str, Any]]:
    """Find a model entry in the cached data.

    The API structure is ``{provider: {model_id: {...}, ...}, ...}``.

    When an exact match isn't found, attempts to match by stripping date
    suffixes or fine-tune prefixes (e.g. ``gpt-4o-2024-08-06`` → ``gpt-4o``).

    For proxy providers (e.g. ``cachibot``), the model_id may contain the
    real upstream provider (e.g. ``openai/gpt-4o``).  In that case, the
    lookup is redirected to the upstream provider in models.dev.
    """
    data = _ensure_loaded()
    if data is None:
        return None

    api_provider = PROVIDER_MAP.get(provider, provider)

    # Try direct lookup in the provider
    entry = _lookup_in_provider(data, api_provider, model_id)
    if entry is not None:
        return entry

    # Proxy provider fallback: model_id may be "upstream_provider/model"
    # (e.g. provider="cachibot", model_id="openai/gpt-4o")
    if "/" in model_id:
        upstream_provider, upstream_model = model_id.split("/", 1)
        upstream_api = PROVIDER_MAP.get(upstream_provider, upstream_provider)
        entry = _lookup_in_provider(data, upstream_api, upstream_model)
        if entry is not None:
            return entry

    # Cross-provider search: for known proxy providers whose models live
    # under a different provider in models.dev (e.g. cachibot's "gpt-4o"
    # is really openai's "gpt-4o").
    if provider in _PROXY_PROVIDERS:
        for p_data in data.values():
            if not isinstance(p_data, dict):
                continue
            models = p_data.get("models", p_data)
            if not isinstance(models, dict):
                continue
            hit = models.get(model_id)
            if hit is not None:
                return hit
            base = _strip_to_base_model(model_id)
            if base is not None:
                hit = models.get(base)
                if hit is not None:
                    return hit

    return None


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

    # models.dev nests actual models under a "models" key
    models = provider_data.get("models", provider_data)
    if not isinstance(models, dict):
        return []

    return list(models.keys())


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


# ── Model Capabilities ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelCapabilities:
    """Normalized capability metadata for an LLM model from models.dev.

    All fields default to ``None`` (unknown) so callers can distinguish
    "the model doesn't support X" from "we have no data about X".
    """

    supports_temperature: Optional[bool] = None
    supports_tool_use: Optional[bool] = None
    supports_structured_output: Optional[bool] = None
    supports_vision: Optional[bool] = None
    is_reasoning: Optional[bool] = None
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    modalities_input: tuple[str, ...] = ()
    modalities_output: tuple[str, ...] = ()


def get_model_capabilities(provider: str, model_id: str) -> Optional[ModelCapabilities]:
    """Return capability metadata for a model, or ``None`` if unavailable.

    Maps models.dev fields to a :class:`ModelCapabilities` instance:

    - ``temperature`` → ``supports_temperature``
    - ``tool_call`` → ``supports_tool_use``
    - ``structured_output`` → ``supports_structured_output``
    - ``"image" in modalities.input`` → ``supports_vision``
    - ``reasoning`` → ``is_reasoning``
    - ``limit.context`` → ``context_window``
    - ``limit.output`` → ``max_output_tokens``
    """
    entry = _lookup_model(provider, model_id)
    if entry is None:
        return None

    # Boolean capabilities (True/False/None)
    supports_temperature: Optional[bool] = None
    if "temperature" in entry:
        supports_temperature = bool(entry["temperature"])

    supports_tool_use: Optional[bool] = None
    if "tool_call" in entry:
        supports_tool_use = bool(entry["tool_call"])

    supports_structured_output: Optional[bool] = None
    if "structured_output" in entry:
        supports_structured_output = bool(entry["structured_output"])

    is_reasoning: Optional[bool] = None
    if "reasoning" in entry:
        is_reasoning = bool(entry["reasoning"])

    # Modalities
    modalities = entry.get("modalities", {})
    modalities_input: tuple[str, ...] = ()
    modalities_output: tuple[str, ...] = ()
    if isinstance(modalities, dict):
        raw_in = modalities.get("input")
        if isinstance(raw_in, (list, tuple)):
            modalities_input = tuple(str(m) for m in raw_in)
        raw_out = modalities.get("output")
        if isinstance(raw_out, (list, tuple)):
            modalities_output = tuple(str(m) for m in raw_out)

    supports_vision: Optional[bool] = None
    if modalities_input:
        supports_vision = "image" in modalities_input

    # Limits
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    limits = entry.get("limit", {})
    if isinstance(limits, dict):
        ctx = limits.get("context")
        if ctx is not None:
            with contextlib.suppress(TypeError, ValueError):
                context_window = int(ctx)
        out = limits.get("output")
        if out is not None:
            with contextlib.suppress(TypeError, ValueError):
                max_output_tokens = int(out)

    return ModelCapabilities(
        supports_temperature=supports_temperature,
        supports_tool_use=supports_tool_use,
        supports_structured_output=supports_structured_output,
        supports_vision=supports_vision,
        is_reasoning=is_reasoning,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        modalities_input=modalities_input,
        modalities_output=modalities_output,
    )
