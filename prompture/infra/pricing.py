"""Pluggable pricing-source registry.

Drivers compute cost via :func:`resolve_rates`, which walks a registry of
:class:`PricingSource` objects in priority order (lowest first) and returns
the first source's rates that include both ``input`` and ``output``.

Two sources ship by default:

* :class:`LocalKBPricingSource` — reads ``cost`` blocks from the per-provider
  JSON files in ``prompture/infra/rates/`` (priority 0). Curated locally so
  Prompture's pricing isn't held hostage by a stale third-party feed.
* :class:`ModelsDevPricingSource` — reads pricing from the cached models.dev
  API data (priority 100). Acts as a fallback for models the local KB hasn't
  curated yet.

The :attr:`Settings.pricing_source` setting selects which built-ins are
registered at startup (``"local_first"``, ``"local_only"``,
``"models_dev_only"``). Custom sources can always be added at runtime via
:func:`register_pricing_source` regardless of that setting.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Per-1M-token rate keys we recognise. Sources may return any subset.
_RATE_KEYS: tuple[str, ...] = ("input", "output", "cache_read", "cache_write", "reasoning")


@runtime_checkable
class PricingSource(Protocol):
    """A source of per-1M-token pricing for ``(provider, model_id)`` pairs.

    Implementations must define ``name`` (unique identifier used for
    deduplication / unregister) and ``priority`` (lower = queried first).
    The convention is:

    * 0–49   — curated, authoritative sources (e.g. local KB)
    * 50–99  — third-party feeds you trust
    * 100+   — best-effort fallbacks (e.g. scraped or auto-imported data)
    """

    name: str
    priority: int

    def get_rates(self, provider: str, model_id: str) -> dict[str, float] | None:
        """Return per-1M rates or ``None`` when the source has no data.

        Returned dict should include ``input`` and ``output`` keys (per 1M
        tokens, USD); ``cache_read``, ``cache_write``, ``reasoning`` are
        optional. Sources that can't supply at least input+output should
        return ``None`` so the resolver can fall through.
        """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_sources: list[PricingSource] = []
_lock = threading.Lock()
_initialized = False


def register_pricing_source(source: PricingSource, *, replace: bool = False) -> None:
    """Add *source* to the registry, ordered by ``source.priority``.

    If ``replace`` is true and a source with the same ``name`` already
    exists, it's removed first.

    Calling this function takes the registry under explicit user control:
    the lazy bootstrap won't install built-in defaults afterwards. Use
    :func:`clear_pricing_sources` to release control.
    """
    global _initialized
    with _lock:
        if replace:
            _sources[:] = [s for s in _sources if s.name != source.name]
        elif any(s.name == source.name for s in _sources):
            raise ValueError(f"Pricing source '{source.name}' is already registered. Pass replace=True to overwrite.")
        _sources.append(source)
        _sources.sort(key=lambda s: (s.priority, s.name))
        # User explicitly registered something — suppress lazy defaults
        # for the rest of this process (or until clear_pricing_sources).
        _initialized = True


def unregister_pricing_source(name: str) -> bool:
    """Remove the named source. Returns True if anything was removed."""
    with _lock:
        before = len(_sources)
        _sources[:] = [s for s in _sources if s.name != name]
        return len(_sources) < before


def get_pricing_sources() -> list[PricingSource]:
    """Return a snapshot of the currently registered sources, in priority order."""
    _ensure_initialized()
    with _lock:
        return list(_sources)


def clear_pricing_sources() -> None:
    """Remove all sources. Mainly useful in tests."""
    global _initialized
    with _lock:
        _sources.clear()
        _initialized = False


def resolve_rates(provider: str, model_id: str) -> dict[str, float] | None:
    """Walk the registry and return the first source's rates for ``(provider, model_id)``.

    A source's result is accepted only if it includes both ``input`` and
    ``output`` (without those, cost calculation is impossible).
    """
    for source in get_pricing_sources():
        try:
            rates = source.get_rates(provider, model_id)
        except Exception:
            logger.debug("Pricing source %r raised; falling through", source.name, exc_info=True)
            continue
        if rates and rates.get("input") is not None and rates.get("output") is not None:
            return rates
    return None


# ---------------------------------------------------------------------------
# Built-in sources
# ---------------------------------------------------------------------------

_RATES_DIR = Path(__file__).resolve().parent / "rates"


def _coerce_rate_dict(raw: Any) -> dict[str, float] | None:
    """Turn a ``cost`` block into a clean float dict, or ``None`` if invalid."""
    if not isinstance(raw, dict):
        return None
    out: dict[str, float] = {}
    for key in _RATE_KEYS:
        val = raw.get(key)
        if val is None:
            continue
        with contextlib.suppress(TypeError, ValueError):
            out[key] = float(val)
    return out or None


def _load_local_rates_kb() -> dict[tuple[str, str], dict[str, float]]:
    """Build the local-KB pricing index from ``infra/rates/*.json`` files.

    Only entries with a ``cost`` object are included.
    """
    kb: dict[tuple[str, str], dict[str, float]] = {}
    for json_file in sorted(_RATES_DIR.glob("*.json")):
        provider = json_file.stem
        try:
            raw: dict[str, Any] = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to parse %s — skipping", json_file)
            continue
        for model_id, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            rates = _coerce_rate_dict(entry.get("cost"))
            if rates is None:
                continue
            if "input" in rates and "output" in rates:
                kb[(provider, model_id)] = rates
    return kb


class LocalKBPricingSource:
    """Pricing source backed by the curated JSON files in ``infra/rates/``.

    Entries are loaded once at construction. Call :meth:`reload` after
    editing the JSON files in a long-running process.
    """

    name = "local_kb"
    priority = 0

    def __init__(self) -> None:
        self._kb: dict[tuple[str, str], dict[str, float]] = _load_local_rates_kb()

    def reload(self) -> None:
        """Rebuild the in-memory KB from disk."""
        self._kb = _load_local_rates_kb()

    def get_rates(self, provider: str, model_id: str) -> dict[str, float] | None:
        # Resolve the prompture provider name to its models.dev key (so
        # provider aliases like "claude" → "anthropic" work for both
        # storage forms — but the JSON files are keyed by the prompture
        # name, so we try that first.)
        from .model_rates import PROVIDER_MAP, _strip_to_base_model

        for prov in (provider, PROVIDER_MAP.get(provider, provider)):
            hit = self._kb.get((prov, model_id))
            if hit is not None:
                return dict(hit)
            base = _strip_to_base_model(model_id)
            if base is not None:
                hit = self._kb.get((prov, base))
                if hit is not None:
                    return dict(hit)
        return None


class ModelsDevPricingSource:
    """Pricing source backed by the cached models.dev API data."""

    name = "models_dev"
    priority = 100

    def get_rates(self, provider: str, model_id: str) -> dict[str, float] | None:
        from .model_rates import _lookup_model

        entry = _lookup_model(provider, model_id)
        if entry is None:
            return None
        return _coerce_rate_dict(entry.get("cost"))


# ---------------------------------------------------------------------------
# Default-source bootstrapping
# ---------------------------------------------------------------------------


def _initialize_default_sources(mode: str | None = None) -> None:
    """Install built-in sources for the requested mode.

    If *mode* is ``None``, reads from ``settings.pricing_source``. Recognised
    modes: ``"local_first"`` (default), ``"local_only"``, ``"models_dev_only"``.
    Unknown values fall back to ``"local_first"`` with a warning.

    Idempotent — calling twice without an intervening
    :func:`clear_pricing_sources` is a no-op.
    """
    global _initialized
    with _lock:
        if _initialized:
            return

        if mode is None:
            try:
                from .settings import settings as _settings

                mode = getattr(_settings, "pricing_source", "local_first")
            except Exception:
                mode = "local_first"

        if mode not in {"local_first", "local_only", "models_dev_only"}:
            logger.warning("Unknown pricing_source=%r — falling back to local_first", mode)
            mode = "local_first"

        # Reset to a clean slate so this can be re-bootstrapped after a
        # clear_pricing_sources() call (used in tests).
        _sources.clear()

        if mode == "models_dev_only":
            _sources.append(ModelsDevPricingSource())
        else:
            _sources.append(LocalKBPricingSource())
            if mode == "local_first":
                _sources.append(ModelsDevPricingSource())

        _sources.sort(key=lambda s: (s.priority, s.name))
        _initialized = True


def _ensure_initialized() -> None:
    """Lazily install default sources on first registry access."""
    if not _initialized:
        _initialize_default_sources()
