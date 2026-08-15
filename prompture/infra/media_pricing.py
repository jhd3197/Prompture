"""Media generation pricing + a public cost-quote function (G16).

LLM pricing is per-token and lives in :mod:`prompture.infra.pricing` /
``rates/*.json`` (``cost: {input, output}``). Media pricing is shaped
differently — per image, per second of video, per character of TTS, or a flat
per-job charge — so it has its own curated source (``rates/_media.json``) and
its own quote function.

:func:`estimate_media_cost` is the ``calculate_dynamic_cost`` equivalent a studio
calls *before* running a node to show a price / enforce a budget, without
instantiating a driver or holding a key. Rates are extensible at runtime via
:func:`register_media_rate`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MEDIA_RATES_PATH = Path(__file__).parent / "media_rates.json"

_file_cache: dict[str, dict[str, Any]] | None = None
_runtime_rates: dict[str, dict[str, Any]] = {}

__all__ = [
    "estimate_media_cost",
    "get_media_rate",
    "load_media_pricing",
    "register_media_rate",
]


def _norm(model_str: str) -> str:
    return (model_str or "").strip().lower()


def load_media_pricing(*, reload: bool = False) -> dict[str, dict[str, Any]]:
    """Load (and cache) the curated media rate table from ``rates/_media.json``."""
    global _file_cache
    if _file_cache is None or reload:
        try:
            raw = json.loads(_MEDIA_RATES_PATH.read_text(encoding="utf-8"))
            _file_cache = {_norm(k): v for k, v in raw.items() if isinstance(v, dict) and not k.startswith("_")}
        except Exception:  # pragma: no cover - defensive
            logger.warning("Failed to load media pricing from %s", _MEDIA_RATES_PATH)
            _file_cache = {}
    return _file_cache


def register_media_rate(model_str: str, rate: dict[str, Any]) -> None:
    """Add or override a media rate at runtime (keyed by ``provider/model``)."""
    _runtime_rates[_norm(model_str)] = dict(rate)


def get_media_rate(model_str: str) -> dict[str, Any] | None:
    """Return the rate dict for ``provider/model`` (or a bare id), else ``None``.

    Runtime-registered rates take precedence over the file table. A bare id
    (no ``/``) matches any entry whose model segment equals it.
    """
    key = _norm(model_str)
    if key in _runtime_rates:
        return dict(_runtime_rates[key])
    table = load_media_pricing()
    if key in table:
        return dict(table[key])
    if "/" not in key:
        for k, v in {**table, **_runtime_rates}.items():
            if k.split("/", 1)[-1] == key:
                return dict(v)
    return None


def estimate_media_cost(
    model_str: str,
    params: dict[str, Any] | None = None,
    **overrides: Any,
) -> float:
    """Quote the USD cost of a media generation before running it.

    Recognized params (via *params* dict and/or keyword *overrides*):
    ``n`` / ``num_images`` (count), ``duration_seconds`` / ``duration`` (video),
    ``characters`` (TTS), ``resolution`` (selects a resolution-specific
    per-second rate when present). Returns ``0.0`` when no rate is known.
    """
    p: dict[str, Any] = {**(params or {}), **overrides}
    rate = get_media_rate(model_str)
    if not rate:
        return 0.0

    n = int(p.get("n", p.get("num_images", 1)) or 1)
    duration = float(p.get("duration_seconds", p.get("duration", 0)) or 0)
    characters = int(p.get("characters", 0) or 0)
    resolution = p.get("resolution")

    cost = 0.0
    per_sec_by_res = rate.get("per_second_by_resolution")
    if duration > 0 and resolution and isinstance(per_sec_by_res, dict) and resolution in per_sec_by_res:
        cost += float(per_sec_by_res[resolution]) * duration * n
    elif duration > 0 and "per_second" in rate:
        cost += float(rate["per_second"]) * duration * n
    if "per_image" in rate:
        cost += float(rate["per_image"]) * n
    if characters > 0 and "per_character" in rate:
        cost += float(rate["per_character"]) * characters
    if cost == 0.0 and "per_job" in rate:
        cost += float(rate["per_job"]) * n

    return round(cost, 6)
