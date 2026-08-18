"""Shared usage recording for media driver hook wrappers.

Media calls bill per image / second / job rather than per token, so the text
drivers' ``_auto_record_usage`` doesn't fit — but the spend is just as real,
and until this helper existed it never reached the usage tracker: media cost
lived only on the immediate response ``meta`` and vanished unless the caller
kept it. Every media ``*_with_hooks`` wrapper now records one
:class:`~prompture.infra.tracker.UsageEvent` (zero tokens, ``cost`` from meta,
unit count in ``metadata``) through the same global tracker as text calls, so
sinks and the SQLite ledger see the whole picture.

Fire-and-forget, mirroring the text bases: recording must never break the
generation it measures.
"""

from __future__ import annotations

from typing import Any


def record_media_usage(
    driver: Any,
    meta: dict[str, Any] | None,
    elapsed_ms: float,
    *,
    modality: str,
    count_key: str,
    status: str = "success",
    error: Exception | None = None,
) -> None:
    """Record one media generation as a usage event. Never raises."""
    try:
        from ..infra.ledger import _resolve_api_key_hash
        from ..infra.tracker import UsageEvent, get_tracker

        tracker = get_tracker()
        if not tracker._enabled:
            return

        meta = meta or {}
        driver_name = str(meta.get("model_name") or getattr(driver, "model", driver.__class__.__name__))

        if "/" in driver_name:
            provider, model = driver_name.split("/", 1)
        else:
            # Same class-name fallback as the text bases, Async prefix stripped
            # so async twins attribute to the same provider.
            cls_name = driver.__class__.__name__.removeprefix("Async")
            provider = cls_name.removesuffix("Driver").lower()
            model = driver_name

        model_name = f"{provider}/{model}" if provider else model

        event = UsageEvent(
            model_name=model_name,
            provider=provider,
            api_key_hash=_resolve_api_key_hash(model_name),
            cost=float(meta.get("cost") or 0.0),
            elapsed_ms=elapsed_ms,
            status=status,
            error_type=type(error).__name__ if error else None,
            metadata={"modality": modality, "count": int(meta.get(count_key) or 0)},
        )
        tracker.record(event)
    except Exception:
        pass  # fire-and-forget
