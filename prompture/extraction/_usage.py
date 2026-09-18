"""Attach extraction outcomes and retry evidence to usage without extra LLM calls."""

from __future__ import annotations

import contextlib
import inspect
import uuid
from functools import wraps
from typing import Any

from pydantic import BaseModel

from ..infra.tracker import UsageEvent, _ctx_extraction, get_tracker


def extraction_attempt(*, retry_attempt: int | None = None, model_attempt: int | None = None) -> None:
    """Update task-local evidence at an actual retry or model fallback boundary."""
    current = _ctx_extraction.get()
    if current is None:
        return
    updated = dict(current)
    if retry_attempt is not None:
        updated["retry_attempt"] = retry_attempt
    if model_attempt is not None:
        updated["model_attempt"] = model_attempt
        updated["is_fallback"] = model_attempt > 0
        updated["retry_attempt"] = 0
    _ctx_extraction.set(updated)


@contextlib.contextmanager
def _extraction_scope():
    parent = _ctx_extraction.get()
    context = dict(parent) if parent else {"extraction_id": str(uuid.uuid4()), "retry_attempt": 0, "is_fallback": False}
    token = _ctx_extraction.set(context)
    outcome: dict[str, Any] = {"extraction_success": False}
    try:
        yield outcome
    finally:
        try:
            with contextlib.suppress(Exception):
                if parent is None:
                    # This is an outcome, never an inference call. Its zero cost cannot
                    # conceal the costs of failed attempts, which have their own events.
                    get_tracker().record(
                        UsageEvent(
                            model_name="extraction",
                            provider="",
                            operation="extraction_outcome",
                            status="success" if outcome["extraction_success"] else "error",
                            metadata={
                                **context,
                                **outcome,
                                "event_kind": "extraction_outcome",
                                "cost_status": "estimated",
                                "rates_available": True,
                                "usage_complete": True,
                            },
                        )
                    )
        finally:
            _ctx_extraction.reset(token)


def _outcome(result: Any, outcome: dict[str, Any]) -> None:
    if not isinstance(result, dict):
        return
    usage = result.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    failed_fields = any(
        isinstance(details, dict) and (details.get("used_default") or details.get("status") != "success")
        for details in (result.get("field_results") or {}).values()
    )
    fallback_used = bool(usage.get("fallback_used")) or failed_fields
    # Stepwise extraction can return partial/error data without raising, or a
    # validated model assembled from defaults after field extraction failures.
    # Neither is a successful extraction of the requested fields.
    validated = isinstance(result.get("model"), BaseModel)
    success = validated and not fallback_used and not result.get("error") and not usage.get("validation_errors")
    outcome.update(extraction_success=bool(success), fallback_used=fallback_used)
    usage["extraction_id"] = (_ctx_extraction.get() or {}).get("extraction_id")


def tracked_extraction(function):
    """Observe the outermost validated extraction, preserving public signatures."""
    if inspect.iscoroutinefunction(function):

        @wraps(function)
        async def asynchronous(*args, **kwargs):
            with _extraction_scope() as outcome:
                result = await function(*args, **kwargs)
                with contextlib.suppress(Exception):
                    _outcome(result, outcome)
                return result

        return asynchronous

    @wraps(function)
    def synchronous(*args, **kwargs):
        with _extraction_scope() as outcome:
            result = function(*args, **kwargs)
            with contextlib.suppress(Exception):
                _outcome(result, outcome)
            return result

    return synchronous
