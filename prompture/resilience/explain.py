"""Human-readable explanations of a resilient route (``meta["route"]``)."""

from __future__ import annotations

from typing import Any


def _describe_error(err: dict[str, Any]) -> str:
    parts = [err.get("category") or "error"]
    if err.get("status_code"):
        parts[0] += f" ({err['status_code']})"
    if err.get("retry_after") is not None:
        parts.append(f"retry after {err['retry_after']:g}s")
    action = err.get("action")
    if action:
        parts.append(
            {
                "retry": "retried",
                "cooldown": "parked",
                "disable_key": "key disabled",
                "failover": "skipped to next",
                "fatal": "request rejected",
            }.get(action, action)
        )
    return ", ".join(parts)


def explain_attempt(attempt: dict[str, Any]) -> str:
    """One line for one attempt record."""
    who = attempt.get("model", "?")
    if attempt.get("key_id"):
        who += f" (key {attempt['key_id']})"
    outcome = attempt.get("outcome")
    ms = f", {attempt['elapsed_ms']:.0f} ms" if attempt.get("elapsed_ms") is not None else ""
    if outcome == "ok":
        return f"{who} — ok{ms}"
    if outcome == "skipped":
        wait = attempt.get("available_in")
        return f"{who} — skipped, unavailable for {wait:.1f}s" if wait else f"{who} — skipped, circuit open"
    if outcome == "unsupported":
        return f"{who} — skipped, doesn't support this call"
    if outcome == "unavailable":
        err = attempt.get("error") or {}
        return f"{who} — couldn't be built: {err.get('message', 'unknown error')}"
    return f"{who} — {_describe_error(attempt.get('error') or {})}{ms}"


def explain_route(route: dict[str, Any] | None) -> str:
    """Multi-line explanation of how a call was routed.

    Accepts ``meta["route"]`` from a resilient driver response, a driver's
    ``last_route``, or an :class:`~.errors.AllTargetsFailedError`'s
    ``attempts`` wrapped as ``{"attempts": [...]}``.
    """
    if not route:
        return "No routing information (the call did not go through a resilient driver)."
    attempts = route.get("attempts") or []
    served = route.get("served_by")
    strategy = route.get("strategy")
    tried = sum(1 for a in attempts if a.get("outcome") in ("ok", "error"))
    if served and any(a.get("outcome") == "ok" for a in attempts):
        head = f"Served by {served}"
        if route.get("key_id"):
            head += f" (key {route['key_id']})"
        head += f" after {tried} attempt{'s' if tried != 1 else ''}"
    else:
        head = f"Failed after {tried} attempt{'s' if tried != 1 else ''}"
    if strategy and strategy != "priority":
        head += f" — strategy: {strategy}"
    lines = [head]
    lines += [f"  {i}. {explain_attempt(a)}" for i, a in enumerate(attempts, 1)]
    for note in route.get("deprioritized") or []:
        window = (note.get("window") or "rate limit").replace("_", " ")
        lines.append(f"  moved last: {note.get('target', '?')} — {note.get('headroom', 0):.0%} of {window} left")
    return "\n".join(lines)
