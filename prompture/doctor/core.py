"""``check_all`` and the :class:`DoctorReport` it returns.

The report walks every capability registered with
:mod:`prompture.capabilities.health` (providers, tools, media, MCP, binaries)
and renders it as a table for people or as a stable, versioned JSON document
(``prompture.doctor/1``) for agents and the companion.

Offline by default: checks inspect configuration, import packages and run
side-effect-free binary probes. Nothing is written, no daemon is started and
no network call is made unless ``live=True``.
"""

from __future__ import annotations

import platform
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..capabilities.health import (
    CATEGORIES,
    STATUS_SEVERITY,
    HealthStatus,
    list_capabilities,
    worst_status,
)
from . import providers as _providers

SCHEMA = "prompture.doctor/1"
SUMMARY_SCHEMA = "prompture.doctor.summary/1"
WATCH_SCHEMA = "prompture.watch/1"

#: Row statuses that mean something is actually broken (``prompture watch`` exits 1).
FAILING_STATUSES = frozenset({"broken", "error"})

#: ``prompture watch`` exit codes.
EXIT_OK = 0
EXIT_BROKEN = 1
EXIT_UPDATE = 2


def _normalize_only(only: str | Iterable[str] | None) -> set[str] | None:
    if only is None:
        return None
    wanted = {only} if isinstance(only, str) else set(only)
    wanted = {w.strip() for w in wanted if w and w.strip()}
    if not wanted:
        return None
    unknown = wanted - set(CATEGORIES)
    if unknown:
        raise ValueError(f"unknown doctor categories: {sorted(unknown)} (choose from {', '.join(CATEGORIES)})")
    return wanted


def _counts(rows: Iterable[HealthStatus]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (STATUS_SEVERITY.get(kv[0], 5), kv[0])))


def _version() -> str:
    from ..infra.updates import installed_version

    return installed_version()


@dataclass
class DoctorReport:
    """Every health row from one doctor run."""

    rows: list[HealthStatus]
    live: bool = False
    only: list[str] | None = None
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def worst(self) -> str:
        return worst_status(self.rows)

    @property
    def failing(self) -> list[HealthStatus]:
        """Rows whose status is ``broken`` or ``error``."""
        return [r for r in self.rows if r.status in FAILING_STATUSES]

    @property
    def ok(self) -> bool:
        """True when nothing is ``broken``/``error`` (missing optional pieces don't count)."""
        return not self.failing

    def by_category(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for cat in [*CATEGORIES, *sorted({r.category for r in self.rows} - set(CATEGORIES))]:
            rows = [r for r in self.rows if r.category == cat]
            if rows:
                out[cat] = {"worst": worst_status(rows), "counts": _counts(rows)}
        return out

    def summary(self) -> dict[str, Any]:
        return {"worst": self.worst, "ok": self.ok, "counts": _counts(self.rows), "by_category": self.by_category()}

    def to_dict(self) -> dict[str, Any]:
        """Stable ``prompture.doctor/1`` document.

        Keys: ``schema``, ``prompture_version``, ``python``, ``platform``,
        ``generated_at``, ``live``, ``only``, ``summary`` (``worst``, ``ok``,
        ``counts``, ``by_category``) and ``checks`` (each a
        :meth:`HealthStatus.to_dict`: ``name``, ``status``, ``category``,
        ``active_backend``, ``message``, ``fix_hint``, ``details``).
        Fields are only ever added within a schema version.
        """
        return {
            "schema": SCHEMA,
            "prompture_version": _version(),
            "python": platform.python_version(),
            "platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
            "generated_at": self.generated_at,
            "live": self.live,
            "only": self.only,
            "summary": self.summary(),
            "checks": [r.to_dict() for r in self.rows],
        }

    def to_table(self, *, width: int = 140) -> str:
        """Plain-text table: capability, status, active backend, message, fix."""
        if not self.rows:
            return "No capabilities registered."
        headers = ("CAPABILITY", "STATUS", "ACTIVE BACKEND", "MESSAGE", "FIX")
        name_w = min(28, max(len(headers[0]), *(len(r.name) for r in self.rows)))
        status_w = max(len(headers[1]), *(len(r.status) for r in self.rows))
        backend_w = min(28, max(len(headers[2]), *(len(r.active_backend or "-") for r in self.rows)))
        msg_w = max(24, min(60, width - name_w - status_w - backend_w - 30))

        def cut(text: str, n: int) -> str:
            text = " ".join(str(text).split())
            return text if len(text) <= n else text[: n - 3] + "..."

        def line(cols: tuple[str, ...]) -> str:
            return (
                f"{cut(cols[0], name_w):<{name_w}}  {cols[1]:<{status_w}}  "
                f"{cut(cols[2], backend_w):<{backend_w}}  {cut(cols[3], msg_w):<{msg_w}}  {cols[4]}"
            ).rstrip()

        out: list[str] = []
        for cat, info in self.by_category().items():
            out.append(f"[{cat}] {info['worst']}")
            out.append(line(headers))
            for r in (r for r in self.rows if r.category == cat):
                fix = "" if r.ok else (r.fix_hint or "")
                out.append(line((r.name, r.status, r.active_backend or "-", r.message or "", fix)))
            out.append("")
        counts = ", ".join(f"{n} {s}" for s, n in _counts(self.rows).items())
        mode = "live" if self.live else "offline"
        out.append(f"Overall: {self.worst} ({counts}) - {mode} check, prompture {_version()}")
        return "\n".join(out)


def check_all(
    live: bool = False,
    only: str | Iterable[str] | None = None,
    *,
    verbose: bool = False,
    timeout: float = _providers.LIVE_TIMEOUT_SECONDS,
) -> DoctorReport:
    """Run every registered capability check and return a :class:`DoctorReport`.

    Args:
        live: Allow real (cheap) network calls: provider model lists, tool
            queries, MCP initialize. Never a paid generation.
        only: Restrict to one or more categories
            (``providers | tools | media | mcp | binaries``).
        verbose: List every provider individually instead of collapsing the
            unconfigured ones into one summary row.
        timeout: Per-provider timeout for live calls, in seconds.

    Raises:
        ValueError: *only* names an unknown category.
    """
    wanted = _normalize_only(only)
    rows: list[HealthStatus] = []
    for cap in list_capabilities():
        if wanted and cap.category not in wanted:
            continue
        if cap.name == _providers.CAPABILITY_NAME and cap.check is _providers._check:
            try:
                rows.extend(_providers.provider_rows(live, verbose=verbose, timeout=timeout))
            except Exception as exc:  # same guarantee as Capability.run
                rows.append(
                    HealthStatus(cap.name, "error", category=cap.category, message=f"health check crashed: {exc}")
                )
            continue
        rows.extend(cap.run(live))
    rows.sort(key=lambda r: (_category_order(r.category), -STATUS_SEVERITY.get(r.status, 5), r.name))
    return DoctorReport(rows=rows, live=live, only=sorted(wanted) if wanted else None)


def _category_order(cat: str) -> int:
    return CATEGORIES.index(cat) if cat in CATEGORIES else len(CATEGORIES)


# ── Companion summary (cheap, cached, never blocks) ──────────────────────

_summary_lock = threading.Lock()
_summary_cache: dict[str, Any] = {"at": 0.0, "value": None, "refreshing": False}
SUMMARY_MAX_AGE = 60.0


def _summarize(report: DoctorReport) -> dict[str, Any]:
    return {
        "schema": SUMMARY_SCHEMA,
        "state": "ready",
        "worst": report.worst,
        "ok": report.ok,
        "counts": _counts(report.rows),
        "by_category": {cat: info["worst"] for cat, info in report.by_category().items()},
        "generated_at": report.generated_at,
    }


def _refresh_summary() -> dict[str, Any]:
    try:
        value = _summarize(check_all(live=False))
    except Exception as exc:
        value = {"schema": SUMMARY_SCHEMA, "state": "error", "error": f"{type(exc).__name__}: {exc}"}
    with _summary_lock:
        _summary_cache.update(at=time.monotonic(), value=value, refreshing=False)
    return value


def capabilities_summary(*, max_age: float = SUMMARY_MAX_AGE, wait: bool = True) -> dict[str, Any]:
    """Offline doctor summary (``prompture.doctor.summary/1``), cached for *max_age* seconds.

    With ``wait=False`` a stale or missing cache is refreshed in a background
    thread and the last known value (or ``{"state": "pending"}``) is returned
    immediately, so request handlers never block on probes.
    """
    with _summary_lock:
        value, at, refreshing = _summary_cache["value"], _summary_cache["at"], _summary_cache["refreshing"]
        fresh = value is not None and time.monotonic() - at < max_age
        if fresh:
            return dict(value)
        if not wait:
            if not refreshing:
                _summary_cache["refreshing"] = True
                threading.Thread(target=_refresh_summary, name="doctor-summary", daemon=True).start()
            return dict(value) if value is not None else {"schema": SUMMARY_SCHEMA, "state": "pending"}
    return dict(_refresh_summary())


def clear_summary_cache() -> None:
    with _summary_lock:
        _summary_cache.update(at=0.0, value=None, refreshing=False)


# ── Watch ────────────────────────────────────────────────────────────────


@dataclass
class WatchResult:
    """Outcome of :func:`run_watch`; ``exit_code`` follows the documented contract."""

    exit_code: int
    report: DoctorReport
    update: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": WATCH_SCHEMA,
            "exit_code": self.exit_code,
            "doctor": self.report.to_dict(),
            "update": self.update.to_dict() if self.update is not None else None,
        }


def watch_exit_code(report: DoctorReport, update: Any | None = None, *, fail_on_update: bool = False) -> int:
    """``1`` if any row is broken/error; else ``2`` if an update is available and
    *fail_on_update*; else ``0``. Offline update checks never fail the run."""
    if not report.ok:
        return EXIT_BROKEN
    if fail_on_update and update is not None and getattr(update, "update_available", False):
        return EXIT_UPDATE
    return EXIT_OK


def run_watch(
    *,
    update_check: bool = True,
    fail_on_update: bool = False,
    only: str | Iterable[str] | None = None,
    **update_kwargs: Any,
) -> WatchResult:
    """Offline doctor + update check in one quick run (for cron / scheduled tasks)."""
    report = check_all(live=False, only=only)
    update = None
    if update_check:
        from ..infra.updates import check_for_update

        update = check_for_update(**update_kwargs)
    return WatchResult(watch_exit_code(report, update, fail_on_update=fail_on_update), report, update)
