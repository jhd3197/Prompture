"""Health statuses and the capability registry that ``prompture doctor`` walks.

Every capability (a search backend chain, the STT pipeline, an MCP server, a
binary) registers a check function that returns one or more
:class:`HealthStatus` rows. Checks must be offline by default: with
``live=False`` they may inspect configuration, import packages and run
side-effect-free probes, but never call a remote service or write anything.
"""

from __future__ import annotations

import importlib
import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..security.redaction import scrub_secrets

logger = logging.getLogger("prompture.capabilities")

Status = Literal["ok", "degraded", "unconfigured", "missing", "broken", "timeout", "error", "skipped"]

# Status ordering for "worst of" summaries.
STATUS_SEVERITY: dict[str, int] = {
    "ok": 0,
    "skipped": 0,
    "unconfigured": 1,
    "degraded": 2,
    "missing": 3,
    "timeout": 4,
    "error": 5,
    "broken": 6,
}

CATEGORIES = ("providers", "tools", "media", "mcp", "binaries")


@dataclass
class HealthStatus:
    """One row of a health report.

    Attributes:
        name: What was checked (``"web_search"``, ``"ffmpeg"``, ``"openai"``).
        status: ``ok | degraded | unconfigured | missing | broken | timeout | error | skipped``.
        category: ``providers | tools | media | mcp | binaries`` (filled from
            the registering capability when left empty).
        active_backend: Backend that would serve a request right now.
        message: Human-readable one-liner.
        fix_hint: Exact next step (env var to set, extra to install, binary to get).
        details: Free-form structured detail (backend list, version, ...).
    """

    name: str
    status: Status
    category: str = ""
    active_backend: str | None = None
    message: str = ""
    fix_hint: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.message = scrub_secrets(self.message or "")
        if self.fix_hint:
            self.fix_hint = scrub_secrets(self.fix_hint)

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "skipped")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CheckFn = Callable[[bool], "HealthStatus | Iterable[HealthStatus]"]


@dataclass
class Capability:
    """A registered health check."""

    name: str
    category: str
    check: CheckFn
    description: str = ""

    def run(self, live: bool = False) -> list[HealthStatus]:
        try:
            result = self.check(live)
        except Exception as exc:  # a broken check must not break doctor
            logger.debug("capability check %s failed", self.name, exc_info=True)
            return [
                HealthStatus(
                    self.name,
                    "error",
                    category=self.category,
                    message=f"health check crashed: {type(exc).__name__}: {exc}",
                )
            ]
        rows = [result] if isinstance(result, HealthStatus) else list(result)
        for row in rows:
            if not row.category:
                row.category = self.category
        return rows


_registry: dict[str, Capability] = {}
_lock = threading.Lock()

# Modules that register built-in capabilities on import. Missing optional
# modules are skipped so a partial install still produces a report.
BUILTIN_CAPABILITY_MODULES: list[str] = [
    "prompture.capabilities.builtin_checks",
    "prompture.tools.web.health",
    "prompture.media.understand.health",
    "prompture.mcp.health",
    "prompture.tools.cli.health",
    "prompture.tools.packs",
]
_loaded_builtins = False


def register_capability(
    name: str,
    category: str,
    check: CheckFn,
    *,
    description: str = "",
    replace: bool = True,
) -> Capability:
    """Register a health check under *name* (replaces an existing one by default)."""
    cap = Capability(name, category, check, description)
    with _lock:
        if not replace and name in _registry:
            return _registry[name]
        _registry[name] = cap
    return cap


def unregister_capability(name: str) -> None:
    with _lock:
        _registry.pop(name, None)


def load_builtin_capabilities() -> None:
    """Import the built-in capability modules once (idempotent)."""
    global _loaded_builtins
    if _loaded_builtins:
        return
    _loaded_builtins = True
    for mod in BUILTIN_CAPABILITY_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError as exc:
            logger.debug("capability module %s not available: %s", mod, exc)


def list_capabilities(category: str | None = None) -> list[Capability]:
    load_builtin_capabilities()
    with _lock:
        caps = list(_registry.values())
    return [c for c in caps if category is None or c.category == category]


def check_capabilities(
    *,
    live: bool = False,
    only: str | Iterable[str] | None = None,
) -> list[HealthStatus]:
    """Run every registered check (optionally only some categories)."""
    wanted = {only} if isinstance(only, str) else set(only) if only else None
    rows: list[HealthStatus] = []
    for cap in list_capabilities():
        if wanted and cap.category not in wanted:
            continue
        rows.extend(cap.run(live))
    return rows


def worst_status(rows: Iterable[HealthStatus]) -> str:
    """Return the most severe status in *rows* (``ok`` when empty)."""
    worst = "ok"
    for row in rows:
        if STATUS_SEVERITY.get(row.status, 5) > STATUS_SEVERITY.get(worst, 0):
            worst = row.status
    return worst
