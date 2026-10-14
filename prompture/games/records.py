"""Turn records for the game harness: what was asked, what came back, what it cost.

Every model call a :class:`~prompture.games.base.GameHarness` makes becomes an
:class:`~prompture.games.base.Attempt`; every turn becomes a :class:`TurnRecord`
attached to its :class:`~prompture.games.base.MoveDecision`. Records are built
from known fields only (never provider objects, headers or environment), so a
record can be written to a trace without leaking credentials.

* :class:`FailureKind` and :func:`classify_failure` separate invalid model output
  from configuration, rate-limit, transport, timeout, cancellation and budget
  failures, using :func:`prompture.resilience.errors.classify_error`.
* :class:`Accounting` totals usage while keeping its completeness honest: a call
  whose cost is unknown is counted as unknown, never as an observed zero.
* :func:`fingerprint` hashes a harness configuration so changed behavior gets a
  new identity while renamed players do not.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..exceptions import PromptureError

RECORD_VERSION = "1"


class FailureKind(str, Enum):
    """Why an attempt or turn failed."""

    INVALID_OUTPUT = "invalid_output"
    """The model answered, but not with a usable legal choice."""
    AUTH = "auth"
    """Credentials, billing or permissions: retrying will not help."""
    CONFIGURATION = "configuration"
    """Unknown model, unsupported feature, bad request or missing setup."""
    RATE_LIMIT = "rate_limit"
    UPSTREAM = "upstream"
    """The provider failed (5xx, overloaded, empty or incomplete reply)."""
    TRANSPORT = "transport"
    """The connection failed or a stream broke."""
    TIMEOUT = "timeout"
    """A request timed out, or the turn's deadline expired."""
    CANCELLED = "cancelled"
    BUDGET = "budget"

    @property
    def transient(self) -> bool:
        """Whether the same request may succeed if sent again later."""
        return self in _TRANSIENT

    @property
    def fatal(self) -> bool:
        """Whether no retry can succeed (fail fast)."""
        return self in (FailureKind.AUTH, FailureKind.CONFIGURATION)


_TRANSIENT = {FailureKind.RATE_LIMIT, FailureKind.UPSTREAM, FailureKind.TRANSPORT, FailureKind.TIMEOUT}
_CATEGORY_KIND = {
    "auth": FailureKind.AUTH,
    "billing": FailureKind.AUTH,
    "quota_exhausted": FailureKind.AUTH,
    "permission": FailureKind.AUTH,
    "config": FailureKind.CONFIGURATION,
    "model_not_found": FailureKind.CONFIGURATION,
    "unsupported": FailureKind.CONFIGURATION,
    "bad_request": FailureKind.CONFIGURATION,
    "context_length": FailureKind.CONFIGURATION,
    "content_filter": FailureKind.INVALID_OUTPUT,
    "rate_limit": FailureKind.RATE_LIMIT,
    "overloaded": FailureKind.UPSTREAM,
    "server_error": FailureKind.UPSTREAM,
    "empty_response": FailureKind.UPSTREAM,
    "conflict": FailureKind.UPSTREAM,
    "timeout": FailureKind.TIMEOUT,
    "connection": FailureKind.TRANSPORT,
}


_TIMEOUT_TEXT = re.compile(r"timed out|timeout", re.I)
_TRANSPORT_TEXT = re.compile(
    r"connection (?:reset|refused|aborted|closed)|stream ended without a completion summary|broken pipe", re.I
)


@dataclass(frozen=True)
class FailureInfo:
    """A classified failure: its kind plus whatever the provider told us."""

    kind: FailureKind
    category: str
    message: str
    status_code: int | None = None
    retry_after: float | None = None
    error_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "category": self.category,
            "message": redact_text(self.message),
            "status_code": self.status_code,
            "retry_after": self.retry_after,
            "error_type": self.error_type,
        }


def classify_failure(exc: BaseException) -> FailureInfo:
    """Classify a failed model call. Unknown errors count as upstream (transient) failures.

    The cause chain, status code and ``Retry-After`` hints are read by
    :func:`~prompture.resilience.errors.classify_error`; string matching is only
    its last resort.
    """
    from ..extraction.streaming import StreamStopped
    from ..resilience.errors import classify_error

    if isinstance(exc, StreamStopped):
        return FailureInfo(FailureKind.CANCELLED, "stopped", str(exc), error_type=type(exc).__name__)
    if isinstance(exc, TurnError):
        return FailureInfo(exc.kind, exc.kind.value, str(exc), error_type=type(exc).__name__)
    if "BudgetExceededError" in [cls.__name__ for cls in type(exc).__mro__]:
        return FailureInfo(FailureKind.BUDGET, "budget", str(exc), error_type=type(exc).__name__)
    info = classify_error(exc)
    kind = _CATEGORY_KIND.get(info.category)
    if kind is None:
        # Last resort for drivers that wrap transport errors in plain text.
        text = str(exc)
        if _TIMEOUT_TEXT.search(text):
            kind = FailureKind.TIMEOUT
        elif _TRANSPORT_TEXT.search(text):
            kind = FailureKind.TRANSPORT
        else:
            kind = FailureKind.CONFIGURATION if info.action.value == "fatal" else FailureKind.UPSTREAM
    return FailureInfo(kind, info.category, info.message, info.status_code, info.retry_after, type(exc).__name__)


# -- errors -------------------------------------------------------------------


class TurnError(PromptureError, RuntimeError):
    """A turn ended without a move. ``record`` holds every attempt made."""

    kind: FailureKind = FailureKind.UPSTREAM

    def __init__(self, message: str, *, kind: FailureKind | None = None, record: TurnRecord | None = None) -> None:
        super().__init__(message)
        if kind is not None:
            self.kind = kind
        self.record = record


class TurnCancelled(TurnError):
    """The turn was cancelled. No move was chosen; none may be played on its behalf."""

    kind = FailureKind.CANCELLED


class TurnFailed(TurnError):
    """The turn failed under a ``forfeit`` or ``stop`` policy, or on a fatal error.

    ``forfeit`` says whether the side should lose the game (strict policy) rather
    than the game simply stopping.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: FailureKind | None = None,
        record: TurnRecord | None = None,
        forfeit: bool = False,
    ) -> None:
        super().__init__(message, kind=kind, record=record)
        self.forfeit = forfeit


class BudgetExhausted(TurnFailed):
    """A call, token, cost or time budget would be exceeded by another request."""

    kind = FailureKind.BUDGET


# -- accounting ---------------------------------------------------------------

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")


def source_meta(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """The driver's own metadata behind an extraction ``usage`` dict.

    Extraction fills in ``cost: 0.0`` when a driver reports no price; the
    driver's original ``meta`` (kept on ``raw_response``) says whether a price
    was really observed.
    """
    if not usage:
        return usage
    raw = usage.get("raw_response")
    if isinstance(raw, dict) and isinstance(raw.get("meta"), dict):
        return raw["meta"]
    return usage


def usage_status(meta: dict[str, Any] | None) -> tuple[str, str]:
    """``(usage_status, cost_status)`` for one call's metadata.

    Usage is ``complete``, ``partial`` or ``unknown``. Cost keeps the driver's own
    ``cost_status`` (``estimated``/``partial``/``unknown``); a bare number with no
    provenance is ``reported``; no number is ``unknown``.
    """
    if not meta:
        return "unknown", "unknown"
    has_tokens = any(isinstance(meta.get(k), (int, float)) for k in USAGE_KEYS[:3])
    if meta.get("usage_complete") is False:
        status = "partial" if has_tokens and meta.get("total_tokens") else "unknown"
    else:
        status = "complete" if has_tokens else "unknown"
    cost_status = meta.get("cost_status")
    if cost_status is None:
        cost_status = "reported" if isinstance(meta.get("cost"), (int, float)) and status != "unknown" else "unknown"
    return status, str(cost_status)


@dataclass
class Accounting:
    """Usage totals plus how complete they are.

    ``tokens``/``cost`` sum what was observed. ``calls`` counts every request
    sent; ``unknown_usage_calls`` and ``unknown_cost_calls`` count requests whose
    usage or price is not known, so a total is only exact when both are zero.
    """

    calls: int = 0
    prompt_tokens: float = 0.0
    completion_tokens: float = 0.0
    total_tokens: float = 0.0
    cost: float = 0.0
    partial_usage_calls: int = 0
    unknown_usage_calls: int = 0
    unknown_cost_calls: int = 0
    partial_cost_calls: int = 0
    abandoned_calls: int = 0
    cost_sources: dict[str, int] = field(default_factory=dict)

    def add(self, usage: dict[str, Any], status: str, cost_status: str, *, abandoned: bool = False) -> None:
        self.calls += 1
        for key in USAGE_KEYS:
            setattr(self, key, getattr(self, key) + float(usage.get(key) or 0))
        if status == "partial":
            self.partial_usage_calls += 1
        elif status == "unknown":
            self.unknown_usage_calls += 1
        if cost_status == "unknown":
            self.unknown_cost_calls += 1
        elif cost_status == "partial":
            self.partial_cost_calls += 1
        self.cost_sources[cost_status] = self.cost_sources.get(cost_status, 0) + 1
        if abandoned:
            self.abandoned_calls += 1

    def merge(self, other: Accounting) -> None:
        for name in (
            "calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost",
            "partial_usage_calls",
            "unknown_usage_calls",
            "unknown_cost_calls",
            "partial_cost_calls",
            "abandoned_calls",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for source, count in other.cost_sources.items():
            self.cost_sources[source] = self.cost_sources.get(source, 0) + count

    @property
    def usage_complete(self) -> bool:
        return not (self.partial_usage_calls or self.unknown_usage_calls)

    @property
    def cost_complete(self) -> bool:
        return not (self.unknown_cost_calls or self.partial_cost_calls)

    @property
    def cost_label(self) -> str:
        """``"$0.0123"`` when exact; ``">= $0.0123 (2 calls unpriced)"`` otherwise."""
        if self.cost_complete:
            return f"${self.cost:.4f}"
        missing = self.unknown_cost_calls + self.partial_cost_calls
        return f">= ${self.cost:.4f} ({missing} call{'s' if missing != 1 else ''} not fully priced)"

    def legacy(self) -> dict[str, float]:
        """The historical ``MoveDecision.usage`` dict (observed sums only)."""
        return {key: getattr(self, key) for key in USAGE_KEYS}

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            **self.legacy(),
            "usage_complete": self.usage_complete,
            "cost_complete": self.cost_complete,
            "partial_usage_calls": self.partial_usage_calls,
            "unknown_usage_calls": self.unknown_usage_calls,
            "unknown_cost_calls": self.unknown_cost_calls,
            "partial_cost_calls": self.partial_cost_calls,
            "abandoned_calls": self.abandoned_calls,
            "cost_sources": dict(self.cost_sources),
        }


# -- redaction ----------------------------------------------------------------

_SECRET_KEY = re.compile(
    r"(api[_-]?key|secret|password|authorization|auth[_-]?token|access[_-]?token|refresh[_-]?token|bearer|"
    r"header|cookie|credential|^token$|^key$|^auth$)",
    re.I,
)
_SECRET_TEXT = [
    re.compile(r"\b(sk|pk|rk|gsk|xai|key)-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{8,}", re.I),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*[=:]\s*['\"]?[^\s'\",;]{6,}"),
]
REDACTED = "[REDACTED]"
# Options that change behavior and are safe to record. Anything else is dropped.
OPTION_ALLOWLIST = (
    "effort",
    "reasoning_effort",
    "reasoning",
    "thinking",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "max_output_tokens",
    "max_completion_tokens",
    "seed",
    "think",
    "num_ctx",
    "num_predict",
    "verbosity",
)


def redact_text(text: str) -> str:
    """``text`` with credential-shaped substrings replaced."""
    for pattern in _SECRET_TEXT:
        text = pattern.sub(REDACTED, text)
    return text


def safe_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """The allowlisted, JSON-safe subset of driver options."""
    kept: dict[str, Any] = {}
    for key in OPTION_ALLOWLIST:
        if options and key in options:
            kept[key] = _json_safe(options[key], 500)
    return kept


def _json_safe(value: Any, limit: int) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return truncate(redact_text(value), limit)
    if isinstance(value, dict):
        return {
            str(k): (REDACTED if _SECRET_KEY.search(str(k)) else _json_safe(v, limit))
            for k, v in list(value.items())[:50]
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, limit) for v in list(value)[:200]]
    return truncate(redact_text(type(value).__name__), limit)


def truncate(text: str | None, limit: int) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text) - limit} chars truncated]"


# -- fingerprints --------------------------------------------------------------


@dataclass(frozen=True)
class Fingerprint:
    """A short digest of a behavior configuration and the configuration itself."""

    digest: str
    config: dict[str, Any]

    def __str__(self) -> str:
        return self.digest

    def differences(self, other: Fingerprint) -> list[str]:
        """Top-level configuration keys whose values differ."""
        keys = set(self.config) | set(other.config)
        return sorted(k for k in keys if self.config.get(k) != other.config.get(k))


def fingerprint(config: dict[str, Any]) -> Fingerprint:
    """Hash a JSON-safe behavior configuration (secrets and display names must not be in it)."""
    clean = _json_safe(config, 10_000)
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)
    return Fingerprint(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], clean)


def text_digest(text: str | None) -> str | None:
    """A short digest of text that should be identified but not stored (a system prompt)."""
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# -- turn records ---------------------------------------------------------------


@dataclass
class TurnRecord:
    """Everything about one turn: identity, attempts, outcome and accounting.

    ``outcome`` is ``accepted`` (a model-selected move), ``rescued`` (a marked
    harness substitute), ``failed`` or ``cancelled``. ``attempts`` are the same
    objects as ``MoveDecision.attempts``.
    """

    run_id: str
    game_id: str
    turn_id: str
    side: str
    ply: int | None
    state_id: str | None
    fingerprint: str
    model: str
    outcome: str = "pending"
    move: str | None = None
    failure: FailureInfo | None = None
    attempts: list[Any] = field(default_factory=list)
    accounting: Accounting = field(default_factory=Accounting)
    started_at: float = 0.0
    elapsed_ms: float = 0.0
    deadline_s: float | None = None
    strategy: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def model_attempts(self) -> int:
        return len(self.attempts)

    def to_dict(self, *, include_prompts: bool = True, max_field_chars: int = 20_000) -> dict[str, Any]:
        return {
            "record_version": RECORD_VERSION,
            "run_id": self.run_id,
            "game_id": self.game_id,
            "turn_id": self.turn_id,
            "side": self.side,
            "ply": self.ply,
            "state_id": self.state_id,
            "fingerprint": self.fingerprint,
            "model": self.model,
            "outcome": self.outcome,
            "move": self.move,
            "failure": self.failure.to_dict() if self.failure else None,
            "started_at": self.started_at,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "deadline_s": self.deadline_s,
            "accounting": self.accounting.to_dict(),
            "strategy": _json_safe(self.strategy, max_field_chars),
            "context": _json_safe(self.context, max_field_chars),
            "notes": [redact_text(n) for n in self.notes],
            "attempts": [
                a.to_dict(include_prompt=include_prompts, max_field_chars=max_field_chars) for a in self.attempts
            ],
        }
