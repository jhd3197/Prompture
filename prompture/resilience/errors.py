"""Error classification: turn any driver exception into a routing decision.

Drivers raise a zoo of exception types — :class:`DriverHTTPError`, raw
``requests`` / ``httpx`` errors, OpenAI / Anthropic SDK errors, or a
``RuntimeError`` wrapping one of those. :func:`classify_error` walks the
exception chain, extracts the HTTP status and any server-requested wait
(``Retry-After`` headers or "try again in 20s" text), and maps the result
onto an :class:`ErrorAction` through an ordered, extensible rule table.
"""

from __future__ import annotations

import email.utils
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..exceptions import DriverError


class ErrorAction(str, Enum):
    """What the router should do after a failed attempt."""

    RETRY = "retry"
    """Transient failure — back off and retry the same target."""

    COOLDOWN = "cooldown"
    """Target is rate limited or overloaded — park it and try the next one."""

    DISABLE_KEY = "disable_key"
    """Credential is unusable (auth, billing, exhausted quota) — park it for a long time."""

    FAILOVER = "failover"
    """This target can't serve this request (unknown model, context too long,
    unsupported feature) — try the next one without penalizing its health."""

    FATAL = "fatal"
    """The request itself is invalid — no target will fix it, raise immediately."""


@dataclass(frozen=True)
class ErrorInfo:
    """Normalized view of a failed driver call."""

    action: ErrorAction
    category: str
    status_code: int | None = None
    retry_after: float | None = None
    message: str = ""
    provider: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "category": self.category,
            "status_code": self.status_code,
            "retry_after": self.retry_after,
            "message": self.message,
        }


@dataclass(frozen=True)
class ErrorRule:
    """One row of the decision table.

    A rule matches when every field that is set matches: ``statuses``
    contains the status code, and ``pattern`` is found in the error text
    (case-insensitive). Rules are checked in priority order (highest first);
    the first match wins.
    """

    category: str
    action: ErrorAction
    pattern: str | None = None
    statuses: frozenset[int] | None = None
    priority: int = 0
    _regex: re.Pattern[str] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.pattern is not None:
            object.__setattr__(self, "_regex", re.compile(self.pattern, re.IGNORECASE))

    def matches(self, status: int | None, text: str) -> bool:
        if self.statuses is not None and status not in self.statuses:
            return False
        if self._regex is not None and not self._regex.search(text):
            return False
        return self.statuses is not None or self._regex is not None


class AllTargetsFailedError(DriverError):
    """Every target in a resilient route failed or was unavailable.

    ``attempts`` holds the per-attempt trace (same shape as
    ``meta["route"]["attempts"]`` on success); ``last_error`` is the final
    underlying exception, also chained as ``__cause__``.
    """

    def __init__(self, message: str, *, attempts: list[dict[str, Any]], last_error: BaseException | None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


# ---------------------------------------------------------------------------
# Decision table
# ---------------------------------------------------------------------------

_A = ErrorAction

# Text rules run before status rules so a 429 that says "insufficient_quota"
# is treated as an exhausted key, not a short rate limit.
_DEFAULT_RULES: list[ErrorRule] = [
    # Exhausted credit / quota looks like a 429 but won't clear in seconds.
    ErrorRule(
        "quota_exhausted",
        _A.DISABLE_KEY,
        pattern=r"insufficient[_ ]quota|quota (?:has been )?exceeded|exceeded your current quota|"
        r"credit balance is too low|out of credits|billing",
        priority=100,
    ),
    ErrorRule(
        "context_length",
        _A.FAILOVER,
        pattern=r"context[_ ]length|maximum context|context window|prompt is too long|too many tokens|"
        r"reduce the length|input is too long",
        priority=90,
    ),
    ErrorRule(
        "model_not_found",
        _A.FAILOVER,
        pattern=r"model[_ ]not[_ ]found|unknown model|no such model|model .{0,80}(?:does not exist|not found|"
        r"is not available|has been deprecated|decommissioned)|invalid model",
        priority=80,
    ),
    ErrorRule(
        "content_filter",
        _A.FAILOVER,
        pattern=r"content[_ ](?:filter|policy|management)|safety (?:system|filter)|flagged",
        priority=70,
    ),
    ErrorRule(
        "unsupported",
        _A.FAILOVER,
        pattern=r"does not support|not supported|unsupported|does not have access|not enabled for",
        priority=60,
    ),
    ErrorRule("overloaded", _A.COOLDOWN, pattern=r"overloaded|at capacity|server is busy", priority=50),
    ErrorRule("rate_limit", _A.COOLDOWN, pattern=r"rate[_ ]limit|too many requests", priority=40),
    # Status-only rules.
    ErrorRule("auth", _A.DISABLE_KEY, statuses=frozenset({401})),
    ErrorRule("billing", _A.DISABLE_KEY, statuses=frozenset({402})),
    ErrorRule("permission", _A.FAILOVER, statuses=frozenset({403})),
    ErrorRule("model_not_found", _A.FAILOVER, statuses=frozenset({404})),
    ErrorRule("timeout", _A.RETRY, statuses=frozenset({408})),
    ErrorRule("conflict", _A.RETRY, statuses=frozenset({409, 425})),
    ErrorRule("context_length", _A.FAILOVER, statuses=frozenset({413})),
    ErrorRule("rate_limit", _A.COOLDOWN, statuses=frozenset({429})),
    ErrorRule("overloaded", _A.COOLDOWN, statuses=frozenset({503, 529})),
    ErrorRule("server_error", _A.RETRY, statuses=frozenset({500, 502, 504, 520, 522, 524})),
    ErrorRule("bad_request", _A.FATAL, statuses=frozenset({400, 422})),
]

_rules: list[ErrorRule] = list(_DEFAULT_RULES)
_rules_lock = threading.Lock()


def register_error_rule(rule: ErrorRule) -> None:
    """Add a rule to the decision table (e.g. a provider-specific error string)."""
    with _rules_lock:
        _rules.append(rule)


def reset_error_rules() -> None:
    """Restore the built-in decision table."""
    with _rules_lock:
        _rules[:] = list(_DEFAULT_RULES)


def _sorted_rules() -> list[ErrorRule]:
    with _rules_lock:
        # Stable sort: equal priorities keep registration order.
        return sorted(_rules, key=lambda r: -r.priority)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

_STATUS_TEXT_RE = re.compile(r"(?:error code|status(?: code)?|http(?:/\d(?:\.\d)?)?)[:=\s]+([45]\d\d)\b", re.IGNORECASE)
# "429 Too Many Requests", "503 Server Error: Service Unavailable" (requests/httpx wording).
_STATUS_REASON_RE = re.compile(
    r"\b([45]\d\d)\s+(?:client error|server error|too many|bad |unauthori|payment|forbidden|not found|"
    r"request timeout|conflict|payload|unprocessable|internal|service unavailable|gateway|overloaded)",
    re.IGNORECASE,
)
_RETRY_TEXT_RE = re.compile(
    r"(?:try again|retry|retrying|wait|reset[s]?)\s+(?:in|after)\s+((?:\d+(?:\.\d+)?\s*(?:ms|h|m|s|seconds?|minutes?)\s*)+)",
    re.IGNORECASE,
)
_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s|seconds?|minutes?)", re.IGNORECASE)

_TIMEOUT_NAMES = ("Timeout", "TimeoutError", "ReadTimeout", "ConnectTimeout", "APITimeoutError")
_CONNECTION_NAMES = (
    "ConnectionError",
    "ConnectError",
    "APIConnectionError",
    "TransportError",
    "RemoteProtocolError",
    "ChunkedEncodingError",
    "ServerDisconnectedError",
)


def _chain(exc: BaseException, limit: int = 6) -> list[BaseException]:
    seen: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and cur not in seen and len(seen) < limit:
        seen.append(cur)
        cur = cur.__cause__ or cur.__context__
    return seen


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and 100 <= val < 600:
            return val
    resp = getattr(exc, "response", None)
    if resp is not None:
        for attr in ("status_code", "status"):
            val = getattr(resp, attr, None)
            if isinstance(val, int) and 100 <= val < 600:
                return val
    code = getattr(exc, "code", None)
    if isinstance(code, int) and 400 <= code < 600:
        return code
    text = str(exc)
    m = _STATUS_TEXT_RE.search(text) or _STATUS_REASON_RE.search(text)
    return int(m.group(1)) if m else None


def parse_duration(text: str) -> float | None:
    """Parse ``"20s"``, ``"1m30s"``, ``"250ms"``, ``"7.5 seconds"`` into seconds."""
    total = 0.0
    found = False
    for num, unit in _DURATION_PART_RE.findall(text):
        found = True
        value = float(num)
        unit = unit.lower()
        if unit == "ms":
            total += value / 1000
        elif unit == "h":
            total += value * 3600
        elif unit.startswith("m"):
            total += value * 60
        else:
            total += value
    return total if found else None


def _headers_of(exc: BaseException) -> Any:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    return headers if headers is not None else getattr(exc, "headers", None)


def _retry_after_from_headers(headers: Any) -> float | None:
    if not headers:
        return None
    try:
        get = headers.get
    except AttributeError:
        return None
    ms = get("retry-after-ms") or get("Retry-After-Ms")
    if ms:
        try:
            return float(ms) / 1000
        except ValueError:
            pass
    ra = get("retry-after") or get("Retry-After")
    if ra:
        try:
            return max(0.0, float(ra))
        except ValueError:
            try:
                dt = email.utils.parsedate_to_datetime(ra)
                return max(0.0, dt.timestamp() - time.time())
            except (TypeError, ValueError):
                pass
    for name in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens", "x-ratelimit-reset"):
        val = get(name)
        if val:
            parsed = parse_duration(str(val))
            if parsed is None:
                try:
                    parsed = float(val)
                except ValueError:
                    parsed = None
            if parsed is not None:
                return parsed
    return None


def _retry_after_of(exc: BaseException) -> float | None:
    val = getattr(exc, "retry_after", None)
    if isinstance(val, (int, float)) and val >= 0:
        return float(val)
    from_headers = _retry_after_from_headers(_headers_of(exc))
    if from_headers is not None:
        return from_headers
    m = _RETRY_TEXT_RE.search(str(exc))
    return parse_duration(m.group(1)) if m else None


def _type_names(exc: BaseException) -> list[str]:
    return [cls.__name__ for cls in type(exc).__mro__]


def _transport_category(exc: BaseException) -> str | None:
    names = _type_names(exc)
    if any(n in _TIMEOUT_NAMES for n in names) or isinstance(exc, TimeoutError):
        return "timeout"
    if any(n in _CONNECTION_NAMES for n in names) or isinstance(exc, ConnectionError):
        return "connection"
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def classify_error(exc: BaseException) -> ErrorInfo:
    """Classify *exc* into an :class:`ErrorInfo` with a routing action."""
    chain = _chain(exc)
    status: int | None = None
    retry_after: float | None = None
    provider: str | None = None
    transport: str | None = None
    for e in chain:
        if status is None:
            status = _status_of(e)
        if retry_after is None:
            retry_after = _retry_after_of(e)
        if provider is None:
            provider = getattr(e, "provider", None) if isinstance(getattr(e, "provider", None), str) else None
        if transport is None:
            transport = _transport_category(e)

    text = " | ".join(str(e) for e in chain if str(e))
    message = str(exc) or type(exc).__name__

    def info(category: str, action: ErrorAction) -> ErrorInfo:
        return ErrorInfo(action, category, status, retry_after, message, provider)

    # Local capability gaps and missing configuration never improve on retry.
    # DriverError subclasses NotImplementedError for historical reasons, so
    # only a *bare* NotImplementedError means "this driver lacks the feature".
    if any(isinstance(e, NotImplementedError) and not isinstance(e, DriverError) for e in chain):
        return info("unsupported", ErrorAction.FAILOVER)
    if any("ConfigurationError" in _type_names(e) for e in chain) and status is None:
        return info("config", ErrorAction.FAILOVER)

    for rule in _sorted_rules():
        if rule.matches(status, text):
            return info(rule.category, rule.action)

    if transport is not None:
        return info(transport, ErrorAction.RETRY)
    if status is not None:
        if status >= 500:
            return info("server_error", ErrorAction.RETRY)
        if status >= 400:
            return info("bad_request", ErrorAction.FATAL)
    if re.search(r"empty response|no content|incomplete", text, re.IGNORECASE):
        return info("empty_response", ErrorAction.RETRY)
    return info("unknown", ErrorAction.FAILOVER)
