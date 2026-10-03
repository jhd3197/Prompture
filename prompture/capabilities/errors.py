"""Typed errors shared by every capability backend.

Each error carries a stable marker in its message (``unsafe_url``,
``response_too_large``, ``challenge_page``) and is registered with
:func:`prompture.resilience.classify_error`, so a :class:`BackendChain` can
decide between "try the next backend" and "stop" the same way drivers do.
"""

from __future__ import annotations

from typing import Any

from ..resilience.errors import ErrorAction, ErrorRule, register_error_rule
from ..security.redaction import scrub_secrets


class CapabilityError(Exception):
    """Base class for capability-layer failures. Messages are always scrubbed."""

    def __init__(self, message: str) -> None:
        super().__init__(scrub_secrets(message))


class UnsafeURLError(CapabilityError, ValueError):
    """The URL is not a public http(s) address (private IP, bad scheme, ...)."""

    def __init__(self, message: str) -> None:
        super().__init__(f"unsafe_url: {message}")


class ResponseTooLargeError(CapabilityError):
    """The response body exceeded the configured byte cap."""

    def __init__(self, message: str, *, limit: int | None = None) -> None:
        super().__init__(f"response_too_large: {message}")
        self.limit = limit


class ChallengePageError(CapabilityError):
    """The server returned a bot-challenge / captcha interstitial instead of content."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(f"challenge_page: {message}")
        self.challenge_status = status_code


class HTTPStatusError(CapabilityError):
    """A non-2xx HTTP response. ``status_code`` and ``headers`` feed ``classify_error``."""

    def __init__(self, message: str, *, status_code: int, headers: Any = None, url: str | None = None) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.headers = headers or {}
        self.url = scrub_secrets(url) if url else url


class BackendUnavailableError(CapabilityError):
    """A backend was asked to run but is not configured / installed."""


class AllBackendsFailedError(CapabilityError):
    """Every backend in a :class:`BackendChain` failed or was unavailable.

    ``attempts`` has the same shape as ``ChainResult.route["attempts"]``.
    """

    def __init__(self, message: str, *, attempts: list[dict[str, Any]], last_error: BaseException | None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


# Every backend receives the same URL, so an unsafe one stops the chain
# instead of being handed to the next (possibly third-party) backend.
_RULES = (
    ErrorRule("unsafe_url", ErrorAction.FATAL, pattern=r"\bunsafe_url:", priority=200),
    ErrorRule("response_too_large", ErrorAction.FAILOVER, pattern=r"\bresponse_too_large:", priority=200),
    ErrorRule("challenge_page", ErrorAction.FAILOVER, pattern=r"\bchallenge_page:", priority=200),
)


def register_capability_error_rules() -> None:
    """Register the capability error markers with ``classify_error`` (idempotent)."""
    from ..resilience.errors import _rules, _rules_lock

    with _rules_lock:
        present = {(r.category, r.pattern) for r in _rules}
    for rule in _RULES:
        if (rule.category, rule.pattern) not in present:
            register_error_rule(rule)


register_capability_error_rules()
