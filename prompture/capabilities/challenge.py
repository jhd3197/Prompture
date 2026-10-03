"""Detect bot-challenge and captcha interstitials.

A single marker is weak evidence (plenty of real pages embed a reCAPTCHA
widget), so detection scores combined markers in the first 4 KB and asks
for more evidence when the status code looks successful.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCAN_BYTES = 4096

# Markers that only show up on interstitials.
_STRONG = (
    "cf_chl_opt",
    "challenge-platform",
    "cf-browser-verification",
    "<title>just a moment...</title>",
    "attention required! | cloudflare",
    "captcha-delivery.com",
    "_incapsula_resource",
    "px-captcha",
    "ddos protection by",
    "/cdn-cgi/challenge",
)

# Markers that also appear on normal pages.
_WEAK = (
    "checking your browser",
    "enable javascript and cookies to continue",
    "please enable cookies",
    "cf-turnstile",
    "g-recaptcha",
    "h-captcha",
    "hcaptcha.com",
    "verify you are human",
    "are you a robot",
    "unusual traffic",
    "access denied",
)

_CHALLENGE_STATUSES = frozenset({403, 429, 503})


def challenge_score(text: str | bytes) -> int:
    """Return the combined marker score of the first :data:`SCAN_BYTES` of *text*."""
    if isinstance(text, bytes):
        head = text[:SCAN_BYTES].decode("utf-8", errors="ignore")
    else:
        head = text[:SCAN_BYTES]
    head = head.lower()
    return sum(2 for m in _STRONG if m in head) + sum(1 for m in _WEAK if m in head)


def is_challenge_page(
    text: str | bytes,
    *,
    status: int | None = None,
    headers: Mapping[str, Any] | None = None,
) -> bool:
    """Return ``True`` when *text* looks like a Cloudflare / captcha interstitial.

    ``cf-mitigated: challenge`` is conclusive. Otherwise a challenge status
    (403/429/503) needs a score of 2 (one strong marker or two weak ones)
    and any other status needs 3.
    """
    if headers:
        lowered = {str(k).lower(): str(v).lower() for k, v in headers.items()}
        if lowered.get("cf-mitigated") == "challenge":
            return True
    score = challenge_score(text)
    threshold = 2 if status in _CHALLENGE_STATUSES else 3
    return score >= threshold
