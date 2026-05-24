"""PII redaction for prompts and responses.

:class:`PIIRedactor` scrubs personal data from text using a curated
set of regex patterns, with a Luhn-check on candidate credit-card
matches to keep false positives down.  Categories covered:

* ``EMAIL`` — RFC-5322-ish, no IDN.
* ``PHONE`` — E.164 / NANP / common international formats.
* ``CREDIT_CARD`` — 13-19 digit Visa/MC/Amex/Discover/JCB candidates
  validated with the Luhn algorithm.
* ``SSN`` — US SSN ``NNN-NN-NNNN``.
* ``IBAN`` — international bank account numbers, length-checked.
* ``IPV4`` and ``IPV6`` addresses.
* ``API_KEY`` — common cloud-key shapes (``sk-...``, ``AKIA...``,
  ``ghp_...``, ``xoxb-...``).
* ``URL_CREDENTIALS`` — ``https://user:pass@host`` style.

Treat this as defense-in-depth, not a compliance solution.  Tune
``categories`` for your threat model and combine with downstream
auditing.

Quick start::

    from prompture.security import PIIRedactor

    redactor = PIIRedactor()
    result = redactor.redact(
        "Contact alice@example.com, card 4111 1111 1111 1111."
    )
    print(result.text)
    # 'Contact [EMAIL], card [CREDIT_CARD].'
    print(result.counts)
    # {'EMAIL': 1, 'CREDIT_CARD': 1}
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


class PIICategory(str, enum.Enum):
    """The kinds of PII the redactor knows about."""

    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    SSN = "SSN"
    IBAN = "IBAN"
    IPV4 = "IPV4"
    IPV6 = "IPV6"
    API_KEY = "API_KEY"
    URL_CREDENTIALS = "URL_CREDENTIALS"


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}"
    # Allow a trailing sentence period — don't require a non-domain
    # character to follow.  We just disallow a TLD continuation like
    # ``[A-Za-z0-9-]`` that would imply a longer TLD.
    r"(?![A-Za-z0-9-])"
)

# Loose international phone: optional +, 7–15 digits with grouping
# punctuation.  Trailing word boundary to avoid swallowing into IDs.
_PHONE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:\+?\d{1,3}[\s.\-]?)?"
    r"(?:\(?\d{2,4}\)?[\s.\-]?)?"
    r"\d{3,4}[\s.\-]?\d{3,4}"
    r"(?![A-Za-z0-9])"
)

# Candidate card numbers: 13-19 digits with optional spaces/dashes.
# We validate Luhn before redacting.
_CARD_CANDIDATE_RE = re.compile(
    r"(?<![\d-])"
    r"(?:\d[ \-]?){12,18}\d"
    r"(?![\d-])"
)

_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")

# IBAN: 2 letters + 2 digits + 11-30 alnum.  Length-validated below.
_IBAN_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{2}[A-Z0-9]{11,30}(?![A-Z0-9])")

_IPV4_RE = re.compile(
    r"(?<!\d)"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d?\d)){3}"
    r"(?!\d)"
)

_IPV6_RE = re.compile(
    r"(?<![0-9A-Fa-f:])"
    r"(?:"
    r"[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4}){7}"
    r"|"
    r"(?:[0-9A-Fa-f]{1,4}:){1,7}:(?:[0-9A-Fa-f]{1,4})?"
    r")"
    r"(?![0-9A-Fa-f:])"
)

# A small but practical list of well-known key shapes.
_API_KEY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("anthropic", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("google_aip", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("aws_access", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_pat", re.compile(r"\bghp_[A-Za-z0-9]{30,}\b")),
    ("github_oauth", re.compile(r"\bgho_[A-Za-z0-9]{30,}\b")),
    ("slack_bot", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("stripe_live", re.compile(r"\b(?:sk|pk|rk)_live_[A-Za-z0-9]{20,}\b")),
    ("stripe_test", re.compile(r"\b(?:sk|pk|rk)_test_[A-Za-z0-9]{20,}\b")),
)

# https://user:pass@host
_URL_CREDS_RE = re.compile(
    r"\b(?:https?|ftp|s3|ssh)://"
    r"[^\s:/@]+:[^\s/@]+@"
    r"[^\s]+",
    re.IGNORECASE,
)


def _luhn_valid(digits: str) -> bool:
    """Return True if *digits* (ASCII numerals only) passes the Luhn check.

    Luhn doubles every second digit counting from the right (i.e. the
    digit immediately left of the check digit, two positions left of
    it, and so on).  Translated to left-to-right iteration that's:
    double when ``i`` has the same parity as ``len(digits) - 2``.
    """
    if len(digits) < 13:
        return False
    total = 0
    parity = (len(digits) - 2) % 2
    for i, ch in enumerate(digits):
        if not ch.isdigit():
            return False
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_length_valid(candidate: str) -> bool:
    """Quick length check for IBAN — country-specific lengths range
    from 15 (Norway) to 34 chars; we accept the broad range."""
    return 15 <= len(candidate) <= 34


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PIIMatch:
    """A single PII match inside an input string."""

    category: PIICategory
    start: int
    end: int
    original: str


@dataclass
class RedactionResult:
    """Outcome of :meth:`PIIRedactor.redact`.

    Attributes:
        text: Redacted text with placeholders substituted in.
        matches: Every :class:`PIIMatch` we found, in left-to-right
            order.
        counts: ``{category_value: n}`` count per category.
    """

    text: str
    matches: list[PIIMatch] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def has_pii(self) -> bool:
        return bool(self.matches)


# ---------------------------------------------------------------------------
# Redactor
# ---------------------------------------------------------------------------


class PIIRedactor:
    """Scrub PII from text.

    Args:
        categories: Which categories to scrub.  Defaults to all.
        placeholder: A callable returning the replacement string for a
            given category.  Default: ``lambda cat: f"[{cat.value}]"``.
            Pass a custom callable to use, e.g., hashed placeholders or
            preserved suffixes (``"[EMAIL:****@example.com]"``).
        custom_patterns: Additional ``{category: list[re.Pattern]}``
            entries.  Custom categories may be passed as strings; they
            are wrapped into :class:`PIICategory`-like dynamic members
            represented as ``str`` placeholders in the output.
        skip_luhn: When ``True``, do not Luhn-check credit-card
            candidates — flag any 13-19 digit run.  Default ``False``.
    """

    DEFAULT_CATEGORIES: frozenset[PIICategory] = frozenset(PIICategory)

    def __init__(
        self,
        *,
        categories: Iterable[PIICategory] | None = None,
        placeholder: Callable[[Any], str] | None = None,
        custom_patterns: dict[str, Iterable[re.Pattern[str]]] | None = None,
        skip_luhn: bool = False,
    ) -> None:
        self.categories = frozenset(categories) if categories is not None else self.DEFAULT_CATEGORIES
        self._placeholder = placeholder or (lambda cat: f"[{cat.value if hasattr(cat, 'value') else cat}]")
        self.skip_luhn = skip_luhn
        # custom_patterns keyed by category-name string; resolved at
        # match time against either a built-in PIICategory or a raw
        # string label.
        self._custom_patterns: dict[str, list[re.Pattern[str]]] = {
            name: list(patterns) for name, patterns in (custom_patterns or {}).items()
        }

    def redact(self, text: str) -> RedactionResult:
        """Scrub PII from *text* and return the rewritten result."""
        if not text:
            return RedactionResult(text=text or "")

        matches = self.find(text)
        if not matches:
            return RedactionResult(text=text)

        # Replace from end to start so offsets stay valid.
        out_chars = list(text)
        for m in sorted(matches, key=lambda m: m.start, reverse=True):
            placeholder = self._placeholder(m.category)
            out_chars[m.start : m.end] = placeholder
        rewritten = "".join(out_chars)

        counts: dict[str, int] = {}
        for m in matches:
            key = m.category.value if isinstance(m.category, PIICategory) else str(m.category)
            counts[key] = counts.get(key, 0) + 1
        return RedactionResult(text=rewritten, matches=matches, counts=counts)

    def find(self, text: str) -> list[PIIMatch]:
        """Return all PII matches in *text* without rewriting it."""
        spans: list[PIIMatch] = []
        # Order matters for overlap handling: URL_CREDENTIALS first
        # so a password embedded in a URL doesn't get re-classified
        # as an API key, then API_KEY before generic patterns, then
        # the rest.
        if PIICategory.URL_CREDENTIALS in self.categories:
            spans.extend(self._scan(text, _URL_CREDS_RE, PIICategory.URL_CREDENTIALS))
        if PIICategory.API_KEY in self.categories:
            for _, pat in _API_KEY_PATTERNS:
                spans.extend(self._scan(text, pat, PIICategory.API_KEY))
        if PIICategory.EMAIL in self.categories:
            spans.extend(self._scan(text, _EMAIL_RE, PIICategory.EMAIL))
        if PIICategory.IPV4 in self.categories:
            spans.extend(self._scan(text, _IPV4_RE, PIICategory.IPV4))
        if PIICategory.IPV6 in self.categories:
            spans.extend(self._scan(text, _IPV6_RE, PIICategory.IPV6))
        if PIICategory.SSN in self.categories:
            spans.extend(self._scan(text, _SSN_RE, PIICategory.SSN))
        if PIICategory.IBAN in self.categories:
            for m in _IBAN_RE.finditer(text):
                candidate = m.group(0)
                if _iban_length_valid(candidate):
                    spans.append(PIIMatch(PIICategory.IBAN, m.start(), m.end(), candidate))
        if PIICategory.CREDIT_CARD in self.categories:
            for m in _CARD_CANDIDATE_RE.finditer(text):
                candidate = m.group(0)
                digits = re.sub(r"[ \-]", "", candidate)
                if self.skip_luhn or _luhn_valid(digits):
                    spans.append(PIIMatch(PIICategory.CREDIT_CARD, m.start(), m.end(), candidate))
        if PIICategory.PHONE in self.categories:
            # Run phone last; we'll dedupe against earlier hits.
            spans.extend(self._scan(text, _PHONE_RE, PIICategory.PHONE))

        # Custom patterns, after built-ins.
        for cat_name, patterns in self._custom_patterns.items():
            for pat in patterns:
                for m in pat.finditer(text):
                    spans.append(
                        PIIMatch(
                            PIICategory(cat_name) if cat_name in PIICategory._value2member_map_ else cat_name,  # type: ignore[arg-type]
                            m.start(),
                            m.end(),
                            m.group(0),
                        )
                    )

        # Resolve overlaps — keep the longest match per overlapping
        # region; ties go to whichever was added first (earlier in
        # priority order above).
        return _resolve_overlaps(spans)

    def _scan(
        self,
        text: str,
        pattern: re.Pattern[str],
        category: PIICategory,
    ) -> list[PIIMatch]:
        return [PIIMatch(category, m.start(), m.end(), m.group(0)) for m in pattern.finditer(text)]


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------


def _resolve_overlaps(spans: list[PIIMatch]) -> list[PIIMatch]:
    """Drop overlapping matches, preferring the earlier (higher-
    priority) and longer span."""
    if not spans:
        return []
    sorted_spans = sorted(
        enumerate(spans),
        key=lambda item: (item[1].start, -(item[1].end - item[1].start), item[0]),
    )
    kept: list[PIIMatch] = []
    last_end = -1
    for _, span in sorted_spans:
        if span.start >= last_end:
            kept.append(span)
            last_end = span.end
        else:
            # Overlap — keep the larger; if equal, keep the earlier-
            # priority (i.e. the one already in ``kept``).
            existing = kept[-1]
            existing_len = existing.end - existing.start
            new_len = span.end - span.start
            if new_len > existing_len:
                kept[-1] = span
                last_end = span.end
    return kept


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_redactor: PIIRedactor | None = None


def redact_pii(text: str) -> str:
    """Quick-and-dirty redact using a shared default :class:`PIIRedactor`.

    Returns just the rewritten text — call :meth:`PIIRedactor.redact`
    directly when you need the full :class:`RedactionResult`.
    """
    global _default_redactor
    if _default_redactor is None:
        _default_redactor = PIIRedactor()
    return _default_redactor.redact(text).text
