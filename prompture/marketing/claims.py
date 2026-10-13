"""Catch made-up claims in marketing copy before it ships.

Models fill gaps with plausible inventions: a testimonial nobody gave, a
"2,000 happy clients" nobody counted, a phone number that rings somewhere else.
:class:`ClaimsChecker` checks copy against a :class:`~prompture.marketing.BrandContext`
in two passes:

1. **Rules, no model needed.**  Every URL, email address and phone number in
   the copy must be one the brand lists; words the brand never uses are flagged.
2. **Claim grounding (optional, needs a driver).**  The copy is broken into
   checkable factual claims (numbers, results, awards, client names,
   testimonials, guarantees) and each is verified against the brand's proof
   points with :class:`~prompture.eval.FaithfulnessEvaluator`.  Opinions and
   calls to action are not claims and are not checked.

Usage::

    from prompture.marketing import BrandContext, ClaimsChecker

    checker = ClaimsChecker(brand, model="openai/gpt-4o-mini")
    report = checker.check(email_body)
    if not report.passed:
        print(report.format())
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

from .context import BrandContext, load_brand

__all__ = ["CLAIM_DECOMPOSITION_PROMPT", "ClaimIssue", "ClaimsChecker", "ClaimsReport"]

#: Pulls out only the claims a reader could fact-check.  ``{answer}`` is substituted.
CLAIM_DECOMPOSITION_PROMPT = """List every checkable factual claim in the MARKETING COPY below.

A checkable claim states one of: a number, percentage or statistic; a result a customer got; a price, discount or \
guarantee; an award, ranking, certification or credential; years in business or number of customers; a named client \
or partner; a quote or testimonial attributed to a person; a fact about the business (location, team, founding).

Skip opinions, feelings, promises about how the reader will feel, questions, and calls to action.
Write each claim as one standalone sentence. Keep numbers and names exactly as written.

MARKETING COPY:
{answer}

Output a JSON array of strings with no other text. Output [] if there are no checkable claims.
"""

_URL_RE = re.compile(
    r"\b(?:https?://[^\s<>()\"']+|www\.[^\s<>()\"']+|[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\."
    r"(?:com|net|org|io|co|ai|app|dev|biz|info|us|uk|ca|au|shop|store|me)\b(?:/[^\s<>()\"']*)?)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_KEYPAD = {
    letter: digit
    for digit, letters in {
        "2": "ABC",
        "3": "DEF",
        "4": "GHI",
        "5": "JKL",
        "6": "MNO",
        "7": "PQRS",
        "8": "TUV",
        "9": "WXYZ",
    }.items()
    for letter in letters
}
_PHONE_RE = re.compile(r"(?<![\w.])(?:\+?\d[\d\s().-]{8,}\d|1[-\s.]?8\d\d[-\s.]?[A-Z0-9][A-Z0-9-]{3,}[A-Z0-9])(?![\w])")


@dataclass
class ClaimIssue:
    """One problem found in the copy.

    ``kind`` is one of ``unknown_url``, ``unknown_email``, ``unknown_phone``,
    ``avoid_word``, ``unsupported_claim`` or ``contradicted_claim``.
    """

    kind: str
    text: str
    detail: str = ""


@dataclass
class ClaimsReport:
    """Result of :meth:`ClaimsChecker.check`.

    Attributes:
        passed: ``True`` when no issue was found.
        issues: Everything that needs fixing or proof.
        supported_claims: Claims the brand's proof points back up.
        claims_checked: How many factual claims were verified (0 without a driver).
        grounding_ran: Whether the model-based pass ran.
        meta: Token and cost totals for the model calls.
    """

    passed: bool
    issues: list[ClaimIssue] = field(default_factory=list)
    supported_claims: list[str] = field(default_factory=list)
    claims_checked: int = 0
    grounding_ran: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def by_kind(self, kind: str) -> list[ClaimIssue]:
        return [issue for issue in self.issues if issue.kind == kind]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self) -> str:
        if self.passed:
            checked = f", {self.claims_checked} claim(s) verified" if self.grounding_ran else ""
            return f"Claims check passed{checked}."
        lines = [f"Claims check found {len(self.issues)} issue(s):"]
        lines.extend(
            f"- [{issue.kind}] {issue.text}" + (f": {issue.detail}" if issue.detail else "") for issue in self.issues
        )
        return "\n".join(lines)


class ClaimsChecker:
    """Check marketing copy against a brand's verified facts.

    Args:
        brand: A :class:`BrandContext`, a dict, or a path to a brand file.
        driver: Any LLM driver with ``generate(prompt, options)``.  Optional;
            without it (and without ``model``) only the rule pass runs.
        model: A ``"provider/model"`` string, used when ``driver`` is not given.
        allowed_domains: Extra domains the copy may mention (partner sites,
            booking tools).  The brand's website is always allowed.
        allowed_contacts: Extra phone numbers or emails the copy may mention.
        treat_silent_as: Passed to :class:`~prompture.eval.FaithfulnessEvaluator`.
            ``"unsupported"`` (default) flags any claim the proof points do not
            mention; ``"neutral"`` flags only contradicted claims.
        llm_options: Options for the model calls.
    """

    def __init__(
        self,
        brand: BrandContext | dict[str, Any] | str,
        driver: Any = None,
        *,
        model: str | None = None,
        allowed_domains: list[str] | None = None,
        allowed_contacts: list[str] | None = None,
        treat_silent_as: str = "unsupported",
        llm_options: dict[str, Any] | None = None,
    ) -> None:
        self.brand = load_brand(brand)
        if driver is None and model:
            from ..drivers import get_driver_for_model

            driver = get_driver_for_model(model)
        self.driver = driver
        self.treat_silent_as = treat_silent_as
        self.llm_options = llm_options

        known = [self.brand.website, *self.brand.contact.values(), *(allowed_contacts or [])]
        self._domains = {_domain(value) for value in [*known, *(allowed_domains or [])] if _domain(value)}
        self._emails = {match.lower() for value in known for match in _EMAIL_RE.findall(value or "")}
        self._phones = {_digits(match) for value in known for match in _PHONE_RE.findall(value or "")}

    def check(self, text: str, *, evidence: list[str] | None = None) -> ClaimsReport:
        """Check ``text``.  ``evidence`` adds facts beyond the brand's proof points (e.g. a cited study)."""
        issues = self._rule_issues(text)
        report = ClaimsReport(passed=False, issues=issues)

        if self.driver is not None:
            self._ground(text, [*self.brand.evidence(), *(evidence or [])], report)

        report.passed = not report.issues
        return report

    # ------------------------------------------------------------------
    # Rule pass
    # ------------------------------------------------------------------

    def _rule_issues(self, text: str) -> list[ClaimIssue]:
        issues: list[ClaimIssue] = []
        seen: set[tuple[str, str]] = set()

        def add(kind: str, value: str, detail: str) -> None:
            if (kind, value.lower()) not in seen:
                seen.add((kind, value.lower()))
                issues.append(ClaimIssue(kind=kind, text=value, detail=detail))

        emails = _EMAIL_RE.findall(text)
        for email in emails:
            if email.lower() not in self._emails:
                add("unknown_email", email, "not a contact address the brand lists")

        without_emails = _EMAIL_RE.sub(" ", text)
        for raw in _URL_RE.findall(without_emails):
            url = raw.rstrip(".,;:!?")
            domain = _domain(url)
            if domain and not any(domain == known or domain.endswith("." + known) for known in self._domains):
                add("unknown_url", url, "not the brand's website or an allowed domain")

        for raw in _PHONE_RE.findall(without_emails):
            phone = raw.strip()
            digits = _digits(phone)
            letters = re.sub(r"[^A-Za-z]", "", phone)
            if len(digits) < 10 and not letters:
                continue  # dates, ranges and short numbers
            if digits not in self._phones:
                add("unknown_phone", phone, "not a phone number the brand lists")

        for word in self.brand.avoid_words:
            if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text, re.IGNORECASE):
                add("avoid_word", word, "the brand never uses this word")
        return issues

    # ------------------------------------------------------------------
    # Grounding pass
    # ------------------------------------------------------------------

    def _ground(self, text: str, evidence: list[str], report: ClaimsReport) -> None:
        from ..eval import EvalError, FaithfulnessEvaluator

        evaluator = FaithfulnessEvaluator(
            self.driver,
            decomposition_prompt=CLAIM_DECOMPOSITION_PROMPT,
            llm_options=self.llm_options,
            treat_silent_as=self.treat_silent_as,
        )
        report.grounding_ran = True
        try:
            result = evaluator.evaluate(text, evidence)
        except EvalError as exc:
            if "no claims" in str(exc):
                return  # nothing checkable in the copy
            raise
        report.meta = result.meta
        report.supported_claims = list(result.supported)
        report.claims_checked = len(result.supported) + len(result.unsupported)
        contradicted = set(result.contradicted)
        for claim in result.unsupported:
            if claim in contradicted:
                report.issues.append(ClaimIssue("contradicted_claim", claim, "the brand's facts say otherwise"))
            else:
                report.issues.append(ClaimIssue("unsupported_claim", claim, "no proof point backs this up"))


def _domain(value: str) -> str:
    value = (value or "").strip()
    if not value or ("@" in value and "/" not in value):
        return ""
    if not re.match(r"^[a-z][a-z0-9+.-]*://", value, re.IGNORECASE):
        value = "http://" + value
    host = (urlparse(value).hostname or "").lower()
    if "." not in host:
        return ""
    return host[4:] if host.startswith("www.") else host


def _digits(value: str) -> str:
    """Digits of a phone number, with letters mapped to keypad digits and a leading 1 dropped."""
    digits = "".join(_KEYPAD.get(ch.upper(), ch) for ch in value or "" if ch.isdigit() or ch.upper() in _KEYPAD)
    return digits[1:] if len(digits) == 11 and digits.startswith("1") else digits
