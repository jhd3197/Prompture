"""Refusal detection for LLM responses.

The :class:`RefusalDetector` runs a fast, deterministic pipeline:

1. **Normalize** the response — lowercase, ASCII-quote, strip markdown
   emphasis, collapse whitespace.
2. **Match** against categorized marker phrases.  Both substring and
   word-boundary matching are supported; word-boundary avoids false
   positives like "I can" matching the start of "I cannot".
3. **Score** each match by category priority and position in the
   normalized text.  Refusals typically appear in the first sentence;
   later matches are downweighted.
4. **Aggregate** into a :class:`RefusalResult` with the decisive
   category, confidence in ``[0.0, 1.0]``, and the markers that fired.

The defaults ship English and Spanish markers (see
:mod:`prompture.refusal.markers`).  Pass ``custom_markers`` to extend
or replace them per detector instance.
"""

from __future__ import annotations

import enum
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

from .markers import DEFAULT_MARKERS


class RefusalCategory(str, enum.Enum):
    """Why a response was classified as a refusal.

    Categories are ordered from strongest to weakest signal — when
    multiple categories fire, the strongest wins.
    """

    HARD_REFUSAL = "hard_refusal"
    POLICY = "policy"
    SOFT_REFUSAL = "soft_refusal"
    DEFLECTION = "deflection"
    SAFETY_DISCLAIMER = "safety_disclaimer"
    EMPTY = "empty"


# Strongest → weakest. Used both for tie-breaking and for the
# per-category confidence base.
_CATEGORY_ORDER: tuple[RefusalCategory, ...] = (
    RefusalCategory.HARD_REFUSAL,
    RefusalCategory.POLICY,
    RefusalCategory.SOFT_REFUSAL,
    RefusalCategory.DEFLECTION,
    RefusalCategory.SAFETY_DISCLAIMER,
)

# Each category's baseline confidence when matched at position 0.
_CATEGORY_BASE_CONFIDENCE: dict[RefusalCategory, float] = {
    RefusalCategory.HARD_REFUSAL: 0.95,
    RefusalCategory.POLICY: 0.85,
    RefusalCategory.SOFT_REFUSAL: 0.70,
    RefusalCategory.DEFLECTION: 0.55,
    RefusalCategory.SAFETY_DISCLAIMER: 0.45,
}

# Categories that trigger ``is_refusal=True`` by default.  Safety
# disclaimers and deflections often accompany partial compliance, so
# they are reported with low confidence but don't tip the boolean.
_DEFAULT_REFUSAL_CATEGORIES: frozenset[RefusalCategory] = frozenset(
    {
        RefusalCategory.HARD_REFUSAL,
        RefusalCategory.POLICY,
        RefusalCategory.SOFT_REFUSAL,
        RefusalCategory.EMPTY,
    }
)


@dataclass(frozen=True)
class RefusalResult:
    """Outcome of running :class:`RefusalDetector` on a single response.

    Attributes:
        is_refusal: Boolean classification.  ``True`` when the
            decisive category is in the detector's refusal-trigger set
            (by default: hard refusal, policy, soft refusal, empty).
        confidence: ``[0.0, 1.0]`` score.  Higher means a stronger
            refusal signal.
        category: The decisive :class:`RefusalCategory` or ``None`` if
            no markers matched.
        matched_markers: All markers that fired, in order of first
            occurrence in the normalized text.
        position: Character offset of the earliest match in the
            normalized text, or ``-1`` if no match.
        normalized_text: The post-normalization text the detector
            scored against — useful for debugging mismatches.
    """

    is_refusal: bool
    confidence: float
    category: RefusalCategory | None
    matched_markers: list[str] = field(default_factory=list)
    position: int = -1
    normalized_text: str = ""

    def __bool__(self) -> bool:
        return self.is_refusal


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Markdown emphasis runs we strip before matching: **bold**, __bold__,
# *italic*, _italic_.  We keep the inner text.
_MD_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|_)(.+?)\1", flags=re.DOTALL)

# Run of whitespace (including newlines, tabs) → single space.
_WS_RE = re.compile(r"\s+")

# Stripped leading filler that sometimes prefixes a refusal:
#   "Sure, but I can't help with that."
#   "Okay — I cannot do that."
# We strip these so position-based scoring measures distance from the
# real start of the response.
_LEADING_FILLER_RE = re.compile(
    r"^(?:sure|okay|ok|well|certainly|of course|alright|"
    r"hmm|thanks|thank you|great question|good question|"
    r"claro|por supuesto|bueno|gracias)"
    r"[\s,.\-!:—]+",
    flags=re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Lowercase, ASCII-quote, strip emphasis, collapse whitespace."""
    if not text:
        return ""
    # Normalize unicode quote / dash variants to their ASCII forms.
    # NFKC folds e.g. "won’t" → "won't" (right single quotation mark
    # → apostrophe) and assorted dashes.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    # Strip markdown emphasis but keep the inner text.
    text = _MD_EMPHASIS_RE.sub(r"\2", text)
    text = text.lower()
    text = _WS_RE.sub(" ", text).strip()
    text = _LEADING_FILLER_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class RefusalDetector:
    """Score LLM responses for refusal behavior.

    Args:
        languages: Marker languages to load.  Defaults to ``["en", "es"]``.
            Unknown language codes are ignored with no error.
        custom_markers: Extra markers, structured as
            ``{category_name: [phrase, ...]}``.  Phrases should be in
            their already-normalized (lowercase, ASCII-quoted) form;
            the detector won't re-normalize the marker list at match
            time.  Categories must match :class:`RefusalCategory`
            values.
        replace_default_markers: When ``True``, ``custom_markers``
            replaces the defaults entirely instead of extending them.
            Default ``False``.
        position_threshold: A match at offset ``> position_threshold``
            in the normalized text is downweighted (real refusals
            almost always appear in the first sentence).  Default 200.
        position_decay: Per-character confidence multiplier for late
            matches, applied beyond ``position_threshold``.  Default
            ``0.4`` — a marker at the very end keeps ~40% of its base
            confidence.
        empty_is_refusal: Treat an empty / whitespace-only response as
            a refusal (with category :attr:`RefusalCategory.EMPTY` and
            confidence 1.0).  Default ``True``.
        empty_min_chars: A response shorter than this many characters
            after normalization counts as "empty".  Default 1.
        refusal_categories: The categories whose presence flips
            ``is_refusal=True``.  Defaults to hard refusal, policy,
            soft refusal, and empty.  Override to make the detector
            more or less strict — e.g. include ``DEFLECTION`` to also
            flag responses that change the subject.
        min_confidence: Floor below which a match doesn't flip
            ``is_refusal=True`` even if its category is in
            ``refusal_categories``.  Default 0.3.
    """

    def __init__(
        self,
        *,
        languages: Iterable[str] = ("en", "es"),
        custom_markers: dict[str, Iterable[str]] | None = None,
        replace_default_markers: bool = False,
        position_threshold: int = 200,
        position_decay: float = 0.4,
        empty_is_refusal: bool = True,
        empty_min_chars: int = 1,
        refusal_categories: Iterable[RefusalCategory] | None = None,
        min_confidence: float = 0.3,
    ) -> None:
        self.position_threshold = position_threshold
        self.position_decay = position_decay
        self.empty_is_refusal = empty_is_refusal
        self.empty_min_chars = max(0, empty_min_chars)
        self.min_confidence = min_confidence
        self.refusal_categories = frozenset(
            refusal_categories if refusal_categories is not None else _DEFAULT_REFUSAL_CATEGORIES
        )

        # Build per-category marker lists.
        markers: dict[RefusalCategory, list[str]] = {c: [] for c in _CATEGORY_ORDER}

        if not replace_default_markers:
            for lang in languages:
                bundle = DEFAULT_MARKERS.get(lang)
                if not bundle:
                    continue
                for cat_name, phrases in bundle.items():
                    cat = _try_category(cat_name)
                    if cat is not None:
                        markers[cat].extend(phrases)

        if custom_markers:
            for cat_name, phrases in custom_markers.items():
                cat = _try_category(cat_name)
                if cat is None:
                    raise ValueError(
                        f"Unknown refusal category: {cat_name!r}. "
                        f"Expected one of: {', '.join(c.value for c in _CATEGORY_ORDER)}"
                    )
                markers[cat].extend(_normalize(p) for p in phrases)

        # De-duplicate while keeping order, longest first so that
        # "i cannot help with that" beats "i cannot" when both fire.
        self._markers: dict[RefusalCategory, list[str]] = {
            cat: sorted(dict.fromkeys(phrases).keys(), key=len, reverse=True) for cat, phrases in markers.items()
        }

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def detect(self, response: str) -> RefusalResult:
        """Classify *response* as a refusal or not."""
        normalized = _normalize(response or "")

        # Empty handling first — short-circuit before scanning markers.
        if len(normalized) < self.empty_min_chars:
            return RefusalResult(
                is_refusal=self.empty_is_refusal,
                confidence=1.0 if self.empty_is_refusal else 0.0,
                category=RefusalCategory.EMPTY if self.empty_is_refusal else None,
                matched_markers=[],
                position=-1,
                normalized_text=normalized,
            )

        best_category: RefusalCategory | None = None
        best_score: float = 0.0
        best_position: int = -1
        all_matches: list[tuple[int, str]] = []

        for category in _CATEGORY_ORDER:
            for phrase in self._markers[category]:
                idx = normalized.find(phrase)
                if idx < 0:
                    continue
                if not _has_word_boundaries(normalized, idx, len(phrase)):
                    continue
                all_matches.append((idx, phrase))
                score = self._score(category, idx)
                if score > best_score:
                    best_score = score
                    best_category = category
                    best_position = idx

        # Order matches by occurrence in the text so the caller sees
        # them in reading order.
        all_matches.sort(key=lambda m: m[0])
        matched_phrases = [phrase for _, phrase in all_matches]

        if best_category is None:
            return RefusalResult(
                is_refusal=False,
                confidence=0.0,
                category=None,
                matched_markers=[],
                position=-1,
                normalized_text=normalized,
            )

        is_refusal = best_category in self.refusal_categories and best_score >= self.min_confidence

        return RefusalResult(
            is_refusal=is_refusal,
            confidence=round(best_score, 4),
            category=best_category,
            matched_markers=matched_phrases,
            position=best_position,
            normalized_text=normalized,
        )

    def is_refusal(self, response: str) -> bool:
        """Boolean shortcut equivalent to ``detect(response).is_refusal``."""
        return self.detect(response).is_refusal

    def detect_many(self, responses: Iterable[str]) -> list[RefusalResult]:
        """Detect over a batch.  Equivalent to a list comprehension."""
        return [self.detect(r) for r in responses]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score(self, category: RefusalCategory, position: int) -> float:
        base = _CATEGORY_BASE_CONFIDENCE[category]
        if position <= self.position_threshold:
            return base
        # Beyond the threshold, decay linearly toward
        # base * position_decay over the next 500 chars.
        overflow = position - self.position_threshold
        # full decay at +500 chars past the threshold
        factor = max(self.position_decay, 1.0 - (overflow / 500.0) * (1.0 - self.position_decay))
        return base * factor


def _try_category(name: str) -> RefusalCategory | None:
    try:
        return RefusalCategory(name)
    except ValueError:
        return None


def _has_word_boundaries(text: str, start: int, length: int) -> bool:
    """Return ``True`` when the substring at ``[start, start+length)``
    looks like a free-standing phrase (not embedded in a larger word).

    Heretic-style substring matching produces false positives such as
    ``"i can"`` matching the start of ``"i can not stand"``.  We
    require that the character before/after the match is either at
    the string edge or not a word character.
    """
    if start > 0:
        prev = text[start - 1]
        if prev.isalnum() or prev == "_":
            return False
    end = start + length
    if end < len(text):
        nxt = text[end]
        if nxt.isalnum() or nxt == "_":
            return False
    return True


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_detector: RefusalDetector | None = None


def is_refusal(response: str) -> bool:
    """Boolean refusal check using a shared default detector.

    Equivalent to ``RefusalDetector().is_refusal(response)`` but reuses
    a single detector instance across calls so marker lists aren't
    rebuilt every invocation.
    """
    global _default_detector
    if _default_detector is None:
        _default_detector = RefusalDetector()
    return _default_detector.is_refusal(response)
