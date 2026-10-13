"""Marketing building blocks: brand context, claims checking, and typed outputs.

* :class:`BrandContext` — the business facts every call starts from, loadable
  from YAML, JSON or a fill-in Markdown file, and accepted as ``brand=`` by
  ``Agent``, ``AsyncAgent``, ``Assistant`` and ``extract_with_model``.
* :class:`ClaimsChecker` — flags invented URLs, emails, phone numbers and
  unproven claims against the brand's verified facts.
* Output models with their limits enforced (:class:`SocialPostSet`,
  :class:`ShotList`, :class:`EmailSequence`, :class:`CoreMessaging`, ...).
* :func:`generate` / :func:`agenerate` — create one of those models for a task,
  with the brand in the system prompt.
"""

from __future__ import annotations

from .claims import CLAIM_DECOMPOSITION_PROMPT, ClaimIssue, ClaimsChecker, ClaimsReport
from .context import BrandColors, BrandContext, load_brand, merge_brand_system_prompt
from .generate import GENERATION_INSTRUCTION, agenerate, generate
from .models import (
    VOICEOVER_WORDS_PER_SECOND,
    ActionItem,
    AudienceMessage,
    CoreMessaging,
    Email,
    EmailSequence,
    IdealCustomer,
    LeadMagnet,
    LeadMagnetIdeas,
    MetricTrend,
    OfferItem,
    OfferStack,
    ScorecardAnalysis,
    ShotList,
    SocialPost,
    SocialPostSet,
    TopTens,
    VideoClip,
    WeekReview,
    analyze_metric,
    find_on_image_text,
    scorecard_brief,
    word_count,
)

__all__ = [
    "CLAIM_DECOMPOSITION_PROMPT",
    "GENERATION_INSTRUCTION",
    "VOICEOVER_WORDS_PER_SECOND",
    "ActionItem",
    "AudienceMessage",
    "BrandColors",
    "BrandContext",
    "ClaimIssue",
    "ClaimsChecker",
    "ClaimsReport",
    "CoreMessaging",
    "Email",
    "EmailSequence",
    "IdealCustomer",
    "LeadMagnet",
    "LeadMagnetIdeas",
    "MetricTrend",
    "OfferItem",
    "OfferStack",
    "ScorecardAnalysis",
    "ShotList",
    "SocialPost",
    "SocialPostSet",
    "TopTens",
    "VideoClip",
    "WeekReview",
    "agenerate",
    "analyze_metric",
    "find_on_image_text",
    "generate",
    "load_brand",
    "merge_brand_system_prompt",
    "scorecard_brief",
    "word_count",
]
