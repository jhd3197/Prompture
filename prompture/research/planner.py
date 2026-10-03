"""Break a research question into sub-questions and search phrasings.

The planner asks the model for a small JSON plan (sub-questions, search
phrasings, relevant platforms and domain packs). When the model call fails
or returns something unusable, a keyword heuristic plan is used instead so a
run never stops at the planning step.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .budget import DepthPreset
from .tools import PACKS, PLATFORMS

#: ``(prompt, schema) -> (json_object, usage)``
AskJson = Callable[[str, dict[str, Any]], tuple[Any, dict[str, Any]]]

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sub_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "queries": {"type": "array", "items": {"type": "string"}},
                    "platforms": {"type": "array", "items": {"type": "string", "enum": list(PLATFORMS)}},
                },
                "required": ["question", "queries"],
            },
        },
        "packs": {"type": "array", "items": {"type": "string", "enum": list(PACKS)}},
    },
    "required": ["sub_questions"],
}

_PLATFORM_HINTS: dict[str, tuple[str, ...]] = {
    "github": (
        "github",
        "repo",
        "repository",
        "library",
        "open source",
        "open-source",
        "sdk",
        "framework",
        "package",
        "npm",
        "pypi",
        "crate",
        "issue tracker",
        "pull request",
    ),
    "hackernews": ("hacker news", "hackernews", "startup", "show hn", "developer opinion", "developers think"),
    "arxiv": (
        "paper",
        "papers",
        "arxiv",
        "research on",
        "study",
        "studies",
        "benchmark",
        "state of the art",
        "state-of-the-art",
        "algorithm",
        "neural",
        "model architecture",
    ),
    "youtube": ("video", "youtube", "talk", "keynote", "tutorial", "podcast", "interview", "lecture", "watch"),
}

_PACK_HINTS: dict[str, tuple[str, ...]] = {
    "finance": (
        "stock",
        "stocks",
        "share price",
        "market cap",
        "earnings",
        "ticker",
        "nasdaq",
        "nyse",
        "crypto",
        "bitcoin",
        "ethereum",
        "token price",
        "dividend",
        "revenue",
        "valuation",
    ),
    "news": ("news", "latest", "today", "this week", "breaking", "announced", "headline", "current events"),
    "dev": ("pypi", "npm", "package", "library", "release notes", "changelog", "github", "api"),
    "places": ("city", "country", "population", "located", "near ", "address", "capital of", "geography"),
}


def _contains(text: str, needle: str) -> bool:
    if needle.endswith(" "):
        return needle in text
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text) is not None


def suggest_platforms(text: str) -> list[str]:
    """Platforms whose keywords appear in *text*."""
    low = (text or "").lower()
    return [p for p, words in _PLATFORM_HINTS.items() if any(_contains(low, w) for w in words)]


def suggest_packs(text: str) -> list[str]:
    """Domain packs whose keywords appear in *text*."""
    low = (text or "").lower()
    return [p for p, words in _PACK_HINTS.items() if any(_contains(low, w) for w in words)]


@dataclass
class PlannedQuestion:
    """One sub-question with its search phrasings and platforms."""

    question: str
    queries: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "queries": list(self.queries), "platforms": list(self.platforms)}


@dataclass
class ResearchPlan:
    """Planner output.

    Attributes:
        sub_questions: Planned sub-questions (at least one).
        packs: Domain packs worth consulting.
        source: ``"llm"`` or ``"heuristic"``.
        usage: Usage of the planning call (empty for the heuristic plan).
        error: Why the heuristic plan was used, when it was.
    """

    sub_questions: list[PlannedQuestion]
    packs: list[str] = field(default_factory=list)
    source: str = "llm"
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def build_plan_prompt(question: str, preset: DepthPreset, *, today: date | None = None) -> str:
    today = today or date.today()
    return (
        "You plan web research. Break the user's question into focused sub-questions that together "
        "answer it, and give search-engine phrasings for each.\n\n"
        f"Today's date: {today.isoformat()}\n"
        f"Question: {question}\n\n"
        "Rules:\n"
        f"- At most {preset.sub_questions} sub-questions; fewer when the question is narrow.\n"
        f"- {preset.phrasings} distinct search queries per sub-question: short keyword queries, varied "
        "wording, include years or names when they matter.\n"
        "- platforms: only where they clearly help — github (code, libraries), hackernews (developer "
        "discussion), arxiv (papers), youtube (talks, videos). Otherwise an empty list.\n"
        "- packs: domain data tools worth consulting — finance (prices, markets), news (current events), "
        "dev (packages, repos), places (geography). Usually empty.\n"
        "Return JSON only."
    )


def heuristic_plan(question: str, preset: DepthPreset, *, error: str | None = None) -> ResearchPlan:
    """Plan without a model: the question itself plus simple phrasing variants."""
    q = " ".join(question.split())
    variants = [q]
    stripped = re.sub(r"^(what|who|when|where|why|how|which|is|are|does|do|can|should)\b\s*", "", q, flags=re.I)
    stripped = stripped.rstrip("?").strip()
    if stripped and stripped.lower() != q.lower():
        variants.append(stripped)
    variants.append(f"{stripped or q} {date.today().year}")
    variants.append(f"{stripped or q} explained")
    queries = list(dict.fromkeys(v for v in variants if v))[: max(1, preset.phrasings)]
    return ResearchPlan(
        sub_questions=[PlannedQuestion(question=q, queries=queries, platforms=suggest_platforms(q))],
        packs=suggest_packs(q)[:2],
        source="heuristic",
        error=error,
    )


def normalize_plan(raw: Any, question: str, preset: DepthPreset) -> ResearchPlan | None:
    """Validate and clamp a model-produced plan. ``None`` when nothing usable came back."""
    if isinstance(raw, list):
        raw = {"sub_questions": raw}
    if not isinstance(raw, dict):
        return None
    items = raw.get("sub_questions") or raw.get("subquestions") or raw.get("questions") or []
    if not isinstance(items, list):
        return None
    planned: list[PlannedQuestion] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, str):
            item = {"question": item}
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("question") or item.get("sub_question") or "").split())
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        queries_raw = item.get("queries") or item.get("search_queries") or []
        if isinstance(queries_raw, str):
            queries_raw = [queries_raw]
        queries = [" ".join(str(x).split()) for x in queries_raw if str(x).strip()]
        queries = list(dict.fromkeys(queries))[: max(1, preset.phrasings)] or [text]
        platforms_raw = item.get("platforms") or []
        if isinstance(platforms_raw, str):
            platforms_raw = [platforms_raw]
        platforms = [str(p).lower().replace(" ", "") for p in platforms_raw]
        platforms = [p for p in dict.fromkeys(platforms) if p in PLATFORMS]
        planned.append(PlannedQuestion(question=text, queries=queries, platforms=platforms))
        if len(planned) >= preset.sub_questions:
            break
    if not planned:
        return None
    packs_raw = raw.get("packs") or []
    if isinstance(packs_raw, str):
        packs_raw = [packs_raw]
    packs = [p for p in dict.fromkeys(str(x).lower() for x in packs_raw) if p in PACKS]
    for p in suggest_packs(question):
        if p not in packs:
            packs.append(p)
    return ResearchPlan(sub_questions=planned, packs=packs[:2], source="llm")


def plan_research(
    question: str,
    preset: DepthPreset,
    ask_json: AskJson | None,
    *,
    today: date | None = None,
) -> ResearchPlan:
    """Plan *question* with the model via *ask_json*; fall back to :func:`heuristic_plan`."""
    if ask_json is None:
        return heuristic_plan(question, preset, error="no planner model")
    try:
        raw, usage = ask_json(build_plan_prompt(question, preset, today=today), PLAN_SCHEMA)
    except Exception as exc:
        return heuristic_plan(question, preset, error=f"planning call failed: {type(exc).__name__}: {exc}")
    plan = normalize_plan(raw, question, preset)
    if plan is None:
        fallback = heuristic_plan(question, preset, error="planner returned no usable sub-questions")
        fallback.usage = dict(usage or {})
        return fallback
    plan.usage = dict(usage or {})
    return plan


__all__ = [
    "PLAN_SCHEMA",
    "PlannedQuestion",
    "ResearchPlan",
    "build_plan_prompt",
    "heuristic_plan",
    "normalize_plan",
    "plan_research",
    "suggest_packs",
    "suggest_platforms",
]
