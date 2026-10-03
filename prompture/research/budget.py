"""Depth presets and budget enforcement for :class:`ResearchAgent`.

A research run spends four things: page fetches, search calls, LLM tokens
(and their cost) and wall-clock time. :class:`ResearchBudget` holds the
caller's limits; anything left as ``None`` comes from the depth preset.
:class:`BudgetTracker` enforces them across the worker threads and produces
the ``budget_used`` block on the final report.

Limits are soft-landing: when one is hit, gathering stops and the agent
synthesizes with whatever it has already read. Time is split so synthesis
always keeps a reserve of the wall-clock budget.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

Depth = Literal["quick", "standard", "deep"]


@dataclass(frozen=True)
class DepthPreset:
    """Fan-out and budget defaults for one research depth.

    Attributes:
        sub_questions: Sub-questions the planner is asked for.
        phrasings: Search phrasings per sub-question.
        results_per_query: ``max_results`` passed to each search.
        fetches_per_question: Pages opened per sub-question.
        max_fetches: Total page fetches across the run.
        max_searches: Total search calls across the run.
        max_tokens: LLM token budget (prompt + completion).
        timeout_s: Wall-clock limit in seconds.
        source_chars: Characters of each opened page given to synthesis.
        answer_tokens: ``max_tokens`` requested for the synthesis call.
    """

    sub_questions: int
    phrasings: int
    results_per_query: int
    fetches_per_question: int
    max_fetches: int
    max_searches: int
    max_tokens: int
    timeout_s: float
    source_chars: int
    answer_tokens: int


DEPTH_PRESETS: dict[str, DepthPreset] = {
    "quick": DepthPreset(
        sub_questions=2,
        phrasings=2,
        results_per_query=5,
        fetches_per_question=2,
        max_fetches=5,
        max_searches=8,
        max_tokens=40_000,
        timeout_s=90.0,
        source_chars=3_000,
        answer_tokens=1_200,
    ),
    "standard": DepthPreset(
        sub_questions=4,
        phrasings=3,
        results_per_query=6,
        fetches_per_question=3,
        max_fetches=12,
        max_searches=20,
        max_tokens=120_000,
        timeout_s=240.0,
        source_chars=5_000,
        answer_tokens=2_000,
    ),
    "deep": DepthPreset(
        sub_questions=6,
        phrasings=4,
        results_per_query=8,
        fetches_per_question=5,
        max_fetches=30,
        max_searches=40,
        max_tokens=300_000,
        timeout_s=600.0,
        source_chars=8_000,
        answer_tokens=3_500,
    ),
}


def get_depth_preset(depth: str) -> DepthPreset:
    """Return the preset for *depth* (``quick``, ``standard`` or ``deep``)."""
    try:
        return DEPTH_PRESETS[depth.lower()]
    except (KeyError, AttributeError):
        valid = ", ".join(DEPTH_PRESETS)
        raise ValueError(f"Unknown research depth {depth!r}. Valid: {valid}") from None


@dataclass(frozen=True)
class ResearchBudget:
    """Caller limits for a research run. ``None`` means "use the depth preset".

    Attributes:
        max_fetches: Pages opened (``read_url`` / ``web_fetch`` / transcripts).
        max_searches: Search calls (web, platform and pack runs).
        max_tokens: LLM tokens across planning, pack runs and synthesis.
        max_cost: USD cost across all LLM calls. No preset default because
            pricing is unknown for some models; set it to cap spend.
        timeout_s: Wall-clock limit in seconds for the whole run.
        synthesis_reserve: Fraction of ``timeout_s`` kept for synthesis.
    """

    max_fetches: int | None = None
    max_searches: int | None = None
    max_tokens: int | None = None
    max_cost: float | None = None
    timeout_s: float | None = None
    synthesis_reserve: float = 0.3

    def resolve(self, preset: DepthPreset) -> ResearchBudget:
        """Fill unset limits from *preset*."""
        return replace(
            self,
            max_fetches=self.max_fetches if self.max_fetches is not None else preset.max_fetches,
            max_searches=self.max_searches if self.max_searches is not None else preset.max_searches,
            max_tokens=self.max_tokens if self.max_tokens is not None else preset.max_tokens,
            timeout_s=self.timeout_s if self.timeout_s is not None else preset.timeout_s,
        )


@dataclass
class BudgetTracker:
    """Thread-safe accounting for one run against a resolved :class:`ResearchBudget`."""

    budget: ResearchBudget
    started: float = field(default_factory=time.monotonic)
    fetches: int = 0
    searches: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    limits_hit: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- time ---------------------------------------------------------------

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining(self) -> float | None:
        """Seconds left in the whole run, or ``None`` when unlimited."""
        if self.budget.timeout_s is None:
            return None
        return max(0.0, self.budget.timeout_s - self.elapsed())

    def gather_remaining(self) -> float | None:
        """Seconds left for planning/search/fetch (synthesis keeps its reserve)."""
        if self.budget.timeout_s is None:
            return None
        reserve = self.budget.timeout_s * min(max(self.budget.synthesis_reserve, 0.0), 0.9)
        return max(0.0, self.budget.timeout_s - reserve - self.elapsed())

    def gather_time_left(self) -> bool:
        left = self.gather_remaining()
        if left is not None and left <= 0:
            self.hit("timeout")
            return False
        return True

    # -- counters -----------------------------------------------------------

    def hit(self, limit: str) -> None:
        with self._lock:
            if limit not in self.limits_hit:
                self.limits_hit.append(limit)

    def try_fetch(self) -> bool:
        """Reserve one fetch slot. ``False`` (and the limit recorded) when none left."""
        if not self.gather_time_left():
            return False
        with self._lock:
            cap = self.budget.max_fetches
            if cap is not None and self.fetches >= cap:
                if "max_fetches" not in self.limits_hit:
                    self.limits_hit.append("max_fetches")
                return False
            self.fetches += 1
            return True

    def try_search(self) -> bool:
        """Reserve one search slot. ``False`` (and the limit recorded) when none left."""
        if not self.gather_time_left():
            return False
        with self._lock:
            cap = self.budget.max_searches
            if cap is not None and self.searches >= cap:
                if "max_searches" not in self.limits_hit:
                    self.limits_hit.append("max_searches")
                return False
            self.searches += 1
            return True

    def record_usage(self, usage: dict[str, Any] | None) -> None:
        """Add one LLM call's usage (``prompt_tokens``, ``completion_tokens``, ``cost``)."""
        usage = usage or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        with self._lock:
            self.llm_calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += total
            self.cost += float(usage.get("cost") or 0.0)

    def llm_budget_left(self) -> bool:
        """``False`` (and the limit recorded) once tokens or cost are spent."""
        b = self.budget
        if b.max_tokens is not None and self.total_tokens >= b.max_tokens:
            self.hit("max_tokens")
            return False
        if b.max_cost is not None and self.cost >= b.max_cost:
            self.hit("max_cost")
            return False
        return True

    def tokens_remaining(self) -> int | None:
        if self.budget.max_tokens is None:
            return None
        return max(0, self.budget.max_tokens - self.total_tokens)

    # -- reporting ----------------------------------------------------------

    @property
    def degraded(self) -> bool:
        return bool(self.limits_hit)

    def usage(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": round(self.cost, 6),
        }

    def snapshot(self) -> dict[str, Any]:
        """The ``budget_used`` block: what was spent against which limit."""
        b = self.budget
        return {
            "fetches": self.fetches,
            "max_fetches": b.max_fetches,
            "searches": self.searches,
            "max_searches": b.max_searches,
            "llm_calls": self.llm_calls,
            "tokens": self.total_tokens,
            "max_tokens": b.max_tokens,
            "cost": round(self.cost, 6),
            "max_cost": b.max_cost,
            "elapsed_s": round(self.elapsed(), 2),
            "timeout_s": b.timeout_s,
            "limits_hit": list(self.limits_hit),
            "degraded": self.degraded,
        }


__all__ = [
    "DEPTH_PRESETS",
    "BudgetTracker",
    "Depth",
    "DepthPreset",
    "ResearchBudget",
    "get_depth_preset",
]
