"""Multi-source web research with citations.

:class:`ResearchAgent` plans sub-questions, searches the web and platforms
(GitHub, Hacker News, arXiv, YouTube) in parallel, reads the best pages and
writes a Markdown answer whose ``[n]`` citations point only at pages it
actually opened — with conflicting claims, coverage gaps and the budget it
spent::

    from prompture.research import ResearchAgent

    report = ResearchAgent("openai/gpt-4o-mini", depth="quick").run("...")
    print(report.to_markdown())

Also available as a tool for other agents (:func:`research_tool`) and as
``prompture research "question"`` on the command line.
"""

from .agent import DEFAULT_MODEL_CANDIDATES, ResearchAgent, default_research_model, research_tool
from .budget import DEPTH_PRESETS, BudgetTracker, DepthPreset, ResearchBudget, get_depth_preset
from .events import ResearchEvent
from .planner import PlannedQuestion, ResearchPlan, heuristic_plan, plan_research
from .ranking import Candidate, CandidatePool, domain_of, normalize_url, select_diverse, strip_tracking
from .report import ReportSource, ResearchReport, SubQuestionResult
from .tools import PACKS, PLATFORMS, ResearchTools, ResearchToolUnavailable

__all__ = [
    "DEFAULT_MODEL_CANDIDATES",
    "DEPTH_PRESETS",
    "PACKS",
    "PLATFORMS",
    "BudgetTracker",
    "Candidate",
    "CandidatePool",
    "DepthPreset",
    "PlannedQuestion",
    "ReportSource",
    "ResearchAgent",
    "ResearchBudget",
    "ResearchEvent",
    "ResearchPlan",
    "ResearchReport",
    "ResearchToolUnavailable",
    "ResearchTools",
    "SubQuestionResult",
    "default_research_model",
    "domain_of",
    "get_depth_preset",
    "heuristic_plan",
    "normalize_url",
    "plan_research",
    "research_tool",
    "select_diverse",
    "strip_tracking",
]
