"""Composable, measurable execution strategies over Prompture's building blocks.

This package is the composition layer described in the adaptive-execution
roadmap.  It adds nothing to the extraction, agent, retrieval, or workflow
engines — it adapts them behind one contract so a caller can say *what* the task
is, hand over a typed output contract and resource limits, and get back a result
that is validated, fully accounted, and able to explain itself.

Layers
------

``outcomes`` / ``types``
    The durable record of one executed task, and the request/result envelope
    every strategy speaks.
``strategies``
    The three fixed strategies — ``direct``, ``retrieve_verify``,
    ``draft_critique`` — each an adapter over machinery that already exists.
``policy``
    **Experimental** bounded adaptive selection, measured routing, explicit
    escalation rules, and abstention.  Opt-in; never a default.
``improve``
    Offline improvement: inert versioned candidates, an optimizer adapter that
    only ever sees the development split, dev/held-out scorecards, and explicit
    promotion with a retained baseline and rollback.
``integration``
    ``Assistant`` entry points, a ``"strategy"`` workflow node type, and
    side-by-side strategy comparison.
``context`` / ``context_eval``
    Shared context allocation, selective skill and tool loading, scoped
    artifacts for large results, safe compaction, and the measurements that
    stop "fewer tokens" from being mistaken for "better".
``fixtures`` / ``scoring`` / ``gates`` / ``harness`` / ``sandbox``
    Versioned dev and held-out evaluation data, deterministic scorers,
    thresholds declared before tuning, a reproducible benchmark runner, and a
    local tool world so tool tasks can be graded on state rather than prose.

What this package deliberately does not do
------------------------------------------

* It does not replace existing return types.  ``extract_with_model``,
  ``RAGPipeline.query``, ``AsyncReviewLoop.arun`` and friends keep their
  signatures and their results; strategies wrap them and keep the underlying
  result on :attr:`~prompture.execution.types.ExecutionResult.raw`.
* It does not treat schema validity, model agreement, or reviewer approval as
  evidence of factual correctness.  Those are four separate fields.
* It does not report an unknown price as ``$0.00``.  See
  :class:`~prompture.execution.outcomes.UsageAccounting`.

Quick start::

    from pydantic import BaseModel
    from prompture.execution import ExecutionRequest, ResourceLimits
    from prompture.execution.strategies import DirectStrategy

    class Contact(BaseModel):
        name: str | None = None
        email: str | None = None

    result = DirectStrategy(model="openai/gpt-4o-mini").run(
        ExecutionRequest(
            task="Ping Ana at ana@example.test about the renewal.",
            output_model=Contact,
            limits=ResourceLimits(max_cost_usd=0.01, max_llm_calls=3),
        )
    )
    print(result.output, result.termination.value)
    print(result.explain())
"""

from __future__ import annotations

from . import improve
from .context import (
    ArtifactRef,
    ArtifactStore,
    CompactionResult,
    ContextAssembly,
    ContextPolicy,
    ContextSection,
    SkillCatalog,
    ToolCatalog,
    compact_messages,
    count_tokens,
    supports_native_tool_discovery,
)
from .context_eval import (
    CompactionReport,
    SelectionCase,
    ToolSelectionReport,
    measure_compaction,
    measure_tool_selection,
)
from .fixtures import (
    BUNDLED_FIXTURE_SETS,
    FixtureCase,
    FixtureSet,
    Split,
    assert_optimization_safe,
    load_bundled_fixture_set,
    load_fixture_set,
    save_fixture_set,
)
from .gates import (
    DEFAULT_GATES,
    CriterionResult,
    GateEvaluation,
    GateVerdict,
    WorkloadGate,
    evaluate_gate,
    gate_for_workload,
)
from .harness import (
    BenchmarkReport,
    CaseRun,
    WorkloadSummary,
    build_request,
    run_fixture_set,
    score_result,
    summarize_runs,
)
from .integration import (
    compare_strategies,
    register_strategy_node,
    request_from_assistant,
    resolve_strategy_instance,
)
from .outcomes import (
    EvidenceReport,
    JsonlOutcomeStore,
    OutcomeRecord,
    TaskCategory,
    TerminationReason,
    UsageAccounting,
    ValidationReport,
)
from .policy import (
    AdaptivePolicy,
    EscalationRule,
    MeasurementKey,
    MeasurementStore,
    PerformanceStats,
    SelectionDecision,
    SelectionRule,
    shadow_compare,
)
from .sandbox import FaultPlan, InjectedFailure, InventoryWorld, ToolCallRecord
from .scoring import CaseScore, score_case, score_document_qa, score_extraction, score_tool_task
from .strategies import (
    DirectStrategy,
    DraftAndCritiqueStrategy,
    ExecutionStrategy,
    RetrieveAndVerifyStrategy,
    StrategyError,
    get_strategy,
    list_strategies,
    register_strategy,
)
from .types import (
    BudgetLedger,
    CancellationToken,
    Cancelled,
    EvidencePassage,
    ExecutionRequest,
    ExecutionResult,
    ResourceLimits,
    StepRecord,
)

__all__ = [
    "BUNDLED_FIXTURE_SETS",
    "DEFAULT_GATES",
    "AdaptivePolicy",
    "ArtifactRef",
    "ArtifactStore",
    "BenchmarkReport",
    "BudgetLedger",
    "CancellationToken",
    "Cancelled",
    "CaseRun",
    "CaseScore",
    "CompactionReport",
    "CompactionResult",
    "ContextAssembly",
    "ContextPolicy",
    "ContextSection",
    "CriterionResult",
    "DirectStrategy",
    "DraftAndCritiqueStrategy",
    "EscalationRule",
    "EvidencePassage",
    "EvidenceReport",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionStrategy",
    "FaultPlan",
    "FixtureCase",
    "FixtureSet",
    "GateEvaluation",
    "GateVerdict",
    "InjectedFailure",
    "InventoryWorld",
    "JsonlOutcomeStore",
    "MeasurementKey",
    "MeasurementStore",
    "OutcomeRecord",
    "PerformanceStats",
    "ResourceLimits",
    "RetrieveAndVerifyStrategy",
    "SelectionCase",
    "SelectionDecision",
    "SelectionRule",
    "SkillCatalog",
    "Split",
    "StepRecord",
    "StrategyError",
    "TaskCategory",
    "TerminationReason",
    "ToolCallRecord",
    "ToolCatalog",
    "ToolSelectionReport",
    "UsageAccounting",
    "ValidationReport",
    "WorkloadGate",
    "WorkloadSummary",
    "assert_optimization_safe",
    "build_request",
    "compact_messages",
    "compare_strategies",
    "count_tokens",
    "evaluate_gate",
    "gate_for_workload",
    "get_strategy",
    "improve",
    "list_strategies",
    "load_bundled_fixture_set",
    "load_fixture_set",
    "measure_compaction",
    "measure_tool_selection",
    "register_strategy",
    "register_strategy_node",
    "request_from_assistant",
    "resolve_strategy_instance",
    "run_fixture_set",
    "save_fixture_set",
    "score_case",
    "score_document_qa",
    "score_extraction",
    "score_result",
    "score_tool_task",
    "shadow_compare",
    "summarize_runs",
    "supports_native_tool_discovery",
]


# Make the ``strategy`` workflow node type available as soon as the execution
# package is imported, mirroring how ``prompture.workflow`` registers its own
# built-ins at import.  Failure is non-fatal and logged, never raised.
register_strategy_node()
