"""Wiring the strategies into the surfaces that already exist.

Three integrations, all additive:

* :func:`request_from_assistant` builds an
  :class:`~prompture.execution.types.ExecutionRequest` out of an
  :class:`~prompture.agents.assistant.Assistant`, so one persona + skills + tools
  bundle can be executed by any of the three strategies without redefining it.
  ``Assistant.run_strategy`` / ``arun_strategy`` are thin wrappers over this.
* :func:`register_strategy_node` adds a ``"strategy"`` workflow node type.  It
  reuses the workflow engine's scheduler, references and accounting exactly as
  the built-in ``llm`` node does — no second engine, no duplicated executor.
* :func:`compare_strategies` runs several strategies over the same request and
  returns their results side by side, which is what the comparison example and
  the shadow runs in :mod:`prompture.execution.policy` are built on.

Nothing here changes an existing return type.  ``Assistant.arun`` still returns
an ``AssistantResult``; the strategy entry points are separate methods, and the
underlying component result stays reachable on
:attr:`~prompture.execution.types.ExecutionResult.raw`.
"""

from __future__ import annotations

import logging
from typing import Any

from .outcomes import TaskCategory
from .strategies.base import ExecutionStrategy
from .strategies.registry import get_strategy
from .types import EvidencePassage, ExecutionRequest, ExecutionResult, ResourceLimits

logger = logging.getLogger("prompture.execution.integration")

__all__ = [
    "compare_strategies",
    "register_strategy_node",
    "request_from_assistant",
    "resolve_strategy_instance",
    "run_strategy_node",
]


def resolve_strategy_instance(strategy: ExecutionStrategy | str, **kwargs: Any) -> ExecutionStrategy:
    """Accept either a built strategy or a registered name."""
    if isinstance(strategy, ExecutionStrategy):
        if kwargs:
            raise ValueError(
                "Strategy keyword arguments are only used when resolving by name; "
                "configure the instance you passed instead."
            )
        return strategy
    return get_strategy(str(strategy), **kwargs)


# ---------------------------------------------------------------------------
# Assistant
# ---------------------------------------------------------------------------


def request_from_assistant(
    assistant: Any,
    prompt: str,
    *,
    category: TaskCategory = TaskCategory.OTHER,
    output_model: Any = None,
    passages: Any = (),
    retriever: Any = None,
    limits: ResourceLimits | None = None,
    allowed_tools: frozenset[str] | None = None,
    variables: dict[str, Any] | None = None,
    **overrides: Any,
) -> ExecutionRequest:
    """Build an :class:`ExecutionRequest` from an ``Assistant`` configuration.

    The assistant's persona *with its skills already composed in* is carried
    over, so the same persona text a strategy sees is the one
    ``Assistant.arun`` would have used — including a callable system prompt,
    which is rendered with the merged template variables.

    Raises:
        ValueError: When the assistant uses the ``coding_agent`` backend.  Those
            shell out to a CLI that owns its own loop; there is no driver for a
            strategy to drive.
    """
    if getattr(assistant, "coding_agent", None):
        raise ValueError(
            "Execution strategies drive an LLM model directly; this Assistant is "
            f"configured with coding_agent={assistant.coding_agent!r}. Use "
            "Assistant.arun() for the coding-agent backend."
        )

    merged_vars = {**dict(getattr(assistant, "variables", {}) or {}), **(variables or {})}
    persona = assistant._composed_persona(merged_vars)
    passage_tuple = tuple(EvidencePassage.from_any(p, index=i) for i, p in enumerate(passages or ()))

    return ExecutionRequest(
        task=prompt,
        category=category,
        output_model=output_model if output_model is not None else getattr(assistant, "output_type", None),
        passages=passage_tuple,
        retriever=retriever,
        tools=assistant._tools_for_agent(),
        allowed_tools=allowed_tools,
        model=getattr(assistant, "model", None),
        persona=persona,
        options=dict(getattr(assistant, "options", {}) or {}),
        limits=limits or ResourceLimits(),
        variables=merged_vars,
        metadata={"assistant": getattr(assistant, "name", "")},
        **overrides,
    )


# ---------------------------------------------------------------------------
# Workflow node
# ---------------------------------------------------------------------------


def run_strategy_node(node: Any, config: dict[str, Any], ctx: Any) -> Any:
    """Workflow runner for the ``"strategy"`` node type.

    Config keys:

    ``strategy``
        Registered strategy name (``"direct"``, ``"retrieve_verify"``,
        ``"draft_critique"``) or a pre-built strategy instance.
    ``model``
        Model string.  Falls back to the run option ``"model"``.
    ``task`` / ``prompt``
        The task text.  Templated references are already resolved by the
        scheduler, exactly as for the built-in ``llm`` node.
    ``instruction``, ``output_schema``, ``passages``, ``options``,
    ``strategy_options``
        Optional; forwarded to the request / strategy constructor.
    ``max_cost_usd``, ``max_llm_calls``, ``max_steps``, ``max_seconds``
        Optional resource bounds for this node's run.

    Outputs ``{"answer", "output", "termination", "ok", "evidence_sources",
    "decisions"}`` and reports the run's usage and cost to the workflow's
    aggregate, so a strategy node is accounted like any other node.

    Failure behaviour matches the built-in ``llm`` node: a run that did not
    complete raises :class:`~prompture.workflow.errors.NodeExecutionError`, so
    the scheduler marks the node failed and skips whatever depended on it.  The
    one exception is a deliberate abstention (``INSUFFICIENT_EVIDENCE`` /
    ``ABSTAINED``), which completes the node with ``ok=False`` so a graph can
    branch on "no grounded answer" without treating it as a crash.

    Note that on the raising path the node's usage does not reach the run's
    aggregate — the engine builds its own empty ``NodeResult`` for a raised
    node.  The calls are still recorded by the driver hooks in
    :mod:`prompture.infra.tracker`, so the spend is not lost, only absent from
    ``RunResult.total_usage``.
    """
    from ..workflow.errors import NodeExecutionError
    from ..workflow.state import NodeResult, NodeStatus

    name = config.get("strategy") or "direct"
    task = config.get("task") or config.get("prompt") or ""
    if not task:
        raise NodeExecutionError(node.id, "strategy node requires config['task'] (or config['prompt'])")

    model = config.get("model") or (ctx.options or {}).get("model")
    strategy_options = dict(config.get("strategy_options") or {})
    if model and "model" not in strategy_options:
        strategy_options["model"] = model
    driver_callbacks = (ctx.options or {}).get("driver_callbacks")

    strategy = resolve_strategy_instance(name, **strategy_options) if isinstance(name, str) else name

    limits = ResourceLimits(
        max_cost_usd=config.get("max_cost_usd"),
        max_llm_calls=config.get("max_llm_calls"),
        max_steps=config.get("max_steps"),
        max_seconds=config.get("max_seconds"),
    )
    options = dict(config.get("options") or {})
    request = ExecutionRequest(
        task=str(task),
        task_id=node.id,
        instruction=str(config.get("instruction") or ""),
        output_schema=config.get("output_schema"),
        passages=tuple(EvidencePassage.from_any(p, index=i) for i, p in enumerate(config.get("passages") or ())),
        model=model,
        options=options,
        limits=limits,
        metadata={"workflow_node": node.id},
    )

    # Bridge the host's driver callbacks the same way the built-in llm node does.
    if driver_callbacks is not None and getattr(strategy, "_driver", None) is not None:
        strategy._driver.callbacks = driver_callbacks

    result = strategy.run(request)

    # A run that failed must fail the *node*, so the scheduler's skip-propagation
    # and error policy work exactly as they do for the built-in ``llm`` node.
    # Raising is how a node reports failure to this engine; returning a FAILED
    # NodeResult would let downstream nodes consume an empty answer.
    #
    # A deliberate abstention is the exception: "the evidence did not support an
    # answer" is a real, structured outcome, so the node completes with
    # ``ok=False`` and the graph can branch on it.
    if not result.ok and not result.abstained:
        raise NodeExecutionError(
            node.id,
            f"strategy {result.strategy!r} terminated as {result.termination.value}"
            + (f": {result.error}" if result.error else "")
            + (f" | decisions: {result.decisions}" if result.decisions else ""),
        )

    usage = {
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "total_tokens": result.usage.total_tokens,
        "cost": result.usage.cost,
    }
    return NodeResult(
        node_id=node.id,
        status=NodeStatus.COMPLETED,
        outputs={
            "answer": result.answer,
            "output": result.output,
            "termination": result.termination.value,
            "ok": result.ok,
            "evidence_sources": list(result.evidence.sources),
            "decisions": list(result.decisions),
        },
        usage=usage,
        cost=result.usage.cost,
        strategy=result.strategy,
        raw={
            "cost_complete": result.usage.cost_complete,
            "budget_note": result.budget_note,
            "steps": [s.to_dict() for s in result.steps],
        },
    )


def register_strategy_node(*, overwrite: bool = True) -> bool:
    """Register the ``"strategy"`` workflow node type.

    Returns ``True`` when registration happened.  Returns ``False`` (without
    raising) when the workflow package cannot be imported, so importing
    :mod:`prompture.execution` never fails because of an optional dependency.
    """
    try:
        from ..workflow.model import Handle, HandleType
        from ..workflow.registry import register_node_type
    except Exception:  # pragma: no cover - defensive
        logger.debug("workflow package unavailable; 'strategy' node type not registered")
        return False

    register_node_type(
        "strategy",
        run_strategy_node,
        output_handles=(
            Handle("answer", HandleType.TEXT),
            Handle("output", HandleType.JSON),
            Handle("termination", HandleType.TEXT),
            Handle("ok", HandleType.BOOLEAN),
        ),
        description="Run a prompture.execution strategy as a workflow node.",
        overwrite=overwrite,
    )
    return True


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_strategies(
    request: ExecutionRequest,
    strategies: dict[str, ExecutionStrategy],
) -> dict[str, ExecutionResult]:
    """Run each strategy over the *same* request and collect the results.

    Each strategy gets its own copy of the request, so one strategy mutating a
    field (a policy narrowing ``allowed_tools``, for instance) cannot affect the
    next.  Failures are captured in the returned envelopes rather than raised,
    so one broken configuration does not hide the others.

    Note that this is only safe to point at side-effecting tools when those
    tools are idempotent or sandboxed — running three strategies against a real
    write API performs the write three times.  The bundled
    :class:`~prompture.execution.sandbox.InventoryWorld` exists for exactly this
    reason.
    """
    results: dict[str, ExecutionResult] = {}
    for label, strategy in strategies.items():
        results[label] = strategy.run(request.with_overrides())
    return results
