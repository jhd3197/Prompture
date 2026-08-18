"""Public sync facade over the scheduler: GraphRunner, run_graph/run_node,
compile_graph, and the workflow-as-function ``CompiledWorkflow``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..groups.types import ErrorPolicy
from .errors import WorkflowError
from .model import Graph
from .scheduler import run_graph as _run_graph
from .scheduler import run_node as _run_node
from .state import NodeResult, RunResult, WorkflowCallbacks

__all__ = [
    "CompiledWorkflow",
    "GraphRunner",
    "compile_graph",
    "run_graph",
    "run_node",
]


class GraphRunner:
    """Stateless, reusable runner for executing graphs."""

    def __init__(self, *, error_policy: ErrorPolicy = ErrorPolicy.fail_fast) -> None:
        self.error_policy = error_policy

    def run(
        self,
        graph: Graph,
        inputs: dict[str, Any] | None = None,
        *,
        error_policy: ErrorPolicy | None = None,
        options: dict[str, Any] | None = None,
        callbacks: WorkflowCallbacks | None = None,
    ) -> RunResult:
        return _run_graph(
            graph,
            inputs,
            error_policy=error_policy or self.error_policy,
            options=options,
            callbacks=callbacks,
        )

    def run_node(
        self,
        graph: Graph,
        node_id: str,
        *,
        inputs: dict[str, Any] | None = None,
        upstream: dict[str, dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> NodeResult:
        return _run_node(graph, node_id, inputs=inputs, upstream=upstream, options=options)


def run_graph(
    graph: Graph,
    inputs: dict[str, Any] | None = None,
    *,
    error_policy: ErrorPolicy = ErrorPolicy.fail_fast,
    options: dict[str, Any] | None = None,
    callbacks: WorkflowCallbacks | None = None,
) -> RunResult:
    """Run a graph to completion (module-level convenience)."""
    return _run_graph(graph, inputs, error_policy=error_policy, options=options, callbacks=callbacks)


def run_node(
    graph: Graph,
    node_id: str,
    *,
    inputs: dict[str, Any] | None = None,
    upstream: dict[str, dict[str, Any]] | None = None,
    options: dict[str, Any] | None = None,
) -> NodeResult:
    """Run a single node given its inputs + upstream outputs (module-level convenience)."""
    return _run_node(graph, node_id, inputs=inputs, upstream=upstream, options=options)


@dataclass
class CompiledWorkflow:
    """A graph callable like a function: ``wf(topic="x") -> {output: value}``."""

    graph: Graph
    runner: GraphRunner
    error_policy: ErrorPolicy = ErrorPolicy.fail_fast
    options: dict[str, Any] = field(default_factory=dict)
    callbacks: WorkflowCallbacks | None = None
    last_run: RunResult | None = None

    def run(
        self,
        inputs: dict[str, Any] | None = None,
        *,
        error_policy: ErrorPolicy | None = None,
        options: dict[str, Any] | None = None,
        callbacks: WorkflowCallbacks | None = None,
    ) -> RunResult:
        run = self.runner.run(
            self.graph,
            inputs,
            error_policy=error_policy or self.error_policy,
            options=options or self.options,
            callbacks=callbacks or self.callbacks,
        )
        self.last_run = run
        return run

    def __call__(self, **inputs: Any) -> dict[str, Any]:
        run = self.run(inputs)
        if not run.ok:
            raise WorkflowError(run.error or f"Workflow {self.graph.id!r} failed")
        return run.outputs

    @property
    def signature(self) -> dict[str, list[str]]:
        return {
            "inputs": [h.name for h in self.graph.inputs],
            "outputs": list(self.graph.outputs),
        }

    def as_tool(self, name: str | None = None, description: str | None = None) -> Any:
        """Wrap the whole workflow as a :class:`ToolDefinition` (agent tool / subgraph)."""
        from ..agents.tools_schema import ToolDefinition

        graph = self.graph
        tool_name = name or graph.id
        properties = {h.name: {"type": "string", "description": h.description or h.name} for h in graph.inputs}
        required = [h.name for h in graph.inputs if h.required]

        def _invoke(**kwargs: Any) -> dict[str, Any]:
            return self(**kwargs)

        return ToolDefinition(
            name=tool_name,
            description=description or graph.name or f"Run the {tool_name} workflow",
            parameters={"type": "object", "properties": properties, "required": required},
            function=_invoke,
        )


def compile_graph(
    graph: Graph,
    *,
    runner: GraphRunner | None = None,
    error_policy: ErrorPolicy = ErrorPolicy.fail_fast,
    options: dict[str, Any] | None = None,
    callbacks: WorkflowCallbacks | None = None,
) -> CompiledWorkflow:
    """Validate *graph* and return a callable :class:`CompiledWorkflow`."""
    graph.validate()
    return CompiledWorkflow(
        graph=graph,
        runner=runner or GraphRunner(error_policy=error_policy),
        error_policy=error_policy,
        options=dict(options or {}),
        callbacks=callbacks,
    )
