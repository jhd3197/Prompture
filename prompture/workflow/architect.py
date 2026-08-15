"""Architect — turn a natural-language description into a validated workflow Graph.

Now that :mod:`prompture.workflow.model` defines a serializable graph, building a
graph from text is structured extraction: an LLM emits a :class:`WorkflowGraphSpec`
(nodes + edges + inputs + outputs) which is converted to a real :class:`Graph` and
validated. Reuses :func:`prompture.extraction.extract_with_model` — no new model
logic. Pass ``current_graph`` to *edit* an existing graph instead of building anew.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from .errors import GraphValidationError, WorkflowError
from .model import Edge, Graph, Handle, Node
from .registry import list_node_types

__all__ = [
    "ArchitectResult",
    "EdgeSpec",
    "NodeSpec",
    "WorkflowGraphSpec",
    "aarchitect",
    "architect",
    "node_type_catalog",
]


class NodeSpec(BaseModel):
    """One node in an LLM-proposed workflow."""

    id: str = Field(..., description="Unique node id (snake_case).")
    type: str = Field(..., description="Node type: input, output, constant, llm, tool, transform, media_gen.")
    config: dict[str, Any] = Field(default_factory=dict, description="Static params + {{refs}} to upstream outputs.")
    title: str | None = Field(default=None, description="Optional human label.")


class EdgeSpec(BaseModel):
    """An explicit node-to-node connection (usually unnecessary — prefer {{refs}})."""

    source: str
    source_handle: str = "value"
    target: str
    target_handle: str = "input"


class WorkflowGraphSpec(BaseModel):
    """The structured graph an Architect LLM produces from a description."""

    name: str | None = Field(default=None, description="Short workflow name.")
    inputs: list[str] = Field(default_factory=list, description="Graph-level input names.")
    nodes: list[NodeSpec] = Field(default_factory=list)
    edges: list[EdgeSpec] = Field(default_factory=list)
    outputs: dict[str, str] = Field(
        default_factory=dict,
        description="Output name -> reference template, e.g. {'brief': '{{summary.outputs.text}}'}.",
    )
    message: str = Field(default="", description="One-sentence explanation of the workflow.")
    suggestions: list[str] = Field(default_factory=list, description="Follow-up ideas to extend the workflow.")

    def to_graph(self, graph_id: str = "workflow", *, validate: bool = True) -> Graph:
        graph = Graph(
            id=graph_id,
            nodes=[Node(id=n.id, type=n.type, config=dict(n.config), title=n.title) for n in self.nodes],
            edges=[Edge(e.source, e.source_handle, e.target, e.target_handle) for e in self.edges],
            inputs=[Handle(name) for name in self.inputs],
            outputs=dict(self.outputs),
            name=self.name,
        )
        if validate:
            graph.validate()
        return graph


@dataclass
class ArchitectResult:
    """Result of an architect run: the graph (if valid) + message/suggestions."""

    graph: Graph | None
    message: str
    suggestions: list[str]
    spec: WorkflowGraphSpec
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.graph is not None and self.error is None


# ── prompt construction ─────────────────────────────────────────────────────

_NODE_DESCRIPTIONS: dict[str, str] = {
    "input": 'graph entry; config {"name": <graph-input>}; outputs.value',
    "output": 'terminal sink; config {"name": <out>, "value": "{{ref}}"}; records the value',
    "constant": 'fixed literal; config {"value": <literal>}; outputs.value',
    "llm": 'text LLM call; config {"model": "provider/model", "prompt": "...{{refs}}...", "options": {}}; outputs.text',
    "tool": 'call a registered tool; config {"tool": <name>, "args": {...}}; outputs.result',
    "transform": 'pure local shaping; config {"op": "format|concat|value|get", ...}; outputs.value',
    "media_gen": 'image/video/music; config {"modality": "image|video|music", "model": "...", "prompt": "...", "options": {}}; outputs.url, outputs.urls',
}


def node_type_catalog() -> str:
    """Return a text catalog of the registered node types for the system prompt."""
    lines = []
    for name in list_node_types():
        desc = _NODE_DESCRIPTIONS.get(name, "(custom node type)")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _system_prompt(current_graph: Graph | None) -> str:
    parts = [
        "You are a workflow Architect. Convert the user's request into a directed-acyclic-graph (DAG) "
        "of nodes that produce the desired result. Output a WorkflowGraphSpec.",
        "",
        "Available node types:",
        node_type_catalog(),
        "",
        "Wiring: a downstream node reads an upstream node's output with a late-bound reference template "
        'embedded in its config, e.g. "prompt": "Summarize {{fetch.outputs.text}}". Forms: '
        "{{node.outputs.<handle>}}, {{node.outputs[0].value}} (first output), {{node.value}} (sole output), "
        "{{inputs.<name>}} (a graph input). Prefer references over explicit edges.",
        "",
        "Rules: unique snake_case node ids; the graph must be acyclic; every {{...}} root must be a real "
        "node id or 'inputs'; set graph 'inputs' for entry values and 'outputs' (name -> reference) for the "
        "final results; give a one-sentence 'message' and 2-4 'suggestions'.",
    ]
    if current_graph is not None:
        parts += [
            "",
            "EDIT the following existing graph per the request (keep ids stable where possible):",
            current_graph.to_json(),
        ]
    return "\n".join(parts)


def _build_result(spec: WorkflowGraphSpec, usage: dict[str, Any], graph_id: str) -> ArchitectResult:
    known = set(list_node_types())
    unknown = sorted({n.type for n in spec.nodes if n.type not in known})
    if unknown:
        return ArchitectResult(
            graph=None,
            message=spec.message,
            suggestions=list(spec.suggestions),
            spec=spec,
            usage=usage,
            error=f"Spec uses unknown node types: {unknown}. Known: {sorted(known)}",
        )
    try:
        graph = spec.to_graph(graph_id)
    except (GraphValidationError, WorkflowError) as exc:
        return ArchitectResult(
            graph=None, message=spec.message, suggestions=list(spec.suggestions), spec=spec, usage=usage, error=str(exc)
        )
    return ArchitectResult(
        graph=graph, message=spec.message, suggestions=list(spec.suggestions), spec=spec, usage=usage
    )


def architect(
    description: str,
    *,
    model: str | None = None,
    current_graph: Graph | None = None,
    graph_id: str = "workflow",
    strategy: str | None = None,
    max_retries: int = 2,
    options: dict[str, Any] | None = None,
) -> ArchitectResult:
    """Build (or edit) a workflow :class:`Graph` from a natural-language *description*.

    Returns an :class:`ArchitectResult`. ``result.ok`` is False (with ``result.error``)
    if the proposed graph references unknown node types or fails validation.
    """
    from ..extraction.core import extract_with_model

    result = extract_with_model(
        WorkflowGraphSpec,
        text=description,
        model_name=model,
        system_prompt=_system_prompt(current_graph),
        strategy=strategy,
        max_retries=max_retries,
        options=options,
    )
    spec = result["model"]
    return _build_result(spec, dict(result.get("usage", {})), graph_id)


async def aarchitect(
    description: str,
    *,
    model: str | None = None,
    current_graph: Graph | None = None,
    graph_id: str = "workflow",
    strategy: str | None = None,
    max_retries: int = 2,
    options: dict[str, Any] | None = None,
) -> ArchitectResult:
    """Async sibling of :func:`architect`."""
    from ..extraction.async_core import extract_with_model as _aextract

    result = await _aextract(
        WorkflowGraphSpec,
        text=description,
        model_name=model,
        system_prompt=_system_prompt(current_graph),
        strategy=strategy,
        max_retries=max_retries,
        options=options,
    )
    spec = result["model"]
    return _build_result(spec, dict(result.get("usage", {})), graph_id)
