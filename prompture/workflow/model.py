"""Declarative, serializable data model for the workflow DAG.

``Graph`` / ``Node`` / ``Edge`` / ``Handle`` are pure dataclasses with
``to_dict`` / ``from_dict`` (and JSON helpers) exactly like
:class:`~prompture.jobs.JobHandle` — JSON-round-trippable, no execution logic,
no driver imports. The effective dependency graph is the union of explicit
``Edge`` wiring and references discovered in each node's ``config`` (via
:func:`~prompture.workflow.references.extract_dependencies`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .errors import CycleError, GraphValidationError
from .references import RESERVED_ROOTS, extract_dependencies

__all__ = [
    "Edge",
    "Graph",
    "Handle",
    "HandleType",
    "Node",
    "build_graph",
]


class HandleType(str, Enum):
    """Coarse, advisory port types — inform coercion, not a full type lattice."""

    ANY = "any"
    TEXT = "text"
    JSON = "json"
    NUMBER = "number"
    BOOLEAN = "boolean"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    ASSET = "asset"
    LIST = "list"

    def is_compatible(self, other: HandleType) -> bool:
        """True if a value typed ``self`` may feed a port typed ``other``."""
        if self is HandleType.ANY or other is HandleType.ANY:
            return True
        media = {HandleType.IMAGE, HandleType.VIDEO, HandleType.AUDIO}
        if HandleType.ASSET in (self, other) and (self in media or other in media):
            return True
        return self is other


def _filter_known(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    return {k: v for k, v in data.items() if k in known}


@dataclass(frozen=True)
class Handle:
    """A typed input/output port on a node (or graph)."""

    name: str
    type: HandleType = HandleType.ANY
    required: bool = True
    default: Any = None
    description: str | None = None
    schema: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Handle:
        data = _filter_known(cls, data)
        if "type" in data and not isinstance(data["type"], HandleType):
            data["type"] = HandleType(data["type"])
        return cls(**data)

    def coerce(self, value: Any) -> Any:
        """Advisory coercion (never raises in MVP). NUMBER strings → int/float."""
        if value is None:
            return self.default
        if self.type is HandleType.NUMBER and isinstance(value, str):
            try:
                return int(value) if value.lstrip("-").isdigit() else float(value)
            except ValueError:
                return value
        return value


@dataclass(frozen=True)
class Edge:
    """An explicit handle-to-handle connection (visual/ordering wire)."""

    source: str
    source_handle: str
    target: str
    target_handle: str
    condition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edge:
        return cls(**_filter_known(cls, data))

    def key(self) -> tuple[str, str, str, str]:
        return (self.source, self.source_handle, self.target, self.target_handle)


@dataclass
class Node:
    """One unit of work: a registry ``type`` plus templated ``config``."""

    id: str
    type: str
    config: dict[str, Any] = field(default_factory=dict)
    inputs: list[Handle] = field(default_factory=list)
    outputs: list[Handle] = field(default_factory=list)
    title: str | None = None
    cache: bool = True
    on_error: str = "fail"  # "fail" | "skip"
    poll: bool = True
    timeout: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "config": self.config,
            "inputs": [h.to_dict() for h in self.inputs],
            "outputs": [h.to_dict() for h in self.outputs],
            "title": self.title,
            "cache": self.cache,
            "on_error": self.on_error,
            "poll": self.poll,
            "timeout": self.timeout,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        data = dict(_filter_known(cls, data))
        data["inputs"] = [Handle.from_dict(h) for h in data.get("inputs", [])]
        data["outputs"] = [Handle.from_dict(h) for h in data.get("outputs", [])]
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> Node:
        return cls.from_dict(json.loads(raw))

    def output_handle(self, name: str) -> Handle | None:
        return next((h for h in self.outputs if h.name == name), None)

    def input_handle(self, name: str) -> Handle | None:
        return next((h for h in self.inputs if h.name == name), None)

    def referenced_node_ids(self) -> set[str]:
        return extract_dependencies(self.config)


@dataclass
class Graph:
    """A serializable DAG of :class:`Node` connected by edges and/or references."""

    id: str = "workflow"
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    inputs: list[Handle] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── construction ──
    def add_node(self, node: Node) -> Graph:
        self.nodes.append(node)
        return self

    def add_edge(self, edge: Edge | str, *handles: str) -> Graph:
        if isinstance(edge, Edge):
            self.edges.append(edge)
        else:
            source_handle, target, target_handle = handles
            self.edges.append(Edge(edge, source_handle, target, target_handle))
        return self

    # ── lookup ──
    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def _node_ids(self) -> set[str]:
        return {n.id for n in self.nodes}

    def predecessors(self, node_id: str) -> list[str]:
        node = self.node(node_id)
        preds = {e.source for e in self.edges if e.target == node_id}
        preds |= node.referenced_node_ids()
        preds.discard(node_id)
        return sorted(preds & self._node_ids())

    def successors(self, node_id: str) -> list[str]:
        ids = self._node_ids()
        out = {e.target for e in self.edges if e.source == node_id}
        for n in self.nodes:
            if node_id in n.referenced_node_ids():
                out.add(n.id)
        out.discard(node_id)
        return sorted(out & ids)

    def input_nodes(self) -> list[Node]:
        return [n for n in self.nodes if not self.predecessors(n.id)]

    def output_nodes(self) -> list[Node]:
        return [n for n in self.nodes if not self.successors(n.id)]

    # ── validation ──
    def validate(self) -> None:
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise GraphValidationError(f"Duplicate node ids: {dupes}")
        id_set = set(ids)

        for e in self.edges:
            if e.source not in id_set:
                raise GraphValidationError(f"Edge source {e.source!r} is not a known node")
            if e.target not in id_set:
                raise GraphValidationError(f"Edge target {e.target!r} is not a known node")

        for n in self.nodes:
            for root in n.referenced_node_ids():
                if root not in id_set and root not in RESERVED_ROOTS:
                    raise GraphValidationError(f"Node {n.id!r} references unknown node {root!r}")
            if n.id in n.referenced_node_ids():
                raise CycleError(f"Node {n.id!r} references itself", nodes=[n.id])

        self.topo_order()  # raises CycleError if not acyclic

    def topo_order(self) -> list[str]:
        """Kahn topological sort, deterministic by insertion order."""
        order_index = {n.id: i for i, n in enumerate(self.nodes)}
        indegree = {n.id: 0 for n in self.nodes}
        adj: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            for pred in self.predecessors(n.id):
                adj[pred].append(n.id)
                indegree[n.id] += 1

        ready = sorted((nid for nid, d in indegree.items() if d == 0), key=lambda x: order_index[x])
        order: list[str] = []
        while ready:
            cur = ready.pop(0)
            order.append(cur)
            for nxt in sorted(adj[cur], key=lambda x: order_index[x]):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)
            ready.sort(key=lambda x: order_index[x])

        if len(order) != len(self.nodes):
            remaining = sorted(set(indegree) - set(order))
            raise CycleError(f"Graph has a cycle among: {remaining}", nodes=remaining)
        return order

    # ── serialization ──
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "inputs": [h.to_dict() for h in self.inputs],
            "outputs": self.outputs,
            "name": self.name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Graph:
        data = dict(_filter_known(cls, data))
        data["nodes"] = [Node.from_dict(n) for n in data.get("nodes", [])]
        data["edges"] = [Edge.from_dict(e) for e in data.get("edges", [])]
        data["inputs"] = [Handle.from_dict(h) for h in data.get("inputs", [])]
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> Graph:
        return cls.from_dict(json.loads(raw))


def build_graph(
    nodes: list[Node],
    edges: list[Edge] | None = None,
    *,
    inputs: list[Handle] | None = None,
    outputs: dict[str, str] | None = None,
    id: str = "workflow",
    name: str | None = None,
) -> Graph:
    """Construct a :class:`Graph`, run :meth:`Graph.validate`, and return it."""
    g = Graph(
        id=id,
        nodes=list(nodes),
        edges=list(edges or []),
        inputs=list(inputs or []),
        outputs=dict(outputs or {}),
        name=name,
    )
    g.validate()
    return g
