"""Persistent, project-scoped catalog of host-approved, read-only workflow graphs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from jsonschema import validate

from ..agents.tools_schema import ToolDefinition, ToolRegistry
from .engine import compile_graph
from .errors import WorkflowError
from .model import Graph
from .recovery import graph_version


class RecipeStore:
    """Bind a catalog to a project, its read-only tool registry and a behavior version.

    The host supplies trusted graphs and a registry containing only permitted reads.
    Bump contract when tool semantics or host policy change without a schema change.
    """

    def __init__(self, path: Path | str, *, scope: str, registry: ToolRegistry, contract: str) -> None:
        if not scope.strip() or not contract.strip():
            raise ValueError("Recipe scope and contract must be nonempty")
        if str(path) == ":memory:":
            raise ValueError("RecipeStore requires a persistent SQLite file")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.scope = scope
        self.registry = registry
        self.contract = contract
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS recipe_defs (
                    scope TEXT, name TEXT, fingerprint TEXT NOT NULL, graph TEXT NOT NULL,
                    PRIMARY KEY(scope, name));
                CREATE TABLE IF NOT EXISTS recipe_routes (
                    scope TEXT, route TEXT, name TEXT NOT NULL, PRIMARY KEY(scope, route));
                CREATE TABLE IF NOT EXISTS recipe_runs (
                    scope TEXT, fingerprint TEXT, event TEXT, PRIMARY KEY(scope, fingerprint, event));
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _fingerprint(self, graph: Graph) -> str:
        schemas = {}
        for node in graph.nodes:
            if node.type in {"input", "output", "constant", "transform"}:
                continue
            if node.type != "tool":
                raise WorkflowError(f"Unsupported recipe node type: {node.type}")
            name = node.config.get("tool")
            tool = self.registry.get(name) if isinstance(name, str) else None
            if tool is None or tool.metadata.get("is_write"):
                raise WorkflowError(f"Recipe tool unavailable for read-only execution: {name}")
            schemas[name] = tool.parameters
        payload = json.dumps([graph_version(graph), self.contract, schemas], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def publish(self, graph: Graph, *, aliases: Iterable[str] = ()) -> str:
        """Atomically publish a reviewed graph and replace its exact, case-sensitive aliases."""
        graph.validate()
        fingerprint = self._fingerprint(graph)
        if isinstance(aliases, str):
            raise TypeError("Pass aliases as a sequence of route names")
        routes = {graph.id, *aliases}
        if not all(isinstance(route, str) and route.strip() for route in routes):
            raise ValueError("Recipe names and aliases must be nonempty strings")
        with self._connect() as db:
            db.execute("DELETE FROM recipe_routes WHERE scope=? AND name=?", (self.scope, graph.id))
            db.executemany("INSERT INTO recipe_routes VALUES (?,?,?)", [(self.scope, r, graph.id) for r in routes])
            db.execute(
                """INSERT INTO recipe_defs VALUES (?,?,?,?) ON CONFLICT(scope,name) DO UPDATE SET
                          fingerprint=excluded.fingerprint, graph=excluded.graph""",
                (self.scope, graph.id, fingerprint, graph.to_json()),
            )
        return fingerprint

    def _load(self, route: str) -> tuple[Graph, str]:
        with self._connect() as db:
            row = db.execute(
                """SELECT graph,fingerprint FROM recipe_routes JOIN recipe_defs USING(scope,name)
                                WHERE scope=? AND route=?""",
                (self.scope, route),
            ).fetchone()
        if row is None:
            raise KeyError(route)
        graph = Graph.from_json(row[0])
        if self._fingerprint(graph) != row[1]:
            raise WorkflowError(f"Recipe {route!r} changed or its contract is stale; publish the verified version")
        return graph, row[1]

    def load(self, route: str) -> Graph:
        """Load the current compatible definition. Inputs and previous outputs are never cached."""
        return self._load(route)[0]

    def run(
        self,
        route: str,
        inputs: dict[str, Any],
        *,
        verify: Callable[[dict[str, Any]], bool],
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Run against fresh tools. Count a verified event once per definition version."""
        graph, fingerprint = self._load(route)
        return self._run_loaded(graph, fingerprint, inputs, verify=verify, event_id=event_id)

    def run_or_learn(
        self,
        name: str,
        inputs: dict[str, Any],
        *,
        build: Callable[[], Graph],
        verify: Callable[[dict[str, Any]], bool],
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Build on an exact-route miss. Promote only after the supplied verifier accepts the fresh result.

        Concurrent cold readers may both execute. Publication and counting are atomic,
        and a concurrent incompatible publication is never overwritten.
        """
        try:
            graph, fingerprint = self._load(name)
        except KeyError:
            pass
        else:
            return self._run_loaded(graph, fingerprint, inputs, verify=verify, event_id=event_id)
        graph = build()
        if not name.strip() or graph.id != name:
            raise ValueError("A new graph must use its requested recipe name")
        fingerprint = self._fingerprint(graph)
        output = self._execute(graph, inputs, verify)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT OR IGNORE INTO recipe_defs VALUES (?,?,?,?)", (self.scope, name, fingerprint, graph.to_json())
            )
            db.execute("INSERT OR IGNORE INTO recipe_routes VALUES (?,?,?)", (self.scope, name, name))
            winner = db.execute(
                """SELECT fingerprint FROM recipe_routes JOIN recipe_defs USING(scope,name)
                                   WHERE scope=? AND route=?""",
                (self.scope, name),
            ).fetchone()
            if winner != (fingerprint,):
                raise WorkflowError(f"Conflicting concurrent publication of {name!r}")
            self._record(db, fingerprint, event_id)
        return output

    def _record(self, db: sqlite3.Connection, fingerprint: str, event_id: str | None) -> None:
        db.execute(
            "INSERT OR IGNORE INTO recipe_runs VALUES (?,?,?)",
            (self.scope, fingerprint, event_id if event_id is not None else uuid4().hex),
        )

    def _run_loaded(
        self,
        graph: Graph,
        fingerprint: str,
        inputs: dict[str, Any],
        *,
        verify: Callable[[dict[str, Any]], bool],
        event_id: str | None,
    ) -> dict[str, Any]:
        output = self._execute(graph, inputs, verify)
        with self._connect() as db:
            self._record(db, fingerprint, event_id)
        return output

    def _execute(
        self, graph: Graph, inputs: dict[str, Any], verify: Callable[[dict[str, Any]], bool]
    ) -> dict[str, Any]:
        compiled = compile_graph(graph, options={"tool_registry": self.registry, "strict_tool_arguments": True})
        schema = compiled.as_tool().parameters
        defaults = {h.name: h.default for h in graph.inputs if not h.required and h.default is not None}
        arguments = {**defaults, **inputs}
        validate(arguments, {**schema, "additionalProperties": False})
        output = compiled(**arguments)
        if not verify(output):
            raise WorkflowError(f"Verification failed for recipe {graph.id!r}")
        return output

    def popular(self) -> list[tuple[str, int]]:
        """Current definitions and their verified usage counts in this scope."""
        with self._connect() as db:
            rows = db.execute(
                """SELECT name,COUNT(event) AS uses,graph,fingerprint FROM recipe_defs
                                 LEFT JOIN recipe_runs USING(scope,fingerprint) WHERE scope=?
                                 GROUP BY name ORDER BY uses DESC,name""",
                (self.scope,),
            ).fetchall()
        eligible = []
        for name, count, raw, fingerprint in rows:
            try:
                compatible = self._fingerprint(Graph.from_json(raw)) == fingerprint
            except WorkflowError:
                continue
            if compatible:
                eligible.append((name, count))
        return eligible

    def as_tool(self, route: str, *, verify: Callable[[dict[str, Any]], bool]) -> ToolDefinition:
        """Expose a verified recipe through the existing native or simulated tool loop."""
        graph, fingerprint = self._load(route)
        tool = compile_graph(graph).as_tool()

        def run(**inputs: Any) -> dict[str, Any]:
            current, version = self._load(route)
            if version != fingerprint:
                raise WorkflowError(f"Re-register changed recipe tool {current.id!r}")
            return self._run_loaded(current, version, inputs, verify=verify, event_id=None)

        tool.function = run
        return tool
