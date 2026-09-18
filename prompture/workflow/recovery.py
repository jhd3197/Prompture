"""Durable workflow state, resume, and external-action reconciliation.

The workflow engine in :mod:`prompture.workflow.scheduler` runs a graph to
completion in memory.  This module makes a run *survivable*: it persists the
definition version, the inputs, every completed node result, the pending work,
errors, pauses, cancellation, and usage — and it resumes from that state without
re-running work that already happened.

Reuse, not replacement
----------------------

Node execution still goes through :func:`prompture.workflow.scheduler.run_node`,
seeded with the persisted upstream outputs.  There is no second executor, no
duplicated reference resolution, and no fork of the node registry.  What is added
is the loop around it, the state, and the honesty about external effects.

The crash window, and what can honestly be promised
---------------------------------------------------

A side-effecting tool call has three failure windows, and only two of them are
recoverable by bookkeeping alone:

1. **Crash before the action.**  The :class:`OperationLedger` has no record.
   Retrying is safe.
2. **Crash after the external action succeeded but before the result was
   saved.**  The ledger holds a ``pending`` intent with no outcome.  This is
   genuinely ambiguous: the action may or may not have happened.  Resume marks
   it :data:`OperationStatus.RECONCILE_REQUIRED` and **stops**, rather than
   guessing.  Where the operation is idempotent, a caller can supply a
   ``reconciler`` that re-checks or safely replays it under the same
   ``operation_id``.
3. **Crash after the result was saved.**  The ledger says ``succeeded`` and the
   node is skipped.

Exactly-once external effects are **not** promised, because the underlying
services cannot generally provide them.  What is promised is: no silent loss of
required work, no blind repetition of an unsafe action, and an explicit
reconciliation state when the truth is unknown.

Costs survive a restart without double counting: usage is stored per completed
node, and a resumed run re-aggregates from the persisted node results rather than
re-executing them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..groups.types import ErrorPolicy
from .errors import WorkflowError
from .model import Graph
from .scheduler import run_node
from .state import NodeResult, NodeStatus, RunResult, RunStatus, WorkflowCallbacks, aggregate_usage

logger = logging.getLogger("prompture.workflow.recovery")

__all__ = [
    "FileWorkflowStateStore",
    "HumanInputRequired",
    "InMemoryWorkflowStateStore",
    "OperationLedger",
    "OperationRecord",
    "OperationStatus",
    "ReconciliationRequired",
    "ResumableGraphRunner",
    "RunLifecycle",
    "SQLiteWorkflowStateStore",
    "WorkflowRunState",
    "WorkflowStateStore",
    "WorkflowVersionMismatch",
    "graph_version",
    "operation_id",
]


# ---------------------------------------------------------------------------
# Errors and signals
# ---------------------------------------------------------------------------


class WorkflowVersionMismatch(WorkflowError):
    """Persisted state belongs to a different version of the graph.

    Carries both versions so a caller can decide between restarting, forcing, or
    migrating.
    """

    def __init__(self, run_id: str, persisted: str, current: str) -> None:
        self.run_id = run_id
        self.persisted = persisted
        self.current = current
        super().__init__(
            f"Run {run_id!r} was persisted against graph version {persisted[:12]} but the "
            f"supplied graph is {current[:12]}. Resume with on_mismatch='restart' to start "
            "over, or 'force' to continue against the new definition (completed node results "
            "from the old definition are kept)."
        )


class HumanInputRequired(WorkflowError):
    """Raised by a node runner to pause the run and wait for a person.

    The run is persisted as :data:`RunLifecycle.PAUSED` with ``prompt`` and
    ``node_id`` recorded; :meth:`ResumableGraphRunner.resume` supplies the
    answer through ``human_input``.
    """

    def __init__(self, prompt: str, *, node_id: str = "", schema: dict[str, Any] | None = None) -> None:
        self.prompt = prompt
        self.node_id = node_id
        self.schema = schema or {}
        super().__init__(f"Human input required for node {node_id or '?'}: {prompt}")


class ReconciliationRequired(WorkflowError):
    """A persisted operation's real-world outcome is unknown.

    Raised on resume when the ledger holds a dispatched-but-unconfirmed
    operation and no reconciler was supplied.  Carries the operation ids so the
    caller can check the external system.
    """

    def __init__(self, run_id: str, operations: list[str]) -> None:
        self.run_id = run_id
        self.operations = operations
        super().__init__(
            f"Run {run_id!r} has {len(operations)} operation(s) that were dispatched but never "
            f"confirmed: {operations}. Their external effect is unknown — this run will not "
            "guess. Supply reconcile=... to check or safely replay them, or resolve them by hand."
        )


# ---------------------------------------------------------------------------
# Operation ledger
# ---------------------------------------------------------------------------


class OperationStatus(str, Enum):
    """Lifecycle of one side-effecting operation."""

    #: Intent recorded; the call may or may not have reached the external system.
    PENDING = "pending"
    #: Confirmed successful, with its result stored.
    SUCCEEDED = "succeeded"
    #: Confirmed failed without effect.
    FAILED = "failed"
    #: Dispatched, never confirmed.  The external effect is unknown.
    RECONCILE_REQUIRED = "reconcile_required"


@dataclass
class OperationRecord:
    """One side-effecting operation with a stable identity."""

    operation_id: str
    node_id: str
    tool: str = ""
    status: OperationStatus = OperationStatus.PENDING
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    attempts: int = 0
    started_at: float = field(default_factory=time.time)
    settled_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperationRecord:
        return cls(
            operation_id=data["operation_id"],
            node_id=data.get("node_id", ""),
            tool=data.get("tool", ""),
            status=OperationStatus(data.get("status", OperationStatus.PENDING.value)),
            arguments=dict(data.get("arguments") or {}),
            result=data.get("result"),
            error=data.get("error"),
            attempts=int(data.get("attempts", 0)),
            started_at=float(data.get("started_at", time.time())),
            settled_at=data.get("settled_at"),
        )


def operation_id(run_id: str, node_id: str, *, tool: str = "", index: int = 0) -> str:
    """A stable identifier for one side-effecting call.

    Stable across restarts by construction: it is derived from the run, the node,
    the tool and the call's position within that node, not from a clock or a
    random source.  That is what lets a replay be recognised as the *same*
    operation by an idempotent external service.
    """
    raw = f"{run_id}|{node_id}|{tool}|{index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class OperationLedger:
    """Records intent before dispatch and the outcome after it.

    The ordering is the whole point.  :meth:`begin` writes ``pending`` *before*
    the external call goes out, so a crash between the two leaves evidence that
    something may have happened.  :meth:`succeed` / :meth:`fail` settle it.

    Example::

        op = operation_id(run_id, node_id, tool="charge_card")
        if ledger.already_succeeded(op):
            return ledger.result_of(op)          # never charge twice
        ledger.begin(op, node_id=node_id, tool="charge_card", arguments=args)
        try:
            result = charge_card(**args, idempotency_key=op)
        except Exception as exc:
            ledger.fail(op, str(exc))
            raise
        ledger.succeed(op, result)
    """

    def __init__(
        self,
        records: Iterable[OperationRecord] = (),
        *,
        on_change: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> None:
        self._records: dict[str, OperationRecord] = {r.operation_id: r for r in records}
        self._lock = threading.RLock()
        self._on_change = on_change

    def _persist(self) -> None:
        # Called before a mutation returns, while the reentrant lock is held.
        # Persistence errors must propagate: dispatch cannot proceed without a
        # durable intent, and an unconfirmed outcome must remain reconcilable.
        if self._on_change is not None:
            self._on_change(self.to_list())

    # ---- writing ------------------------------------------------------

    def begin(
        self,
        op_id: str,
        *,
        node_id: str,
        tool: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> OperationRecord:
        """Record the intent to dispatch.  Call this *before* the external call."""
        with self._lock:
            record = self._records.get(op_id)
            if record is None:
                record = OperationRecord(
                    operation_id=op_id, node_id=node_id, tool=tool, arguments=dict(arguments or {})
                )
                self._records[op_id] = record
            record.status = OperationStatus.PENDING
            record.attempts += 1
            record.settled_at = None
            self._persist()
            return record

    def succeed(self, op_id: str, result: Any = None) -> OperationRecord:
        with self._lock:
            record = self._records.setdefault(op_id, OperationRecord(operation_id=op_id, node_id=""))
            record.status = OperationStatus.SUCCEEDED
            record.result = result
            record.error = None
            record.settled_at = time.time()
            self._persist()
            return record

    def fail(self, op_id: str, error: str) -> OperationRecord:
        """Settle as failed *with no external effect*.

        Only use this when the failure is known to have happened before the
        external system acted (a validation error, a refused connection).  When
        that is not known, leave the record pending and let resume classify it as
        reconcile-required.
        """
        with self._lock:
            record = self._records.setdefault(op_id, OperationRecord(operation_id=op_id, node_id=""))
            record.status = OperationStatus.FAILED
            record.error = error
            record.settled_at = time.time()
            self._persist()
            return record

    def mark_reconcile_required(self, op_id: str) -> OperationRecord:
        with self._lock:
            record = self._records.setdefault(op_id, OperationRecord(operation_id=op_id, node_id=""))
            record.status = OperationStatus.RECONCILE_REQUIRED
            self._persist()
            return record

    # ---- reading ------------------------------------------------------

    def get(self, op_id: str) -> OperationRecord | None:
        with self._lock:
            return self._records.get(op_id)

    def already_succeeded(self, op_id: str) -> bool:
        record = self.get(op_id)
        return record is not None and record.status is OperationStatus.SUCCEEDED

    def result_of(self, op_id: str) -> Any:
        record = self.get(op_id)
        return None if record is None else record.result

    def unsettled(self) -> list[OperationRecord]:
        """Operations that were dispatched and never confirmed."""
        with self._lock:
            return [
                r
                for r in self._records.values()
                if r.status in {OperationStatus.PENDING, OperationStatus.RECONCILE_REQUIRED}
            ]

    def records(self) -> list[OperationRecord]:
        with self._lock:
            return list(self._records.values())

    def to_list(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records()]

    @classmethod
    def from_list(
        cls,
        items: Iterable[dict[str, Any]],
        *,
        on_change: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> OperationLedger:
        return cls((OperationRecord.from_dict(i) for i in items), on_change=on_change)


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------


def graph_version(graph: Graph) -> str:
    """A content hash of the graph definition.

    Two graphs with the same nodes, configs, edges and outputs have the same
    version; any change to the definition produces a different one, which is
    what makes a mismatch detectable instead of silently resumed.
    """
    payload = json.dumps(graph.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RunLifecycle(str, Enum):
    """Persisted lifecycle of a durable run.

    A superset of :class:`~prompture.workflow.state.RunStatus`: it adds the
    states that only exist once a run can outlive its process.
    """

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    RECONCILE_REQUIRED = "reconcile_required"

    @property
    def is_terminal(self) -> bool:
        """The run is not currently executing."""
        return self in {RunLifecycle.COMPLETED, RunLifecycle.FAILED, RunLifecycle.CANCELLED}

    @property
    def is_resumable(self) -> bool:
        """Whether :meth:`ResumableGraphRunner.resume` will accept this run.

        ``FAILED`` is resumable on purpose — retrying the node that failed is the
        main reason durable state exists. ``COMPLETED`` and ``CANCELLED`` are
        not: finishing twice and un-cancelling are both decisions the caller has
        to make explicitly, by starting a new run.
        """
        return self in {
            RunLifecycle.PENDING,
            RunLifecycle.RUNNING,
            RunLifecycle.PAUSED,
            RunLifecycle.FAILED,
            RunLifecycle.RECONCILE_REQUIRED,
        }


@dataclass
class WorkflowRunState:
    """Everything needed to continue a run in a fresh process.

    Attributes:
        run_id: Stable id for the run.
        graph_id / graph_version: Which definition this belongs to.
        inputs: The graph inputs.  Large payloads should be passed by reference
            (a URI, an artifact handle) rather than inlined — the state is meant
            to be small enough to write after every node.
        node_results: Completed / failed / skipped node results, by node id.
        pending: Node ids still to run.
        status: See :class:`RunLifecycle`.
        error: First error message, when the run failed.
        paused_node / pause_prompt / pause_schema: Set when a node asked for
            human input.
        human_input: The answer supplied on resume, keyed by node id.
        operations: The side-effect ledger, serialised.
        usage: Aggregated usage over completed nodes.  Recomputed from
            ``node_results`` on every save, so a resume cannot double-count.
        attempts: How many times this run has been started or resumed.
    """

    run_id: str
    graph_id: str
    graph_version: str
    inputs: dict[str, Any] = field(default_factory=dict)
    node_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)
    status: RunLifecycle = RunLifecycle.PENDING
    error: str | None = None
    paused_node: str | None = None
    pause_prompt: str = ""
    pause_schema: dict[str, Any] = field(default_factory=dict)
    human_input: dict[str, Any] = field(default_factory=dict)
    operations: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- derived ------------------------------------------------------

    @property
    def completed_nodes(self) -> set[str]:
        """Nodes that finished successfully and must not be re-run."""
        return {nid for nid, data in self.node_results.items() if data.get("status") == NodeStatus.COMPLETED.value}

    @property
    def settled_nodes(self) -> set[str]:
        """Nodes with any terminal result (completed, failed, or skipped)."""
        return set(self.node_results)

    def outputs_of(self, node_id: str) -> dict[str, Any]:
        return dict((self.node_results.get(node_id) or {}).get("outputs") or {})

    def ledger(self) -> OperationLedger:
        return OperationLedger.from_list(self.operations)

    def recompute_usage(self) -> dict[str, Any]:
        """Re-derive total usage from the persisted node results.

        Deriving rather than accumulating is what keeps costs correct across a
        restart: a resumed run adds nothing for nodes it did not execute, and
        cannot double-count the ones it did.
        """
        self.usage = aggregate_usage([data.get("usage") or {} for data in self.node_results.values()])
        return self.usage

    def record_node(self, result: NodeResult) -> None:
        self.node_results[result.node_id] = result.to_dict()
        if result.node_id in self.pending:
            self.pending.remove(result.node_id)
        self.recompute_usage()
        self.updated_at = time.time()

    # ---- serialisation ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowRunState:
        return cls(
            run_id=data["run_id"],
            graph_id=data.get("graph_id", ""),
            graph_version=data.get("graph_version", ""),
            inputs=dict(data.get("inputs") or {}),
            node_results=dict(data.get("node_results") or {}),
            pending=list(data.get("pending") or []),
            status=RunLifecycle(data.get("status", RunLifecycle.PENDING.value)),
            error=data.get("error"),
            paused_node=data.get("paused_node"),
            pause_prompt=data.get("pause_prompt", ""),
            pause_schema=dict(data.get("pause_schema") or {}),
            human_input=dict(data.get("human_input") or {}),
            operations=list(data.get("operations") or []),
            usage=dict(data.get("usage") or {}),
            attempts=int(data.get("attempts", 0)),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            version=int(data.get("version", 1)),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, raw: str) -> WorkflowRunState:
        return cls.from_dict(json.loads(raw))


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


@runtime_checkable
class WorkflowStateStore(Protocol):
    """Persistence contract for :class:`WorkflowRunState`.

    Implementations must be thread-safe and must make :meth:`save` durable
    before returning — the whole recovery story depends on a save that survives
    the process that made it.
    """

    def save(self, state: WorkflowRunState) -> None: ...

    def load(self, run_id: str) -> WorkflowRunState | None: ...

    def list_runs(self, *, status: RunLifecycle | None = None) -> list[WorkflowRunState]: ...

    def delete(self, run_id: str) -> bool: ...


class InMemoryWorkflowStateStore:
    """Process-local store.  Useful in tests and for a single-process run."""

    def __init__(self) -> None:
        self._runs: dict[str, WorkflowRunState] = {}
        self._lock = threading.RLock()

    def save(self, state: WorkflowRunState) -> None:
        with self._lock:
            # Store a copy so a later in-place mutation cannot retroactively
            # rewrite what was "persisted".
            self._runs[state.run_id] = WorkflowRunState.from_dict(state.to_dict())

    def load(self, run_id: str) -> WorkflowRunState | None:
        with self._lock:
            found = self._runs.get(run_id)
            return None if found is None else WorkflowRunState.from_dict(found.to_dict())

    def list_runs(self, *, status: RunLifecycle | None = None) -> list[WorkflowRunState]:
        with self._lock:
            runs = [WorkflowRunState.from_dict(r.to_dict()) for r in self._runs.values()]
        if status is not None:
            runs = [r for r in runs if r.status is status]
        runs.sort(key=lambda r: r.updated_at, reverse=True)
        return runs

    def delete(self, run_id: str) -> bool:
        with self._lock:
            return self._runs.pop(run_id, None) is not None


class FileWorkflowStateStore:
    """One JSON file per run, written atomically.

    Writes go to a temp file in the same directory and are then ``os.replace``d,
    so a crash mid-write can never leave a half-written state that a resume would
    read as truth.
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, run_id: str) -> Path:
        safe = run_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self._dir / f"{safe}.json"

    def save(self, state: WorkflowRunState) -> None:
        target = self._path(state.run_id)
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(state.to_json())
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, target)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

    def load(self, run_id: str) -> WorkflowRunState | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        try:
            return WorkflowRunState.from_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("could not read workflow state %s: %s", path, exc)
            return None

    def list_runs(self, *, status: RunLifecycle | None = None) -> list[WorkflowRunState]:
        runs: list[WorkflowRunState] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                runs.append(WorkflowRunState.from_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        if status is not None:
            runs = [r for r in runs if r.status is status]
        runs.sort(key=lambda r: r.updated_at, reverse=True)
        return runs

    def delete(self, run_id: str) -> bool:
        path = self._path(run_id)
        with self._lock:
            if path.exists():
                path.unlink()
                return True
        return False


class SQLiteWorkflowStateStore:
    """Single-file database store, mirroring the checkpoint store's patterns."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        default = Path.home() / ".prompture" / "workflows" / "runs.db"
        self._path = Path(db_path) if db_path is not None else default
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._path), timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_db(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    run_id TEXT PRIMARY KEY,
                    graph_id TEXT,
                    status TEXT,
                    updated_at REAL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON workflow_runs(status)")

    def save(self, state: WorkflowRunState) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workflow_runs (run_id, graph_id, status, updated_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    graph_id=excluded.graph_id,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    payload=excluded.payload
                """,
                (state.run_id, state.graph_id, state.status.value, state.updated_at, state.to_json()),
            )

    def load(self, run_id: str) -> WorkflowRunState | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT payload FROM workflow_runs WHERE run_id = ?", (run_id,)).fetchone()
        return WorkflowRunState.from_json(row[0]) if row else None

    def list_runs(self, *, status: RunLifecycle | None = None) -> list[WorkflowRunState]:
        query = "SELECT payload FROM workflow_runs"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status.value,)
        query += " ORDER BY updated_at DESC"
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [WorkflowRunState.from_json(row[0]) for row in rows]

    def delete(self, run_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM workflow_runs WHERE run_id = ?", (run_id,))
            return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# The resumable runner
# ---------------------------------------------------------------------------


#: ``(record) -> OperationStatus`` or ``(record) -> (OperationStatus, result)``.
#:
#: Decides what really happened to a dispatched-but-unconfirmed operation.
#: Returning ``SUCCEEDED`` or ``FAILED`` resolves it; ``RECONCILE_REQUIRED``
#: leaves it for a person.
#:
#: The two-tuple form matters more than it looks: when a process dies between an
#: external call succeeding and its result being written, the *result* is lost
#: even though the effect happened.  A reconciler that queries the external
#: system can hand that value back, and the resumed node then returns the real
#: outcome instead of ``None``.
Reconciler = Callable[[OperationRecord], "OperationStatus | tuple[OperationStatus, Any]"]


class ResumableGraphRunner:
    """Runs a graph with durable state, and resumes a compatible run.

    Args:
        store: Where state is persisted.  Defaults to an in-memory store, which
            gives no durability — pass a
            :class:`FileWorkflowStateStore` or :class:`SQLiteWorkflowStateStore`
            for a run that must survive the process.
        error_policy: ``fail_fast`` stops scheduling after the first failure;
            ``continue_on_error`` lets independent branches finish.
        callbacks: The same :class:`~prompture.workflow.state.WorkflowCallbacks`
            the in-memory engine takes.

    Example::

        runner = ResumableGraphRunner(store=FileWorkflowStateStore("./runs"))
        result = runner.start(graph, {"topic": "x"}, run_id="run-1")
        # ... process dies ...
        result = runner.resume(graph, "run-1")
    """

    def __init__(
        self,
        *,
        store: WorkflowStateStore | None = None,
        error_policy: ErrorPolicy = ErrorPolicy.fail_fast,
        callbacks: WorkflowCallbacks | None = None,
    ) -> None:
        self.store = store or InMemoryWorkflowStateStore()
        self.error_policy = error_policy
        self.callbacks = callbacks

    # ---- entry points -------------------------------------------------

    def start(
        self,
        graph: Graph,
        inputs: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
        options: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunResult:
        """Begin a new durable run.

        The initial state — definition version, inputs, and the full pending node
        list — is persisted *before* the first node executes, so even a crash
        during node one leaves a resumable record.
        """
        graph.validate()
        state = WorkflowRunState(
            run_id=run_id or uuid.uuid4().hex,
            graph_id=graph.id,
            graph_version=graph_version(graph),
            inputs=dict(inputs or {}),
            pending=graph.topo_order(),
            status=RunLifecycle.PENDING,
            metadata=dict(metadata or {}),
        )
        self.store.save(state)
        return self._drive(graph, state, options=options)

    def resume(
        self,
        graph: Graph,
        run_id: str,
        *,
        options: dict[str, Any] | None = None,
        human_input: Any = None,
        on_mismatch: str = "fail",
        reconcile: Reconciler | None = None,
    ) -> RunResult:
        """Continue a persisted run.

        Args:
            graph: The graph definition to continue against.
            run_id: Which run.
            options: Run options (tool registries, driver callbacks, …).  These
                are *not* persisted — they hold live objects — so a resuming
                process must supply them again.
            human_input: The answer to a pause, when the run is paused.
            on_mismatch: What to do when the graph version differs from the
                persisted one.  ``"fail"`` (default) raises
                :class:`WorkflowVersionMismatch`; ``"restart"`` discards the
                completed results and runs the new definition from the start;
                ``"force"`` continues, keeping results produced by the old
                definition — sound only when the change did not affect the nodes
                already run, which is the caller's judgement to make.
            reconcile: Resolver for dispatched-but-unconfirmed operations.

        Raises:
            KeyError: No such run.
            WorkflowError: The run is already terminal.
            WorkflowVersionMismatch: Definition changed and ``on_mismatch='fail'``.
            ReconciliationRequired: Unconfirmed operations and no reconciler.
        """
        state = self.store.load(run_id)
        if state is None:
            raise KeyError(f"No persisted workflow run {run_id!r}")
        if not state.status.is_resumable:
            raise WorkflowError(
                f"Run {run_id!r} is already {state.status.value}; it cannot be resumed. Start a new run instead."
            )

        current = graph_version(graph)
        if state.graph_version != current:
            if on_mismatch == "fail":
                raise WorkflowVersionMismatch(run_id, state.graph_version, current)
            if on_mismatch == "restart":
                logger.warning(
                    "run %s: graph changed; discarding %d completed node result(s)", run_id, len(state.node_results)
                )
                state.node_results.clear()
                state.pending = graph.topo_order()
                state.error = None
            elif on_mismatch == "force":
                logger.warning(
                    "run %s: graph changed; continuing with %d result(s) produced by the previous definition",
                    run_id,
                    len(state.node_results),
                )
                state.metadata.setdefault("version_mismatch_forced", []).append(
                    {"from": state.graph_version, "to": current, "at": time.time()}
                )
            else:
                raise ValueError(f"on_mismatch must be 'fail', 'restart', or 'force' (got {on_mismatch!r})")
            state.graph_version = current

        self._resolve_operations(state, reconcile)

        if state.paused_node is not None:
            if human_input is None:
                raise WorkflowError(
                    f"Run {run_id!r} is paused at node {state.paused_node!r} waiting for input: "
                    f"{state.pause_prompt}. Resume with human_input=..."
                )
            state.human_input[state.paused_node] = human_input
            state.paused_node = None
            state.pause_prompt = ""
            state.pause_schema = {}

        state.status = RunLifecycle.RUNNING
        self.store.save(state)
        return self._drive(graph, state, options=options)

    def cancel(self, run_id: str, *, reason: str = "cancelled by caller") -> WorkflowRunState:
        """Mark a persisted run cancelled.

        Cooperative: it stops the *next* node from being scheduled.  A node
        already executing runs to completion, and any external effect it has
        already caused stands.
        """
        state = self.store.load(run_id)
        if state is None:
            raise KeyError(f"No persisted workflow run {run_id!r}")
        state.status = RunLifecycle.CANCELLED
        state.error = reason
        state.updated_at = time.time()
        self.store.save(state)
        return state

    def state(self, run_id: str) -> WorkflowRunState | None:
        return self.store.load(run_id)

    # ---- internals ----------------------------------------------------

    def _resolve_operations(self, state: WorkflowRunState, reconcile: Reconciler | None) -> None:
        """Classify or resolve operations that were dispatched but never confirmed."""
        ledger = state.ledger()
        unsettled = ledger.unsettled()
        if not unsettled:
            return

        if reconcile is None:
            for record in unsettled:
                ledger.mark_reconcile_required(record.operation_id)
            state.operations = ledger.to_list()
            state.status = RunLifecycle.RECONCILE_REQUIRED
            state.updated_at = time.time()
            self.store.save(state)
            raise ReconciliationRequired(state.run_id, [r.operation_id for r in unsettled])

        still_unknown: list[str] = []
        for record in unsettled:
            outcome = reconcile(record)
            if isinstance(outcome, tuple):
                verdict, recovered = outcome
            else:
                verdict, recovered = outcome, record.result
            if verdict is OperationStatus.SUCCEEDED:
                ledger.succeed(record.operation_id, recovered)
            elif verdict is OperationStatus.FAILED:
                ledger.fail(record.operation_id, record.error or "reconciled as failed")
            else:
                ledger.mark_reconcile_required(record.operation_id)
                still_unknown.append(record.operation_id)
        state.operations = ledger.to_list()
        state.updated_at = time.time()
        self.store.save(state)

        if still_unknown:
            state.status = RunLifecycle.RECONCILE_REQUIRED
            self.store.save(state)
            raise ReconciliationRequired(state.run_id, still_unknown)

    def _drive(
        self,
        graph: Graph,
        state: WorkflowRunState,
        *,
        options: dict[str, Any] | None,
    ) -> RunResult:
        """The scheduling loop.  Persists after every node."""
        started = time.perf_counter()
        state.attempts += 1
        state.status = RunLifecycle.RUNNING
        self.store.save(state)

        order = graph.topo_order()

        # Only *completed* work is carried forward. A node that failed, or was
        # skipped because something upstream of it failed, is retried — resuming
        # a run that inherited its own previous failure would skip everything
        # and report the same failure forever.
        state.node_results = {
            nid: data for nid, data in state.node_results.items() if data.get("status") == NodeStatus.COMPLETED.value
        }
        results: dict[str, NodeResult] = {nid: _node_result_from_dict(data) for nid, data in state.node_results.items()}
        state.error = None
        errors: list[tuple[str, str]] = []
        any_failed = False

        def persist_operations(operations: list[dict[str, Any]]) -> None:
            state.operations = operations
            state.updated_at = time.time()
            self.store.save(state)

        ledger = OperationLedger.from_list(state.operations, on_change=persist_operations)
        run_options = dict(options or {})
        run_options.setdefault("run_id", state.run_id)
        run_options.setdefault("operation_ledger", ledger)
        run_options.setdefault("human_input", dict(state.human_input))

        for nid in order:
            if nid in results:
                continue  # completed in a previous attempt — never re-run

            # Cancellation is checked between nodes, from the store, so another
            # process can stop a long run.
            latest = self.store.load(state.run_id)
            if latest is not None and latest.status is RunLifecycle.CANCELLED:
                state.status = RunLifecycle.CANCELLED
                state.error = latest.error
                state.operations = ledger.to_list()
                self.store.save(state)
                return self._to_run_result(graph, state, results, errors, started, RunStatus.FAILED)

            preds = graph.predecessors(nid)
            blocked = any(p in results and not results[p].status.is_success for p in preds)
            if blocked or (any_failed and self.error_policy is ErrorPolicy.fail_fast):
                result = NodeResult(nid, NodeStatus.SKIPPED)
                results[nid] = result
                state.record_node(result)
                state.operations = ledger.to_list()
                self.store.save(state)
                continue

            _emit(self.callbacks, "on_node_start", nid)
            upstream = {p: results[p].outputs for p in preds if p in results}
            node_started = time.perf_counter()
            try:
                result = run_node(graph, nid, inputs=state.inputs, upstream=upstream, options=run_options)
            except Exception as exc:
                # ``run_node`` wraps a runner's exception in NodeExecutionError,
                # so the pause signal arrives as the __cause__.
                cause = exc.__cause__ if isinstance(exc, WorkflowError) and exc.__cause__ else exc
                if isinstance(cause, HumanInputRequired):
                    state.status = RunLifecycle.PAUSED
                    state.paused_node = nid
                    state.pause_prompt = cause.prompt
                    state.pause_schema = dict(cause.schema)
                    state.operations = ledger.to_list()
                    state.updated_at = time.time()
                    self.store.save(state)
                    return self._to_run_result(graph, state, results, errors, started, RunStatus.PENDING)
                result = NodeResult(
                    nid,
                    NodeStatus.FAILED,
                    error=str(cause),
                    error_type=type(cause).__name__,
                    elapsed_ms=(time.perf_counter() - node_started) * 1000,
                )
                results[nid] = result
                state.record_node(result)
                state.operations = ledger.to_list()
                errors.append((nid, str(cause)))
                any_failed = True
                state.error = state.error or str(cause)
                state.status = RunLifecycle.FAILED
                self.store.save(state)
                _emit(self.callbacks, "on_node_error", nid, exc)
                if self.error_policy is ErrorPolicy.raise_on_error:
                    raise
                continue

            results[nid] = result
            state.record_node(result)
            state.operations = ledger.to_list()
            self.store.save(state)
            _emit(self.callbacks, "on_node_complete", nid, result)

        status = RunStatus.FAILED if any_failed else RunStatus.COMPLETED
        state.status = RunLifecycle.FAILED if any_failed else RunLifecycle.COMPLETED
        state.pending = []
        state.operations = ledger.to_list()
        state.updated_at = time.time()
        self.store.save(state)
        return self._to_run_result(graph, state, results, errors, started, status)

    def _to_run_result(
        self,
        graph: Graph,
        state: WorkflowRunState,
        results: dict[str, NodeResult],
        errors: list[tuple[str, str]],
        started: float,
        status: RunStatus,
    ) -> RunResult:
        # Output collection goes through the engine's own resolver, seeded with
        # the persisted results, so a resumed run renders its outputs exactly
        # the way an uninterrupted one would.
        from .scheduler import ExecutionContext, _collect_outputs
        from .state import NodeResultStore

        store = NodeResultStore()
        for nid, result in results.items():
            store.set(nid, result)
        ctx = ExecutionContext(graph=graph, inputs=state.inputs, store=store)
        outputs = _collect_outputs(graph, ctx, results)

        run = RunResult(
            graph_id=graph.id,
            status=status,
            node_results=results,
            outputs=outputs,
            total_usage=state.recompute_usage(),
            order=graph.topo_order(),
            errors=errors,
            error=state.error or (errors[0][1] if errors else None),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        _emit(self.callbacks, "on_graph_complete", run)
        return run


def _node_result_from_dict(data: dict[str, Any]) -> NodeResult:
    return NodeResult(
        node_id=data["node_id"],
        status=NodeStatus(data.get("status", NodeStatus.PENDING.value)),
        outputs=dict(data.get("outputs") or {}),
        usage=dict(data.get("usage") or {}),
        cost=float(data.get("cost", 0.0) or 0.0),
        strategy=data.get("strategy"),
        error=data.get("error"),
        error_type=data.get("error_type"),
        elapsed_ms=float(data.get("elapsed_ms", 0.0) or 0.0),
    )


def _emit(callbacks: WorkflowCallbacks | None, event: str, *args: Any) -> None:
    if callbacks is None:
        return
    callback = getattr(callbacks, event, None)
    if callback is None:
        return
    try:
        callback(*args)
    except Exception:  # pragma: no cover - a bad hook must not break a run
        logger.debug("workflow callback %s raised", event, exc_info=True)
