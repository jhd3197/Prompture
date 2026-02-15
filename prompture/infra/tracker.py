"""Unified event-level usage tracker for Prompture.

Records every LLM call as an individual event with full context
(model, tokens, cost, timing, conversation, agent, tool, custom tags)
in a local SQLite database at ``~/.prompture/usage/usage.db``.

Context propagation uses Python's :mod:`contextvars` so that nesting
``session()``, ``conversation()``, ``agent()``, ``tool()``, and
``operation()`` scopes automatically tags child events without
changing function signatures.

External tools (CachiBot, AgentSite, etc.) can query the SQLite
database directly for aggregation, budgeting, and dashboards.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("prompture.tracker")

# ---------------------------------------------------------------------------
# Context variables for scope propagation
# ---------------------------------------------------------------------------

_ctx_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "usage_session_id", default=None
)
_ctx_conversation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "usage_conversation_id", default=None
)
_ctx_agent_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "usage_agent_id", default=None
)
_ctx_tool_name: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "usage_tool_name", default=None
)
_ctx_operation: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "usage_operation", default=None
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class UsageEvent:
    """A single LLM call event."""

    id: str = ""
    timestamp: str = ""
    model_name: str = ""
    provider: str = ""
    api_key_hash: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    elapsed_ms: float = 0.0
    session_id: str | None = None
    conversation_id: str | None = None
    agent_id: str | None = None
    tool_name: str | None = None
    operation: str | None = None
    cache_hit: bool = False
    status: str = "success"
    error_type: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class UsageSummary:
    """Aggregated usage statistics."""

    total_events: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_elapsed_ms: float = 0.0
    models: dict[str, float] = field(default_factory=dict)
    providers: dict[str, float] = field(default_factory=dict)


@dataclass
class BudgetStatus:
    """Result of a budget check."""

    scope: str = ""
    limit_cost: float | None = None
    limit_tokens: int | None = None
    current_cost: float = 0.0
    current_tokens: int = 0
    exceeded: bool = False
    remaining_cost: float | None = None
    remaining_tokens: int | None = None


class BudgetExceededError(Exception):
    """Raised when a pre-call budget check fails."""

    def __init__(self, status: BudgetStatus) -> None:
        self.status = status
        super().__init__(
            f"Budget exceeded for scope '{status.scope}': "
            f"cost ${status.current_cost:.4f} / ${status.limit_cost:.4f}"
            if status.limit_cost is not None
            else f"Budget exceeded for scope '{status.scope}': "
            f"tokens {status.current_tokens:,} / {status.limit_tokens:,}"
        )


# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS usage_events (
    id                TEXT PRIMARY KEY,
    timestamp         TEXT NOT NULL,
    model_name        TEXT NOT NULL,
    provider          TEXT NOT NULL,
    api_key_hash      TEXT DEFAULT '',
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens      INTEGER DEFAULT 0,
    cost              REAL DEFAULT 0.0,
    elapsed_ms        REAL DEFAULT 0.0,
    session_id        TEXT,
    conversation_id   TEXT,
    agent_id          TEXT,
    tool_name         TEXT,
    operation         TEXT,
    cache_hit         INTEGER DEFAULT 0,
    status            TEXT DEFAULT 'success',
    error_type        TEXT,
    tags              TEXT,
    metadata          TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON usage_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_model ON usage_events(model_name);
CREATE INDEX IF NOT EXISTS idx_events_provider ON usage_events(provider);
CREATE INDEX IF NOT EXISTS idx_events_session ON usage_events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_conversation ON usage_events(conversation_id);
CREATE INDEX IF NOT EXISTS idx_events_agent ON usage_events(agent_id);

CREATE TABLE IF NOT EXISTS usage_budgets (
    scope        TEXT PRIMARY KEY,
    limit_cost   REAL,
    limit_tokens INTEGER,
    period       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

-- Pre-built views for external tools
CREATE VIEW IF NOT EXISTS daily_costs AS
    SELECT date(timestamp) AS day,
           SUM(cost) AS total_cost,
           SUM(total_tokens) AS total_tokens,
           COUNT(*) AS event_count
    FROM usage_events
    GROUP BY date(timestamp);

CREATE VIEW IF NOT EXISTS model_costs AS
    SELECT model_name,
           SUM(cost) AS total_cost,
           SUM(prompt_tokens) AS total_prompt_tokens,
           SUM(completion_tokens) AS total_completion_tokens,
           SUM(total_tokens) AS total_tokens,
           COUNT(*) AS event_count
    FROM usage_events
    GROUP BY model_name;

CREATE VIEW IF NOT EXISTS provider_costs AS
    SELECT provider,
           SUM(cost) AS total_cost,
           SUM(total_tokens) AS total_tokens,
           COUNT(*) AS event_count
    FROM usage_events
    GROUP BY provider;

CREATE VIEW IF NOT EXISTS conversation_costs AS
    SELECT conversation_id,
           SUM(cost) AS total_cost,
           SUM(total_tokens) AS total_tokens,
           COUNT(*) AS event_count
    FROM usage_events
    WHERE conversation_id IS NOT NULL
    GROUP BY conversation_id;

CREATE VIEW IF NOT EXISTS agent_costs AS
    SELECT agent_id,
           SUM(cost) AS total_cost,
           SUM(total_tokens) AS total_tokens,
           COUNT(*) AS event_count
    FROM usage_events
    WHERE agent_id IS NOT NULL
    GROUP BY agent_id;

-- Backward-compat view matching old model_ledger schema
CREATE VIEW IF NOT EXISTS model_usage AS
    SELECT model_name,
           api_key_hash,
           COUNT(*) AS use_count,
           SUM(total_tokens) AS total_tokens,
           SUM(cost) AS total_cost,
           MIN(timestamp) AS first_used,
           MAX(timestamp) AS last_used,
           'success' AS last_status
    FROM usage_events
    GROUP BY model_name, api_key_hash;
"""

_INSERT_SQL = """
INSERT INTO usage_events (
    id, timestamp, model_name, provider, api_key_hash,
    prompt_tokens, completion_tokens, total_tokens, cost, elapsed_ms,
    session_id, conversation_id, agent_id, tool_name, operation,
    cache_hit, status, error_type, tags, metadata
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


# ---------------------------------------------------------------------------
# UsageTracker
# ---------------------------------------------------------------------------


class UsageTracker:
    """Event-based SQLite usage tracker.

    Thread-safe.  Writes are batched for performance and flushed when
    the buffer reaches ``flush_threshold`` or on explicit ``flush()``.

    Args:
        db_path: Path to the SQLite database. Defaults to
            ``~/.prompture/usage/usage.db``.
        enabled: Whether tracking is active.  When ``False``, ``record()``
            is a no-op.
        flush_threshold: Number of events to buffer before auto-flushing.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        enabled: bool = True,
        flush_threshold: int = 10,
    ) -> None:
        default_dir = Path.home() / ".prompture" / "usage"
        self._db_path = Path(db_path) if db_path else default_dir / "usage.db"
        self._enabled = enabled
        self._flush_threshold = max(1, flush_threshold)
        self._buffer: list[UsageEvent] = []
        self._lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._initialized = False

    # ------------------------------------------------------------------ #
    # Lazy init
    # ------------------------------------------------------------------ #

    def _ensure_db(self) -> None:
        """Create the database and schema on first use.

        Uses an independent ``_init_lock`` to avoid deadlocking with
        ``_lock`` (which guards the event buffer and may already be held
        when ``_flush_locked`` triggers lazy init via ``_connect``).
        """
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            try:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(self._db_path))
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.executescript(_SCHEMA_SQL)
                    conn.commit()
                finally:
                    conn.close()
                self._initialized = True
            except Exception:
                logger.debug("Failed to initialize usage tracker DB", exc_info=True)

    def _connect(self) -> sqlite3.Connection:
        self._ensure_db()
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record(self, event: UsageEvent) -> None:
        """Record a usage event.  Fire-and-forget — never raises."""
        if not self._enabled:
            return
        try:
            # Inject context vars if not already set on the event
            if event.session_id is None:
                event.session_id = _ctx_session_id.get()
            if event.conversation_id is None:
                event.conversation_id = _ctx_conversation_id.get()
            if event.agent_id is None:
                event.agent_id = _ctx_agent_id.get()
            if event.tool_name is None:
                event.tool_name = _ctx_tool_name.get()
            if event.operation is None:
                event.operation = _ctx_operation.get()

            with self._lock:
                self._buffer.append(event)
                if len(self._buffer) >= self._flush_threshold:
                    self._flush_locked()
        except Exception:
            logger.debug("Failed to record usage event", exc_info=True)

    def flush(self) -> None:
        """Flush buffered events to disk."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Flush buffer while holding _lock.  Never raises."""
        if not self._buffer:
            return
        events = list(self._buffer)
        self._buffer.clear()
        try:
            conn = self._connect()
            try:
                conn.executemany(
                    _INSERT_SQL,
                    [
                        (
                            e.id,
                            e.timestamp,
                            e.model_name,
                            e.provider,
                            e.api_key_hash,
                            e.prompt_tokens,
                            e.completion_tokens,
                            e.total_tokens,
                            e.cost,
                            e.elapsed_ms,
                            e.session_id,
                            e.conversation_id,
                            e.agent_id,
                            e.tool_name,
                            e.operation,
                            1 if e.cache_hit else 0,
                            e.status,
                            e.error_type,
                            json.dumps(e.tags) if e.tags else None,
                            json.dumps(e.metadata) if e.metadata else None,
                        )
                        for e in events
                    ],
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logger.debug("Failed to flush usage events to DB", exc_info=True)

    # ------------------------------------------------------------------ #
    # Context managers (scope propagation via contextvars)
    # ------------------------------------------------------------------ #

    @contextlib.contextmanager
    def session(self, session_id: str | None = None):
        """Set the session scope for nested calls."""
        sid = session_id or str(uuid.uuid4())
        token = _ctx_session_id.set(sid)
        try:
            yield sid
        finally:
            _ctx_session_id.reset(token)

    @contextlib.contextmanager
    def conversation(self, conversation_id: str):
        """Set the conversation scope for nested calls."""
        token = _ctx_conversation_id.set(conversation_id)
        try:
            yield conversation_id
        finally:
            _ctx_conversation_id.reset(token)

    @contextlib.contextmanager
    def agent(self, agent_id: str):
        """Set the agent scope for nested calls."""
        token = _ctx_agent_id.set(agent_id)
        try:
            yield agent_id
        finally:
            _ctx_agent_id.reset(token)

    @contextlib.contextmanager
    def tool(self, tool_name: str):
        """Set the tool scope for nested calls."""
        token = _ctx_tool_name.set(tool_name)
        try:
            yield tool_name
        finally:
            _ctx_tool_name.reset(token)

    @contextlib.contextmanager
    def operation(self, operation_name: str):
        """Set the operation scope for nested calls."""
        token = _ctx_operation.set(operation_name)
        try:
            yield operation_name
        finally:
            _ctx_operation.reset(token)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def query(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
        agent_id: str | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Query usage events with filters."""
        self.flush()
        conditions: list[str] = []
        params: list[Any] = []

        if start is not None:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            conditions.append("timestamp <= ?")
            params.append(end)
        if model is not None:
            conditions.append("model_name = ?")
            params.append(model)
        if provider is not None:
            conditions.append("provider = ?")
            params.append(provider)
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        if conversation_id is not None:
            conditions.append("conversation_id = ?")
            params.append(conversation_id)
        if agent_id is not None:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM usage_events WHERE {where} ORDER BY timestamp DESC LIMIT ?"  # nosec B608
        params.append(limit)

        try:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()
        except Exception:
            logger.debug("Failed to query usage events", exc_info=True)
            return []

    def summary(self, **filters: Any) -> UsageSummary:
        """Get aggregated usage summary with optional filters."""
        events = self.query(**filters)
        s = UsageSummary()
        s.total_events = len(events)
        for e in events:
            s.total_prompt_tokens += e.get("prompt_tokens", 0)
            s.total_completion_tokens += e.get("completion_tokens", 0)
            s.total_tokens += e.get("total_tokens", 0)
            s.total_cost += e.get("cost", 0.0)
            s.total_elapsed_ms += e.get("elapsed_ms", 0.0)

            model = e.get("model_name", "")
            s.models[model] = s.models.get(model, 0.0) + e.get("cost", 0.0)

            prov = e.get("provider", "")
            s.providers[prov] = s.providers.get(prov, 0.0) + e.get("cost", 0.0)

        return s

    def cost_today(self) -> float:
        """Total cost for today (UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._sum_cost_for_date_prefix(today)

    def cost_this_month(self) -> float:
        """Total cost for the current month (UTC)."""
        month_prefix = datetime.now(timezone.utc).strftime("%Y-%m")
        return self._sum_cost_for_date_prefix(month_prefix)

    def _sum_cost_for_date_prefix(self, prefix: str) -> float:
        """Sum costs for events whose timestamp starts with *prefix*."""
        self.flush()
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COALESCE(SUM(cost), 0) FROM usage_events WHERE timestamp LIKE ?",
                    (f"{prefix}%",),
                ).fetchone()
                return float(row[0]) if row else 0.0
            finally:
                conn.close()
        except Exception:
            logger.debug("Failed to sum costs", exc_info=True)
            return 0.0

    def cost_by_model(self) -> dict[str, float]:
        """Cost breakdown by model name."""
        self.flush()
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT model_name, SUM(cost) AS total FROM usage_events GROUP BY model_name"
                ).fetchall()
                return {r["model_name"]: r["total"] for r in rows}
            finally:
                conn.close()
        except Exception:
            logger.debug("Failed to get cost by model", exc_info=True)
            return {}

    def cost_by_provider(self) -> dict[str, float]:
        """Cost breakdown by provider."""
        self.flush()
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT provider, SUM(cost) AS total FROM usage_events GROUP BY provider"
                ).fetchall()
                return {r["provider"]: r["total"] for r in rows}
            finally:
                conn.close()
        except Exception:
            logger.debug("Failed to get cost by provider", exc_info=True)
            return {}

    # ------------------------------------------------------------------ #
    # Budget management
    # ------------------------------------------------------------------ #

    def set_budget(
        self,
        scope: str,
        *,
        limit_cost: float | None = None,
        limit_tokens: int | None = None,
        period: str = "monthly",
    ) -> None:
        """Set or update a budget for a scope (e.g. 'global', 'agent:pm')."""
        self._ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        try:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO usage_budgets (scope, limit_cost, limit_tokens, period, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(scope) DO UPDATE SET
                        limit_cost = excluded.limit_cost,
                        limit_tokens = excluded.limit_tokens,
                        period = excluded.period,
                        updated_at = excluded.updated_at
                    """,
                    (scope, limit_cost, limit_tokens, period, now, now),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logger.debug("Failed to set budget for scope %s", scope, exc_info=True)

    def check_budget(self, scope: str = "global") -> BudgetStatus:
        """Check budget status for a scope."""
        self.flush()
        status = BudgetStatus(scope=scope)
        try:
            conn = self._connect()
            try:
                # Get budget
                budget_row = conn.execute(
                    "SELECT * FROM usage_budgets WHERE scope = ?", (scope,)
                ).fetchone()
                if budget_row is None:
                    return status

                status.limit_cost = budget_row["limit_cost"]
                status.limit_tokens = budget_row["limit_tokens"]
                period = budget_row["period"]

                # Get current usage for the period
                if period == "daily":
                    prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                elif period == "monthly":
                    prefix = datetime.now(timezone.utc).strftime("%Y-%m")
                else:
                    prefix = ""  # all time

                if prefix:
                    usage_row = conn.execute(
                        "SELECT COALESCE(SUM(cost), 0) AS total_cost, "
                        "COALESCE(SUM(total_tokens), 0) AS total_tokens "
                        "FROM usage_events WHERE timestamp LIKE ?",
                        (f"{prefix}%",),
                    ).fetchone()
                else:
                    usage_row = conn.execute(
                        "SELECT COALESCE(SUM(cost), 0) AS total_cost, "
                        "COALESCE(SUM(total_tokens), 0) AS total_tokens "
                        "FROM usage_events"
                    ).fetchone()

                status.current_cost = float(usage_row["total_cost"])
                status.current_tokens = int(usage_row["total_tokens"])

                # Check exceeded
                if status.limit_cost is not None:
                    status.remaining_cost = status.limit_cost - status.current_cost
                    if status.current_cost >= status.limit_cost:
                        status.exceeded = True
                if status.limit_tokens is not None:
                    status.remaining_tokens = status.limit_tokens - status.current_tokens
                    if status.current_tokens >= status.limit_tokens:
                        status.exceeded = True

            finally:
                conn.close()
        except Exception:
            logger.debug("Failed to check budget for scope %s", scope, exc_info=True)

        return status

    # ------------------------------------------------------------------ #
    # Public cost calculation API
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_cost(
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate USD cost for a model call.

        This is the **public** API for cost calculation, replacing
        the private ``CostMixin._calculate_cost()`` that external
        consumers previously depended on.

        Resolution order:
        1. Live rates from ``model_rates.get_model_rates()`` (per 1M tokens).
        2. Zero if no rate data is available.
        """
        try:
            from .model_rates import get_model_rates

            provider = model_name.split("/", 1)[0] if "/" in model_name else model_name
            model = model_name.split("/", 1)[1] if "/" in model_name else model_name

            rates = get_model_rates(provider, model)
            if rates and (rates.get("input") or rates.get("output")):
                prompt_cost = (input_tokens / 1_000_000) * rates["input"]
                completion_cost = (output_tokens / 1_000_000) * rates["output"]
                return round(prompt_cost + completion_cost, 6)
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------ #
    # DriverCallbacks integration
    # ------------------------------------------------------------------ #

    def as_callbacks(self, **context: Any):
        """Return a :class:`DriverCallbacks` wired to this tracker.

        Extra keyword arguments are stored as default context on
        each recorded event (e.g. ``session_id``, ``agent_id``).
        """
        from .callbacks import DriverCallbacks

        tracker = self
        ctx = context

        def _on_response(info: dict[str, Any]) -> None:
            meta = info.get("meta", {})
            driver = info.get("driver", "")
            # Parse provider/model from driver string
            if "/" in driver:
                provider, model = driver.split("/", 1)
            else:
                provider = driver
                model = driver

            event = UsageEvent(
                model_name=f"{provider}/{model}" if provider != model else model,
                provider=provider,
                prompt_tokens=meta.get("prompt_tokens", 0),
                completion_tokens=meta.get("completion_tokens", 0),
                total_tokens=meta.get("total_tokens", 0),
                cost=meta.get("cost", 0.0),
                elapsed_ms=info.get("elapsed_ms", 0.0),
                session_id=ctx.get("session_id"),
                conversation_id=ctx.get("conversation_id"),
                agent_id=ctx.get("agent_id"),
            )
            tracker.record(event)

        def _on_error(info: dict[str, Any]) -> None:
            driver = info.get("driver", "")
            error = info.get("error")
            if "/" in driver:
                provider, model = driver.split("/", 1)
            else:
                provider = driver
                model = driver

            event = UsageEvent(
                model_name=f"{provider}/{model}" if provider != model else model,
                provider=provider,
                status="error",
                error_type=type(error).__name__ if error else None,
                session_id=ctx.get("session_id"),
                conversation_id=ctx.get("conversation_id"),
                agent_id=ctx.get("agent_id"),
            )
            tracker.record(event)

        return DriverCallbacks(on_response=_on_response, on_error=_on_error)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_tracker: UsageTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> UsageTracker:
    """Return the module-level singleton tracker (lazily created)."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = UsageTracker()
    return _tracker


def configure_tracker(
    *,
    enabled: bool = True,
    db_path: str | None = None,
    flush_threshold: int = 10,
) -> UsageTracker:
    """Configure (or reconfigure) the global tracker singleton."""
    global _tracker
    with _tracker_lock:
        _tracker = UsageTracker(
            db_path=db_path,
            enabled=enabled,
            flush_threshold=flush_threshold,
        )
    return _tracker
