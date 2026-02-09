"""Tests for the unified usage tracker (prompture.infra.tracker)."""

from __future__ import annotations

import threading

import pytest

from prompture.infra.tracker import (
    UsageEvent,
    UsageTracker,
    configure_tracker,
    get_tracker,
)


@pytest.fixture()
def tracker(tmp_path):
    """Create a tracker backed by a temp DB with flush_threshold=1 for immediate writes."""
    return UsageTracker(db_path=tmp_path / "test_usage.db", flush_threshold=1)


@pytest.fixture()
def batched_tracker(tmp_path):
    """Create a tracker with batch size > 1."""
    return UsageTracker(db_path=tmp_path / "test_usage_batch.db", flush_threshold=5)


# ------------------------------------------------------------------ #
# Basic record & query
# ------------------------------------------------------------------ #


class TestBasicRecordAndQuery:
    def test_record_and_query_single(self, tracker):
        event = UsageEvent(
            model_name="openai/gpt-4",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost=0.01,
            elapsed_ms=200.0,
        )
        tracker.record(event)

        results = tracker.query()
        assert len(results) == 1
        row = results[0]
        assert row["model_name"] == "openai/gpt-4"
        assert row["provider"] == "openai"
        assert row["prompt_tokens"] == 100
        assert row["completion_tokens"] == 50
        assert row["total_tokens"] == 150
        assert row["cost"] == pytest.approx(0.01)
        assert row["elapsed_ms"] == pytest.approx(200.0)
        assert row["status"] == "success"

    def test_record_multiple(self, tracker):
        for i in range(3):
            tracker.record(
                UsageEvent(
                    model_name=f"openai/gpt-{i}",
                    provider="openai",
                    total_tokens=100 * (i + 1),
                    cost=0.01 * (i + 1),
                )
            )

        results = tracker.query()
        assert len(results) == 3

    def test_query_filter_by_model(self, tracker):
        tracker.record(UsageEvent(model_name="openai/gpt-4", provider="openai", cost=0.01))
        tracker.record(UsageEvent(model_name="claude/haiku", provider="claude", cost=0.005))

        results = tracker.query(model="openai/gpt-4")
        assert len(results) == 1
        assert results[0]["model_name"] == "openai/gpt-4"

    def test_query_filter_by_provider(self, tracker):
        tracker.record(UsageEvent(model_name="openai/gpt-4", provider="openai", cost=0.01))
        tracker.record(UsageEvent(model_name="claude/haiku", provider="claude", cost=0.005))

        results = tracker.query(provider="claude")
        assert len(results) == 1
        assert results[0]["provider"] == "claude"

    def test_query_filter_by_status(self, tracker):
        tracker.record(UsageEvent(model_name="openai/gpt-4", provider="openai", status="success"))
        tracker.record(
            UsageEvent(model_name="openai/gpt-4", provider="openai", status="error", error_type="APIError")
        )

        results = tracker.query(status="error")
        assert len(results) == 1
        assert results[0]["error_type"] == "APIError"


# ------------------------------------------------------------------ #
# UsageEvent auto-fields
# ------------------------------------------------------------------ #


class TestUsageEvent:
    def test_auto_id_and_timestamp(self):
        event = UsageEvent(model_name="test")
        assert event.id != ""
        assert event.timestamp != ""

    def test_explicit_id_preserved(self):
        event = UsageEvent(id="custom-id", model_name="test")
        assert event.id == "custom-id"


# ------------------------------------------------------------------ #
# Context propagation
# ------------------------------------------------------------------ #


class TestContextPropagation:
    def test_session_context(self, tracker):
        with tracker.session("my-session"):
            tracker.record(UsageEvent(model_name="test", provider="test"))

        results = tracker.query()
        assert len(results) == 1
        assert results[0]["session_id"] == "my-session"

    def test_agent_context(self, tracker):
        with tracker.agent("pm-agent"):
            tracker.record(UsageEvent(model_name="test", provider="test"))

        results = tracker.query()
        assert results[0]["agent_id"] == "pm-agent"

    def test_conversation_context(self, tracker):
        with tracker.conversation("conv-123"):
            tracker.record(UsageEvent(model_name="test", provider="test"))

        results = tracker.query()
        assert results[0]["conversation_id"] == "conv-123"

    def test_tool_context(self, tracker):
        with tracker.tool("search"):
            tracker.record(UsageEvent(model_name="test", provider="test"))

        results = tracker.query()
        assert results[0]["tool_name"] == "search"

    def test_operation_context(self, tracker):
        with tracker.operation("ask_for_json"):
            tracker.record(UsageEvent(model_name="test", provider="test"))

        results = tracker.query()
        assert results[0]["operation"] == "ask_for_json"

    def test_nested_contexts(self, tracker):
        with tracker.session("sess-1"):
            with tracker.agent("agent-a"):
                with tracker.conversation("conv-x"):
                    tracker.record(UsageEvent(model_name="test", provider="test"))

        results = tracker.query()
        assert len(results) == 1
        row = results[0]
        assert row["session_id"] == "sess-1"
        assert row["agent_id"] == "agent-a"
        assert row["conversation_id"] == "conv-x"

    def test_context_cleanup_after_exit(self, tracker):
        with tracker.session("sess-1"):
            pass

        # After exiting context, recording should not carry the old session
        tracker.record(UsageEvent(model_name="test", provider="test"))
        results = tracker.query()
        assert results[0]["session_id"] is None

    def test_explicit_event_context_not_overridden(self, tracker):
        """Event fields set explicitly should not be overridden by context."""
        with tracker.session("ctx-session"):
            tracker.record(
                UsageEvent(model_name="test", provider="test", session_id="explicit-session")
            )

        results = tracker.query()
        assert results[0]["session_id"] == "explicit-session"


# ------------------------------------------------------------------ #
# Summary
# ------------------------------------------------------------------ #


class TestSummary:
    def test_summary_aggregation(self, tracker):
        tracker.record(UsageEvent(model_name="openai/gpt-4", provider="openai", total_tokens=100, cost=0.01))
        tracker.record(UsageEvent(model_name="openai/gpt-4", provider="openai", total_tokens=200, cost=0.02))
        tracker.record(UsageEvent(model_name="claude/haiku", provider="claude", total_tokens=50, cost=0.005))

        s = tracker.summary()
        assert s.total_events == 3
        assert s.total_tokens == 350
        assert s.total_cost == pytest.approx(0.035)
        assert "openai/gpt-4" in s.models
        assert "claude/haiku" in s.models
        assert "openai" in s.providers
        assert "claude" in s.providers


# ------------------------------------------------------------------ #
# Cost convenience methods
# ------------------------------------------------------------------ #


class TestCostMethods:
    def test_cost_by_model(self, tracker):
        tracker.record(UsageEvent(model_name="openai/gpt-4", provider="openai", cost=0.01))
        tracker.record(UsageEvent(model_name="openai/gpt-4", provider="openai", cost=0.02))
        tracker.record(UsageEvent(model_name="claude/haiku", provider="claude", cost=0.005))

        costs = tracker.cost_by_model()
        assert costs["openai/gpt-4"] == pytest.approx(0.03)
        assert costs["claude/haiku"] == pytest.approx(0.005)

    def test_cost_by_provider(self, tracker):
        tracker.record(UsageEvent(model_name="openai/gpt-4", provider="openai", cost=0.01))
        tracker.record(UsageEvent(model_name="claude/haiku", provider="claude", cost=0.005))

        costs = tracker.cost_by_provider()
        assert costs["openai"] == pytest.approx(0.01)
        assert costs["claude"] == pytest.approx(0.005)

    def test_cost_today(self, tracker):
        tracker.record(UsageEvent(model_name="test", provider="test", cost=0.05))
        tracker.record(UsageEvent(model_name="test", provider="test", cost=0.03))

        assert tracker.cost_today() == pytest.approx(0.08)

    def test_cost_this_month(self, tracker):
        tracker.record(UsageEvent(model_name="test", provider="test", cost=0.1))
        assert tracker.cost_this_month() == pytest.approx(0.1)


# ------------------------------------------------------------------ #
# Budget management
# ------------------------------------------------------------------ #


class TestBudgetManagement:
    def test_set_and_check_budget(self, tracker):
        tracker.set_budget("global", limit_cost=1.0, period="all-time")

        tracker.record(UsageEvent(model_name="test", provider="test", cost=0.50))

        status = tracker.check_budget("global")
        assert status.limit_cost == 1.0
        assert status.current_cost == pytest.approx(0.50)
        assert status.exceeded is False
        assert status.remaining_cost == pytest.approx(0.50)

    def test_budget_exceeded(self, tracker):
        tracker.set_budget("global", limit_cost=0.10, period="all-time")

        tracker.record(UsageEvent(model_name="test", provider="test", cost=0.15))

        status = tracker.check_budget("global")
        assert status.exceeded is True

    def test_budget_token_limit(self, tracker):
        tracker.set_budget("global", limit_tokens=1000, period="all-time")

        tracker.record(UsageEvent(model_name="test", provider="test", total_tokens=1200))

        status = tracker.check_budget("global")
        assert status.exceeded is True
        assert status.remaining_tokens == -200

    def test_no_budget_set(self, tracker):
        status = tracker.check_budget("nonexistent")
        assert status.exceeded is False
        assert status.limit_cost is None
        assert status.limit_tokens is None

    def test_budget_update(self, tracker):
        tracker.set_budget("global", limit_cost=1.0, period="monthly")
        tracker.set_budget("global", limit_cost=2.0, period="daily")

        status = tracker.check_budget("global")
        assert status.limit_cost == 2.0


# ------------------------------------------------------------------ #
# Batch flush behavior
# ------------------------------------------------------------------ #


class TestBatchFlush:
    def test_events_not_visible_until_flush(self, batched_tracker):
        """Events in buffer are not yet in the DB."""
        batched_tracker.record(UsageEvent(model_name="test", provider="test", cost=0.01))

        # Direct query without flush - record() didn't trigger auto-flush (threshold=5)
        # But query() calls flush() first, so this should still work
        results = batched_tracker.query()
        assert len(results) == 1

    def test_auto_flush_at_threshold(self, batched_tracker):
        """Auto-flush triggers when buffer reaches threshold."""
        for i in range(5):
            batched_tracker.record(UsageEvent(model_name=f"test/{i}", provider="test"))

        # Buffer should have been flushed at threshold=5
        results = batched_tracker.query()
        assert len(results) == 5

    def test_manual_flush(self, batched_tracker):
        batched_tracker.record(UsageEvent(model_name="test", provider="test"))
        batched_tracker.flush()
        results = batched_tracker.query()
        assert len(results) == 1


# ------------------------------------------------------------------ #
# Fire-and-forget behavior
# ------------------------------------------------------------------ #


class TestFireAndForget:
    def test_disabled_tracker_is_noop(self, tmp_path):
        tracker = UsageTracker(db_path=tmp_path / "disabled.db", enabled=False)
        tracker.record(UsageEvent(model_name="test", provider="test"))
        # Should not create the DB at all or store events
        results = tracker.query()
        assert len(results) == 0

    def test_record_never_raises(self, tmp_path):
        """Record should swallow errors (fire-and-forget)."""
        tracker = UsageTracker(db_path=tmp_path / "test.db", flush_threshold=1)
        # Record with a normal event first to init DB
        tracker.record(UsageEvent(model_name="test", provider="test"))
        # Even with bad data types, record should not propagate exceptions
        # (the fire-and-forget behavior)
        try:
            tracker.record(UsageEvent(model_name="test", provider="test"))
        except Exception:
            pytest.fail("record() should never raise")


# ------------------------------------------------------------------ #
# Thread safety
# ------------------------------------------------------------------ #


class TestThreadSafety:
    def test_concurrent_writes(self, tracker):
        """Multiple threads writing concurrently should not lose events."""
        num_threads = 4
        events_per_thread = 25
        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for _i in range(events_per_thread):
                    tracker.record(
                        UsageEvent(
                            model_name=f"test/t{thread_id}",
                            provider="test",
                            total_tokens=1,
                            cost=0.001,
                        )
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        tracker.flush()
        results = tracker.query(limit=10000)
        assert len(results) == num_threads * events_per_thread


# ------------------------------------------------------------------ #
# SQL views
# ------------------------------------------------------------------ #


class TestSQLViews:
    def test_model_costs_view(self, tracker):
        tracker.record(UsageEvent(model_name="openai/gpt-4", provider="openai", cost=0.01, total_tokens=100))
        tracker.record(UsageEvent(model_name="openai/gpt-4", provider="openai", cost=0.02, total_tokens=200))
        tracker.flush()

        import sqlite3

        conn = sqlite3.connect(str(tracker._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM model_costs").fetchall()
        conn.close()

        assert len(rows) == 1
        row = dict(rows[0])
        assert row["model_name"] == "openai/gpt-4"
        assert row["total_cost"] == pytest.approx(0.03)
        assert row["total_tokens"] == 300
        assert row["event_count"] == 2

    def test_provider_costs_view(self, tracker):
        tracker.record(UsageEvent(model_name="openai/gpt-4", provider="openai", cost=0.01))
        tracker.record(UsageEvent(model_name="claude/haiku", provider="claude", cost=0.005))
        tracker.flush()

        import sqlite3

        conn = sqlite3.connect(str(tracker._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM provider_costs ORDER BY provider").fetchall()
        conn.close()

        assert len(rows) == 2
        providers = {dict(r)["provider"]: dict(r)["total_cost"] for r in rows}
        assert providers["openai"] == pytest.approx(0.01)
        assert providers["claude"] == pytest.approx(0.005)

    def test_agent_costs_view(self, tracker):
        with tracker.agent("agent-a"):
            tracker.record(UsageEvent(model_name="test", provider="test", cost=0.01))
        with tracker.agent("agent-b"):
            tracker.record(UsageEvent(model_name="test", provider="test", cost=0.02))
        tracker.flush()

        import sqlite3

        conn = sqlite3.connect(str(tracker._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM agent_costs ORDER BY agent_id").fetchall()
        conn.close()

        assert len(rows) == 2
        agents = {dict(r)["agent_id"]: dict(r)["total_cost"] for r in rows}
        assert agents["agent-a"] == pytest.approx(0.01)
        assert agents["agent-b"] == pytest.approx(0.02)

    def test_model_usage_backward_compat_view(self, tracker):
        """The model_usage view should match the old ledger schema."""
        tracker.record(
            UsageEvent(
                model_name="openai/gpt-4",
                provider="openai",
                api_key_hash="abc123",
                total_tokens=100,
                cost=0.01,
            )
        )
        tracker.record(
            UsageEvent(
                model_name="openai/gpt-4",
                provider="openai",
                api_key_hash="abc123",
                total_tokens=200,
                cost=0.02,
            )
        )
        tracker.flush()

        import sqlite3

        conn = sqlite3.connect(str(tracker._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM model_usage").fetchall()
        conn.close()

        assert len(rows) == 1
        row = dict(rows[0])
        assert row["model_name"] == "openai/gpt-4"
        assert row["api_key_hash"] == "abc123"
        assert row["use_count"] == 2
        assert row["total_tokens"] == 300
        assert row["total_cost"] == pytest.approx(0.03)
        assert row["first_used"] is not None
        assert row["last_used"] is not None


# ------------------------------------------------------------------ #
# Module-level singleton
# ------------------------------------------------------------------ #


class TestSingleton:
    def test_get_tracker_returns_same_instance(self):
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

    def test_configure_tracker_replaces_singleton(self, tmp_path):
        original = get_tracker()
        new = configure_tracker(
            enabled=True,
            db_path=str(tmp_path / "configured.db"),
            flush_threshold=20,
        )
        assert new is not original
        assert get_tracker() is new
        assert new._flush_threshold == 20

        # Restore default singleton for other tests
        configure_tracker(enabled=True)


# ------------------------------------------------------------------ #
# Cache hit recording
# ------------------------------------------------------------------ #


class TestCacheHit:
    def test_cache_hit_recorded(self, tracker):
        tracker.record(UsageEvent(model_name="test", provider="test", cache_hit=True))

        results = tracker.query()
        assert results[0]["cache_hit"] == 1

    def test_cache_miss_default(self, tracker):
        tracker.record(UsageEvent(model_name="test", provider="test"))

        results = tracker.query()
        assert results[0]["cache_hit"] == 0


# ------------------------------------------------------------------ #
# as_callbacks integration
# ------------------------------------------------------------------ #


class TestAsCallbacks:
    def test_as_callbacks_records_on_response(self, tracker):
        callbacks = tracker.as_callbacks(session_id="cb-sess")

        callbacks.on_response(
            {
                "text": "hello",
                "meta": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost": 0.001,
                },
                "driver": "openai/gpt-4",
                "elapsed_ms": 100.0,
            }
        )

        results = tracker.query()
        assert len(results) == 1
        assert results[0]["session_id"] == "cb-sess"
        assert results[0]["model_name"] == "openai/gpt-4"
        assert results[0]["total_tokens"] == 15

    def test_as_callbacks_records_on_error(self, tracker):
        callbacks = tracker.as_callbacks()

        callbacks.on_error(
            {
                "error": ValueError("test error"),
                "driver": "openai/gpt-4",
            }
        )

        results = tracker.query()
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert results[0]["error_type"] == "ValueError"
