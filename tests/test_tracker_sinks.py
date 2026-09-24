"""Tests for UsageTracker sinks — host-owned destinations for usage events."""

from __future__ import annotations

import pytest

from prompture.infra.tracker import UsageEvent, UsageTracker, configure_tracker, get_tracker


def _event(**kw) -> UsageEvent:
    defaults = {"model_name": "test/model", "provider": "test", "cost": 0.01, "total_tokens": 10}
    defaults.update(kw)
    return UsageEvent(**defaults)


def test_sink_receives_recorded_events(tmp_path):
    seen: list[UsageEvent] = []
    tracker = UsageTracker(db_path=tmp_path / "u.db", sinks=[seen.append])

    tracker.record(_event())

    assert len(seen) == 1
    assert seen[0].model_name == "test/model"
    assert seen[0].cost == 0.01


def test_sink_gets_context_injected_event(tmp_path):
    seen: list[UsageEvent] = []
    tracker = UsageTracker(db_path=tmp_path / "u.db", sinks=[seen.append])

    with tracker.session("s-1"), tracker.agent("agent-x"), tracker.operation("op-y"):
        tracker.record(_event())

    assert seen[0].session_id == "s-1"
    assert seen[0].agent_id == "agent-x"
    assert seen[0].operation == "op-y"


def test_failing_sink_does_not_break_recording_or_other_sinks(tmp_path):
    seen: list[UsageEvent] = []

    def bad_sink(event: UsageEvent) -> None:
        raise RuntimeError("boom")

    tracker = UsageTracker(db_path=tmp_path / "u.db", flush_threshold=1, sinks=[bad_sink, seen.append])

    tracker.record(_event())  # must not raise

    assert len(seen) == 1
    # The SQLite write still happened despite the raising sink.
    summary = tracker.summary()
    assert summary.total_events == 1


def test_persist_false_skips_sqlite_but_fans_out(tmp_path):
    seen: list[UsageEvent] = []
    db = tmp_path / "u.db"
    tracker = UsageTracker(db_path=db, flush_threshold=1, persist=False, sinks=[seen.append])

    tracker.record(_event())
    tracker.flush()

    assert len(seen) == 1
    assert not db.exists()


def test_add_and_remove_sink(tmp_path):
    seen: list[UsageEvent] = []
    tracker = UsageTracker(db_path=tmp_path / "u.db")

    tracker.add_sink(seen.append)
    tracker.add_sink(seen.append)  # duplicate registration is a no-op
    tracker.record(_event())
    assert len(seen) == 1

    tracker.remove_sink(seen.append)
    tracker.remove_sink(seen.append)  # unknown sink is ignored
    tracker.record(_event())
    assert len(seen) == 1


def test_disabled_tracker_does_not_fan_out(tmp_path):
    seen: list[UsageEvent] = []
    tracker = UsageTracker(db_path=tmp_path / "u.db", enabled=False, sinks=[seen.append])

    tracker.record(_event())

    assert seen == []


def test_configure_tracker_wires_sinks_globally(tmp_path):
    seen: list[UsageEvent] = []
    configure_tracker(db_path=str(tmp_path / "u.db"), sinks=[seen.append], persist=False)
    try:
        get_tracker().record(_event())
        assert len(seen) == 1
    finally:
        configure_tracker(enabled=False)


@pytest.mark.parametrize("exported", ["UsageSink", "UsageEvent", "UsageTracker"])
def test_public_exports(exported):
    import prompture

    assert hasattr(prompture, exported)


def test_event_exposes_cost_provenance_and_rate_limits():
    event = _event(
        metadata={
            "cost_status": "estimated",
            "pricing": {"source": "local_kb", "currency": "USD"},
            "rate_limits": {
                "source": "headers",
                "observed_at": 1_800_000_000.0,
                "windows": {"tokens": {"limit": 100, "remaining": 25, "resets_at": 1_800_000_060.0}},
            },
        }
    )
    assert event.cost_status == "estimated"
    assert event.cost_source == "local_kb"
    limits = event.rate_limits
    assert limits is not None
    assert limits.headroom == 0.25
    assert limits.tightest_window == "tokens"


def test_event_without_provenance_returns_none():
    event = _event()
    assert (event.cost_status, event.cost_source, event.rate_limits) == (None, None, None)


def test_driver_usage_reaches_sinks_with_provenance(tmp_path, monkeypatch):
    from prompture.drivers.base import Driver
    from prompture.infra import tracker as tracker_module

    seen: list[UsageEvent] = []
    monkeypatch.setattr(tracker_module, "_tracker", UsageTracker(db_path=tmp_path / "u.db", sinks=[seen.append]))

    class StubDriver(Driver):
        model = "openai/gpt-4o-mini"

        def generate(self, prompt, options):
            return {}

    StubDriver()._auto_record_usage(
        {
            "meta": {
                "model_name": "openai/gpt-4o-mini",
                "cost": 0.001,
                "cost_status": "estimated",
                "pricing": {"source": "models.dev"},
                "rate_limits": {"windows": {"requests": {"limit": 60, "remaining": 1}}, "observed_at": 1.0},
                "raw_response": {"big": "payload"},
            }
        },
        12.0,
    )
    assert len(seen) == 1
    assert seen[0].cost_source == "models.dev"
    assert seen[0].rate_limits is not None
    assert seen[0].rate_limits.windows["requests"].remaining == 1
    assert "raw_response" not in seen[0].metadata
