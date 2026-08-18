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
