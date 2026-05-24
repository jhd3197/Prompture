"""Tests for prompture.checkpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from prompture.checkpoints import (
    Checkpoint,
    CheckpointManager,
    FileCheckpointStore,
    InMemoryCheckpointStore,
    RunStatus,
    SQLiteCheckpointStore,
    restore_conversation,
    snapshot_conversation,
)

# ----------------------------------------------------------------------
# Checkpoint dataclass round-trip
# ----------------------------------------------------------------------


class TestCheckpoint:
    def test_round_trip(self) -> None:
        cp = Checkpoint(
            run_id="r1",
            messages=[{"role": "user", "content": "hi"}],
            iteration=3,
            status=RunStatus.paused.value,
            prompt="hi",
            metadata={"todo": ["a", "b"]},
        )
        d = cp.to_dict()
        cp2 = Checkpoint.from_dict(d)
        assert cp2.run_id == "r1"
        assert cp2.iteration == 3
        assert cp2.status == RunStatus.paused.value
        assert cp2.metadata == {"todo": ["a", "b"]}

    def test_from_dict_ignores_unknown_keys(self) -> None:
        cp = Checkpoint.from_dict({"run_id": "r1", "extra_garbage": 99})
        assert cp.run_id == "r1"

    def test_default_status_is_running(self) -> None:
        cp = Checkpoint(run_id="r1")
        assert cp.status == "running"


# ----------------------------------------------------------------------
# InMemoryCheckpointStore
# ----------------------------------------------------------------------


class TestInMemoryStore:
    def test_save_load(self) -> None:
        store = InMemoryCheckpointStore()
        cp = Checkpoint(run_id="r1", iteration=2)
        store.save(cp)
        loaded = store.load("r1")
        assert loaded is not None
        assert loaded.iteration == 2

    def test_load_missing_returns_none(self) -> None:
        assert InMemoryCheckpointStore().load("nope") is None

    def test_save_upserts_by_run_id(self) -> None:
        store = InMemoryCheckpointStore()
        store.save(Checkpoint(run_id="r1", iteration=1))
        store.save(Checkpoint(run_id="r1", iteration=5))
        assert store.load("r1").iteration == 5
        assert len(store.list_runs()) == 1

    def test_list_runs_filter_by_status(self) -> None:
        store = InMemoryCheckpointStore()
        store.save(Checkpoint(run_id="a", status=RunStatus.completed.value))
        store.save(Checkpoint(run_id="b", status=RunStatus.running.value))
        runs = store.list_runs(status=RunStatus.completed.value)
        assert [c.run_id for c in runs] == ["a"]

    def test_delete(self) -> None:
        store = InMemoryCheckpointStore()
        store.save(Checkpoint(run_id="r1"))
        assert store.delete("r1") is True
        assert store.delete("r1") is False


# ----------------------------------------------------------------------
# FileCheckpointStore
# ----------------------------------------------------------------------


class TestFileStore:
    def test_save_load(self, tmp_path) -> None:
        store = FileCheckpointStore(tmp_path)
        cp = Checkpoint(run_id="r1", iteration=2, messages=[{"role": "user", "content": "hi"}])
        store.save(cp)
        loaded = store.load("r1")
        assert loaded is not None
        assert loaded.iteration == 2
        assert loaded.messages[0]["content"] == "hi"

    def test_load_missing(self, tmp_path) -> None:
        assert FileCheckpointStore(tmp_path).load("nope") is None

    def test_path_traversal_neutralised(self, tmp_path) -> None:
        store = FileCheckpointStore(tmp_path)
        # Hostile run id with separators should not escape the directory.
        store.save(Checkpoint(run_id="../../etc/passwd"))
        # File should still exist within tmp_path
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        assert files[0].parent == tmp_path

    def test_list_runs(self, tmp_path) -> None:
        store = FileCheckpointStore(tmp_path)
        store.save(Checkpoint(run_id="a"))
        store.save(Checkpoint(run_id="b"))
        runs = store.list_runs()
        assert {c.run_id for c in runs} == {"a", "b"}

    def test_atomic_write_doesnt_leave_temp_files(self, tmp_path) -> None:
        store = FileCheckpointStore(tmp_path)
        store.save(Checkpoint(run_id="r1"))
        # No leftover temp file in the directory.
        leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".chkpt-")]
        assert not leftovers

    def test_delete(self, tmp_path) -> None:
        store = FileCheckpointStore(tmp_path)
        store.save(Checkpoint(run_id="r1"))
        assert store.delete("r1") is True
        assert store.delete("r1") is False

    def test_corrupt_file_returns_none(self, tmp_path) -> None:
        store = FileCheckpointStore(tmp_path)
        bad = tmp_path / "broken.json"
        bad.write_text("{not json")
        assert store.load("broken") is None


# ----------------------------------------------------------------------
# SQLiteCheckpointStore
# ----------------------------------------------------------------------


class TestSQLiteStore:
    def test_save_load(self, tmp_path) -> None:
        store = SQLiteCheckpointStore(db_path=tmp_path / "cp.db")
        store.save(Checkpoint(run_id="r1", iteration=4))
        loaded = store.load("r1")
        assert loaded is not None
        assert loaded.iteration == 4

    def test_persistence_across_instances(self, tmp_path) -> None:
        path = tmp_path / "cp.db"
        SQLiteCheckpointStore(db_path=path).save(Checkpoint(run_id="r1", iteration=9))
        loaded = SQLiteCheckpointStore(db_path=path).load("r1")
        assert loaded is not None
        assert loaded.iteration == 9

    def test_list_runs_filter(self, tmp_path) -> None:
        store = SQLiteCheckpointStore(db_path=tmp_path / "cp.db")
        store.save(Checkpoint(run_id="a", status=RunStatus.completed.value))
        store.save(Checkpoint(run_id="b", status=RunStatus.running.value))
        runs = store.list_runs(status=RunStatus.running.value)
        assert [c.run_id for c in runs] == ["b"]

    def test_delete(self, tmp_path) -> None:
        store = SQLiteCheckpointStore(db_path=tmp_path / "cp.db")
        store.save(Checkpoint(run_id="r1"))
        assert store.delete("r1") is True
        assert store.delete("r1") is False


# ----------------------------------------------------------------------
# snapshot_conversation / restore_conversation
# ----------------------------------------------------------------------


def _fake_conv(messages: list[dict[str, Any]] | None = None) -> MagicMock:
    conv = MagicMock()
    conv.messages = list(messages) if messages else [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    conv.usage = {"total_tokens": 42}
    conv.system_prompt = "you are helpful"
    conv.model_name = "openai/gpt-4o"
    return conv


class TestSnapshot:
    def test_snapshot_captures_conversation_fields(self) -> None:
        conv = _fake_conv()
        cp = snapshot_conversation(conv, iteration=2, prompt="hi", agent_name="researcher")
        assert cp.messages == conv.messages
        assert cp.usage == conv.usage
        assert cp.system_prompt == "you are helpful"
        assert cp.iteration == 2
        assert cp.agent_name == "researcher"
        assert cp.model == "openai/gpt-4o"

    def test_metadata_is_copied(self) -> None:
        conv = _fake_conv()
        md = {"todo": ["a"]}
        cp = snapshot_conversation(conv, metadata=md)
        assert cp.metadata == md
        md["todo"].append("b")
        # Snapshot's metadata is decoupled
        assert cp.metadata == {"todo": ["a"]}


# ----------------------------------------------------------------------
# CheckpointManager
# ----------------------------------------------------------------------


class TestCheckpointManager:
    def test_save_conversation_writes_to_store(self) -> None:
        store = InMemoryCheckpointStore()
        mgr = CheckpointManager(store, run_id="run-1", agent_name="a")
        conv = _fake_conv()
        cp = mgr.save_conversation(conv, iteration=3, prompt="go")
        assert cp.run_id == "run-1"
        assert cp.agent_name == "a"
        assert store.load("run-1").iteration == 3

    def test_save_agent_returns_none_without_conversation(self) -> None:
        agent = MagicMock()
        agent.conversation = None
        agent._conversation = None
        mgr = CheckpointManager(InMemoryCheckpointStore(), run_id="r")
        assert mgr.save_agent(agent) is None

    def test_save_agent_uses_persistent_conversation(self) -> None:
        agent = MagicMock()
        agent.conversation = _fake_conv()
        mgr = CheckpointManager(InMemoryCheckpointStore(), run_id="r")
        cp = mgr.save_agent(agent, iteration=1)
        assert cp is not None
        assert cp.iteration == 1
        assert cp.messages == agent.conversation.messages

    def test_load_returns_latest(self) -> None:
        store = InMemoryCheckpointStore()
        mgr = CheckpointManager(store, run_id="r")
        mgr.save_conversation(_fake_conv(), iteration=1)
        mgr.save_conversation(_fake_conv(), iteration=2)
        assert mgr.load().iteration == 2

    def test_load_missing_returns_none(self) -> None:
        mgr = CheckpointManager(InMemoryCheckpointStore(), run_id="ghost")
        assert mgr.load() is None
        assert mgr.exists() is False

    def test_mark_updates_status(self) -> None:
        mgr = CheckpointManager(InMemoryCheckpointStore(), run_id="r")
        mgr.save_conversation(_fake_conv())
        mgr.mark(RunStatus.completed.value, metadata={"final": True})
        cp = mgr.load()
        assert cp.status == RunStatus.completed.value
        assert cp.metadata["final"] is True

    def test_mark_without_existing_returns_none(self) -> None:
        mgr = CheckpointManager(InMemoryCheckpointStore(), run_id="ghost")
        assert mgr.mark(RunStatus.failed.value) is None

    def test_resume_raises_when_no_checkpoint(self) -> None:
        mgr = CheckpointManager(InMemoryCheckpointStore(), run_id="ghost")
        with pytest.raises(LookupError):
            mgr.resume_conversation(model="openai/gpt-4o-mini")

    def test_delete(self) -> None:
        store = InMemoryCheckpointStore()
        mgr = CheckpointManager(store, run_id="r")
        mgr.save_conversation(_fake_conv())
        assert mgr.delete() is True
        assert mgr.delete() is False


# ----------------------------------------------------------------------
# Round-trip with a real Conversation (using a stub driver)
# ----------------------------------------------------------------------


class TestRoundTripWithRealConversation:
    """Smoke test: snapshot a real Conversation, then restore it."""

    def test_round_trip(self) -> None:
        from prompture.agents.conversation import Conversation
        from prompture.drivers.base import Driver

        class StubDriver(Driver):
            supports_messages = True
            supports_tool_use = False

            def __init__(self) -> None:
                self.model = "stub/model"

            def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
                return self._make()

            def generate_messages(
                self, messages: list[dict[str, Any]], options: dict[str, Any]
            ) -> dict[str, Any]:
                return self._make()

            def _make(self) -> dict[str, Any]:
                return {
                    "text": "ok",
                    "meta": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                        "cost": 0.0,
                        "raw_response": {},
                    },
                }

        conv = Conversation(driver=StubDriver(), system_prompt="be brief")
        conv.ask("hi")
        cp = snapshot_conversation(conv, iteration=1)
        assert cp.messages
        assert cp.system_prompt == "be brief"

        restored = restore_conversation(cp, driver=StubDriver())
        # History was carried over
        assert restored.messages == cp.messages
        # New question continues from the existing history
        restored.ask("again")
        assert len(restored.messages) >= len(cp.messages) + 1
