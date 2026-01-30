"""Tests for the persistence module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from prompture.persistence import ConversationStore, load_from_file, save_to_file

# ------------------------------------------------------------------
# File-based persistence
# ------------------------------------------------------------------


class TestFilePersistence:
    """Test save_to_file / load_from_file."""

    def test_round_trip(self, tmp_path: Path):
        data = {"version": 1, "conversation_id": "abc", "messages": [{"role": "user", "content": "hi"}]}
        path = tmp_path / "conv.json"
        save_to_file(data, path)
        loaded = load_from_file(path)
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "sub" / "dir" / "conv.json"
        save_to_file({"version": 1}, path)
        assert path.exists()

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_from_file(tmp_path / "nonexistent.json")

    def test_unicode_content(self, tmp_path: Path):
        data = {"text": "héllo wörld 日本語"}
        path = tmp_path / "unicode.json"
        save_to_file(data, path)
        loaded = load_from_file(path)
        assert loaded["text"] == "héllo wörld 日本語"


# ------------------------------------------------------------------
# ConversationStore
# ------------------------------------------------------------------


def _make_conv_data(
    conv_id: str = "c1",
    model: str = "openai/gpt-4",
    tags: list[str] | None = None,
    turn_count: int = 1,
    created_at: str = "2026-01-01T00:00:00+00:00",
    last_active: str = "2026-01-01T00:00:00+00:00",
) -> dict[str, Any]:
    return {
        "version": 1,
        "conversation_id": conv_id,
        "model_name": model,
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {
            "created_at": created_at,
            "last_active": last_active,
            "turn_count": turn_count,
            "tags": tags or [],
        },
    }


class TestConversationStore:
    """Test SQLite-backed ConversationStore."""

    @pytest.fixture()
    def store(self, tmp_path: Path) -> ConversationStore:
        return ConversationStore(db_path=tmp_path / "test.db")

    def test_save_and_load(self, store: ConversationStore):
        data = _make_conv_data("c1")
        store.save("c1", data)
        loaded = store.load("c1")
        assert loaded is not None
        assert loaded["conversation_id"] == "c1"

    def test_load_missing_returns_none(self, store: ConversationStore):
        assert store.load("nonexistent") is None

    def test_delete(self, store: ConversationStore):
        store.save("c1", _make_conv_data("c1"))
        assert store.delete("c1") is True
        assert store.load("c1") is None

    def test_delete_nonexistent_returns_false(self, store: ConversationStore):
        assert store.delete("nonexistent") is False

    def test_find_by_tag(self, store: ConversationStore):
        store.save("c1", _make_conv_data("c1", tags=["demo", "support"]))
        store.save("c2", _make_conv_data("c2", tags=["demo"]))
        store.save("c3", _make_conv_data("c3", tags=["other"]))

        results = store.find_by_tag("demo")
        ids = [r["id"] for r in results]
        assert "c1" in ids
        assert "c2" in ids
        assert "c3" not in ids

    def test_find_by_tag_empty(self, store: ConversationStore):
        assert store.find_by_tag("nonexistent") == []

    def test_find_by_id(self, store: ConversationStore):
        store.save("c1", _make_conv_data("c1", tags=["demo"], model="openai/gpt-4"))
        summary = store.find_by_id("c1")
        assert summary is not None
        assert summary["id"] == "c1"
        assert summary["model_name"] == "openai/gpt-4"
        assert "demo" in summary["tags"]

    def test_find_by_id_missing(self, store: ConversationStore):
        assert store.find_by_id("nonexistent") is None

    def test_list_all(self, store: ConversationStore):
        store.save("c1", _make_conv_data("c1", last_active="2026-01-01T00:00:00+00:00"))
        store.save("c2", _make_conv_data("c2", last_active="2026-01-02T00:00:00+00:00"))
        store.save("c3", _make_conv_data("c3", last_active="2026-01-03T00:00:00+00:00"))

        results = store.list_all()
        assert len(results) == 3
        # Ordered by last_active DESC
        assert results[0]["id"] == "c3"
        assert results[1]["id"] == "c2"
        assert results[2]["id"] == "c1"

    def test_list_all_with_limit(self, store: ConversationStore):
        for i in range(5):
            store.save(f"c{i}", _make_conv_data(f"c{i}", last_active=f"2026-01-0{i + 1}T00:00:00+00:00"))
        results = store.list_all(limit=2)
        assert len(results) == 2

    def test_list_all_with_offset(self, store: ConversationStore):
        for i in range(5):
            store.save(f"c{i}", _make_conv_data(f"c{i}", last_active=f"2026-01-0{i + 1}T00:00:00+00:00"))
        results = store.list_all(limit=2, offset=2)
        assert len(results) == 2

    def test_tag_update_on_resave(self, store: ConversationStore):
        store.save("c1", _make_conv_data("c1", tags=["old-tag"]))
        store.save("c1", _make_conv_data("c1", tags=["new-tag"]))
        summary = store.find_by_id("c1")
        assert summary is not None
        assert "new-tag" in summary["tags"]
        assert "old-tag" not in summary["tags"]

    def test_summary_includes_turn_count(self, store: ConversationStore):
        store.save("c1", _make_conv_data("c1", turn_count=5))
        summary = store.find_by_id("c1")
        assert summary is not None
        assert summary["turn_count"] == 5
