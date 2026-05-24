"""Tests for prompture.session_memory."""

from __future__ import annotations

import time

import pytest

from prompture.session_memory import (
    InMemorySessionStore,
    MemoryFact,
    MemoryKind,
    SessionMemory,
    SQLiteSessionStore,
    default_summarizer,
)

# ----------------------------------------------------------------------
# MemoryFact
# ----------------------------------------------------------------------


class TestMemoryFact:
    def test_default_ids_unique(self) -> None:
        a = MemoryFact(user_id="u", content="x")
        b = MemoryFact(user_id="u", content="x")
        assert a.id != b.id

    def test_round_trip_dict(self) -> None:
        f = MemoryFact(user_id="u", content="x", kind=MemoryKind.preference.value, importance=0.9)
        f2 = MemoryFact.from_dict(f.to_dict())
        assert f2.id == f.id
        assert f2.kind == MemoryKind.preference.value
        assert f2.importance == 0.9


# ----------------------------------------------------------------------
# InMemorySessionStore
# ----------------------------------------------------------------------


class TestInMemoryStore:
    def test_add_and_list_isolates_users(self) -> None:
        store = InMemorySessionStore()
        store.add(MemoryFact(user_id="alice", content="a"))
        store.add(MemoryFact(user_id="bob", content="b"))
        assert len(store.list("alice")) == 1
        assert len(store.list("bob")) == 1
        assert store.list("alice")[0].content == "a"

    def test_session_filter(self) -> None:
        store = InMemorySessionStore()
        store.add(MemoryFact(user_id="u", session_id="s1", content="one"))
        store.add(MemoryFact(user_id="u", session_id="s2", content="two"))
        store.add(MemoryFact(user_id="u", content="user-wide"))
        assert [f.content for f in store.list("u", session_id="s1")] == ["one"]
        assert [f.content for f in store.list("u", session_id="__user__")] == ["user-wide"]
        # session_id=None means "any" session
        assert {f.content for f in store.list("u")} == {"one", "two", "user-wide"}

    def test_kind_filter(self) -> None:
        store = InMemorySessionStore()
        store.add(MemoryFact(user_id="u", content="a", kind="fact"))
        store.add(MemoryFact(user_id="u", content="b", kind="preference"))
        out = store.list("u", kinds=("preference",))
        assert [f.content for f in out] == ["b"]

    def test_upsert_by_id(self) -> None:
        store = InMemorySessionStore()
        f = MemoryFact(id="fixed", user_id="u", content="old")
        store.add(f)
        store.add(MemoryFact(id="fixed", user_id="u", content="new"))
        results = store.list("u")
        assert len(results) == 1
        assert results[0].content == "new"

    def test_search_scores_overlap_and_importance(self) -> None:
        store = InMemorySessionStore()
        store.add(MemoryFact(user_id="u", content="paris france travel", importance=0.1))
        store.add(MemoryFact(user_id="u", content="paris food guide", importance=0.9))
        store.add(MemoryFact(user_id="u", content="berlin trip", importance=0.5))
        out = store.search("u", "paris food")
        assert out[0].content == "paris food guide"
        assert {f.content for f in out} == {"paris food guide", "paris france travel"}

    def test_search_empty_query_returns_recent(self) -> None:
        store = InMemorySessionStore()
        store.add(MemoryFact(user_id="u", content="a", ts=time.time() - 100))
        store.add(MemoryFact(user_id="u", content="b", ts=time.time()))
        out = store.search("u", "", k=5)
        assert [f.content for f in out] == ["a", "b"]

    def test_delete_by_session(self) -> None:
        store = InMemorySessionStore()
        store.add(MemoryFact(user_id="u", session_id="s1", content="a"))
        store.add(MemoryFact(user_id="u", session_id="s1", content="b"))
        store.add(MemoryFact(user_id="u", session_id="s2", content="c"))
        removed = store.delete("u", session_id="s1")
        assert removed == 2
        assert [f.content for f in store.list("u")] == ["c"]

    def test_delete_by_fact_id(self) -> None:
        store = InMemorySessionStore()
        f = MemoryFact(user_id="u", content="a")
        store.add(f)
        store.add(MemoryFact(user_id="u", content="b"))
        assert store.delete("u", fact_id=f.id) == 1
        assert [f.content for f in store.list("u")] == ["b"]

    def test_delete_requires_filter(self) -> None:
        store = InMemorySessionStore()
        with pytest.raises(ValueError, match="Refusing"):
            store.delete("u")


# ----------------------------------------------------------------------
# SQLiteSessionStore
# ----------------------------------------------------------------------


class TestSQLiteStore:
    def test_add_and_list(self, tmp_path) -> None:
        store = SQLiteSessionStore(db_path=tmp_path / "mem.db")
        store.add(MemoryFact(user_id="alice", content="hello"))
        out = store.list("alice")
        assert len(out) == 1
        assert out[0].content == "hello"

    def test_persistence_across_instances(self, tmp_path) -> None:
        path = tmp_path / "mem.db"
        s1 = SQLiteSessionStore(db_path=path)
        s1.add(MemoryFact(user_id="alice", content="persist me"))
        s2 = SQLiteSessionStore(db_path=path)
        out = s2.list("alice")
        assert len(out) == 1
        assert out[0].content == "persist me"

    def test_search(self, tmp_path) -> None:
        store = SQLiteSessionStore(db_path=tmp_path / "mem.db")
        store.add(MemoryFact(user_id="u", content="weather is sunny"))
        store.add(MemoryFact(user_id="u", content="tax is increasing"))
        out = store.search("u", "weather")
        assert len(out) == 1
        assert "weather" in out[0].content

    def test_delete_by_session(self, tmp_path) -> None:
        store = SQLiteSessionStore(db_path=tmp_path / "mem.db")
        store.add(MemoryFact(user_id="u", session_id="s1", content="a"))
        store.add(MemoryFact(user_id="u", session_id="s2", content="b"))
        assert store.delete("u", session_id="s1") == 1
        remaining = [f.content for f in store.list("u")]
        assert remaining == ["b"]

    def test_metadata_round_trip(self, tmp_path) -> None:
        store = SQLiteSessionStore(db_path=tmp_path / "mem.db")
        store.add(MemoryFact(user_id="u", content="x", metadata={"foo": 1, "bar": [1, 2]}))
        out = store.list("u")
        assert out[0].metadata == {"foo": 1, "bar": [1, 2]}


# ----------------------------------------------------------------------
# SessionMemory high-level API
# ----------------------------------------------------------------------


class TestSessionMemory:
    def test_remember_writes_user_wide_by_default(self) -> None:
        mem = SessionMemory(user_id="alice", session_id="s1")
        f = mem.remember("loves espresso", kind=MemoryKind.preference.value)
        assert f.user_id == "alice"
        assert f.session_id is None  # user-wide

    def test_remember_session_scoped(self) -> None:
        mem = SessionMemory(user_id="alice", session_id="s1")
        f = mem.remember("booked flight to Paris", session_scoped=True)
        assert f.session_id == "s1"

    def test_recall_merges_session_and_user_wide(self) -> None:
        store = InMemorySessionStore()
        mem = SessionMemory(user_id="u", session_id="s1", store=store)
        mem.remember("alice loves dark mode")  # user-wide
        mem.remember("currently booking paris flight", session_scoped=True)
        hits = mem.recall("paris")
        assert any("paris" in f.content for f in hits)
        hits2 = mem.recall("dark")
        assert any("dark mode" in f.content for f in hits2)

    def test_recall_excluding_user_wide(self) -> None:
        store = InMemorySessionStore()
        mem = SessionMemory(user_id="u", session_id="s1", store=store)
        mem.remember("user-wide fact about coffee")
        mem.remember("session note about coffee", session_scoped=True)
        only_session = mem.recall("coffee", include_user_wide=False)
        # All hits should belong to this session
        assert all(f.session_id == "s1" for f in only_session)

    def test_render_for_prompt_relevance(self) -> None:
        mem = SessionMemory(user_id="u", session_id="s1")
        mem.remember("Pref: concise answers", kind=MemoryKind.preference.value)
        mem.remember("Pref: avoid emojis", kind=MemoryKind.preference.value)
        text = mem.render_for_prompt("emojis")
        assert "Long-term memory" in text
        assert "avoid emojis" in text

    def test_render_for_prompt_empty(self) -> None:
        mem = SessionMemory(user_id="u")
        assert mem.render_for_prompt("anything") == ""

    def test_render_for_prompt_recent_when_no_query(self) -> None:
        mem = SessionMemory(user_id="u", recall_k=2)
        mem.remember("one")
        mem.remember("two")
        mem.remember("three")
        text = mem.render_for_prompt()
        assert "two" in text
        assert "three" in text
        assert "one" not in text

    def test_forget(self) -> None:
        mem = SessionMemory(user_id="u")
        f = mem.remember("to delete")
        assert mem.forget(f.id) is True
        assert mem.forget("nope") is False

    def test_clear_session(self) -> None:
        mem = SessionMemory(user_id="u", session_id="s1")
        mem.remember("user-wide")
        mem.remember("session", session_scoped=True)
        removed = mem.clear_session()
        assert removed == 1
        remaining = mem.all()
        assert [f.content for f in remaining] == ["user-wide"]

    def test_clear_session_without_session_id_clears_user_wide(self) -> None:
        mem = SessionMemory(user_id="u")
        mem.remember("user-wide")
        removed = mem.clear_session()
        assert removed == 1
        assert mem.all() == []

    def test_summarize_uses_default_summarizer(self) -> None:
        mem = SessionMemory(user_id="u", session_id="s1")
        msgs = [
            {"role": "user", "content": "first message"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "last message"},
        ]
        f = mem.summarize(msgs)
        assert f is not None
        assert f.kind == MemoryKind.summary.value
        assert "first message" in f.content or "last message" in f.content
        assert f.metadata["message_count"] == 3

    def test_summarize_empty_returns_none(self) -> None:
        mem = SessionMemory(user_id="u")
        assert mem.summarize([]) is None

    def test_summarize_with_custom_summarizer(self) -> None:
        mem = SessionMemory(user_id="u")
        f = mem.summarize(
            [{"role": "user", "content": "hi"}],
            summarizer=lambda msgs: f"CUSTOM:{len(msgs)}",
        )
        assert f is not None
        assert f.content == "CUSTOM:1"

    def test_default_summarizer_handles_no_user_messages(self) -> None:
        assert default_summarizer([{"role": "assistant", "content": "hi"}]) == ""

    def test_empty_user_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="user_id"):
            SessionMemory(user_id="")
