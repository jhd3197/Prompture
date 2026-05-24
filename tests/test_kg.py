"""Tests for prompture.kg."""

from __future__ import annotations

from prompture.kg import (
    Entity,
    InMemoryEntityStore,
    KnowledgeGraph,
    Mention,
    Relation,
    SQLiteEntityStore,
)

# ----------------------------------------------------------------------
# Type round trips
# ----------------------------------------------------------------------


class TestTypes:
    def test_entity_round_trip(self) -> None:
        e = Entity(
            id="person:alice",
            name="Alice",
            type="person",
            aliases=("Al", "A. Smith"),
            attributes={"age": 30},
        )
        d = e.to_dict()
        assert Entity.from_dict(d).aliases == ("Al", "A. Smith")

    def test_entity_make_id(self) -> None:
        assert Entity.make_id("Sam Altman", "person") == "person:sam_altman"
        assert Entity.make_id("OpenAI") == "openai"

    def test_relation_default_id(self) -> None:
        r1 = Relation(subject_id="a", predicate="p", object_id="b")
        r2 = Relation(subject_id="a", predicate="p", object_id="b")
        assert r1.id != r2.id

    def test_mention_round_trip(self) -> None:
        m = Mention(entity_id="e", text="Alice", source="doc.txt", start=10, end=15)
        m2 = Mention.from_dict(m.to_dict())
        assert m2.entity_id == "e"
        assert m2.start == 10


# ----------------------------------------------------------------------
# InMemoryEntityStore
# ----------------------------------------------------------------------


class TestInMemoryStore:
    def test_upsert_get(self) -> None:
        store = InMemoryEntityStore()
        e = Entity(id="e1", name="Alice", type="person")
        store.upsert_entity(e)
        assert store.get_entity("e1").name == "Alice"

    def test_find_by_name(self) -> None:
        store = InMemoryEntityStore()
        store.upsert_entity(Entity(id="e1", name="Alice"))
        store.upsert_entity(Entity(id="e2", name="Bob"))
        out = store.find_entities(name="alice")
        assert [e.id for e in out] == ["e1"]

    def test_find_by_alias(self) -> None:
        store = InMemoryEntityStore()
        store.upsert_entity(Entity(id="e1", name="Alice", aliases=("Al",)))
        out = store.find_entities(name="Al")
        assert out and out[0].id == "e1"

    def test_find_by_type(self) -> None:
        store = InMemoryEntityStore()
        store.upsert_entity(Entity(id="e1", name="Alice", type="person"))
        store.upsert_entity(Entity(id="e2", name="OpenAI", type="organization"))
        people = store.find_entities(type="person")
        assert [e.id for e in people] == ["e1"]

    def test_upsert_reindexes_aliases(self) -> None:
        store = InMemoryEntityStore()
        store.upsert_entity(Entity(id="e1", name="Alice", aliases=("Al",)))
        # Drop the alias on a re-upsert
        store.upsert_entity(Entity(id="e1", name="Alice", aliases=()))
        assert store.find_entities(name="Al") == []

    def test_delete_cascades_relations(self) -> None:
        store = InMemoryEntityStore()
        store.upsert_entity(Entity(id="a", name="A"))
        store.upsert_entity(Entity(id="b", name="B"))
        rel = Relation(subject_id="a", predicate="knows", object_id="b")
        store.add_relation(rel)
        store.delete_entity("a")
        assert store.relations_for("b") == []

    def test_relations_direction(self) -> None:
        store = InMemoryEntityStore()
        store.upsert_entity(Entity(id="a", name="A"))
        store.upsert_entity(Entity(id="b", name="B"))
        r = Relation(subject_id="a", predicate="knows", object_id="b")
        store.add_relation(r)
        assert len(store.relations_for("a", as_object=False)) == 1
        assert store.relations_for("a", as_subject=False) == []
        assert len(store.relations_for("b", as_subject=False)) == 1

    def test_relations_predicate_filter(self) -> None:
        store = InMemoryEntityStore()
        store.upsert_entity(Entity(id="a", name="A"))
        store.upsert_entity(Entity(id="b", name="B"))
        store.add_relation(Relation(subject_id="a", predicate="knows", object_id="b"))
        store.add_relation(Relation(subject_id="a", predicate="works_with", object_id="b"))
        knows = store.relations_for("a", predicate="knows")
        assert len(knows) == 1
        assert knows[0].predicate == "knows"

    def test_mentions(self) -> None:
        store = InMemoryEntityStore()
        store.upsert_entity(Entity(id="e1", name="Alice"))
        store.add_mention(Mention(entity_id="e1", text="Alice"))
        store.add_mention(Mention(entity_id="e1", text="Al"))
        out = store.mentions_for("e1")
        assert {m.text for m in out} == {"Alice", "Al"}


# ----------------------------------------------------------------------
# SQLiteEntityStore
# ----------------------------------------------------------------------


class TestSQLiteStore:
    def test_upsert_get(self, tmp_path) -> None:
        store = SQLiteEntityStore(db_path=tmp_path / "kg.db")
        store.upsert_entity(Entity(id="e1", name="Alice", type="person"))
        assert store.get_entity("e1").type == "person"

    def test_find_by_name_and_alias(self, tmp_path) -> None:
        store = SQLiteEntityStore(db_path=tmp_path / "kg.db")
        store.upsert_entity(Entity(id="e1", name="Alice", aliases=("Al",)))
        store.upsert_entity(Entity(id="e2", name="Bob"))
        assert [e.id for e in store.find_entities(name="alice")] == ["e1"]
        # Alias search via LIKE
        alias_hits = store.find_entities(name="Al")
        assert any(e.id == "e1" for e in alias_hits)

    def test_persistence_across_instances(self, tmp_path) -> None:
        path = tmp_path / "kg.db"
        SQLiteEntityStore(db_path=path).upsert_entity(Entity(id="e1", name="Alice"))
        loaded = SQLiteEntityStore(db_path=path).get_entity("e1")
        assert loaded is not None
        assert loaded.name == "Alice"

    def test_cascade_delete(self, tmp_path) -> None:
        store = SQLiteEntityStore(db_path=tmp_path / "kg.db")
        store.upsert_entity(Entity(id="a", name="A"))
        store.upsert_entity(Entity(id="b", name="B"))
        store.add_relation(Relation(subject_id="a", predicate="p", object_id="b"))
        store.delete_entity("a")
        assert store.relations_for("b") == []

    def test_mentions(self, tmp_path) -> None:
        store = SQLiteEntityStore(db_path=tmp_path / "kg.db")
        store.upsert_entity(Entity(id="e1", name="Alice"))
        store.add_mention(Mention(entity_id="e1", text="Alice", source="doc1.txt"))
        out = store.mentions_for("e1")
        assert len(out) == 1
        assert out[0].source == "doc1.txt"


# ----------------------------------------------------------------------
# KnowledgeGraph
# ----------------------------------------------------------------------


class TestKnowledgeGraph:
    def test_upsert_entity_by_name(self) -> None:
        kg = KnowledgeGraph()
        e = kg.upsert_entity("Sam Altman", type="person")
        assert e.id == "person:sam_altman"
        assert e.type == "person"

    def test_upsert_merges_aliases_and_attrs(self) -> None:
        kg = KnowledgeGraph()
        kg.upsert_entity("OpenAI", type="organization", aliases=("OAI",), attributes={"hq": "SF"})
        merged = kg.upsert_entity(
            "OpenAI",
            type="organization",
            aliases=("Open AI",),
            attributes={"hq": "San Francisco", "founded": 2015},
        )
        # Aliases merge, hq is overwritten, founded is added
        assert set(merged.aliases) == {"OAI", "Open AI"}
        assert merged.attributes["hq"] == "San Francisco"
        assert merged.attributes["founded"] == 2015

    def test_add_relation_resolves_strings(self) -> None:
        kg = KnowledgeGraph()
        sam = kg.upsert_entity("Sam Altman", type="person")
        rel = kg.add_relation(sam, "works_at", "OpenAI")
        oai = kg.resolve("OpenAI")
        assert oai is not None
        assert rel.subject_id == sam.id
        assert rel.object_id == oai.id

    def test_neighbors_directional(self) -> None:
        kg = KnowledgeGraph()
        sam = kg.upsert_entity("Sam", type="person")
        oai = kg.upsert_entity("OpenAI", type="organization")
        kg.add_relation(sam, "works_at", oai)
        out_rels = kg.neighbors(sam, direction="out")
        in_rels = kg.neighbors(oai, direction="in")
        assert len(out_rels) == 1
        assert len(in_rels) == 1
        assert kg.neighbors(sam, direction="in") == []

    def test_neighbors_predicate_filter(self) -> None:
        kg = KnowledgeGraph()
        a = kg.upsert_entity("A")
        b = kg.upsert_entity("B")
        kg.add_relation(a, "knows", b)
        kg.add_relation(a, "founded", b)
        knows = kg.neighbors(a, predicate="knows")
        assert len(knows) == 1
        assert knows[0].predicate == "knows"

    def test_mention_with_string_creates_entity(self) -> None:
        kg = KnowledgeGraph()
        m = kg.mention("Paris", "Paris", source="doc1")
        assert kg.resolve("Paris") is not None
        assert m.source == "doc1"

    def test_resolve_returns_none_when_missing(self) -> None:
        kg = KnowledgeGraph()
        assert kg.resolve("Nobody") is None

    def test_resolve_filters_by_type(self) -> None:
        kg = KnowledgeGraph()
        kg.upsert_entity("Paris", type="location")
        kg.upsert_entity("Paris", type="person")
        loc = kg.resolve("Paris", type="location")
        assert loc is not None
        assert loc.type == "location"

    def test_full_round_trip_with_sqlite(self, tmp_path) -> None:
        store = SQLiteEntityStore(db_path=tmp_path / "kg.db")
        kg = KnowledgeGraph(store)
        sam = kg.upsert_entity("Sam", type="person")
        kg.add_relation(sam, "works_at", "OpenAI")

        # Re-open
        kg2 = KnowledgeGraph(SQLiteEntityStore(db_path=tmp_path / "kg.db"))
        sam2 = kg2.resolve("Sam")
        assert sam2 is not None
        assert len(kg2.neighbors(sam2)) == 1
