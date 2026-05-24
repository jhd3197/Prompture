"""Storage backends for the knowledge graph.

Two implementations ship out of the box:

- :class:`InMemoryEntityStore` — fast, thread-safe, no dependencies.
- :class:`SQLiteEntityStore` — file-backed durable storage, single DB
  file at ``~/.prompture/kg/kg.db`` by default.

Both satisfy the :class:`EntityStore` protocol.  Bring-your-own
backends (Neo4j, Memgraph, JanusGraph, Zep's GraphRAG) implement the
same eight methods.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .types import Entity, Mention, Relation

_DEFAULT_DB_DIR = Path.home() / ".prompture" / "kg"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "kg.db"


@runtime_checkable
class EntityStore(Protocol):
    """Persistence contract for :class:`KnowledgeGraph`."""

    def upsert_entity(self, entity: Entity) -> None: ...
    def get_entity(self, entity_id: str) -> Entity | None: ...
    def find_entities(
        self, *, name: str | None = None, type: str | None = None, limit: int | None = None
    ) -> list[Entity]: ...
    def delete_entity(self, entity_id: str) -> bool: ...

    def add_relation(self, relation: Relation) -> None: ...
    def relations_for(
        self,
        entity_id: str,
        *,
        as_subject: bool = True,
        as_object: bool = True,
        predicate: str | None = None,
    ) -> list[Relation]: ...

    def add_mention(self, mention: Mention) -> None: ...
    def mentions_for(self, entity_id: str) -> list[Mention]: ...


# ----------------------------------------------------------------------
# In-memory
# ----------------------------------------------------------------------


class InMemoryEntityStore:
    """Thread-safe in-process :class:`EntityStore`."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entities: dict[str, Entity] = {}
        # name (lowercased) → set of entity ids — supports alias lookup
        self._by_name: dict[str, set[str]] = defaultdict(set)
        self._by_type: dict[str, set[str]] = defaultdict(set)
        self._relations: dict[str, Relation] = {}
        self._rel_by_subject: dict[str, set[str]] = defaultdict(set)
        self._rel_by_object: dict[str, set[str]] = defaultdict(set)
        self._mentions: dict[str, list[Mention]] = defaultdict(list)

    # ---- entities ------------------------------------------------------

    def upsert_entity(self, entity: Entity) -> None:
        with self._lock:
            existing = self._entities.get(entity.id)
            if existing is not None:
                # Drop old index entries before re-indexing.
                self._unindex_entity(existing)
            self._entities[entity.id] = entity
            self._index_entity(entity)

    def _index_entity(self, entity: Entity) -> None:
        self._by_name[entity.name.lower()].add(entity.id)
        for alias in entity.aliases:
            self._by_name[alias.lower()].add(entity.id)
        self._by_type[entity.type].add(entity.id)

    def _unindex_entity(self, entity: Entity) -> None:
        self._by_name[entity.name.lower()].discard(entity.id)
        for alias in entity.aliases:
            self._by_name[alias.lower()].discard(entity.id)
        self._by_type[entity.type].discard(entity.id)

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._lock:
            return self._entities.get(entity_id)

    def find_entities(
        self,
        *,
        name: str | None = None,
        type: str | None = None,
        limit: int | None = None,
    ) -> list[Entity]:
        with self._lock:
            if name is not None:
                ids: Iterable[str] = self._by_name.get(name.lower(), ())
            elif type is not None:
                ids = self._by_type.get(type, ())
            else:
                ids = self._entities.keys()
            if type is not None and name is not None:
                ids = [i for i in ids if self._entities[i].type == type]
            out = [self._entities[i] for i in ids]
        out.sort(key=lambda e: e.ts, reverse=True)
        if limit is not None and limit >= 0:
            out = out[:limit]
        return out

    def delete_entity(self, entity_id: str) -> bool:
        with self._lock:
            ent = self._entities.pop(entity_id, None)
            if ent is None:
                return False
            self._unindex_entity(ent)
            # Cascade: drop edges referencing this id.
            for rel_id in list(self._rel_by_subject.get(entity_id, ())) + list(
                self._rel_by_object.get(entity_id, ())
            ):
                rel = self._relations.pop(rel_id, None)
                if rel is not None:
                    self._rel_by_subject[rel.subject_id].discard(rel.id)
                    self._rel_by_object[rel.object_id].discard(rel.id)
            self._rel_by_subject.pop(entity_id, None)
            self._rel_by_object.pop(entity_id, None)
            self._mentions.pop(entity_id, None)
            return True

    # ---- relations -----------------------------------------------------

    def add_relation(self, relation: Relation) -> None:
        with self._lock:
            self._relations[relation.id] = relation
            self._rel_by_subject[relation.subject_id].add(relation.id)
            self._rel_by_object[relation.object_id].add(relation.id)

    def relations_for(
        self,
        entity_id: str,
        *,
        as_subject: bool = True,
        as_object: bool = True,
        predicate: str | None = None,
    ) -> list[Relation]:
        with self._lock:
            ids: set[str] = set()
            if as_subject:
                ids.update(self._rel_by_subject.get(entity_id, ()))
            if as_object:
                ids.update(self._rel_by_object.get(entity_id, ()))
            rels = [self._relations[i] for i in ids]
        if predicate is not None:
            rels = [r for r in rels if r.predicate == predicate]
        rels.sort(key=lambda r: r.ts, reverse=True)
        return rels

    # ---- mentions ------------------------------------------------------

    def add_mention(self, mention: Mention) -> None:
        with self._lock:
            self._mentions[mention.entity_id].append(mention)

    def mentions_for(self, entity_id: str) -> list[Mention]:
        with self._lock:
            return list(self._mentions.get(entity_id, ()))


# ----------------------------------------------------------------------
# SQLite
# ----------------------------------------------------------------------


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    attributes TEXT NOT NULL DEFAULT '{}',
    ts REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);

CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_id TEXT NOT NULL,
    attributes TEXT NOT NULL DEFAULT '{}',
    ts REAL NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES entities(id) ON DELETE CASCADE,
    FOREIGN KEY (object_id) REFERENCES entities(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_id);
CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_id);
CREATE INDEX IF NOT EXISTS idx_relations_predicate ON relations(predicate);

CREATE TABLE IF NOT EXISTS mentions (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT,
    span_start INTEGER,
    span_end INTEGER,
    confidence REAL NOT NULL DEFAULT 1.0,
    attributes TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_mentions_entity ON mentions(entity_id);
"""


class SQLiteEntityStore:
    """File-backed durable :class:`EntityStore`."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    # ---- entities ------------------------------------------------------

    def upsert_entity(self, entity: Entity) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO entities (id, name, type, aliases, attributes, ts)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        type = excluded.type,
                        aliases = excluded.aliases,
                        attributes = excluded.attributes,
                        ts = excluded.ts
                    """,
                    (
                        entity.id,
                        entity.name,
                        entity.type,
                        json.dumps(list(entity.aliases), ensure_ascii=False),
                        json.dumps(entity.attributes, ensure_ascii=False),
                        entity.ts,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM entities WHERE id = ?", (entity_id,)
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return None
        return self._row_to_entity(row)

    def find_entities(
        self,
        *,
        name: str | None = None,
        type: str | None = None,
        limit: int | None = None,
    ) -> list[Entity]:
        sql = "SELECT * FROM entities"
        params: list[Any] = []
        wheres: list[str] = []
        if name is not None:
            wheres.append("(LOWER(name) = ? OR aliases LIKE ?)")
            params.extend([name.lower(), f'%"{name}"%'])
        if type is not None:
            wheres.append("type = ?")
            params.append(type)
        if wheres:
            sql += " WHERE " + " AND ".join(wheres)
        sql += " ORDER BY ts DESC"
        if limit is not None and limit >= 0:
            sql += " LIMIT ?"
            params.append(limit)

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
            finally:
                conn.close()
        return [self._row_to_entity(r) for r in rows]

    def delete_entity(self, entity_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    # ---- relations -----------------------------------------------------

    def add_relation(self, relation: Relation) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO relations (id, subject_id, predicate, object_id, attributes, ts)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        subject_id = excluded.subject_id,
                        predicate = excluded.predicate,
                        object_id = excluded.object_id,
                        attributes = excluded.attributes,
                        ts = excluded.ts
                    """,
                    (
                        relation.id,
                        relation.subject_id,
                        relation.predicate,
                        relation.object_id,
                        json.dumps(relation.attributes, ensure_ascii=False),
                        relation.ts,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def relations_for(
        self,
        entity_id: str,
        *,
        as_subject: bool = True,
        as_object: bool = True,
        predicate: str | None = None,
    ) -> list[Relation]:
        if not (as_subject or as_object):
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if as_subject:
            clauses.append("subject_id = ?")
            params.append(entity_id)
        if as_object:
            clauses.append("object_id = ?")
            params.append(entity_id)
        sql = f"SELECT * FROM relations WHERE ({' OR '.join(clauses)})"
        if predicate is not None:
            sql += " AND predicate = ?"
            params.append(predicate)
        sql += " ORDER BY ts DESC"

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
            finally:
                conn.close()
        return [self._row_to_relation(r) for r in rows]

    # ---- mentions ------------------------------------------------------

    def add_mention(self, mention: Mention) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO mentions
                        (id, entity_id, text, source, span_start, span_end, confidence, attributes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mention.id,
                        mention.entity_id,
                        mention.text,
                        mention.source,
                        mention.start,
                        mention.end,
                        mention.confidence,
                        json.dumps(mention.attributes, ensure_ascii=False),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def mentions_for(self, entity_id: str) -> list[Mention]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM mentions WHERE entity_id = ?", (entity_id,)
                ).fetchall()
            finally:
                conn.close()
        return [self._row_to_mention(r) for r in rows]

    # ---- row decoders --------------------------------------------------

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            name=row["name"],
            type=row["type"],
            aliases=tuple(json.loads(row["aliases"]) if row["aliases"] else ()),
            attributes=json.loads(row["attributes"]) if row["attributes"] else {},
            ts=float(row["ts"]),
        )

    @staticmethod
    def _row_to_relation(row: sqlite3.Row) -> Relation:
        return Relation(
            id=row["id"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_id=row["object_id"],
            attributes=json.loads(row["attributes"]) if row["attributes"] else {},
            ts=float(row["ts"]),
        )

    @staticmethod
    def _row_to_mention(row: sqlite3.Row) -> Mention:
        return Mention(
            id=row["id"],
            entity_id=row["entity_id"],
            text=row["text"],
            source=row["source"],
            start=row["span_start"],
            end=row["span_end"],
            confidence=float(row["confidence"]),
            attributes=json.loads(row["attributes"]) if row["attributes"] else {},
        )
