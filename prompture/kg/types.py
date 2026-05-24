"""Types for the knowledge-graph entity store."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def _slugify(value: str) -> str:
    """Lowercase + collapse whitespace, used to canonicalise entity ids."""
    return " ".join(value.lower().split())


@dataclass
class Entity:
    """A node in the knowledge graph.

    Attributes:
        id: Stable identifier.  Use a slug, URI, or wikidata id when
            available; otherwise :meth:`KnowledgeGraph.upsert_entity`
            derives one from ``name``.
        name: Canonical human-readable name (used for display).
        type: Coarse class (``"person"``, ``"organization"``, ``"city"``,
            ``"product"``, etc.).  Free-form; no enum is enforced.
        aliases: Alternative surface forms that resolve to this entity.
        attributes: Free-form structured payload (birth date, market
            cap, GPS coords).  Backends serialise as JSON.
        ts: Wall-clock timestamp of last update.
    """

    id: str
    name: str
    type: str = "thing"
    aliases: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    @classmethod
    def make_id(cls, name: str, type: str | None = None) -> str:
        """Derive a deterministic id from a *name* (and optional *type*)."""
        slug = _slugify(name).replace(" ", "_")
        if type:
            return f"{_slugify(type)}:{slug}"
        return slug

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "aliases": list(self.aliases),
            "attributes": self.attributes,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        return cls(
            id=data["id"],
            name=data["name"],
            type=data.get("type", "thing"),
            aliases=tuple(data.get("aliases") or ()),
            attributes=dict(data.get("attributes") or {}),
            ts=float(data.get("ts", time.time())),
        )


@dataclass
class Relation:
    """A directed edge between two entities.

    Attributes:
        id: Stable identifier (auto-generated UUID4 hex when omitted).
        subject_id: Source entity id.
        predicate: Relation label (``"works_at"``, ``"located_in"``,
            ``"acquired"``).  Free-form; no enum is enforced.
        object_id: Target entity id.
        attributes: Free-form structured payload (start_date,
            confidence, source).
        ts: Wall-clock timestamp.
    """

    subject_id: str
    predicate: str
    object_id: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    attributes: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "attributes": self.attributes,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relation:
        return cls(
            id=data.get("id", uuid.uuid4().hex),
            subject_id=data["subject_id"],
            predicate=data["predicate"],
            object_id=data["object_id"],
            attributes=dict(data.get("attributes") or {}),
            ts=float(data.get("ts", time.time())),
        )


@dataclass
class Mention:
    """A reference to an entity inside a piece of source text.

    Attributes:
        entity_id: The mentioned entity.
        text: The exact surface form (the slice of source text).
        source: Identifier for the document the mention came from (URL,
            file path, document id).
        start: Character offset of the mention's start, when available.
        end: Character offset (exclusive) of the mention's end.
        confidence: Detector confidence on ``[0.0, 1.0]``.  Defaults to
            ``1.0`` for hand-asserted mentions.
        attributes: Free-form structured payload.
    """

    entity_id: str
    text: str
    source: str | None = None
    start: int | None = None
    end: int | None = None
    confidence: float = 1.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity_id": self.entity_id,
            "text": self.text,
            "source": self.source,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Mention:
        return cls(
            id=data.get("id", uuid.uuid4().hex),
            entity_id=data["entity_id"],
            text=data["text"],
            source=data.get("source"),
            start=data.get("start"),
            end=data.get("end"),
            confidence=float(data.get("confidence", 1.0)),
            attributes=dict(data.get("attributes") or {}),
        )
