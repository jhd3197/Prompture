"""High-level :class:`KnowledgeGraph` API and LLM-backed extraction helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .stores import EntityStore, InMemoryEntityStore
from .types import Entity, Mention, Relation

if TYPE_CHECKING:
    pass


class KnowledgeGraph:
    """Convenience facade over an :class:`EntityStore`.

    The store does the raw bookkeeping; this class layers on the
    ergonomics callers actually want: entity resolution by name,
    one-call ``upsert_entity`` returning a typed :class:`Entity`,
    neighbourhood traversal, and helpers that pair with the
    :func:`extract_entities` / :func:`extract_relations` LLM extractors.

    Args:
        store: Backing :class:`EntityStore`.  Defaults to
            :class:`InMemoryEntityStore`.

    Example::

        kg = KnowledgeGraph()
        sam = kg.upsert_entity("Sam Altman", type="person")
        oai = kg.upsert_entity("OpenAI", type="organization")
        kg.add_relation(sam, "works_at", oai)
        for r in kg.neighbors(sam):
            print(r.predicate, "→", kg.get(r.object_id).name)
    """

    def __init__(self, store: EntityStore | None = None) -> None:
        self._store: EntityStore = store if store is not None else InMemoryEntityStore()

    # ------------------------------------------------------------------
    # Direct accessors
    # ------------------------------------------------------------------

    @property
    def store(self) -> EntityStore:
        return self._store

    def get(self, entity_id: str) -> Entity | None:
        """Return an entity by id, or ``None`` if not found."""
        return self._store.get_entity(entity_id)

    def find(
        self,
        *,
        name: str | None = None,
        type: str | None = None,
        limit: int | None = None,
    ) -> list[Entity]:
        """Search entities by name (exact, case-insensitive, alias-aware) or type."""
        return self._store.find_entities(name=name, type=type, limit=limit)

    def resolve(self, name: str, *, type: str | None = None) -> Entity | None:
        """Look up the most recently-updated entity matching *name*.

        Returns ``None`` when no match exists.  Filters by *type* when
        supplied — useful when names collide across categories
        (``Paris`` the city vs ``Paris`` the person).
        """
        matches = self.find(name=name, type=type, limit=1)
        return matches[0] if matches else None

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def upsert_entity(
        self,
        name_or_entity: str | Entity,
        *,
        type: str = "thing",
        aliases: Iterable[str] = (),
        attributes: dict[str, Any] | None = None,
        entity_id: str | None = None,
    ) -> Entity:
        """Insert a new entity or merge into an existing one.

        Accepts either a fully-built :class:`Entity` (returned
        unchanged after persisting) or a name + descriptor pair.

        Merge semantics:
        - The id is the canonical key.  If *entity_id* is omitted, the
          name+type slug from :meth:`Entity.make_id` is used.
        - Aliases and attributes are *merged* with any existing record;
          attribute values supplied here win on conflict.
        - The most recent timestamp wins.
        """
        if isinstance(name_or_entity, Entity):
            self._store.upsert_entity(name_or_entity)
            return name_or_entity

        eid = entity_id or Entity.make_id(name_or_entity, type=type)
        existing = self._store.get_entity(eid)
        if existing is not None:
            merged_aliases = tuple(dict.fromkeys((*existing.aliases, *aliases)))
            merged_attrs = {**existing.attributes, **(attributes or {})}
            entity = Entity(
                id=eid,
                name=name_or_entity or existing.name,
                type=type or existing.type,
                aliases=merged_aliases,
                attributes=merged_attrs,
            )
        else:
            entity = Entity(
                id=eid,
                name=name_or_entity,
                type=type,
                aliases=tuple(aliases),
                attributes=dict(attributes or {}),
            )
        self._store.upsert_entity(entity)
        return entity

    def add_relation(
        self,
        subject: str | Entity,
        predicate: str,
        object: str | Entity,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> Relation:
        """Record a directed edge between two entities.

        Strings are resolved against the graph; unknown names are
        upserted as ``type="thing"`` so the edge always lands on real
        nodes.
        """
        subj_id = subject.id if isinstance(subject, Entity) else self.upsert_entity(subject).id
        obj_id = object.id if isinstance(object, Entity) else self.upsert_entity(object).id
        relation = Relation(
            subject_id=subj_id,
            predicate=predicate,
            object_id=obj_id,
            attributes=dict(attributes or {}),
        )
        self._store.add_relation(relation)
        return relation

    def mention(
        self,
        entity: str | Entity,
        text: str,
        *,
        source: str | None = None,
        start: int | None = None,
        end: int | None = None,
        confidence: float = 1.0,
        attributes: dict[str, Any] | None = None,
    ) -> Mention:
        """Record a surface-form mention of an entity in a document.

        String *entity* values are resolved or upserted as
        ``type="thing"``.
        """
        ent = entity if isinstance(entity, Entity) else self.upsert_entity(entity)
        m = Mention(
            entity_id=ent.id,
            text=text,
            source=source,
            start=start,
            end=end,
            confidence=confidence,
            attributes=dict(attributes or {}),
        )
        self._store.add_mention(m)
        return m

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def neighbors(
        self,
        entity: str | Entity,
        *,
        predicate: str | None = None,
        direction: str = "both",
    ) -> list[Relation]:
        """Return relations touching *entity*.

        Args:
            entity: An :class:`Entity` or its id.
            predicate: Restrict to a specific predicate when set.
            direction: ``"out"`` (subject only), ``"in"`` (object
                only), or ``"both"`` (default).
        """
        eid = entity.id if isinstance(entity, Entity) else entity
        as_subject = direction in ("out", "both")
        as_object = direction in ("in", "both")
        return self._store.relations_for(
            eid,
            as_subject=as_subject,
            as_object=as_object,
            predicate=predicate,
        )

    def mentions(self, entity: str | Entity) -> list[Mention]:
        """Return every mention of *entity*."""
        eid = entity.id if isinstance(entity, Entity) else entity
        return self._store.mentions_for(eid)


# ----------------------------------------------------------------------
# LLM-backed extraction helpers
# ----------------------------------------------------------------------


_ENTITY_EXTRACTION_PROMPT = (
    "Extract every distinct entity mentioned in the text. "
    "Return a JSON array where each element has 'name' (canonical "
    "display name), 'type' (one of: person, organization, location, "
    "product, event, concept, other), and 'aliases' (array of "
    "alternative surface forms found in the text). "
    "Do not include duplicates."
)

_RELATION_EXTRACTION_PROMPT = (
    "Extract every directed relation between entities in the text. "
    "Return a JSON array where each element has 'subject' (entity name), "
    "'predicate' (verb-like phrase in snake_case, e.g. 'works_at', "
    "'located_in', 'founded'), and 'object' (entity name). "
    "Only emit relations explicitly supported by the text."
)


def _resolve_extract_with_model() -> Any:
    """Lazy import to avoid a heavy dependency at module import time."""
    from ..extraction.core import extract_with_model

    return extract_with_model


def extract_entities(
    text: str,
    *,
    model: str,
    graph: KnowledgeGraph | None = None,
    source: str | None = None,
    instruction: str = _ENTITY_EXTRACTION_PROMPT,
) -> list[Entity]:
    """Extract entities from free text using an LLM and upsert into *graph*.

    Uses :func:`prompture.extraction.extract_with_model` with an inline
    Pydantic schema, so it benefits from the same retry / cleanup
    handling as the rest of the extraction stack.

    Args:
        text: Free-form text to analyse.
        model: Provider/model string (e.g. ``"openai/gpt-4o-mini"``).
        graph: When supplied, every extracted entity is upserted into
            it and surface-form mentions are recorded.  When ``None``
            entities are still returned but nothing is persisted.
        source: Document id propagated onto :class:`Mention` records.
        instruction: Override the default extraction directive.

    Returns:
        List of :class:`Entity` records (newly created or upserted).
    """
    from pydantic import BaseModel, Field

    extract_with_model = _resolve_extract_with_model()

    class _ExtractedEntity(BaseModel):
        name: str = Field(..., description="Canonical display name")
        type: str = Field("other", description="Coarse class")
        aliases: list[str] = Field(default_factory=list)

    class _EntityList(BaseModel):
        entities: list[_ExtractedEntity] = Field(default_factory=list)

    result = extract_with_model(
        _EntityList,
        text,
        model=model,
        instruction_template=instruction,
    )

    extracted: list[_ExtractedEntity] = []
    payload = getattr(result, "data", result)
    if isinstance(payload, _EntityList):
        extracted = payload.entities
    elif isinstance(payload, dict) and "entities" in payload:
        extracted = [_ExtractedEntity.model_validate(e) for e in payload["entities"]]

    entities: list[Entity] = []
    for item in extracted:
        if graph is not None:
            ent = graph.upsert_entity(item.name, type=item.type, aliases=tuple(item.aliases))
            for alias in [item.name, *item.aliases]:
                if alias and alias in text:
                    graph.mention(ent, alias, source=source)
        else:
            ent = Entity(
                id=Entity.make_id(item.name, type=item.type),
                name=item.name,
                type=item.type,
                aliases=tuple(item.aliases),
            )
        entities.append(ent)
    return entities


def extract_relations(
    text: str,
    *,
    model: str,
    graph: KnowledgeGraph,
    instruction: str = _RELATION_EXTRACTION_PROMPT,
) -> list[Relation]:
    """Extract directed relations from text and add them to *graph*.

    Subjects/objects are resolved against existing entities by name
    (case-insensitive, alias-aware); unknown names are upserted as
    ``type="thing"`` so the edge always lands on real nodes.

    Args:
        text: Free-form text to analyse.
        model: Provider/model string.
        graph: Required — relations are persisted into this graph.
        instruction: Override the default extraction directive.

    Returns:
        List of :class:`Relation` records that were added.
    """
    from pydantic import BaseModel, Field

    extract_with_model = _resolve_extract_with_model()

    class _ExtractedRelation(BaseModel):
        subject: str
        predicate: str
        object: str

    class _RelationList(BaseModel):
        relations: list[_ExtractedRelation] = Field(default_factory=list)

    result = extract_with_model(
        _RelationList,
        text,
        model=model,
        instruction_template=instruction,
    )
    payload = getattr(result, "data", result)
    if isinstance(payload, _RelationList):
        extracted = payload.relations
    elif isinstance(payload, dict) and "relations" in payload:
        extracted = [_ExtractedRelation.model_validate(r) for r in payload["relations"]]
    else:
        extracted = []

    out: list[Relation] = []
    for r in extracted:
        subj = graph.resolve(r.subject) or graph.upsert_entity(r.subject)
        obj = graph.resolve(r.object) or graph.upsert_entity(r.object)
        out.append(graph.add_relation(subj, r.predicate, obj))
    return out
