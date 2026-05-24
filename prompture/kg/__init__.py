"""Knowledge-graph entity store for Prompture.

Tracks entities (people, organisations, products, places) and the
relations between them as documents flow through extraction.  Designed
to pair with:

- :mod:`prompture.swarm` — agents can attach extracted entities to the
  shared environment.
- :mod:`prompture.rag` — retrievers can ground citations against
  entity-mention spans.
- :mod:`prompture.extraction` — :func:`extract_entities` uses the
  existing Pydantic-typed extraction machinery.

Three layers:

- :class:`Entity`, :class:`Relation`, :class:`Mention` — graph atoms.
- :class:`EntityStore` — pluggable backend protocol.  Ships with
  :class:`InMemoryEntityStore` and :class:`SQLiteEntityStore`.
- :class:`KnowledgeGraph` — high-level API: ``upsert_entity`` /
  ``add_relation`` / ``mention`` / ``neighbors`` / ``find``.
- :func:`extract_entities` / :func:`extract_relations` — LLM-backed
  helpers that consume free text and populate the graph.
"""

from .core import KnowledgeGraph, extract_entities, extract_relations
from .stores import (
    EntityStore,
    InMemoryEntityStore,
    SQLiteEntityStore,
)
from .types import Entity, Mention, Relation

__all__ = [
    "Entity",
    "EntityStore",
    "InMemoryEntityStore",
    "KnowledgeGraph",
    "Mention",
    "Relation",
    "SQLiteEntityStore",
    "extract_entities",
    "extract_relations",
]
