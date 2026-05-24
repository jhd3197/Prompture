#!/usr/bin/env python3
"""Deep-research example — wiring citations + session memory + checkpoints + KG.

A small Perplexity-style flow:

1. Pretend a retriever found three sources about a topic.
2. Build a citation-instructed prompt with CitationTracker.
3. Ask the LLM (one call) for a cited answer.
4. Parse the citations back into typed records.
5. Stash a user preference in SessionMemory so the next session is
   primed.
6. Save a Checkpoint so the run survives a process restart.
7. Add the cited entities to a KnowledgeGraph.

Usage:
    python examples/deep_research_example.py

Requires:
    - A valid API key for the configured MODEL (defaults to
      openai/gpt-4o-mini).  Swap to an Ollama / local model to run free.
"""

from __future__ import annotations

from prompture import (
    Checkpoint,
    CheckpointManager,
    CitationTracker,
    Conversation,
    InMemoryCheckpointStore,
    InMemorySessionStore,
    KnowledgeGraph,
    SessionMemory,
    Source,
)

MODEL = "openai/gpt-4o-mini"
USER_ID = "alice"
SESSION_ID = "trip-research-2026-05"
RUN_ID = "paris-deep-dive"


# ── Step 1: pretend the retriever found these (in a real flow this is RAG) ──

sources = [
    Source(
        id="1",
        title="Wikipedia: Paris",
        text=(
            "Paris is the capital and most populous city of France, "
            "with a population of 2,102,650 within its administrative limits."
        ),
    ),
    Source(
        id="2",
        title="Visit Paris — Eiffel Tower",
        text=(
            "The Eiffel Tower was completed in 1889 for the World's Fair and "
            "stands 330 metres tall on the Champ de Mars."
        ),
    ),
    Source(
        id="3",
        title="Louvre Museum facts",
        text=(
            "The Louvre is the world's most-visited art museum. It was "
            "originally a fortress built by King Philip II in the late 12th century."
        ),
    ),
]


# ── Step 2: build the citation-instructed prompt ────────────────────────────

tracker = CitationTracker(sources=sources, question="What should I see on a 2-day Paris trip?")
prompt = tracker.build_prompt()

print("=" * 60)
print("Step 1: Sending citation-instructed prompt to LLM")
print("=" * 60)
print(prompt[:400] + "…\n")


# ── Step 3: ask the LLM once ────────────────────────────────────────────────

conv = Conversation(model_name=MODEL)
response = conv.ask(prompt)


# ── Step 4: parse the citations ─────────────────────────────────────────────

answer = tracker.parse_response(response, usage=conv.usage)

print("=" * 60)
print("Step 2: Parsed cited answer")
print("=" * 60)
print(f"Grounded: {answer.is_grounded} | Coverage: {answer.coverage:.0%}")
print(f"Cited sources: {sorted(answer.cited_source_ids)}")
print(f"\n{answer.clean_answer}\n")
for c in answer.citations:
    print(f"  • {c.text!r:60s} → {c.source_ids}")


# ── Step 5: remember a user preference for next session ────────────────────

memory = SessionMemory(
    user_id=USER_ID,
    session_id=SESSION_ID,
    store=InMemorySessionStore(),
)
memory.remember("Prefers museum-heavy itineraries over nightlife.", kind="preference", importance=0.8)
memory.remember(f"Currently researching: {tracker._question}", session_scoped=True)

print("=" * 60)
print("Step 3: Session memory")
print("=" * 60)
print(memory.render_for_prompt("paris"))


# ── Step 6: checkpoint so we can resume ────────────────────────────────────

store = InMemoryCheckpointStore()
mgr = CheckpointManager(store, run_id=RUN_ID, agent_name="research-agent")
mgr.save_conversation(
    conv,
    iteration=1,
    prompt=tracker._question or "",
    metadata={
        "cited_sources": sorted(answer.cited_source_ids),
        "coverage": answer.coverage,
    },
)
mgr.mark("completed")

print("=" * 60)
print("Step 4: Checkpoint saved")
print("=" * 60)
cp: Checkpoint | None = mgr.load()
assert cp is not None
print(f"run_id        : {cp.run_id}")
print(f"status        : {cp.status}")
print(f"iteration     : {cp.iteration}")
print(f"messages saved: {len(cp.messages)}")
print(f"coverage meta : {cp.metadata.get('coverage')}")


# ── Step 7: drop the cited entities into a knowledge graph ─────────────────

kg = KnowledgeGraph()
kg.upsert_entity("Paris", type="location")
kg.upsert_entity("Eiffel Tower", type="landmark")
kg.upsert_entity("Louvre", type="museum", aliases=("Louvre Museum",))

kg.add_relation("Eiffel Tower", "located_in", "Paris")
kg.add_relation("Louvre", "located_in", "Paris")

# Mark which entities came up in cited claims.
for cit in answer.citations:
    for ent_name in ("Paris", "Eiffel Tower", "Louvre"):
        if ent_name.lower() in cit.text.lower():
            kg.mention(ent_name, ent_name, source=f"source:{cit.first_source()}")

print()
print("=" * 60)
print("Step 5: Knowledge graph")
print("=" * 60)
paris = kg.resolve("Paris")
assert paris is not None
print(f"Neighbors of {paris.name}:")
for rel in kg.neighbors(paris, direction="in"):
    other = kg.get(rel.subject_id)
    other_name = other.name if other is not None else rel.subject_id
    print(f"  ← {other_name} {rel.predicate} → {paris.name}")

for ent_name in ("Paris", "Eiffel Tower", "Louvre"):
    mentions = kg.mentions(ent_name)
    print(f"{ent_name:14s}: {len(mentions)} mention(s)")
