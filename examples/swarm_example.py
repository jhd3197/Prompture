#!/usr/bin/env python3
"""Swarm runtime example — populating a shared environment with reacting agents.

Demonstrates:
1. Building a population of Persona-driven Agents
2. Wrapping them as SwarmAgents with custom roles and observation policies
3. Seeding the environment with a news event
4. Running a multi-tick swarm with the default broadcast scheduler
5. Switching to RoundRobinScheduler and SampleScheduler
6. Reading the per-agent timeline, event log, and memory
7. Stopping early via stop_when

Usage:
    python examples/swarm_example.py

Requires:
    - A valid API key for the LLM provider used (OpenAI here).
    - Or set MODEL to an Ollama / local model to run free.
"""

from __future__ import annotations

from prompture import Agent, Persona
from prompture.swarm import (
    Environment,
    InMemoryStore,
    RoundRobinScheduler,
    SampleScheduler,
    Swarm,
    SwarmAgent,
    SwarmCallbacks,
)

MODEL = "openai/gpt-4o-mini"


# ── Section 1: Build the population ──────────────────────────────────────

journalist = Agent(
    MODEL,
    system_prompt=Persona(
        name="journalist",
        system_prompt=(
            "You are a city-desk reporter. Answer in one short factual sentence "
            "describing the angle you would investigate next."
        ),
    ),
)

contrarian = Agent(
    MODEL,
    system_prompt=Persona(
        name="contrarian",
        system_prompt=(
            "You are a contrarian commentator. Answer in one short sentence "
            "challenging the most recent claim."
        ),
    ),
)

resident = Agent(
    MODEL,
    system_prompt=Persona(
        name="resident",
        system_prompt=(
            "You are a long-time city resident. Answer in one short sentence "
            "expressing how the news affects your daily life."
        ),
    ),
)

memory = InMemoryStore()
env = Environment()

agents = [
    SwarmAgent(journalist, name="journalist", role="city reporter"),
    SwarmAgent(contrarian, name="contrarian", role="loud sceptic"),
    SwarmAgent(resident, name="resident", role="long-time resident"),
]


# ── Section 2: Default swarm — every agent every tick ────────────────────

print("=" * 60)
print("Section 1: AllScheduler — every agent every tick")
print("=" * 60)

swarm = Swarm(
    agents=agents,
    env=env,
    memory=memory,
    max_ticks=3,
    callbacks=SwarmCallbacks(
        on_tick_start=lambda t, names: print(f"  → tick {t}: {names}"),
    ),
)
result = swarm.run(seed="The city council just passed a 12% transit tax.")

print(f"\nticks_run = {result.ticks_run}")
print(f"events    = {len(result.events)}")
print(f"cost      = ${result.aggregate_usage['total_cost']:.4f}")
print()
print("Timeline:")
for step in result.timeline:
    snippet = (step.output or "").replace("\n", " ")[:80]
    print(f"  [t{step.tick}] {step.agent_id:<11} → {snippet}")


# ── Section 3: RoundRobin — one agent per tick ───────────────────────────

print()
print("=" * 60)
print("Section 2: RoundRobinScheduler — one voice at a time")
print("=" * 60)

rr_swarm = Swarm(
    agents=agents,
    env=Environment(),
    memory=InMemoryStore(),
    scheduler=RoundRobinScheduler(),
    max_ticks=4,
)
rr_swarm.environment.seed("Storm warning: heavy rain expected tomorrow.")
rr_result = rr_swarm.run()
for step in rr_result.timeline:
    print(f"  [t{step.tick}] {step.agent_id}")


# ── Section 4: Sampling — gossip-style stochastic activation ─────────────

print()
print("=" * 60)
print("Section 3: SampleScheduler — random k per tick")
print("=" * 60)

import random

sample_swarm = Swarm(
    agents=agents,
    env=Environment(),
    scheduler=SampleScheduler(k=2, rng=random.Random(7)),
    max_ticks=3,
)
sample_swarm.environment.seed("Tech startup announces hiring spree.")
sample_result = sample_swarm.run()
for step in sample_result.timeline:
    snippet = (step.output or "").replace("\n", " ")[:80]
    print(f"  [t{step.tick}] {step.agent_id:<11} → {snippet}")


# ── Section 5: stop_when — end the run on a custom condition ─────────────

print()
print("=" * 60)
print("Section 4: stop_when — bail when consensus emerges")
print("=" * 60)


def consensus_reached(env, tick) -> bool:
    """Stop once the contrarian has echoed a key term twice."""
    contrarian_says = [
        e.content.lower()
        for e in env.events
        if e.agent_id == "contrarian"
    ]
    return sum("agree" in c for c in contrarian_says) >= 2


early_swarm = Swarm(
    agents=agents,
    env=Environment(),
    max_ticks=10,
    stop_when=consensus_reached,
)
early_swarm.environment.seed("Mayor proposes a city-wide bike network.")
early_result = early_swarm.run()
print(f"stopped_reason: {early_result.stopped_reason}")
print(f"ticks_run     : {early_result.ticks_run}")
