"""Swarm intelligence runtime for Prompture.

A :class:`Swarm` orchestrates a population of :class:`SwarmAgent`
participants against a shared :class:`Environment` over multiple
rounds (ticks).  Each agent observes a filtered slice of the
environment's event log and (optional) long-term memory, then acts —
writing observations back into the environment for the next round.

Top-level pieces:

- :class:`Swarm` — the orchestrator (ticks, scheduling, budget,
  callbacks, error policy).
- :class:`SwarmAgent` — a participant wrapping any
  :class:`~prompture.agents.Agent`-shaped object via composition.
- :class:`Environment` — append-only event log + mutable world state.
- :class:`MemoryStore` / :class:`InMemoryStore` — per-agent long-term
  memory contract and default backend.
- :class:`Scheduler` and friends — pluggable tick activation policies.
- :class:`Event`, :class:`SwarmResult`, :class:`SwarmStep`,
  :class:`SwarmCallbacks` — data types.

Example::

    from prompture import Agent, Persona
    from prompture.swarm import Swarm, SwarmAgent, Environment, InMemoryStore

    journalist = Agent(
        "openai/gpt-4o",
        system_prompt=Persona(name="journo", system_prompt="You are a city desk reporter."),
    )
    contrarian = Agent(
        "openai/gpt-4o",
        system_prompt=Persona(name="contra", system_prompt="You always argue the opposite."),
    )

    memory = InMemoryStore()
    swarm = Swarm(
        agents=[
            SwarmAgent(journalist, name="journo", role="reporter"),
            SwarmAgent(contrarian, name="contra", role="contrarian"),
        ],
        memory=memory,
        max_ticks=4,
    )
    result = swarm.run(seed="Breaking: city council passes new tax.")
    for ev in result.events:
        print(ev.tick, ev.agent_id, ev.content)
"""

from .environment import ENV_AGENT_ID, Environment
from .memory import InMemoryStore, MemoryStore
from .scheduler import (
    AllScheduler,
    CallableScheduler,
    PriorityScheduler,
    RoundRobinScheduler,
    SampleScheduler,
    Scheduler,
    make_scheduler,
)
from .swarm import Swarm
from .swarm_agent import (
    SwarmAgent,
    default_observe,
    default_post_act,
    default_remember,
)
from .types import (
    Event,
    EventKind,
    Memory,
    SwarmCallbacks,
    SwarmResult,
    SwarmStep,
    aggregate_usage,
)

__all__ = [
    "ENV_AGENT_ID",
    "AllScheduler",
    "CallableScheduler",
    "Environment",
    "Event",
    "EventKind",
    "InMemoryStore",
    "Memory",
    "MemoryStore",
    "PriorityScheduler",
    "RoundRobinScheduler",
    "SampleScheduler",
    "Scheduler",
    "Swarm",
    "SwarmAgent",
    "SwarmCallbacks",
    "SwarmResult",
    "SwarmStep",
    "aggregate_usage",
    "default_observe",
    "default_post_act",
    "default_remember",
    "make_scheduler",
]
