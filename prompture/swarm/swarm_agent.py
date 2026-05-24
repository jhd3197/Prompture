"""Swarm participant wrapping a Prompture :class:`~prompture.agents.Agent`.

:class:`SwarmAgent` composes (does not subclass) an existing agent so
that any object exposing a ``run(prompt) -> AgentResult``-shaped result
can participate — :class:`Agent`, :class:`DeepAgent`, a mock agent in
tests, or even a :class:`~prompture.groups.GroupAsAgent` wrapper.

The wrapper supplies three pluggable policies:

- **observe**: assemble the prompt this agent will see this tick.
- **post_act**: write the agent's output back to the environment
  (typically as an :class:`~prompture.swarm.types.Event`).
- **remember**: persist the result to the swarm's memory store.

Each policy has a sensible default; override any of them by passing a
callable in the constructor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..agents.types import AgentResult
    from .environment import Environment
    from .memory import MemoryStore


@runtime_checkable
class _AgentLike(Protocol):
    """The minimal duck-typed interface a swarm participant needs.

    Anything exposing ``run(prompt: str) -> object`` with an
    ``output_text`` attribute on the result satisfies this — including
    :class:`~prompture.agents.Agent`, :class:`~prompture.agents.DeepAgent`,
    and :class:`~prompture.groups.GroupAsAgent`.
    """

    def run(self, prompt: str, **kwargs: Any) -> Any: ...


ObserveFn = Callable[["SwarmAgent", "Environment", int], str]
PostActFn = Callable[["SwarmAgent", "AgentResult", "Environment", int], None]
RememberFn = Callable[["SwarmAgent", "AgentResult", "Environment", int], None]


# ----------------------------------------------------------------------
# Default policies
# ----------------------------------------------------------------------


def default_observe(agent: SwarmAgent, env: Environment, tick: int) -> str:
    """Default observation: persona-relevant slice of the event log + memory.

    Builds a prompt with three sections:

    1. ``ROLE`` — agent name and role (if any).
    2. ``WORLD`` — relevant world-state keys (when ``observe_state_keys``
       is non-empty on the swarm agent).
    3. ``RECENT EVENTS`` — events visible to this agent since the last
       tick it acted (or all events on the first run), filtered by
       ``observe_kinds`` and ``observe_topics`` when set.
    4. ``MEMORY`` — last *k* memories, when a memory store is attached.
    5. ``TICK`` — current round and the action prompt the swarm passed in.

    Returns a single concatenated string the underlying agent will see.
    """
    parts: list[str] = []

    role_line = f"You are agent '{agent.name}'."
    if agent.role:
        role_line += f" Role: {agent.role}."
    parts.append(role_line)

    if agent.observe_state_keys:
        snippets = []
        for key in agent.observe_state_keys:
            if key in env.state:
                snippets.append(f"- {key}: {env.state[key]}")
        if snippets:
            parts.append("## World state\n" + "\n".join(snippets))

    since = agent._last_acted_tick + 1 if agent._last_acted_tick is not None else None
    events = env.events_for(
        agent.name,
        since_tick=since,
        last_n=agent.observe_window,
        kinds=agent.observe_kinds,
        topics=agent.observe_topics,
        include_own=False,
    )
    if events:
        lines = [f"- [t{e.tick}] {e.agent_id} ({e.kind}): {e.content}" for e in events]
        parts.append("## Recent events\n" + "\n".join(lines))

    if agent.memory is not None and agent.memory_recent_k > 0:
        memories = agent.memory.recent(agent.name, k=agent.memory_recent_k)
        if memories:
            lines = [f"- {m.content}" for m in memories]
            parts.append("## Your memory\n" + "\n".join(lines))

    parts.append(f"## Tick {tick}\n{agent._pending_prompt}")
    return "\n\n".join(parts)


def default_post_act(
    agent: SwarmAgent,
    result: AgentResult,
    env: Environment,
    tick: int,
) -> None:
    """Default action: emit a broadcast ``say`` event with the agent's output."""
    text = getattr(result, "output_text", str(result))
    env.emit(
        agent.name,
        text,
        kind=agent.default_event_kind,
        topic=agent.default_topic,
        metadata={"output_type": type(getattr(result, "output", None)).__name__},
    )


def default_remember(
    agent: SwarmAgent,
    result: AgentResult,
    env: Environment,
    tick: int,
) -> None:
    """Default memory write: store this tick's output as a memory entry."""
    if agent.memory is None:
        return
    text = getattr(result, "output_text", str(result))
    agent.memory.add(
        agent.name,
        text,
        tick=tick,
        metadata={"kind": "self_act"},
    )


# ----------------------------------------------------------------------
# SwarmAgent
# ----------------------------------------------------------------------


@dataclass
class SwarmAgent:
    """A swarm participant wrapping an underlying agent.

    Use the constructor directly; the dataclass is mutable so swarms
    can update tick bookkeeping on each instance.

    Args:
        agent: The underlying agent.  Anything with a ``run(prompt)``
            method whose result exposes ``output_text``.
        name: Unique identifier within the swarm.  Used in event
            attribution and memory partitioning.
        role: Optional short role description (e.g. ``"journalist"``,
            ``"contrarian voter"``).  Appears in the default observation.
        memory: Optional :class:`MemoryStore` for this agent's long-term
            memory.  When ``None`` no memory is read or written.
        observe: Override for the observation policy.  Signature
            ``(agent, env, tick) -> str``.
        post_act: Override for the post-action policy.  Signature
            ``(agent, result, env, tick) -> None``.
        remember: Override for the memory-write policy.  Signature
            ``(agent, result, env, tick) -> None``.
        observe_window: Cap on how many events the default observer
            includes per tick.  Use ``None`` to disable the cap.
            Default 20.
        observe_kinds: Optional iterable of event kinds to include in
            the observation.  When ``None`` every kind is included.
        observe_topics: Optional iterable of topics to include in the
            observation.
        observe_state_keys: Iterable of world-state keys to surface in
            the default observation.
        memory_recent_k: Number of recent memories the default observer
            should include.  Default 5.
        default_event_kind: The ``kind`` written by :func:`default_post_act`.
            Default ``"say"``.
        default_topic: Optional topic for the default emitted event.
        priority: Static priority hint readable by
            :class:`~prompture.swarm.scheduler.PriorityScheduler` or
            custom schedulers.
    """

    agent: _AgentLike
    name: str
    role: str = ""
    memory: MemoryStore | None = None
    observe: ObserveFn | None = None
    post_act: PostActFn | None = None
    remember: RememberFn | None = None
    observe_window: int | None = 20
    observe_kinds: tuple[str, ...] | None = None
    observe_topics: tuple[str, ...] | None = None
    observe_state_keys: tuple[str, ...] = ()
    memory_recent_k: int = 5
    default_event_kind: str = "say"
    default_topic: str | None = None
    priority: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    # Mutable bookkeeping — updated by the Swarm runtime.
    _pending_prompt: str = field(default="", repr=False)
    _last_acted_tick: int | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Tick-side API used by Swarm
    # ------------------------------------------------------------------

    def build_observation(self, env: Environment, tick: int) -> str:
        """Run the observation policy and return the prompt."""
        fn = self.observe or default_observe
        return fn(self, env, tick)

    def run_step(self, env: Environment, tick: int, prompt: str) -> AgentResult:
        """Execute one swarm tick for this agent.

        Sets the pending prompt, invokes the observation policy, runs
        the underlying agent, fires the post-action and remember
        policies, and updates ``_last_acted_tick``.
        """
        self._pending_prompt = prompt
        observation = self.build_observation(env, tick)
        result = self.agent.run(observation)

        post_act = self.post_act or default_post_act
        remember = self.remember or default_remember
        post_act(self, result, env, tick)
        remember(self, result, env, tick)

        self._last_acted_tick = tick
        return result  # type: ignore[return-value]
