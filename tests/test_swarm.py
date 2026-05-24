"""Tests for the prompture.swarm runtime."""

from __future__ import annotations

import random
from typing import Any
from unittest.mock import MagicMock

import pytest

from prompture.agents.types import AgentResult, AgentState
from prompture.groups.types import ErrorPolicy
from prompture.swarm import (
    AllScheduler,
    CallableScheduler,
    Environment,
    Event,
    InMemoryStore,
    PriorityScheduler,
    RoundRobinScheduler,
    SampleScheduler,
    Swarm,
    SwarmAgent,
    SwarmCallbacks,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _mock_agent(name: str, outputs: list[str] | None = None) -> MagicMock:
    """Create a MagicMock agent that returns successive canned outputs."""
    outputs = outputs or [f"{name}-output"]
    agent = MagicMock()
    agent._counter = 0

    def _run(prompt: str, **_: Any) -> AgentResult:
        idx = min(agent._counter, len(outputs) - 1)
        agent._counter += 1
        text = outputs[idx]
        return AgentResult(
            output=text,
            output_text=text,
            messages=[],
            usage={},
            state=AgentState.idle,
            run_usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "total_cost": 0.001,
            },
        )

    agent.run.side_effect = _run
    agent.name = name
    return agent


def _make_swarm_agent(
    name: str,
    outputs: list[str] | None = None,
    **kwargs: Any,
) -> SwarmAgent:
    inner = _mock_agent(name, outputs)
    return SwarmAgent(agent=inner, name=name, **kwargs)


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------


class TestEnvironment:
    def test_emit_and_events(self) -> None:
        env = Environment()
        env.advance_tick()
        ev = env.emit("alice", "hello", kind="say")
        assert isinstance(ev, Event)
        assert ev.tick == 1
        assert ev.agent_id == "alice"
        assert env.events[0].content == "hello"

    def test_broadcast_visible_to_all(self) -> None:
        env = Environment()
        env.emit("alice", "broadcast")
        assert len(env.events_for("bob")) == 1
        assert len(env.events_for("carol")) == 1

    def test_targeted_visible_only_to_targets_and_author(self) -> None:
        env = Environment()
        env.emit("alice", "private", targets=("bob",))
        assert len(env.events_for("bob")) == 1
        assert len(env.events_for("carol")) == 0
        # Author always sees their own when include_own=True (default)
        assert len(env.events_for("alice")) == 1
        # ...but not when include_own=False
        assert len(env.events_for("alice", include_own=False)) == 0

    def test_since_tick_filter(self) -> None:
        env = Environment()
        env.advance_tick()
        env.emit("a", "t1")
        env.advance_tick()
        env.emit("a", "t2")
        recent = env.events_for("b", since_tick=2)
        assert len(recent) == 1
        assert recent[0].content == "t2"

    def test_kind_and_topic_filter(self) -> None:
        env = Environment()
        env.emit("a", "x", kind="say", topic="news")
        env.emit("a", "y", kind="act", topic="news")
        env.emit("a", "z", kind="say", topic="weather")
        out = env.events_for("b", kinds=("say",), topics=("news",))
        assert len(out) == 1
        assert out[0].content == "x"

    def test_state_and_callbacks(self) -> None:
        state_log: list[tuple[str, Any]] = []
        event_log: list[Event] = []
        env = Environment(
            initial_state={"weather": "sunny"},
            on_state_update=lambda k, v: state_log.append((k, v)),
            on_event=lambda e: event_log.append(e),
        )
        assert env.get("weather") == "sunny"
        env.set("weather", "rain")
        env.emit("a", "hi")
        assert state_log == [("weather", "rain")]
        assert len(event_log) == 1

    def test_seed_helper(self) -> None:
        env = Environment()
        env.seed("the news")
        ev = env.events[0]
        assert ev.kind == "seed"
        assert ev.agent_id == "__env__"
        assert ev.content == "the news"

    def test_last_n_caps_results(self) -> None:
        env = Environment()
        for i in range(5):
            env.emit("a", f"msg{i}")
        out = env.events_for("b", last_n=2)
        assert [e.content for e in out] == ["msg3", "msg4"]


# ----------------------------------------------------------------------
# Memory
# ----------------------------------------------------------------------


class TestInMemoryStore:
    def test_add_and_recent(self) -> None:
        m = InMemoryStore()
        m.add("alice", "first")
        m.add("alice", "second")
        m.add("alice", "third")
        recent = m.recent("alice", k=2)
        assert [r.content for r in recent] == ["second", "third"]

    def test_search_returns_keyword_matches(self) -> None:
        m = InMemoryStore()
        m.add("alice", "the weather is sunny")
        m.add("alice", "tax policy is changing")
        m.add("alice", "weather forecast next week")
        out = m.search("alice", "weather")
        assert len(out) == 2
        assert all("weather" in r.content for r in out)

    def test_search_empty_query_falls_back_to_recent(self) -> None:
        m = InMemoryStore()
        m.add("alice", "one")
        m.add("alice", "two")
        out = m.search("alice", "")
        assert [r.content for r in out] == ["one", "two"]

    def test_per_agent_isolation(self) -> None:
        m = InMemoryStore()
        m.add("alice", "alice secret")
        m.add("bob", "bob secret")
        assert len(m.recent("alice")) == 1
        assert len(m.recent("bob")) == 1
        assert m.recent("alice")[0].content == "alice secret"

    def test_clear(self) -> None:
        m = InMemoryStore()
        m.add("alice", "a")
        m.add("bob", "b")
        m.clear("alice")
        assert m.recent("alice") == []
        assert len(m.recent("bob")) == 1
        m.clear()
        assert m.recent("bob") == []


# ----------------------------------------------------------------------
# Schedulers
# ----------------------------------------------------------------------


class TestSchedulers:
    def test_all_scheduler_returns_every_agent(self) -> None:
        env = Environment()
        agents = [_make_swarm_agent(f"a{i}") for i in range(3)]
        out = AllScheduler().select(agents, env, 1)
        assert out == agents

    def test_sample_scheduler_returns_k_distinct(self) -> None:
        env = Environment()
        agents = [_make_swarm_agent(f"a{i}") for i in range(5)]
        rng = random.Random(42)
        out = SampleScheduler(k=3, rng=rng).select(agents, env, 1)
        assert len(out) == 3
        assert len({a.name for a in out}) == 3

    def test_sample_scheduler_clamps_to_population(self) -> None:
        env = Environment()
        agents = [_make_swarm_agent(f"a{i}") for i in range(2)]
        out = SampleScheduler(k=10).select(agents, env, 1)
        assert len(out) == 2

    def test_round_robin_rotates(self) -> None:
        env = Environment()
        agents = [_make_swarm_agent(f"a{i}") for i in range(3)]
        sched = RoundRobinScheduler()
        picks = [sched.select(agents, env, t)[0].name for t in range(1, 5)]
        assert picks == ["a0", "a1", "a2", "a0"]

    def test_round_robin_batch(self) -> None:
        env = Environment()
        agents = [_make_swarm_agent(f"a{i}") for i in range(4)]
        sched = RoundRobinScheduler(batch_size=2)
        out1 = [a.name for a in sched.select(agents, env, 1)]
        out2 = [a.name for a in sched.select(agents, env, 2)]
        assert out1 == ["a0", "a1"]
        assert out2 == ["a2", "a3"]

    def test_priority_scheduler_filters_and_orders(self) -> None:
        env = Environment()
        a = _make_swarm_agent("a", priority=0.0)
        b = _make_swarm_agent("b", priority=3.0)
        c = _make_swarm_agent("c", priority=2.0)
        sched = PriorityScheduler(lambda a, e, t: a.priority, threshold=1.0)
        out = sched.select([a, b, c], env, 1)
        assert [x.name for x in out] == ["b", "c"]

    def test_priority_scheduler_max_active(self) -> None:
        env = Environment()
        agents = [_make_swarm_agent(f"a{i}", priority=float(i + 1)) for i in range(4)]
        sched = PriorityScheduler(lambda a, e, t: a.priority, max_active=2)
        out = sched.select(agents, env, 1)
        assert [x.name for x in out] == ["a3", "a2"]

    def test_callable_scheduler(self) -> None:
        env = Environment()
        agents = [_make_swarm_agent(f"a{i}") for i in range(3)]
        sched = CallableScheduler(lambda ags, e, t: [a for a in ags if a.name != "a1"])
        out = sched.select(agents, env, 1)
        assert [x.name for x in out] == ["a0", "a2"]


# ----------------------------------------------------------------------
# SwarmAgent
# ----------------------------------------------------------------------


class TestSwarmAgent:
    def test_default_observe_includes_role_and_events(self) -> None:
        env = Environment()
        env.seed("breaking news")
        sa = _make_swarm_agent("alice", role="reporter")
        obs = sa.build_observation(env, tick=1)
        assert "alice" in obs
        assert "reporter" in obs
        assert "breaking news" in obs

    def test_default_observe_includes_memory(self) -> None:
        env = Environment()
        memory = InMemoryStore()
        memory.add("alice", "I covered taxes last year")
        sa = _make_swarm_agent("alice", memory=memory)
        obs = sa.build_observation(env, tick=1)
        assert "taxes last year" in obs

    def test_default_observe_filters_by_kind(self) -> None:
        env = Environment()
        env.emit("bob", "loud", kind="shout")
        env.emit("bob", "quiet", kind="say")
        sa = _make_swarm_agent("alice", observe_kinds=("say",))
        obs = sa.build_observation(env, tick=1)
        assert "quiet" in obs
        assert "loud" not in obs

    def test_run_step_emits_event_and_records_tick(self) -> None:
        env = Environment()
        sa = _make_swarm_agent("alice", outputs=["I disagree"])
        sa.run_step(env, tick=1, prompt="React.")
        assert sa._last_acted_tick == 1
        assert len(env.events) == 1
        assert env.events[0].content == "I disagree"
        assert env.events[0].agent_id == "alice"

    def test_run_step_writes_memory(self) -> None:
        env = Environment()
        memory = InMemoryStore()
        sa = _make_swarm_agent("alice", outputs=["a fresh take"], memory=memory)
        sa.run_step(env, tick=1, prompt="React.")
        recent = memory.recent("alice")
        assert len(recent) == 1
        assert recent[0].content == "a fresh take"
        assert recent[0].tick == 1

    def test_custom_post_act_overrides_default(self) -> None:
        env = Environment()
        called: list[str] = []

        def custom_post_act(agent, result, env, tick):
            called.append(result.output_text)
            env.emit(agent.name, f"transformed: {result.output_text}", kind="custom")

        sa = _make_swarm_agent("alice", outputs=["raw"], post_act=custom_post_act)
        sa.run_step(env, tick=1, prompt="React.")
        assert called == ["raw"]
        assert env.events[0].kind == "custom"
        assert env.events[0].content == "transformed: raw"


# ----------------------------------------------------------------------
# Swarm
# ----------------------------------------------------------------------


class TestSwarm:
    def test_runs_max_ticks(self) -> None:
        agents = [
            _make_swarm_agent("a", outputs=["a1", "a2", "a3"]),
            _make_swarm_agent("b", outputs=["b1", "b2", "b3"]),
        ]
        swarm = Swarm(agents=agents, max_ticks=3)
        result = swarm.run(seed="go")
        assert result.ticks_run == 3
        assert result.success
        # 2 agents x 3 ticks
        assert len(result.timeline) == 6
        # seed + 6 say events
        assert len(result.events) == 7

    def test_aggregate_usage(self) -> None:
        agents = [_make_swarm_agent("a"), _make_swarm_agent("b")]
        swarm = Swarm(agents=agents, max_ticks=2)
        result = swarm.run()
        assert result.aggregate_usage["total_tokens"] == 15 * 4
        assert result.aggregate_usage["total_cost"] == pytest.approx(0.001 * 4)

    def test_duplicate_names_rejected(self) -> None:
        agents = [_make_swarm_agent("a"), _make_swarm_agent("a")]
        with pytest.raises(ValueError, match="Duplicate"):
            Swarm(agents=agents)

    def test_empty_agents_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            Swarm(agents=[])

    def test_stop_when(self) -> None:
        agents = [_make_swarm_agent("a")]
        swarm = Swarm(agents=agents, max_ticks=10, stop_when=lambda env, t: t > 2)
        result = swarm.run()
        assert result.ticks_run == 2
        assert "stop_when" in (result.stopped_reason or "")

    def test_stop_method(self) -> None:
        agents = [_make_swarm_agent("a"), _make_swarm_agent("b")]
        swarm: Swarm | None = None

        def on_tick_complete(tick: int) -> None:
            assert swarm is not None
            if tick == 2:
                swarm.stop("external halt")

        swarm = Swarm(
            agents=agents,
            max_ticks=10,
            callbacks=SwarmCallbacks(on_tick_complete=on_tick_complete),
        )
        result = swarm.run()
        assert result.ticks_run == 2
        assert result.stopped_reason == "external halt"

    def test_fail_fast_on_error(self) -> None:
        good = _make_swarm_agent("good")
        bad = MagicMock()
        bad.run.side_effect = RuntimeError("boom")
        bad_sa = SwarmAgent(agent=bad, name="bad")
        swarm = Swarm(agents=[good, bad_sa], max_ticks=3)
        result = swarm.run()
        assert not result.success
        assert result.errors
        assert result.errors[0][0] == "bad"
        # fail_fast: ends after first errored tick
        assert result.ticks_run == 1

    def test_continue_on_error(self) -> None:
        good = _make_swarm_agent("good")
        bad = MagicMock()
        bad.run.side_effect = RuntimeError("boom")
        bad_sa = SwarmAgent(agent=bad, name="bad")
        swarm = Swarm(
            agents=[good, bad_sa],
            max_ticks=2,
            error_policy=ErrorPolicy.continue_on_error,
        )
        result = swarm.run()
        assert not result.success
        assert len(result.errors) == 2  # bad errors both ticks
        assert result.ticks_run == 2

    def test_raise_on_error(self) -> None:
        bad = MagicMock()
        bad.run.side_effect = RuntimeError("boom")
        swarm = Swarm(
            agents=[SwarmAgent(agent=bad, name="bad")],
            max_ticks=3,
            error_policy=ErrorPolicy.raise_on_error,
        )
        with pytest.raises(RuntimeError, match="boom"):
            swarm.run()

    def test_round_robin_scheduling(self) -> None:
        agents = [
            _make_swarm_agent("a", outputs=["a"] * 10),
            _make_swarm_agent("b", outputs=["b"] * 10),
            _make_swarm_agent("c", outputs=["c"] * 10),
        ]
        swarm = Swarm(agents=agents, scheduler=RoundRobinScheduler(), max_ticks=4)
        result = swarm.run()
        actors = [s.agent_id for s in result.timeline]
        assert actors == ["a", "b", "c", "a"]

    def test_default_memory_applied_to_agents(self) -> None:
        memory = InMemoryStore()
        agents = [_make_swarm_agent("a"), _make_swarm_agent("b")]
        swarm = Swarm(agents=agents, memory=memory, max_ticks=1)
        swarm.run()
        assert memory.recent("a")
        assert memory.recent("b")

    def test_per_agent_memory_not_overridden(self) -> None:
        global_mem = InMemoryStore()
        local_mem = InMemoryStore()
        agents = [
            _make_swarm_agent("a", memory=local_mem),
            _make_swarm_agent("b"),
        ]
        Swarm(agents=agents, memory=global_mem, max_ticks=1).run()
        # a kept its own memory store
        assert local_mem.recent("a")
        assert not global_mem.recent("a")
        # b inherited the default
        assert global_mem.recent("b")

    def test_callbacks_fire(self) -> None:
        ticks_started: list[int] = []
        agents_completed: list[str] = []
        events_seen: list[Event] = []
        swarm = Swarm(
            agents=[_make_swarm_agent("a")],
            max_ticks=2,
            callbacks=SwarmCallbacks(
                on_tick_start=lambda t, names: ticks_started.append(t),
                on_agent_complete=lambda name, r, t: agents_completed.append(name),
                on_event=lambda e: events_seen.append(e),
            ),
        )
        swarm.run(seed="hi")
        assert ticks_started == [1, 2]
        assert agents_completed == ["a", "a"]
        # seed + 2 say events
        assert len(events_seen) == 3

    def test_budget_cap_stops_run(self) -> None:
        agents = [_make_swarm_agent("a"), _make_swarm_agent("b")]
        # Each agent run costs 0.001; cap at 0.0025 → run for 2 agents on tick 1
        # then bail at the start of tick 2.
        swarm = Swarm(agents=agents, max_ticks=5, max_total_cost=0.0025)
        result = swarm.run()
        assert result.stopped_reason is not None
        assert "budget" in result.stopped_reason

    def test_action_prompt_callable(self) -> None:
        captured: list[str] = []

        def custom_observe(agent, env, tick):
            captured.append(agent._pending_prompt)
            return agent._pending_prompt

        agent = _make_swarm_agent("a", observe=custom_observe)
        swarm = Swarm(
            agents=[agent],
            action_prompt=lambda env, tick: f"tick {tick} go",
            max_ticks=2,
        )
        swarm.run()
        assert captured == ["tick 1 go", "tick 2 go"]

    def test_action_prompt_template(self) -> None:
        captured: list[str] = []

        def custom_observe(agent, env, tick):
            captured.append(agent._pending_prompt)
            return agent._pending_prompt

        agent = _make_swarm_agent("a", observe=custom_observe)
        swarm = Swarm(agents=[agent], action_prompt="go tick={tick}", max_ticks=2)
        swarm.run()
        assert captured == ["go tick=1", "go tick=2"]

    def test_result_export_serialises(self) -> None:
        swarm = Swarm(agents=[_make_swarm_agent("a")], max_ticks=1)
        result = swarm.run(seed="hi")
        export = result.export()
        assert export["ticks_run"] == 1
        assert export["success"] is True
        assert export["events"][0]["kind"] == "seed"
        assert export["timeline"][0]["agent_id"] == "a"


# ----------------------------------------------------------------------
# End-to-end interaction scenario
# ----------------------------------------------------------------------


def test_two_agent_dialogue_uses_each_others_events() -> None:
    """The second agent's observation includes the first agent's emission."""
    seen_by_bob: list[str] = []

    def alice_observe(agent, env, tick):
        return agent._pending_prompt  # noop

    def bob_observe(agent, env, tick):
        events = env.events_for(agent.name)
        seen_by_bob.append(" | ".join(e.content for e in events))
        return agent._pending_prompt

    alice = SwarmAgent(
        agent=_mock_agent("alice", ["alice says hi"]),
        name="alice",
        observe=alice_observe,
    )
    bob = SwarmAgent(
        agent=_mock_agent("bob", ["bob says yo"]),
        name="bob",
        observe=bob_observe,
    )

    swarm = Swarm(agents=[alice, bob], max_ticks=1)
    result = swarm.run(seed="hello world")

    # Bob's observation step happens after Alice's emission this tick,
    # so it should include the seed + alice's say.
    assert seen_by_bob == ["hello world | alice says hi"]
    assert result.success
    assert len(result.events) == 3  # seed + 2 says
