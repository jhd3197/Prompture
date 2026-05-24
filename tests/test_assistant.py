"""Unit tests for the Assistant abstraction (no live LLM calls)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from prompture import (
    Assistant,
    AssistantResult,
    Persona,
    SkillInfo,
    clear_assistant_registry,
    get_assistant,
    get_assistant_names,
    register_assistant,
    unregister_assistant,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_assistant_registry()
    yield
    clear_assistant_registry()


@pytest.fixture
def persona() -> Persona:
    return Persona(
        name="dev",
        system_prompt="You are a {{role}}. Workspace: {{workspace}}.",
        variables={"role": "developer", "workspace": "/tmp"},
    )


@pytest.fixture
def skill() -> SkillInfo:
    return SkillInfo(
        name="html-builder",
        description="Builds semantic HTML5 pages.",
        instructions="Always use semantic tags. Prefer <section> over <div>.",
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_requires_exactly_one_backend(persona):
    with pytest.raises(ValueError, match="exactly one"):
        Assistant(name="bad", persona=persona)

    with pytest.raises(ValueError, match="exactly one"):
        Assistant(name="bad", persona=persona, model="openai/gpt-4o", coding_agent="claude")


def test_planning_requires_llm_backend(persona):
    with pytest.raises(ValueError, match="enable_planning"):
        Assistant(
            name="bad",
            persona=persona,
            coding_agent="claude",
            enable_planning=True,
        )


def test_llm_assistant_constructs(persona):
    a = Assistant(name="dev", persona=persona, model="openai/gpt-4o")
    assert a.model == "openai/gpt-4o"
    assert a.coding_agent is None


def test_coding_agent_assistant_constructs(persona):
    a = Assistant(name="dev-cli", persona=persona, coding_agent="claude")
    assert a.coding_agent == "claude"
    assert a.model is None


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_with_vars_merges(persona):
    a = Assistant(name="dev", persona=persona, model="openai/gpt-4o")
    b = a.with_vars(role="senior developer")
    assert b.variables["role"] == "senior developer"
    assert a.variables == {}  # original unchanged


def test_with_skills_appends(persona, skill):
    a = Assistant(name="dev", persona=persona, model="openai/gpt-4o")
    b = a.with_skills(skill)
    assert len(b.skills) == 1
    assert b.skills[0].name == "html-builder"
    assert len(a.skills) == 0


def test_composed_persona_includes_skill_instructions(persona, skill):
    a = Assistant(name="dev", persona=persona, skills=[skill], model="openai/gpt-4o")
    composed = a._composed_persona({})
    rendered = composed.render()
    assert "Always use semantic tags" in rendered
    assert "Skill: html-builder" in rendered


def test_composed_persona_substitutes_vars(persona):
    a = Assistant(
        name="dev",
        persona=persona,
        model="openai/gpt-4o",
        variables={"role": "architect"},
    )
    composed = a._composed_persona({"workspace": "/srv/site"})
    rendered = composed.render()
    assert "architect" in rendered
    assert "/srv/site" in rendered


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_roundtrip(persona):
    a = Assistant(name="web-dev", persona=persona, model="openai/gpt-4o")
    register_assistant(a)
    assert get_assistant("web-dev") is a
    assert "web-dev" in get_assistant_names()
    assert unregister_assistant("web-dev") is True
    assert get_assistant("web-dev") is None


def test_register_method_returns_self(persona):
    a = Assistant(name="web-dev", persona=persona, model="openai/gpt-4o")
    assert a.register() is a
    assert get_assistant("web-dev") is a


def test_from_registry_classmethod(persona):
    a = Assistant(name="web-dev", persona=persona, model="openai/gpt-4o")
    register_assistant(a)
    assert Assistant.from_registry("web-dev") is a


def test_from_registry_unknown_raises():
    with pytest.raises(KeyError, match="No Assistant registered"):
        Assistant.from_registry("nope")


# ---------------------------------------------------------------------------
# build_async_agent
# ---------------------------------------------------------------------------


def test_build_async_agent_returns_async_agent(persona):
    from prompture import AsyncAgent

    a = Assistant(
        name="dev",
        persona=persona,
        model="openai/gpt-4o",
        output_key="page_output",
    )
    agent = a.build_async_agent(role="senior")
    assert isinstance(agent, AsyncAgent)
    assert agent.name == "dev"
    assert agent.output_key == "page_output"


def test_build_async_agent_returns_deep_when_planning_enabled(persona):
    from prompture import AsyncDeepAgent

    a = Assistant(
        name="dev",
        persona=persona,
        model="openai/gpt-4o",
        enable_planning=True,
        deep_agent_options={"enable_summarization": False},
    )
    agent = a.build_async_agent()
    assert isinstance(agent, AsyncDeepAgent)


def test_build_async_agent_rejects_coding_agent_backend(persona):
    a = Assistant(name="cli", persona=persona, coding_agent="claude")
    with pytest.raises(ValueError, match="only available for the LLM backend"):
        a.build_async_agent()


def test_agent_options_forward_to_async_agent_kwargs(persona):
    """agent_options must spread into AsyncAgent.__init__ so callers can
    set budget controls (max_cost, budget_policy, fallback_models, ...)
    that the Assistant doesn't expose as first-class fields."""
    from prompture import BudgetPolicy

    a = Assistant(
        name="dev",
        persona=persona,
        model="openai/gpt-4o",
        agent_options={
            "max_iterations": 5,
            "max_cost": 0.50,
            "budget_policy": BudgetPolicy.warn_and_continue,
            "fallback_models": ["openai/gpt-4o-mini"],
        },
    )
    with patch("prompture.agents.async_agent.AsyncAgent.__init__", return_value=None) as init:
        a.build_async_agent()
    assert init.called
    kwargs = init.call_args.kwargs
    assert kwargs["max_iterations"] == 5
    assert kwargs["max_cost"] == 0.50
    assert kwargs["budget_policy"] is BudgetPolicy.warn_and_continue
    assert kwargs["fallback_models"] == ["openai/gpt-4o-mini"]


def test_agent_options_ignored_when_planning_enabled(persona):
    """enable_planning=True routes through AsyncDeepAgent, so agent_options
    must NOT leak through — deep_agent_options is the right escape hatch."""
    from prompture import AsyncDeepAgent

    a = Assistant(
        name="dev",
        persona=persona,
        model="openai/gpt-4o",
        enable_planning=True,
        agent_options={"max_iterations": 5},
    )
    with patch("prompture.agents.async_deep_agent.AsyncDeepAgent.__init__", return_value=None) as init:
        agent = a.build_async_agent()
    assert isinstance(agent, AsyncDeepAgent)
    assert "max_iterations" not in init.call_args.kwargs


def test_agent_options_empty_by_default(persona):
    """Default Assistant carries an empty agent_options dict; nothing is
    forwarded so callers who don't opt in see no behavior change."""
    a = Assistant(name="dev", persona=persona, model="openai/gpt-4o")
    assert a.agent_options == {}
    with patch("prompture.agents.async_agent.AsyncAgent.__init__", return_value=None) as init:
        a.build_async_agent()
    kwargs = init.call_args.kwargs
    for budget_kw in ("max_iterations", "max_cost", "budget_policy", "fallback_models"):
        assert budget_kw not in kwargs, f"{budget_kw} should not appear without agent_options"


def test_build_async_agent_applies_skills_to_system_prompt(persona, skill):
    a = Assistant(
        name="dev",
        persona=persona,
        skills=[skill],
        model="openai/gpt-4o",
    )
    agent = a.build_async_agent()
    rendered = agent._system_prompt.render()
    assert "Always use semantic tags" in rendered


# ---------------------------------------------------------------------------
# Coding-agent backend (mocked)
# ---------------------------------------------------------------------------


def test_coding_agent_run_dispatches_to_arun_coding_agent(persona):
    a = Assistant(name="cli-dev", persona=persona, coding_agent="claude")
    fake = type(
        "FakeRes",
        (),
        {
            "output": "<!DOCTYPE html><html></html>",
            "cost_usd": 0.01,
            "input_tokens": 100,
            "output_tokens": 50,
            "session_id": "sess-1",
            "ok": True,
        },
    )()
    with patch(
        "prompture.infra.coding_agents.arun_coding_agent",
        new=AsyncMock(return_value=fake),
    ) as m:
        result = asyncio.run(a.arun("Build /about.html", workspace="/tmp"))
    assert isinstance(result, AssistantResult)
    assert result.backend == "coding_agent"
    assert result.output.startswith("<!DOCTYPE")
    assert result.cost_usd == 0.01
    assert result.session_id == "sess-1"
    # Persona text must have been prepended to the task.
    sent_task = m.call_args.args[1]
    assert "developer" in sent_task  # rendered persona variable
    assert "Build /about.html" in sent_task


def test_coding_agent_auto_resolves_first_healthy(persona):
    a = Assistant(name="cli-dev", persona=persona, coding_agent="auto")
    fake_specs = [
        type("S", (), {"id": "claude", "available": True, "healthy": True})(),
        type("S", (), {"id": "codex", "available": True, "healthy": True})(),
    ]
    fake_result = type(
        "R",
        (),
        {
            "output": "x",
            "cost_usd": None,
            "input_tokens": None,
            "output_tokens": None,
            "session_id": None,
        },
    )()
    with (
        patch(
            "prompture.infra.discovery.get_available_coding_agents",
            return_value=fake_specs,
        ),
        patch(
            "prompture.infra.coding_agents.arun_coding_agent",
            new=AsyncMock(return_value=fake_result),
        ) as runner,
    ):
        asyncio.run(a.arun("hi"))
    assert runner.call_args.args[0] == "claude"


def test_coding_agent_auto_raises_when_none_healthy(persona):
    a = Assistant(name="cli-dev", persona=persona, coding_agent="auto")
    with patch(
        "prompture.infra.discovery.get_available_coding_agents",
        return_value=[],
    ):
        with pytest.raises(RuntimeError, match="no healthy coding-agent"):
            asyncio.run(a.arun("hi"))
