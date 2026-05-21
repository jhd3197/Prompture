"""Tests for pick_best_coding_agent and the new capability flags."""

from __future__ import annotations

from unittest.mock import patch

from prompture import pick_best_coding_agent
from prompture.infra.coding_agent_specs import CODING_AGENT_SPECS


def _info(agent_id: str, *, healthy: bool = True, available: bool = True):
    return type(
        "Info",
        (),
        {
            "id": agent_id,
            "name": agent_id,
            "available": available,
            "healthy": healthy,
            "binary": agent_id,
            "custom_path": False,
            "source": "PATH",
            "verified": True,
            "error": None,
        },
    )()


# ---------------------------------------------------------------------------
# Capability flags on the specs themselves
# ---------------------------------------------------------------------------


def test_claude_advertises_session_resume_and_questions():
    spec = CODING_AGENT_SPECS["claude"]
    assert spec.supports_session_resume is True
    assert spec.supports_questions is True
    assert spec.supports_tool_use is True
    assert spec.supports_structured_output is True


def test_codex_advertises_session_resume_and_questions():
    spec = CODING_AGENT_SPECS["codex"]
    assert spec.supports_session_resume is True
    assert spec.supports_questions is True
    assert spec.supports_structured_output is True


def test_aider_lacks_advanced_capabilities():
    spec = CODING_AGENT_SPECS["aider"]
    # No JSON event parser → no structured output, questions, or resume
    # via Prompture's runner today.
    assert spec.supports_structured_output is False
    assert spec.supports_questions is False
    assert spec.supports_session_resume is False
    # All registered CLIs still execute tools.
    assert spec.supports_tool_use is True


# ---------------------------------------------------------------------------
# pick_best_coding_agent
# ---------------------------------------------------------------------------


def test_picks_preferred_when_healthy():
    with patch(
        "prompture.infra.discovery.get_available_coding_agents",
        return_value=[_info("codex"), _info("claude")],
    ):
        chosen = pick_best_coding_agent(prefer=["claude", "codex"], verify=False)
    assert chosen is not None
    assert chosen.id == "claude"


def test_falls_through_when_preferred_unhealthy():
    with patch(
        "prompture.infra.discovery.get_available_coding_agents",
        return_value=[_info("codex"), _info("claude", healthy=False)],
    ):
        # Note: unhealthy entries are dropped by get_available_coding_agents
        # when include_unavailable=False, so simulate that here.
        pass

    healthy_only = [_info("codex")]
    with patch(
        "prompture.infra.discovery.get_available_coding_agents",
        return_value=healthy_only,
    ):
        chosen = pick_best_coding_agent(prefer=["claude", "codex"], verify=False)
    assert chosen is not None
    assert chosen.id == "codex"


def test_require_session_resume_filters_out_aider():
    with patch(
        "prompture.infra.discovery.get_available_coding_agents",
        return_value=[_info("aider"), _info("claude")],
    ):
        chosen = pick_best_coding_agent(
            require_session_resume=True, verify=False
        )
    assert chosen is not None
    assert chosen.id == "claude"


def test_returns_none_when_no_requirements_satisfied():
    with patch(
        "prompture.infra.discovery.get_available_coding_agents",
        return_value=[_info("aider")],
    ):
        chosen = pick_best_coding_agent(
            require_session_resume=True, verify=False
        )
    assert chosen is None


def test_returns_none_when_nothing_installed():
    with patch(
        "prompture.infra.discovery.get_available_coding_agents",
        return_value=[],
    ):
        chosen = pick_best_coding_agent(verify=False)
    assert chosen is None


def test_require_structured_output_filters_aider():
    with patch(
        "prompture.infra.discovery.get_available_coding_agents",
        return_value=[_info("aider"), _info("codex")],
    ):
        chosen = pick_best_coding_agent(
            require_structured_output=True, verify=False
        )
    assert chosen is not None
    assert chosen.id == "codex"
