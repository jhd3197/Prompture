"""Tests for Tukuy SecurityContext integration.

Covers Agent/AsyncAgent security_context and auto_approve_safe_only params,
PythonSandbox.to_security_context(), apply_security_context(), ToolDefinition
security_metadata, and re-exports.

All tests are unit tests -- no LLM calls required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from tukuy import Skill, skill
from tukuy.safety import SecurityContext, reset_security_context, set_security_context

from prompture import ToolDefinition, ToolRegistry, apply_security_context
from prompture.agents.agent import Agent
from prompture.agents.async_agent import AsyncAgent
from prompture.drivers.base import Driver
from prompture.integrations.tukuy_bridge import skill_to_tool_definition, skills_to_registry
from prompture.sandbox.sandbox import PythonSandbox

# ------------------------------------------------------------------
# Helpers: mock drivers & tukuy skills
# ------------------------------------------------------------------


class MockDriver(Driver):
    """Minimal mock driver for agent construction and execution."""

    supports_messages = True
    supports_tool_use = False

    def __init__(self):
        self.model = "mock-model"

    def generate(self, prompt, options):
        return self._response()

    def generate_messages(self, messages, options):
        return self._response()

    def _response(self):
        return {
            "text": "Hello",
            "meta": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.001,
                "raw_response": {},
            },
        }


@skill(name="safe_skill", description="A safe skill")
def safe_skill(x: int) -> int:
    return x * 2


@skill(name="dangerous_skill", description="Has side effects", side_effects=True)
def dangerous_skill(x: int) -> int:
    return x * 3


@skill(name="network_skill", description="Needs network", requires_network=True)
def network_skill(url: str) -> str:
    return f"fetched {url}"


# ------------------------------------------------------------------
# TestAgentSecurityContextParameter
# ------------------------------------------------------------------


class TestAgentSecurityContextParameter:
    """Test that security_context param is accepted and stored."""

    def test_default_is_none(self):
        agent = Agent(driver=MockDriver())
        assert agent._security_context is None

    def test_accepts_security_context(self):
        ctx = SecurityContext()
        agent = Agent(driver=MockDriver(), security_context=ctx)
        assert agent._security_context is ctx

    def test_auto_approve_default_false(self):
        agent = Agent(driver=MockDriver())
        assert agent._auto_approve_safe_only is False

    def test_auto_approve_set_true(self):
        agent = Agent(driver=MockDriver(), auto_approve_safe_only=True)
        assert agent._auto_approve_safe_only is True

    def test_async_agent_default_is_none(self):
        agent = AsyncAgent(driver=MockDriver())
        assert agent._security_context is None

    def test_async_agent_accepts_security_context(self):
        ctx = SecurityContext()
        agent = AsyncAgent(driver=MockDriver(), security_context=ctx)
        assert agent._security_context is ctx

    def test_async_agent_auto_approve_default_false(self):
        agent = AsyncAgent(driver=MockDriver())
        assert agent._auto_approve_safe_only is False

    def test_async_agent_auto_approve_set_true(self):
        agent = AsyncAgent(driver=MockDriver(), auto_approve_safe_only=True)
        assert agent._auto_approve_safe_only is True


# ------------------------------------------------------------------
# TestSecurityContextLifecycle
# ------------------------------------------------------------------


class TestSecurityContextLifecycle:
    """Test that security context is activated and deactivated during agent runs."""

    def test_context_set_during_execution(self):
        """Verify set/reset are called with the right context during run."""
        ctx = SecurityContext()
        agent = Agent(driver=MockDriver(), security_context=ctx)

        # Track calls by wrapping the real functions
        set_calls: list[Any] = []
        reset_calls: list[Any] = []
        original_set = set_security_context
        original_reset = reset_security_context

        def tracking_set(sc):
            token = original_set(sc)
            set_calls.append((sc, token))
            return token

        def tracking_reset(token):
            reset_calls.append(token)
            return original_reset(token)

        with (
            patch("tukuy.safety.set_security_context", side_effect=tracking_set),
            patch("tukuy.safety.reset_security_context", side_effect=tracking_reset),
        ):
            agent.run("Hello")

        assert len(set_calls) == 1
        assert set_calls[0][0] is ctx
        assert len(reset_calls) == 1

    def test_context_not_set_when_none(self):
        """No calls to set/reset when security_context is None."""
        agent = Agent(driver=MockDriver())
        # When security_context is None, the code never imports tukuy.safety,
        # so there's nothing to patch. We verify by checking _security_context.
        assert agent._security_context is None
        # Just verify agent.run works without error
        result = agent.run("Hello")
        assert result.output_text == "Hello"

    def test_context_reset_on_error(self):
        """reset_security_context is called even if conv.ask() raises."""

        class ErrorDriver(MockDriver):
            def generate_messages(self, messages, options):
                raise RuntimeError("boom")

        ctx = SecurityContext()
        agent = Agent(driver=ErrorDriver(), security_context=ctx)

        reset_calls: list[Any] = []
        original_set = set_security_context
        original_reset = reset_security_context

        def tracking_set(sc):
            return original_set(sc)

        def tracking_reset(token):
            reset_calls.append(token)
            return original_reset(token)

        with (
            patch("tukuy.safety.set_security_context", side_effect=tracking_set),
            patch("tukuy.safety.reset_security_context", side_effect=tracking_reset),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                agent.run("Hello")

        assert len(reset_calls) == 1


# ------------------------------------------------------------------
# TestPythonSandboxBridge
# ------------------------------------------------------------------


class TestPythonSandboxBridge:
    """Test PythonSandbox.to_security_context() conversion."""

    def test_basic_conversion(self):
        sandbox = PythonSandbox(
            allowed_read_paths=["/tmp/read"],
            allowed_write_paths=["/tmp/write"],
            working_directory="/tmp/work",
        )
        sc = sandbox.to_security_context()
        assert isinstance(sc, SecurityContext)

    def test_read_paths_converted(self):
        sandbox = PythonSandbox(
            allowed_read_paths=["/tmp/a", "/tmp/b"],
        )
        sc = sandbox.to_security_context()
        read_paths = sc.allowed_read_paths
        # Paths are converted to strings
        assert len(read_paths) == 2
        path_strs = {str(p) for p in read_paths}
        assert any("a" in p for p in path_strs)
        assert any("b" in p for p in path_strs)

    def test_write_paths_converted(self):
        sandbox = PythonSandbox(
            allowed_write_paths=["/tmp/out"],
        )
        sc = sandbox.to_security_context()
        write_paths = sc.allowed_write_paths
        assert len(write_paths) == 1

    def test_working_directory(self):
        sandbox = PythonSandbox(
            working_directory="/tmp/work",
        )
        sc = sandbox.to_security_context()
        assert sc.working_directory is not None
        assert "work" in str(sc.working_directory)

    def test_no_working_directory(self):
        sandbox = PythonSandbox()
        sc = sandbox.to_security_context()
        assert sc.working_directory is None

    def test_empty_paths(self):
        sandbox = PythonSandbox()
        sc = sandbox.to_security_context()
        assert len(sc.allowed_read_paths) == 0
        assert len(sc.allowed_write_paths) == 0


# ------------------------------------------------------------------
# TestApplySecurityContext
# ------------------------------------------------------------------


class TestApplySecurityContext:
    """Test apply_security_context() wraps tukuy tools and passes non-tukuy through."""

    def test_wraps_tukuy_tools(self):
        reg = skills_to_registry([safe_skill])
        ctx = SecurityContext()
        scoped = apply_security_context(reg, ctx)
        assert len(scoped) == 1
        # Tool should still work
        result = scoped.execute("safe_skill", {"x": 5})
        assert result == 10

    def test_passes_non_tukuy_through(self):
        reg = ToolRegistry()
        reg.register(lambda x: x + 1, name="inc", description="Increment")
        ctx = SecurityContext()
        scoped = apply_security_context(reg, ctx)
        result = scoped.execute("inc", {"x": 5})
        assert result == 6

    def test_preserves_skill_attribute(self):
        reg = skills_to_registry([safe_skill])
        ctx = SecurityContext()
        scoped = apply_security_context(reg, ctx)
        td = scoped.get("safe_skill")
        assert td is not None
        assert hasattr(td.function, "__skill__")
        assert isinstance(td.function.__skill__, Skill)

    def test_returns_new_registry(self):
        reg = skills_to_registry([safe_skill])
        ctx = SecurityContext()
        scoped = apply_security_context(reg, ctx)
        assert scoped is not reg

    def test_mixed_registry(self):
        reg = skills_to_registry([safe_skill])
        reg.register(lambda x: x - 1, name="dec", description="Decrement")
        ctx = SecurityContext()
        scoped = apply_security_context(reg, ctx)
        assert len(scoped) == 2
        assert scoped.execute("safe_skill", {"x": 3}) == 6
        assert scoped.execute("dec", {"x": 3}) == 2


# ------------------------------------------------------------------
# TestAutoApproveSafeOnly
# ------------------------------------------------------------------


class TestAutoApproveSafeOnly:
    """Test auto_approve_safe_only gating on tool execution."""

    def test_safe_tool_passes(self):
        agent = Agent(
            driver=MockDriver(),
            tools=[safe_skill],
            auto_approve_safe_only=True,
        )
        # Build context to wrap tools
        from prompture.infra.session import UsageSession

        session = UsageSession()
        ctx = agent._build_run_context("test", None, session, [], 0)
        wrapped = agent._wrap_tools_with_context(ctx, session)

        # safe_skill has no side_effects/requires_network, should work
        result = wrapped.execute("safe_skill", {"x": 5})
        assert result == 10

    def test_dangerous_tool_raises_approval(self):
        agent = Agent(
            driver=MockDriver(),
            tools=[dangerous_skill],
            auto_approve_safe_only=True,
        )
        from prompture.infra.session import UsageSession

        session = UsageSession()
        ctx = agent._build_run_context("test", None, session, [], 0)
        wrapped = agent._wrap_tools_with_context(ctx, session)

        # dangerous_skill has side_effects=True, should raise ApprovalRequired
        # which gets caught by the wrapper and returned as error string
        result = wrapped.execute("dangerous_skill", {"x": 5})
        assert isinstance(result, str)
        assert "approval" in result.lower()

    def test_network_tool_raises_approval(self):
        agent = Agent(
            driver=MockDriver(),
            tools=[network_skill],
            auto_approve_safe_only=True,
        )
        from prompture.infra.session import UsageSession

        session = UsageSession()
        ctx = agent._build_run_context("test", None, session, [], 0)
        wrapped = agent._wrap_tools_with_context(ctx, session)

        result = wrapped.execute("network_skill", {"url": "http://test.com"})
        assert isinstance(result, str)
        assert "approval" in result.lower()

    def test_disabled_by_default(self):
        agent = Agent(
            driver=MockDriver(),
            tools=[dangerous_skill],
        )
        from prompture.infra.session import UsageSession

        session = UsageSession()
        ctx = agent._build_run_context("test", None, session, [], 0)
        wrapped = agent._wrap_tools_with_context(ctx, session)

        # auto_approve_safe_only is False, so dangerous tool should run
        result = wrapped.execute("dangerous_skill", {"x": 5})
        assert result == 15

    def test_native_tools_not_affected(self):
        def my_tool(x: int) -> int:
            """Multiply by 4."""
            return x * 4

        agent = Agent(
            driver=MockDriver(),
            tools=[my_tool],
            auto_approve_safe_only=True,
        )
        from prompture.infra.session import UsageSession

        session = UsageSession()
        ctx = agent._build_run_context("test", None, session, [], 0)
        wrapped = agent._wrap_tools_with_context(ctx, session)

        # Native tool has no __skill__, should pass
        result = wrapped.execute("my_tool", {"x": 5})
        assert result == 20


# ------------------------------------------------------------------
# TestToolDefinitionSecurityMetadata
# ------------------------------------------------------------------


class TestToolDefinitionSecurityMetadata:
    """Test ToolDefinition.security_metadata property."""

    def test_tukuy_tool_has_metadata(self):
        td = skill_to_tool_definition(safe_skill)
        meta = td.security_metadata
        assert meta is not None
        assert meta["name"] == "safe_skill"
        assert meta["is_tukuy_skill"] is True
        assert meta["side_effects"] is False
        assert meta["requires_network"] is False

    def test_side_effects_skill(self):
        td = skill_to_tool_definition(dangerous_skill)
        meta = td.security_metadata
        assert meta is not None
        assert meta["side_effects"] is True

    def test_network_skill(self):
        td = skill_to_tool_definition(network_skill)
        meta = td.security_metadata
        assert meta is not None
        assert meta["requires_network"] is True

    def test_native_tool_returns_none(self):
        td = ToolDefinition(
            name="add",
            description="Add two numbers",
            parameters={"type": "object", "properties": {}},
            function=lambda: None,
        )
        assert td.security_metadata is None


# ------------------------------------------------------------------
# TestReExports
# ------------------------------------------------------------------


class TestReExports:
    """Test that apply_security_context and TukuySecurityContext are importable."""

    def test_apply_security_context_importable(self):
        from prompture import apply_security_context

        assert callable(apply_security_context)

    def test_tukuy_security_context_importable(self):
        from prompture import TukuySecurityContext

        assert TukuySecurityContext is SecurityContext
