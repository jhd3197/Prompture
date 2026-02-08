"""Tests for the tukuy bridge module (prompture/tukuy_bridge.py).

All tests are unit tests — no LLM calls required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from tukuy import Chain, SafetyPolicy, Skill, SkillContext, SkillResult, skill

from prompture import (
    ToolDefinition,
    ToolRegistry,
    TukuyChainStep,
    apply_safety_policy,
    make_transform_chain,
    registry_to_skill_dict,
    skill_to_tool_definition,
    skills_to_registry,
    tool_definition_to_skill,
)

# ------------------------------------------------------------------
# Helpers: define some tukuy skills for testing
# ------------------------------------------------------------------


@skill(name="double", description="Double a number")
def double(x: int) -> int:
    return x * 2


@skill(name="greet", description="Greet someone by name")
def greet(name: str) -> str:
    return f"Hello, {name}!"


@skill(name="failing", description="Always fails")
def failing(x: int) -> int:
    raise ValueError("intentional failure")


@skill(name="net_skill", description="Needs network", requires_network=True)
def net_skill(url: str) -> str:
    return f"fetched {url}"


# ------------------------------------------------------------------
# TestSkillToToolDefinition
# ------------------------------------------------------------------


class TestSkillToToolDefinition:
    """Test converting a tukuy @skill to a ToolDefinition."""

    def test_from_decorated_function(self):
        td = skill_to_tool_definition(double)
        assert td.name == "double"
        assert td.description == "Double a number"
        assert "type" in td.parameters
        assert td.parameters["type"] == "object"

    def test_from_skill_instance(self):
        skill_obj = double.__skill__
        td = skill_to_tool_definition(skill_obj)
        assert td.name == "double"

    def test_wrapper_execution_success(self):
        td = skill_to_tool_definition(double)
        result = td.function(x=5)
        assert result == 10

    def test_wrapper_execution_error(self):
        td = skill_to_tool_definition(failing)
        result = td.function(x=1)
        assert isinstance(result, str)
        assert result.startswith("Error:")

    def test_wrapper_has_skill_attribute(self):
        td = skill_to_tool_definition(double)
        assert hasattr(td.function, "__skill__")
        assert isinstance(td.function.__skill__, Skill)

    def test_type_error_on_bad_input(self):
        with pytest.raises(TypeError):
            skill_to_tool_definition("not a skill")

    def test_parameters_schema_structure(self):
        td = skill_to_tool_definition(greet)
        assert td.parameters["type"] == "object"
        # Should have properties (may vary based on tukuy's schema extraction)
        assert isinstance(td.parameters, dict)


# ------------------------------------------------------------------
# TestSkillsToRegistry
# ------------------------------------------------------------------


class TestSkillsToRegistry:
    """Test batch conversion of skills to a ToolRegistry."""

    def test_batch_conversion(self):
        reg = skills_to_registry([double, greet])
        assert len(reg) == 2

    def test_registry_names(self):
        reg = skills_to_registry([double, greet])
        assert "double" in reg.names
        assert "greet" in reg.names

    def test_execute_via_registry(self):
        reg = skills_to_registry([double])
        result = reg.execute("double", {"x": 7})
        assert result == 14

    def test_empty_list(self):
        reg = skills_to_registry([])
        assert len(reg) == 0


# ------------------------------------------------------------------
# TestToolDefinitionToSkill
# ------------------------------------------------------------------


class TestToolDefinitionToSkill:
    """Test reverse bridge: ToolDefinition -> Skill."""

    def test_conversion(self):
        td = ToolDefinition(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
            function=lambda a, b: a + b,
        )
        sk = tool_definition_to_skill(td)
        assert isinstance(sk, Skill)
        assert sk.descriptor.name == "add"
        assert sk.descriptor.description == "Add two numbers"

    def test_round_trip_preserves_name(self):
        td_orig = skill_to_tool_definition(double)
        sk = tool_definition_to_skill(td_orig)
        assert sk.descriptor.name == "double"

    def test_round_trip_preserves_behavior(self):
        td_orig = skill_to_tool_definition(greet)
        sk = tool_definition_to_skill(td_orig)
        result = sk.invoke(name="World")
        assert result.success
        assert result.value == "Hello, World!"


# ------------------------------------------------------------------
# TestRegistryToSkillDict
# ------------------------------------------------------------------


class TestRegistryToSkillDict:
    """Test converting a ToolRegistry to a tukuy-compatible skill dict."""

    def test_dict_output(self):
        reg = skills_to_registry([double, greet])
        d = registry_to_skill_dict(reg)
        assert isinstance(d, dict)
        assert "double" in d
        assert "greet" in d

    def test_tukuy_skills_return_skill_objects(self):
        reg = skills_to_registry([double])
        d = registry_to_skill_dict(reg)
        assert isinstance(d["double"], Skill)

    def test_native_tools_return_skill_objects(self):
        reg = ToolRegistry()
        reg.register(lambda x: x * 3, name="triple", description="Triple a number")
        d = registry_to_skill_dict(reg)
        assert isinstance(d["triple"], Skill)
        assert d["triple"].descriptor.name == "triple"


# ------------------------------------------------------------------
# TestTukuyChainStep
# ------------------------------------------------------------------


class TestTukuyChainStep:
    """Test the TukuyChainStep pipeline adapter."""

    def test_run_calls_chain(self):
        chain = Chain(["strip", "lowercase"])
        step = TukuyChainStep(chain, name="clean")
        result = step.run("  HELLO  ")
        assert result == "hello"

    def test_name_attribute(self):
        chain = Chain(["strip"])
        step = TukuyChainStep(chain)
        assert step.name == "tukuy_chain"

    def test_custom_name(self):
        chain = Chain(["strip"])
        step = TukuyChainStep(chain, name="my_cleaner")
        assert step.name == "my_cleaner"

    def test_returns_string(self):
        chain = Chain(["strip"])
        step = TukuyChainStep(chain)
        result = step.run("  test  ")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_arun(self):
        chain = Chain(["strip", "lowercase"])
        step = TukuyChainStep(chain, name="async_clean")
        result = await step.arun("  WORLD  ")
        assert result == "world"


# ------------------------------------------------------------------
# TestApplySafetyPolicy
# ------------------------------------------------------------------


class TestApplySafetyPolicy:
    """Test safety policy gating on tool registries."""

    def test_compliant_skill_passes(self):
        reg = skills_to_registry([double])
        policy = SafetyPolicy.permissive()
        gated = apply_safety_policy(reg, policy)
        result = gated.execute("double", {"x": 5})
        assert result == 10

    def test_violating_skill_blocked(self):
        reg = skills_to_registry([net_skill])
        policy = SafetyPolicy(allow_network=False)
        gated = apply_safety_policy(reg, policy)
        result = gated.execute("net_skill", {"url": "http://example.com"})
        assert isinstance(result, str)
        assert "Safety policy violation" in result

    def test_native_tool_unaffected(self):
        reg = ToolRegistry()
        reg.register(lambda x: x + 1, name="inc", description="Increment")
        policy = SafetyPolicy(allow_network=False)
        gated = apply_safety_policy(reg, policy)
        result = gated.execute("inc", {"x": 5})
        assert result == 6

    def test_returns_new_registry(self):
        reg = skills_to_registry([double])
        policy = SafetyPolicy.permissive()
        gated = apply_safety_policy(reg, policy)
        assert gated is not reg

    def test_preserves_tool_count(self):
        reg = skills_to_registry([double, greet])
        policy = SafetyPolicy.permissive()
        gated = apply_safety_policy(reg, policy)
        assert len(gated) == 2


# ------------------------------------------------------------------
# TestMakeTransformChain
# ------------------------------------------------------------------


class TestMakeTransformChain:
    """Test the make_transform_chain convenience wrapper."""

    def test_strip(self):
        fn = make_transform_chain("strip")
        assert fn("  hello  ") == "hello"

    def test_lowercase(self):
        fn = make_transform_chain("lowercase")
        assert fn("HELLO") == "hello"

    def test_multiple_transforms(self):
        fn = make_transform_chain("strip", "lowercase")
        assert fn("  HELLO  ") == "hello"

    def test_returns_callable(self):
        fn = make_transform_chain("strip")
        assert callable(fn)


# ------------------------------------------------------------------
# TestToolRegistryMethods
# ------------------------------------------------------------------


class TestToolRegistryMethods:
    """Test the add_tukuy_skill/add_tukuy_skills methods on ToolRegistry."""

    def test_add_tukuy_skill(self):
        reg = ToolRegistry()
        td = reg.add_tukuy_skill(double)
        assert td.name == "double"
        assert "double" in reg.names

    def test_add_tukuy_skill_with_name_override(self):
        reg = ToolRegistry()
        td = reg.add_tukuy_skill(double, name="multiply_by_two")
        assert td.name == "multiply_by_two"
        assert "multiply_by_two" in reg.names

    def test_add_tukuy_skill_with_description_override(self):
        reg = ToolRegistry()
        td = reg.add_tukuy_skill(double, description="Custom description")
        assert td.description == "Custom description"

    def test_add_tukuy_skills_batch(self):
        reg = ToolRegistry()
        tds = reg.add_tukuy_skills([double, greet])
        assert len(tds) == 2
        assert "double" in reg.names
        assert "greet" in reg.names

    def test_execute_added_skill(self):
        reg = ToolRegistry()
        reg.add_tukuy_skill(double)
        result = reg.execute("double", {"x": 3})
        assert result == 6


# ------------------------------------------------------------------
# TestAgentAddTukuyTools
# ------------------------------------------------------------------


class TestAgentAddTukuyTools:
    """Test Agent.add_tukuy_tools() populates _tools."""

    def test_agent_add_tukuy_tools(self):
        from prompture import Agent

        # Use a mock driver to avoid needing a real model
        mock_driver = MagicMock()
        agent = Agent(driver=mock_driver)
        agent.add_tukuy_tools([double, greet])
        assert "double" in agent._tools.names
        assert "greet" in agent._tools.names

    def test_async_agent_add_tukuy_tools(self):
        from prompture import AsyncAgent

        mock_driver = MagicMock()
        agent = AsyncAgent(driver=mock_driver)
        agent.add_tukuy_tools([double, greet])
        assert "double" in agent._tools.names
        assert "greet" in agent._tools.names


# ------------------------------------------------------------------
# TestPipelineChainStep
# ------------------------------------------------------------------


class TestPipelineChainStep:
    """Test TukuyChainStep normalizes and executes in SkillPipeline.run()."""

    def test_chain_step_normalizes(self):
        from prompture.pipeline.pipeline import SkillPipeline

        chain = Chain(["strip", "lowercase"])
        step = TukuyChainStep(chain, name="clean")

        # This should not raise — TukuyChainStep is accepted as a step
        pipeline = SkillPipeline(
            steps=[step],
            model_name="openai/gpt-4o",
            share_conversation=False,
        )
        assert len(pipeline.steps) == 1

    def test_chain_step_executes_in_pipeline(self):
        from prompture.pipeline.pipeline import SkillPipeline

        chain = Chain(["strip", "lowercase"])
        step = TukuyChainStep(chain, name="normalizer")

        pipeline = SkillPipeline(
            steps=[step],
            model_name="openai/gpt-4o",
            share_conversation=False,
        )
        result = pipeline.run("  HELLO WORLD  ")
        assert result.success
        assert result.final_output == "hello world"
        assert len(result.steps) == 1
        assert result.steps[0].skill_name == "normalizer"
        assert result.steps[0].success

    def test_chain_step_state_passing(self):
        from prompture.pipeline.pipeline import SkillPipeline

        chain1 = Chain(["strip"])
        chain2 = Chain(["lowercase"])
        step1 = TukuyChainStep(chain1, name="strip_step")
        step2 = TukuyChainStep(chain2, name="lower_step")

        pipeline = SkillPipeline(
            steps=[step1, step2],
            model_name="openai/gpt-4o",
            share_conversation=False,
        )
        result = pipeline.run("  HELLO  ")
        assert result.success
        assert result.final_output == "hello"
        assert len(result.steps) == 2


# ------------------------------------------------------------------
# TestReExports
# ------------------------------------------------------------------


class TestReExports:
    """Test that tukuy bridge and tukuy types are re-exported from prompture."""

    def test_bridge_functions(self):
        from prompture import (
            apply_safety_policy,
            make_transform_chain,
            registry_to_skill_dict,
            skill_to_tool_definition,
            skills_to_registry,
            tool_definition_to_skill,
        )

        assert callable(skill_to_tool_definition)
        assert callable(skills_to_registry)
        assert callable(tool_definition_to_skill)
        assert callable(registry_to_skill_dict)
        assert callable(apply_safety_policy)
        assert callable(make_transform_chain)

    def test_chain_step_class(self):
        from prompture import TukuyChainStep

        assert TukuyChainStep is not None

    def test_tukuy_type_aliases(self):
        from prompture import (
            TukuyChain,
            TukuySafetyPolicy,
            TukuySkill,
            TukuySkillContext,
            TukuySkillResult,
            tukuy_branch,
            tukuy_parallel,
            tukuy_skill,
        )

        # Verify they are the actual tukuy types
        assert TukuyChain is Chain
        assert TukuySkill is Skill
        assert TukuySafetyPolicy is SafetyPolicy
        assert TukuySkillContext is SkillContext
        assert TukuySkillResult is SkillResult
        assert callable(tukuy_skill)
        assert callable(tukuy_branch)
        assert callable(tukuy_parallel)


# ------------------------------------------------------------------
# TestSecurityMetadataEnriched (tukuy >= 0.0.20)
# ------------------------------------------------------------------


@skill(
    name="enriched_skill",
    description="A skill with UI metadata",
    risk_level="moderate",
    display_name="Enriched Skill",
    icon="wrench",
    group="tools",
)
def enriched_skill(x: int) -> int:
    return x + 1


@skill(
    name="hidden_skill",
    description="A hidden skill",
    hidden=True,
    deprecated="Use enriched_skill instead",
)
def hidden_skill(x: int) -> int:
    return x


class TestSecurityMetadataEnriched:
    """Test enriched security_metadata with UI metadata from tukuy >= 0.0.20."""

    def test_risk_level_present(self):
        td = skill_to_tool_definition(enriched_skill)
        meta = td.security_metadata
        assert meta is not None
        assert "risk_level" in meta
        assert meta["risk_level"] == "moderate"

    def test_display_name_present(self):
        td = skill_to_tool_definition(enriched_skill)
        meta = td.security_metadata
        assert meta is not None
        assert meta["display_name"] == "Enriched Skill"

    def test_icon_present(self):
        td = skill_to_tool_definition(enriched_skill)
        meta = td.security_metadata
        assert meta is not None
        assert meta["icon"] == "wrench"

    def test_group_present(self):
        td = skill_to_tool_definition(enriched_skill)
        meta = td.security_metadata
        assert meta is not None
        assert meta["group"] == "tools"

    def test_hidden_absent_by_default(self):
        td = skill_to_tool_definition(enriched_skill)
        meta = td.security_metadata
        assert meta is not None
        assert "hidden" not in meta

    def test_hidden_present_when_true(self):
        td = skill_to_tool_definition(hidden_skill)
        meta = td.security_metadata
        assert meta is not None
        assert meta.get("hidden") is True

    def test_deprecated_absent_by_default(self):
        td = skill_to_tool_definition(enriched_skill)
        meta = td.security_metadata
        assert meta is not None
        assert "deprecated" not in meta

    def test_deprecated_present_when_set(self):
        td = skill_to_tool_definition(hidden_skill)
        meta = td.security_metadata
        assert meta is not None
        assert meta["deprecated"] == "Use enriched_skill instead"

    def test_basic_fields_still_present(self):
        td = skill_to_tool_definition(enriched_skill)
        meta = td.security_metadata
        assert meta is not None
        assert meta["name"] == "enriched_skill"
        assert meta["is_tukuy_skill"] is True


# ------------------------------------------------------------------
# TestFilterAvailableSkills
# ------------------------------------------------------------------


class TestFilterAvailableSkills:
    """Test filter_available_skills bridge function."""

    def test_permissive_keeps_all(self):
        from prompture import filter_available_skills

        reg = skills_to_registry([double, greet, net_skill])
        filtered = filter_available_skills(reg, policy=SafetyPolicy.permissive())
        assert len(filtered) == 3

    def test_restrictive_blocks_network(self):
        from prompture import filter_available_skills

        reg = skills_to_registry([double, net_skill])
        filtered = filter_available_skills(reg, policy=SafetyPolicy(allow_network=False))
        assert "double" in filtered.names
        assert "net_skill" not in filtered.names

    def test_native_tools_pass_through(self):
        from prompture import filter_available_skills

        reg = ToolRegistry()
        reg.register(lambda x: x + 1, name="inc", description="Increment")
        reg.add_tukuy_skill(net_skill)
        filtered = filter_available_skills(reg, policy=SafetyPolicy(allow_network=False))
        assert "inc" in filtered.names
        assert "net_skill" not in filtered.names

    def test_no_policy_keeps_all(self):
        from prompture import filter_available_skills

        reg = skills_to_registry([double, net_skill])
        filtered = filter_available_skills(reg)
        assert len(filtered) == 2


# ------------------------------------------------------------------
# TestDiscoverAndRegisterPlugins
# ------------------------------------------------------------------


class TestDiscoverAndRegisterPlugins:
    """Test discover_and_register_plugins bridge function."""

    def test_available_plugins_registered(self):
        from tukuy.plugins.base import TransformerPlugin

        from prompture import discover_and_register_plugins

        class TestPlugin(TransformerPlugin):
            @property
            def skills(self):
                return {"double": double.__skill__}

            @property
            def transformers(self):
                return {}

        plugin = TestPlugin("test_plugin")
        reg = discover_and_register_plugins([plugin])
        assert "double" in reg.names

    def test_empty_list_returns_empty_registry(self):
        from prompture import discover_and_register_plugins

        reg = discover_and_register_plugins([])
        assert len(reg) == 0


# ------------------------------------------------------------------
# TestSkillToToolDefinitionConfig
# ------------------------------------------------------------------


class TestSkillToToolDefinitionConfig:
    """Test skill_to_tool_definition with config parameter."""

    def test_config_passed_through(self):
        """When config is provided, a SkillContext with that config should be injected."""
        td = skill_to_tool_definition(double, config={"api_key": "test123"})
        # The wrapper still works (config is injected but skill may ignore it)
        result = td.function(x=5)
        assert result == 10

    def test_no_config_no_context(self):
        """When config is None (default), no context injection happens."""
        td = skill_to_tool_definition(double)
        result = td.function(x=5)
        assert result == 10


# ------------------------------------------------------------------
# TestNewReExports (tukuy >= 0.0.20)
# ------------------------------------------------------------------


class TestNewReExports:
    """Test that all 10 new tukuy re-exports are importable from prompture."""

    def test_risk_level(self):
        from tukuy import RiskLevel

        from prompture import TukuyRiskLevel

        assert TukuyRiskLevel is RiskLevel

    def test_config_scope(self):
        from tukuy import ConfigScope

        from prompture import TukuyConfigScope

        assert TukuyConfigScope is ConfigScope

    def test_config_param(self):
        from tukuy import ConfigParam

        from prompture import TukuyConfigParam

        assert TukuyConfigParam is ConfigParam

    def test_plugin_manifest(self):
        from tukuy import PluginManifest

        from prompture import TukuyPluginManifest

        assert TukuyPluginManifest is PluginManifest

    def test_plugin_requirements(self):
        from tukuy import PluginRequirements

        from prompture import TukuyPluginRequirements

        assert TukuyPluginRequirements is PluginRequirements

    def test_availability_reason(self):
        from tukuy import AvailabilityReason

        from prompture import TukuyAvailabilityReason

        assert TukuyAvailabilityReason is AvailabilityReason

    def test_skill_availability(self):
        from tukuy import SkillAvailability

        from prompture import TukuySkillAvailability

        assert TukuySkillAvailability is SkillAvailability

    def test_plugin_discovery_result(self):
        from tukuy import PluginDiscoveryResult

        from prompture import TukuyPluginDiscoveryResult

        assert TukuyPluginDiscoveryResult is PluginDiscoveryResult

    def test_get_available_skills(self):
        from tukuy import get_available_skills

        from prompture import tukuy_get_available_skills

        assert tukuy_get_available_skills is get_available_skills

    def test_discover_plugins(self):
        from tukuy import discover_plugins

        from prompture import tukuy_discover_plugins

        assert tukuy_discover_plugins is discover_plugins
