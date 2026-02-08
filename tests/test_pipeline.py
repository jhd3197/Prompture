"""Tests for skill pipeline module."""

from unittest.mock import MagicMock, patch

import pytest

from prompture.groups.types import ErrorPolicy
from prompture.agents.persona import Persona
from prompture.pipeline.pipeline import (
    PipelineResult,
    PipelineStep,
    SkillPipeline,
    StepResult,
    _inject_state,
    create_pipeline,
)
from prompture.agents.skills import SkillInfo


class TestInjectState:
    """Tests for state injection helper."""

    def test_simple_replacement(self):
        """Should replace placeholders with state values."""
        result = _inject_state("Hello {name}!", {"name": "World"})
        assert result == "Hello World!"

    def test_multiple_replacements(self):
        """Should replace multiple placeholders."""
        result = _inject_state("{greeting} {name}!", {"greeting": "Hello", "name": "World"})
        assert result == "Hello World!"

    def test_missing_key_preserved(self):
        """Missing keys should be preserved as-is."""
        result = _inject_state("Hello {name} and {other}!", {"name": "World"})
        assert result == "Hello World and {other}!"

    def test_empty_state(self):
        """Empty state should preserve all placeholders."""
        result = _inject_state("Hello {name}!", {})
        assert result == "Hello {name}!"

    def test_no_placeholders(self):
        """String without placeholders should pass through."""
        result = _inject_state("Hello World!", {"name": "Test"})
        assert result == "Hello World!"


class TestPipelineStep:
    """Tests for PipelineStep dataclass."""

    def test_basic_creation(self):
        """Should create step with minimal args."""
        step = PipelineStep(skill="test-skill")
        assert step.skill == "test-skill"
        assert step.output_key is None
        assert step.input_template is None
        assert step.condition is None

    def test_full_creation(self):
        """Should create step with all args."""

        def condition(state):
            return True

        step = PipelineStep(
            skill="test-skill",
            output_key="result",
            input_template="Process {input}",
            condition=condition,
        )

        assert step.skill == "test-skill"
        assert step.output_key == "result"
        assert step.input_template == "Process {input}"
        assert step.condition is condition


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_successful_step(self):
        """Should represent successful step."""
        result = StepResult(
            step_index=0,
            skill_name="test",
            output_key="step_0",
            output="Hello",
            success=True,
            duration_ms=100.0,
            usage={"total_tokens": 50},
        )

        assert result.success
        assert not result.skipped
        assert result.error is None

    def test_failed_step(self):
        """Should represent failed step."""
        result = StepResult(
            step_index=1,
            skill_name="test",
            output_key="step_1",
            success=False,
            error="Something went wrong",
        )

        assert not result.success
        assert result.error == "Something went wrong"

    def test_skipped_step(self):
        """Should represent skipped step."""
        result = StepResult(
            step_index=2,
            skill_name="test",
            output_key="step_2",
            skipped=True,
        )

        assert result.skipped
        assert result.success  # Skipped is still considered success


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_to_dict(self):
        """Should serialize to dict."""
        result = PipelineResult(
            final_output="Final text",
            state={"input": "test", "step_0": "output"},
            steps=[StepResult(step_index=0, skill_name="skill1", output_key="step_0", output="output", success=True)],
            usage={"total_tokens": 100, "cost": 0.001},
            success=True,
            elapsed_ms=500.0,
        )

        d = result.to_dict()

        assert d["final_output"] == "Final text"
        assert d["success"]
        assert len(d["steps"]) == 1
        assert d["steps"][0]["skill_name"] == "skill1"
        assert d["usage"]["total_tokens"] == 100
        assert d["elapsed_ms"] == 500.0


class TestSkillPipelineNormalization:
    """Tests for step normalization in SkillPipeline."""

    def test_normalize_string_steps(self):
        """Should normalize string skill names."""
        pipeline = SkillPipeline(["skill1", "skill2"], model_name="openai/gpt-4o")

        assert len(pipeline.steps) == 2
        assert all(isinstance(s, PipelineStep) for s in pipeline.steps)
        assert pipeline.steps[0].skill == "skill1"
        assert pipeline.steps[0].output_key == "step_0"
        assert pipeline.steps[1].output_key == "step_1"

    def test_normalize_persona_steps(self):
        """Should normalize Persona instances."""
        persona = Persona(name="test", system_prompt="Test prompt")
        pipeline = SkillPipeline([persona], model_name="openai/gpt-4o")

        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].skill is persona

    def test_normalize_skill_info_steps(self):
        """Should normalize SkillInfo instances."""
        skill = SkillInfo(name="test-skill", description="Test", instructions="Do something")
        pipeline = SkillPipeline([skill], model_name="openai/gpt-4o")

        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].skill is skill

    def test_normalize_pipeline_step_preserves_config(self):
        """Should preserve PipelineStep configuration."""
        step = PipelineStep(
            skill="skill1",
            output_key="custom_key",
            input_template="Custom template",
        )
        pipeline = SkillPipeline([step], model_name="openai/gpt-4o")

        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].output_key == "custom_key"
        assert pipeline.steps[0].input_template == "Custom template"

    def test_normalize_mixed_steps(self):
        """Should handle mixed step types."""
        persona = Persona(name="persona1", system_prompt="Test")
        step = PipelineStep(skill="skill1", output_key="result")

        pipeline = SkillPipeline(["string_skill", persona, step], model_name="openai/gpt-4o")

        assert len(pipeline.steps) == 3
        assert pipeline.steps[0].skill == "string_skill"
        assert pipeline.steps[1].skill is persona
        assert pipeline.steps[2].output_key == "result"


class TestSkillPipelineConfiguration:
    """Tests for SkillPipeline configuration."""

    def test_default_configuration(self):
        """Should use default configuration."""
        pipeline = SkillPipeline(["skill1"], model_name="openai/gpt-4o")

        assert pipeline.model_name == "openai/gpt-4o"
        assert pipeline.share_conversation is True
        assert pipeline.error_policy == ErrorPolicy.fail_fast
        assert pipeline.system_prompt is None

    def test_custom_configuration(self):
        """Should accept custom configuration."""
        pipeline = SkillPipeline(
            ["skill1"],
            model_name="claude/claude-sonnet-4",
            share_conversation=False,
            error_policy=ErrorPolicy.continue_on_error,
            system_prompt="Custom prompt",
            options={"temperature": 0.5},
        )

        assert pipeline.model_name == "claude/claude-sonnet-4"
        assert pipeline.share_conversation is False
        assert pipeline.error_policy == ErrorPolicy.continue_on_error
        assert pipeline.system_prompt == "Custom prompt"
        assert pipeline.options["temperature"] == 0.5


class TestSkillPipelineResolveSkill:
    """Tests for skill resolution."""

    def test_resolve_persona(self):
        """Should resolve Persona directly."""
        persona = Persona(name="test", system_prompt="Test")
        pipeline = SkillPipeline([], model_name="openai/gpt-4o")

        name, resolved = pipeline._resolve_skill(persona)

        assert name == "test"
        assert resolved is persona

    def test_resolve_skill_info(self):
        """Should convert SkillInfo to Persona."""
        skill = SkillInfo(name="test-skill", description="Test desc", instructions="Instructions")
        pipeline = SkillPipeline([], model_name="openai/gpt-4o")

        name, resolved = pipeline._resolve_skill(skill)

        assert name == "test-skill"
        assert isinstance(resolved, Persona)
        assert resolved.system_prompt == "Instructions"

    def test_resolve_string_not_found(self):
        """Should raise on unknown skill name."""
        pipeline = SkillPipeline([], model_name="openai/gpt-4o")

        with pytest.raises(ValueError, match="not found"):
            pipeline._resolve_skill("nonexistent-skill")


class TestSkillPipelineAggregateUsage:
    """Tests for usage aggregation."""

    def test_aggregate_empty(self):
        """Should handle empty usage list."""
        pipeline = SkillPipeline([], model_name="openai/gpt-4o")
        result = pipeline._aggregate_usage([])

        assert result["total_tokens"] == 0
        assert result["cost"] == 0.0
        assert result["call_count"] == 0

    def test_aggregate_single(self):
        """Should aggregate single usage."""
        pipeline = SkillPipeline([], model_name="openai/gpt-4o")
        result = pipeline._aggregate_usage(
            [{"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 0.001}]
        )

        assert result["prompt_tokens"] == 10
        assert result["completion_tokens"] == 20
        assert result["total_tokens"] == 30
        assert result["cost"] == 0.001
        assert result["call_count"] == 1

    def test_aggregate_multiple(self):
        """Should aggregate multiple usages."""
        pipeline = SkillPipeline([], model_name="openai/gpt-4o")
        result = pipeline._aggregate_usage(
            [
                {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 0.001},
                {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40, "cost": 0.002},
            ]
        )

        assert result["prompt_tokens"] == 25
        assert result["completion_tokens"] == 45
        assert result["total_tokens"] == 70
        assert result["cost"] == 0.003
        assert result["call_count"] == 2


class TestSkillPipelineRun:
    """Tests for pipeline execution."""

    @patch("prompture.agents.conversation.Conversation")
    def test_run_with_persona(self, mock_conv_cls):
        """Should run pipeline with persona steps."""
        # Mock conversation
        mock_conv = MagicMock()
        mock_conv.ask.return_value = "Step output"
        mock_conv.usage.to_dict.return_value = {"total_tokens": 50, "cost": 0.001}
        mock_conv_cls.return_value = mock_conv

        # Need to patch where Conversation is imported
        with patch.object(SkillPipeline, "run") as mock_run:
            # Setup return value
            mock_run.return_value = PipelineResult(
                final_output="Step output",
                state={"input": "Input text", "step_0": "Step output"},
                steps=[
                    StepResult(step_index=0, skill_name="test", output_key="step_0", output="Step output", success=True)
                ],
                usage={"total_tokens": 50, "cost": 0.001},
                success=True,
            )

            persona = Persona(name="test", system_prompt="Test prompt")
            pipeline = SkillPipeline([persona], model_name="openai/gpt-4o")

            result = pipeline.run("Input text")

            assert result.success
            assert result.final_output == "Step output"
            assert "input" in result.state
            assert "step_0" in result.state

    def test_run_multiple_steps_structure(self):
        """Should set up multiple steps correctly."""
        p1 = Persona(name="step1", system_prompt="Step 1")
        p2 = Persona(name="step2", system_prompt="Step 2")
        pipeline = SkillPipeline([p1, p2], model_name="openai/gpt-4o")

        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].skill is p1
        assert pipeline.steps[1].skill is p2

    def test_run_with_input_template_structure(self):
        """Should preserve input template in steps."""
        p1 = Persona(name="step1", system_prompt="Step 1")
        step2 = PipelineStep(
            skill=Persona(name="step2", system_prompt="Step 2"),
            input_template="Process: {step_0}",
            output_key="result",
        )

        pipeline = SkillPipeline([p1, step2], model_name="openai/gpt-4o")

        assert pipeline.steps[1].input_template == "Process: {step_0}"
        assert pipeline.steps[1].output_key == "result"

    def test_run_with_condition_structure(self):
        """Should preserve condition in steps."""

        def condition_fn(state):
            return len(state.get("input", "")) > 10

        step = PipelineStep(
            skill=Persona(name="s1", system_prompt="P1"),
            condition=condition_fn,
        )

        pipeline = SkillPipeline([step], model_name="openai/gpt-4o")

        assert pipeline.steps[0].condition is condition_fn

    def test_error_policy_configuration(self):
        """Should respect error policy configuration."""
        pipeline = SkillPipeline(
            [Persona(name="s1", system_prompt="P1")],
            model_name="openai/gpt-4o",
            error_policy=ErrorPolicy.fail_fast,
        )

        assert pipeline.error_policy == ErrorPolicy.fail_fast

    def test_continue_on_error_policy(self):
        """Should accept continue_on_error policy."""
        pipeline = SkillPipeline(
            [Persona(name="s1", system_prompt="P1")],
            model_name="openai/gpt-4o",
            error_policy=ErrorPolicy.continue_on_error,
        )

        assert pipeline.error_policy == ErrorPolicy.continue_on_error

    def test_pipeline_steps_empty(self):
        """Should handle empty pipeline."""
        pipeline = SkillPipeline([], model_name="openai/gpt-4o")
        assert pipeline.steps == []

    def test_initial_state_in_config(self):
        """Pipeline should accept options for configuration."""
        pipeline = SkillPipeline(
            [],
            model_name="openai/gpt-4o",
            options={"temperature": 0.5},
        )
        assert pipeline.options["temperature"] == 0.5


class TestSkillPipelineRepr:
    """Tests for __repr__."""

    def test_repr_string_skills(self):
        """Should show string skill names."""
        pipeline = SkillPipeline(["skill1", "skill2"], model_name="openai/gpt-4o")
        repr_str = repr(pipeline)

        assert "SkillPipeline" in repr_str
        assert "skill1" in repr_str
        assert "skill2" in repr_str
        assert "openai/gpt-4o" in repr_str

    def test_repr_persona_skills(self):
        """Should show persona names."""
        p1 = Persona(name="persona1", system_prompt="P1")
        pipeline = SkillPipeline([p1], model_name="openai/gpt-4o")
        repr_str = repr(pipeline)

        assert "persona1" in repr_str


class TestCreatePipeline:
    """Tests for create_pipeline convenience function."""

    def test_basic_usage(self):
        """Should create pipeline with positional steps."""
        pipeline = create_pipeline("skill1", "skill2", model_name="openai/gpt-4o")

        assert isinstance(pipeline, SkillPipeline)
        assert len(pipeline.steps) == 2
        assert pipeline.model_name == "openai/gpt-4o"

    def test_with_kwargs(self):
        """Should pass kwargs to SkillPipeline."""
        pipeline = create_pipeline(
            "skill1",
            model_name="claude/sonnet",
            share_conversation=False,
            error_policy=ErrorPolicy.continue_on_error,
        )

        assert pipeline.model_name == "claude/sonnet"
        assert not pipeline.share_conversation
        assert pipeline.error_policy == ErrorPolicy.continue_on_error
