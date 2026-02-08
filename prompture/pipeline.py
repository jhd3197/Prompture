"""Skill pipelines for Prompture.

Chain skills into multi-step workflows with shared context and state passing.
Supports both synchronous and asynchronous execution with configurable error
handling policies.

Features:
- Step normalization (skills, personas, or strings from registry)
- State passing between pipeline steps with template substitution
- Conditional step execution
- Shared conversation context option
- Sync and async execution modes
- Aggregated usage tracking
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

from .group_types import ErrorPolicy

if TYPE_CHECKING:
    from .persona import Persona
    from .skills import SkillInfo

logger = logging.getLogger("prompture.pipeline")


@dataclass
class PipelineStep:
    """A single step in a skill pipeline.

    Args:
        skill: The skill to execute. Can be:
            - SkillInfo instance
            - Persona instance
            - String name from skill registry
        output_key: Key to store this step's output in the pipeline state.
            If None, uses "step_{index}".
        input_template: Template for constructing input to this step.
            Supports ``{key}`` placeholders for state variables.
            If None, uses the previous step's output or initial input.
        condition: Optional callable ``(state) -> bool``. If provided and
            returns False, the step is skipped.
    """

    skill: Union[SkillInfo, Persona, str]
    output_key: str | None = None
    input_template: str | None = None
    condition: Callable[[dict[str, Any]], bool] | None = None


@dataclass
class StepResult:
    """Result of executing a single pipeline step.

    Args:
        step_index: Index of this step in the pipeline.
        skill_name: Name of the skill that was executed.
        output_key: The state key where output was stored.
        output: The output text from this step.
        success: Whether the step completed successfully.
        skipped: Whether the step was skipped due to condition.
        error: Error message if step failed.
        duration_ms: Execution time in milliseconds.
        usage: Token usage for this step.
    """

    step_index: int
    skill_name: str
    output_key: str
    output: str = ""
    success: bool = True
    skipped: bool = False
    error: str | None = None
    duration_ms: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of pipeline execution.

    Args:
        final_output: The output from the last executed step.
        state: Final state dict containing all step outputs.
        steps: List of per-step results.
        usage: Aggregated token/cost usage across all steps.
        success: Whether all steps completed successfully.
        error: Error message if pipeline failed.
        elapsed_ms: Total execution time in milliseconds.
    """

    final_output: str
    state: dict[str, Any]
    steps: list[StepResult]
    usage: dict[str, Any]
    success: bool
    error: str | None = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "final_output": self.final_output,
            "state": self.state,
            "steps": [
                {
                    "step_index": s.step_index,
                    "skill_name": s.skill_name,
                    "output_key": s.output_key,
                    "output": s.output,
                    "success": s.success,
                    "skipped": s.skipped,
                    "error": s.error,
                    "duration_ms": s.duration_ms,
                    "usage": s.usage,
                }
                for s in self.steps
            ],
            "usage": self.usage,
            "success": self.success,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
        }


def _inject_state(template: str, state: dict[str, Any]) -> str:
    """Replace ``{key}`` placeholders with state values.

    Unknown keys pass through unchanged.

    Args:
        template: Template string with placeholders.
        state: Dict of key-value pairs for substitution.

    Returns:
        Template with placeholders replaced.
    """

    def _replacer(m: re.Match[str]) -> str:
        key = m.group(1)
        if key in state:
            return str(state[key])
        return m.group(0)  # leave unchanged

    return re.sub(r"\{(\w+)\}", _replacer, template)


class SkillPipeline:
    """Chain skills into multi-step workflows with shared context.

    A pipeline executes a sequence of skills (or personas) in order,
    passing the output of each step to the next via shared state.
    Supports optional conversation sharing for context continuity.

    Args:
        steps: List of pipeline steps. Can be:
            - SkillInfo instances
            - Persona instances
            - String skill names (looked up from registry)
            - PipelineStep instances for full control
        model_name: Default model string for all steps.
        share_conversation: If True, use a single Conversation instance
            across all steps for context continuity.
        error_policy: How to handle step failures.
        system_prompt: Optional system prompt for shared conversation mode.
        options: Additional driver options for all steps.

    Example:
        >>> from prompture import SkillPipeline
        >>> pipeline = SkillPipeline([
        ...     "extract-entities",
        ...     "classify-sentiment",
        ...     PipelineStep(
        ...         skill="summarize",
        ...         input_template="Summarize the entities {entities} with sentiment {sentiment}",
        ...         condition=lambda state: len(state.get("entities", "")) > 100
        ...     )
        ... ], model_name="openai/gpt-4o")
        >>> result = pipeline.run("Customer feedback text...")
    """

    def __init__(
        self,
        steps: list[Union[SkillInfo, Persona, str, PipelineStep]],
        model_name: str = "openai/gpt-4o",
        share_conversation: bool = True,
        error_policy: ErrorPolicy = ErrorPolicy.fail_fast,
        system_prompt: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.steps = self._normalize_steps(steps)
        self.model_name = model_name
        self.share_conversation = share_conversation
        self.error_policy = error_policy
        self.system_prompt = system_prompt
        self.options = options or {}

    def _normalize_steps(
        self,
        steps: list[Union[SkillInfo, Persona, str, PipelineStep]],
    ) -> list[PipelineStep]:
        """Convert various step formats to PipelineStep instances.

        Args:
            steps: Mixed list of step specifications.

        Returns:
            List of normalized PipelineStep instances.
        """
        result: list[PipelineStep] = []
        for i, step in enumerate(steps):
            if isinstance(step, PipelineStep):
                # Assign default output_key if not set
                if step.output_key is None:
                    step = PipelineStep(
                        skill=step.skill,
                        output_key=f"step_{i}",
                        input_template=step.input_template,
                        condition=step.condition,
                    )
                result.append(step)
            elif isinstance(step, str):
                # Skill name from registry
                result.append(PipelineStep(skill=step, output_key=f"step_{i}"))
            elif hasattr(step, "chain") and hasattr(step, "run"):
                # TukuyChainStep
                result.append(PipelineStep(skill=step, output_key=f"step_{i}"))
            else:
                # SkillInfo or Persona directly
                result.append(PipelineStep(skill=step, output_key=f"step_{i}"))
        return result

    def _resolve_skill(self, skill: Union[SkillInfo, Persona, str]) -> tuple[str, Persona]:
        """Resolve a skill reference to a Persona instance.

        Args:
            skill: Skill specification (SkillInfo, Persona, or name string).

        Returns:
            Tuple of (skill_name, persona).

        Raises:
            ValueError: If skill name not found in registry.
        """
        from .persona import Persona
        from .skills import SkillInfo, get_skill

        if isinstance(skill, str):
            # Look up from skill registry
            skill_info = get_skill(skill)
            if skill_info is None:
                # Try persona registry as fallback
                from .persona import get_persona

                persona = get_persona(skill)
                if persona is not None:
                    return skill, persona
                raise ValueError(f"Skill '{skill}' not found in registry")
            return skill_info.name, skill_info.as_persona()

        if isinstance(skill, SkillInfo):
            return skill.name, skill.as_persona()

        if isinstance(skill, Persona):
            return skill.name, skill

        raise TypeError(f"Invalid skill type: {type(skill).__name__}")

    def _aggregate_usage(self, usages: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate usage stats from multiple steps.

        Args:
            usages: List of usage dicts from each step.

        Returns:
            Combined usage dict.
        """
        agg: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "call_count": len(usages),
        }
        for u in usages:
            agg["prompt_tokens"] += u.get("prompt_tokens", 0)
            agg["completion_tokens"] += u.get("completion_tokens", 0)
            agg["total_tokens"] += u.get("total_tokens", 0)
            agg["cost"] += u.get("cost", 0.0)
        return agg

    def run(
        self,
        input_text: str,
        initial_state: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute pipeline synchronously.

        Args:
            input_text: Initial input text for the first step.
            initial_state: Optional pre-populated state dict.

        Returns:
            PipelineResult with final output and all step results.

        Example:
            >>> result = pipeline.run("Analyze this customer feedback...")
            >>> print(result.final_output)
            >>> print(f"Total cost: ${result.usage['cost']:.4f}")
        """
        from .conversation import Conversation

        t0 = time.perf_counter()
        state: dict[str, Any] = dict(initial_state) if initial_state else {}
        state["input"] = input_text

        step_results: list[StepResult] = []
        usages: list[dict[str, Any]] = []
        final_output = input_text
        current_input = input_text

        # Create shared conversation if enabled
        shared_conv: Conversation | None = None
        if self.share_conversation:
            shared_conv = Conversation(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                options=self.options,
            )

        for i, step in enumerate(self.steps):
            step_t0 = time.perf_counter()
            output_key = step.output_key or f"step_{i}"

            # Check condition
            if step.condition is not None:
                try:
                    should_run = step.condition(state)
                except Exception as e:
                    logger.warning("Condition check failed for step %d: %s", i, e)
                    should_run = True  # Default to running on condition error

                if not should_run:
                    step_results.append(
                        StepResult(
                            step_index=i,
                            skill_name=str(step.skill),
                            output_key=output_key,
                            skipped=True,
                            duration_ms=(time.perf_counter() - step_t0) * 1000,
                        )
                    )
                    logger.debug("Step %d skipped due to condition", i)
                    continue

            # Handle TukuyChainStep directly (no LLM call needed)
            from .tukuy_bridge import TukuyChainStep

            if isinstance(step.skill, TukuyChainStep):
                if step.input_template:
                    step_input = _inject_state(step.input_template, state)
                else:
                    step_input = current_input
                try:
                    step_output = step.skill.run(step_input)
                    state[output_key] = step_output
                    current_input = step_output
                    final_output = step_output
                    step_results.append(
                        StepResult(
                            step_index=i,
                            skill_name=step.skill.name,
                            output_key=output_key,
                            output=str(step_output),
                            success=True,
                            duration_ms=(time.perf_counter() - step_t0) * 1000,
                        )
                    )
                except Exception as e:
                    error_msg = str(e)
                    step_results.append(
                        StepResult(
                            step_index=i,
                            skill_name=step.skill.name,
                            output_key=output_key,
                            success=False,
                            error=error_msg,
                            duration_ms=(time.perf_counter() - step_t0) * 1000,
                        )
                    )
                    if self.error_policy == ErrorPolicy.raise_on_error:
                        raise
                    if self.error_policy == ErrorPolicy.fail_fast:
                        break
                continue

            # Resolve skill to persona
            try:
                skill_name, persona = self._resolve_skill(step.skill)
            except Exception as e:
                error_msg = f"Failed to resolve skill: {e}"
                step_results.append(
                    StepResult(
                        step_index=i,
                        skill_name=str(step.skill),
                        output_key=output_key,
                        success=False,
                        error=error_msg,
                        duration_ms=(time.perf_counter() - step_t0) * 1000,
                    )
                )
                if self.error_policy == ErrorPolicy.raise_on_error:
                    raise
                if self.error_policy == ErrorPolicy.fail_fast:
                    break
                continue

            # Build step input
            if step.input_template:
                step_input = _inject_state(step.input_template, state)
            else:
                step_input = current_input

            # Execute step
            try:
                if shared_conv is not None:
                    # Use shared conversation - apply persona as system message extension
                    rendered_prompt = persona.render()
                    full_prompt = f"{rendered_prompt}\n\n{step_input}"
                    response = shared_conv.ask(full_prompt)
                    step_output = response
                    step_usage = shared_conv.usage.to_dict() if hasattr(shared_conv.usage, "to_dict") else {}
                else:
                    # Create new conversation per step
                    conv = Conversation(
                        model_name=self.model_name,
                        system_prompt=persona.render(),
                        options=self.options,
                    )
                    response = conv.ask(step_input)
                    step_output = response
                    step_usage = conv.usage.to_dict() if hasattr(conv.usage, "to_dict") else {}

                # Store in state
                state[output_key] = step_output
                current_input = step_output
                final_output = step_output

                step_results.append(
                    StepResult(
                        step_index=i,
                        skill_name=skill_name,
                        output_key=output_key,
                        output=step_output,
                        success=True,
                        duration_ms=(time.perf_counter() - step_t0) * 1000,
                        usage=step_usage,
                    )
                )
                usages.append(step_usage)

                logger.debug("Step %d (%s) completed successfully", i, skill_name)

            except Exception as e:
                error_msg = str(e)
                step_results.append(
                    StepResult(
                        step_index=i,
                        skill_name=skill_name,
                        output_key=output_key,
                        success=False,
                        error=error_msg,
                        duration_ms=(time.perf_counter() - step_t0) * 1000,
                    )
                )

                logger.error("Step %d (%s) failed: %s", i, skill_name, error_msg)

                if self.error_policy == ErrorPolicy.raise_on_error:
                    raise
                if self.error_policy == ErrorPolicy.fail_fast:
                    break
                # continue_on_error: proceed to next step

        elapsed_ms = (time.perf_counter() - t0) * 1000
        success = all(s.success or s.skipped for s in step_results)

        return PipelineResult(
            final_output=final_output,
            state=state,
            steps=step_results,
            usage=self._aggregate_usage(usages),
            success=success,
            error=None if success else "One or more steps failed",
            elapsed_ms=elapsed_ms,
        )

    async def run_async(
        self,
        input_text: str,
        initial_state: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute pipeline asynchronously.

        Runs the synchronous pipeline in a thread pool executor
        to avoid blocking the event loop. For true async execution
        with async conversations, use AsyncSkillPipeline.

        Args:
            input_text: Initial input text for the first step.
            initial_state: Optional pre-populated state dict.

        Returns:
            PipelineResult with final output and all step results.

        Example:
            >>> result = await pipeline.run_async("Analyze this feedback...")
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.run(input_text, initial_state),
        )

    def __repr__(self) -> str:
        step_names = []
        for s in self.steps:
            if isinstance(s.skill, str):
                step_names.append(s.skill)
            elif hasattr(s.skill, "name"):
                step_names.append(s.skill.name)
            else:
                step_names.append(str(s.skill))
        return f"SkillPipeline({step_names}, model={self.model_name})"


def create_pipeline(
    *steps: Union[SkillInfo, Persona, str, PipelineStep],
    model_name: str = "openai/gpt-4o",
    share_conversation: bool = True,
    **kwargs: Any,
) -> SkillPipeline:
    """Convenience function to create a skill pipeline.

    Args:
        *steps: Pipeline steps as positional arguments.
        model_name: Default model string for all steps.
        share_conversation: If True, share context between steps.
        **kwargs: Additional SkillPipeline parameters.

    Returns:
        Configured SkillPipeline instance.

    Example:
        >>> pipeline = create_pipeline(
        ...     "extract-entities",
        ...     "classify-sentiment",
        ...     model_name="openai/gpt-4o",
        ... )
    """
    return SkillPipeline(
        steps=list(steps),
        model_name=model_name,
        share_conversation=share_conversation,
        **kwargs,
    )
