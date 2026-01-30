"""Agent framework for Prompture.

Provides a reusable :class:`Agent` that wraps a ReAct-style loop around
:class:`~prompture.conversation.Conversation`, with optional structured
output via Pydantic models and tool support via :class:`ToolRegistry`.

Example::

    from prompture import Agent

    agent = Agent("openai/gpt-4o", system_prompt="You are a helpful assistant.")
    result = agent.run("What is the capital of France?")
    print(result.output)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from .agent_types import AgentResult, AgentState, AgentStep, StepType
from .conversation import Conversation
from .driver import Driver
from .tools import clean_json_text
from .tools_schema import ToolRegistry

logger = logging.getLogger("prompture.agent")

_OUTPUT_PARSE_MAX_RETRIES = 3


class Agent:
    """A reusable agent that executes a ReAct loop with tool support.

    Each call to :meth:`run` creates a fresh :class:`Conversation`,
    preventing state leakage between runs.  The Agent itself is a
    template holding model config, tools, and system prompt.

    Args:
        model: Model string in ``"provider/model"`` format.
        driver: Pre-built driver instance (useful for testing).
        tools: Initial tools as a list of callables or a
            :class:`ToolRegistry`.
        system_prompt: System prompt prepended to every run.
        output_type: Optional Pydantic model class.  When set, the
            final LLM response is parsed and validated against this type.
        max_iterations: Maximum tool-use rounds per run.
        options: Extra driver options forwarded to every call.
    """

    def __init__(
        self,
        model: str = "",
        *,
        driver: Driver | None = None,
        tools: list[Callable[..., Any]] | ToolRegistry | None = None,
        system_prompt: str | None = None,
        output_type: type[BaseModel] | None = None,
        max_iterations: int = 10,
        options: dict[str, Any] | None = None,
    ) -> None:
        if not model and driver is None:
            raise ValueError("Either model or driver must be provided")

        self._model = model
        self._driver = driver
        self._system_prompt = system_prompt
        self._output_type = output_type
        self._max_iterations = max_iterations
        self._options = dict(options) if options else {}

        # Build internal tool registry
        self._tools = ToolRegistry()
        if isinstance(tools, ToolRegistry):
            self._tools = tools
        elif tools is not None:
            for fn in tools:
                self._tools.register(fn)

        self._state = AgentState.idle
        self._stop_requested = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tool(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to register a function as a tool on this agent.

        Returns the original function unchanged.
        """
        self._tools.register(fn)
        return fn

    @property
    def state(self) -> AgentState:
        """Current lifecycle state of the agent."""
        return self._state

    def stop(self) -> None:
        """Request graceful shutdown after the current iteration."""
        self._stop_requested = True

    def run(self, prompt: str) -> AgentResult:
        """Execute the agent loop to completion.

        Creates a fresh :class:`Conversation`, sends the prompt,
        handles any tool calls, and optionally parses the final
        response into an ``output_type`` Pydantic model.
        """
        self._state = AgentState.running
        self._stop_requested = False
        steps: list[AgentStep] = []

        try:
            result = self._execute(prompt, steps)
            self._state = AgentState.idle
            return result
        except Exception:
            self._state = AgentState.errored
            raise

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_system_prompt(self) -> str | None:
        """Build the system prompt, appending output schema if needed."""
        parts: list[str] = []
        if self._system_prompt:
            parts.append(self._system_prompt)

        if self._output_type is not None:
            schema = self._output_type.model_json_schema()
            schema_str = json.dumps(schema, indent=2)
            parts.append(
                "You MUST respond with a single JSON object (no markdown, "
                "no extra text) that validates against this JSON schema:\n"
                f"{schema_str}\n\n"
                "Use double quotes for keys and strings. "
                "If a value is unknown use null."
            )

        return "\n\n".join(parts) if parts else None

    def _build_conversation(self) -> Conversation:
        """Create a fresh Conversation for a single run."""
        kwargs: dict[str, Any] = {
            "system_prompt": self._resolve_system_prompt(),
            "tools": self._tools if self._tools else None,
            "max_tool_rounds": self._max_iterations,
        }
        if self._options:
            kwargs["options"] = self._options

        if self._driver is not None:
            kwargs["driver"] = self._driver
        else:
            kwargs["model_name"] = self._model

        return Conversation(**kwargs)

    def _execute(self, prompt: str, steps: list[AgentStep]) -> AgentResult:
        """Core execution: run conversation, extract steps, parse output."""
        conv = self._build_conversation()

        # Ask the conversation (handles full tool loop internally)
        t0 = time.perf_counter()
        response_text = conv.ask(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Extract steps and tool calls from conversation messages
        all_tool_calls: list[dict[str, Any]] = []
        self._extract_steps(conv.messages, steps, all_tool_calls)

        # Handle output_type parsing
        if self._output_type is not None:
            output, output_text = self._parse_output(conv, response_text, steps, all_tool_calls, elapsed_ms)
        else:
            output = response_text
            output_text = response_text

        return AgentResult(
            output=output,
            output_text=output_text,
            messages=conv.messages,
            usage=conv.usage,
            steps=steps,
            all_tool_calls=all_tool_calls,
            state=AgentState.idle,
        )

    def _extract_steps(
        self,
        messages: list[dict[str, Any]],
        steps: list[AgentStep],
        all_tool_calls: list[dict[str, Any]],
    ) -> None:
        """Scan conversation messages and populate steps and tool_calls."""
        now = time.time()

        for msg in messages:
            role = msg.get("role", "")

            if role == "assistant":
                tc_list = msg.get("tool_calls", [])
                if tc_list:
                    # Assistant message with tool calls
                    for tc in tc_list:
                        fn = tc.get("function", {})
                        name = fn.get("name", tc.get("name", ""))
                        raw_args = fn.get("arguments", tc.get("arguments", "{}"))
                        if isinstance(raw_args, str):
                            try:
                                args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                args = {}
                        else:
                            args = raw_args

                        steps.append(
                            AgentStep(
                                step_type=StepType.tool_call,
                                timestamp=now,
                                content=msg.get("content", ""),
                                tool_name=name,
                                tool_args=args,
                            )
                        )
                        all_tool_calls.append({"name": name, "arguments": args, "id": tc.get("id", "")})
                else:
                    # Final assistant message (no tool calls)
                    steps.append(
                        AgentStep(
                            step_type=StepType.output,
                            timestamp=now,
                            content=msg.get("content", ""),
                        )
                    )

            elif role == "tool":
                steps.append(
                    AgentStep(
                        step_type=StepType.tool_result,
                        timestamp=now,
                        content=msg.get("content", ""),
                        tool_name=msg.get("tool_call_id"),
                    )
                )

    def _parse_output(
        self,
        conv: Conversation,
        response_text: str,
        steps: list[AgentStep],
        all_tool_calls: list[dict[str, Any]],
        elapsed_ms: float,
    ) -> tuple[Any, str]:
        """Try to parse ``response_text`` as the output_type, with retries."""
        assert self._output_type is not None

        last_error: Exception | None = None
        text = response_text

        for attempt in range(_OUTPUT_PARSE_MAX_RETRIES):
            try:
                cleaned = clean_json_text(text)
                parsed = json.loads(cleaned)
                model_instance = self._output_type.model_validate(parsed)
                return model_instance, text
            except Exception as exc:
                last_error = exc
                if attempt < _OUTPUT_PARSE_MAX_RETRIES - 1:
                    logger.debug("Output parse attempt %d failed: %s", attempt + 1, exc)
                    retry_msg = (
                        f"Your previous response could not be parsed as valid JSON "
                        f"matching the required schema. Error: {exc}\n\n"
                        f"Please try again and respond ONLY with valid JSON."
                    )
                    text = conv.ask(retry_msg)

                    # Record the retry step
                    self._extract_steps(conv.messages[-2:], steps, all_tool_calls)

        raise ValueError(
            f"Failed to parse output as {self._output_type.__name__} "
            f"after {_OUTPUT_PARSE_MAX_RETRIES} attempts: {last_error}"
        )
