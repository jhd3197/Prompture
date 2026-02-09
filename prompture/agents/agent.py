"""Agent framework for Prompture.

Provides a reusable :class:`Agent` that wraps a ReAct-style loop around
:class:`~prompture.conversation.Conversation`, with optional structured
output via Pydantic models and tool support via :class:`ToolRegistry`.

Example::

    from prompture import Agent

    agent = Agent("openai/gpt-4o", system_prompt="You are a helpful assistant.")
    result = agent.run("What is the capital of France?")
    print(result.output)

    # Using a Persona for reusable, templated system prompts:
    from prompture import Persona

    persona = Persona(
        name="analyst",
        system_prompt="You are {{agent_name}}, a data analyst.\\nWorkspace: {{workspace}}",
        variables={"agent_name": "DataBot"},
    )
    agent = Agent("openai/gpt-4o", system_prompt=persona)
    result = agent.run("Summarize the quarterly data.")
"""

from __future__ import annotations

import inspect
import json
import logging
import time
import typing
from collections.abc import Callable, Generator, Iterator
from typing import Any, Generic

from pydantic import BaseModel

from .types import (
    AgentCallbacks,
    AgentResult,
    AgentState,
    AgentStep,
    ApprovalRequired,
    DepsType,
    ModelRetry,
    RunContext,
    StepType,
    StreamEvent,
    StreamEventType,
)
from ..infra.callbacks import DriverCallbacks
from .conversation import Conversation
from ..drivers.base import Driver
from .persona import Persona
from ..infra.session import UsageSession
from ..extraction.tools import clean_json_text
from .tools_schema import ToolDefinition, ToolRegistry

logger = logging.getLogger("prompture.agent")

_OUTPUT_PARSE_MAX_RETRIES = 3
_OUTPUT_GUARDRAIL_MAX_RETRIES = 3


# ------------------------------------------------------------------
# Module-level helpers for RunContext injection
# ------------------------------------------------------------------


def _tool_wants_context(fn: Callable[..., Any]) -> bool:
    """Check whether *fn*'s first parameter is annotated as :class:`RunContext`.

    Uses :func:`typing.get_type_hints` to resolve string annotations
    (from ``from __future__ import annotations``).  Falls back to raw
    ``__annotations__`` when ``get_type_hints`` cannot resolve local types.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    if not params:
        return False

    first_param = params[0]
    if first_param == "self":
        if len(params) < 2:
            return False
        first_param = params[1]

    # Try get_type_hints first (resolves string annotations)
    annotation = None
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
        annotation = hints.get(first_param)
    except Exception:
        # get_type_hints can fail with local/forward references; fall back to raw annotation
        pass

    # Fallback: inspect raw annotation (may be a string)
    if annotation is None:
        raw = sig.parameters[first_param].annotation
        if raw is inspect.Parameter.empty:
            return False
        annotation = raw

    # String annotation: check if it starts with "RunContext"
    if isinstance(annotation, str):
        return annotation == "RunContext" or annotation.startswith("RunContext[")

    # Direct match
    if annotation is RunContext:
        return True

    # Generic alias: RunContext[X]
    origin = getattr(annotation, "__origin__", None)
    return origin is RunContext


def _get_first_param_name(fn: Callable[..., Any]) -> str:
    """Return the name of the first non-self parameter of *fn*."""
    sig = inspect.signature(fn)
    for name, _param in sig.parameters.items():
        if name != "self":
            return name
    return ""


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------


class Agent(Generic[DepsType]):
    """A reusable agent that executes a ReAct loop with tool support.

    Each call to :meth:`run` creates a fresh :class:`Conversation`,
    preventing state leakage between runs.  The Agent itself is a
    template holding model config, tools, and system prompt.

    Args:
        model: Model string in ``"provider/model"`` format.
        driver: Pre-built driver instance (useful for testing).
        tools: Initial tools as a list of callables or a
            :class:`ToolRegistry`.
        system_prompt: System prompt prepended to every run.  May also
            be a callable ``(RunContext) -> str`` for dynamic prompts.
        output_type: Optional Pydantic model class.  When set, the
            final LLM response is parsed and validated against this type.
        max_iterations: Maximum tool-use rounds per run.
        max_cost: Soft budget in USD.  When exceeded, output parse and
            guardrail retries are skipped.
        options: Extra driver options forwarded to every LLM call.
            Common keys include ``"temperature"`` (float, 0.0-2.0),
            ``"max_tokens"`` (int), and ``"top_p"`` (float).
            Example: ``options={"temperature": 0.7, "max_tokens": 1024}``.
        deps_type: Type hint for dependencies (for docs/IDE only).
        agent_callbacks: Agent-level observability callbacks.
        input_guardrails: Functions called before the prompt is sent.
        output_guardrails: Functions called after output is parsed.
        persistent_conversation: When ``True``, subsequent ``run()``
            calls reuse the same :class:`Conversation` so the model
            sees the full multi-turn history.  Use
            :meth:`clear_history` to reset.  Default ``False``
            preserves the existing one-shot behaviour.
        security_context: Optional tukuy ``SecurityContext`` activated
            around tool execution so tukuy skills run with filesystem
            and network scoping.  When ``None`` (default) no scoping
            is applied.
        auto_approve_safe_only: When ``True``, tools backed by tukuy
            skills that declare ``side_effects=True`` or
            ``requires_network=True`` will raise
            :class:`ApprovalRequired` before execution.  Default
            ``False``.
        skill_config: Optional configuration dict injected as a
            :class:`SkillContext` into tukuy skill ``invoke()`` calls.
            When ``None`` (default) no config is injected.
    """

    def __init__(
        self,
        model: str = "",
        *,
        driver: Driver | None = None,
        tools: list[Callable[..., Any]] | ToolRegistry | None = None,
        system_prompt: str | Persona | Callable[..., str] | None = None,
        output_type: type[BaseModel] | None = None,
        max_iterations: int = 10,
        max_cost: float | None = None,
        options: dict[str, Any] | None = None,
        deps_type: type | None = None,
        agent_callbacks: AgentCallbacks | None = None,
        input_guardrails: list[Callable[..., Any]] | None = None,
        output_guardrails: list[Callable[..., Any]] | None = None,
        name: str = "",
        description: str = "",
        output_key: str | None = None,
        persistent_conversation: bool = False,
        security_context: Any | None = None,
        auto_approve_safe_only: bool = False,
        skill_config: dict[str, Any] | None = None,
    ) -> None:
        if not model and driver is None:
            raise ValueError("Either model or driver must be provided")

        self._model = model
        self._driver = driver
        self._system_prompt = system_prompt
        self._output_type = output_type
        self._max_iterations = max_iterations
        self._max_cost = max_cost
        self._options = dict(options) if options else {}
        self._deps_type = deps_type
        self._agent_callbacks = agent_callbacks or AgentCallbacks()
        self._input_guardrails = list(input_guardrails) if input_guardrails else []
        self._output_guardrails = list(output_guardrails) if output_guardrails else []
        self.name = name
        self.description = description
        self.output_key = output_key
        self._persistent_conversation = persistent_conversation
        self._security_context = security_context
        self._auto_approve_safe_only = auto_approve_safe_only
        self._skill_config = skill_config
        self._conversation: Conversation | None = None

        # Build internal tool registry
        self._tools = ToolRegistry()
        if isinstance(tools, ToolRegistry):
            self._tools = tools
        elif tools is not None:
            for fn in tools:
                self._tools.register(fn)

        self._lifecycle = AgentState.idle
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

    def add_tukuy_tools(self, skills: list[Any]) -> None:
        """Register tukuy skills as tools on this agent.

        Args:
            skills: List of tukuy ``Skill`` instances or ``@skill``-decorated functions.
        """
        self._tools.add_tukuy_skills(skills)

    @property
    def state(self) -> AgentState:
        """Current lifecycle state of the agent."""
        return self._lifecycle

    def stop(self) -> None:
        """Request graceful shutdown after the current iteration."""
        self._stop_requested = True

    @property
    def conversation(self) -> Conversation | None:
        """The current persistent conversation, or ``None``."""
        return self._conversation

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Message history from the persistent conversation, or ``[]``."""
        if self._conversation is not None:
            return self._conversation.messages
        return []

    def clear_history(self) -> None:
        """Reset the persistent conversation history."""
        if self._conversation is not None:
            self._conversation.clear()

    def as_tool(
        self,
        name: str | None = None,
        description: str | None = None,
        custom_output_extractor: Callable[[AgentResult], str] | None = None,
    ) -> ToolDefinition:
        """Wrap this Agent as a callable tool for another Agent.

        Creates a :class:`ToolDefinition` whose function accepts a ``prompt``
        string, runs this agent, and returns the output text.

        Args:
            name: Tool name (defaults to ``self.name`` or ``"agent_tool"``).
            description: Tool description (defaults to ``self.description``).
            custom_output_extractor: Optional function to extract a string
                from :class:`AgentResult`.  Defaults to ``result.output_text``.
        """
        tool_name = name or self.name or "agent_tool"
        tool_desc = description or self.description or f"Run agent {tool_name}"
        agent = self
        extractor = custom_output_extractor

        def _call_agent(prompt: str) -> str:
            """Run the wrapped agent with the given prompt."""
            result = agent.run(prompt)
            _call_agent._last_agent_result = result
            if extractor is not None:
                return extractor(result)
            return result.output_text

        _call_agent._source_agent = agent
        _call_agent._last_agent_result = None

        return ToolDefinition(
            name=tool_name,
            description=tool_desc,
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The prompt to send to the agent"},
                },
                "required": ["prompt"],
            },
            function=_call_agent,
        )

    def run(self, prompt: str, *, deps: Any = None) -> AgentResult:
        """Execute the agent loop to completion.

        Creates a fresh :class:`Conversation`, sends the prompt,
        handles any tool calls, and optionally parses the final
        response into an ``output_type`` Pydantic model.

        Args:
            prompt: The user prompt to send.
            deps: Optional dependencies injected into :class:`RunContext`.
        """
        self._lifecycle = AgentState.running
        self._stop_requested = False
        steps: list[AgentStep] = []

        try:
            result = self._execute(prompt, steps, deps)
            self._lifecycle = AgentState.idle
            return result
        except Exception:
            self._lifecycle = AgentState.errored
            raise

    # ------------------------------------------------------------------
    # RunContext helpers
    # ------------------------------------------------------------------

    def _build_run_context(
        self,
        prompt: str,
        deps: Any,
        session: UsageSession,
        messages: list[dict[str, Any]],
        iteration: int,
    ) -> RunContext[Any]:
        """Create a :class:`RunContext` snapshot for the current run."""
        return RunContext(
            deps=deps,
            model=self._model,
            usage=session.summary(),
            messages=list(messages),
            iteration=iteration,
            prompt=prompt,
        )

    # ------------------------------------------------------------------
    # Tool wrapping (RunContext injection + ModelRetry + callbacks)
    # ------------------------------------------------------------------

    def _wrap_tools_with_context(self, ctx: RunContext[Any], session: UsageSession | None = None) -> ToolRegistry:
        """Return a new :class:`ToolRegistry` with wrapped tool functions.

        For each registered tool:
        - If the tool's first param is ``RunContext``, inject *ctx* automatically.
        - Catch :class:`ModelRetry` and convert to an error string.
        - Fire ``agent_callbacks.on_tool_start`` / ``on_tool_end``.
        - Strip the ``RunContext`` parameter from the JSON schema sent to the LLM.
        - If the tool wraps a child agent (via ``as_tool``), aggregate its
          usage into the parent *session*.
        """
        if not self._tools:
            return ToolRegistry()

        new_registry = ToolRegistry()

        cb = self._agent_callbacks

        for td in self._tools.definitions:
            wants_ctx = _tool_wants_context(td.function)
            original_fn = td.function
            tool_name = td.name

            def _make_wrapper(
                _fn: Callable[..., Any],
                _wants: bool,
                _name: str,
                _cb: AgentCallbacks = cb,
                _session: UsageSession | None = session,
            ) -> Callable[..., Any]:
                def wrapper(**kwargs: Any) -> Any:
                    if _cb.on_tool_start:
                        _cb.on_tool_start(_name, kwargs)
                    try:
                        # Inject skill config via SkillContext for tukuy skills
                        if self._skill_config is not None:
                            _skill_obj = getattr(_fn, "__skill__", None)
                            if _skill_obj is not None:
                                from tukuy import SkillContext

                                kwargs["context"] = SkillContext(config=self._skill_config)

                        # Auto-approve gate: block tukuy skills with side effects
                        if self._auto_approve_safe_only:
                            skill_obj = getattr(_fn, "__skill__", None)
                            if skill_obj is not None:
                                desc = skill_obj.descriptor
                                has_side_effects = getattr(desc, "side_effects", False)
                                has_network = getattr(desc, "requires_network", False)
                                if has_side_effects or has_network:
                                    raise ApprovalRequired(
                                        tool_name=_name,
                                        action="execute tool with side effects",
                                        details={
                                            "side_effects": has_side_effects,
                                            "requires_network": has_network,
                                            "skill_name": desc.name,
                                        },
                                    )

                        if _wants:
                            result = _fn(ctx, **kwargs)
                        else:
                            result = _fn(**kwargs)
                    except ApprovalRequired as exc:
                        # Handle approval request
                        if _cb.on_approval_needed:
                            approved = _cb.on_approval_needed(exc.tool_name, exc.action, exc.details)
                            if approved:
                                # Retry the tool call after approval
                                try:
                                    if _wants:
                                        result = _fn(ctx, **kwargs)
                                    else:
                                        result = _fn(**kwargs)
                                except ApprovalRequired:
                                    # Tool raised ApprovalRequired again - don't loop
                                    result = f"Error: Tool '{_name}' requires approval but approval was already granted"
                                except ModelRetry as retry_exc:
                                    result = f"Error: {retry_exc.message}"
                            else:
                                result = f"Error: Tool '{_name}' execution denied - approval required: {exc.action}"
                        else:
                            result = f"Error: Tool '{_name}' requires approval but no approval handler is configured"
                    except ModelRetry as exc:
                        result = f"Error: {exc.message}"

                    # Aggregate child agent usage to parent session
                    if _session is not None and hasattr(_fn, "_source_agent"):
                        agent_result = getattr(_fn, "_last_agent_result", None)
                        if agent_result is not None and hasattr(agent_result, "run_usage"):
                            child_usage = agent_result.run_usage
                            child_name = getattr(_fn._source_agent, "name", "") or _name
                            _session.record(
                                {
                                    "meta": {
                                        "prompt_tokens": child_usage.get("prompt_tokens", 0),
                                        "completion_tokens": child_usage.get("completion_tokens", 0),
                                        "total_tokens": child_usage.get("total_tokens", 0),
                                        "cost": child_usage.get("cost", 0.0),
                                    },
                                    "driver": f"sub-agent:{child_name}",
                                }
                            )

                    if _cb.on_tool_end:
                        _cb.on_tool_end(_name, result)
                    return result

                return wrapper

            wrapped = _make_wrapper(original_fn, wants_ctx, tool_name)

            # Build schema: strip RunContext param if present
            params = dict(td.parameters)
            if wants_ctx:
                ctx_param_name = _get_first_param_name(td.function)
                props = dict(params.get("properties", {}))
                props.pop(ctx_param_name, None)
                params = dict(params)
                params["properties"] = props
                req = list(params.get("required", []))
                if ctx_param_name in req:
                    req.remove(ctx_param_name)
                if req:
                    params["required"] = req
                elif "required" in params:
                    del params["required"]

            new_td = ToolDefinition(
                name=td.name,
                description=td.description,
                parameters=params,
                function=wrapped,
            )
            new_registry.add(new_td)

        return new_registry

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------

    def _run_input_guardrails(self, ctx: RunContext[Any], prompt: str) -> str:
        """Execute input guardrails in order. Returns the (possibly transformed) prompt.

        Each guardrail receives ``(ctx, prompt)`` and may:
        - Return a ``str`` to transform the prompt.
        - Return ``None`` to leave it unchanged.
        - Raise :class:`GuardrailError` to reject entirely.
        """
        for guardrail in self._input_guardrails:
            result = guardrail(ctx, prompt)
            if result is not None:
                prompt = result
        return prompt

    def _run_output_guardrails(
        self,
        ctx: RunContext[Any],
        result: AgentResult,
        conv: Conversation,
        session: UsageSession,
        steps: list[AgentStep],
        all_tool_calls: list[dict[str, Any]],
    ) -> AgentResult:
        """Execute output guardrails. Returns the (possibly modified) result.

        Each guardrail receives ``(ctx, result)`` and may:
        - Return ``None`` to pass (no change).
        - Return an :class:`AgentResult` to modify the result.
        - Raise :class:`ModelRetry` to re-prompt the LLM (up to 3 retries).
        """
        for guardrail in self._output_guardrails:
            for attempt in range(_OUTPUT_GUARDRAIL_MAX_RETRIES):
                try:
                    guard_result = guardrail(ctx, result)
                    if guard_result is not None:
                        result = guard_result
                    break  # guardrail passed
                except ModelRetry as exc:
                    if self._is_over_budget(session):
                        logger.debug("Over budget, skipping output guardrail retry")
                        break
                    if attempt >= _OUTPUT_GUARDRAIL_MAX_RETRIES - 1:
                        raise ValueError(
                            f"Output guardrail failed after {_OUTPUT_GUARDRAIL_MAX_RETRIES} retries: {exc.message}"
                        ) from exc
                    # Re-prompt the LLM
                    retry_text = conv.ask(
                        f"Your response did not pass validation. Error: {exc.message}\n\nPlease try again."
                    )
                    self._extract_steps(conv.messages[-2:], steps, all_tool_calls)

                    # Re-parse if output_type is set
                    if self._output_type is not None:
                        try:
                            cleaned = clean_json_text(retry_text)
                            parsed = json.loads(cleaned)
                            output = self._output_type.model_validate(parsed)
                        except Exception:
                            output = retry_text
                    else:
                        output = retry_text

                    result = AgentResult(
                        output=output,
                        output_text=retry_text,
                        messages=conv.messages,
                        usage=conv.usage,
                        steps=steps,
                        all_tool_calls=all_tool_calls,
                        state=AgentState.idle,
                        run_usage=session.summary(),
                    )
        return result

    # ------------------------------------------------------------------
    # Budget check
    # ------------------------------------------------------------------

    def _is_over_budget(self, session: UsageSession) -> bool:
        """Return True if max_cost is set and the session has exceeded it."""
        if self._max_cost is None:
            return False
        return session.cost >= self._max_cost

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_system_prompt(self, ctx: RunContext[Any] | None = None) -> str | None:
        """Build the system prompt, appending output schema if needed."""
        parts: list[str] = []

        if self._system_prompt is not None:
            if isinstance(self._system_prompt, Persona):
                # Render Persona with RunContext variables if available
                render_kwargs: dict[str, Any] = {}
                if ctx is not None:
                    render_kwargs["model"] = ctx.model
                    render_kwargs["iteration"] = ctx.iteration
                parts.append(self._system_prompt.render(**render_kwargs))
            elif callable(self._system_prompt) and not isinstance(self._system_prompt, str):
                if ctx is not None:
                    parts.append(self._system_prompt(ctx))
                else:
                    # Fallback: call without context (shouldn't happen in normal flow)
                    parts.append(self._system_prompt(None))  # type: ignore[arg-type]
            else:
                parts.append(str(self._system_prompt))

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

    def _build_conversation(
        self,
        system_prompt: str | None = None,
        tools: ToolRegistry | None = None,
        driver_callbacks: DriverCallbacks | None = None,
    ) -> Conversation:
        """Create or reuse a Conversation for a run.

        When ``persistent_conversation=True`` and a conversation already
        exists, it is reused (with updated tools and callbacks).
        Otherwise a fresh :class:`Conversation` is created.
        """
        # Reuse existing conversation in persistent mode
        if self._persistent_conversation and self._conversation is not None:
            conv = self._conversation
            # Update tools and callbacks for this run
            if tools is not None:
                conv._tools = tools
            if driver_callbacks is not None:
                conv._driver.callbacks = driver_callbacks
            return conv

        effective_tools = tools if tools is not None else (self._tools if self._tools else None)

        kwargs: dict[str, Any] = {
            "system_prompt": system_prompt if system_prompt is not None else self._resolve_system_prompt(),
            "tools": effective_tools,
            "max_tool_rounds": self._max_iterations,
        }
        if self._options:
            kwargs["options"] = self._options
        if driver_callbacks is not None:
            kwargs["callbacks"] = driver_callbacks

        if self._driver is not None:
            kwargs["driver"] = self._driver
        else:
            kwargs["model_name"] = self._model

        conv = Conversation(**kwargs)
        if self._persistent_conversation:
            self._conversation = conv
        return conv

    def _execute(self, prompt: str, steps: list[AgentStep], deps: Any) -> AgentResult:
        """Core execution: run conversation, extract steps, parse output."""
        from ..infra.tracker import get_tracker

        tracker = get_tracker()

        # 1. Create per-run UsageSession and wire into DriverCallbacks
        session = UsageSession()
        driver_callbacks = DriverCallbacks(
            on_response=session.record,
            on_error=session.record_error,
        )

        # 2. Build initial RunContext
        ctx = self._build_run_context(prompt, deps, session, [], 0)

        # 3. Run input guardrails
        effective_prompt = self._run_input_guardrails(ctx, prompt)

        # 4. Resolve system prompt (call it if callable, passing ctx)
        resolved_system_prompt = self._resolve_system_prompt(ctx)

        # 5. Wrap tools with context (pass session for child agent usage aggregation)
        wrapped_tools = self._wrap_tools_with_context(ctx, session)

        # 6. Build Conversation
        conv = self._build_conversation(
            system_prompt=resolved_system_prompt,
            tools=wrapped_tools if wrapped_tools else None,
            driver_callbacks=driver_callbacks,
        )

        # 7. Fire on_iteration callback
        if self._agent_callbacks.on_iteration:
            self._agent_callbacks.on_iteration(0)

        # 8. Ask the conversation (handles full tool loop internally)
        agent_name = self.name or self.__class__.__name__
        with tracker.agent(agent_name):
            t0 = time.perf_counter()
            security_token = None
            if self._security_context is not None:
                from tukuy.safety import set_security_context

                security_token = set_security_context(self._security_context)
            try:
                response_text = conv.ask(effective_prompt)
            finally:
                if security_token is not None:
                    from tukuy.safety import reset_security_context

                    reset_security_context(security_token)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # 9. Extract steps and tool calls from conversation messages
            all_tool_calls: list[dict[str, Any]] = []
            self._extract_steps(conv.messages, steps, all_tool_calls)

            # Handle output_type parsing
            if self._output_type is not None:
                output, output_text = self._parse_output(conv, response_text, steps, all_tool_calls, elapsed_ms, session)
            else:
                output = response_text
                output_text = response_text

            # Build result with run_usage
            result = AgentResult(
                output=output,
                output_text=output_text,
                messages=conv.messages,
                usage=conv.usage,
                steps=steps,
                all_tool_calls=all_tool_calls,
                state=AgentState.idle,
                run_usage=session.summary(),
            )

            # 10. Run output guardrails
            if self._output_guardrails:
                result = self._run_output_guardrails(ctx, result, conv, session, steps, all_tool_calls)

            # 11. Fire callbacks
            if self._agent_callbacks.on_step:
                for step in steps:
                    self._agent_callbacks.on_step(step)

            if self._agent_callbacks.on_output:
                self._agent_callbacks.on_output(result)

            if self._agent_callbacks.on_message:
                self._agent_callbacks.on_message(result.output_text)

            return result

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
            # Extract usage from message meta if present
            msg_meta = msg.get("meta")
            step_usage = None
            if isinstance(msg_meta, dict):
                step_usage = {
                    "prompt_tokens": msg_meta.get("prompt_tokens", 0),
                    "completion_tokens": msg_meta.get("completion_tokens", 0),
                    "total_tokens": msg_meta.get("total_tokens", 0),
                    "cost": msg_meta.get("cost", 0.0),
                }

            if role == "assistant":
                content = msg.get("content", "") or ""

                # Extract thinking content from <think> tags
                thinking_text = self._extract_thinking(content)
                if thinking_text and self._agent_callbacks.on_thinking:
                    self._agent_callbacks.on_thinking(thinking_text)
                    # Also record as a think step
                    steps.append(
                        AgentStep(
                            step_type=StepType.think,
                            timestamp=now,
                            content=thinking_text,
                            usage=step_usage,
                        )
                    )

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
                                content=content,
                                tool_name=name,
                                tool_args=args,
                                usage=step_usage,
                            )
                        )
                        all_tool_calls.append({"name": name, "arguments": args, "id": tc.get("id", "")})
                else:
                    # Final assistant message (no tool calls)
                    steps.append(
                        AgentStep(
                            step_type=StepType.output,
                            timestamp=now,
                            content=content,
                            usage=step_usage,
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

    def _extract_thinking(self, content: str) -> str | None:
        """Extract thinking content from <think> tags.

        Some models (like DeepSeek, Qwen) emit chain-of-thought reasoning
        within <think>...</think> tags. This method extracts that content.

        Args:
            content: The assistant message content.

        Returns:
            The thinking text if found, None otherwise.
        """
        import re

        # Match <think>...</think> tags (case-insensitive, allows multiline)
        pattern = r"<think>(.*?)</think>"
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        if matches:
            # Join multiple thinking blocks with newlines
            return "\n".join(match.strip() for match in matches)
        return None

    def _parse_output(
        self,
        conv: Conversation,
        response_text: str,
        steps: list[AgentStep],
        all_tool_calls: list[dict[str, Any]],
        elapsed_ms: float,
        session: UsageSession | None = None,
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
                    # Check budget before retrying
                    if session is not None and self._is_over_budget(session):
                        logger.debug("Over budget, skipping output parse retry")
                        break
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

    # ------------------------------------------------------------------
    # iter() — step-by-step inspection
    # ------------------------------------------------------------------

    def iter(self, prompt: str, *, deps: Any = None) -> AgentIterator:
        """Execute the agent loop and iterate over steps.

        Returns an :class:`AgentIterator` that yields :class:`AgentStep`
        objects.  After iteration completes, the final :class:`AgentResult`
        is available via :attr:`AgentIterator.result`.

        Note:
            In Phase 3c the conversation's tool loop runs to completion
            before steps are yielded.  True mid-loop yielding is deferred.
        """
        gen = self._execute_iter(prompt, deps)
        return AgentIterator(gen)

    def _execute_iter(self, prompt: str, deps: Any) -> Generator[AgentStep, None, AgentResult]:
        """Generator that executes the agent loop and yields each step."""
        self._lifecycle = AgentState.running
        self._stop_requested = False
        steps: list[AgentStep] = []

        try:
            result = self._execute(prompt, steps, deps)
            # Yield each step one at a time
            yield from result.steps
            self._lifecycle = AgentState.idle
            return result
        except Exception:
            self._lifecycle = AgentState.errored
            raise

    # ------------------------------------------------------------------
    # run_stream() — streaming output
    # ------------------------------------------------------------------

    def run_stream(self, prompt: str, *, deps: Any = None) -> StreamedAgentResult:
        """Execute the agent loop with streaming output.

        Returns a :class:`StreamedAgentResult` that yields
        :class:`StreamEvent` objects.  After iteration completes, the
        final :class:`AgentResult` is available via
        :attr:`StreamedAgentResult.result`.

        When tools are registered, streaming falls back to non-streaming
        ``conv.ask()`` and yields the full response as a single
        ``text_delta`` event.
        """
        gen = self._execute_stream(prompt, deps)
        return StreamedAgentResult(gen)

    def _execute_stream(self, prompt: str, deps: Any) -> Generator[StreamEvent, None, AgentResult]:
        """Generator that executes the agent loop and yields stream events."""
        self._lifecycle = AgentState.running
        self._stop_requested = False
        steps: list[AgentStep] = []

        try:
            # 1. Create per-run UsageSession and wire into DriverCallbacks
            session = UsageSession()
            driver_callbacks = DriverCallbacks(
                on_response=session.record,
                on_error=session.record_error,
            )

            # 2. Build initial RunContext
            ctx = self._build_run_context(prompt, deps, session, [], 0)

            # 3. Run input guardrails
            effective_prompt = self._run_input_guardrails(ctx, prompt)

            # 4. Resolve system prompt
            resolved_system_prompt = self._resolve_system_prompt(ctx)

            # 5. Wrap tools with context (pass session for child agent usage aggregation)
            wrapped_tools = self._wrap_tools_with_context(ctx, session)
            has_tools = bool(wrapped_tools)

            # 6. Build Conversation
            conv = self._build_conversation(
                system_prompt=resolved_system_prompt,
                tools=wrapped_tools if wrapped_tools else None,
                driver_callbacks=driver_callbacks,
            )

            # 7. Fire on_iteration callback
            if self._agent_callbacks.on_iteration:
                self._agent_callbacks.on_iteration(0)

            security_token = None
            if self._security_context is not None:
                from tukuy.safety import set_security_context

                security_token = set_security_context(self._security_context)
            try:
                if has_tools:
                    # Tools registered: stream tool events and final response
                    response_text = ""
                    for event in conv.ask_with_tool_events(effective_prompt):
                        if event["type"] == "tool_call":
                            yield StreamEvent(
                                event_type=StreamEventType.tool_call,
                                data=event,
                            )
                        elif event["type"] == "tool_result":
                            yield StreamEvent(
                                event_type=StreamEventType.tool_result,
                                data=event,
                            )
                        elif event["type"] == "text_delta":
                            response_text += event["text"]
                            yield StreamEvent(
                                event_type=StreamEventType.text_delta,
                                data=event["text"],
                            )
                else:
                    # No tools: use streaming if available
                    response_text = ""
                    stream_iter: Iterator[str] = conv.ask_stream(effective_prompt)
                    for chunk in stream_iter:
                        response_text += chunk
                        yield StreamEvent(
                            event_type=StreamEventType.text_delta,
                            data=chunk,
                        )
            finally:
                if security_token is not None:
                    from tukuy.safety import reset_security_context

                    reset_security_context(security_token)

            # 8. Extract steps
            all_tool_calls: list[dict[str, Any]] = []
            self._extract_steps(conv.messages, steps, all_tool_calls)

            # 9. Parse output
            if self._output_type is not None:
                output, output_text = self._parse_output(conv, response_text, steps, all_tool_calls, 0.0, session)
            else:
                output = response_text
                output_text = response_text

            # 10. Build result
            result = AgentResult(
                output=output,
                output_text=output_text,
                messages=conv.messages,
                usage=conv.usage,
                steps=steps,
                all_tool_calls=all_tool_calls,
                state=AgentState.idle,
                run_usage=session.summary(),
            )

            # 11. Run output guardrails
            if self._output_guardrails:
                result = self._run_output_guardrails(ctx, result, conv, session, steps, all_tool_calls)

            # 12. Fire callbacks
            if self._agent_callbacks.on_step:
                for step in steps:
                    self._agent_callbacks.on_step(step)
            if self._agent_callbacks.on_output:
                self._agent_callbacks.on_output(result)

            if self._agent_callbacks.on_message:
                self._agent_callbacks.on_message(result.output_text)

            # 13. Yield final output event
            yield StreamEvent(
                event_type=StreamEventType.output,
                data=result,
            )

            self._lifecycle = AgentState.idle
            return result
        except Exception:
            self._lifecycle = AgentState.errored
            raise


# ------------------------------------------------------------------
# AgentIterator
# ------------------------------------------------------------------


class AgentIterator:
    """Wraps the :meth:`Agent.iter` generator, capturing the final result.

    After iteration completes (the ``for`` loop ends), the
    :attr:`result` property holds the :class:`AgentResult`.
    """

    def __init__(self, gen: Generator[AgentStep, None, AgentResult]) -> None:
        self._gen = gen
        self._result: AgentResult | None = None

    def __iter__(self) -> AgentIterator:
        return self

    def __next__(self) -> AgentStep:
        try:
            return next(self._gen)
        except StopIteration as e:
            self._result = e.value
            raise

    @property
    def result(self) -> AgentResult | None:
        """The final :class:`AgentResult`, available after iteration completes."""
        return self._result


# ------------------------------------------------------------------
# StreamedAgentResult
# ------------------------------------------------------------------


class StreamedAgentResult:
    """Wraps the :meth:`Agent.run_stream` generator, capturing the final result.

    Yields :class:`StreamEvent` objects during iteration.  After iteration
    completes, the :attr:`result` property holds the :class:`AgentResult`.
    """

    def __init__(self, gen: Generator[StreamEvent, None, AgentResult]) -> None:
        self._gen = gen
        self._result: AgentResult | None = None

    def __iter__(self) -> StreamedAgentResult:
        return self

    def __next__(self) -> StreamEvent:
        try:
            return next(self._gen)
        except StopIteration as e:
            self._result = e.value
            raise

    @property
    def result(self) -> AgentResult | None:
        """The final :class:`AgentResult`, available after iteration completes."""
        return self._result
