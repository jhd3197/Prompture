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

import asyncio
import concurrent.futures
import contextvars
import inspect
import json
import logging
import time
import typing
from collections.abc import Callable, Generator
from typing import Any, Generic

from pydantic import BaseModel

from ..drivers.base import Driver
from ..extraction.tools import clean_json_text
from ..infra.budget import BudgetPolicy, resolve_budget_policy
from ..infra.callbacks import DriverCallbacks
from ..infra.provider_env import ProviderEnvironment
from ..infra.session import UsageSession
from .conversation import Conversation
from .persona import Persona
from .tools_schema import ToolDefinition, ToolRegistry
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

logger = logging.getLogger("prompture.agent")

_OUTPUT_PARSE_MAX_RETRIES = 3
_OUTPUT_GUARDRAIL_MAX_RETRIES = 3
_DEFAULT_MAX_AGENT_DEPTH = 5

# Global recursion depth counter — inherited by child agents via contextvars
_agent_depth: contextvars.ContextVar[int] = contextvars.ContextVar("_agent_depth", default=0)


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


def _parse_simulated_tool_call(content: str) -> tuple[str, dict[str, Any]] | None:
    """Recognise a prompted (simulated) tool call recorded as assistant text.

    Drivers without ``supports_tool_use`` go through
    :meth:`Conversation._ask_with_simulated_tools`, which records the round as
    a plain assistant message whose content is the protocol JSON, followed by a
    plain user message holding the result.  Neither carries ``tool_calls`` nor
    the ``tool`` role, so a reader that only understands the native shape
    reports the whole run as model output and leaks this JSON as the answer.

    Returns ``(name, arguments)`` or ``None`` when *content* is ordinary prose.
    """
    text = clean_json_text(content or "").strip()
    if not text.startswith("{"):
        return None
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    # Explicit discriminator, or the inferred shape parse_simulated_response
    # also accepts when the model omits "type".
    if obj.get("type") != "tool_call" and not ("name" in obj and "arguments" in obj):
        return None
    name = obj.get("name")
    args = obj.get("arguments", {})
    if not isinstance(name, str) or not name or not isinstance(args, dict):
        return None
    return name, args


def _invoke_cb_sync(callback: Callable[..., Any], *cb_args: Any) -> Any:
    """Invoke *callback* from sync code, driving awaitables to completion.

    Async callbacks (e.g. ``async def on_approval_needed(...)``) return a
    coroutine when called bare — which is always truthy and previously
    caused silent auto-approval.  This bridge detects awaitables and runs
    them: directly with :func:`asyncio.run` when no loop is running, or on
    a worker thread (carrying over ``contextvars``) when a loop is already
    running in the current thread.
    """
    result = callback(*cb_args)
    if not inspect.isawaitable(result):
        return result

    async def _await_result() -> Any:
        return await result

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        ctx_snapshot = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(ctx_snapshot.run, lambda: asyncio.run(_await_result())).result()
    return asyncio.run(_await_result())


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
        tool_timeout: Per-tool wall-clock budget in seconds.  A tool that
            overruns it yields an error result the model can react to
            instead of wedging the run forever.  ``None`` (default) means
            no timeout.  Can be overridden per call via
            ``options={"tool_timeout": ...}``.
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
        max_tokens: int | None = None,
        budget_policy: BudgetPolicy | str | None = None,
        fallback_models: list[str] | None = None,
        on_model_fallback: Callable[..., Any] | None = None,
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
        max_tool_result_length: int | None = None,
        tool_timeout: float | None = None,
        max_depth: int = _DEFAULT_MAX_AGENT_DEPTH,
        env: ProviderEnvironment | None = None,
    ) -> None:
        if not model and driver is None:
            raise ValueError("Either model or driver must be provided")

        self._model = model
        self._driver = driver
        self._env = env
        self._max_depth = max_depth
        self._system_prompt = system_prompt
        self._output_type = output_type
        self._max_iterations = max_iterations
        self._max_cost = max_cost
        self._max_tokens = max_tokens
        self._budget_policy = resolve_budget_policy(budget_policy)
        self._fallback_models = list(fallback_models) if fallback_models else None
        self._on_model_fallback = on_model_fallback
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
        self._max_tool_result_length = max_tool_result_length
        self._tool_timeout = tool_timeout
        self._conversation: Conversation | None = None
        # The conversation driving the current run.  Tracked separately from
        # ``_conversation`` (which only holds persistent ones) so that
        # ``stop()`` can reach the loop of a non-persistent run too.
        self._active_conversation: Conversation | None = None

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
        """Request graceful shutdown after the current iteration.

        Sets the agent-level flag and forwards the request to the
        conversation driving the current run (cooperative stop) so its
        tool-round loop exits gracefully between rounds instead of starting
        another round.  Works for non-persistent agents too: the in-flight
        conversation is tracked in ``_active_conversation`` for the duration
        of the run, so the flag is never merely decorative.

        Safe to call from a tool, a callback, or another thread.
        """
        self._stop_requested = True
        conv = self._active_conversation or self._conversation
        if conv is not None:
            conv.request_stop()

    @property
    def callbacks(self) -> AgentCallbacks:
        """Current agent-level callbacks."""
        return self._agent_callbacks

    @callbacks.setter
    def callbacks(self, value: AgentCallbacks) -> None:
        """Set agent-level callbacks."""
        self._agent_callbacks = value

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
            _call_agent._last_agent_result = result  # type: ignore[attr-defined]
            if extractor is not None:
                return extractor(result)
            return result.output_text

        _call_agent._source_agent = agent  # type: ignore[attr-defined]
        _call_agent._last_agent_result = None  # type: ignore[attr-defined]

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

        Raises:
            RecursionError: If the agent nesting depth exceeds ``max_depth``.
        """
        current_depth = _agent_depth.get()
        if current_depth >= self._max_depth:
            raise RecursionError(f"Agent recursion depth exceeded: {current_depth} >= {self._max_depth}")
        token = _agent_depth.set(current_depth + 1)
        self._lifecycle = AgentState.running
        self._stop_requested = False
        self._active_conversation = None
        steps: list[AgentStep] = []

        try:
            result = self._execute(prompt, steps, deps)
            self._lifecycle = AgentState.idle
            return result
        except Exception:
            self._lifecycle = AgentState.errored
            raise
        finally:
            _agent_depth.reset(token)

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

    def _make_live_ctx_fn(
        self,
        prompt: str,
        deps: Any,
        session: UsageSession,
        run_state: dict[str, Any],
    ) -> Callable[[], RunContext[Any]]:
        """Return a factory that builds a fresh :class:`RunContext` per call.

        The factory reads the current conversation from *run_state* (key
        ``"conv"``) so tools invoked in later tool rounds see live
        ``iteration``/``messages``/``usage`` instead of the snapshot taken
        at the start of the run.  ``run_state["conv"]`` must be assigned
        once the conversation is built; before that, an empty context is
        produced.
        """

        def _live_ctx() -> RunContext[Any]:
            conv = run_state.get("conv")
            messages = conv.messages if conv is not None else []
            iteration = max(
                0,
                sum(1 for m in messages if m.get("role") == "assistant" and m.get("tool_calls")) - 1,
            )
            return self._build_run_context(prompt, deps, session, messages, iteration)

        return _live_ctx

    # ------------------------------------------------------------------
    # Tool wrapping (RunContext injection + ModelRetry + callbacks)
    # ------------------------------------------------------------------

    def _wrap_tools_with_context(
        self,
        ctx: RunContext[Any],
        session: UsageSession | None = None,
        ctx_fn: Callable[[], RunContext[Any]] | None = None,
        tool_timings: list[dict[str, Any]] | None = None,
    ) -> ToolRegistry:
        """Return a new :class:`ToolRegistry` with wrapped tool functions.

        For each registered tool:
        - If the tool's first param is ``RunContext``, inject the live context
          automatically.  When *ctx_fn* is provided it is called at each tool
          invocation so tools see the current iteration/messages; otherwise the
          static *ctx* snapshot is used.
        - Catch :class:`ModelRetry` and convert to an error string.
        - Fire ``agent_callbacks.on_tool_start`` / ``on_tool_end`` (awaitable
          callbacks are driven to completion via :func:`_invoke_cb_sync`).
        - Strip the ``RunContext`` parameter from the JSON schema sent to the LLM.
        - If the tool wraps a child agent (via ``as_tool``), aggregate its
          usage into the parent *session*.
        - When *tool_timings* is provided, append a
          ``{"name", "timestamp", "duration_ms"}`` record per invocation so
          step extraction can populate ``AgentStep.duration_ms``.
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
                _ctx_fn: Callable[[], RunContext[Any]] | None = ctx_fn,
                _timings: list[dict[str, Any]] | None = tool_timings,
            ) -> Callable[..., Any]:
                def wrapper(**kwargs: Any) -> Any:
                    call_ctx = _ctx_fn() if _ctx_fn is not None else ctx
                    start = time.perf_counter()
                    if _cb.on_tool_start:
                        _invoke_cb_sync(_cb.on_tool_start, _name, kwargs)
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
                            result = _fn(call_ctx, **kwargs)
                        else:
                            result = _fn(**kwargs)
                    except ApprovalRequired as exc:
                        # Handle approval request
                        if _cb.on_approval_needed:
                            approved = _invoke_cb_sync(_cb.on_approval_needed, exc.tool_name, exc.action, exc.details)
                            if approved:
                                # Retry the tool call after approval
                                try:
                                    if _wants:
                                        result = _fn(call_ctx, **kwargs)
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

                    if _timings is not None:
                        _timings.append(
                            {
                                "name": _name,
                                "timestamp": time.time(),
                                "duration_ms": (time.perf_counter() - start) * 1000,
                            }
                        )

                    if _cb.on_tool_end:
                        _invoke_cb_sync(_cb.on_tool_end, _name, result)
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
                # Carry host annotations across the wrap. Without this the field
                # is silently emptied before any tool reaches the agent loop, so
                # a host cannot gate execution on metadata it set itself.
                metadata=td.metadata,
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
                    self._extract_steps(
                        conv.messages[-2:],
                        steps,
                        all_tool_calls,
                        getattr(conv, "_full_tool_results", None),
                    )

                    # Re-parse if output_type is set
                    output: Any
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
        """Return True if max_cost or max_tokens is set and the session has exceeded it."""
        if self._max_cost is not None and session.cost >= self._max_cost:
            return True
        return self._max_tokens is not None and session.total_tokens >= self._max_tokens

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_system_prompt(self, ctx: RunContext[Any] | None = None) -> str | None:
        """Build the system prompt, appending output schema if needed.

        Assembly order (stable across iterations to maximise prompt-cache hits
        on Anthropic / OpenAI auto-caching providers):

        1. **persona / system_prompt content** — first, so it forms the cache
           prefix. The Persona ``render(model=..., iteration=...)`` call only
           substitutes placeholders the template actually references; a persona
           template that includes ``{{iteration}}`` will rotate the cache key
           every turn, defeating caching. Avoid per-turn variables in personas
           if you want cache reuse.
        2. **JSON output schema instruction** — second. Stable across calls
           because ``self._output_type`` is fixed at Agent construction.

        Variable per-call content (user query, retrieved RAG context, etc.)
        belongs in the message stream, not the system prompt.
        """
        parts: list[str] = []

        if self._system_prompt is not None:
            if isinstance(self._system_prompt, Persona):
                # Render Persona with RunContext variables if available.
                # NOTE: ``iteration`` makes the system prompt change every turn
                # if the persona template references it — that's intentional
                # for advanced cases but it breaks prompt caching. See docstring.
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
                    parts.append(self._system_prompt(None))
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
            # If the agent's driver was swapped after the conversation was
            # built (e.g. a budget fallback forced a rebuild), the cached
            # conversation still holds the old driver — discard it and fall
            # through to build a fresh one.
            driver_stale = self._driver is not None and getattr(conv, "_driver", None) is not self._driver
            if not driver_stale:
                # Respect the newly resolved system prompt for this run.
                if system_prompt is not None:
                    conv.system_prompt = system_prompt
                # Update tools and callbacks for this run.
                # NOTE: assigning ``callbacks`` mutates the driver instance,
                # which may be shared (e.g. passed to several agents).  A
                # persistent-conversation agent should treat its driver as
                # exclusively owned by the agent.
                if tools is not None:
                    conv._tools = tools
                if driver_callbacks is not None:
                    conv._driver.callbacks = driver_callbacks
                # Propagate before_turn hook (subclasses may set this)
                hook = getattr(self, "_before_turn_hook", None)
                if hook is not None:
                    conv._before_turn = hook
                return self._track_active(conv)
            self._conversation = None

        effective_tools = tools if tools is not None else (self._tools if self._tools else None)

        kwargs: dict[str, Any] = {
            "system_prompt": system_prompt if system_prompt is not None else self._resolve_system_prompt(),
            "tools": effective_tools,
            "max_tool_rounds": self._max_iterations,
        }
        # Propagate before_turn hook from subclasses (e.g. DeepAgent)
        hook = getattr(self, "_before_turn_hook", None)
        if hook is not None:
            kwargs["before_turn"] = hook
        if self._max_tool_result_length is not None:
            kwargs["max_tool_result_length"] = self._max_tool_result_length
        if self._tool_timeout is not None:
            kwargs["tool_timeout"] = self._tool_timeout
        if self._options:
            kwargs["options"] = self._options
        if driver_callbacks is not None:
            kwargs["callbacks"] = driver_callbacks

        if self._driver is not None:
            kwargs["driver"] = self._driver
        else:
            kwargs["model_name"] = self._model
            if self._env is not None:
                kwargs["env"] = self._env

        # Forward budget params when budget enforcement is active
        if self._budget_policy is not None:
            kwargs["budget_policy"] = self._budget_policy
            if self._max_cost is not None:
                kwargs["max_cost"] = self._max_cost
            if self._max_tokens is not None:
                kwargs["max_tokens"] = self._max_tokens
            if self._fallback_models is not None:
                kwargs["fallback_models"] = self._fallback_models
            if self._on_model_fallback is not None:
                kwargs["on_model_fallback"] = self._on_model_fallback

        conv = Conversation(**kwargs)
        if self._persistent_conversation:
            self._conversation = conv
        return self._track_active(conv)

    def _track_active(self, conv: Conversation) -> Conversation:
        """Record *conv* as the conversation driving the current run.

        Lets :meth:`stop` reach the tool-round loop of a non-persistent run.
        If ``stop()`` was already called between the run starting and the
        conversation being built, the request is replayed onto *conv* so it
        is not lost to that race.
        """
        self._active_conversation = conv
        if self._stop_requested:
            conv.request_stop()
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

        # 5. Wrap tools with context (pass session for child agent usage
        #    aggregation; ctx_fn refreshes RunContext per tool round)
        run_state: dict[str, Any] = {}
        tool_timings: list[dict[str, Any]] = []
        wrapped_tools = self._wrap_tools_with_context(
            ctx,
            session,
            ctx_fn=self._make_live_ctx_fn(prompt, deps, session, run_state),
            tool_timings=tool_timings,
        )

        # 6. Build Conversation
        conv = self._build_conversation(
            system_prompt=resolved_system_prompt,
            tools=wrapped_tools if wrapped_tools else None,
            driver_callbacks=driver_callbacks,
        )
        run_state["conv"] = conv

        # 7. Fire on_iteration callback
        if self._agent_callbacks.on_iteration:
            self._agent_callbacks.on_iteration(0)

        # 8. Ask the conversation (handles full tool loop internally)
        #    Note: budget policy is enforced inside the Conversation
        #    (``_check_budget``); the agent-level pre-call check was removed
        #    because it read a fresh, always-empty UsageSession.
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

            # 10. Extract steps and tool calls from conversation messages
            all_tool_calls: list[dict[str, Any]] = []
            self._extract_steps(
                conv.messages,
                steps,
                all_tool_calls,
                getattr(conv, "_full_tool_results", None),
                tool_timings,
            )

            # Handle output_type parsing
            if self._output_type is not None:
                output, output_text = self._parse_output(
                    conv, response_text, steps, all_tool_calls, elapsed_ms, session
                )
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

            # 11. Run output guardrails
            if self._output_guardrails:
                result = self._run_output_guardrails(ctx, result, conv, session, steps, all_tool_calls)

            # 12. Fire callbacks
            if self._agent_callbacks.on_thinking:
                for step in steps:
                    if step.step_type == StepType.think and step.content:
                        self._agent_callbacks.on_thinking(step.content)
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
        full_tool_results: dict[str, str] | None = None,
        tool_timings: list[dict[str, Any]] | None = None,
    ) -> None:
        """Scan conversation messages and populate steps and tool_calls.

        Args:
            messages: Conversation messages to scan.
            steps: List to append :class:`AgentStep` records to.
            all_tool_calls: List to append tool-call dicts to.
            full_tool_results: Optional map of tool_call_id to the full
                (pre-truncation) tool result string.
            tool_timings: Optional list of ``{"name", "timestamp",
                "duration_ms"}`` records captured by the tool wrappers.
                Consumed in FIFO order per tool name to populate
                ``duration_ms`` and a real execution timestamp on
                ``tool_result`` steps.
        """

        now = time.time()

        # Map tool_call_id -> tool name from assistant tool_calls messages so
        # tool_result steps record the real tool name, not the call id.
        id_to_name: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    tc_fn = tc.get("function", {})
                    tc_name = tc_fn.get("name", tc.get("name", ""))
                    tc_id = tc.get("id", "")
                    if tc_id and tc_name:
                        id_to_name[tc_id] = tc_name

        # FIFO queues of timing records per tool name
        timings_by_name: dict[str, list[dict[str, Any]]] = {}
        for rec in tool_timings or []:
            timings_by_name.setdefault(rec.get("name", ""), []).append(rec)

        # Name of a simulated tool call awaiting its result, which the prompted
        # loop records as the next user message rather than a "tool" message.
        pending_sim_tool: str | None = None

        for msg in messages:
            role = msg.get("role", "")
            # Claim (and clear) any simulated call left by the previous message.
            sim_tool = pending_sim_tool
            pending_sim_tool = None
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
                if thinking_text:
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
                    simulated = _parse_simulated_tool_call(content)
                    if simulated is not None:
                        # Prompted tool call: same step shape as the native path
                        # so a host sees one run, not two dialects.
                        sim_name, sim_args = simulated
                        steps.append(
                            AgentStep(
                                step_type=StepType.tool_call,
                                timestamp=now,
                                content="",
                                tool_name=sim_name,
                                tool_args=sim_args,
                                usage=step_usage,
                            )
                        )
                        all_tool_calls.append({"name": sim_name, "arguments": sim_args, "id": ""})
                        pending_sim_tool = sim_name
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

            elif role == "user" and sim_tool is not None:
                # The prompted loop stores a tool result as a user message.
                steps.append(
                    AgentStep(
                        step_type=StepType.tool_result,
                        timestamp=now,
                        content=msg.get("content", "") or "",
                        tool_name=sim_tool,
                    )
                )

            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                full_result = None
                if full_tool_results and tool_call_id:
                    full_result = full_tool_results.get(tool_call_id)
                # Resolve the real tool name from the originating assistant
                # tool_calls message (fall back to the raw id).
                tool_name = id_to_name.get(tool_call_id, tool_call_id)
                # Pop the matching timing record (FIFO) if available.
                timing = None
                if tool_name is not None:
                    queue = timings_by_name.get(tool_name)
                    if queue:
                        timing = queue.pop(0)
                steps.append(
                    AgentStep(
                        step_type=StepType.tool_result,
                        timestamp=timing["timestamp"] if timing else now,
                        content=msg.get("content", ""),
                        tool_name=tool_name,
                        tool_result=full_result,
                        duration_ms=timing["duration_ms"] if timing else 0.0,
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
                    self._extract_steps(
                        conv.messages[-2:],
                        steps,
                        all_tool_calls,
                        getattr(conv, "_full_tool_results", None),
                    )

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
        """Generator that executes the agent loop and yields each step.

        Raises:
            RecursionError: If the agent nesting depth exceeds ``max_depth``.
        """
        current_depth = _agent_depth.get()
        if current_depth >= self._max_depth:
            raise RecursionError(f"Agent recursion depth exceeded: {current_depth} >= {self._max_depth}")
        token = _agent_depth.set(current_depth + 1)
        self._lifecycle = AgentState.running
        self._stop_requested = False
        self._active_conversation = None
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
        finally:
            _agent_depth.reset(token)

    # ------------------------------------------------------------------
    # run_stream() — streaming output
    # ------------------------------------------------------------------

    def run_stream(self, prompt: str, *, deps: Any = None) -> StreamedAgentResult:
        """Execute the agent loop with streaming output.

        Returns a :class:`StreamedAgentResult` that yields
        :class:`StreamEvent` objects.  After iteration completes, the
        final :class:`AgentResult` is available via
        :attr:`StreamedAgentResult.result`.

        When tools are registered, the tool loop runs via
        ``conv.ask_with_tool_events()``: ``tool_call`` and ``tool_result``
        events are emitted as tools execute, and the final LLM response is
        yielded as a single ``text_delta`` event (per-turn text is not
        token-streamed in this mode).  Without tools, the driver's native
        streaming is used when available.
        """
        gen = self._execute_stream(prompt, deps)
        return StreamedAgentResult(gen)

    def _execute_stream(self, prompt: str, deps: Any) -> Generator[StreamEvent, None, AgentResult]:
        """Generator that executes the agent loop and yields stream events.

        Raises:
            RecursionError: If the agent nesting depth exceeds ``max_depth``.
        """
        current_depth = _agent_depth.get()
        if current_depth >= self._max_depth:
            raise RecursionError(f"Agent recursion depth exceeded: {current_depth} >= {self._max_depth}")
        token = _agent_depth.set(current_depth + 1)
        self._lifecycle = AgentState.running
        self._stop_requested = False
        self._active_conversation = None
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
            run_state: dict[str, Any] = {}
            tool_timings: list[dict[str, Any]] = []
            wrapped_tools = self._wrap_tools_with_context(
                ctx,
                session,
                ctx_fn=self._make_live_ctx_fn(prompt, deps, session, run_state),
                tool_timings=tool_timings,
            )
            has_tools = bool(wrapped_tools)

            # 6. Build Conversation
            conv = self._build_conversation(
                system_prompt=resolved_system_prompt,
                tools=wrapped_tools if wrapped_tools else None,
                driver_callbacks=driver_callbacks,
            )
            run_state["conv"] = conv

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
                    for chunk in conv._ask_stream_raw(effective_prompt):
                        if chunk["type"] == "delta":
                            response_text += chunk["text"]
                            yield StreamEvent(
                                event_type=StreamEventType.text_delta,
                                data=chunk["text"],
                            )
                        elif chunk["type"] == "thinking_delta":
                            yield StreamEvent(
                                event_type=StreamEventType.thinking_delta,
                                data=chunk["text"],
                            )
            finally:
                if security_token is not None:
                    from tukuy.safety import reset_security_context

                    reset_security_context(security_token)

            # 8. Extract steps
            all_tool_calls: list[dict[str, Any]] = []
            self._extract_steps(
                conv.messages,
                steps,
                all_tool_calls,
                getattr(conv, "_full_tool_results", None),
                tool_timings,
            )

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
            if self._agent_callbacks.on_thinking:
                for step in steps:
                    if step.step_type == StepType.think and step.content:
                        self._agent_callbacks.on_thinking(step.content)
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
        finally:
            _agent_depth.reset(token)

    # ------------------------------------------------------------------
    # run_live() — interleaved tool calling with streaming text deltas
    # ------------------------------------------------------------------

    def run_live(self, prompt: str, *, deps: Any = None) -> LiveAgentResult:
        """Execute the agent loop yielding interleaved
        :class:`~prompture.agents.live_events.LiveEvent` objects.

        Unlike :meth:`run_stream`, ``run_live`` forwards the driver's
        native streaming event sequence (Anthropic SSE / OpenAI deltas)
        so the caller sees text and reasoning deltas *between* tool calls
        within a single LLM turn — the "Claude Code feel".

        Drivers without ``supports_streaming_tool_use`` fall back to the
        base-class synthetic event sequence (one bundle per turn).

        Returns a :class:`LiveAgentResult` iterable. After iteration
        completes, :attr:`LiveAgentResult.result` holds the final
        :class:`AgentResult`.
        """
        gen = self._execute_live(prompt, deps)
        return LiveAgentResult(gen)

    def _execute_live(self, prompt: str, deps: Any) -> Generator[Any, None, AgentResult]:
        """Generator that drives ask_live and yields :class:`LiveEvent`."""
        from ..infra.tracker import get_tracker
        from .live_events import TextDelta, ThinkingDelta

        current_depth = _agent_depth.get()
        if current_depth >= self._max_depth:
            raise RecursionError(f"Agent recursion depth exceeded: {current_depth} >= {self._max_depth}")
        token = _agent_depth.set(current_depth + 1)
        self._lifecycle = AgentState.running
        self._stop_requested = False
        self._active_conversation = None
        steps: list[AgentStep] = []

        try:
            session = UsageSession()
            driver_callbacks = DriverCallbacks(
                on_response=session.record,
                on_error=session.record_error,
            )
            ctx = self._build_run_context(prompt, deps, session, [], 0)
            effective_prompt = self._run_input_guardrails(ctx, prompt)
            resolved_system_prompt = self._resolve_system_prompt(ctx)
            run_state: dict[str, Any] = {}
            tool_timings: list[dict[str, Any]] = []
            wrapped_tools = self._wrap_tools_with_context(
                ctx,
                session,
                ctx_fn=self._make_live_ctx_fn(prompt, deps, session, run_state),
                tool_timings=tool_timings,
            )

            conv = self._build_conversation(
                system_prompt=resolved_system_prompt,
                tools=wrapped_tools if wrapped_tools else None,
                driver_callbacks=driver_callbacks,
            )
            run_state["conv"] = conv

            if self._agent_callbacks.on_iteration:
                self._agent_callbacks.on_iteration(0)

            tracker = get_tracker()
            agent_name = self.name or self.__class__.__name__

            response_text_parts: list[str] = []
            thinking_parts: list[str] = []

            with tracker.agent(agent_name):
                security_token = None
                if self._security_context is not None:
                    from tukuy.safety import set_security_context

                    security_token = set_security_context(self._security_context)
                try:
                    for event in conv.ask_live(effective_prompt):
                        if isinstance(event, TextDelta):
                            response_text_parts.append(event.text)
                        elif isinstance(event, ThinkingDelta):
                            thinking_parts.append(event.text)
                        yield event
                finally:
                    if security_token is not None:
                        from tukuy.safety import reset_security_context

                        reset_security_context(security_token)

            response_text = "".join(response_text_parts)
            all_tool_calls: list[dict[str, Any]] = []
            self._extract_steps(
                conv.messages,
                steps,
                all_tool_calls,
                getattr(conv, "_full_tool_results", None),
                tool_timings,
            )

            if self._output_type is not None:
                output, output_text = self._parse_output(conv, response_text, steps, all_tool_calls, 0.0, session)
            else:
                output = response_text
                output_text = response_text

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

            if self._output_guardrails:
                result = self._run_output_guardrails(ctx, result, conv, session, steps, all_tool_calls)

            if self._agent_callbacks.on_step:
                for step in steps:
                    self._agent_callbacks.on_step(step)
            if self._agent_callbacks.on_output:
                self._agent_callbacks.on_output(result)
            if self._agent_callbacks.on_message:
                self._agent_callbacks.on_message(result.output_text)

            self._lifecycle = AgentState.idle
            return result
        except Exception:
            self._lifecycle = AgentState.errored
            raise
        finally:
            _agent_depth.reset(token)


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


# ------------------------------------------------------------------
# LiveAgentResult
# ------------------------------------------------------------------


class LiveAgentResult:
    """Wraps the :meth:`Agent.run_live` generator, capturing the final result.

    Yields :class:`~prompture.agents.live_events.LiveEvent` objects during
    iteration.  After iteration completes, :attr:`result` holds the final
    :class:`AgentResult`.
    """

    def __init__(self, gen: Generator[Any, None, AgentResult]) -> None:
        self._gen = gen
        self._result: AgentResult | None = None

    def __iter__(self) -> LiveAgentResult:
        return self

    def __next__(self) -> Any:
        try:
            return next(self._gen)
        except StopIteration as e:
            self._result = e.value
            raise

    @property
    def result(self) -> AgentResult | None:
        """The final :class:`AgentResult`, available after iteration completes."""
        return self._result
