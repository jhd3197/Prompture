"""Async Agent framework for Prompture.

Provides :class:`AsyncAgent`, the async counterpart of :class:`~prompture.agent.Agent`.
All methods are ``async`` and use :class:`~prompture.async_conversation.AsyncConversation`.

Example::

    from prompture import AsyncAgent

    agent = AsyncAgent("openai/gpt-4o", system_prompt="You are helpful.")
    result = await agent.run("What is 2 + 2?")
    print(result.output)
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
from collections.abc import AsyncGenerator, Callable
from typing import Any, Generic

from pydantic import BaseModel

from ..extraction.tools import clean_json_text
from ..infra.budget import BudgetPolicy, resolve_budget_policy
from ..infra.callbacks import DriverCallbacks
from ..infra.provider_env import ProviderEnvironment
from ..infra.session import UsageSession
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

logger = logging.getLogger("prompture.async_agent")

_OUTPUT_PARSE_MAX_RETRIES = 3
_OUTPUT_GUARDRAIL_MAX_RETRIES = 3
_DEFAULT_MAX_AGENT_DEPTH = 5

# Share the same ContextVar with the sync Agent so depth tracking crosses
# agent boundaries (e.g. a sync Agent calling an AsyncAgent as a tool).
from .agent import _agent_depth

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _is_async_callable(fn: Callable[..., Any]) -> bool:
    """Check if *fn* is an async callable (coroutine function or has async ``__call__``)."""
    if asyncio.iscoroutinefunction(fn):
        return True
    # Check if the object has an async __call__ method (callable class)
    dunder_call = type(fn).__call__ if callable(fn) else None
    return dunder_call is not None and asyncio.iscoroutinefunction(dunder_call)


def _run_awaitable_sync(awaitable: Any) -> Any:
    """Drive *awaitable* to completion from sync code.

    Used only as a fallback when a sync caller (``ToolRegistry.execute``)
    hits an async tool/callback.  When a loop is already running in the
    current thread the awaitable runs on a worker thread with the caller's
    ``contextvars`` copied over, so values such as tukuy's
    ``SecurityContext`` and ``current_tool_call_id`` survive the hop.
    The preferred path is the ``_async_fn`` hook awaited by
    :meth:`ToolRegistry.aexecute`, which needs no thread bridge at all.
    """

    async def _await_it() -> Any:
        return await awaitable

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        ctx_snapshot = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(ctx_snapshot.run, lambda: asyncio.run(_await_it())).result()
    return asyncio.run(_await_it())


def _tool_wants_context(fn: Callable[..., Any]) -> bool:
    """Check whether *fn*'s first parameter is annotated as :class:`RunContext`."""
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    if not params:
        return False

    first_param = params[0]
    if first_param == "self":
        if len(params) < 2:
            return False
        first_param = params[1]

    annotation = None
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
        annotation = hints.get(first_param)
    except Exception:
        # get_type_hints can fail with local/forward references; fall back to raw annotation
        pass

    if annotation is None:
        raw = sig.parameters[first_param].annotation
        if raw is inspect.Parameter.empty:
            return False
        annotation = raw

    if isinstance(annotation, str):
        return annotation == "RunContext" or annotation.startswith("RunContext[")

    if annotation is RunContext:
        return True

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
# AsyncAgent
# ------------------------------------------------------------------


class AsyncAgent(Generic[DepsType]):
    """Async agent that executes a ReAct loop with tool support.

    Mirrors :class:`~prompture.agent.Agent` but uses
    :class:`~prompture.async_conversation.AsyncConversation` and
    ``async`` methods throughout.

    Args:
        model: Model string in ``"provider/model"`` format.
        driver: Pre-built async driver instance.
        tools: Initial tools as a list of callables or a :class:`ToolRegistry`.
        system_prompt: System prompt prepended to every run.  May also be a
            callable ``(RunContext) -> str`` for dynamic prompts.
        output_type: Optional Pydantic model class for structured output.
        max_iterations: Maximum tool-use rounds per run.
        max_cost: Soft budget in USD.
        options: Extra driver options forwarded to every LLM call.
            Common keys include ``"temperature"`` (float, 0.0-2.0),
            ``"max_tokens"`` (int), and ``"top_p"`` (float).
            Example: ``options={"temperature": 0.7, "max_tokens": 1024}``.
        deps_type: Type hint for dependencies.
        agent_callbacks: Agent-level observability callbacks.
        input_guardrails: Functions called before the prompt is sent.
        output_guardrails: Functions called after output is parsed.
        persistent_conversation: When ``True``, subsequent ``run()``
            calls reuse the same :class:`AsyncConversation` so the model
            sees the full multi-turn history.  Default ``False``.
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
        driver: Any | None = None,
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
        self._conversation: Any = None
        # The conversation driving the current run.  Tracked separately from
        # ``_conversation`` (which only holds persistent ones) so that
        # ``stop()`` can reach the loop of a non-persistent run too.
        self._active_conversation: Any = None

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
        """Decorator to register a function as a tool on this agent."""
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

        Sets the agent-level flag and forwards the request to the conversation
        driving the current run (cooperative stop) so its tool-round loop
        exits gracefully between rounds instead of starting another round.
        Works for non-persistent agents too: the in-flight conversation is
        tracked in ``_active_conversation`` for the duration of the run, so
        the flag is never merely decorative.

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
    def conversation(self) -> Any:
        """The current persistent conversation, or ``None``."""
        return self._conversation

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Message history from the persistent conversation, or ``[]``."""
        if self._conversation is not None:
            return self._conversation.messages  # type: ignore[no-any-return]
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
        """Wrap this AsyncAgent as a callable tool for another Agent.

        Creates a :class:`ToolDefinition` whose function accepts a ``prompt``
        string, runs this agent (bridging async to sync), and returns the
        output text.

        Args:
            name: Tool name (defaults to ``self.name`` or ``"agent_tool"``).
            description: Tool description (defaults to ``self.description``).
            custom_output_extractor: Optional function to extract a string
                from :class:`AgentResult`.
        """
        tool_name = name or self.name or "agent_tool"
        tool_desc = description or self.description or f"Run agent {tool_name}"
        agent = self
        extractor = custom_output_extractor

        def _call_agent(prompt: str) -> str:
            """Run the wrapped async agent with the given prompt."""
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, agent.run(prompt)).result()
            else:
                result = asyncio.run(agent.run(prompt))

            _call_agent._last_agent_result = result  # type: ignore[attr-defined]
            if extractor is not None:
                return extractor(result)
            return result.output_text

        async def _call_agent_async(prompt: str) -> str:
            """Run the wrapped async agent, awaited in the current loop.

            Registered as ``_async_fn`` so :meth:`ToolRegistry.aexecute`
            awaits this directly instead of using the thread bridge in
            ``_call_agent`` (which drops ``contextvars`` such as tukuy's
            ``SecurityContext`` and ``current_tool_call_id``).
            """
            result = await agent.run(prompt)
            _call_agent._last_agent_result = result  # type: ignore[attr-defined]
            if extractor is not None:
                extracted = extractor(result)
                if inspect.isawaitable(extracted):
                    extracted = await extracted
                return extracted
            return result.output_text

        _call_agent._source_agent = agent  # type: ignore[attr-defined]
        _call_agent._last_agent_result = None  # type: ignore[attr-defined]
        _call_agent._async_fn = _call_agent_async  # type: ignore[attr-defined]

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

    async def run(
        self,
        prompt: str,
        *,
        deps: Any = None,
        images: list[Any] | None = None,
    ) -> AgentResult:
        """Execute the agent loop to completion (async).

        Creates a fresh conversation, sends the prompt, handles tool calls,
        and optionally parses the final response into ``output_type``.

        Args:
            prompt: The user prompt.
            deps: Optional dependencies.
            images: Optional list of :class:`ImageInput` for vision models.

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
            result = await self._execute(prompt, steps, deps, images=images)
            self._lifecycle = AgentState.idle
            return result
        except Exception:
            self._lifecycle = AgentState.errored
            raise
        finally:
            _agent_depth.reset(token)

    def iter(
        self,
        prompt: str,
        *,
        deps: Any = None,
        images: list[Any] | None = None,
    ) -> AsyncAgentIterator:
        """Execute the agent loop and iterate over steps asynchronously.

        Returns an :class:`AsyncAgentIterator` yielding :class:`AgentStep` objects.
        After iteration, :attr:`AsyncAgentIterator.result` holds the final result.
        """
        gen = self._execute_iter(prompt, deps, images=images)
        return AsyncAgentIterator(gen)

    def run_stream(
        self,
        prompt: str,
        *,
        deps: Any = None,
        images: list[Any] | None = None,
    ) -> AsyncStreamedAgentResult:
        """Execute the agent loop with streaming output (async).

        Returns an :class:`AsyncStreamedAgentResult` yielding :class:`StreamEvent` objects.
        """
        gen = self._execute_stream(prompt, deps, images=images)
        return AsyncStreamedAgentResult(gen)

    # ------------------------------------------------------------------
    # Async callback helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _invoke_callback(cb: Callable[..., Any], *args: Any) -> Any:
        """Invoke a callback, awaiting it if it's async."""
        if _is_async_callable(cb):
            return await cb(*args)
        return cb(*args)

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
        - Fire ``agent_callbacks.on_tool_start`` / ``on_tool_end``.
        - Strip the ``RunContext`` parameter from the JSON schema sent to the LLM.
        - If the tool wraps a child agent (via ``as_tool``), aggregate its
          usage into the parent *session*.
        - When *tool_timings* is provided, append a
          ``{"name", "timestamp", "duration_ms"}`` record per invocation.

        Async tool functions additionally get an ``async`` wrapper attached
        as ``_async_fn`` on the sync wrapper.  :meth:`ToolRegistry.aexecute`
        prefers and awaits that hook, so inside ``AsyncConversation`` the
        coroutine runs on the current event loop — no thread/asyncio.run
        bridge, and ``contextvars`` (tukuy ``SecurityContext``,
        ``current_tool_call_id``) propagate naturally.  The sync wrapper
        remains as a fallback for plain :meth:`ToolRegistry.execute` calls.
        """
        if not self._tools:
            return ToolRegistry()

        new_registry = ToolRegistry()
        cb = self._agent_callbacks

        for td in self._tools.definitions:
            wants_ctx = _tool_wants_context(td.function)
            original_fn = td.function
            tool_name = td.name
            is_async = _is_async_callable(original_fn)

            def _make_wrapper(
                _fn: Callable[..., Any],
                _wants: bool,
                _name: str,
                _is_async: bool,
                _cb: AgentCallbacks = cb,
                _session: UsageSession | None = session,
                _ctx_fn: Callable[[], RunContext[Any]] | None = ctx_fn,
                _timings: list[dict[str, Any]] | None = tool_timings,
            ) -> Callable[..., Any]:
                def _invoke_cb_sync(callback: Callable[..., Any], *cb_args: Any) -> Any:
                    """Invoke a possibly-async callback from a sync tool wrapper."""
                    result = callback(*cb_args)
                    if inspect.isawaitable(result):
                        return _run_awaitable_sync(result)
                    return result

                def _prepare_call(kwargs: dict[str, Any]) -> dict[str, Any]:
                    """Apply skill-config injection and the auto-approve gate."""
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
                    return kwargs

                def _call_args(call_ctx: RunContext[Any]) -> tuple[Any, ...]:
                    return (call_ctx,) if _wants else ()

                def _aggregate_child_usage() -> None:
                    """Aggregate child agent usage to the parent session."""
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

                def _record_timing(start: float) -> None:
                    if _timings is not None:
                        _timings.append(
                            {
                                "name": _name,
                                "timestamp": time.time(),
                                "duration_ms": (time.perf_counter() - start) * 1000,
                            }
                        )

                def wrapper(**kwargs: Any) -> Any:
                    call_ctx = _ctx_fn() if _ctx_fn is not None else ctx
                    start = time.perf_counter()
                    if _cb.on_tool_start:
                        _invoke_cb_sync(_cb.on_tool_start, _name, kwargs)
                    try:
                        _prepare_call(kwargs)
                        if _is_async:
                            result = _run_awaitable_sync(_fn(*_call_args(call_ctx), **kwargs))
                        else:
                            result = _fn(*_call_args(call_ctx), **kwargs)
                    except ApprovalRequired as exc:
                        # Handle approval request
                        if _cb.on_approval_needed:
                            approved = _invoke_cb_sync(_cb.on_approval_needed, exc.tool_name, exc.action, exc.details)
                            if approved:
                                # Retry the tool call after approval
                                try:
                                    if _is_async:
                                        result = _run_awaitable_sync(_fn(*_call_args(call_ctx), **kwargs))
                                    else:
                                        result = _fn(*_call_args(call_ctx), **kwargs)
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

                    _aggregate_child_usage()
                    _record_timing(start)

                    if _cb.on_tool_end:
                        _invoke_cb_sync(_cb.on_tool_end, _name, result)
                    return result

                async def async_wrapper(**kwargs: Any) -> Any:
                    """Fully-async wrapper: awaited directly via ``_async_fn``."""
                    call_ctx = _ctx_fn() if _ctx_fn is not None else ctx
                    start = time.perf_counter()
                    if _cb.on_tool_start:
                        await AsyncAgent._invoke_callback(_cb.on_tool_start, _name, kwargs)
                    try:
                        _prepare_call(kwargs)
                        result = await _fn(*_call_args(call_ctx), **kwargs)
                    except ApprovalRequired as exc:
                        # Handle approval request
                        if _cb.on_approval_needed:
                            approved = await AsyncAgent._invoke_callback(
                                _cb.on_approval_needed, exc.tool_name, exc.action, exc.details
                            )
                            if approved:
                                # Retry the tool call after approval
                                try:
                                    result = await _fn(*_call_args(call_ctx), **kwargs)
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

                    _aggregate_child_usage()
                    _record_timing(start)

                    if _cb.on_tool_end:
                        await AsyncAgent._invoke_callback(_cb.on_tool_end, _name, result)
                    return result

                if _is_async:
                    wrapper._async_fn = async_wrapper  # type: ignore[attr-defined]
                return wrapper

            wrapped = _make_wrapper(original_fn, wants_ctx, tool_name, is_async)

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
        for guardrail in self._input_guardrails:
            result = guardrail(ctx, prompt)
            if result is not None:
                prompt = result
        return prompt

    async def _run_output_guardrails(
        self,
        ctx: RunContext[Any],
        result: AgentResult,
        conv: Any,
        session: UsageSession,
        steps: list[AgentStep],
        all_tool_calls: list[dict[str, Any]],
    ) -> AgentResult:
        for guardrail in self._output_guardrails:
            for attempt in range(_OUTPUT_GUARDRAIL_MAX_RETRIES):
                try:
                    guard_result = guardrail(ctx, result)
                    if guard_result is not None:
                        result = guard_result
                    break
                except ModelRetry as exc:
                    if self._is_over_budget(session):
                        break
                    if attempt >= _OUTPUT_GUARDRAIL_MAX_RETRIES - 1:
                        raise ValueError(
                            f"Output guardrail failed after {_OUTPUT_GUARDRAIL_MAX_RETRIES} retries: {exc.message}"
                        ) from exc
                    retry_text = await conv.ask(
                        f"Your response did not pass validation. Error: {exc.message}\n\nPlease try again."
                    )
                    self._extract_steps(
                        conv.messages[-2:],
                        steps,
                        all_tool_calls,
                        getattr(conv, "_full_tool_results", None),
                    )

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
        if self._max_cost is not None and session.cost >= self._max_cost:
            return True
        return bool(self._max_tokens is not None and session.total_tokens >= self._max_tokens)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_system_prompt(self, ctx: RunContext[Any] | None = None) -> str | None:
        parts: list[str] = []

        if self._system_prompt is not None:
            if isinstance(self._system_prompt, Persona):
                render_kwargs: dict[str, Any] = {}
                if ctx is not None:
                    render_kwargs["model"] = ctx.model
                    render_kwargs["iteration"] = ctx.iteration
                parts.append(self._system_prompt.render(**render_kwargs))
            elif callable(self._system_prompt) and not isinstance(self._system_prompt, str):
                if ctx is not None:
                    parts.append(self._system_prompt(ctx))
                else:
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
    ) -> Any:
        """Create or reuse an AsyncConversation for a run.

        When ``persistent_conversation=True`` and a conversation already
        exists, it is reused (with updated tools and callbacks).
        Otherwise a fresh :class:`AsyncConversation` is created.
        """
        from .async_conversation import AsyncConversation

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
                # NOTE: assigning ``callbacks`` mutates the driver instance,
                # which may be shared (e.g. passed to several agents).  A
                # persistent-conversation agent should treat its driver as
                # exclusively owned by the agent.
                if tools is not None:
                    conv._tools = tools
                if driver_callbacks is not None:
                    conv._driver.callbacks = driver_callbacks
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

        conv = AsyncConversation(**kwargs)
        if self._persistent_conversation:
            self._conversation = conv
        return self._track_active(conv)

    def _track_active(self, conv: Any) -> Any:
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

    async def _execute(
        self,
        prompt: str,
        steps: list[AgentStep],
        deps: Any,
        *,
        images: list[Any] | None = None,
    ) -> AgentResult:
        """Core async execution: run conversation, extract steps, parse output."""
        from ..infra.tracker import get_tracker

        tracker = get_tracker()

        # 1. Create per-run UsageSession
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

        # 6. Build AsyncConversation
        conv = self._build_conversation(
            system_prompt=resolved_system_prompt,
            tools=wrapped_tools if wrapped_tools else None,
            driver_callbacks=driver_callbacks,
        )
        run_state["conv"] = conv

        # 7. Fire on_iteration callback
        if self._agent_callbacks.on_iteration:
            await self._invoke_callback(self._agent_callbacks.on_iteration, 0)

        # 8. Ask the conversation (handles full tool loop internally)
        #    Note: budget policy is enforced inside the conversation; the
        #    agent-level pre-call check was removed because it read a fresh,
        #    always-empty UsageSession.
        agent_name = self.name or self.__class__.__name__
        with tracker.agent(agent_name):
            t0 = time.perf_counter()
            security_token = None
            if self._security_context is not None:
                from tukuy.safety import set_security_context

                security_token = set_security_context(self._security_context)
            try:
                response_text = await conv.ask(effective_prompt, images=images)
            finally:
                if security_token is not None:
                    from tukuy.safety import reset_security_context

                    reset_security_context(security_token)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # 9. Extract steps and tool calls
            all_tool_calls: list[dict[str, Any]] = []
            full_results = getattr(conv, "_full_tool_results", None)
            self._extract_steps(conv.messages, steps, all_tool_calls, full_results, tool_timings)

            # Handle output_type parsing
            if self._output_type is not None:
                output, output_text = await self._parse_output(
                    conv, response_text, steps, all_tool_calls, elapsed_ms, session
                )
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

            # 10. Run output guardrails
            if self._output_guardrails:
                result = await self._run_output_guardrails(ctx, result, conv, session, steps, all_tool_calls)

            # 11. Fire callbacks (async-aware)
            if self._agent_callbacks.on_thinking:
                for step in steps:
                    if step.step_type == StepType.think and step.content:
                        await self._invoke_callback(self._agent_callbacks.on_thinking, step.content)
            if self._agent_callbacks.on_step:
                for step in steps:
                    await self._invoke_callback(self._agent_callbacks.on_step, step)
            if self._agent_callbacks.on_output:
                await self._invoke_callback(self._agent_callbacks.on_output, result)

            if self._agent_callbacks.on_message:
                await self._invoke_callback(self._agent_callbacks.on_message, result.output_text)

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
                    steps.append(
                        AgentStep(
                            step_type=StepType.output,
                            timestamp=now,
                            content=content,
                            usage=step_usage,
                        )
                    )

            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                # Use the full (pre-truncation) result when available,
                # falling back to the (possibly truncated) message content.
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
            return "\n".join(match.strip() for match in matches)
        return None

    async def _parse_output(
        self,
        conv: Any,
        response_text: str,
        steps: list[AgentStep],
        all_tool_calls: list[dict[str, Any]],
        elapsed_ms: float,
        session: UsageSession | None = None,
    ) -> tuple[Any, str]:
        """Try to parse ``response_text`` as the output_type, with retries (async)."""
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
                    if session is not None and self._is_over_budget(session):
                        break
                    retry_msg = (
                        f"Your previous response could not be parsed as valid JSON "
                        f"matching the required schema. Error: {exc}\n\n"
                        f"Please try again and respond ONLY with valid JSON."
                    )
                    text = await conv.ask(retry_msg)
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
    # iter() — async step-by-step
    # ------------------------------------------------------------------

    async def _execute_iter(
        self,
        prompt: str,
        deps: Any,
        *,
        images: list[Any] | None = None,
    ) -> AsyncGenerator[AgentStep]:
        """Async generator that executes the agent loop and yields each step.

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
            result = await self._execute(prompt, steps, deps, images=images)
            for step in result.steps:
                yield step
            self._lifecycle = AgentState.idle
            # Store result on the generator for retrieval
            self._last_iter_result = result
        except Exception:
            self._lifecycle = AgentState.errored
            raise
        finally:
            _agent_depth.reset(token)

    # ------------------------------------------------------------------
    # run_stream() — async streaming
    # ------------------------------------------------------------------

    async def _execute_stream(
        self,
        prompt: str,
        deps: Any,
        *,
        images: list[Any] | None = None,
    ) -> AsyncGenerator[StreamEvent]:
        """Async generator that executes the agent loop and yields stream events.

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
            # 1. Setup
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
            has_tools = bool(wrapped_tools)

            conv = self._build_conversation(
                system_prompt=resolved_system_prompt,
                tools=wrapped_tools if wrapped_tools else None,
                driver_callbacks=driver_callbacks,
            )
            run_state["conv"] = conv

            if self._agent_callbacks.on_iteration:
                await self._invoke_callback(self._agent_callbacks.on_iteration, 0)

            security_token = None
            if self._security_context is not None:
                from tukuy.safety import set_security_context

                security_token = set_security_context(self._security_context)
            try:
                if has_tools:
                    response_text = ""
                    async for event in conv.ask_with_tool_events(effective_prompt, images=images):
                        if event["type"] == "tool_call":
                            yield StreamEvent(event_type=StreamEventType.tool_call, data=event)
                        elif event["type"] == "tool_result":
                            yield StreamEvent(event_type=StreamEventType.tool_result, data=event)
                        elif event["type"] == "text_delta":
                            response_text += event["text"]
                            yield StreamEvent(event_type=StreamEventType.text_delta, data=event["text"])
                else:
                    response_text = ""
                    async for chunk in conv._ask_stream_raw(effective_prompt, images=images):
                        if chunk["type"] == "delta":
                            response_text += chunk["text"]
                            yield StreamEvent(event_type=StreamEventType.text_delta, data=chunk["text"])
                        elif chunk["type"] == "thinking_delta":
                            yield StreamEvent(event_type=StreamEventType.thinking_delta, data=chunk["text"])
            finally:
                if security_token is not None:
                    from tukuy.safety import reset_security_context

                    reset_security_context(security_token)

            # Extract steps
            all_tool_calls: list[dict[str, Any]] = []
            full_results = getattr(conv, "_full_tool_results", None)
            self._extract_steps(conv.messages, steps, all_tool_calls, full_results, tool_timings)

            # Parse output
            if self._output_type is not None:
                output, output_text = await self._parse_output(conv, response_text, steps, all_tool_calls, 0.0, session)
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
                result = await self._run_output_guardrails(ctx, result, conv, session, steps, all_tool_calls)

            if self._agent_callbacks.on_step:
                for step in steps:
                    await self._invoke_callback(self._agent_callbacks.on_step, step)
            if self._agent_callbacks.on_output:
                await self._invoke_callback(self._agent_callbacks.on_output, result)

            if self._agent_callbacks.on_message:
                await self._invoke_callback(self._agent_callbacks.on_message, result.output_text)

            yield StreamEvent(event_type=StreamEventType.output, data=result)

            self._lifecycle = AgentState.idle
            self._last_stream_result = result
        except Exception:
            self._lifecycle = AgentState.errored
            raise
        finally:
            _agent_depth.reset(token)

    # ------------------------------------------------------------------
    # run_live() — interleaved tool calling with streaming text deltas
    # ------------------------------------------------------------------

    def run_live(
        self,
        prompt: str,
        *,
        deps: Any = None,
        images: list[Any] | None = None,
    ) -> AsyncLiveAgentResult:
        """Execute the async agent loop yielding interleaved
        :class:`~prompture.agents.live_events.LiveEvent` objects.

        Async sibling of :meth:`Agent.run_live`. Returns an
        :class:`AsyncLiveAgentResult` you ``async for`` over; after the
        stream completes, ``.result`` holds the final
        :class:`AgentResult`.
        """
        gen = self._execute_live(prompt, deps, images=images)
        return AsyncLiveAgentResult(gen, agent=self)

    async def _execute_live(
        self,
        prompt: str,
        deps: Any,
        *,
        images: list[Any] | None = None,
    ) -> AsyncGenerator[Any]:
        """Async generator that drives ask_live and yields :class:`LiveEvent`."""
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
                await self._invoke_callback(self._agent_callbacks.on_iteration, 0)

            tracker = get_tracker()
            agent_name = self.name or self.__class__.__name__

            response_text_parts: list[str] = []

            with tracker.agent(agent_name):
                security_token = None
                if self._security_context is not None:
                    from tukuy.safety import set_security_context

                    security_token = set_security_context(self._security_context)
                try:
                    async for event in conv.ask_live(effective_prompt, images=images):
                        if isinstance(event, TextDelta):
                            response_text_parts.append(event.text)
                        elif isinstance(event, ThinkingDelta):
                            pass
                        yield event
                finally:
                    if security_token is not None:
                        from tukuy.safety import reset_security_context

                        reset_security_context(security_token)

            response_text = "".join(response_text_parts)
            all_tool_calls: list[dict[str, Any]] = []
            full_results = getattr(conv, "_full_tool_results", None)
            self._extract_steps(conv.messages, steps, all_tool_calls, full_results, tool_timings)

            if self._output_type is not None:
                output, output_text = await self._parse_output(conv, response_text, steps, all_tool_calls, 0.0, session)
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
                result = await self._run_output_guardrails(ctx, result, conv, session, steps, all_tool_calls)

            if self._agent_callbacks.on_step:
                for step in steps:
                    await self._invoke_callback(self._agent_callbacks.on_step, step)
            if self._agent_callbacks.on_output:
                await self._invoke_callback(self._agent_callbacks.on_output, result)
            if self._agent_callbacks.on_message:
                await self._invoke_callback(self._agent_callbacks.on_message, result.output_text)

            self._lifecycle = AgentState.idle
            self._last_live_result = result
        except Exception:
            self._lifecycle = AgentState.errored
            raise
        finally:
            _agent_depth.reset(token)


# ------------------------------------------------------------------
# AsyncAgentIterator
# ------------------------------------------------------------------


class AsyncAgentIterator:
    """Wraps the :meth:`AsyncAgent.iter` async generator, capturing the final result.

    After async iteration completes, :attr:`result` holds the :class:`AgentResult`.
    """

    def __init__(self, gen: AsyncGenerator[AgentStep]) -> None:
        self._gen = gen
        self._result: AgentResult | None = None
        self._agent: AsyncAgent[Any] | None = None

    def __aiter__(self) -> AsyncAgentIterator:
        return self

    async def __anext__(self) -> AgentStep:
        try:
            return await self._gen.__anext__()
        except StopAsyncIteration:
            # Try to capture the result from the agent
            ag_frame = getattr(self._gen, "ag_frame", None)
            agent = ag_frame and ag_frame.f_locals.get("self") if ag_frame else None
            if agent and hasattr(agent, "_last_iter_result"):
                self._result = agent._last_iter_result
            raise

    @property
    def result(self) -> AgentResult | None:
        """The final :class:`AgentResult`, available after iteration completes."""
        return self._result


# ------------------------------------------------------------------
# AsyncStreamedAgentResult
# ------------------------------------------------------------------


class AsyncStreamedAgentResult:
    """Wraps the :meth:`AsyncAgent.run_stream` async generator.

    Yields :class:`StreamEvent` objects.  After iteration completes,
    :attr:`result` holds the :class:`AgentResult`.
    """

    def __init__(self, gen: AsyncGenerator[StreamEvent]) -> None:
        self._gen = gen
        self._result: AgentResult | None = None

    def __aiter__(self) -> AsyncStreamedAgentResult:
        return self

    async def __anext__(self) -> StreamEvent:
        try:
            event = await self._gen.__anext__()
            # Capture result from the output event
            if event.event_type == StreamEventType.output and isinstance(event.data, AgentResult):
                self._result = event.data
            return event
        except StopAsyncIteration:
            raise

    @property
    def result(self) -> AgentResult | None:
        """The final :class:`AgentResult`, available after iteration completes."""
        return self._result


# ------------------------------------------------------------------
# AsyncLiveAgentResult
# ------------------------------------------------------------------


class AsyncLiveAgentResult:
    """Wraps the :meth:`AsyncAgent.run_live` async generator.

    Yields :class:`~prompture.agents.live_events.LiveEvent` objects.
    After async iteration completes, :attr:`result` holds the final
    :class:`AgentResult`.

    Async generators cannot return values via :class:`StopAsyncIteration`
    the way sync generators do, so the wrapper keeps a reference to the
    originating agent and reads ``_last_live_result`` from it when the
    stream terminates.
    """

    def __init__(self, gen: AsyncGenerator[Any], *, agent: AsyncAgent[Any] | None = None) -> None:
        self._gen = gen
        self._agent = agent
        self._result: AgentResult | None = None

    def __aiter__(self) -> AsyncLiveAgentResult:
        return self

    async def __anext__(self) -> Any:
        try:
            return await self._gen.__anext__()
        except StopAsyncIteration:
            if self._agent is not None:
                self._result = getattr(self._agent, "_last_live_result", None)
            raise

    @property
    def result(self) -> AgentResult | None:
        """The final :class:`AgentResult`, available after iteration completes."""
        return self._result
