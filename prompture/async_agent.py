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
import inspect
import json
import logging
import time
import typing
from collections.abc import AsyncGenerator, Callable
from typing import Any, Generic

from pydantic import BaseModel

from .agent_types import (
    AgentCallbacks,
    AgentResult,
    AgentState,
    AgentStep,
    DepsType,
    ModelRetry,
    RunContext,
    StepType,
    StreamEvent,
    StreamEventType,
)
from .callbacks import DriverCallbacks
from .persona import Persona
from .session import UsageSession
from .tools import clean_json_text
from .tools_schema import ToolDefinition, ToolRegistry

logger = logging.getLogger("prompture.async_agent")

_OUTPUT_PARSE_MAX_RETRIES = 3
_OUTPUT_GUARDRAIL_MAX_RETRIES = 3


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
        options: Extra driver options forwarded to every call.
        deps_type: Type hint for dependencies.
        agent_callbacks: Agent-level observability callbacks.
        input_guardrails: Functions called before the prompt is sent.
        output_guardrails: Functions called after output is parsed.
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
        options: dict[str, Any] | None = None,
        deps_type: type | None = None,
        agent_callbacks: AgentCallbacks | None = None,
        input_guardrails: list[Callable[..., Any]] | None = None,
        output_guardrails: list[Callable[..., Any]] | None = None,
        name: str = "",
        description: str = "",
        output_key: str | None = None,
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
        """Decorator to register a function as a tool on this agent."""
        self._tools.register(fn)
        return fn

    @property
    def state(self) -> AgentState:
        """Current lifecycle state of the agent."""
        return self._state

    def stop(self) -> None:
        """Request graceful shutdown after the current iteration."""
        self._stop_requested = True

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
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, agent.run(prompt)).result()
            else:
                result = asyncio.run(agent.run(prompt))

            if extractor is not None:
                return extractor(result)
            return result.output_text

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

    async def run(self, prompt: str, *, deps: Any = None) -> AgentResult:
        """Execute the agent loop to completion (async).

        Creates a fresh conversation, sends the prompt, handles tool calls,
        and optionally parses the final response into ``output_type``.
        """
        self._state = AgentState.running
        self._stop_requested = False
        steps: list[AgentStep] = []

        try:
            result = await self._execute(prompt, steps, deps)
            self._state = AgentState.idle
            return result
        except Exception:
            self._state = AgentState.errored
            raise

    async def iter(self, prompt: str, *, deps: Any = None) -> AsyncAgentIterator:
        """Execute the agent loop and iterate over steps asynchronously.

        Returns an :class:`AsyncAgentIterator` yielding :class:`AgentStep` objects.
        After iteration, :attr:`AsyncAgentIterator.result` holds the final result.
        """
        gen = self._execute_iter(prompt, deps)
        return AsyncAgentIterator(gen)

    async def run_stream(self, prompt: str, *, deps: Any = None) -> AsyncStreamedAgentResult:
        """Execute the agent loop with streaming output (async).

        Returns an :class:`AsyncStreamedAgentResult` yielding :class:`StreamEvent` objects.
        """
        gen = self._execute_stream(prompt, deps)
        return AsyncStreamedAgentResult(gen)

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

    # ------------------------------------------------------------------
    # Tool wrapping (RunContext injection + ModelRetry + callbacks)
    # ------------------------------------------------------------------

    def _wrap_tools_with_context(self, ctx: RunContext[Any]) -> ToolRegistry:
        """Return a new :class:`ToolRegistry` with wrapped tool functions.

        All wrappers are **sync** so they work with ``ToolRegistry.execute()``.
        For async tool functions, the wrapper uses
        ``asyncio.get_event_loop().run_until_complete()`` as a fallback.
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
            ) -> Callable[..., Any]:
                def wrapper(**kwargs: Any) -> Any:
                    if _cb.on_tool_start:
                        _cb.on_tool_start(_name, kwargs)
                    try:
                        if _wants:
                            call_args = (ctx,)
                        else:
                            call_args = ()

                        if _is_async:
                            coro = _fn(*call_args, **kwargs)
                            # Try to get running loop; if none, use asyncio.run()
                            try:
                                loop = asyncio.get_running_loop()
                            except RuntimeError:
                                loop = None
                            if loop is not None and loop.is_running():
                                # We're inside an async context — create a new thread
                                import concurrent.futures

                                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                                    result = pool.submit(asyncio.run, coro).result()
                            else:
                                result = asyncio.run(coro)
                        else:
                            result = _fn(*call_args, **kwargs)
                    except ModelRetry as exc:
                        result = f"Error: {exc.message}"
                    if _cb.on_tool_end:
                        _cb.on_tool_end(_name, result)
                    return result

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
                    self._extract_steps(conv.messages[-2:], steps, all_tool_calls)

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
        if self._max_cost is None:
            return False
        return session.total_cost >= self._max_cost

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
    ) -> Any:
        """Create a fresh AsyncConversation for a single run."""
        from .async_conversation import AsyncConversation

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

        return AsyncConversation(**kwargs)

    async def _execute(self, prompt: str, steps: list[AgentStep], deps: Any) -> AgentResult:
        """Core async execution: run conversation, extract steps, parse output."""
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

        # 5. Wrap tools with context
        wrapped_tools = self._wrap_tools_with_context(ctx)

        # 6. Build AsyncConversation
        conv = self._build_conversation(
            system_prompt=resolved_system_prompt,
            tools=wrapped_tools if wrapped_tools else None,
            driver_callbacks=driver_callbacks,
        )

        # 7. Fire on_iteration callback
        if self._agent_callbacks.on_iteration:
            self._agent_callbacks.on_iteration(0)

        # 8. Ask the conversation (handles full tool loop internally)
        t0 = time.perf_counter()
        response_text = await conv.ask(effective_prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 9. Extract steps and tool calls
        all_tool_calls: list[dict[str, Any]] = []
        self._extract_steps(conv.messages, steps, all_tool_calls)

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

        # 11. Fire callbacks
        if self._agent_callbacks.on_step:
            for step in steps:
                self._agent_callbacks.on_step(step)
        if self._agent_callbacks.on_output:
            self._agent_callbacks.on_output(result)

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

            if role == "assistant":
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
                                content=msg.get("content", ""),
                                tool_name=name,
                                tool_args=args,
                            )
                        )
                        all_tool_calls.append({"name": name, "arguments": args, "id": tc.get("id", "")})
                else:
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
                    self._extract_steps(conv.messages[-2:], steps, all_tool_calls)

        raise ValueError(
            f"Failed to parse output as {self._output_type.__name__} "
            f"after {_OUTPUT_PARSE_MAX_RETRIES} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    # iter() — async step-by-step
    # ------------------------------------------------------------------

    async def _execute_iter(self, prompt: str, deps: Any) -> AsyncGenerator[AgentStep, None]:
        """Async generator that executes the agent loop and yields each step."""
        self._state = AgentState.running
        self._stop_requested = False
        steps: list[AgentStep] = []

        try:
            result = await self._execute(prompt, steps, deps)
            for step in result.steps:
                yield step
            self._state = AgentState.idle
            # Store result on the generator for retrieval
            self._last_iter_result = result
        except Exception:
            self._state = AgentState.errored
            raise

    # ------------------------------------------------------------------
    # run_stream() — async streaming
    # ------------------------------------------------------------------

    async def _execute_stream(self, prompt: str, deps: Any) -> AsyncGenerator[StreamEvent, None]:
        """Async generator that executes the agent loop and yields stream events."""
        self._state = AgentState.running
        self._stop_requested = False
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
            wrapped_tools = self._wrap_tools_with_context(ctx)
            has_tools = bool(wrapped_tools)

            conv = self._build_conversation(
                system_prompt=resolved_system_prompt,
                tools=wrapped_tools if wrapped_tools else None,
                driver_callbacks=driver_callbacks,
            )

            if self._agent_callbacks.on_iteration:
                self._agent_callbacks.on_iteration(0)

            if has_tools:
                response_text = await conv.ask(effective_prompt)
                yield StreamEvent(event_type=StreamEventType.text_delta, data=response_text)
            else:
                response_text = ""
                async for chunk in conv.ask_stream(effective_prompt):
                    response_text += chunk
                    yield StreamEvent(event_type=StreamEventType.text_delta, data=chunk)

            # Extract steps
            all_tool_calls: list[dict[str, Any]] = []
            self._extract_steps(conv.messages, steps, all_tool_calls)

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
                    self._agent_callbacks.on_step(step)
            if self._agent_callbacks.on_output:
                self._agent_callbacks.on_output(result)

            yield StreamEvent(event_type=StreamEventType.output, data=result)

            self._state = AgentState.idle
            self._last_stream_result = result
        except Exception:
            self._state = AgentState.errored
            raise


# ------------------------------------------------------------------
# AsyncAgentIterator
# ------------------------------------------------------------------


class AsyncAgentIterator:
    """Wraps the :meth:`AsyncAgent.iter` async generator, capturing the final result.

    After async iteration completes, :attr:`result` holds the :class:`AgentResult`.
    """

    def __init__(self, gen: AsyncGenerator[AgentStep, None]) -> None:
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
            agent = self._gen.ag_frame and self._gen.ag_frame.f_locals.get("self")
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

    def __init__(self, gen: AsyncGenerator[StreamEvent, None]) -> None:
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
