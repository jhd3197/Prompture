"""Assistant — a Persona + Skills + Tools + execution backend bundle.

An ``Assistant`` packages everything needed to put an LLM to work on a
task without callers having to wire ``AsyncAgent`` / ``AsyncDeepAgent`` /
coding-agent CLIs themselves.  It composes:

* A :class:`~prompture.agents.persona.Persona` — the system prompt and
  default variables.
* Zero or more :class:`~prompture.agents.skills.SkillInfo` — extra
  instructions appended to the persona under a ``## Skills`` section.
* Optional Python callables passed through as tools to the underlying
  ``AsyncAgent``.
* Exactly one execution backend: either an LLM ``model`` id (handled by
  ``AsyncAgent`` / ``AsyncDeepAgent``) or a ``coding_agent`` CLI id
  routed through :func:`~prompture.infra.coding_agents.run_coding_agent`.

Both backends return an :class:`AssistantResult` with a uniform surface
(``output``, ``cost_usd``, ``input_tokens``, ``output_tokens``,
``session_id``, ``raw``) so callers can swap backends without changing
downstream code.

Quick start::

    from prompture import Assistant, Persona

    web_dev = Assistant(
        name="web-developer",
        persona=Persona(
            name="web_developer",
            system_prompt=(
                "You are a senior web developer. Build {{page_type}} pages "
                "using semantic HTML5 and accessible markup."
            ),
        ),
        model="openai/gpt-4o",
    )

    result = await web_dev.arun(
        "Build the about page for a coffee shop.",
        page_type="landing",
    )
    print(result.output)

Registry usage::

    Assistant.register(web_dev)
    same = Assistant.from_registry("web-developer")

Coding-agent backend (delegate to an installed CLI)::

    cli_dev = Assistant(
        name="cli-developer",
        persona=Persona(name="cli_dev", system_prompt="You write production HTML."),
        coding_agent="claude",  # or "auto" to pick the first healthy CLI
    )
    result = await cli_dev.arun("Build /about.html for a coffee shop.")
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
from collections.abc import AsyncIterator
from typing import Any

from .persona import Persona
from .skills import SkillInfo
from .types import AgentResult

logger = logging.getLogger("prompture.assistant")


# ---------------------------------------------------------------------------
# Result wrapper
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AssistantResult:
    """Uniform result returned by ``Assistant.arun`` regardless of backend.

    ``raw`` holds the underlying object — an :class:`AgentResult` for the
    LLM backend, or a :class:`CodingAgentRunResult` for the coding-agent
    CLI backend — so callers that need backend-specific fields can still
    reach them.
    """

    output: str
    backend: str  # "llm" | "coding_agent"
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    session_id: str | None = None
    raw: Any = None

    @property
    def ok(self) -> bool:
        """Best-effort success check across backends."""
        if self.raw is None:
            return bool(self.output)
        # CodingAgentRunResult has .ok; AgentResult is considered ok if
        # we got non-empty output.
        ok_attr = getattr(self.raw, "ok", None)
        if isinstance(ok_attr, bool):
            return ok_attr
        return bool(self.output)


# ---------------------------------------------------------------------------
# Assistant
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Assistant:
    """A reusable bundle of persona + skills + tools + execution backend.

    Exactly one of ``model`` or ``coding_agent`` must be set.  ``model``
    routes through ``AsyncAgent`` (or ``AsyncDeepAgent`` when
    ``enable_planning=True``); ``coding_agent`` shells out to the named
    CLI via :func:`~prompture.infra.coding_agents.astream_coding_agent`.

    Args:
        name: Short identifier (used by the registry).
        persona: The base :class:`Persona`.
        skills: Optional list of :class:`SkillInfo` whose ``instructions``
            are appended to the persona under a ``## Skills`` section.
        tools: Optional list of Python callables registered as tools on
            the underlying ``AsyncAgent``.  Ignored for the coding-agent
            backend (the CLI has its own toolset).
        model: LLM model id (e.g. ``"openai/gpt-4o"``).  Mutually
            exclusive with ``coding_agent``.
        coding_agent: Coding-agent CLI id (``"claude"``, ``"codex"``,
            ``"aider"``, …) or ``"auto"`` to pick the first healthy CLI.
            Mutually exclusive with ``model``.
        description: Free-form description.
        variables: Default template variables for the persona.  Merged
            with any kwargs passed to ``arun`` / ``astream`` (call-site
            wins).
        enable_planning: When True (and ``model`` is set), wrap the
            ``AsyncAgent`` in :class:`AsyncDeepAgent` so the model gets
            ``write_todos`` and streaming plan updates.
        output_type: Optional Pydantic model passed to ``AsyncAgent`` for
            structured-output parsing.  Ignored for the coding-agent
            backend.
        options: Driver options forwarded to the underlying agent's
            ``options`` kwarg (driver-level settings like
            ``{"max_tokens": 8000, "timeout": 600}``). Distinct from
            ``agent_options`` below, which targets the agent constructor
            itself.
        agent_options: Extra kwargs spread into ``AsyncAgent.__init__``
            when ``enable_planning=False``. Use this for budget controls
            and other agent-level settings the Assistant doesn't expose
            as first-class fields, e.g.
            ``{"max_iterations": 5, "max_cost": 0.50,
            "budget_policy": "hard_stop",
            "fallback_models": ["openai/gpt-4o-mini"]}``.
        coding_agent_options: Extra kwargs forwarded to
            ``run_coding_agent`` / ``astream_coding_agent`` (e.g.
            ``approval_mode``, ``cwd``, ``timeout``).
        deep_agent_options: Extra kwargs spread into
            ``AsyncDeepAgent.__init__`` when ``enable_planning=True``
            (e.g. ``{"enable_vfs": False, "enable_summarization": False,
            "max_iterations": 30}``).
    """

    name: str
    persona: Persona
    skills: tuple[SkillInfo, ...] = ()
    # Either a tuple of Python callables OR a ToolRegistry — both are
    # accepted by the underlying AsyncAgent.
    tools: Any = ()
    model: str | None = None
    coding_agent: str | None = None
    description: str = ""
    variables: dict[str, Any] = dataclasses.field(default_factory=dict)
    enable_planning: bool = False
    output_type: Any = None
    output_key: str | None = None
    options: dict[str, Any] = dataclasses.field(default_factory=dict)
    coding_agent_options: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Extra kwargs forwarded to AsyncAgent.__init__ when enable_planning is
    # False (e.g. {"max_iterations": 5, "max_cost": 0.50,
    # "budget_policy": "hard_stop", "fallback_models": ["openai/gpt-4o-mini"]}).
    # Anything AsyncAgent accepts works here — see its signature for the
    # full list (max_tokens, persistent_conversation, output_key, ...).
    agent_options: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Extra kwargs forwarded to AsyncDeepAgent when enable_planning=True
    # (e.g. {"enable_vfs": False, "enable_summarization": False}).
    deep_agent_options: dict[str, Any] = dataclasses.field(default_factory=dict)

    # ----- construction --------------------------------------------------

    def __post_init__(self) -> None:
        # Normalise skills to an immutable tuple.  Tools may be either a
        # tuple of callables OR a ToolRegistry — both are passed
        # through to AsyncAgent unchanged.
        if not isinstance(self.skills, tuple):
            object.__setattr__(self, "skills", tuple(self.skills))
        if isinstance(self.tools, (list, set)):
            object.__setattr__(self, "tools", tuple(self.tools))

        if bool(self.model) == bool(self.coding_agent):
            raise ValueError(
                "Assistant requires exactly one of `model` or `coding_agent` "
                "to be set (got "
                f"model={self.model!r}, coding_agent={self.coding_agent!r})."
            )
        if self.enable_planning and not self.model:
            raise ValueError(
                "enable_planning=True only applies to the LLM backend (set `model=...`, not `coding_agent=...`)."
            )

    # ----- composition ---------------------------------------------------

    def with_vars(self, **vars: Any) -> Assistant:
        """Return a new Assistant with `vars` merged into `variables`."""
        merged = {**self.variables, **vars}
        return dataclasses.replace(self, variables=merged)

    def with_skills(self, *skills: SkillInfo) -> Assistant:
        """Return a new Assistant with additional skills appended."""
        return dataclasses.replace(self, skills=self.skills + tuple(skills))

    def _composed_persona(self, call_vars: dict[str, Any]) -> Persona:
        """Persona + skill instructions, with all template vars applied."""
        merged_vars = {**self.persona.variables, **self.variables, **call_vars}
        persona = self.persona

        for skill in self.skills:
            block = f"## Skill: {skill.name}\n{skill.description}\n\n{skill.instructions}".rstrip()
            persona = persona.extend(block)

        # Push merged variables back so render() (called by AsyncAgent)
        # has everything it needs.
        if merged_vars and merged_vars != dict(persona.variables):
            persona = dataclasses.replace(persona, variables=dict(merged_vars))
        return persona

    # ----- execution -----------------------------------------------------

    async def arun(
        self,
        prompt: str,
        *,
        deps: Any = None,
        cwd: str | None = None,
        **vars: Any,
    ) -> AssistantResult:
        """Run the assistant to completion.

        Args:
            prompt: User prompt.
            deps: Forwarded to ``AsyncAgent.run(deps=...)`` (LLM backend
                only).  Ignored for the coding-agent backend.
            cwd: Working directory for the coding-agent CLI.  Ignored for
                the LLM backend.
            **vars: Per-call template variables merged into the persona.
        """
        if self.model:
            return await self._run_llm(prompt, deps=deps, call_vars=vars)
        return await self._run_coding_agent(prompt, cwd=cwd, call_vars=vars)

    async def astream(
        self,
        prompt: str,
        *,
        deps: Any = None,
        cwd: str | None = None,
        **vars: Any,
    ) -> AsyncIterator[Any]:
        """Stream events from the assistant.

        For the LLM backend yields :class:`StreamEvent`; for the
        coding-agent backend yields :class:`CodingAgentEvent`.  Callers
        that want a uniform shape should branch on ``assistant.model``
        vs ``assistant.coding_agent`` and adapt accordingly.
        """
        if self.model:
            agent = self._build_async_agent(call_vars=vars)
            async for event in agent.run_stream(prompt, deps=deps):
                yield event
            return

        from ..infra.coding_agents import astream_coding_agent

        agent_id = self._resolve_coding_agent_id()
        task = self._compose_coding_agent_task(prompt, call_vars=vars)
        async for event in astream_coding_agent(
            agent_id,
            task,
            cwd=cwd,
            **self.coding_agent_options,
        ):
            yield event

    # ----- LLM backend ---------------------------------------------------

    def build_async_agent(self, **vars: Any) -> Any:
        """Construct the underlying ``AsyncAgent`` (or ``AsyncDeepAgent``).

        Use this when you want the Assistant as a *configuration* bundle
        (persona + skills + tools + model + options) but need to drive
        the underlying agent yourself — through a group, a streaming
        callback machinery, or any other orchestration layer that
        operates on :class:`AsyncAgent` directly.

        Args:
            **vars: Per-call template variables merged into the persona
                before rendering.  Equivalent to the per-call kwargs of
                :meth:`arun` / :meth:`astream`, but applied at
                construction time.

        Returns:
            An :class:`AsyncAgent` when ``enable_planning=False``,
            otherwise an :class:`AsyncDeepAgent`.

        Raises:
            ValueError: When this Assistant is configured with a
                ``coding_agent`` backend (it shells out to a CLI and has
                no in-process agent to expose).
        """
        if self.coding_agent:
            raise ValueError(
                "build_async_agent() is only available for the LLM backend; "
                f"this Assistant uses coding_agent={self.coding_agent!r}."
            )
        return self._build_async_agent(call_vars=vars)

    def _build_async_agent(self, *, call_vars: dict[str, Any]) -> Any:
        persona = self._composed_persona(call_vars)
        tools_arg = self._tools_for_agent()
        if self.enable_planning:
            from .async_deep_agent import AsyncDeepAgent

            return AsyncDeepAgent(
                self.model or "",
                system_prompt=persona,
                tools=tools_arg,
                name=self.name,
                description=self.description,
                enable_planning=True,
                options=dict(self.options) or None,
                **dict(self.deep_agent_options),
            )

        from .async_agent import AsyncAgent

        return AsyncAgent(
            self.model or "",
            system_prompt=persona,
            tools=tools_arg,
            output_type=self.output_type,
            output_key=self.output_key,
            name=self.name,
            description=self.description,
            options=dict(self.options) or None,
            **dict(self.agent_options),
        )

    def _tools_for_agent(self) -> Any:
        """Coerce ``self.tools`` into the shape ``AsyncAgent`` expects.

        Returns ``None`` when nothing is configured, the underlying
        ``ToolRegistry`` when one was passed in directly, or a list of
        callables otherwise.
        """
        if not self.tools:
            return None
        # Tuples / lists of callables → list (AsyncAgent expects a list).
        if isinstance(self.tools, tuple):
            return list(self.tools) or None
        # Any other object (e.g. ToolRegistry) is passed through verbatim.
        return self.tools

    async def _run_llm(
        self,
        prompt: str,
        *,
        deps: Any,
        call_vars: dict[str, Any],
    ) -> AssistantResult:
        agent = self._build_async_agent(call_vars=call_vars)
        result: AgentResult = await agent.run(prompt, deps=deps)
        # AgentResult.usage is a DICT (prompt_tokens/completion_tokens/cost) —
        # getattr on a dict returns the default, so the attribute-style reads
        # left cost_usd/input_tokens/output_tokens permanently None on the LLM
        # backend and every host that trusted them recorded $0.00 forever.
        usage = getattr(result, "usage", None)
        if isinstance(usage, dict):
            cost_usd = usage.get("cost")
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
        else:
            cost_usd = getattr(usage, "total_cost_usd", None) if usage else None
            input_tokens = getattr(usage, "input_tokens", None) if usage else None
            output_tokens = getattr(usage, "output_tokens", None) if usage else None
        return AssistantResult(
            output=getattr(result, "output", "") or "",
            backend="llm",
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            session_id=None,
            raw=result,
        )

    # ----- coding-agent backend -----------------------------------------

    def _resolve_coding_agent_id(self) -> str:
        agent_id = self.coding_agent or ""
        if agent_id != "auto":
            return agent_id

        from ..infra.discovery import pick_best_coding_agent

        chosen = pick_best_coding_agent(verify=True)
        if chosen is None:
            raise RuntimeError(
                "Assistant coding_agent='auto' but no healthy coding-agent CLI "
                "was found on PATH (tried Claude Code, Codex, Aider, etc.)."
            )
        return chosen.id

    def _compose_coding_agent_task(
        self,
        prompt: str,
        *,
        call_vars: dict[str, Any],
    ) -> str:
        """Coding-agent CLIs don't accept a separate system prompt; the
        persona body is prepended to the task so the CLI still receives
        the role/skills/constraints we configured."""
        persona = self._composed_persona(call_vars)
        rendered = persona.render(**call_vars)
        return f"{rendered}\n\n---\n\n{prompt}"

    async def _run_coding_agent(
        self,
        prompt: str,
        *,
        cwd: str | None,
        call_vars: dict[str, Any],
    ) -> AssistantResult:
        from ..infra.coding_agents import arun_coding_agent

        agent_id = self._resolve_coding_agent_id()
        task = self._compose_coding_agent_task(prompt, call_vars=call_vars)
        result = await arun_coding_agent(
            agent_id,
            task,
            cwd=cwd,
            **self.coding_agent_options,
        )
        return AssistantResult(
            output=getattr(result, "output", "") or "",
            backend="coding_agent",
            cost_usd=getattr(result, "cost_usd", None),
            input_tokens=getattr(result, "input_tokens", None),
            output_tokens=getattr(result, "output_tokens", None),
            session_id=getattr(result, "session_id", None),
            raw=result,
        )

    # ----- sync convenience --------------------------------------------

    def run(self, prompt: str, **kwargs: Any) -> AssistantResult:
        """Blocking wrapper around :meth:`arun` for sync callers."""
        return asyncio.run(self.arun(prompt, **kwargs))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_registry_lock = threading.RLock()
_ASSISTANT_REGISTRY: dict[str, Assistant] = {}


def register_assistant(assistant: Assistant) -> None:
    """Register an Assistant by name (overwrites any prior entry)."""
    with _registry_lock:
        _ASSISTANT_REGISTRY[assistant.name] = assistant


def get_assistant(name: str) -> Assistant | None:
    """Look up a registered Assistant by name."""
    with _registry_lock:
        return _ASSISTANT_REGISTRY.get(name)


def get_assistant_names() -> list[str]:
    """List all registered Assistant names."""
    with _registry_lock:
        return sorted(_ASSISTANT_REGISTRY)


def unregister_assistant(name: str) -> bool:
    """Remove a registered Assistant.  Returns True if it existed."""
    with _registry_lock:
        return _ASSISTANT_REGISTRY.pop(name, None) is not None


def clear_assistant_registry() -> None:
    """Remove every registered Assistant.  Primarily for tests."""
    with _registry_lock:
        _ASSISTANT_REGISTRY.clear()


# Classmethod-style entry points on the Assistant itself for ergonomic use.
def _from_registry(cls: type[Assistant], name: str) -> Assistant:
    found = get_assistant(name)
    if found is None:
        raise KeyError(f"No Assistant registered as {name!r}. Known: {get_assistant_names()}")
    return found


def _register(self: Assistant) -> Assistant:
    register_assistant(self)
    return self


Assistant.from_registry = classmethod(_from_registry)  # type: ignore[attr-defined]
Assistant.register = _register  # type: ignore[attr-defined]


__all__ = [
    "Assistant",
    "AssistantResult",
    "clear_assistant_registry",
    "get_assistant",
    "get_assistant_names",
    "register_assistant",
    "unregister_assistant",
]
