"""Bridge between tukuy's skill/chain/safety system and Prompture's agent/tool architecture.

Converts tukuy ``@skill``-decorated functions into Prompture :class:`ToolDefinition`
objects, wraps tukuy :class:`Chain` as pipeline steps, and applies
:class:`SafetyPolicy` to gate tool execution.

All tukuy imports are lazy to avoid import-time errors if tukuy is not installed.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

from ..agents.tools_schema import ToolDefinition, ToolRegistry

logger = logging.getLogger("prompture.tukuy_bridge")


# ------------------------------------------------------------------
# Skill → ToolDefinition
# ------------------------------------------------------------------


def skill_to_tool_definition(skill_or_fn: Any, *, config: dict[str, Any] | None = None) -> ToolDefinition:
    """Convert a tukuy :class:`Skill` or ``@skill``-decorated function to a :class:`ToolDefinition`.

    Uses tukuy's ``bridges._normalize()`` and ``bridges._wrap_as_parameters()``
    to extract name, description, and JSON Schema parameters.  The returned
    tool's ``function`` calls ``skill.invoke(**kwargs)`` and returns
    ``result.value`` on success or ``f"Error: {result.error}"`` on failure.

    Args:
        skill_or_fn: A tukuy ``Skill`` instance or a ``@skill``-decorated function.
        config: Optional configuration dict injected as a
            :class:`SkillContext` into ``invoke()`` when no context is
            already present in kwargs.

    Returns:
        A :class:`ToolDefinition` wrapping the skill.

    Raises:
        TypeError: If *skill_or_fn* is not a recognised tukuy skill type.
    """
    from tukuy.bridges import _normalize, _wrap_as_parameters

    skill_obj = _normalize(skill_or_fn)
    desc = skill_obj.descriptor

    parameters = _wrap_as_parameters(skill_obj)

    # Build wrapper that calls invoke() and unwraps SkillResult
    def _wrapper(**kwargs: Any) -> Any:
        invoke_kwargs = dict(kwargs)
        if config is not None and "context" not in invoke_kwargs:
            from tukuy import SkillContext

            ctx = SkillContext(config=config)
            invoke_kwargs["context"] = ctx
        result = skill_obj.invoke(**invoke_kwargs)
        if result.success:
            return result.value
        return f"Error: {result.error}"

    # For async skill functions, attach a dedicated async wrapper that
    # uses ainvoke() so tukuy's error handling and timing are correct.
    if inspect.iscoroutinefunction(skill_obj.fn):

        async def _async_wrapper(**kwargs: Any) -> Any:
            invoke_kwargs = dict(kwargs)
            if config is not None and "context" not in invoke_kwargs:
                from tukuy import SkillContext

                ctx = SkillContext(config=config)
                invoke_kwargs["context"] = ctx
            result = await skill_obj.ainvoke(**invoke_kwargs)
            if result.success:
                return result.value
            return f"Error: {result.error}"

        _wrapper._async_fn = _async_wrapper  # type: ignore[attr-defined]

    # Attach the original skill for reverse-bridge detection
    _wrapper.__skill__ = skill_obj  # type: ignore[attr-defined]

    return ToolDefinition(
        name=desc.name,
        description=desc.description or f"Call {desc.name}",
        parameters=parameters,
        function=_wrapper,
    )


def skills_to_registry(skills: list[Any]) -> ToolRegistry:
    """Batch-convert tukuy skills to a :class:`ToolRegistry`.

    Args:
        skills: List of tukuy ``Skill`` instances or ``@skill``-decorated functions.

    Returns:
        A populated :class:`ToolRegistry`.
    """
    registry = ToolRegistry()
    for s in skills:
        td = skill_to_tool_definition(s)
        registry.add(td)
    return registry


# ------------------------------------------------------------------
# ToolDefinition → Skill (reverse bridge)
# ------------------------------------------------------------------


def tool_definition_to_skill(td: ToolDefinition) -> Any:
    """Convert a :class:`ToolDefinition` back to a tukuy :class:`Skill`.

    Builds a :class:`SkillDescriptor` from the tool's name, description,
    and parameters, then wraps the tool's function as a ``Skill``.

    Args:
        td: The Prompture tool definition to convert.

    Returns:
        A tukuy ``Skill`` instance.
    """
    from tukuy import Skill, SkillDescriptor

    descriptor = SkillDescriptor(
        name=td.name,
        description=td.description,
        input_schema=td.parameters,
    )
    return Skill(descriptor=descriptor, fn=td.function)


# ------------------------------------------------------------------
# Registry ↔ skill dict
# ------------------------------------------------------------------


def registry_to_skill_dict(registry: ToolRegistry) -> dict[str, Any]:
    """Convert a :class:`ToolRegistry` to a dict compatible with tukuy's ``dispatch_openai()``/``dispatch_anthropic()``.

    For tools that originated from tukuy skills (detected via ``__skill__``
    attribute on the wrapper function), the original decorated function is used.
    For native Prompture tools, a reverse bridge via :func:`tool_definition_to_skill`
    is applied.

    Args:
        registry: The Prompture tool registry.

    Returns:
        A ``{name: skill_or_fn}`` dict usable with tukuy dispatch functions.
    """
    result: dict[str, Any] = {}
    for td in registry.definitions:
        # Check if the tool's function came from a tukuy skill
        skill_obj = getattr(td.function, "__skill__", None)
        if skill_obj is not None:
            result[td.name] = skill_obj
        else:
            result[td.name] = tool_definition_to_skill(td)
    return result


# ------------------------------------------------------------------
# TukuyChainStep — pipeline adapter
# ------------------------------------------------------------------


class TukuyChainStep:
    """Adapter that makes a tukuy :class:`Chain` usable as a :class:`SkillPipeline` step.

    Args:
        chain: A tukuy ``Chain`` instance.
        name: Display name for this step (default ``"tukuy_chain"``).
    """

    def __init__(self, chain: Any, *, name: str = "tukuy_chain") -> None:
        self.chain = chain
        self.name = name

    def run(self, input_text: str) -> str:
        """Execute the chain synchronously.

        Args:
            input_text: The input string to transform.

        Returns:
            The chain's output as a string.
        """
        result = self.chain.run(input_text)
        return str(result)

    async def arun(self, input_text: str) -> str:
        """Execute the chain asynchronously.

        Args:
            input_text: The input string to transform.

        Returns:
            The chain's output as a string.
        """
        result = await self.chain.arun(input_text)
        return str(result)


# ------------------------------------------------------------------
# Safety policy gating
# ------------------------------------------------------------------


def apply_safety_policy(registry: ToolRegistry, policy: Any) -> ToolRegistry:
    """Return a new :class:`ToolRegistry` where tukuy-backed tools are gated by *policy*.

    For each tool whose function has a ``__skill__`` attribute (i.e. it was
    created from a tukuy skill), the wrapper is replaced with one that calls
    ``policy.validate()`` before execution.  If validation fails, the tool
    returns an error string.

    Non-tukuy tools pass through unchanged.

    Args:
        registry: The source tool registry.
        policy: A tukuy ``SafetyPolicy`` instance.

    Returns:
        A new :class:`ToolRegistry` with safety-gated tools.
    """
    new_registry = ToolRegistry()

    for td in registry.definitions:
        skill_obj = getattr(td.function, "__skill__", None)
        if skill_obj is not None:
            # Gate this tool with the safety policy
            original_fn = td.function

            def _make_gated(fn: Callable[..., Any], skill: Any) -> Callable[..., Any]:
                def _gated_wrapper(**kwargs: Any) -> Any:
                    violations = policy.validate(skill.descriptor)
                    if violations:
                        msgs = "; ".join(v.message for v in violations)
                        return f"Error: Safety policy violation: {msgs}"
                    return fn(**kwargs)

                # Preserve the __skill__ attribute
                _gated_wrapper.__skill__ = skill  # type: ignore[attr-defined]
                return _gated_wrapper

            gated = _make_gated(original_fn, skill_obj)
            new_td = ToolDefinition(
                name=td.name,
                description=td.description,
                parameters=td.parameters,
                function=gated,
            )
            new_registry.add(new_td)
        else:
            # Non-tukuy tool: pass through
            new_registry.add(td)

    return new_registry


# ------------------------------------------------------------------
# Security context gating
# ------------------------------------------------------------------


def apply_security_context(registry: ToolRegistry, security_context: Any) -> ToolRegistry:
    """Return a new :class:`ToolRegistry` where tukuy-backed tools run inside *security_context*.

    For each tool whose function has a ``__skill__`` attribute (i.e. it was
    created from a tukuy skill), the wrapper is replaced with one that calls
    ``set_security_context()`` before execution and ``reset_security_context()``
    after (even on error).

    Non-tukuy tools pass through unchanged.

    Args:
        registry: The source tool registry.
        security_context: A tukuy ``SecurityContext`` instance.

    Returns:
        A new :class:`ToolRegistry` with security-scoped tools.
    """
    from tukuy.safety import reset_security_context, set_security_context

    new_registry = ToolRegistry()

    for td in registry.definitions:
        skill_obj = getattr(td.function, "__skill__", None)
        if skill_obj is not None:
            original_fn = td.function

            def _make_scoped(fn: Callable[..., Any], skill: Any) -> Callable[..., Any]:
                def _scoped_wrapper(**kwargs: Any) -> Any:
                    token = set_security_context(security_context)
                    try:
                        return fn(**kwargs)
                    finally:
                        reset_security_context(token)

                _scoped_wrapper.__skill__ = skill  # type: ignore[attr-defined]
                return _scoped_wrapper

            scoped = _make_scoped(original_fn, skill_obj)
            new_td = ToolDefinition(
                name=td.name,
                description=td.description,
                parameters=td.parameters,
                function=scoped,
            )
            new_registry.add(new_td)
        else:
            new_registry.add(td)

    return new_registry


# ------------------------------------------------------------------
# Availability filtering
# ------------------------------------------------------------------


def filter_available_skills(
    registry: ToolRegistry,
    *,
    policy: Any | None = None,
) -> ToolRegistry:
    """Return a new :class:`ToolRegistry` containing only available tukuy skills.

    Uses tukuy's :func:`get_available_skills` with a virtual plugin wrapper
    to filter a :class:`ToolRegistry`.  Non-tukuy tools always pass through.

    Args:
        registry: The source tool registry.
        policy: Optional tukuy ``SafetyPolicy``.  When ``None``, all
            skills are considered available.

    Returns:
        A new :class:`ToolRegistry` with only available tools.
    """
    from tukuy import get_available_skills
    from tukuy.plugins.base import TransformerPlugin

    new_registry = ToolRegistry()

    # Collect tukuy skills for availability check
    tukuy_tools: list[tuple[ToolDefinition, Any]] = []
    for td in registry.definitions:
        skill_obj = getattr(td.function, "__skill__", None)
        if skill_obj is not None:
            tukuy_tools.append((td, skill_obj))
        else:
            # Non-tukuy tools pass through unchanged
            new_registry.add(td)

    if not tukuy_tools:
        return new_registry

    # Build a virtual plugin wrapping the registry's tukuy skills
    skill_dict = {s.descriptor.name: s for _, s in tukuy_tools}

    class _VirtualPlugin(TransformerPlugin):
        @property
        def skills(self) -> dict[str, Any]:
            return skill_dict

        @property
        def transformers(self) -> dict[str, Any]:
            return {}

    virtual = _VirtualPlugin("_prompture_filter")
    available = get_available_skills([virtual], policy=policy)
    available_names = {a.skill.descriptor.name for a in available if a.available}

    for td, skill_obj in tukuy_tools:
        if skill_obj.descriptor.name in available_names:
            new_registry.add(td)

    return new_registry


def discover_and_register_plugins(
    plugins: list[Any],
    *,
    config: dict[str, Any] | None = None,
) -> ToolRegistry:
    """Discover available plugins and register their skills into a new :class:`ToolRegistry`.

    Takes a list of tukuy ``TransformerPlugin`` instances, runs
    :func:`discover_plugins`, and registers all skills from available
    plugins.

    Args:
        plugins: List of tukuy ``TransformerPlugin`` instances.
        config: Optional configuration dict forwarded to
            :func:`skill_to_tool_definition`.

    Returns:
        A :class:`ToolRegistry` populated with skills from available plugins.
    """
    from tukuy import discover_plugins

    registry = ToolRegistry()

    if not plugins:
        return registry

    results = discover_plugins(plugins)
    for result in results:
        if result.available:
            for skill_obj in result.plugin.skills.values():
                td = skill_to_tool_definition(skill_obj, config=config)
                registry.add(td)

    return registry


# ------------------------------------------------------------------
# Transform chain convenience
# ------------------------------------------------------------------


def make_transform_chain(*transforms: str) -> Callable[[str], str]:
    """Return a callable that applies tukuy transforms to a string.

    Convenience wrapper around tukuy's :class:`Chain`.

    Args:
        *transforms: Transform names (e.g. ``"strip"``, ``"lowercase"``).

    Returns:
        A callable ``(str) -> str`` that applies the transforms in order.
    """
    from tukuy import Chain

    chain = Chain(list(transforms))

    def _apply(value: str) -> str:
        return str(chain.run(value))

    return _apply
