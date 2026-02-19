"""Switchable reasoning strategies for extraction prompts.

Provides a registry of reasoning strategies that augment content prompts
before JSON schema instructions are appended. Pure prompt-level — no driver
changes, no multi-call orchestration.

Example usage::

    from prompture import extract_with_model

    # Use a built-in strategy by name
    result = extract_with_model(
        MyModel, text, model_name="openai/gpt-4",
        reasoning_strategy="plan-and-solve",
    )

    # Register and use a custom strategy
    from prompture import ReasoningStrategy, register_reasoning_strategy

    my_strategy = ReasoningStrategy(
        name="my-cot",
        template="Think step by step.\\n\\n{content_prompt}",
    )
    register_reasoning_strategy("my-cot", my_strategy)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Protocol, Union, runtime_checkable

logger = logging.getLogger("prompture.extraction.reasoning")

# ---------------------------------------------------------------------------
# Protocol & dataclass
# ---------------------------------------------------------------------------


@runtime_checkable
class ReasoningStrategyProtocol(Protocol):
    """Runtime-checkable protocol for reasoning strategies."""

    def augment_prompt(self, content_prompt: str) -> str: ...


@dataclass(frozen=True)
class ReasoningStrategy:
    """A frozen, immutable reasoning strategy.

    Attributes:
        name: Short identifier (e.g. ``"plan-and-solve"``).
        template: Prompt template containing a ``{content_prompt}`` placeholder.
        description: Human-readable summary of the strategy.
    """

    name: str
    template: str
    description: str = ""

    def augment_prompt(self, content_prompt: str) -> str:
        """Insert *content_prompt* into the strategy template."""
        return self.template.format(content_prompt=content_prompt)


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------

PLAN_AND_SOLVE = ReasoningStrategy(
    name="plan-and-solve",
    template=(
        "Let's devise a plan to extract the requested information, then follow it step by step.\n"
        "\n"
        "Plan:\n"
        "1. Identify the entities and fields the schema requires.\n"
        "2. Locate supporting evidence in the source text for each field.\n"
        "3. Verify each value against the text before committing it.\n"
        "4. Assemble the final structured output.\n"
        "\n"
        "{content_prompt}"
    ),
    description="PS+ style: devise a plan, identify entities, locate evidence, verify, assemble.",
)

SELF_DISCOVER = ReasoningStrategy(
    name="self-discover",
    template=(
        "Select and apply the most useful reasoning modules for this extraction task:\n"
        "- Critical reading: identify key claims and supporting details.\n"
        "- Decomposition: break the request into independent sub-extractions.\n"
        "- Evidence grounding: anchor every extracted value to a specific text span.\n"
        "- Type checking: ensure each value matches the expected data type.\n"
        "\n"
        "Apply the selected modules, then extract:\n"
        "\n"
        "{content_prompt}"
    ),
    description="Self-Discover: select reasoning modules, apply, then extract.",
)


class _AutoReasoningStrategy:
    """Sentinel registered under ``"auto"`` so it appears in the registry.

    Calling :meth:`augment_prompt` directly is an error — callers must
    resolve ``"auto"`` via :func:`auto_select_reasoning_strategy` first,
    which needs the raw text and schema that are unavailable through the
    simple protocol.
    """

    name = "auto"
    template = ""
    description = "Auto-select the best strategy (or none) based on text + schema complexity."

    def augment_prompt(self, content_prompt: str) -> str:  # pragma: no cover
        raise RuntimeError(
            "The 'auto' reasoning strategy cannot be applied directly. "
            "Resolve it first with auto_select_reasoning_strategy(text, schema)."
        )


AUTO_REASONING = _AutoReasoningStrategy()

# ---------------------------------------------------------------------------
# Thread-safe registry
# ---------------------------------------------------------------------------

_registry_lock = threading.Lock()
_registry: dict[str, ReasoningStrategy] = {}


def register_reasoning_strategy(
    name: str,
    strategy: ReasoningStrategy,
    *,
    overwrite: bool = False,
) -> None:
    """Register a reasoning strategy under *name*.

    Args:
        name: Registry key (lowercased automatically).
        strategy: The strategy instance to register.
        overwrite: If ``True``, silently replace an existing entry.

    Raises:
        ValueError: If *name* is already registered and *overwrite* is ``False``.
    """
    name = name.lower()
    with _registry_lock:
        if name in _registry and not overwrite:
            raise ValueError(f"Reasoning strategy '{name}' is already registered. Use overwrite=True to replace it.")
        _registry[name] = strategy
    logger.debug("Registered reasoning strategy: %s", name)


def unregister_reasoning_strategy(name: str) -> bool:
    """Remove a strategy from the registry.

    Returns:
        ``True`` if the strategy was present and removed, ``False`` otherwise.
    """
    name = name.lower()
    with _registry_lock:
        if name in _registry:
            del _registry[name]
            logger.debug("Unregistered reasoning strategy: %s", name)
            return True
    return False


def get_reasoning_strategy(name: str) -> ReasoningStrategy:
    """Look up a strategy by name.

    Raises:
        KeyError: If *name* is not registered.
    """
    name = name.lower()
    with _registry_lock:
        if name not in _registry:
            raise KeyError(f"Unknown reasoning strategy: '{name}'")
        return _registry[name]


def list_reasoning_strategies() -> list[str]:
    """Return a sorted list of registered strategy names."""
    with _registry_lock:
        return sorted(_registry)


def reset_reasoning_strategy_registry() -> None:
    """Clear the registry and re-register built-in strategies.

    Intended for testing teardown.
    """
    with _registry_lock:
        _registry.clear()
    _register_builtins()


# ---------------------------------------------------------------------------
# apply helper
# ---------------------------------------------------------------------------


def auto_select_reasoning_strategy(
    text: str,
    schema: dict[str, Any],
) -> str | None:
    """Pick the best reasoning strategy (or *None*) based on heuristics.

    Uses :class:`~prompture.pipeline.routing.ModelRouter` to compute a
    complexity score from *text* and *schema*, then maps the score to a
    strategy:

    * **< 0.3** → ``None`` (simple extraction, skip overhead)
    * **0.3 – 0.6, or ``requires_reasoning``** → ``"plan-and-solve"``
    * **> 0.6** → ``"self-discover"``

    The import of ``ModelRouter`` is deferred to avoid circular
    dependencies.

    Args:
        text: Raw input text that will be sent to the LLM.
        schema: JSON schema dict describing the expected output.

    Returns:
        A strategy name (``"plan-and-solve"`` or ``"self-discover"``) or
        ``None`` if no strategy is beneficial.
    """
    # Lazy import to avoid circular deps (routing → extraction → reasoning)
    from ..pipeline.routing import ModelRouter

    router = ModelRouter()
    analysis = router.analyze_complexity(text, schema)

    if analysis.complexity_score > 0.6:
        logger.debug(
            "auto reasoning: score=%.2f → self-discover",
            analysis.complexity_score,
        )
        return "self-discover"

    if analysis.complexity_score >= 0.3 or analysis.requires_reasoning:
        logger.debug(
            "auto reasoning: score=%.2f requires_reasoning=%s → plan-and-solve",
            analysis.complexity_score,
            analysis.requires_reasoning,
        )
        return "plan-and-solve"

    logger.debug(
        "auto reasoning: score=%.2f → None (skip)",
        analysis.complexity_score,
    )
    return None


def _strategy_name(strategy: Union[str, ReasoningStrategyProtocol, None]) -> str | None:
    """Return a serialisable name for *strategy*, or ``None``."""
    if strategy is None:
        return None
    if isinstance(strategy, str):
        return strategy
    return getattr(strategy, "name", type(strategy).__name__)


def apply_reasoning_strategy(
    content_prompt: str,
    strategy: Union[str, ReasoningStrategyProtocol, None],
) -> str:
    """Resolve and apply a reasoning strategy to *content_prompt*.

    Args:
        content_prompt: The original prompt text.
        strategy: A strategy name (looked up in the registry), a strategy
            instance implementing ``augment_prompt``, or ``None`` (no-op).

    Returns:
        The (possibly augmented) prompt string.

    Raises:
        KeyError: If a string name is not found in the registry.
        TypeError: If *strategy* is neither ``None``, a ``str``, nor a
            ``ReasoningStrategyProtocol`` implementor.
    """
    if strategy is None:
        return content_prompt

    if isinstance(strategy, str):
        resolved = get_reasoning_strategy(strategy)
        return resolved.augment_prompt(content_prompt)

    if isinstance(strategy, ReasoningStrategyProtocol):
        return strategy.augment_prompt(content_prompt)

    raise TypeError(
        f"reasoning_strategy must be None, a string, or implement "
        f"ReasoningStrategyProtocol, got {type(strategy).__name__}"
    )


# ---------------------------------------------------------------------------
# Auto-register built-ins on import
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    for s in (PLAN_AND_SOLVE, SELF_DISCOVER):
        register_reasoning_strategy(s.name, s, overwrite=True)
    # Register the auto sentinel so list/get work
    register_reasoning_strategy(AUTO_REASONING.name, AUTO_REASONING, overwrite=True)  # type: ignore[arg-type]


_register_builtins()
