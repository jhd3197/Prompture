"""Prompture's canonical capability set as MCP-ready ``ToolDefinition``s.

This is the SDK-independent core of the MCP integration: one function that
returns Prompture's generation/discovery/pricing surfaces as
:class:`~prompture.agents.tools_schema.ToolDefinition` objects. The MCP server
registers these; the same set is equally usable by agents directly.
"""

from __future__ import annotations

from typing import Any

from ..agents.tools_schema import ToolDefinition, tool_from_function
from ..media.agent_tools import media_tool_definitions

__all__ = [
    "prompture_estimate_cost",
    "prompture_generate",
    "prompture_tool_definitions",
]


def prompture_generate(model: str, prompt: str) -> str:
    """Generate text from an LLM.

    Args:
        model: A ``provider/model`` string, e.g. ``"openai/gpt-4o-mini"``.
        prompt: The prompt to send.

    Returns:
        The generated text.
    """
    from ..drivers import get_driver_for_model

    driver = get_driver_for_model(model)
    resp = driver.generate(prompt, {})
    return resp.get("text", "")


def prompture_estimate_cost(model: str, n: int = 1, duration_seconds: float = 0.0, characters: int = 0) -> float:
    """Estimate the USD cost of a media generation before running it.

    Args:
        model: A ``provider/model`` string, e.g. ``"muapi/kling-video-v2-1"``.
        n: Number of images / items.
        duration_seconds: Clip length for video / lipsync.
        characters: Character count for TTS.

    Returns:
        Estimated cost in USD (0.0 if unknown).
    """
    from ..infra.media_pricing import estimate_media_cost

    return estimate_media_cost(model, n=n, duration_seconds=duration_seconds, characters=characters)


def prompture_tool_definitions(
    *,
    media: bool = True,
    llm: bool = True,
    pricing: bool = True,
    extra: list[ToolDefinition] | None = None,
) -> list[ToolDefinition]:
    """Return Prompture's capabilities as a list of :class:`ToolDefinition`.

    Includes (by default): the media tools (``generate_image`` / ``edit_image`` /
    ``generate_video`` / ``generate_music`` / ``resume_media_job`` /
    ``list_media_models``), text ``generate_text``, and ``estimate_cost``.
    """
    defs: list[ToolDefinition] = []
    if media:
        defs.extend(media_tool_definitions())
    if llm:
        defs.append(tool_from_function(prompture_generate, name="generate_text"))
    if pricing:
        defs.append(tool_from_function(prompture_estimate_cost, name="estimate_cost"))
    if extra:
        defs.extend(extra)
    return defs


def _coerce_tool_definitions(tools: Any) -> list[ToolDefinition]:
    """Accept a list of ToolDefinitions, a ToolRegistry, or None (→ defaults)."""
    if tools is None:
        return prompture_tool_definitions()
    if isinstance(tools, list):
        return tools
    definitions = getattr(tools, "definitions", None)  # ToolRegistry.definitions
    if definitions is not None:
        return list(definitions)
    raise TypeError(f"Cannot derive ToolDefinitions from {type(tools).__name__}")
