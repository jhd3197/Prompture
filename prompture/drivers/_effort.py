"""One reasoning-effort option for every provider.

Callers pass ``options["effort"]`` (``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``, ``"max"``,
or ``"none"`` where a model can switch reasoning off); ``"reasoning_effort"`` is accepted as
the same thing. Each driver family puts it where its API reads it:

* ``"anthropic"``: ``output_config.effort`` (see ``claude_driver``).
* ``"openai"``: ``reasoning_effort`` on Chat Completions style APIs (OpenAI, Azure, Groq, xAI,
  OpenAI-compatible servers).
* ``"openrouter"``: ``reasoning.effort``.
* ``"google"``: ``thinking_config.thinking_level`` (model-dependent).

Explicit provider-specific fields take precedence. Google maps generic effort to
the model's thinking levels; other providers validate the supplied value themselves.
"""

from __future__ import annotations

from typing import Any

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def requested_effort(options: dict[str, Any] | None) -> str | None:
    """The effort a call asks for (``effort``, else ``reasoning_effort``), or None."""
    if not options:
        return None
    value = options.get("effort") or options.get("reasoning_effort")
    return str(value) if value else None


def google_thinking_level(effort: str, model: str = "") -> str:
    """Map effort without sending unsupported levels to older Gemini models."""
    if model.removeprefix("models/") == "gemini-4-argon" and effort == "medium":
        return "medium"
    return "low" if effort in ("none", "minimal", "low", "medium") else "high"


def apply_effort(request: dict[str, Any], options: dict[str, Any] | None, style: str, *, model: str = "") -> None:
    """Put the requested effort into ``request`` the way ``style``'s API takes it."""
    effort = requested_effort(options)
    if effort is None:
        return
    if style == "openai":
        request.setdefault("reasoning_effort", effort)
    elif style == "openrouter":
        reasoning = request.setdefault("reasoning", {})
        if isinstance(reasoning, dict):
            reasoning.setdefault("effort", effort)
    elif style == "google":
        thinking = request.setdefault("thinking_config", {})
        if isinstance(thinking, dict) and "thinking_budget" not in thinking:
            thinking.setdefault("thinking_level", google_thinking_level(effort, model))
    else:
        raise ValueError(f"Unknown effort style: {style!r}")
