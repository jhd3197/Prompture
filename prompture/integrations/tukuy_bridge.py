"""Deprecated alias for :mod:`prompture.extraction.tukuy_bridge`.

.. deprecated::
    This module moved to ``prompture.extraction.tukuy_bridge``.  Import from
    there instead.

Everything is re-exported from the canonical module rather than duplicated.
Two copies of the source meant two *distinct* ``current_tool_call_id``
ContextVars, so a tool executed via one module's var was invisible to code
reading the other's — and any fix applied to one copy silently skipped the
other.
"""

from __future__ import annotations

from ..extraction.tukuy_bridge import (
    TukuyChainStep,
    apply_safety_policy,
    apply_security_context,
    current_tool_call_id,
    discover_and_register_plugins,
    filter_available_skills,
    make_transform_chain,
    registry_to_skill_dict,
    skill_to_tool_definition,
    skills_to_registry,
    tool_definition_to_skill,
)

__all__ = [
    "TukuyChainStep",
    "apply_safety_policy",
    "apply_security_context",
    "current_tool_call_id",
    "discover_and_register_plugins",
    "filter_available_skills",
    "make_transform_chain",
    "registry_to_skill_dict",
    "skill_to_tool_definition",
    "skills_to_registry",
    "tool_definition_to_skill",
]
