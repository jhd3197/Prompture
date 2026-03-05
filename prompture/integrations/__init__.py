"""Third-party integrations and bridges.

.. deprecated::
    This package has been consolidated into ``prompture.extraction``.
    Import from ``prompture.extraction.tukuy_bridge`` instead.
"""

import contextlib

with contextlib.suppress(ImportError):  # tukuy not installed
    from ..extraction.tukuy_bridge import (
        TukuyChainStep,
        apply_safety_policy,
        apply_security_context,
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
    "discover_and_register_plugins",
    "filter_available_skills",
    "make_transform_chain",
    "registry_to_skill_dict",
    "skill_to_tool_definition",
    "skills_to_registry",
    "tool_definition_to_skill",
]
