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
    "mcp_session_from_stdio",
    "register_mcp_tools",
    "register_mcp_tools_sync",
    "registry_to_skill_dict",
    "skill_to_tool_definition",
    "skills_to_registry",
    "tool_definition_to_skill",
]

# MCP bridge is exported lazily so importing this package never requires
# the optional ``mcp`` dependency (pip install 'prompture[mcp]').
_LAZY_MCP_EXPORTS = {
    "register_mcp_tools",
    "register_mcp_tools_sync",
    "mcp_session_from_stdio",
}


def __getattr__(name: str):
    if name in _LAZY_MCP_EXPORTS:
        from . import mcp_bridge

        return getattr(mcp_bridge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
