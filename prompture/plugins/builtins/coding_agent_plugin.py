"""Built-in plugin: coding-agent CLIs you are signed in to, used as models.

``claude-code/<model>`` runs Claude Code and ``codex/<model>`` runs Codex CLI (see
``drivers.coding_agent_driver``). Each counts as configured when its CLI is installed.
"""

from __future__ import annotations

from ...drivers.provider_descriptors import DriverSpec, ProviderDescriptor
from ..base import ProviderPlugin


def _installed(driver_name: str):  # type: ignore[no-untyped-def]
    def check(env=None) -> bool:  # type: ignore[no-untyped-def]
        from ...drivers import coding_agent_driver

        return getattr(coding_agent_driver, driver_name).available()

    return check


class CodingAgentPlugin(ProviderPlugin):
    name = "coding_agent_builtin"
    version = "1.0.0"

    def descriptors(self) -> list[ProviderDescriptor]:
        return [
            ProviderDescriptor(
                name="claude-code",
                llm_sync=DriverSpec("coding_agent_driver.ClaudeCodeDriver", {}, "default"),
                llm_async=DriverSpec("async_coding_agent_driver.AsyncClaudeCodeDriver", {}, "default"),
                display_name="Claude Code (your Claude login)",
                is_configured_fn=_installed("ClaudeCodeDriver"),
            ),
            ProviderDescriptor(
                name="codex",
                llm_sync=DriverSpec("coding_agent_driver.CodexDriver", {}, "default"),
                llm_async=DriverSpec("async_coding_agent_driver.AsyncCodexDriver", {}, "default"),
                display_name="Codex (your ChatGPT login)",
                is_configured_fn=_installed("CodexDriver"),
            ),
        ]
