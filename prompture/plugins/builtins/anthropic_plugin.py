"""Built-in Anthropic/Claude provider plugin."""

from __future__ import annotations

from ...drivers.provider_descriptors import ProviderDescriptor
from ..base import ProviderPlugin
from ._aliases import _llm, build_alias


class AnthropicPlugin(ProviderPlugin):
    name = "anthropic_builtin"
    version = "1.0.0"

    def descriptors(self) -> list[ProviderDescriptor]:
        claude_kw = {"api_key": "claude_api_key"}
        claude_desc = ProviderDescriptor(
            name="claude",
            **_llm(
                "claude_driver",
                "ClaudeDriver",
                "async_claude_driver",
                "AsyncClaudeDriver",
                claude_kw,
                "claude_model",
            ),
            display_name="Anthropic",
            is_configured_check="claude_api_key",
            list_models_kwargs=[("api_key", "claude_api_key", "CLAUDE_API_KEY")],
            models_dev_name="anthropic",
        )
        return [claude_desc, build_alias("anthropic", claude_desc, {"llm"})]
