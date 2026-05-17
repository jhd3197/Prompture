"""Built-in misc/extras provider plugin: Mistral, DeepSeek."""

from __future__ import annotations

from ...drivers.provider_descriptors import DriverSpec, ProviderDescriptor
from ..base import ProviderPlugin
from ._aliases import _llm, build_alias


class ExtrasPlugin(ProviderPlugin):
    name = "extras_builtin"
    version = "1.0.0"

    def descriptors(self) -> list[ProviderDescriptor]:
        descs: list[ProviderDescriptor] = []

        # Mistral AI
        mistral_kw = {"api_key": "mistral_api_key"}
        mistral_desc = ProviderDescriptor(
            name="mistral",
            **_llm(
                "mistral_driver",
                "MistralDriver",
                "async_mistral_driver",
                "AsyncMistralDriver",
                mistral_kw,
                "mistral_model",
            ),
            moderation_sync=DriverSpec(
                "mistral_moderation_driver.MistralModerationDriver",
                mistral_kw,
                "mistral-moderation-latest",
            ),
            moderation_async=DriverSpec(
                "async_mistral_moderation_driver.AsyncMistralModerationDriver",
                mistral_kw,
                "mistral-moderation-latest",
            ),
            display_name="Mistral AI",
            is_configured_check="mistral_api_key",
            list_models_kwargs=[("api_key", "mistral_api_key", "MISTRAL_API_KEY")],
            models_dev_name="mistral",
        )
        descs.append(mistral_desc)

        # DeepSeek
        ds_kw = {"api_key": "deepseek_api_key"}
        descs.append(
            ProviderDescriptor(
                name="deepseek",
                **_llm(
                    "deepseek_driver",
                    "DeepSeekDriver",
                    "async_deepseek_driver",
                    "AsyncDeepSeekDriver",
                    ds_kw,
                    "deepseek_model",
                ),
                display_name="DeepSeek",
                is_configured_check="deepseek_api_key",
                list_models_kwargs=[("api_key", "deepseek_api_key", "DEEPSEEK_API_KEY")],
                models_dev_name="deepseek",
            )
        )

        descs.append(build_alias("mistralai", mistral_desc, {"llm", "moderation"}))

        return descs
