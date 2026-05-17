"""Built-in OpenAI-compatible provider plugin: Groq, Grok, OpenRouter, Moonshot, ModelScope, Z.ai, generic."""

from __future__ import annotations

from ...drivers.provider_descriptors import DriverSpec, ProviderDescriptor
from ..base import ProviderPlugin
from ._aliases import _llm, build_alias


class OpenAICompatiblePlugin(ProviderPlugin):
    name = "openai_compatible_builtin"
    version = "1.0.0"

    def descriptors(self) -> list[ProviderDescriptor]:
        descs: list[ProviderDescriptor] = []

        # Groq
        groq_kw = {"api_key": "groq_api_key"}
        groq_desc = ProviderDescriptor(
            name="groq",
            **_llm(
                "groq_driver",
                "GroqDriver",
                "async_groq_driver",
                "AsyncGroqDriver",
                groq_kw,
                "groq_model",
            ),
            display_name="Groq",
            is_configured_check="groq_api_key",
            list_models_kwargs=[("api_key", "groq_api_key", "GROQ_API_KEY")],
            models_dev_name="groq",
        )
        descs.append(groq_desc)

        # Grok / xAI
        grok_kw = {"api_key": "grok_api_key"}
        grok_desc = ProviderDescriptor(
            name="grok",
            **_llm(
                "grok_driver",
                "GrokDriver",
                "async_grok_driver",
                "AsyncGrokDriver",
                grok_kw,
                "grok_model",
            ),
            img_gen_sync=DriverSpec("grok_img_gen_driver.GrokImageGenDriver", grok_kw, "grok-2-image"),
            img_gen_async=DriverSpec(
                "async_grok_img_gen_driver.AsyncGrokImageGenDriver",
                grok_kw,
                "grok-2-image",
            ),
            video_gen_sync=DriverSpec("grok_video_gen_driver.GrokVideoGenDriver", grok_kw, "grok_video_model"),
            video_gen_async=DriverSpec(
                "async_grok_video_gen_driver.AsyncGrokVideoGenDriver",
                grok_kw,
                "grok_video_model",
            ),
            display_name="xAI Grok",
            is_configured_check="grok_api_key",
            list_models_kwargs=[("api_key", "grok_api_key", "GROK_API_KEY")],
            models_dev_name="xai",
        )
        descs.append(grok_desc)

        # OpenRouter
        or_kw = {"api_key": "openrouter_api_key"}
        openrouter_desc = ProviderDescriptor(
            name="openrouter",
            **_llm(
                "openrouter_driver",
                "OpenRouterDriver",
                "async_openrouter_driver",
                "AsyncOpenRouterDriver",
                or_kw,
                "openrouter_model",
            ),
            display_name="OpenRouter",
            is_configured_check="openrouter_api_key",
            list_models_kwargs=[("api_key", "openrouter_api_key", "OPENROUTER_API_KEY")],
            models_dev_name="openrouter",
        )
        descs.append(openrouter_desc)

        # Moonshot
        moon_kw = {"api_key": "moonshot_api_key", "endpoint": "moonshot_endpoint"}
        descs.append(
            ProviderDescriptor(
                name="moonshot",
                **_llm(
                    "moonshot_driver",
                    "MoonshotDriver",
                    "async_moonshot_driver",
                    "AsyncMoonshotDriver",
                    moon_kw,
                    "moonshot_model",
                ),
                display_name="Moonshot",
                is_configured_check="moonshot_api_key",
                list_models_kwargs=[
                    ("api_key", "moonshot_api_key", "MOONSHOT_API_KEY"),
                    ("endpoint", "moonshot_endpoint", "MOONSHOT_ENDPOINT"),
                ],
                models_dev_name="moonshotai",
            )
        )

        # ModelScope
        ms_kw = {"api_key": "modelscope_api_key", "endpoint": "modelscope_endpoint"}
        descs.append(
            ProviderDescriptor(
                name="modelscope",
                **_llm(
                    "modelscope_driver",
                    "ModelScopeDriver",
                    "async_modelscope_driver",
                    "AsyncModelScopeDriver",
                    ms_kw,
                    "modelscope_model",
                ),
                display_name="ModelScope",
                is_configured_check="modelscope_api_key",
            )
        )

        # Z.ai (Zhipu)
        zai_kw = {"api_key": "zhipu_api_key", "endpoint": "zhipu_endpoint"}
        zai_desc = ProviderDescriptor(
            name="zai",
            **_llm(
                "zai_driver",
                "ZaiDriver",
                "async_zai_driver",
                "AsyncZaiDriver",
                zai_kw,
                "zhipu_model",
            ),
            display_name="Z.ai (Zhipu)",
            is_configured_check="zhipu_api_key",
            models_dev_name="zai",
        )
        descs.append(zai_desc)

        # Generic OpenAI-compatible profile router
        oac_kw: dict[str, str] = {}
        descs.append(
            ProviderDescriptor(
                name="openai_compatible",
                **_llm(
                    "openai_compatible_driver",
                    "OpenAICompatibleDriver",
                    "async_openai_compatible_driver",
                    "AsyncOpenAICompatibleDriver",
                    oac_kw,
                    "",
                ),
                display_name="OpenAI-Compatible",
                is_configured_check=None,
                list_models_kwargs=[],
                models_dev_name=None,
            )
        )

        # Aliases
        descs.append(build_alias("xai", grok_desc, {"llm", "img_gen", "video_gen"}))
        descs.append(build_alias("zhipu", zai_desc, {"llm"}))

        return descs
