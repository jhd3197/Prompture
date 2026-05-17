"""Built-in OpenAI provider plugin (LLM, STT, TTS, image, embedding, moderation)."""

from __future__ import annotations

from ...drivers.provider_descriptors import DriverSpec, ProviderDescriptor
from ..base import ProviderPlugin
from ._aliases import _llm, build_alias


class OpenAIPlugin(ProviderPlugin):
    name = "openai_builtin"
    version = "1.0.0"

    def descriptors(self) -> list[ProviderDescriptor]:
        oai_kw = {"api_key": "openai_api_key"}
        openai_desc = ProviderDescriptor(
            name="openai",
            **_llm(
                "openai_driver",
                "OpenAIDriver",
                "async_openai_driver",
                "AsyncOpenAIDriver",
                oai_kw,
                "openai_model",
            ),
            stt_sync=DriverSpec("openai_stt_driver.OpenAISTTDriver", oai_kw, "whisper-1"),
            stt_async=DriverSpec("async_openai_stt_driver.AsyncOpenAISTTDriver", oai_kw, "whisper-1"),
            tts_sync=DriverSpec("openai_tts_driver.OpenAITTSDriver", oai_kw, "tts-1"),
            tts_async=DriverSpec("async_openai_tts_driver.AsyncOpenAITTSDriver", oai_kw, "tts-1"),
            img_gen_sync=DriverSpec("openai_img_gen_driver.OpenAIImageGenDriver", oai_kw, "dall-e-3"),
            img_gen_async=DriverSpec("async_openai_img_gen_driver.AsyncOpenAIImageGenDriver", oai_kw, "dall-e-3"),
            embedding_sync=DriverSpec(
                "openai_embedding_driver.OpenAIEmbeddingDriver",
                oai_kw,
                "text-embedding-3-small",
            ),
            embedding_async=DriverSpec(
                "async_openai_embedding_driver.AsyncOpenAIEmbeddingDriver",
                oai_kw,
                "text-embedding-3-small",
            ),
            moderation_sync=DriverSpec(
                "openai_moderation_driver.OpenAIModerationDriver",
                oai_kw,
                "omni-moderation-latest",
            ),
            moderation_async=DriverSpec(
                "async_openai_moderation_driver.AsyncOpenAIModerationDriver",
                oai_kw,
                "omni-moderation-latest",
            ),
            display_name="OpenAI",
            is_configured_check="openai_api_key",
            list_models_kwargs=[("api_key", "openai_api_key", "OPENAI_API_KEY")],
            models_dev_name="openai",
        )
        return [
            openai_desc,
            build_alias("chatgpt", openai_desc, {"llm", "embedding", "moderation"}),
            build_alias("dalle", openai_desc, {"img_gen"}),
        ]
