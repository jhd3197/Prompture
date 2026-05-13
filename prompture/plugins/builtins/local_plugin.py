"""Built-in local/self-hosted provider plugin: Ollama, LM Studio, local HTTP, AirLLM, HuggingFace, CachiBot."""

from __future__ import annotations

from ...drivers.provider_descriptors import DriverSpec, ProviderDescriptor
from ..base import ProviderPlugin
from ._aliases import _llm, build_alias


class LocalPlugin(ProviderPlugin):
    name = "local_builtin"
    version = "1.0.0"

    def descriptors(self) -> list[ProviderDescriptor]:
        descs: list[ProviderDescriptor] = []

        # Ollama
        ollama_kw = {"endpoint": "ollama_endpoint"}
        descs.append(
            ProviderDescriptor(
                name="ollama",
                **_llm(
                    "ollama_driver",
                    "OllamaDriver",
                    "async_ollama_driver",
                    "AsyncOllamaDriver",
                    ollama_kw,
                    "ollama_model",
                ),
                embedding_sync=DriverSpec(
                    "ollama_embedding_driver.OllamaEmbeddingDriver",
                    ollama_kw,
                    "nomic-embed-text",
                ),
                embedding_async=DriverSpec(
                    "async_ollama_embedding_driver.AsyncOllamaEmbeddingDriver",
                    ollama_kw,
                    "nomic-embed-text",
                ),
                display_name="Ollama",
                always_available=True,
                list_models_kwargs=[("endpoint", "ollama_endpoint", "OLLAMA_ENDPOINT")],
            )
        )

        # LM Studio
        lms_kw = {"endpoint": "lmstudio_endpoint", "api_key": "lmstudio_api_key"}
        lmstudio_desc = ProviderDescriptor(
            name="lmstudio",
            **_llm(
                "lmstudio_driver",
                "LMStudioDriver",
                "async_lmstudio_driver",
                "AsyncLMStudioDriver",
                lms_kw,
                "lmstudio_model",
            ),
            display_name="LM Studio",
            always_available=True,
            list_models_kwargs=[
                ("endpoint", "lmstudio_endpoint", "LMSTUDIO_ENDPOINT"),
                ("api_key", "lmstudio_api_key", "LMSTUDIO_API_KEY"),
            ],
        )
        descs.append(lmstudio_desc)

        # Local HTTP
        local_kw = {"endpoint": "local_http_endpoint"}
        descs.append(
            ProviderDescriptor(
                name="local_http",
                llm_sync=DriverSpec("local_http_driver.LocalHTTPDriver", local_kw, "local_http_model"),
                llm_async=DriverSpec(
                    "async_local_http_driver.AsyncLocalHTTPDriver",
                    local_kw,
                    "local_http_model",
                ),
                display_name="Local HTTP",
                always_available=True,
            )
        )

        # CachiBot
        cb_kw = {"api_key": "cachibot_api_key", "endpoint": "cachibot_endpoint"}
        descs.append(
            ProviderDescriptor(
                name="cachibot",
                **_llm(
                    "cachibot_driver",
                    "CachiBotDriver",
                    "async_cachibot_driver",
                    "AsyncCachiBotDriver",
                    cb_kw,
                    "openai/gpt-4o-mini",
                ),
                display_name="CachiBot",
                is_configured_check="cachibot_api_key",
                list_models_kwargs=[
                    ("api_key", "cachibot_api_key", "CACHIBOT_API_KEY"),
                    ("endpoint", "cachibot_endpoint", "CACHIBOT_ENDPOINT"),
                ],
            )
        )

        # AirLLM
        air_kw = {"compression": "airllm_compression"}
        descs.append(
            ProviderDescriptor(
                name="airllm",
                llm_sync=DriverSpec("airllm_driver.AirLLMDriver", air_kw, "airllm_model"),
                llm_async=DriverSpec("async_airllm_driver.AsyncAirLLMDriver", air_kw, "airllm_model"),
                display_name="AirLLM",
                always_available=True,
            )
        )

        # HuggingFace
        hf_kw = {"endpoint": "hf_endpoint", "token": "hf_token"}  # nosec B105
        hf_desc = ProviderDescriptor(
            name="huggingface",
            **_llm(
                "hugging_driver",
                "HuggingFaceDriver",
                "async_hugging_driver",
                "AsyncHuggingFaceDriver",
                hf_kw,
                "bert-base-uncased",
            ),
            display_name="Hugging Face",
        )
        descs.append(hf_desc)

        # Aliases
        descs.append(build_alias("lm_studio", lmstudio_desc, {"llm"}))
        descs.append(build_alias("lm-studio", lmstudio_desc, {"llm"}))
        descs.append(build_alias("hf", hf_desc, {"llm"}))

        return descs
