"""Built-in embedding + rerank provider plugin: Cohere, Voyage, Jina, Nomic, Mixedbread."""

from __future__ import annotations

from ...drivers.provider_descriptors import DriverSpec, ProviderDescriptor
from ..base import ProviderPlugin
from ._aliases import _llm, build_alias


class EmbeddingRerankPlugin(ProviderPlugin):
    name = "embedding_rerank_builtin"
    version = "1.0.0"

    def descriptors(self) -> list[ProviderDescriptor]:
        descs: list[ProviderDescriptor] = []

        # Cohere (LLM + embeddings + rerank)
        cohere_kw = {"api_key": "cohere_api_key"}
        descs.append(
            ProviderDescriptor(
                name="cohere",
                **_llm(
                    "cohere_driver",
                    "CohereDriver",
                    "async_cohere_driver",
                    "AsyncCohereDriver",
                    cohere_kw,
                    "cohere_model",
                ),
                embedding_sync=DriverSpec(
                    "cohere_embedding_driver.CohereEmbeddingDriver",
                    cohere_kw,
                    "cohere_embedding_model",
                ),
                embedding_async=DriverSpec(
                    "async_cohere_embedding_driver.AsyncCohereEmbeddingDriver",
                    cohere_kw,
                    "cohere_embedding_model",
                ),
                rerank_sync=DriverSpec("cohere_rerank_driver.CohereRerankDriver", cohere_kw, "rerank-v3.5"),
                rerank_async=DriverSpec(
                    "async_cohere_rerank_driver.AsyncCohereRerankDriver",
                    cohere_kw,
                    "rerank-v3.5",
                ),
                display_name="Cohere",
                is_configured_check="cohere_api_key",
                list_models_kwargs=[("api_key", "cohere_api_key", "COHERE_API_KEY")],
                models_dev_name="cohere",
            )
        )

        # Voyage AI
        voyage_kw = {"api_key": "voyage_api_key"}
        descs.append(
            ProviderDescriptor(
                name="voyage",
                embedding_sync=DriverSpec(
                    "voyage_embedding_driver.VoyageEmbeddingDriver",
                    voyage_kw,
                    "voyage_embedding_model",
                ),
                embedding_async=DriverSpec(
                    "async_voyage_embedding_driver.AsyncVoyageEmbeddingDriver",
                    voyage_kw,
                    "voyage_embedding_model",
                ),
                rerank_sync=DriverSpec("voyage_rerank_driver.VoyageRerankDriver", voyage_kw, "rerank-2.5"),
                rerank_async=DriverSpec(
                    "async_voyage_rerank_driver.AsyncVoyageRerankDriver",
                    voyage_kw,
                    "rerank-2.5",
                ),
                display_name="Voyage AI",
                is_configured_check="voyage_api_key",
                models_dev_name=None,
            )
        )

        # Jina AI
        jina_kw = {"api_key": "jina_api_key"}
        descs.append(
            ProviderDescriptor(
                name="jina",
                embedding_sync=DriverSpec(
                    "jina_embedding_driver.JinaEmbeddingDriver",
                    jina_kw,
                    "jina_embedding_model",
                ),
                embedding_async=DriverSpec(
                    "async_jina_embedding_driver.AsyncJinaEmbeddingDriver",
                    jina_kw,
                    "jina_embedding_model",
                ),
                rerank_sync=DriverSpec(
                    "jina_rerank_driver.JinaRerankDriver",
                    jina_kw,
                    "jina-reranker-v2-base-multilingual",
                ),
                rerank_async=DriverSpec(
                    "async_jina_rerank_driver.AsyncJinaRerankDriver",
                    jina_kw,
                    "jina-reranker-v2-base-multilingual",
                ),
                display_name="Jina AI",
                is_configured_check="jina_api_key",
                models_dev_name=None,
            )
        )

        # Nomic
        nomic_kw = {"api_key": "nomic_api_key"}
        descs.append(
            ProviderDescriptor(
                name="nomic",
                embedding_sync=DriverSpec(
                    "nomic_embedding_driver.NomicEmbeddingDriver",
                    nomic_kw,
                    "nomic_embedding_model",
                ),
                embedding_async=DriverSpec(
                    "async_nomic_embedding_driver.AsyncNomicEmbeddingDriver",
                    nomic_kw,
                    "nomic_embedding_model",
                ),
                display_name="Nomic",
                is_configured_check="nomic_api_key",
                models_dev_name=None,
            )
        )

        # Mixedbread
        mxbai_kw = {"api_key": "mixedbread_api_key"}
        mixedbread_desc = ProviderDescriptor(
            name="mixedbread",
            embedding_sync=DriverSpec(
                "mxbai_embedding_driver.MxbaiEmbeddingDriver",
                mxbai_kw,
                "mxbai_embedding_model",
            ),
            embedding_async=DriverSpec(
                "async_mxbai_embedding_driver.AsyncMxbaiEmbeddingDriver",
                mxbai_kw,
                "mxbai_embedding_model",
            ),
            rerank_sync=DriverSpec("mxbai_rerank_driver.MxbaiRerankDriver", mxbai_kw, "mxbai_rerank_model"),
            rerank_async=DriverSpec(
                "async_mxbai_rerank_driver.AsyncMxbaiRerankDriver",
                mxbai_kw,
                "mxbai_rerank_model",
            ),
            display_name="Mixedbread",
            is_configured_check="mixedbread_api_key",
            models_dev_name=None,
        )
        descs.append(mixedbread_desc)

        descs.append(build_alias("mxbai", mixedbread_desc, {"embedding", "rerank"}))

        return descs
