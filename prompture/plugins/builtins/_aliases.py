"""Shared helper for building alias :class:`ProviderDescriptor` instances."""

from __future__ import annotations

from ...drivers.provider_descriptors import ProviderDescriptor


def build_alias(
    alias_name: str,
    canonical: ProviderDescriptor,
    modalities: set[str],
) -> ProviderDescriptor:
    """Build an alias :class:`ProviderDescriptor` pointing to *canonical*.

    *modalities* is a set of modality prefixes (``"llm"``, ``"img_gen"``,
    ``"video_gen"``, ``"tts"``, ``"stt"``, ``"embedding"``, ``"rerank"``,
    ``"moderation"``). Each prefix inherits both the sync and async spec.
    """
    return ProviderDescriptor(
        name=alias_name,
        alias_for=canonical.name,
        llm_sync=canonical.llm_sync if "llm" in modalities else None,
        llm_async=canonical.llm_async if "llm" in modalities else None,
        stt_sync=canonical.stt_sync if "stt" in modalities else None,
        stt_async=canonical.stt_async if "stt" in modalities else None,
        tts_sync=canonical.tts_sync if "tts" in modalities else None,
        tts_async=canonical.tts_async if "tts" in modalities else None,
        img_gen_sync=canonical.img_gen_sync if "img_gen" in modalities else None,
        img_gen_async=canonical.img_gen_async if "img_gen" in modalities else None,
        video_gen_sync=canonical.video_gen_sync if "video_gen" in modalities else None,
        video_gen_async=canonical.video_gen_async if "video_gen" in modalities else None,
        embedding_sync=canonical.embedding_sync if "embedding" in modalities else None,
        embedding_async=canonical.embedding_async if "embedding" in modalities else None,
        rerank_sync=canonical.rerank_sync if "rerank" in modalities else None,
        rerank_async=canonical.rerank_async if "rerank" in modalities else None,
        moderation_sync=canonical.moderation_sync if "moderation" in modalities else None,
        moderation_async=canonical.moderation_async if "moderation" in modalities else None,
        is_configured_check=canonical.is_configured_check,
        is_configured_fn=canonical.is_configured_fn,
        always_available=canonical.always_available,
        list_models_kwargs=canonical.list_models_kwargs,
        models_dev_name=None,  # aliases don't map to models.dev independently
    )


def _llm(
    sync_mod: str,
    sync_cls: str,
    async_mod: str,
    async_cls: str,
    kwarg_map: dict[str, str],
    default_model: str,
) -> dict[str, object]:
    """Helper to build ``llm_sync`` + ``llm_async`` :class:`DriverSpec` pair."""
    from ...drivers.provider_descriptors import DriverSpec

    return {
        "llm_sync": DriverSpec(f"{sync_mod}.{sync_cls}", kwarg_map, default_model),
        "llm_async": DriverSpec(f"{async_mod}.{async_cls}", kwarg_map, default_model),
    }
