"""Discovery module for auto-detecting available models."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import sys
from typing import Any, Literal, overload

from ..drivers.provider_descriptors import PROVIDER_DESCRIPTOR_MAP, ProviderDescriptor, _resolve_cls
from .cache import MemoryCacheBackend
from .provider_env import ProviderEnvironment
from .settings import settings

logger = logging.getLogger(__name__)

# Default TTL for discovery cache (seconds)
_DISCOVERY_CACHE_TTL = 300  # 5 minutes

# Module-level cache for discovery results.  Thread-safe via MemoryCacheBackend's
# internal lock.  Keyed by a string describing the call parameters.
_discovery_cache = MemoryCacheBackend(maxsize=32)


def _cfg_value(
    env: ProviderEnvironment | None,
    attr: str,
    env_var: str | None = None,
) -> str | None:
    """Resolve a config value: env -> settings -> os.getenv.

    When *env* is provided, checks ``env.<attr>`` first, then ``settings.<attr>``.
    When *env* is ``None``, checks ``settings.<attr>`` then ``os.getenv(env_var)``.
    """
    if env is not None:
        return env.resolve(attr)
    return getattr(settings, attr, None) or (os.getenv(env_var) if env_var else None)


def _is_provider_configured(provider_name: str, env: ProviderEnvironment | None = None) -> bool:
    """Determine if a provider is configured, using its ProviderDescriptor metadata."""
    desc = PROVIDER_DESCRIPTOR_MAP.get(provider_name)
    if desc is None:
        return False

    # Skip aliases — only check canonical providers.
    if desc.alias_for is not None:
        return False

    # Always-available providers (ollama, lmstudio, airllm).
    if desc.always_available:
        # local_http is special: only available if LOCAL_HTTP_ENDPOINT is set.
        return not (provider_name == "local_http" and not os.getenv("LOCAL_HTTP_ENDPOINT"))

    # Complex check (e.g. Azure).
    if desc.is_configured_fn is not None:
        return desc.is_configured_fn(env=env)

    # Simple check: a settings attribute must be truthy.
    if desc.is_configured_check:
        env_var = desc.is_configured_check.upper()
        return bool(_cfg_value(env, desc.is_configured_check, env_var))

    return False


def _get_list_models_kwargs(
    provider: str,
    env: ProviderEnvironment | None = None,
) -> dict[str, Any]:
    """Return keyword arguments for ``driver_cls.list_models()`` based on provider config."""
    desc = PROVIDER_DESCRIPTOR_MAP.get(provider)
    if desc is None:
        return {}

    kw: dict[str, Any] = {}
    for ctor_kwarg, settings_attr, env_var in desc.list_models_kwargs:
        kw[ctor_kwarg] = _cfg_value(env, settings_attr, env_var)
    return kw


_SOURCE_LABELS: dict[str, str] = {
    "api": "live",
    "static": "known",
    "catalog": "catalog",
}


def _config_via(desc: ProviderDescriptor) -> str:
    """Derive the env var or label that provides access for this provider."""
    if desc.always_available:
        return "local"
    if desc.is_configured_check:
        return desc.is_configured_check.upper()
    # Complex providers (Azure) — use name-based convention.
    return f"{desc.name.upper()}_API_KEY"


def _get_driver_class(provider_name: str) -> type | None:
    """Resolve the sync LLM driver class for a provider from its descriptor."""
    desc = PROVIDER_DESCRIPTOR_MAP.get(provider_name)
    if desc is None or desc.llm_sync is None:
        return None
    return _resolve_cls(desc.llm_sync.cls_path)


@overload
def get_available_models(
    *,
    env: ProviderEnvironment | None = None,
    include_capabilities: Literal[False] = False,
    include_source: Literal[False] = False,
    grouped: Literal[False] = False,
    verified_only: bool = False,
    force_refresh: bool = False,
    cache_ttl: int | None = None,
) -> list[str]: ...


@overload
def get_available_models(  # type: ignore[overload-overlap]
    *,
    env: ProviderEnvironment | None = None,
    include_capabilities: Literal[True],
    include_source: bool = False,
    grouped: Literal[False] = False,
    verified_only: bool = False,
    force_refresh: bool = False,
    cache_ttl: int | None = None,
) -> list[dict[str, Any]]: ...


@overload
def get_available_models(
    *,
    env: ProviderEnvironment | None = None,
    include_capabilities: Literal[False] = False,
    include_source: Literal[True] = ...,
    grouped: Literal[False] = False,
    verified_only: bool = False,
    force_refresh: bool = False,
    cache_ttl: int | None = None,
) -> list[dict[str, str]]: ...


@overload
def get_available_models(
    *,
    env: ProviderEnvironment | None = None,
    include_capabilities: bool = False,
    include_source: bool = False,
    grouped: Literal[True] = ...,
    verified_only: bool = False,
    force_refresh: bool = False,
    cache_ttl: int | None = None,
) -> dict[str, dict[str, Any]]: ...


def get_available_models(
    *,
    env: ProviderEnvironment | None = None,
    include_capabilities: bool = False,
    include_source: bool = False,
    grouped: bool = False,
    verified_only: bool = False,
    force_refresh: bool = False,
    cache_ttl: int | None = None,
) -> list[str] | list[dict[str, Any]] | list[dict[str, str]] | dict[str, dict[str, Any]]:
    """Auto-detect available models based on configured drivers and environment variables.

    Iterates through supported providers and checks if they are configured
    (e.g. API key present).  For static drivers, returns models from their
    ``MODEL_PRICING`` keys.  For dynamic drivers (like Ollama), attempts to
    fetch available models from the endpoint.

    Results are cached in memory with a configurable TTL to avoid redundant
    provider queries on repeated calls.

    Args:
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton (current behavior).
        include_capabilities: When ``True``, return enriched dicts with
            ``model``, ``provider``, ``model_id``, and ``capabilities``
            fields instead of plain ``"provider/model_id"`` strings.
        include_source: When ``True`` (and ``grouped=False`` and
            ``include_capabilities=False``), return ``list[dict]`` with
            ``{"model": ..., "source": "live"|"known"|"catalog"}`` instead
            of plain strings.
        grouped: When ``True``, return a ``dict`` keyed by provider name
            with ``display_name``, ``source``, ``config_via``, and
            ``models`` fields.
        verified_only: When ``True``, only return models that have been
            successfully used (as recorded by the usage ledger).
        force_refresh: When ``True``, bypass the cache and re-query all
            providers.
        cache_ttl: Cache time-to-live in seconds.  Defaults to
            ``_DISCOVERY_CACHE_TTL`` (300 s / 5 min).

    Returns:
        A sorted list of unique model strings (default), enriched dicts,
        source-annotated dicts, or a grouped dict by provider.
    """
    ttl = cache_ttl if cache_ttl is not None else _DISCOVERY_CACHE_TTL

    # Build cache key — include a hash of env-provided fields to avoid cross-bot pollution
    mode = "grouped" if grouped else ("source" if include_source else str(include_capabilities))
    if env is not None:
        env_sig = tuple(sorted(k for k, v in vars(env).items() if v is not None))
        env_hash = hashlib.md5(str(env_sig).encode(), usedforsecurity=False).hexdigest()[:8]
        cache_key = f"models:{mode}:{verified_only}:env={env_hash}"
    else:
        cache_key = f"models:{mode}:{verified_only}"

    if not force_refresh:
        cached = _discovery_cache.get(cache_key)
        if cached is not None:
            logger.debug("Discovery cache hit for key=%s", cache_key)
            return cached  # type: ignore[no-any-return]

    logger.debug("Discovery cache miss for key=%s — querying providers", cache_key)
    # Maps model string -> source ("api", "static", "catalog"). First source wins.
    model_sources: dict[str, str] = {}
    configured_providers: set[str] = set()
    # Providers where list_models() returned an authoritative model list.
    # For these, we trust the API response and skip MODEL_PRICING / models.dev additions.
    api_authoritative_providers: set[str] = set()

    # Iterate canonical providers from descriptors (skip aliases).
    for provider_name, desc in PROVIDER_DESCRIPTOR_MAP.items():
        if desc.alias_for is not None:
            continue

        try:
            if not _is_provider_configured(provider_name, env):
                continue

            configured_providers.add(provider_name)

            driver_cls = _get_driver_class(provider_name)
            if driver_cls is None:
                continue

            # API-based Detection: call driver's list_models() classmethod.
            # When this succeeds, the API response is authoritative — it reflects
            # exactly which models the API key has access to.
            api_kwargs = _get_list_models_kwargs(provider_name, env=env)
            api_succeeded = False
            try:
                api_models = driver_cls.list_models(**api_kwargs)  # type: ignore[attr-defined]
                if api_models is not None:
                    api_succeeded = True
                    api_authoritative_providers.add(provider_name)
                    for model_id in api_models:
                        model_str = f"{provider_name}/{model_id}"
                        if model_str not in model_sources:
                            model_sources[model_str] = "api"
            except Exception as e:
                logger.warning("list_models() failed for %s: %s", provider_name, e)

            # Static Detection: Get models from the capabilities knowledge base
            # (JSON rate files) when the API didn't return an authoritative list
            # (failed, returned None, or the driver doesn't implement list_models).
            if not api_succeeded:
                from .model_rates import get_kb_models_for_provider

                for model_id in get_kb_models_for_provider(provider_name):
                    model_str = f"{provider_name}/{model_id}"
                    if model_str not in model_sources:
                        model_sources[model_str] = "static"

        except Exception as e:
            logger.warning(f"Error detecting models for provider {provider_name}: {e}")
            continue

    # Enrich with live model list from models.dev cache — but only for providers
    # where the API didn't return an authoritative list.  When list_models()
    # succeeded, the API already told us exactly which models the key can use;
    # adding the full models.dev catalog would surface models the user can't
    # actually access (e.g. GPT-5 when the key only has GPT-4o access).
    from .model_rates import PROVIDER_MAP, get_all_provider_models

    for prompture_name, api_name in PROVIDER_MAP.items():
        if prompture_name in configured_providers and prompture_name not in api_authoritative_providers:
            for model_id in get_all_provider_models(api_name):
                model_str = f"{prompture_name}/{model_id}"
                if model_str not in model_sources:
                    model_sources[model_str] = "catalog"

    sorted_models = sorted(model_sources.keys())

    # --- verified_only filtering ---
    verified_set: set[str] | None = None
    if verified_only or include_capabilities:
        try:
            from .ledger import _get_ledger

            ledger = _get_ledger()
            verified_set = ledger.get_verified_models()
        except Exception:
            logger.debug("Could not load ledger for verified models", exc_info=True)
            verified_set = set()

    if verified_only and verified_set is not None:
        sorted_models = [m for m in sorted_models if m in verified_set]

    # ── grouped mode ────────────────────────────────────────────────────
    if grouped:
        groups: dict[str, dict[str, Any]] = {}
        for model_str in sorted_models:
            provider = model_str.split("/", 1)[0]
            model_id = model_str.split("/", 1)[1] if "/" in model_str else model_str
            if provider not in groups:
                prov_desc = PROVIDER_DESCRIPTOR_MAP.get(provider)
                source_raw = model_sources.get(model_str, "static")
                groups[provider] = {
                    "display_name": prov_desc.display_name if prov_desc else provider,
                    "source": _SOURCE_LABELS.get(source_raw, source_raw),
                    "config_via": _config_via(prov_desc) if prov_desc else "unknown",
                    "models": [],
                }
            groups[provider]["models"].append(model_id)
        _discovery_cache.set(cache_key, groups, ttl=ttl)
        return groups

    # ── include_source mode ───────────────────────────────────────────
    if include_source and not include_capabilities:
        source_list: list[dict[str, str]] = [
            {
                "model": m,
                "source": _SOURCE_LABELS.get(model_sources.get(m, "static"), model_sources.get(m, "static")),
            }
            for m in sorted_models
        ]
        _discovery_cache.set(cache_key, source_list, ttl=ttl)
        return source_list

    if not include_capabilities:
        _discovery_cache.set(cache_key, sorted_models, ttl=ttl)
        return sorted_models

    # Build enriched dicts with capabilities and lifecycle from models.dev
    from .model_rates import get_model_capabilities, get_model_lifecycle

    # Fetch all ledger stats for annotation (keyed by model_name)
    ledger_stats: dict[str, dict[str, Any]] = {}
    try:
        from .ledger import _get_ledger

        for row in _get_ledger().get_all_stats():
            name = row["model_name"]
            if name not in ledger_stats:
                ledger_stats[name] = row
            else:
                # Aggregate across API key hashes
                existing = ledger_stats[name]
                existing["use_count"] += row["use_count"]
                existing["total_tokens"] += row["total_tokens"]
                existing["total_cost"] += row["total_cost"]
                if row["last_used"] > existing["last_used"]:
                    existing["last_used"] = row["last_used"]
    except Exception:
        logger.debug("Could not load ledger stats for enrichment", exc_info=True)

    enriched: list[dict[str, Any]] = []
    for model_str in sorted_models:
        parts = model_str.split("/", 1)
        provider = parts[0]
        model_id = parts[1] if len(parts) > 1 else parts[0]

        caps = get_model_capabilities(provider, model_id)
        caps_dict = dataclasses.asdict(caps) if caps is not None else None

        lifecycle = get_model_lifecycle(provider, model_id)

        entry: dict[str, Any] = {
            "model": model_str,
            "provider": provider,
            "model_id": model_id,
            "capabilities": caps_dict,
            "source": model_sources.get(model_str, "unknown"),
            "verified": verified_set is not None and model_str in verified_set,
            "status": lifecycle.get("status") if lifecycle else None,
            "family": lifecycle.get("family") if lifecycle else None,
            "release_date": lifecycle.get("release_date") if lifecycle else None,
            "superseded_by": lifecycle.get("superseded_by") if lifecycle else None,
            "end_of_support": lifecycle.get("end_of_support") if lifecycle else None,
        }

        stats = ledger_stats.get(model_str)
        if stats:
            entry["last_used"] = stats["last_used"]
            entry["use_count"] = stats["use_count"]
        else:
            entry["last_used"] = None
            entry["use_count"] = 0

        enriched.append(entry)

    _discovery_cache.set(cache_key, enriched, ttl=ttl)
    return enriched


def display_available_models(
    *,
    env: ProviderEnvironment | None = None,
    force_refresh: bool = False,
    return_string: bool = False,
) -> str | None:
    """Print a human-friendly grouped view of available models.

    Args:
        env: Optional per-consumer environment for isolated API keys.
        force_refresh: Bypass cache and re-query providers.
        return_string: When ``True``, return the formatted string instead
            of printing it.

    Returns:
        The formatted string when *return_string* is ``True``, else ``None``.
    """
    grouped = get_available_models(grouped=True, env=env, force_refresh=force_refresh)

    lines: list[str] = []
    lines.append("Available Models")
    lines.append("=" * 60)
    lines.append("")

    for provider, info in sorted(grouped.items()):
        display = info.get("display_name") or provider
        config = info.get("config_via", "unknown")
        source = info.get("source", "unknown")
        models: list[str] = info.get("models", [])
        count = len(models)

        if config == "local":
            header = f"{display} (local)  [{source} - {count} model{'s' if count != 1 else ''}]"
        else:
            header = f"{display} (via {config})  [{source} - {count} model{'s' if count != 1 else ''}]"
        lines.append(header)

        # 3-column layout
        col_width = 24
        for i in range(0, len(models), 3):
            row = models[i : i + 3]
            lines.append("  " + "".join(m.ljust(col_width) for m in row).rstrip())

        lines.append("")

    lines.append("Legend: live = confirmed by API | known = from rate files | catalog = from models.dev")

    output = "\n".join(lines)

    if return_string:
        return output

    print(output, file=sys.stdout)
    return None


def clear_discovery_cache() -> None:
    """Clear the in-memory discovery cache, forcing the next call to re-query providers."""
    _discovery_cache.clear()


def get_available_audio_models(
    *,
    env: ProviderEnvironment | None = None,
    modality: str | None = None,
) -> list[str]:
    """Auto-detect available audio models (STT and TTS) based on configured API keys.

    Checks which audio providers are configured and returns their supported models.

    Args:
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton (current behavior).
        modality: Filter by ``"stt"`` or ``"tts"``. Returns both when ``None``.

    Returns:
        A sorted list of unique model strings (e.g. ``"openai/whisper-1"``).
    """
    from ..drivers.openai_stt_driver import OpenAISTTDriver
    from ..drivers.openai_tts_driver import OpenAITTSDriver

    available: set[str] = set()

    # OpenAI audio models (requires same openai_api_key as LLM)
    if _cfg_value(env, "openai_api_key", "OPENAI_API_KEY"):
        if modality is None or modality == "stt":
            for model_id in OpenAISTTDriver.AUDIO_PRICING:
                available.add(f"openai/{model_id}")
        if modality is None or modality == "tts":
            for model_id in OpenAITTSDriver.AUDIO_PRICING:
                available.add(f"openai/{model_id}")

    # ElevenLabs audio models
    elevenlabs_key = _cfg_value(env, "elevenlabs_api_key", "ELEVENLABS_API_KEY")
    elevenlabs_endpoint = getattr(settings, "elevenlabs_endpoint", None) or "https://api.elevenlabs.io/v1"
    if elevenlabs_key:
        from ..drivers.elevenlabs_stt_driver import ElevenLabsSTTDriver
        from ..drivers.elevenlabs_tts_driver import ElevenLabsTTSDriver

        if modality is None or modality == "stt":
            # STT: static list (no API listing endpoint for STT models)
            stt_models = ElevenLabsSTTDriver.list_models(api_key=elevenlabs_key, endpoint=elevenlabs_endpoint)
            if stt_models:
                for model_id in stt_models:
                    available.add(f"elevenlabs/{model_id}")

        if modality is None or modality == "tts":
            # TTS: dynamic discovery via GET /v1/models, fallback to AUDIO_PRICING
            tts_models = ElevenLabsTTSDriver.list_models(api_key=elevenlabs_key, endpoint=elevenlabs_endpoint)
            if tts_models is not None:
                for model_id in tts_models:
                    available.add(f"elevenlabs/{model_id}")
            else:
                for model_id in ElevenLabsTTSDriver.AUDIO_PRICING:
                    available.add(f"elevenlabs/{model_id}")

    # Dynamic discovery: check modalities_output from models.dev capabilities
    # for any models that the pricing dicts don't know about yet.
    from .model_rates import get_model_capabilities

    for model_str in get_available_models(env=env):
        parts = model_str.split("/", 1)
        if len(parts) != 2:
            continue
        provider, model_id = parts
        caps = get_model_capabilities(provider, model_id)
        if caps and "audio" in caps.modalities_output and (modality is None or modality == "tts"):
            available.add(model_str)

    return sorted(available)


def get_available_image_gen_models(
    *,
    env: ProviderEnvironment | None = None,
) -> list[str]:
    """Auto-detect available image generation models based on configured API keys.

    Checks which image gen providers are configured and returns their supported models.

    Args:
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton (current behavior).

    Returns:
        A sorted list of unique model strings (e.g. ``"openai/dall-e-3"``).
    """
    from ..drivers.google_img_gen_driver import GoogleImageGenDriver
    from ..drivers.grok_img_gen_driver import GrokImageGenDriver
    from ..drivers.openai_img_gen_driver import OpenAIImageGenDriver
    from ..drivers.stability_img_gen_driver import StabilityImageGenDriver

    available: set[str] = set()

    # OpenAI image gen models (requires openai_api_key)
    if _cfg_value(env, "openai_api_key", "OPENAI_API_KEY"):
        for model_id in OpenAIImageGenDriver.IMAGE_PRICING:
            available.add(f"openai/{model_id}")

    # Google image gen models (requires google_api_key)
    if _cfg_value(env, "google_api_key", "GOOGLE_API_KEY"):
        for model_id in GoogleImageGenDriver.IMAGE_PRICING:
            available.add(f"google/{model_id}")

    # Stability AI image gen models
    stability_key = _cfg_value(env, "stability_api_key", "STABILITY_API_KEY")
    if stability_key:
        for model_id in StabilityImageGenDriver.IMAGE_PRICING:
            available.add(f"stability/{model_id}")

    # Grok/xAI image gen models (requires grok_api_key)
    if _cfg_value(env, "grok_api_key", "GROK_API_KEY"):
        for model_id in GrokImageGenDriver.IMAGE_PRICING:
            available.add(f"grok/{model_id}")

    # Dynamic discovery: check modalities_output from models.dev capabilities
    # for any models that the pricing dicts don't know about yet.
    from .model_rates import get_model_capabilities

    for model_str in get_available_models(env=env):
        parts = model_str.split("/", 1)
        if len(parts) != 2:
            continue
        provider, model_id = parts
        caps = get_model_capabilities(provider, model_id)
        if caps and "image" in caps.modalities_output:
            available.add(model_str)

    return sorted(available)


def get_available_embedding_models(
    *,
    env: ProviderEnvironment | None = None,
) -> list[str]:
    """Auto-detect available embedding models based on configured API keys.

    Checks which embedding providers are configured and returns their supported models.

    Args:
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton (current behavior).

    Returns:
        A sorted list of unique model strings (e.g. ``"openai/text-embedding-3-small"``).
    """
    from ..drivers.openai_embedding_driver import OpenAIEmbeddingDriver

    available: set[str] = set()

    # OpenAI embedding models (requires openai_api_key)
    if _cfg_value(env, "openai_api_key", "OPENAI_API_KEY"):
        for model_id in OpenAIEmbeddingDriver.EMBEDDING_PRICING:
            available.add(f"openai/{model_id}")

    # Ollama embedding models (always available — local)
    available.add("ollama/nomic-embed-text")
    available.add("ollama/all-minilm")
    available.add("ollama/mxbai-embed-large")
    available.add("ollama/snowflake-arctic-embed")
    available.add("ollama/bge-m3")

    return sorted(available)
