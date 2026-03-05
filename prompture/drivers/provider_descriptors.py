"""Single source of truth for all built-in provider definitions.

Each ``ProviderDescriptor`` describes one canonical provider (or an alias for
one) and carries enough metadata to:

* register sync + async driver factories for every modality (LLM, STT, TTS,
  image-gen, embedding)
* populate ``PROVIDER_DRIVER_MAP`` / ``ASYNC_PROVIDER_DRIVER_MAP``
* drive the discovery module's ``is_configured`` / ``list_models_kwargs`` logic
* generate the ``PROVIDER_MAP`` in ``model_rates.py``

Adding a new provider is a single ``ProviderDescriptor(...)`` entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class DriverSpec:
    """Recipe for one modality driver (sync *or* async).

    Attributes:
        cls_path: Dotted import path relative to ``prompture.drivers``, e.g.
            ``"openai_driver.OpenAIDriver"``.  Resolved lazily to avoid
            circular imports.
        kwarg_map: Maps constructor kwarg → settings attribute name.
        default_model: Either a settings attribute name (e.g. ``"openai_model"``)
            or a literal model string (e.g. ``"gpt-4o-mini"``).
            ``getattr(settings, x, x)`` resolves both.
    """

    cls_path: str
    kwarg_map: dict[str, str]
    default_model: str


@dataclass
class ProviderDescriptor:
    """Full description of one provider (or alias)."""

    name: str

    # If set, this name is an alias for another canonical provider.
    alias_for: str | None = None

    # Modality specs (sync, async) — None means the provider doesn't support that modality.
    llm_sync: DriverSpec | None = None
    llm_async: DriverSpec | None = None

    stt_sync: DriverSpec | None = None
    stt_async: DriverSpec | None = None
    tts_sync: DriverSpec | None = None
    tts_async: DriverSpec | None = None

    img_gen_sync: DriverSpec | None = None
    img_gen_async: DriverSpec | None = None

    embedding_sync: DriverSpec | None = None
    embedding_async: DriverSpec | None = None

    # Human-friendly name for display purposes (e.g. "OpenAI", "Google Gemini").
    # Aliases get None.
    display_name: str | None = None

    # Discovery: how to tell if the provider is configured.
    # Simple case: a settings attribute that must be truthy (e.g. "openai_api_key").
    is_configured_check: str | None = None
    # Complex case (e.g. Azure): a callable returning bool.
    is_configured_fn: Callable[..., bool] | None = None
    # Providers that are always available (local servers).
    always_available: bool = False

    # Discovery: kwargs for list_models().
    # Each entry is (ctor_kwarg, settings_attr, env_var_fallback | None).
    list_models_kwargs: list[tuple[str, str, str | None]] = field(default_factory=list)

    # model_rates.py: maps this provider to a models.dev provider name.
    models_dev_name: str | None = None


# ── Lazy class resolution ──────────────────────────────────────────────────

_cls_cache: dict[str, type[Any]] = {}


def _resolve_cls(cls_path: str) -> type[Any]:
    """Resolve ``"module.ClassName"`` relative to ``prompture.drivers``."""
    if cls_path in _cls_cache:
        return _cls_cache[cls_path]
    module_part, cls_name = cls_path.rsplit(".", 1)
    import importlib

    mod = importlib.import_module(f"prompture.drivers.{module_part}")
    cls: type[Any] = getattr(mod, cls_name)
    _cls_cache[cls_path] = cls
    return cls


# ── Factory builders ───────────────────────────────────────────────────────


def _make_factory(spec: DriverSpec) -> Callable[[str | None], object]:
    """Build a closure that constructs a driver from *spec*, reading settings at call time."""

    def factory(model: str | None = None) -> object:
        from ..infra.settings import settings

        cls = _resolve_cls(spec.cls_path)
        kwargs: dict[str, Any] = {}
        for ctor_kwarg, attr_name in spec.kwarg_map.items():
            kwargs[ctor_kwarg] = getattr(settings, attr_name, None)
        kwargs["model"] = model or getattr(settings, spec.default_model, spec.default_model)
        return cls(**kwargs)

    return factory


# ── Provider descriptor list ──────────────────────────────────────────────


def _llm(
    sync_mod: str, sync_cls: str, async_mod: str, async_cls: str, kwarg_map: dict[str, str], default_model: str
) -> dict[str, Any]:
    """Helper to build llm_sync + llm_async specs."""
    return {
        "llm_sync": DriverSpec(f"{sync_mod}.{sync_cls}", kwarg_map, default_model),
        "llm_async": DriverSpec(f"{async_mod}.{async_cls}", kwarg_map, default_model),
    }


def _build_descriptors() -> list[ProviderDescriptor]:
    """Define all built-in providers and aliases."""
    descriptors: list[ProviderDescriptor] = []

    # ── OpenAI ─────────────────────────────────────────────────────────
    _oai_kw = {"api_key": "openai_api_key"}
    descriptors.append(
        ProviderDescriptor(
            name="openai",
            **_llm(
                "openai_driver", "OpenAIDriver", "async_openai_driver", "AsyncOpenAIDriver", _oai_kw, "openai_model"
            ),
            stt_sync=DriverSpec("openai_stt_driver.OpenAISTTDriver", _oai_kw, "whisper-1"),
            stt_async=DriverSpec("async_openai_stt_driver.AsyncOpenAISTTDriver", _oai_kw, "whisper-1"),
            tts_sync=DriverSpec("openai_tts_driver.OpenAITTSDriver", _oai_kw, "tts-1"),
            tts_async=DriverSpec("async_openai_tts_driver.AsyncOpenAITTSDriver", _oai_kw, "tts-1"),
            img_gen_sync=DriverSpec("openai_img_gen_driver.OpenAIImageGenDriver", _oai_kw, "dall-e-3"),
            img_gen_async=DriverSpec("async_openai_img_gen_driver.AsyncOpenAIImageGenDriver", _oai_kw, "dall-e-3"),
            embedding_sync=DriverSpec(
                "openai_embedding_driver.OpenAIEmbeddingDriver", _oai_kw, "text-embedding-3-small"
            ),
            embedding_async=DriverSpec(
                "async_openai_embedding_driver.AsyncOpenAIEmbeddingDriver", _oai_kw, "text-embedding-3-small"
            ),
            display_name="OpenAI",
            is_configured_check="openai_api_key",
            list_models_kwargs=[("api_key", "openai_api_key", "OPENAI_API_KEY")],
            models_dev_name="openai",
        )
    )

    # ── Claude ─────────────────────────────────────────────────────────
    _claude_kw = {"api_key": "claude_api_key"}
    descriptors.append(
        ProviderDescriptor(
            name="claude",
            **_llm(
                "claude_driver", "ClaudeDriver", "async_claude_driver", "AsyncClaudeDriver", _claude_kw, "claude_model"
            ),
            display_name="Anthropic",
            is_configured_check="claude_api_key",
            list_models_kwargs=[("api_key", "claude_api_key", "CLAUDE_API_KEY")],
            models_dev_name="anthropic",
        )
    )

    # ── Google ─────────────────────────────────────────────────────────
    _google_kw = {"api_key": "google_api_key"}
    descriptors.append(
        ProviderDescriptor(
            name="google",
            **_llm(
                "google_driver", "GoogleDriver", "async_google_driver", "AsyncGoogleDriver", _google_kw, "google_model"
            ),
            img_gen_sync=DriverSpec(
                "google_img_gen_driver.GoogleImageGenDriver", _google_kw, "imagen-3.0-generate-002"
            ),
            img_gen_async=DriverSpec(
                "async_google_img_gen_driver.AsyncGoogleImageGenDriver", _google_kw, "imagen-3.0-generate-002"
            ),
            display_name="Google Gemini",
            is_configured_check="google_api_key",
            list_models_kwargs=[("api_key", "google_api_key", "GOOGLE_API_KEY")],
            models_dev_name="google",
        )
    )

    # ── Groq ───────────────────────────────────────────────────────────
    _groq_kw = {"api_key": "groq_api_key"}
    descriptors.append(
        ProviderDescriptor(
            name="groq",
            **_llm("groq_driver", "GroqDriver", "async_groq_driver", "AsyncGroqDriver", _groq_kw, "groq_model"),
            display_name="Groq",
            is_configured_check="groq_api_key",
            list_models_kwargs=[("api_key", "groq_api_key", "GROQ_API_KEY")],
            models_dev_name="groq",
        )
    )

    # ── Grok / xAI ────────────────────────────────────────────────────
    _grok_kw = {"api_key": "grok_api_key"}
    descriptors.append(
        ProviderDescriptor(
            name="grok",
            **_llm("grok_driver", "GrokDriver", "async_grok_driver", "AsyncGrokDriver", _grok_kw, "grok_model"),
            img_gen_sync=DriverSpec("grok_img_gen_driver.GrokImageGenDriver", _grok_kw, "grok-2-image"),
            img_gen_async=DriverSpec("async_grok_img_gen_driver.AsyncGrokImageGenDriver", _grok_kw, "grok-2-image"),
            display_name="xAI Grok",
            is_configured_check="grok_api_key",
            list_models_kwargs=[("api_key", "grok_api_key", "GROK_API_KEY")],
            models_dev_name="xai",
        )
    )

    # ── OpenRouter ────────────────────────────────────────────────────
    _or_kw = {"api_key": "openrouter_api_key"}
    descriptors.append(
        ProviderDescriptor(
            name="openrouter",
            **_llm(
                "openrouter_driver",
                "OpenRouterDriver",
                "async_openrouter_driver",
                "AsyncOpenRouterDriver",
                _or_kw,
                "openrouter_model",
            ),
            display_name="OpenRouter",
            is_configured_check="openrouter_api_key",
            list_models_kwargs=[("api_key", "openrouter_api_key", "OPENROUTER_API_KEY")],
            models_dev_name="openrouter",
        )
    )

    # ── Moonshot ──────────────────────────────────────────────────────
    _moon_kw = {"api_key": "moonshot_api_key", "endpoint": "moonshot_endpoint"}
    descriptors.append(
        ProviderDescriptor(
            name="moonshot",
            **_llm(
                "moonshot_driver",
                "MoonshotDriver",
                "async_moonshot_driver",
                "AsyncMoonshotDriver",
                _moon_kw,
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

    # ── ModelScope ────────────────────────────────────────────────────
    _ms_kw = {"api_key": "modelscope_api_key", "endpoint": "modelscope_endpoint"}
    descriptors.append(
        ProviderDescriptor(
            name="modelscope",
            **_llm(
                "modelscope_driver",
                "ModelScopeDriver",
                "async_modelscope_driver",
                "AsyncModelScopeDriver",
                _ms_kw,
                "modelscope_model",
            ),
            display_name="ModelScope",
            is_configured_check="modelscope_api_key",
        )
    )

    # ── Z.ai (Zhipu) ─────────────────────────────────────────────────
    _zai_kw = {"api_key": "zhipu_api_key", "endpoint": "zhipu_endpoint"}
    descriptors.append(
        ProviderDescriptor(
            name="zai",
            **_llm("zai_driver", "ZaiDriver", "async_zai_driver", "AsyncZaiDriver", _zai_kw, "zhipu_model"),
            display_name="Z.ai (Zhipu)",
            is_configured_check="zhipu_api_key",
            models_dev_name="zai",
        )
    )

    # ── Ollama ────────────────────────────────────────────────────────
    _ollama_kw = {"endpoint": "ollama_endpoint"}
    descriptors.append(
        ProviderDescriptor(
            name="ollama",
            **_llm(
                "ollama_driver", "OllamaDriver", "async_ollama_driver", "AsyncOllamaDriver", _ollama_kw, "ollama_model"
            ),
            embedding_sync=DriverSpec("ollama_embedding_driver.OllamaEmbeddingDriver", _ollama_kw, "nomic-embed-text"),
            embedding_async=DriverSpec(
                "async_ollama_embedding_driver.AsyncOllamaEmbeddingDriver", _ollama_kw, "nomic-embed-text"
            ),
            display_name="Ollama",
            always_available=True,
            list_models_kwargs=[("endpoint", "ollama_endpoint", "OLLAMA_ENDPOINT")],
        )
    )

    # ── LM Studio ────────────────────────────────────────────────────
    _lms_kw = {"endpoint": "lmstudio_endpoint", "api_key": "lmstudio_api_key"}
    descriptors.append(
        ProviderDescriptor(
            name="lmstudio",
            **_llm(
                "lmstudio_driver",
                "LMStudioDriver",
                "async_lmstudio_driver",
                "AsyncLMStudioDriver",
                _lms_kw,
                "lmstudio_model",
            ),
            display_name="LM Studio",
            always_available=True,
            list_models_kwargs=[
                ("endpoint", "lmstudio_endpoint", "LMSTUDIO_ENDPOINT"),
                ("api_key", "lmstudio_api_key", "LMSTUDIO_API_KEY"),
            ],
        )
    )

    # ── Local HTTP ───────────────────────────────────────────────────
    _local_kw = {"endpoint": "local_http_endpoint"}
    descriptors.append(
        ProviderDescriptor(
            name="local_http",
            llm_sync=DriverSpec("local_http_driver.LocalHTTPDriver", _local_kw, "local_http_model"),
            llm_async=DriverSpec("async_local_http_driver.AsyncLocalHTTPDriver", _local_kw, "local_http_model"),
            display_name="Local HTTP",
            always_available=True,
        )
    )

    # ── Azure ────────────────────────────────────────────────────────
    _azure_kw = {
        "api_key": "azure_api_key",
        "endpoint": "azure_api_endpoint",
        "deployment_id": "azure_deployment_id",
        "claude_api_key": "azure_claude_api_key",
        "claude_endpoint": "azure_claude_endpoint",
        "mistral_api_key": "azure_mistral_api_key",
        "mistral_endpoint": "azure_mistral_endpoint",
    }
    descriptors.append(
        ProviderDescriptor(
            name="azure",
            **_llm("azure_driver", "AzureDriver", "async_azure_driver", "AsyncAzureDriver", _azure_kw, "gpt-4o-mini"),
            display_name="Azure",
            is_configured_fn=_azure_is_configured,
            models_dev_name="azure",
        )
    )

    # ── CachiBot ─────────────────────────────────────────────────────
    _cb_kw = {"api_key": "cachibot_api_key", "endpoint": "cachibot_endpoint"}
    descriptors.append(
        ProviderDescriptor(
            name="cachibot",
            **_llm(
                "cachibot_driver",
                "CachiBotDriver",
                "async_cachibot_driver",
                "AsyncCachiBotDriver",
                _cb_kw,
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

    # ── AirLLM ───────────────────────────────────────────────────────
    _air_kw = {"compression": "airllm_compression"}
    descriptors.append(
        ProviderDescriptor(
            name="airllm",
            llm_sync=DriverSpec("airllm_driver.AirLLMDriver", _air_kw, "airllm_model"),
            llm_async=DriverSpec("async_airllm_driver.AsyncAirLLMDriver", _air_kw, "airllm_model"),
            display_name="AirLLM",
            always_available=True,
        )
    )

    # ── HuggingFace ──────────────────────────────────────────────────
    _hf_kw = {"endpoint": "hf_endpoint", "token": "hf_token"}  # nosec B105
    descriptors.append(
        ProviderDescriptor(
            name="huggingface",
            **_llm(
                "hugging_driver",
                "HuggingFaceDriver",
                "async_hugging_driver",
                "AsyncHuggingFaceDriver",
                _hf_kw,
                "bert-base-uncased",
            ),
            display_name="Hugging Face",
        )
    )

    # ── ElevenLabs (audio only) ──────────────────────────────────────
    _el_kw = {"api_key": "elevenlabs_api_key", "endpoint": "elevenlabs_endpoint"}
    descriptors.append(
        ProviderDescriptor(
            name="elevenlabs",
            stt_sync=DriverSpec("elevenlabs_stt_driver.ElevenLabsSTTDriver", _el_kw, "scribe_v1"),
            stt_async=DriverSpec("async_elevenlabs_stt_driver.AsyncElevenLabsSTTDriver", _el_kw, "scribe_v1"),
            tts_sync=DriverSpec("elevenlabs_tts_driver.ElevenLabsTTSDriver", {**_el_kw}, "elevenlabs_tts_model"),
            tts_async=DriverSpec(
                "async_elevenlabs_tts_driver.AsyncElevenLabsTTSDriver", {**_el_kw}, "elevenlabs_tts_model"
            ),
            display_name="ElevenLabs",
            is_configured_check="elevenlabs_api_key",
            models_dev_name="elevenlabs",
        )
    )

    # ── Stability AI (image gen only) ────────────────────────────────
    _stab_kw = {"api_key": "stability_api_key", "endpoint": "stability_endpoint"}
    descriptors.append(
        ProviderDescriptor(
            name="stability",
            img_gen_sync=DriverSpec("stability_img_gen_driver.StabilityImageGenDriver", _stab_kw, "stable-image-core"),
            img_gen_async=DriverSpec(
                "async_stability_img_gen_driver.AsyncStabilityImageGenDriver", _stab_kw, "stable-image-core"
            ),
            display_name="Stability AI",
            is_configured_check="stability_api_key",
        )
    )

    # ── Aliases ───────────────────────────────────────────────────────
    # Each alias specifies exactly which modalities it covers, matching
    # the original per-file registrations.  Format:
    #   (alias_name, canonical_name, set_of_modality_prefixes_to_inherit)
    # "llm" → llm_sync + llm_async, "img_gen" → img_gen_sync + img_gen_async, etc.
    _aliases: list[tuple[str, str, set[str]]] = [
        ("anthropic", "claude", {"llm"}),
        ("gemini", "google", {"llm", "img_gen"}),
        ("chatgpt", "openai", {"llm", "embedding"}),
        ("xai", "grok", {"llm", "img_gen"}),
        ("lm_studio", "lmstudio", {"llm"}),
        ("lm-studio", "lmstudio", {"llm"}),
        ("zhipu", "zai", {"llm"}),
        ("hf", "huggingface", {"llm"}),
        ("dalle", "openai", {"img_gen"}),
    ]

    # Build a lookup of canonical descriptors by name.
    canonical_map = {d.name: d for d in descriptors}

    for alias_name, canonical_name, modalities in _aliases:
        canon = canonical_map[canonical_name]
        alias_desc = ProviderDescriptor(
            name=alias_name,
            alias_for=canonical_name,
            llm_sync=canon.llm_sync if "llm" in modalities else None,
            llm_async=canon.llm_async if "llm" in modalities else None,
            stt_sync=canon.stt_sync if "stt" in modalities else None,
            stt_async=canon.stt_async if "stt" in modalities else None,
            tts_sync=canon.tts_sync if "tts" in modalities else None,
            tts_async=canon.tts_async if "tts" in modalities else None,
            img_gen_sync=canon.img_gen_sync if "img_gen" in modalities else None,
            img_gen_async=canon.img_gen_async if "img_gen" in modalities else None,
            embedding_sync=canon.embedding_sync if "embedding" in modalities else None,
            embedding_async=canon.embedding_async if "embedding" in modalities else None,
            is_configured_check=canon.is_configured_check,
            is_configured_fn=canon.is_configured_fn,
            always_available=canon.always_available,
            list_models_kwargs=canon.list_models_kwargs,
            models_dev_name=None,  # aliases don't map to models.dev independently
        )
        descriptors.append(alias_desc)

    return descriptors


def _azure_is_configured(env: Any = None) -> bool:
    """Azure has a complex multi-backend configuration check."""
    import os

    from ..infra.settings import settings

    if env is not None:
        if env.resolve("azure_api_key") and env.resolve("azure_api_endpoint"):
            return True
    else:
        if (getattr(settings, "azure_api_key", None) or os.getenv("AZURE_API_KEY")) and (
            getattr(settings, "azure_api_endpoint", None) or os.getenv("AZURE_API_ENDPOINT")
        ):
            return True

    if settings.azure_claude_api_key or os.getenv("AZURE_CLAUDE_API_KEY"):
        return True
    if settings.azure_mistral_api_key or os.getenv("AZURE_MISTRAL_API_KEY"):
        return True

    try:
        from .azure_config import has_azure_config_resolver, has_registered_configs

        if has_registered_configs() or has_azure_config_resolver():
            return True
    except ImportError:
        pass

    return False


# ── Module-level singletons ───────────────────────────────────────────────

PROVIDER_DESCRIPTORS: list[ProviderDescriptor] = _build_descriptors()
PROVIDER_DESCRIPTOR_MAP: dict[str, ProviderDescriptor] = {d.name: d for d in PROVIDER_DESCRIPTORS}


# ── Bulk registration helper ──────────────────────────────────────────────


def register_all_builtin_drivers() -> None:
    """Register factories for every modality of every built-in provider."""
    from .registry import (
        register_async_driver,
        register_async_embedding_driver,
        register_async_img_gen_driver,
        register_async_stt_driver,
        register_async_tts_driver,
        register_driver,
        register_embedding_driver,
        register_img_gen_driver,
        register_stt_driver,
        register_tts_driver,
    )

    _MODALITY_REGISTRARS = {
        "llm_sync": register_driver,
        "llm_async": register_async_driver,
        "stt_sync": register_stt_driver,
        "stt_async": register_async_stt_driver,
        "tts_sync": register_tts_driver,
        "tts_async": register_async_tts_driver,
        "img_gen_sync": register_img_gen_driver,
        "img_gen_async": register_async_img_gen_driver,
        "embedding_sync": register_embedding_driver,
        "embedding_async": register_async_embedding_driver,
    }

    for desc in PROVIDER_DESCRIPTORS:
        for attr, registrar in _MODALITY_REGISTRARS.items():
            spec: DriverSpec | None = getattr(desc, attr, None)
            if spec is not None:
                registrar(desc.name, _make_factory(spec), overwrite=True)


def build_provider_driver_map(*, is_async: bool = False) -> dict[str, tuple[type, dict[str, str], str]]:
    """Derive the ``PROVIDER_DRIVER_MAP`` (or async variant) from descriptors.

    Returns a dict mapping provider name → ``(DriverClass, kwarg_map, default_model)``.
    Only includes providers that have an LLM spec *and* whose LLM spec has a
    non-empty ``kwarg_map`` (i.e. providers that support per-env construction).
    """
    attr = "llm_async" if is_async else "llm_sync"
    result: dict[str, tuple[type, dict[str, str], str]] = {}
    for desc in PROVIDER_DESCRIPTORS:
        spec: DriverSpec | None = getattr(desc, attr, None)
        if spec is None:
            continue
        cls = _resolve_cls(spec.cls_path)
        result[desc.name] = (cls, dict(spec.kwarg_map), spec.default_model)
    return result
