"""Provider descriptor data shapes + lazy registry populated by plugins.

Each :class:`ProviderDescriptor` describes one canonical provider (or an alias
for one) and carries enough metadata to:

* register sync + async driver factories for every modality (LLM, STT, TTS,
  image-gen, video-gen, embedding, rerank, moderation),
* populate ``PROVIDER_DRIVER_MAP`` / ``ASYNC_PROVIDER_DRIVER_MAP``,
* drive the discovery module's ``is_configured`` / ``list_models_kwargs`` logic,
* generate the ``PROVIDER_MAP`` in ``model_rates.py``.

The list of built-in :class:`ProviderDescriptor` instances is no longer hard-coded
in this module. Instead, it is contributed by :class:`prompture.plugins.ProviderPlugin`
instances. Built-in plugins live in :mod:`prompture.plugins.builtins`. Third
parties can register additional plugins via the ``prompture.providers``
entry-point group.

The module-level names ``PROVIDER_DESCRIPTORS`` and ``PROVIDER_DESCRIPTOR_MAP``
remain accessible (via PEP 562 ``__getattr__``) for backwards compatibility.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


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

    video_gen_sync: DriverSpec | None = None
    video_gen_async: DriverSpec | None = None

    embedding_sync: DriverSpec | None = None
    embedding_async: DriverSpec | None = None

    rerank_sync: DriverSpec | None = None
    rerank_async: DriverSpec | None = None

    moderation_sync: DriverSpec | None = None
    moderation_async: DriverSpec | None = None

    lipsync_sync: DriverSpec | None = None
    lipsync_async: DriverSpec | None = None

    music_sync: DriverSpec | None = None
    music_async: DriverSpec | None = None

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


# ── Lazy plugin-backed descriptor registry ────────────────────────────────

_PROVIDER_DESCRIPTORS_CACHE: list[ProviderDescriptor] | None = None
_PROVIDER_DESCRIPTOR_MAP_CACHE: dict[str, ProviderDescriptor] | None = None


def _ensure_loaded() -> None:
    """Populate the descriptor caches from the plugin system if not yet loaded."""
    global _PROVIDER_DESCRIPTORS_CACHE, _PROVIDER_DESCRIPTOR_MAP_CACHE
    if _PROVIDER_DESCRIPTORS_CACHE is not None:
        return
    from ..plugins.discovery import load_plugins

    _PROVIDER_DESCRIPTORS_CACHE = load_plugins()
    _PROVIDER_DESCRIPTOR_MAP_CACHE = {d.name: d for d in _PROVIDER_DESCRIPTORS_CACHE}


def _reset_descriptor_cache() -> None:
    """Reset the descriptor cache. Intended for tests only."""
    global _PROVIDER_DESCRIPTORS_CACHE, _PROVIDER_DESCRIPTOR_MAP_CACHE
    _PROVIDER_DESCRIPTORS_CACHE = None
    _PROVIDER_DESCRIPTOR_MAP_CACHE = None


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for ``PROVIDER_DESCRIPTORS`` / ``PROVIDER_DESCRIPTOR_MAP``."""
    if name == "PROVIDER_DESCRIPTORS":
        _ensure_loaded()
        return _PROVIDER_DESCRIPTORS_CACHE
    if name == "PROVIDER_DESCRIPTOR_MAP":
        _ensure_loaded()
        return _PROVIDER_DESCRIPTOR_MAP_CACHE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── Bulk registration helper ──────────────────────────────────────────────


def register_all_builtin_drivers() -> None:
    """Register factories for every modality of every built-in provider."""
    _ensure_loaded()
    assert _PROVIDER_DESCRIPTORS_CACHE is not None  # nosec B101 — set by _ensure_loaded

    from .registry import (
        register_async_driver,
        register_async_embedding_driver,
        register_async_img_gen_driver,
        register_async_lipsync_driver,
        register_async_moderation_driver,
        register_async_music_driver,
        register_async_rerank_driver,
        register_async_stt_driver,
        register_async_tts_driver,
        register_async_video_gen_driver,
        register_driver,
        register_embedding_driver,
        register_img_gen_driver,
        register_lipsync_driver,
        register_moderation_driver,
        register_music_driver,
        register_rerank_driver,
        register_stt_driver,
        register_tts_driver,
        register_video_gen_driver,
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
        "video_gen_sync": register_video_gen_driver,
        "video_gen_async": register_async_video_gen_driver,
        "embedding_sync": register_embedding_driver,
        "embedding_async": register_async_embedding_driver,
        "rerank_sync": register_rerank_driver,
        "rerank_async": register_async_rerank_driver,
        "moderation_sync": register_moderation_driver,
        "moderation_async": register_async_moderation_driver,
        "lipsync_sync": register_lipsync_driver,
        "lipsync_async": register_async_lipsync_driver,
        "music_sync": register_music_driver,
        "music_async": register_async_music_driver,
    }

    for desc in _PROVIDER_DESCRIPTORS_CACHE:
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
    _ensure_loaded()
    assert _PROVIDER_DESCRIPTORS_CACHE is not None  # nosec B101 — set by _ensure_loaded

    attr = "llm_async" if is_async else "llm_sync"
    result: dict[str, tuple[type, dict[str, str], str]] = {}
    for desc in _PROVIDER_DESCRIPTORS_CACHE:
        spec: DriverSpec | None = getattr(desc, attr, None)
        if spec is None:
            continue
        cls = _resolve_cls(spec.cls_path)
        result[desc.name] = (cls, dict(spec.kwarg_map), spec.default_model)
    return result
