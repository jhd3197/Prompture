"""Driver registry with plugin support.

This module provides a public API for registering custom drivers and
supports auto-discovery of drivers via Python entry points.

Example usage:
    # Register a custom driver
    from prompture import register_driver

    def my_driver_factory(model=None):
        return MyCustomDriver(model=model)

    register_driver("my_provider", my_driver_factory)

    # Now you can use it
    driver = get_driver_for_model("my_provider/my-model")

For entry point discovery, add to your package's pyproject.toml:
    [project.entry-points."prompture.drivers"]
    my_provider = "my_package.drivers:my_driver_factory"

    [project.entry-points."prompture.async_drivers"]
    my_provider = "my_package.drivers:my_async_driver_factory"
"""

from __future__ import annotations

from .driver_registry import DriverFactory, DriverRegistry

# Re-export the type alias so existing ``from .registry import DriverFactory`` keeps working.
__all__ = ["DriverFactory"]

# ── Registry instances ─────────────────────────────────────────────────────

_llm_sync = DriverRegistry("LLM sync", "prompture.drivers", error_prefix="")
_llm_async = DriverRegistry("LLM async", "prompture.async_drivers", error_prefix="")

_stt_sync = DriverRegistry("STT sync", "prompture.stt_drivers", error_prefix="STT ")
_stt_async = DriverRegistry("STT async", "prompture.async_stt_drivers", error_prefix="async STT ")
_tts_sync = DriverRegistry("TTS sync", "prompture.tts_drivers", error_prefix="TTS ")
_tts_async = DriverRegistry("TTS async", "prompture.async_tts_drivers", error_prefix="async TTS ")

_img_gen_sync = DriverRegistry("image gen sync", "prompture.img_gen_drivers", error_prefix="image gen ")
_img_gen_async = DriverRegistry("image gen async", "prompture.async_img_gen_drivers", error_prefix="async image gen ")

_embedding_sync = DriverRegistry("embedding sync", "prompture.embedding_drivers", error_prefix="embedding ")
_embedding_async = DriverRegistry(
    "embedding async", "prompture.async_embedding_drivers", error_prefix="async embedding "
)

# ── LLM sync ───────────────────────────────────────────────────────────────


def register_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register a custom driver factory for a provider name.

    Args:
        name: Provider name (e.g., "my_provider"). Will be lowercased.
        factory: A callable that takes an optional model name and returns
                 a driver instance. The driver must implement the
                 ``Driver`` interface (specifically ``generate()``).
        overwrite: If True, allow overwriting an existing registration.
                   Defaults to False.

    Raises:
        ValueError: If a driver with this name is already registered
                    and overwrite=False.

    Example:
        >>> def my_factory(model=None):
        ...     return MyDriver(model=model or "default-model")
        >>> register_driver("my_provider", my_factory)
        >>> driver = get_driver_for_model("my_provider/custom-model")
    """
    _llm_sync.register(name, factory, overwrite=overwrite)


def register_async_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register a custom async driver factory for a provider name.

    Args:
        name: Provider name (e.g., "my_provider"). Will be lowercased.
        factory: A callable that takes an optional model name and returns
                 an async driver instance. The driver must implement the
                 ``AsyncDriver`` interface (specifically ``async generate()``).
        overwrite: If True, allow overwriting an existing registration.
                   Defaults to False.

    Raises:
        ValueError: If an async driver with this name is already registered
                    and overwrite=False.

    Example:
        >>> def my_async_factory(model=None):
        ...     return MyAsyncDriver(model=model or "default-model")
        >>> register_async_driver("my_provider", my_async_factory)
        >>> driver = get_async_driver_for_model("my_provider/custom-model")
    """
    _llm_async.register(name, factory, overwrite=overwrite)


def unregister_driver(name: str) -> bool:
    """Unregister a sync driver by name.

    Args:
        name: Provider name to unregister.

    Returns:
        True if the driver was unregistered, False if it wasn't registered.
    """
    return _llm_sync.unregister(name)


def unregister_async_driver(name: str) -> bool:
    """Unregister an async driver by name.

    Args:
        name: Provider name to unregister.

    Returns:
        True if the driver was unregistered, False if it wasn't registered.
    """
    return _llm_async.unregister(name)


def list_registered_drivers() -> list[str]:
    """Return a sorted list of registered sync driver names."""
    return _llm_sync.list_names()


def list_registered_async_drivers() -> list[str]:
    """Return a sorted list of registered async driver names."""
    return _llm_async.list_names()


def is_driver_registered(name: str) -> bool:
    """Check if a sync driver is registered."""
    return _llm_sync.is_registered(name)


def is_async_driver_registered(name: str) -> bool:
    """Check if an async driver is registered."""
    return _llm_async.is_registered(name)


def get_driver_factory(name: str) -> DriverFactory:
    """Get a registered sync driver factory by name.

    Args:
        name: Provider name.

    Returns:
        The factory function.

    Raises:
        ValueError: If the driver is not registered.
    """
    return _llm_sync.get_factory(name)


def get_async_driver_factory(name: str) -> DriverFactory:
    """Get a registered async driver factory by name.

    Args:
        name: Provider name.

    Returns:
        The factory function.

    Raises:
        ValueError: If the async driver is not registered.
    """
    return _llm_async.get_factory(name)


def load_entry_point_drivers() -> tuple[int, int]:
    """Load drivers from installed packages via entry points.

    Returns:
        A tuple of (sync_count, async_count) indicating how many drivers
        were loaded from entry points.
    """
    return (_llm_sync.load_entry_points(), _llm_async.load_entry_points())


# ── STT ────────────────────────────────────────────────────────────────────


def register_stt_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register a sync STT driver factory for a provider name."""
    _stt_sync.register(name, factory, overwrite=overwrite)


def register_async_stt_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register an async STT driver factory for a provider name."""
    _stt_async.register(name, factory, overwrite=overwrite)


def unregister_stt_driver(name: str) -> bool:
    """Unregister a sync STT driver by name."""
    return _stt_sync.unregister(name)


def unregister_async_stt_driver(name: str) -> bool:
    """Unregister an async STT driver by name."""
    return _stt_async.unregister(name)


def list_registered_stt_drivers() -> list[str]:
    """Return a sorted list of registered sync STT driver names."""
    return _stt_sync.list_names()


def list_registered_async_stt_drivers() -> list[str]:
    """Return a sorted list of registered async STT driver names."""
    return _stt_async.list_names()


def is_stt_driver_registered(name: str) -> bool:
    """Check if a sync STT driver is registered."""
    return _stt_sync.is_registered(name)


def is_async_stt_driver_registered(name: str) -> bool:
    """Check if an async STT driver is registered."""
    return _stt_async.is_registered(name)


def get_stt_driver_factory(name: str) -> DriverFactory:
    """Get a registered sync STT driver factory by name."""
    return _stt_sync.get_factory(name)


def get_async_stt_driver_factory(name: str) -> DriverFactory:
    """Get a registered async STT driver factory by name."""
    return _stt_async.get_factory(name)


# ── TTS ────────────────────────────────────────────────────────────────────


def register_tts_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register a sync TTS driver factory for a provider name."""
    _tts_sync.register(name, factory, overwrite=overwrite)


def register_async_tts_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register an async TTS driver factory for a provider name."""
    _tts_async.register(name, factory, overwrite=overwrite)


def unregister_tts_driver(name: str) -> bool:
    """Unregister a sync TTS driver by name."""
    return _tts_sync.unregister(name)


def unregister_async_tts_driver(name: str) -> bool:
    """Unregister an async TTS driver by name."""
    return _tts_async.unregister(name)


def list_registered_tts_drivers() -> list[str]:
    """Return a sorted list of registered sync TTS driver names."""
    return _tts_sync.list_names()


def list_registered_async_tts_drivers() -> list[str]:
    """Return a sorted list of registered async TTS driver names."""
    return _tts_async.list_names()


def is_tts_driver_registered(name: str) -> bool:
    """Check if a sync TTS driver is registered."""
    return _tts_sync.is_registered(name)


def is_async_tts_driver_registered(name: str) -> bool:
    """Check if an async TTS driver is registered."""
    return _tts_async.is_registered(name)


def get_tts_driver_factory(name: str) -> DriverFactory:
    """Get a registered sync TTS driver factory by name."""
    return _tts_sync.get_factory(name)


def get_async_tts_driver_factory(name: str) -> DriverFactory:
    """Get a registered async TTS driver factory by name."""
    return _tts_async.get_factory(name)


# ── Audio entry points ─────────────────────────────────────────────────────


def load_audio_entry_point_drivers() -> tuple[int, int, int, int]:
    """Load audio drivers from installed packages via entry points.

    Returns:
        A tuple of (stt_sync, stt_async, tts_sync, tts_async) counts.
    """
    return (
        _stt_sync.load_entry_points(),
        _stt_async.load_entry_points(),
        _tts_sync.load_entry_points(),
        _tts_async.load_entry_points(),
    )


# ── Image Gen ──────────────────────────────────────────────────────────────


def register_img_gen_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register a sync image generation driver factory for a provider name."""
    _img_gen_sync.register(name, factory, overwrite=overwrite)


def register_async_img_gen_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register an async image generation driver factory for a provider name."""
    _img_gen_async.register(name, factory, overwrite=overwrite)


def unregister_img_gen_driver(name: str) -> bool:
    """Unregister a sync image gen driver by name."""
    return _img_gen_sync.unregister(name)


def unregister_async_img_gen_driver(name: str) -> bool:
    """Unregister an async image gen driver by name."""
    return _img_gen_async.unregister(name)


def list_registered_img_gen_drivers() -> list[str]:
    """Return a sorted list of registered sync image gen driver names."""
    return _img_gen_sync.list_names()


def list_registered_async_img_gen_drivers() -> list[str]:
    """Return a sorted list of registered async image gen driver names."""
    return _img_gen_async.list_names()


def is_img_gen_driver_registered(name: str) -> bool:
    """Check if a sync image gen driver is registered."""
    return _img_gen_sync.is_registered(name)


def is_async_img_gen_driver_registered(name: str) -> bool:
    """Check if an async image gen driver is registered."""
    return _img_gen_async.is_registered(name)


def get_img_gen_driver_factory(name: str) -> DriverFactory:
    """Get a registered sync image gen driver factory by name."""
    return _img_gen_sync.get_factory(name)


def get_async_img_gen_driver_factory(name: str) -> DriverFactory:
    """Get a registered async image gen driver factory by name."""
    return _img_gen_async.get_factory(name)


def load_img_gen_entry_point_drivers() -> tuple[int, int]:
    """Load image gen drivers from installed packages via entry points.

    Returns:
        A tuple of (sync_count, async_count) counts.
    """
    return (_img_gen_sync.load_entry_points(), _img_gen_async.load_entry_points())


# ── Embedding ──────────────────────────────────────────────────────────────


def register_embedding_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register a sync embedding driver factory for a provider name."""
    _embedding_sync.register(name, factory, overwrite=overwrite)


def register_async_embedding_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register an async embedding driver factory for a provider name."""
    _embedding_async.register(name, factory, overwrite=overwrite)


def unregister_embedding_driver(name: str) -> bool:
    """Unregister a sync embedding driver by name."""
    return _embedding_sync.unregister(name)


def unregister_async_embedding_driver(name: str) -> bool:
    """Unregister an async embedding driver by name."""
    return _embedding_async.unregister(name)


def list_registered_embedding_drivers() -> list[str]:
    """Return a sorted list of registered sync embedding driver names."""
    return _embedding_sync.list_names()


def list_registered_async_embedding_drivers() -> list[str]:
    """Return a sorted list of registered async embedding driver names."""
    return _embedding_async.list_names()


def is_embedding_driver_registered(name: str) -> bool:
    """Check if a sync embedding driver is registered."""
    return _embedding_sync.is_registered(name)


def is_async_embedding_driver_registered(name: str) -> bool:
    """Check if an async embedding driver is registered."""
    return _embedding_async.is_registered(name)


def get_embedding_driver_factory(name: str) -> DriverFactory:
    """Get a registered sync embedding driver factory by name."""
    return _embedding_sync.get_factory(name)


def get_async_embedding_driver_factory(name: str) -> DriverFactory:
    """Get a registered async embedding driver factory by name."""
    return _embedding_async.get_factory(name)


def load_embedding_entry_point_drivers() -> tuple[int, int]:
    """Load embedding drivers from installed packages via entry points.

    Returns:
        A tuple of (sync_count, async_count) counts.
    """
    return (_embedding_sync.load_entry_points(), _embedding_async.load_entry_points())


# ── Internal helpers (used by __init__.py and async_registry.py) ───────────


def _get_sync_registry() -> dict[str, DriverFactory]:
    """Get the internal sync registry dict (for internal use by drivers/__init__.py)."""
    return _llm_sync.dict


def _get_async_registry() -> dict[str, DriverFactory]:
    """Get the internal async registry dict (for internal use by drivers/async_registry.py)."""
    return _llm_async.dict


def _get_stt_registry() -> dict[str, DriverFactory]:
    return _stt_sync.dict


def _get_async_stt_registry() -> dict[str, DriverFactory]:
    return _stt_async.dict


def _get_tts_registry() -> dict[str, DriverFactory]:
    return _tts_sync.dict


def _get_async_tts_registry() -> dict[str, DriverFactory]:
    return _tts_async.dict


def _get_img_gen_registry() -> dict[str, DriverFactory]:
    return _img_gen_sync.dict


def _get_async_img_gen_registry() -> dict[str, DriverFactory]:
    return _img_gen_async.dict


def _get_embedding_registry() -> dict[str, DriverFactory]:
    return _embedding_sync.dict


def _get_async_embedding_registry() -> dict[str, DriverFactory]:
    return _embedding_async.dict


def _reset_registries() -> None:
    """Reset registries to empty state (for testing only)."""
    _llm_sync.reset()
    _llm_async.reset()
    _stt_sync.reset()
    _stt_async.reset()
    _tts_sync.reset()
    _tts_async.reset()
    _img_gen_sync.reset()
    _img_gen_async.reset()
    _embedding_sync.reset()
    _embedding_async.reset()


# ── Backwards-compat aliases for internal dicts (used by tests) ────────────
# These are live references to the underlying dicts so that test fixtures
# which do ``_IMG_GEN_REGISTRY.clear()`` / ``.update()`` still work.

_SYNC_REGISTRY = _llm_sync._registry
_ASYNC_REGISTRY = _llm_async._registry
_STT_REGISTRY = _stt_sync._registry
_ASYNC_STT_REGISTRY = _stt_async._registry
_TTS_REGISTRY = _tts_sync._registry
_ASYNC_TTS_REGISTRY = _tts_async._registry
_IMG_GEN_REGISTRY = _img_gen_sync._registry
_ASYNC_IMG_GEN_REGISTRY = _img_gen_async._registry
_EMBEDDING_REGISTRY = _embedding_sync._registry
_ASYNC_EMBEDDING_REGISTRY = _embedding_async._registry
