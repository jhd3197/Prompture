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

import logging
import sys
from typing import Callable

logger = logging.getLogger("prompture.drivers.registry")

# Type alias for driver factory functions
# A factory takes an optional model name and returns a driver instance
DriverFactory = Callable[[str | None], object]

# Internal registries - populated by built-in drivers and plugins
_SYNC_REGISTRY: dict[str, DriverFactory] = {}
_ASYNC_REGISTRY: dict[str, DriverFactory] = {}

# Audio driver registries (STT and TTS)
_STT_REGISTRY: dict[str, DriverFactory] = {}
_ASYNC_STT_REGISTRY: dict[str, DriverFactory] = {}
_TTS_REGISTRY: dict[str, DriverFactory] = {}
_ASYNC_TTS_REGISTRY: dict[str, DriverFactory] = {}

# Image generation driver registries
_IMG_GEN_REGISTRY: dict[str, DriverFactory] = {}
_ASYNC_IMG_GEN_REGISTRY: dict[str, DriverFactory] = {}

# Track whether entry points have been loaded
_entry_points_loaded = False
_audio_entry_points_loaded = False
_img_gen_entry_points_loaded = False


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
    name = name.lower()
    if name in _SYNC_REGISTRY and not overwrite:
        raise ValueError(f"Driver '{name}' is already registered. Use overwrite=True to replace it.")
    _SYNC_REGISTRY[name] = factory
    logger.debug("Registered sync driver: %s", name)


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
    name = name.lower()
    if name in _ASYNC_REGISTRY and not overwrite:
        raise ValueError(f"Async driver '{name}' is already registered. Use overwrite=True to replace it.")
    _ASYNC_REGISTRY[name] = factory
    logger.debug("Registered async driver: %s", name)


def unregister_driver(name: str) -> bool:
    """Unregister a sync driver by name.

    Args:
        name: Provider name to unregister.

    Returns:
        True if the driver was unregistered, False if it wasn't registered.
    """
    name = name.lower()
    if name in _SYNC_REGISTRY:
        del _SYNC_REGISTRY[name]
        logger.debug("Unregistered sync driver: %s", name)
        return True
    return False


def unregister_async_driver(name: str) -> bool:
    """Unregister an async driver by name.

    Args:
        name: Provider name to unregister.

    Returns:
        True if the driver was unregistered, False if it wasn't registered.
    """
    name = name.lower()
    if name in _ASYNC_REGISTRY:
        del _ASYNC_REGISTRY[name]
        logger.debug("Unregistered async driver: %s", name)
        return True
    return False


def list_registered_drivers() -> list[str]:
    """Return a sorted list of registered sync driver names."""
    _ensure_entry_points_loaded()
    return sorted(_SYNC_REGISTRY.keys())


def list_registered_async_drivers() -> list[str]:
    """Return a sorted list of registered async driver names."""
    _ensure_entry_points_loaded()
    return sorted(_ASYNC_REGISTRY.keys())


def is_driver_registered(name: str) -> bool:
    """Check if a sync driver is registered.

    Args:
        name: Provider name to check.

    Returns:
        True if the driver is registered.
    """
    _ensure_entry_points_loaded()
    return name.lower() in _SYNC_REGISTRY


def is_async_driver_registered(name: str) -> bool:
    """Check if an async driver is registered.

    Args:
        name: Provider name to check.

    Returns:
        True if the async driver is registered.
    """
    _ensure_entry_points_loaded()
    return name.lower() in _ASYNC_REGISTRY


def get_driver_factory(name: str) -> DriverFactory:
    """Get a registered sync driver factory by name.

    Args:
        name: Provider name.

    Returns:
        The factory function.

    Raises:
        ValueError: If the driver is not registered.
    """
    _ensure_entry_points_loaded()
    name = name.lower()
    if name not in _SYNC_REGISTRY:
        raise ValueError(f"Unsupported provider '{name}'")
    return _SYNC_REGISTRY[name]


def get_async_driver_factory(name: str) -> DriverFactory:
    """Get a registered async driver factory by name.

    Args:
        name: Provider name.

    Returns:
        The factory function.

    Raises:
        ValueError: If the async driver is not registered.
    """
    _ensure_entry_points_loaded()
    name = name.lower()
    if name not in _ASYNC_REGISTRY:
        raise ValueError(f"Unsupported provider '{name}'")
    return _ASYNC_REGISTRY[name]


def load_entry_point_drivers() -> tuple[int, int]:
    """Load drivers from installed packages via entry points.

    This function scans for packages that define entry points in the
    ``prompture.drivers`` and ``prompture.async_drivers`` groups.

    Returns:
        A tuple of (sync_count, async_count) indicating how many drivers
        were loaded from entry points.

    Example pyproject.toml for a plugin package:
        [project.entry-points."prompture.drivers"]
        my_provider = "my_package.drivers:create_my_driver"

        [project.entry-points."prompture.async_drivers"]
        my_provider = "my_package.drivers:create_my_async_driver"
    """
    global _entry_points_loaded

    sync_count = 0
    async_count = 0

    # Python 3.9+ has importlib.metadata in stdlib
    # Python 3.8 needs importlib_metadata backport
    if sys.version_info >= (3, 10):
        from importlib.metadata import entry_points

        sync_eps = entry_points(group="prompture.drivers")
        async_eps = entry_points(group="prompture.async_drivers")
    else:
        from importlib.metadata import entry_points

        all_eps = entry_points()
        sync_eps = all_eps.get("prompture.drivers", [])
        async_eps = all_eps.get("prompture.async_drivers", [])

    # Load sync drivers
    for ep in sync_eps:
        try:
            # Skip if already registered (built-in drivers take precedence)
            if ep.name.lower() in _SYNC_REGISTRY:
                logger.debug("Skipping entry point driver '%s' (already registered)", ep.name)
                continue

            factory = ep.load()
            _SYNC_REGISTRY[ep.name.lower()] = factory
            sync_count += 1
            logger.info("Loaded sync driver from entry point: %s", ep.name)
        except Exception:
            logger.exception("Failed to load sync driver entry point: %s", ep.name)

    # Load async drivers
    for ep in async_eps:
        try:
            # Skip if already registered (built-in drivers take precedence)
            if ep.name.lower() in _ASYNC_REGISTRY:
                logger.debug("Skipping entry point async driver '%s' (already registered)", ep.name)
                continue

            factory = ep.load()
            _ASYNC_REGISTRY[ep.name.lower()] = factory
            async_count += 1
            logger.info("Loaded async driver from entry point: %s", ep.name)
        except Exception:
            logger.exception("Failed to load async driver entry point: %s", ep.name)

    _entry_points_loaded = True
    return (sync_count, async_count)


def _ensure_entry_points_loaded() -> None:
    """Ensure entry points have been loaded (lazy initialization)."""
    global _entry_points_loaded
    if not _entry_points_loaded:
        load_entry_point_drivers()


def _get_sync_registry() -> dict[str, DriverFactory]:
    """Get the internal sync registry dict (for internal use by drivers/__init__.py)."""
    _ensure_entry_points_loaded()
    return _SYNC_REGISTRY


def _get_async_registry() -> dict[str, DriverFactory]:
    """Get the internal async registry dict (for internal use by drivers/async_registry.py)."""
    _ensure_entry_points_loaded()
    return _ASYNC_REGISTRY


def _reset_registries() -> None:
    """Reset registries to empty state (for testing only)."""
    global _entry_points_loaded, _audio_entry_points_loaded, _img_gen_entry_points_loaded
    _SYNC_REGISTRY.clear()
    _ASYNC_REGISTRY.clear()
    _STT_REGISTRY.clear()
    _ASYNC_STT_REGISTRY.clear()
    _TTS_REGISTRY.clear()
    _ASYNC_TTS_REGISTRY.clear()
    _IMG_GEN_REGISTRY.clear()
    _ASYNC_IMG_GEN_REGISTRY.clear()
    _entry_points_loaded = False
    _audio_entry_points_loaded = False
    _img_gen_entry_points_loaded = False


# ── STT Driver Registration ───────────────────────────────────────────────


def register_stt_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register a sync STT driver factory for a provider name.

    Args:
        name: Provider name (e.g., "openai"). Will be lowercased.
        factory: Callable that takes an optional model name and returns an STT driver.
        overwrite: If True, allow overwriting an existing registration.
    """
    name = name.lower()
    if name in _STT_REGISTRY and not overwrite:
        raise ValueError(f"STT driver '{name}' is already registered. Use overwrite=True to replace it.")
    _STT_REGISTRY[name] = factory
    logger.debug("Registered sync STT driver: %s", name)


def register_async_stt_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register an async STT driver factory for a provider name."""
    name = name.lower()
    if name in _ASYNC_STT_REGISTRY and not overwrite:
        raise ValueError(f"Async STT driver '{name}' is already registered. Use overwrite=True to replace it.")
    _ASYNC_STT_REGISTRY[name] = factory
    logger.debug("Registered async STT driver: %s", name)


def unregister_stt_driver(name: str) -> bool:
    """Unregister a sync STT driver by name."""
    name = name.lower()
    if name in _STT_REGISTRY:
        del _STT_REGISTRY[name]
        return True
    return False


def unregister_async_stt_driver(name: str) -> bool:
    """Unregister an async STT driver by name."""
    name = name.lower()
    if name in _ASYNC_STT_REGISTRY:
        del _ASYNC_STT_REGISTRY[name]
        return True
    return False


def list_registered_stt_drivers() -> list[str]:
    """Return a sorted list of registered sync STT driver names."""
    _ensure_audio_entry_points_loaded()
    return sorted(_STT_REGISTRY.keys())


def list_registered_async_stt_drivers() -> list[str]:
    """Return a sorted list of registered async STT driver names."""
    _ensure_audio_entry_points_loaded()
    return sorted(_ASYNC_STT_REGISTRY.keys())


def is_stt_driver_registered(name: str) -> bool:
    """Check if a sync STT driver is registered."""
    _ensure_audio_entry_points_loaded()
    return name.lower() in _STT_REGISTRY


def is_async_stt_driver_registered(name: str) -> bool:
    """Check if an async STT driver is registered."""
    _ensure_audio_entry_points_loaded()
    return name.lower() in _ASYNC_STT_REGISTRY


def get_stt_driver_factory(name: str) -> DriverFactory:
    """Get a registered sync STT driver factory by name."""
    _ensure_audio_entry_points_loaded()
    name = name.lower()
    if name not in _STT_REGISTRY:
        raise ValueError(f"Unsupported STT provider '{name}'")
    return _STT_REGISTRY[name]


def get_async_stt_driver_factory(name: str) -> DriverFactory:
    """Get a registered async STT driver factory by name."""
    _ensure_audio_entry_points_loaded()
    name = name.lower()
    if name not in _ASYNC_STT_REGISTRY:
        raise ValueError(f"Unsupported async STT provider '{name}'")
    return _ASYNC_STT_REGISTRY[name]


# ── TTS Driver Registration ───────────────────────────────────────────────


def register_tts_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register a sync TTS driver factory for a provider name.

    Args:
        name: Provider name (e.g., "openai"). Will be lowercased.
        factory: Callable that takes an optional model name and returns a TTS driver.
        overwrite: If True, allow overwriting an existing registration.
    """
    name = name.lower()
    if name in _TTS_REGISTRY and not overwrite:
        raise ValueError(f"TTS driver '{name}' is already registered. Use overwrite=True to replace it.")
    _TTS_REGISTRY[name] = factory
    logger.debug("Registered sync TTS driver: %s", name)


def register_async_tts_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register an async TTS driver factory for a provider name."""
    name = name.lower()
    if name in _ASYNC_TTS_REGISTRY and not overwrite:
        raise ValueError(f"Async TTS driver '{name}' is already registered. Use overwrite=True to replace it.")
    _ASYNC_TTS_REGISTRY[name] = factory
    logger.debug("Registered async TTS driver: %s", name)


def unregister_tts_driver(name: str) -> bool:
    """Unregister a sync TTS driver by name."""
    name = name.lower()
    if name in _TTS_REGISTRY:
        del _TTS_REGISTRY[name]
        return True
    return False


def unregister_async_tts_driver(name: str) -> bool:
    """Unregister an async TTS driver by name."""
    name = name.lower()
    if name in _ASYNC_TTS_REGISTRY:
        del _ASYNC_TTS_REGISTRY[name]
        return True
    return False


def list_registered_tts_drivers() -> list[str]:
    """Return a sorted list of registered sync TTS driver names."""
    _ensure_audio_entry_points_loaded()
    return sorted(_TTS_REGISTRY.keys())


def list_registered_async_tts_drivers() -> list[str]:
    """Return a sorted list of registered async TTS driver names."""
    _ensure_audio_entry_points_loaded()
    return sorted(_ASYNC_TTS_REGISTRY.keys())


def is_tts_driver_registered(name: str) -> bool:
    """Check if a sync TTS driver is registered."""
    _ensure_audio_entry_points_loaded()
    return name.lower() in _TTS_REGISTRY


def is_async_tts_driver_registered(name: str) -> bool:
    """Check if an async TTS driver is registered."""
    _ensure_audio_entry_points_loaded()
    return name.lower() in _ASYNC_TTS_REGISTRY


def get_tts_driver_factory(name: str) -> DriverFactory:
    """Get a registered sync TTS driver factory by name."""
    _ensure_audio_entry_points_loaded()
    name = name.lower()
    if name not in _TTS_REGISTRY:
        raise ValueError(f"Unsupported TTS provider '{name}'")
    return _TTS_REGISTRY[name]


def get_async_tts_driver_factory(name: str) -> DriverFactory:
    """Get a registered async TTS driver factory by name."""
    _ensure_audio_entry_points_loaded()
    name = name.lower()
    if name not in _ASYNC_TTS_REGISTRY:
        raise ValueError(f"Unsupported async TTS provider '{name}'")
    return _ASYNC_TTS_REGISTRY[name]


# ── Audio Registry Internals ──────────────────────────────────────────────


def _get_stt_registry() -> dict[str, DriverFactory]:
    """Get the internal sync STT registry dict."""
    _ensure_audio_entry_points_loaded()
    return _STT_REGISTRY


def _get_async_stt_registry() -> dict[str, DriverFactory]:
    """Get the internal async STT registry dict."""
    _ensure_audio_entry_points_loaded()
    return _ASYNC_STT_REGISTRY


def _get_tts_registry() -> dict[str, DriverFactory]:
    """Get the internal sync TTS registry dict."""
    _ensure_audio_entry_points_loaded()
    return _TTS_REGISTRY


def _get_async_tts_registry() -> dict[str, DriverFactory]:
    """Get the internal async TTS registry dict."""
    _ensure_audio_entry_points_loaded()
    return _ASYNC_TTS_REGISTRY


def load_audio_entry_point_drivers() -> tuple[int, int, int, int]:
    """Load audio drivers from installed packages via entry points.

    Scans for ``prompture.stt_drivers``, ``prompture.async_stt_drivers``,
    ``prompture.tts_drivers``, and ``prompture.async_tts_drivers`` groups.

    Returns:
        A tuple of (stt_sync, stt_async, tts_sync, tts_async) counts.
    """
    global _audio_entry_points_loaded

    counts = [0, 0, 0, 0]
    groups = [
        ("prompture.stt_drivers", _STT_REGISTRY, 0),
        ("prompture.async_stt_drivers", _ASYNC_STT_REGISTRY, 1),
        ("prompture.tts_drivers", _TTS_REGISTRY, 2),
        ("prompture.async_tts_drivers", _ASYNC_TTS_REGISTRY, 3),
    ]

    if sys.version_info >= (3, 10):
        from importlib.metadata import entry_points as _ep_func

        for group_name, registry, idx in groups:
            for ep in _ep_func(group=group_name):
                try:
                    if ep.name.lower() in registry:
                        continue
                    factory = ep.load()
                    registry[ep.name.lower()] = factory
                    counts[idx] += 1
                    logger.info("Loaded audio driver from entry point: %s (%s)", ep.name, group_name)
                except Exception:
                    logger.exception("Failed to load audio driver entry point: %s", ep.name)
    else:
        from importlib.metadata import entry_points as _ep_func

        all_eps = _ep_func()
        for group_name, registry, idx in groups:
            for ep in all_eps.get(group_name, []):
                try:
                    if ep.name.lower() in registry:
                        continue
                    factory = ep.load()
                    registry[ep.name.lower()] = factory
                    counts[idx] += 1
                except Exception:
                    logger.exception("Failed to load audio driver entry point: %s", ep.name)

    _audio_entry_points_loaded = True
    return (counts[0], counts[1], counts[2], counts[3])


def _ensure_audio_entry_points_loaded() -> None:
    """Ensure audio entry points have been loaded (lazy initialization)."""
    global _audio_entry_points_loaded
    if not _audio_entry_points_loaded:
        load_audio_entry_point_drivers()


# ── Image Gen Driver Registration ─────────────────────────────────────────


def register_img_gen_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register a sync image generation driver factory for a provider name.

    Args:
        name: Provider name (e.g., "openai"). Will be lowercased.
        factory: Callable that takes an optional model name and returns an image gen driver.
        overwrite: If True, allow overwriting an existing registration.
    """
    name = name.lower()
    if name in _IMG_GEN_REGISTRY and not overwrite:
        raise ValueError(f"Image gen driver '{name}' is already registered. Use overwrite=True to replace it.")
    _IMG_GEN_REGISTRY[name] = factory
    logger.debug("Registered sync image gen driver: %s", name)


def register_async_img_gen_driver(name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
    """Register an async image generation driver factory for a provider name."""
    name = name.lower()
    if name in _ASYNC_IMG_GEN_REGISTRY and not overwrite:
        raise ValueError(f"Async image gen driver '{name}' is already registered. Use overwrite=True to replace it.")
    _ASYNC_IMG_GEN_REGISTRY[name] = factory
    logger.debug("Registered async image gen driver: %s", name)


def unregister_img_gen_driver(name: str) -> bool:
    """Unregister a sync image gen driver by name."""
    name = name.lower()
    if name in _IMG_GEN_REGISTRY:
        del _IMG_GEN_REGISTRY[name]
        return True
    return False


def unregister_async_img_gen_driver(name: str) -> bool:
    """Unregister an async image gen driver by name."""
    name = name.lower()
    if name in _ASYNC_IMG_GEN_REGISTRY:
        del _ASYNC_IMG_GEN_REGISTRY[name]
        return True
    return False


def list_registered_img_gen_drivers() -> list[str]:
    """Return a sorted list of registered sync image gen driver names."""
    _ensure_img_gen_entry_points_loaded()
    return sorted(_IMG_GEN_REGISTRY.keys())


def list_registered_async_img_gen_drivers() -> list[str]:
    """Return a sorted list of registered async image gen driver names."""
    _ensure_img_gen_entry_points_loaded()
    return sorted(_ASYNC_IMG_GEN_REGISTRY.keys())


def is_img_gen_driver_registered(name: str) -> bool:
    """Check if a sync image gen driver is registered."""
    _ensure_img_gen_entry_points_loaded()
    return name.lower() in _IMG_GEN_REGISTRY


def is_async_img_gen_driver_registered(name: str) -> bool:
    """Check if an async image gen driver is registered."""
    _ensure_img_gen_entry_points_loaded()
    return name.lower() in _ASYNC_IMG_GEN_REGISTRY


def get_img_gen_driver_factory(name: str) -> DriverFactory:
    """Get a registered sync image gen driver factory by name."""
    _ensure_img_gen_entry_points_loaded()
    name = name.lower()
    if name not in _IMG_GEN_REGISTRY:
        raise ValueError(f"Unsupported image gen provider '{name}'")
    return _IMG_GEN_REGISTRY[name]


def get_async_img_gen_driver_factory(name: str) -> DriverFactory:
    """Get a registered async image gen driver factory by name."""
    _ensure_img_gen_entry_points_loaded()
    name = name.lower()
    if name not in _ASYNC_IMG_GEN_REGISTRY:
        raise ValueError(f"Unsupported async image gen provider '{name}'")
    return _ASYNC_IMG_GEN_REGISTRY[name]


# ── Image Gen Registry Internals ──────────────────────────────────────────


def _get_img_gen_registry() -> dict[str, DriverFactory]:
    """Get the internal sync image gen registry dict."""
    _ensure_img_gen_entry_points_loaded()
    return _IMG_GEN_REGISTRY


def _get_async_img_gen_registry() -> dict[str, DriverFactory]:
    """Get the internal async image gen registry dict."""
    _ensure_img_gen_entry_points_loaded()
    return _ASYNC_IMG_GEN_REGISTRY


def load_img_gen_entry_point_drivers() -> tuple[int, int]:
    """Load image gen drivers from installed packages via entry points.

    Scans for ``prompture.img_gen_drivers`` and ``prompture.async_img_gen_drivers`` groups.

    Returns:
        A tuple of (sync_count, async_count) counts.
    """
    global _img_gen_entry_points_loaded

    counts = [0, 0]
    groups = [
        ("prompture.img_gen_drivers", _IMG_GEN_REGISTRY, 0),
        ("prompture.async_img_gen_drivers", _ASYNC_IMG_GEN_REGISTRY, 1),
    ]

    if sys.version_info >= (3, 10):
        from importlib.metadata import entry_points as _ep_func

        for group_name, registry, idx in groups:
            for ep in _ep_func(group=group_name):
                try:
                    if ep.name.lower() in registry:
                        continue
                    factory = ep.load()
                    registry[ep.name.lower()] = factory
                    counts[idx] += 1
                    logger.info("Loaded image gen driver from entry point: %s (%s)", ep.name, group_name)
                except Exception:
                    logger.exception("Failed to load image gen driver entry point: %s", ep.name)
    else:
        from importlib.metadata import entry_points as _ep_func

        all_eps = _ep_func()
        for group_name, registry, idx in groups:
            for ep in all_eps.get(group_name, []):
                try:
                    if ep.name.lower() in registry:
                        continue
                    factory = ep.load()
                    registry[ep.name.lower()] = factory
                    counts[idx] += 1
                except Exception:
                    logger.exception("Failed to load image gen driver entry point: %s", ep.name)

    _img_gen_entry_points_loaded = True
    return (counts[0], counts[1])


def _ensure_img_gen_entry_points_loaded() -> None:
    """Ensure image gen entry points have been loaded (lazy initialization)."""
    global _img_gen_entry_points_loaded
    if not _img_gen_entry_points_loaded:
        load_img_gen_entry_point_drivers()
