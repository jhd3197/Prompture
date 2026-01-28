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

# Track whether entry points have been loaded
_entry_points_loaded = False


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
    global _entry_points_loaded
    _SYNC_REGISTRY.clear()
    _ASYNC_REGISTRY.clear()
    _entry_points_loaded = False
