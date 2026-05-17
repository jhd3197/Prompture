"""Plugin discovery via built-ins + ``prompture.providers`` entry points."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from ..drivers.provider_descriptors import ProviderDescriptor
from .base import ProviderPlugin

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "prompture.providers"

_DISCOVERED_PLUGINS: dict[str, ProviderPlugin] = {}
_BUILTINS_LOADED = False
_EXTERNALS_LOADED = False


def _reset_plugin_cache() -> None:
    """Reset the discovery cache. Intended for tests only."""
    global _BUILTINS_LOADED, _EXTERNALS_LOADED
    _DISCOVERED_PLUGINS.clear()
    _BUILTINS_LOADED = False
    _EXTERNALS_LOADED = False


def discover_plugins() -> list[ProviderPlugin]:
    """Discover all registered :class:`ProviderPlugin` instances.

    Combines built-in plugins (always loaded) with any third-party plugins
    declared via the ``prompture.providers`` entry-point group. Results are
    cached after the first call.
    """
    _load_builtins()
    _load_external_entry_points()
    return list(_DISCOVERED_PLUGINS.values())


def _load_builtins() -> None:
    """Load the built-in plugin set. Idempotent."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from .builtins import BUILTIN_PLUGINS

    for plugin in BUILTIN_PLUGINS:
        _DISCOVERED_PLUGINS[plugin.name] = plugin
    _BUILTINS_LOADED = True


def _load_external_entry_points() -> None:
    """Load third-party plugins declared via ``prompture.providers`` entry points."""
    global _EXTERNALS_LOADED
    if _EXTERNALS_LOADED:
        return
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:
        # Older importlib.metadata API (Python <3.10)
        eps = entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]

    for ep in eps:
        try:
            loaded = ep.load()
            plugin = loaded() if isinstance(loaded, type) else loaded
            if not isinstance(plugin, ProviderPlugin):
                logger.warning(
                    "Entry point %s did not return a ProviderPlugin instance; skipping.",
                    ep.name,
                )
                continue
            if plugin.name in _DISCOVERED_PLUGINS:
                logger.warning(
                    "Plugin %s already registered (overriding via entry point).",
                    plugin.name,
                )
            _DISCOVERED_PLUGINS[plugin.name] = plugin
        except Exception as exc:
            logger.error("Failed to load plugin entry point %s: %s", ep.name, exc, exc_info=True)
    _EXTERNALS_LOADED = True


def load_plugins() -> list[ProviderDescriptor]:
    """Discover all plugins and return their combined :class:`ProviderDescriptor` list.

    Canonical descriptors are collected first across all plugins; alias-only
    plugins (or alias entries inside a plugin) can rely on canonical names
    being present. Plugins whose ``descriptors()`` raises are logged and
    skipped without crashing the load.
    """
    plugins = discover_plugins()
    canonical: list[ProviderDescriptor] = []
    aliases: list[ProviderDescriptor] = []
    for plugin in plugins:
        try:
            for desc in plugin.descriptors():
                if desc.alias_for is None:
                    canonical.append(desc)
                else:
                    aliases.append(desc)
        except Exception as exc:
            logger.error(
                "Plugin %s failed to provide descriptors: %s",
                plugin.name,
                exc,
                exc_info=True,
            )
    return canonical + aliases
