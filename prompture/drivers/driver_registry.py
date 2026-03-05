"""Generic driver registry with entry-point discovery.

Provides ``DriverRegistry``, a typed wrapper around a ``dict[str, DriverFactory]``
that standardises registration, lookup, and plugin loading for any modality
(LLM, STT, TTS, image-gen, embedding) in both sync and async variants.
"""

from __future__ import annotations

import logging
import sys
from typing import Callable

logger = logging.getLogger("prompture.drivers.registry")

# A factory takes an optional model name and returns a driver instance.
DriverFactory = Callable[[str | None], object]


class DriverRegistry:
    """Registry of named driver factories for a single modality + sync/async variant.

    Parameters:
        modality: Human label used in log/error messages (e.g. ``"LLM"``, ``"STT"``).
        entry_point_group: importlib entry-point group name for plugin discovery
            (e.g. ``"prompture.drivers"``).  ``None`` disables entry-point loading.
    """

    def __init__(
        self,
        modality: str,
        entry_point_group: str | None = None,
        error_prefix: str = "",
    ) -> None:
        self._modality = modality
        self._entry_point_group = entry_point_group
        self._error_prefix = error_prefix
        self._registry: dict[str, DriverFactory] = {}
        self._entry_points_loaded = False

    # -- public API ----------------------------------------------------------

    def register(self, name: str, factory: DriverFactory, *, overwrite: bool = False) -> None:
        """Register a driver factory under *name* (lowercased)."""
        name = name.lower()
        if name in self._registry and not overwrite:
            raise ValueError(
                f"{self._modality} driver '{name}' is already registered. Use overwrite=True to replace it."
            )
        self._registry[name] = factory
        logger.debug("Registered %s driver: %s", self._modality, name)

    def unregister(self, name: str) -> bool:
        """Remove *name* from the registry.  Returns ``True`` if it existed."""
        name = name.lower()
        if name in self._registry:
            del self._registry[name]
            logger.debug("Unregistered %s driver: %s", self._modality, name)
            return True
        return False

    def is_registered(self, name: str) -> bool:
        self._ensure_entry_points_loaded()
        return name.lower() in self._registry

    def list_names(self) -> list[str]:
        """Return a sorted list of registered driver names."""
        self._ensure_entry_points_loaded()
        return sorted(self._registry.keys())

    def get_factory(self, name: str) -> DriverFactory:
        """Return the factory for *name*, raising ``ValueError`` if missing."""
        self._ensure_entry_points_loaded()
        name = name.lower()
        if name not in self._registry:
            raise ValueError(f"Unsupported {self._error_prefix}provider '{name}'")
        return self._registry[name]

    @property
    def dict(self) -> dict[str, DriverFactory]:
        """Direct reference to the internal dict (backwards compat)."""
        self._ensure_entry_points_loaded()
        return self._registry

    def reset(self) -> None:
        """Clear all registrations and mark entry points as unloaded (testing only)."""
        self._registry.clear()
        self._entry_points_loaded = False

    # -- entry-point loading -------------------------------------------------

    def load_entry_points(self) -> int:
        """Discover and load plugins from the configured entry-point group.

        Returns the number of new drivers loaded.  Skips names already present
        so that built-in drivers take precedence over plugins.
        """
        if self._entry_point_group is None:
            self._entry_points_loaded = True
            return 0

        count = 0
        if sys.version_info >= (3, 10):
            from importlib.metadata import entry_points

            eps = entry_points(group=self._entry_point_group)
        else:
            from importlib.metadata import entry_points

            eps = entry_points().get(self._entry_point_group, [])

        for ep in eps:
            try:
                if ep.name.lower() in self._registry:
                    logger.debug(
                        "Skipping entry-point %s driver '%s' (already registered)",
                        self._modality,
                        ep.name,
                    )
                    continue
                factory = ep.load()
                self._registry[ep.name.lower()] = factory
                count += 1
                logger.info(
                    "Loaded %s driver from entry point: %s (%s)",
                    self._modality,
                    ep.name,
                    self._entry_point_group,
                )
            except Exception:
                logger.exception("Failed to load %s entry point: %s", self._modality, ep.name)

        self._entry_points_loaded = True
        return count

    def _ensure_entry_points_loaded(self) -> None:
        if not self._entry_points_loaded:
            self.load_entry_points()
