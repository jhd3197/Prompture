"""Plugin-based provider discovery system for Prompture.

Built-in providers are loaded from :mod:`prompture.plugins.builtins`.
Third-party providers may register additional :class:`ProviderPlugin` instances
via the ``prompture.providers`` entry-point group.
"""

from __future__ import annotations

from .base import ProviderPlugin
from .discovery import _load_builtins, discover_plugins, load_plugins

__all__ = ["ProviderPlugin", "_load_builtins", "discover_plugins", "load_plugins"]
