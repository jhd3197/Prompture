"""Model resolution with layered fallback chains.

Provides a generic :class:`ModelResolver` that walks an ordered list of
:data:`ResolutionLayer` callables until one returns a non-empty model string
for the requested slot.  Fallback slots (e.g. ``"utility"`` → ``"default"``)
are tried automatically when the primary slot yields nothing.

Quick start::

    from prompture.pipeline.resolver import (
        ModelResolver, dict_layer, attr_layer, SLOT_DEFAULT, SLOT_UTILITY,
    )

    resolver = ModelResolver(layers=[
        dict_layer({"utility": "openai/gpt-4o-mini", "default": "openai/gpt-4o"}),
    ])
    model = resolver.resolve(SLOT_UTILITY)   # "openai/gpt-4o-mini"
    model = resolver.resolve(SLOT_DEFAULT)   # "openai/gpt-4o"
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..exceptions import ConfigurationError

# ── Slot constants ─────────────────────────────────────────────────────────

SLOT_DEFAULT: str = "default"
SLOT_UTILITY: str = "utility"
SLOT_IMAGE: str = "image"
SLOT_AUDIO: str = "audio"
SLOT_EMBEDDING: str = "embedding"
SLOT_STRUCTURED: str = "structured"

# ── Types ──────────────────────────────────────────────────────────────────

ResolutionLayer = Callable[[str], str | None]
"""A callable that receives a slot name and returns a model string or ``None``."""

# ── Exceptions ─────────────────────────────────────────────────────────────


class NoModelConfiguredError(ConfigurationError):
    """Raised when no model is configured anywhere in the fallback chain."""

    def __init__(self, slot: str = "default") -> None:
        self.slot = slot
        super().__init__(
            f"No AI model configured for slot '{slot}'. "
            "Please set a default model in settings or pass one explicitly."
        )


# ── Default fallback mapping ──────────────────────────────────────────────

DEFAULT_FALLBACK_SLOTS: dict[str, list[str]] = {
    "utility": ["default"],
    "structured": ["default"],
}

# ── Layer factories ────────────────────────────────────────────────────────


def dict_layer(mapping: dict[str, str | None]) -> ResolutionLayer:
    """Create a resolution layer backed by a plain dictionary.

    Args:
        mapping: Maps slot names to model strings (or ``None``).

    Returns:
        A :data:`ResolutionLayer` callable.

    Example::

        layer = dict_layer({"utility": "openai/gpt-4o-mini", "default": "openai/gpt-4o"})
        layer("utility")   # "openai/gpt-4o-mini"
        layer("image")     # None
    """

    def _resolve(slot: str) -> str | None:
        val = mapping.get(slot)
        if val and str(val).strip():
            return str(val).strip()
        return None

    return _resolve


def attr_layer(
    obj: object,
    attr_map: dict[str, str] | None = None,
) -> ResolutionLayer:
    """Create a resolution layer that reads attributes from an object.

    Args:
        obj: Any object whose attributes hold model strings
            (e.g. a config section, a DB record).
        attr_map: Maps slot names to attribute names on *obj*.
            Defaults to ``{"default": "model", "utility": "utility_model"}``.

    Returns:
        A :data:`ResolutionLayer` callable.

    Example::

        class Cfg:
            model = "openai/gpt-4o"
            utility_model = "openai/gpt-4o-mini"

        layer = attr_layer(Cfg())
        layer("default")   # "openai/gpt-4o"
        layer("utility")   # "openai/gpt-4o-mini"
    """
    if attr_map is None:
        attr_map = {"default": "model", "utility": "utility_model"}

    def _resolve(slot: str) -> str | None:
        attr_name = attr_map.get(slot)  # type: ignore[union-attr]
        if attr_name is None:
            return None
        val = getattr(obj, attr_name, None)
        if val and str(val).strip():
            return str(val).strip()
        return None

    return _resolve


# ── ModelResolver ──────────────────────────────────────────────────────────


@dataclass
class ModelResolver:
    """Walk an ordered list of layers to resolve a model string for a slot.

    Attributes:
        layers: Ordered highest-to-lowest priority list of
            :data:`ResolutionLayer` callables.
        fallback_slots: Maps a slot to a list of alternative slots to try
            when the primary slot yields nothing.  Defaults to
            ``{"utility": ["default"], "structured": ["default"]}``.
    """

    layers: list[ResolutionLayer] = field(default_factory=list)
    fallback_slots: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_FALLBACK_SLOTS))

    def resolve(self, slot: str) -> str:
        """Resolve a model string for *slot*, raising on failure.

        Walks every layer for the primary *slot*, then for each fallback
        slot in order.

        Raises:
            NoModelConfiguredError: If no layer returns a model string.
        """
        result = self._try_resolve(slot)
        if result is not None:
            return result

        # Try fallback slots
        for fallback in self.fallback_slots.get(slot, []):
            result = self._try_resolve(fallback)
            if result is not None:
                return result

        raise NoModelConfiguredError(slot)

    def resolve_or(self, slot: str, default: str) -> str:
        """Resolve a model string for *slot*, returning *default* on failure."""
        try:
            return self.resolve(slot)
        except NoModelConfiguredError:
            return default

    def add_layer(self, layer: ResolutionLayer, priority: int = -1) -> None:
        """Insert a layer at *priority* index (``-1`` means append to end)."""
        if priority < 0:
            self.layers.append(layer)
        else:
            self.layers.insert(priority, layer)

    # ── internals ──────────────────────────────────────────────────────

    def _try_resolve(self, slot: str) -> str | None:
        """Walk layers for *slot*, return first non-empty result."""
        for layer in self.layers:
            result = layer(slot)
            if result is not None:
                return result
        return None


# ── Convenience function ───────────────────────────────────────────────────


def resolve_model(
    slot: str,
    layers: list[ResolutionLayer],
    fallback_slots: dict[str, list[str]] | None = None,
) -> str:
    """One-shot model resolution without constructing a :class:`ModelResolver`.

    Args:
        slot: The slot to resolve (e.g. :data:`SLOT_UTILITY`).
        layers: Ordered list of :data:`ResolutionLayer` callables.
        fallback_slots: Optional fallback mapping.  Uses
            :data:`DEFAULT_FALLBACK_SLOTS` when ``None``.

    Returns:
        The resolved model string.

    Raises:
        NoModelConfiguredError: If no layer returns a model string.
    """
    kwargs: dict[str, object] = {"layers": layers}
    if fallback_slots is not None:
        kwargs["fallback_slots"] = fallback_slots
    return ModelResolver(**kwargs).resolve(slot)  # type: ignore[arg-type]
