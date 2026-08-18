"""Muapi.ai aggregator driver (async) — thin subclass of the async aggregator base."""

from __future__ import annotations

from typing import Any

from ..infra.cost_mixin import ImageCostMixin, VideoCostMixin
from .async_aggregator_base import (
    AsyncAggregatorImageDriver,
    AsyncAggregatorLipsyncDriver,
    AsyncAggregatorMusicDriver,
    AsyncAggregatorVideoDriver,
)
from .muapi_aggregator_driver import _MuapiMixin


class AsyncMuapiImageGenDriver(ImageCostMixin, _MuapiMixin, AsyncAggregatorImageDriver):
    """Async image generation + i2i editing via Muapi."""

    IMAGE_PRICING: dict[str, dict[str, float]] = {}


class AsyncMuapiVideoGenDriver(VideoCostMixin, _MuapiMixin, AsyncAggregatorVideoDriver):
    """Async video generation via Muapi."""

    VIDEO_PRICING: dict[str, dict[str, Any]] = {}


class AsyncMuapiLipsyncDriver(_MuapiMixin, AsyncAggregatorLipsyncDriver):
    """Async lipsync via Muapi."""


class AsyncMuapiMusicGenDriver(_MuapiMixin, AsyncAggregatorMusicDriver):
    """Async music generation via Muapi."""
