"""Provider capability registry for structured output strategy selection.

Maps providers to their known capabilities (JSON mode, JSON schema,
tool use, streaming, vision) so the extraction layer can automatically
pick the best structured-output strategy without trial-and-error.

The registry is populated from driver flags by default.  Users can
override entries via :func:`override_capabilities` for custom setups
(e.g. a local model behind a generic HTTP driver that actually supports
tool calling).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("prompture.capabilities")

_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Capability snapshot for a provider or specific model."""

    json_mode: bool | None = None
    json_schema: bool | None = None
    tool_use: bool | None = None
    streaming: bool | None = None
    vision: bool | None = None

    def best_strategy(self) -> str:
        """Return the best StructuredOutputStrategy name for this provider.

        Resolution order:
        1. ``provider_native`` — if json_mode or json_schema is supported.
        2. ``tool_call`` — if tool_use is supported.
        3. ``prompted_repair`` — always available, universal fallback.
        """
        if self.json_schema or self.json_mode:
            return "provider_native"
        if self.tool_use:
            return "tool_call"
        return "prompted_repair"


# Internal registries: provider-level and model-level overrides.
_provider_caps: dict[str, ProviderCapabilities] = {}
_model_caps: dict[str, ProviderCapabilities] = {}  # keyed by "provider/model"
_user_overrides: dict[str, ProviderCapabilities] = {}  # user overrides (highest priority)


def _caps_from_driver(driver: Any) -> ProviderCapabilities:
    """Build ProviderCapabilities from a driver instance's flags."""
    return ProviderCapabilities(
        json_mode=getattr(driver, "supports_json_mode", None),
        json_schema=getattr(driver, "supports_json_schema", None),
        tool_use=getattr(driver, "supports_tool_use", None),
        streaming=getattr(driver, "supports_streaming", None),
        vision=getattr(driver, "supports_vision", None),
    )


def register_provider(name: str, caps: ProviderCapabilities) -> None:
    """Register capabilities for a provider (e.g. ``"openai"``)."""
    with _lock:
        _provider_caps[name] = caps


def register_model(model_str: str, caps: ProviderCapabilities) -> None:
    """Register capabilities for a specific model (e.g. ``"openai/gpt-3.5-turbo"``)."""
    with _lock:
        _model_caps[model_str] = caps


def override_capabilities(key: str, caps: ProviderCapabilities) -> None:
    """User-level override. Takes highest priority over auto-detected caps.

    *key* can be a provider name (``"ollama"``) or a full model string
    (``"ollama/llama3:8b"``).
    """
    with _lock:
        _user_overrides[key] = caps


def clear_overrides() -> None:
    """Remove all user overrides (useful for testing)."""
    with _lock:
        _user_overrides.clear()


def get_capabilities(
    model_str: str,
    *,
    driver: Any | None = None,
) -> ProviderCapabilities:
    """Resolve capabilities for a model string like ``"openai/gpt-4"``.

    Resolution order (first match wins at each field level):
    1. User overrides (exact model match, then provider match).
    2. Model-level registry.
    3. Provider-level registry.
    4. Live driver instance flags (if *driver* is provided).
    5. models.dev metadata.
    6. Unknown (all ``None``).
    """
    if not isinstance(model_str, str) or not model_str:
        # Unknown model string — fall back to driver flags or empty caps
        if driver is not None:
            return _caps_from_driver(driver)
        return ProviderCapabilities()

    provider = model_str.split("/", 1)[0] if "/" in model_str else model_str

    with _lock:
        # 1. User overrides (exact model, then provider)
        if model_str in _user_overrides:
            return _user_overrides[model_str]
        if provider in _user_overrides:
            return _user_overrides[provider]

        # 2. Model-level registry
        if model_str in _model_caps:
            return _model_caps[model_str]

        # 3. Provider-level registry
        if provider in _provider_caps:
            return _provider_caps[provider]

    # 4. Live driver flags
    if driver is not None:
        caps = _caps_from_driver(driver)
        # Cache for next time
        with _lock:
            _model_caps[model_str] = caps
        return caps

    # 5. models.dev metadata
    if "/" in model_str:
        model_id = model_str.split("/", 1)[1]
        try:
            from .model_rates import get_model_capabilities

            mc = get_model_capabilities(provider, model_id)
            if mc is not None:
                caps = ProviderCapabilities(
                    json_mode=mc.supports_structured_output if mc.supports_structured_output is not None else None,
                    json_schema=mc.supports_structured_output,
                    tool_use=mc.supports_tool_use,
                    vision=mc.supports_vision,
                )
                with _lock:
                    _model_caps[model_str] = caps
                return caps
        except Exception:
            pass

    # 6. Unknown
    return ProviderCapabilities()


def _resolve_driver_class(spec: Any) -> type | None:
    """Lazily resolve a DriverSpec's cls_path to the actual class."""
    cls_path = getattr(spec, "cls_path", None)
    if not cls_path or not isinstance(cls_path, str):
        return None
    try:
        module_path, cls_name = cls_path.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(f"prompture.drivers.{module_path}")
        return getattr(mod, cls_name, None)
    except Exception:
        return None


def _populate_from_descriptors() -> None:
    """Auto-populate provider-level caps from PROVIDER_DESCRIPTOR_MAP."""
    try:
        from ..drivers.provider_descriptors import PROVIDER_DESCRIPTOR_MAP

        for name, desc in PROVIDER_DESCRIPTOR_MAP.items():
            if desc.alias_for:
                continue
            # Use the LLM sync driver spec to read class-level capability flags
            spec = desc.llm_sync
            if spec is None:
                continue
            driver_cls = _resolve_driver_class(spec)
            if driver_cls is None:
                continue
            caps = ProviderCapabilities(
                json_mode=getattr(driver_cls, "supports_json_mode", False),
                json_schema=getattr(driver_cls, "supports_json_schema", False),
                tool_use=getattr(driver_cls, "supports_tool_use", False),
                streaming=getattr(driver_cls, "supports_streaming", False),
                vision=getattr(driver_cls, "supports_vision", False),
            )
            register_provider(name, caps)
    except Exception:
        logger.debug("Failed to auto-populate capabilities from descriptors", exc_info=True)


# Auto-populate on import
_populate_from_descriptors()
