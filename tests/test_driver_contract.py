"""Driver capability contract tests.

For every LLM driver registered in ``PROVIDER_DESCRIPTOR_MAP``, verifies
that the declared ``supports_*`` flags match the methods that are actually
overridden — bidirectionally.  Catches the class of bug where async ollama
claims streaming support without implementing ``generate_messages_stream``
(which causes ``auto_select_strategy`` to recommend a strategy the driver
cannot honor).
"""

from __future__ import annotations

import pytest

from prompture.drivers.async_base import AsyncDriver
from prompture.drivers.base import Driver
from prompture.drivers.provider_descriptors import (
    PROVIDER_DESCRIPTOR_MAP,
    _resolve_cls,
)

# Capability flag → method that must be overridden when the flag is True.
# Limited to flags whose support is determined by overriding a single method.
# (json_mode / json_schema / vision are negotiated through option flags inside
# ``generate``-family methods, not by overriding a separate method, so they're
# excluded from this contract.)
_FLAG_TO_METHOD = {
    "supports_streaming": "generate_messages_stream",
    "supports_tool_use": "generate_messages_with_tools",
    "supports_messages": "generate_messages",
}


def _collect_llm_driver_classes() -> list[tuple[str, type]]:
    """Resolve every LLM driver class declared in the descriptor map.

    Skips aliases (they share the canonical's classes) and drivers whose
    optional dependencies are not installed in the test environment.
    """
    pairs: list[tuple[str, type]] = []
    for name, desc in PROVIDER_DESCRIPTOR_MAP.items():
        if desc.alias_for:
            continue
        for kind, spec in (("sync", desc.llm_sync), ("async", desc.llm_async)):
            if spec is None:
                continue
            try:
                cls = _resolve_cls(spec.cls_path)
            except Exception:
                continue
            pairs.append((f"{name}-{kind}", cls))
    return pairs


_DRIVER_CLASSES = _collect_llm_driver_classes()


def _base_for(cls: type) -> type:
    if issubclass(cls, AsyncDriver):
        return AsyncDriver
    if issubclass(cls, Driver):
        return Driver
    raise AssertionError(f"{cls.__name__} is not a Driver or AsyncDriver subclass")


@pytest.mark.parametrize(
    "label,cls",
    _DRIVER_CLASSES,
    ids=[label for label, _ in _DRIVER_CLASSES],
)
def test_declared_flags_match_implementation(label: str, cls: type) -> None:
    """Bidirectional: ``supports_X = True`` ⇔ corresponding method overridden."""
    base = _base_for(cls)
    failures: list[str] = []
    for flag, method_name in _FLAG_TO_METHOD.items():
        declared = bool(getattr(cls, flag, False))
        cls_method = getattr(cls, method_name, None)
        base_method = getattr(base, method_name, None)
        implemented = cls_method is not None and cls_method is not base_method

        if declared and not implemented:
            failures.append(
                f"{flag}=True but {method_name} not overridden — "
                f"auto-strategy will recommend an unsupported feature"
            )
        elif implemented and not declared:
            failures.append(
                f"{method_name} overridden but {flag}=False — "
                f"capability is hidden from auto-strategy selection"
            )
    assert not failures, "{} ({}):\n  - {}".format(
        cls.__name__, label, "\n  - ".join(failures)
    )


def test_capability_resolution_uses_driver_instance() -> None:
    """Regression: live driver flags must trump the provider-level registry.

    Pre-fix, ``_populate_from_descriptors`` registered sync flags at the
    provider level and ``get_capabilities`` returned them before consulting
    the live driver instance.  Async ollama therefore claimed streaming
    support (inherited from the sync sibling's registry entry) even though
    ``AsyncOllamaDriver`` does not implement ``generate_messages_stream``.
    """
    from prompture.drivers.async_ollama_driver import AsyncOllamaDriver
    from prompture.drivers.ollama_driver import OllamaDriver
    from prompture.infra.capabilities import get_capabilities

    assert OllamaDriver.supports_streaming is True
    assert getattr(AsyncOllamaDriver, "supports_streaming", False) is False

    async_driver = AsyncOllamaDriver(endpoint="http://localhost:11434")
    caps = get_capabilities("ollama/llama3", driver=async_driver)
    assert caps.streaming is False, (
        "Capability registry returned streaming=True for an AsyncOllamaDriver "
        "instance — the provider-level cache is shadowing live driver flags."
    )


def test_capability_resolution_user_override_beats_driver() -> None:
    """User overrides must continue to win over live driver flags."""
    from prompture.drivers.async_ollama_driver import AsyncOllamaDriver
    from prompture.infra.capabilities import (
        ProviderCapabilities,
        clear_overrides,
        get_capabilities,
        override_capabilities,
    )

    try:
        override_capabilities(
            "ollama", ProviderCapabilities(json_mode=True, streaming=True)
        )
        async_driver = AsyncOllamaDriver(endpoint="http://localhost:11434")
        caps = get_capabilities("ollama/llama3", driver=async_driver)
        assert caps.streaming is True, "user override must trump live driver flags"
    finally:
        clear_overrides()
