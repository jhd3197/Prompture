"""Driver capability contract tests.

For every LLM driver registered in ``PROVIDER_DESCRIPTOR_MAP``, verifies
that the declared ``supports_*`` flags match the methods that are actually
overridden — bidirectionally.  Catches the class of bug where async ollama
claimed streaming support without implementing ``generate_messages_stream``
(which caused ``auto_select_strategy`` to recommend a strategy the driver
could not honor).

The validation logic lives in :mod:`prompture.testing.driver_contract` so
third-party plugin authors can apply the same contract to their own
drivers.
"""

from __future__ import annotations

import pytest

from prompture.drivers.provider_descriptors import (
    PROVIDER_DESCRIPTOR_MAP,
    _resolve_cls,
)
from prompture.testing import assert_driver_capabilities


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


@pytest.mark.parametrize(
    "label,cls",
    _DRIVER_CLASSES,
    ids=[label for label, _ in _DRIVER_CLASSES],
)
def test_declared_flags_match_implementation(label: str, cls: type) -> None:
    """Every built-in driver must satisfy the capability contract."""
    assert_driver_capabilities(cls)


def test_capability_resolution_uses_driver_instance() -> None:
    """Regression: live driver flags must trump the provider-level registry.

    Pre-fix, ``_populate_from_descriptors`` registered sync flags at the
    provider level and ``get_capabilities`` returned them before consulting
    the live driver instance, so async drivers could claim capabilities
    they didn't implement.

    We synthesize a stripped-down ``AsyncOllamaDriver`` subclass that
    drops streaming support to exercise the lookup path with a known
    mismatch between provider-level claims and instance-level reality.
    """
    from prompture.drivers.async_base import AsyncDriver
    from prompture.drivers.async_ollama_driver import AsyncOllamaDriver
    from prompture.drivers.ollama_driver import OllamaDriver
    from prompture.infra.capabilities import get_capabilities

    assert OllamaDriver.supports_streaming is True
    assert AsyncOllamaDriver.supports_streaming is True

    class NoStreamAsyncOllama(AsyncOllamaDriver):
        supports_streaming = False
        # Drop the override so it falls back to AsyncDriver's NotImplementedError default.
        generate_messages_stream = AsyncDriver.generate_messages_stream

    async_driver = NoStreamAsyncOllama(endpoint="http://localhost:11434")
    caps = get_capabilities("ollama/llama3", driver=async_driver)
    assert caps.streaming is False, (
        "Capability registry returned streaming=True for an async driver "
        "instance that explicitly opts out — the provider-level cache is "
        "shadowing live driver flags."
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
        override_capabilities("ollama", ProviderCapabilities(json_mode=True, streaming=True))
        async_driver = AsyncOllamaDriver(endpoint="http://localhost:11434")
        caps = get_capabilities("ollama/llama3", driver=async_driver)
        assert caps.streaming is True, "user override must trump live driver flags"
    finally:
        clear_overrides()


def test_validate_driver_capabilities_returns_violations() -> None:
    """Programmatic API returns a list — useful for plugin authors who
    want to integrate the check into their own assertions or CI logic."""
    from prompture.drivers.base import Driver
    from prompture.drivers.ollama_driver import OllamaDriver
    from prompture.testing import validate_driver_capabilities

    assert validate_driver_capabilities(OllamaDriver) == []

    # Synthetic class with declared-but-unimplemented streaming
    class BrokenDriver(OllamaDriver):
        pass

    BrokenDriver.supports_streaming = True
    BrokenDriver.generate_messages_stream = Driver.generate_messages_stream

    violations = validate_driver_capabilities(BrokenDriver)
    assert any("supports_streaming" in v for v in violations)
