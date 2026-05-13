"""Phase 14 — plugin-based provider discovery tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prompture.drivers.provider_descriptors import (
    PROVIDER_DESCRIPTOR_MAP,
    PROVIDER_DESCRIPTORS,
    DriverSpec,
    ProviderDescriptor,
)
from prompture.plugins import ProviderPlugin, discover_plugins, load_plugins
from prompture.plugins import discovery as _discovery
from prompture.plugins.builtins import BUILTIN_PLUGINS

# ── ABC enforcement ───────────────────────────────────────────────────────


def test_provider_plugin_abc_requires_descriptors():
    """A subclass that doesn't implement descriptors() cannot be instantiated."""

    class BadPlugin(ProviderPlugin):
        name = "bad"

    with pytest.raises(TypeError):
        BadPlugin()  # type: ignore[abstract]


def test_provider_plugin_subclass_works():
    """A subclass that implements descriptors() is instantiable."""

    class GoodPlugin(ProviderPlugin):
        name = "good"

        def descriptors(self):
            return []

    plugin = GoodPlugin()
    assert plugin.descriptors() == []


# ── Built-in discovery ────────────────────────────────────────────────────


def test_discover_plugins_returns_builtins():
    plugins = discover_plugins()
    plugin_names = {p.name for p in plugins}
    assert "openai_builtin" in plugin_names
    assert "anthropic_builtin" in plugin_names
    assert "google_builtin" in plugin_names
    # Every BUILTIN_PLUGINS entry must surface.
    for bp in BUILTIN_PLUGINS:
        assert bp.name in plugin_names


def test_load_plugins_returns_known_provider_names():
    descs = load_plugins()
    names = {d.name for d in descs}
    for expected in {
        "openai",
        "claude",
        "google",
        "google_vertexai",
        "azure",
        "bedrock",
        "groq",
        "grok",
        "openrouter",
        "moonshot",
        "modelscope",
        "zai",
        "openai_compatible",
        "ollama",
        "lmstudio",
        "local_http",
        "airllm",
        "huggingface",
        "cachibot",
        "elevenlabs",
        "cartesia",
        "deepgram",
        "assemblyai",
        "stability",
        "runway",
        "kling",
        "fal",
        "ideogram",
        "bfl",
        "minimax",
        "luma",
        "pika",
        "cohere",
        "voyage",
        "jina",
        "nomic",
        "mixedbread",
        "mistral",
        "deepseek",
    }:
        assert expected in names, f"missing canonical provider: {expected}"


def test_aliases_present_via_load_plugins():
    descs = load_plugins()
    names = {d.name for d in descs}
    for alias in {
        "anthropic",
        "gemini",
        "chatgpt",
        "xai",
        "lm_studio",
        "lm-studio",
        "zhipu",
        "hf",
        "dalle",
        "runwayml",
        "vertex",
        "vertexai",
        "hailuo",
        "mistralai",
        "dream-machine",
        "lumalabs",
        "flux",
        "mxbai",
    }:
        assert alias in names, f"missing alias: {alias}"


# ── Backwards compatibility ───────────────────────────────────────────────


def test_provider_descriptor_map_backwards_compatible():
    """The lazy PROVIDER_DESCRIPTOR_MAP attribute still surfaces the expected providers."""
    assert "openai" in PROVIDER_DESCRIPTOR_MAP
    assert "claude" in PROVIDER_DESCRIPTOR_MAP
    assert "chatgpt" in PROVIDER_DESCRIPTOR_MAP
    assert "mxbai" in PROVIDER_DESCRIPTOR_MAP
    assert isinstance(PROVIDER_DESCRIPTOR_MAP["openai"], ProviderDescriptor)


def test_provider_descriptors_list_backwards_compatible():
    assert isinstance(PROVIDER_DESCRIPTORS, list)
    assert len(PROVIDER_DESCRIPTORS) > 30  # 39 canonical + 18 aliases = 57


def test_get_driver_for_model_still_works():
    from prompture.drivers import get_driver_for_model

    driver = get_driver_for_model("openai/gpt-4o-mini")
    assert driver.__class__.__name__ == "OpenAIDriver"


def test_alias_descriptor_inherits_canonical_specs():
    chatgpt = PROVIDER_DESCRIPTOR_MAP["chatgpt"]
    openai = PROVIDER_DESCRIPTOR_MAP["openai"]
    assert chatgpt.alias_for == "openai"
    assert chatgpt.llm_sync == openai.llm_sync
    assert chatgpt.embedding_sync == openai.embedding_sync
    # chatgpt alias does NOT inherit img_gen
    assert chatgpt.img_gen_sync is None


def test_dalle_alias_only_inherits_img_gen():
    dalle = PROVIDER_DESCRIPTOR_MAP["dalle"]
    openai = PROVIDER_DESCRIPTOR_MAP["openai"]
    assert dalle.alias_for == "openai"
    assert dalle.img_gen_sync == openai.img_gen_sync
    assert dalle.llm_sync is None


# ── Plugin failure isolation ──────────────────────────────────────────────


def test_failing_plugin_is_logged_and_skipped(caplog):
    """A plugin whose descriptors() raises must be skipped without crashing the load."""

    class CrashyPlugin(ProviderPlugin):
        name = "crashy"

        def descriptors(self):
            raise RuntimeError("boom")

    # Save state
    saved = dict(_discovery._DISCOVERED_PLUGINS)
    saved_b = _discovery._BUILTINS_LOADED
    saved_e = _discovery._EXTERNALS_LOADED
    try:
        _discovery._DISCOVERED_PLUGINS.clear()
        _discovery._BUILTINS_LOADED = False
        _discovery._EXTERNALS_LOADED = True  # skip external entry-point loading
        # Inject built-ins + our crashy plugin
        _discovery._load_builtins()
        _discovery._DISCOVERED_PLUGINS["crashy"] = CrashyPlugin()

        descs = load_plugins()
        names = {d.name for d in descs}
        # Built-ins still loaded
        assert "openai" in names
        # Crashy plugin produced nothing
        assert "crashy" not in names
    finally:
        _discovery._DISCOVERED_PLUGINS.clear()
        _discovery._DISCOVERED_PLUGINS.update(saved)
        _discovery._BUILTINS_LOADED = saved_b
        _discovery._EXTERNALS_LOADED = saved_e


# ── External plugin injection ─────────────────────────────────────────────


def test_external_plugin_descriptor_surfaces():
    """Manually inserting a plugin into the discovery cache surfaces its descriptors."""

    class FakePlugin(ProviderPlugin):
        name = "fake_external"

        def descriptors(self):
            return [
                ProviderDescriptor(
                    name="fake_provider",
                    llm_sync=DriverSpec("openai_driver.OpenAIDriver", {"api_key": "openai_api_key"}, "gpt-4o-mini"),
                    display_name="Fake",
                )
            ]

    saved = dict(_discovery._DISCOVERED_PLUGINS)
    saved_b = _discovery._BUILTINS_LOADED
    saved_e = _discovery._EXTERNALS_LOADED
    try:
        _discovery._DISCOVERED_PLUGINS.clear()
        _discovery._BUILTINS_LOADED = False
        _discovery._EXTERNALS_LOADED = True
        _discovery._load_builtins()
        _discovery._DISCOVERED_PLUGINS["fake_external"] = FakePlugin()

        descs = load_plugins()
        names = {d.name for d in descs}
        assert "fake_provider" in names
    finally:
        _discovery._DISCOVERED_PLUGINS.clear()
        _discovery._DISCOVERED_PLUGINS.update(saved)
        _discovery._BUILTINS_LOADED = saved_b
        _discovery._EXTERNALS_LOADED = saved_e


def test_external_entry_point_loaded_via_mock():
    """Simulate an entry point producing a ProviderPlugin via importlib.metadata."""

    class EpPlugin(ProviderPlugin):
        name = "ep_plugin"

        def descriptors(self):
            return [ProviderDescriptor(name="ep_provider", display_name="EP")]

    fake_ep = MagicMock()
    fake_ep.name = "ep_plugin"
    fake_ep.load.return_value = EpPlugin

    saved = dict(_discovery._DISCOVERED_PLUGINS)
    saved_b = _discovery._BUILTINS_LOADED
    saved_e = _discovery._EXTERNALS_LOADED
    try:
        _discovery._DISCOVERED_PLUGINS.clear()
        _discovery._BUILTINS_LOADED = False
        _discovery._EXTERNALS_LOADED = False

        with patch("prompture.plugins.discovery.entry_points", return_value=[fake_ep]):
            plugins = discover_plugins()

        plugin_names = {p.name for p in plugins}
        assert "ep_plugin" in plugin_names

        descs = load_plugins()
        names = {d.name for d in descs}
        assert "ep_provider" in names
    finally:
        _discovery._DISCOVERED_PLUGINS.clear()
        _discovery._DISCOVERED_PLUGINS.update(saved)
        _discovery._BUILTINS_LOADED = saved_b
        _discovery._EXTERNALS_LOADED = saved_e


def test_entry_point_load_failure_is_logged():
    """A broken entry point logs an error and does not crash discovery."""
    broken_ep = MagicMock()
    broken_ep.name = "broken"
    broken_ep.load.side_effect = ImportError("missing dep")

    saved = dict(_discovery._DISCOVERED_PLUGINS)
    saved_b = _discovery._BUILTINS_LOADED
    saved_e = _discovery._EXTERNALS_LOADED
    try:
        _discovery._DISCOVERED_PLUGINS.clear()
        _discovery._BUILTINS_LOADED = False
        _discovery._EXTERNALS_LOADED = False

        with patch("prompture.plugins.discovery.entry_points", return_value=[broken_ep]):
            plugins = discover_plugins()
        plugin_names = {p.name for p in plugins}
        # Built-ins still load
        assert "openai_builtin" in plugin_names
        # Broken plugin not present
        assert "broken" not in plugin_names
    finally:
        _discovery._DISCOVERED_PLUGINS.clear()
        _discovery._DISCOVERED_PLUGINS.update(saved)
        _discovery._BUILTINS_LOADED = saved_b
        _discovery._EXTERNALS_LOADED = saved_e


def test_non_plugin_entry_point_is_skipped():
    """An entry point that returns a non-ProviderPlugin value is skipped."""

    class NotAPlugin:
        pass

    fake_ep = MagicMock()
    fake_ep.name = "not_a_plugin"
    fake_ep.load.return_value = NotAPlugin

    saved = dict(_discovery._DISCOVERED_PLUGINS)
    saved_b = _discovery._BUILTINS_LOADED
    saved_e = _discovery._EXTERNALS_LOADED
    try:
        _discovery._DISCOVERED_PLUGINS.clear()
        _discovery._BUILTINS_LOADED = False
        _discovery._EXTERNALS_LOADED = False

        with patch("prompture.plugins.discovery.entry_points", return_value=[fake_ep]):
            discover_plugins()

        assert "not_a_plugin" not in _discovery._DISCOVERED_PLUGINS
    finally:
        _discovery._DISCOVERED_PLUGINS.clear()
        _discovery._DISCOVERED_PLUGINS.update(saved)
        _discovery._BUILTINS_LOADED = saved_b
        _discovery._EXTERNALS_LOADED = saved_e


def test_duplicate_plugin_name_external_overrides_builtin(caplog):
    """An external plugin with a duplicate name overrides the built-in and emits a warning."""

    class OverridingPlugin(ProviderPlugin):
        name = "openai_builtin"  # collides with built-in

        def descriptors(self):
            return []

    fake_ep = MagicMock()
    fake_ep.name = "openai_builtin"
    fake_ep.load.return_value = OverridingPlugin

    saved = dict(_discovery._DISCOVERED_PLUGINS)
    saved_b = _discovery._BUILTINS_LOADED
    saved_e = _discovery._EXTERNALS_LOADED
    try:
        _discovery._DISCOVERED_PLUGINS.clear()
        _discovery._BUILTINS_LOADED = False
        _discovery._EXTERNALS_LOADED = False

        with patch("prompture.plugins.discovery.entry_points", return_value=[fake_ep]):
            with caplog.at_level("WARNING", logger="prompture.plugins.discovery"):
                discover_plugins()

        # The override took effect
        assert isinstance(_discovery._DISCOVERED_PLUGINS["openai_builtin"], OverridingPlugin)
        # And a warning was logged
        assert any("already registered" in rec.message for rec in caplog.records)
    finally:
        _discovery._DISCOVERED_PLUGINS.clear()
        _discovery._DISCOVERED_PLUGINS.update(saved)
        _discovery._BUILTINS_LOADED = saved_b
        _discovery._EXTERNALS_LOADED = saved_e


# ── Discovery integration ────────────────────────────────────────────────


def test_infra_discovery_still_works():
    """The infra.discovery module iterates PROVIDER_DESCRIPTORS — verify it doesn't crash."""
    from prompture.infra.discovery import get_available_models

    # Should not raise; result is an iterable (list or dict, depending on configured providers).
    result = get_available_models()
    assert result is not None
    iter(result)
