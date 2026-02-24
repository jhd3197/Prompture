"""Unit tests for Azure per-model config resolution system."""

import pytest

from prompture.drivers.azure_config import (
    classify_backend,
    clear_azure_configs,
    register_azure_config,
    resolve_config,
    set_azure_config_resolver,
    unregister_azure_config,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the global config registry and resolver between tests."""
    clear_azure_configs()
    set_azure_config_resolver(None)
    yield
    clear_azure_configs()
    set_azure_config_resolver(None)


# ------------------------------------------------------------------
# classify_backend
# ------------------------------------------------------------------


class TestClassifyBackend:
    def test_gpt_models(self):
        assert classify_backend("gpt-4o") == "openai"
        assert classify_backend("gpt-4o-mini") == "openai"
        assert classify_backend("gpt-4-turbo") == "openai"
        assert classify_backend("gpt-5-mini") == "openai"

    def test_o_series_models(self):
        assert classify_backend("o1-preview") == "openai"
        assert classify_backend("o3-mini") == "openai"
        assert classify_backend("o4-mini") == "openai"

    def test_claude_models(self):
        assert classify_backend("claude-sonnet-4-20250514") == "claude"
        assert classify_backend("claude-haiku-4-5-20251001") == "claude"
        assert classify_backend("claude-sonnet-4-6") == "claude"

    def test_mistral_models(self):
        assert classify_backend("mistral-large-latest") == "mistral"
        assert classify_backend("mixtral-8x7b") == "mistral"

    def test_unknown_defaults_to_openai(self):
        assert classify_backend("some-custom-model") == "openai"
        assert classify_backend("phi-3") == "openai"

    def test_case_insensitive(self):
        assert classify_backend("GPT-4o") == "openai"
        assert classify_backend("Claude-Sonnet-4-20250514") == "claude"
        assert classify_backend("Mistral-Large-Latest") == "mistral"


# ------------------------------------------------------------------
# register_azure_config / resolve_config
# ------------------------------------------------------------------


class TestConfigRegistry:
    def test_register_and_resolve(self):
        config = {
            "endpoint": "https://eastus.openai.azure.com/",
            "api_key": "key-eastus",
            "deployment_id": "gpt-4o",
        }
        register_azure_config("gpt-4o", config)

        resolved = resolve_config("gpt-4o")
        assert resolved == config

    def test_unregister(self):
        register_azure_config("gpt-4o", {"endpoint": "x", "api_key": "y"})
        unregister_azure_config("gpt-4o")

        with pytest.raises(ValueError, match="No Azure config found"):
            resolve_config("gpt-4o")

    def test_clear_all(self):
        register_azure_config("gpt-4o", {"endpoint": "x", "api_key": "y"})
        register_azure_config("gpt-4o-mini", {"endpoint": "x2", "api_key": "y2"})
        clear_azure_configs()

        with pytest.raises(ValueError, match="No Azure config found"):
            resolve_config("gpt-4o")

    def test_multiple_models_different_configs(self):
        config_east = {"endpoint": "https://eastus.azure.com/", "api_key": "key-east"}
        config_west = {"endpoint": "https://westus.azure.com/", "api_key": "key-west"}

        register_azure_config("gpt-4o", config_east)
        register_azure_config("gpt-4o-mini", config_west)

        assert resolve_config("gpt-4o") == config_east
        assert resolve_config("gpt-4o-mini") == config_west


# ------------------------------------------------------------------
# resolve_config priority chain
# ------------------------------------------------------------------


class TestResolveConfigPriority:
    def test_override_takes_highest_priority(self):
        register_azure_config("gpt-4o", {"endpoint": "registry", "api_key": "r"})
        set_azure_config_resolver(lambda m: {"endpoint": "resolver", "api_key": "v"})
        default = {"endpoint": "default", "api_key": "d"}
        override = {"endpoint": "override", "api_key": "o"}

        resolved = resolve_config("gpt-4o", override=override, default_config=default)
        assert resolved["endpoint"] == "override"

    def test_resolver_over_registry(self):
        register_azure_config("gpt-4o", {"endpoint": "registry", "api_key": "r"})
        set_azure_config_resolver(lambda m: {"endpoint": "resolver", "api_key": "v"})

        resolved = resolve_config("gpt-4o")
        assert resolved["endpoint"] == "resolver"

    def test_registry_over_default(self):
        register_azure_config("gpt-4o", {"endpoint": "registry", "api_key": "r"})
        default = {"endpoint": "default", "api_key": "d"}

        resolved = resolve_config("gpt-4o", default_config=default)
        assert resolved["endpoint"] == "registry"

    def test_default_fallback(self):
        default = {"endpoint": "default", "api_key": "d"}

        resolved = resolve_config("gpt-4o", default_config=default)
        assert resolved["endpoint"] == "default"

    def test_no_config_raises(self):
        with pytest.raises(ValueError, match="No Azure config found for 'gpt-4o'"):
            resolve_config("gpt-4o")

    def test_empty_default_raises(self):
        """Default config with no endpoint/key should not be used."""
        default = {"endpoint": None, "api_key": None}
        with pytest.raises(ValueError, match="No Azure config found"):
            resolve_config("gpt-4o", default_config=default)

    def test_resolver_returns_none_falls_through(self):
        """If resolver returns None/empty, fall through to registry."""
        register_azure_config("gpt-4o", {"endpoint": "registry", "api_key": "r"})
        set_azure_config_resolver(lambda m: None)

        resolved = resolve_config("gpt-4o")
        assert resolved["endpoint"] == "registry"


# ------------------------------------------------------------------
# set_azure_config_resolver
# ------------------------------------------------------------------


class TestConfigResolver:
    def test_resolver_called_with_model_name(self):
        calls = []

        def resolver(model):
            calls.append(model)
            return {"endpoint": "resolved", "api_key": "k"}

        set_azure_config_resolver(resolver)
        resolve_config("gpt-4o")

        assert calls == ["gpt-4o"]

    def test_clear_resolver(self):
        set_azure_config_resolver(lambda m: {"endpoint": "x", "api_key": "y"})
        set_azure_config_resolver(None)

        with pytest.raises(ValueError, match="No Azure config found"):
            resolve_config("gpt-4o")
