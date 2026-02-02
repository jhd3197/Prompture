"""Unit tests for Azure multi-backend driver dispatch and client caching."""

from unittest.mock import MagicMock, patch

import pytest

from prompture.drivers.azure_config import (
    clear_azure_configs,
    register_azure_config,
    set_azure_config_resolver,
)
from prompture.drivers.azure_driver import AzureDriver


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the global config registry and resolver between tests."""
    clear_azure_configs()
    set_azure_config_resolver(None)
    yield
    clear_azure_configs()
    set_azure_config_resolver(None)


def _make_driver(**kwargs):
    """Create an AzureDriver with defaults that won't hit env vars."""
    defaults = {
        "api_key": "test-key",
        "endpoint": "https://test.openai.azure.com/",
        "deployment_id": "gpt-4o",
        "model": "gpt-4o",
    }
    defaults.update(kwargs)
    return AzureDriver(**defaults)


# ------------------------------------------------------------------
# Backend dispatch
# ------------------------------------------------------------------


class TestBackendDispatch:
    def test_openai_model_dispatches_to_openai_backend(self):
        driver = _make_driver(model="gpt-4o")

        with patch.object(driver, "_generate_openai", return_value={"text": "ok", "meta": {}}) as mock:
            driver._do_generate([{"role": "user", "content": "hi"}], {})
            mock.assert_called_once()

    def test_claude_model_dispatches_to_claude_backend(self):
        driver = _make_driver(model="claude-sonnet-4-20250514")

        with patch.object(driver, "_generate_claude", return_value={"text": "ok", "meta": {}}) as mock:
            driver._do_generate([{"role": "user", "content": "hi"}], {})
            mock.assert_called_once()

    def test_mistral_model_dispatches_to_openai_backend(self):
        """Mistral uses OpenAI-compatible protocol."""
        driver = _make_driver(model="mistral-large-latest")

        with patch.object(driver, "_generate_openai", return_value={"text": "ok", "meta": {}}) as mock:
            driver._do_generate([{"role": "user", "content": "hi"}], {})
            mock.assert_called_once()

    def test_tool_dispatch_openai(self):
        driver = _make_driver(model="gpt-4o")

        with patch.object(
            driver,
            "_generate_openai_with_tools",
            return_value={"text": "", "meta": {}, "tool_calls": [], "stop_reason": "stop"},
        ) as mock:
            driver.generate_messages_with_tools([{"role": "user", "content": "hi"}], [], {})
            mock.assert_called_once()

    def test_tool_dispatch_claude(self):
        driver = _make_driver(model="claude-sonnet-4-20250514")

        with patch.object(
            driver,
            "_generate_claude_with_tools",
            return_value={"text": "", "meta": {}, "tool_calls": [], "stop_reason": "end_turn"},
        ) as mock:
            driver.generate_messages_with_tools([{"role": "user", "content": "hi"}], [], {})
            mock.assert_called_once()


# ------------------------------------------------------------------
# Config resolution in driver
# ------------------------------------------------------------------


class TestDriverConfigResolution:
    def test_uses_default_config(self):
        driver = _make_driver()

        with patch.object(driver, "_generate_openai", return_value={"text": "ok", "meta": {}}) as mock:
            driver._do_generate([{"role": "user", "content": "hi"}], {})
            # Check config passed has our default endpoint
            call_args = mock.call_args
            config = call_args[0][2]  # third positional arg is config
            assert config["endpoint"] == "https://test.openai.azure.com/"

    def test_uses_registered_config(self):
        register_azure_config(
            "gpt-4o-mini",
            {
                "endpoint": "https://westus.azure.com/",
                "api_key": "west-key",
                "deployment_id": "gpt-4o-mini-west",
            },
        )

        driver = _make_driver(model="gpt-4o-mini")

        with patch.object(driver, "_generate_openai", return_value={"text": "ok", "meta": {}}) as mock:
            driver._do_generate([{"role": "user", "content": "hi"}], {})
            config = mock.call_args[0][2]
            assert config["endpoint"] == "https://westus.azure.com/"
            assert config["api_key"] == "west-key"

    def test_uses_per_call_override(self):
        driver = _make_driver()
        override = {
            "endpoint": "https://override.azure.com/",
            "api_key": "override-key",
            "deployment_id": "override-deploy",
        }

        with patch.object(driver, "_generate_openai", return_value={"text": "ok", "meta": {}}) as mock:
            driver._do_generate(
                [{"role": "user", "content": "hi"}],
                {"azure_config": override},
            )
            config = mock.call_args[0][2]
            assert config["endpoint"] == "https://override.azure.com/"

    def test_resolver_callback(self):
        set_azure_config_resolver(
            lambda m: {
                "endpoint": f"https://{m}.azure.com/",
                "api_key": f"key-{m}",
                "deployment_id": m,
            }
        )

        driver = _make_driver(model="gpt-4o")

        with patch.object(driver, "_generate_openai", return_value={"text": "ok", "meta": {}}) as mock:
            driver._do_generate([{"role": "user", "content": "hi"}], {})
            config = mock.call_args[0][2]
            assert config["endpoint"] == "https://gpt-4o.azure.com/"


# ------------------------------------------------------------------
# Client caching
# ------------------------------------------------------------------


class TestClientCaching:
    @patch("prompture.drivers.azure_driver.AzureOpenAI")
    def test_same_config_reuses_client(self, mock_azure_cls):
        mock_client = MagicMock()
        mock_azure_cls.return_value = mock_client

        driver = _make_driver()
        config = {"endpoint": "https://test.azure.com/", "api_key": "k", "api_version": "2024-02-15-preview"}

        client1 = driver._get_openai_client(config)
        client2 = driver._get_openai_client(config)

        assert client1 is client2
        assert mock_azure_cls.call_count == 1

    @patch("prompture.drivers.azure_driver.AzureOpenAI")
    def test_different_config_creates_new_client(self, mock_azure_cls):
        mock_azure_cls.return_value = MagicMock()

        driver = _make_driver()
        config1 = {"endpoint": "https://east.azure.com/", "api_key": "k1", "api_version": "2024-02-15-preview"}
        config2 = {"endpoint": "https://west.azure.com/", "api_key": "k2", "api_version": "2024-02-15-preview"}

        driver._get_openai_client(config1)
        driver._get_openai_client(config2)

        assert mock_azure_cls.call_count == 2


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------


class TestBackwardCompat:
    def test_constructor_without_model(self):
        """Old-style construction with just env var args still works."""
        driver = AzureDriver(
            api_key="key",
            endpoint="https://test.azure.com/",
            deployment_id="gpt-4o",
        )
        assert driver.model == "gpt-4o-mini"  # default
        assert driver._default_config["api_key"] == "key"
        assert driver._default_config["endpoint"] == "https://test.azure.com/"

    def test_no_eager_validation(self):
        """Constructor should NOT raise even with missing config."""
        driver = AzureDriver()  # no args, no env vars
        assert driver.model == "gpt-4o-mini"

    @patch.dict("os.environ", {}, clear=False)
    def test_deployment_id_defaults_to_model(self):
        """If no deployment_id in config, model name is used."""
        # Remove AZURE_DEPLOYMENT_ID from env so it doesn't interfere
        import os

        old_val = os.environ.pop("AZURE_DEPLOYMENT_ID", None)
        try:
            driver = AzureDriver(
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
                model="gpt-4o",
            )
            # deployment_id should be None in default config
            assert driver._default_config["deployment_id"] is None
        finally:
            if old_val is not None:
                os.environ["AZURE_DEPLOYMENT_ID"] = old_val
