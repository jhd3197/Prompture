import os
from unittest.mock import patch

from prompture.drivers.ollama_driver import OllamaDriver
from prompture.infra.discovery import get_available_models


class TestDiscovery:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    def test_get_available_models_openai(self):
        """Test that OpenAI models are detected when API key is present."""
        models = get_available_models()
        # Check for some known OpenAI models
        assert "openai/gpt-4o" in models
        assert "openai/gpt-3.5-turbo" in models

    @patch.dict(os.environ, {}, clear=True)
    def test_get_available_models_no_keys(self):
        """Test that no static models are returned when no keys are present (except maybe Ollama/LMStudio if configured)."""
        # Ensure no env vars are set
        with (
            patch("prompture.infra.settings.settings.openai_api_key", None),
            patch("prompture.infra.settings.settings.claude_api_key", None),
            patch("prompture.infra.settings.settings.google_api_key", None),
        ):
            models = get_available_models()
            # Should not contain openai models
            assert not any(m.startswith("openai/") for m in models)

    @patch.object(OllamaDriver, "list_models", return_value=["llama3:latest", "mistral:latest"])
    def test_get_available_models_ollama(self, mock_list):
        """Test Ollama dynamic discovery via list_models()."""
        models = get_available_models()

        assert "ollama/llama3:latest" in models
        assert "ollama/mistral:latest" in models

    @patch.object(OllamaDriver, "list_models", return_value=None)
    def test_get_available_models_ollama_fail(self, mock_list):
        """Test Ollama discovery failure handling."""
        models = get_available_models()
        # Should not crash, just not include ollama models (unless some other provider is active)
        assert not any(m.startswith("ollama/") for m in models)
