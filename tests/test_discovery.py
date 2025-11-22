import os
import pytest
from unittest.mock import patch, MagicMock
from prompture.discovery import get_available_models
from prompture.drivers import OpenAIDriver, OllamaDriver

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
        with patch("prompture.settings.settings.openai_api_key", None), \
             patch("prompture.settings.settings.claude_api_key", None), \
             patch("prompture.settings.settings.google_api_key", None):
            
            models = get_available_models()
            # Should not contain openai models
            assert not any(m.startswith("openai/") for m in models)
            
    @patch("requests.get")
    def test_get_available_models_ollama(self, mock_get):
        """Test Ollama dynamic discovery."""
        # Mock successful Ollama response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3:latest"},
                {"name": "mistral:latest"}
            ]
        }
        mock_get.return_value = mock_response
        
        models = get_available_models()
        
        assert "ollama/llama3:latest" in models
        assert "ollama/mistral:latest" in models

    @patch("requests.get")
    def test_get_available_models_ollama_fail(self, mock_get):
        """Test Ollama discovery failure handling."""
        mock_get.side_effect = Exception("Connection refused")
        
        models = get_available_models()
        # Should not crash, just not include ollama models (unless some other provider is active)
        assert not any(m.startswith("ollama/") for m in models)
