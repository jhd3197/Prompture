"""Tests for per-driver list_models() classmethods and the shared helper."""

import os
from unittest.mock import MagicMock, patch

from prompture.drivers.base import Driver, _fetch_openai_compatible_models

# ---------------------------------------------------------------------------
# Shared helper: _fetch_openai_compatible_models
# ---------------------------------------------------------------------------


class TestFetchOpenAICompatibleModels:
    @patch("prompture.drivers.base.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"id": "gpt-4o"},
                {"id": "gpt-4o-mini"},
            ]
        }
        mock_get.return_value = mock_resp

        result = _fetch_openai_compatible_models("https://api.example.com/v1", api_key="sk-test")
        assert result == ["gpt-4o", "gpt-4o-mini"]
        mock_get.assert_called_once()

        # Verify URL and headers
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://api.example.com/v1/models"
        assert call_args[1]["headers"]["Authorization"] == "Bearer sk-test"

    @patch("prompture.drivers.base.requests.get")
    def test_non_200_returns_none(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        result = _fetch_openai_compatible_models("https://api.example.com/v1", api_key="bad-key")
        assert result is None

    @patch("prompture.drivers.base.requests.get")
    def test_connection_error_returns_none(self, mock_get):
        mock_get.side_effect = ConnectionError("refused")

        result = _fetch_openai_compatible_models("https://api.example.com/v1", api_key="sk-test")
        assert result is None

    @patch("prompture.drivers.base.requests.get")
    def test_no_api_key(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "model-1"}]}
        mock_get.return_value = mock_resp

        result = _fetch_openai_compatible_models("http://localhost:1234/v1")
        assert result == ["model-1"]

        # Verify no Authorization header
        call_args = mock_get.call_args
        assert "Authorization" not in call_args[1]["headers"]

    @patch("prompture.drivers.base.requests.get")
    def test_extra_headers(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "model-1"}]}
        mock_get.return_value = mock_resp

        result = _fetch_openai_compatible_models(
            "https://api.example.com/v1",
            api_key="sk-test",
            headers={"X-Custom": "value"},
        )
        assert result == ["model-1"]
        call_args = mock_get.call_args
        assert call_args[1]["headers"]["X-Custom"] == "value"
        assert call_args[1]["headers"]["Authorization"] == "Bearer sk-test"

    @patch("prompture.drivers.base.requests.get")
    def test_skips_entries_without_id(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"id": "model-a"},
                {"name": "no-id-field"},  # missing "id"
                {"id": ""},  # empty id
                {"id": "model-b"},
            ]
        }
        mock_get.return_value = mock_resp

        result = _fetch_openai_compatible_models("https://api.example.com/v1")
        assert result == ["model-a", "model-b"]

    @patch("prompture.drivers.base.requests.get")
    def test_trailing_slash_stripped(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "m"}]}
        mock_get.return_value = mock_resp

        _fetch_openai_compatible_models("https://api.example.com/v1/")
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://api.example.com/v1/models"


# ---------------------------------------------------------------------------
# Driver base class
# ---------------------------------------------------------------------------


class TestDriverBaseListModels:
    def test_default_returns_none(self):
        assert Driver.list_models() is None


# ---------------------------------------------------------------------------
# OpenAI driver
# ---------------------------------------------------------------------------


class TestOpenAIDriverListModels:
    @patch("prompture.drivers.base.requests.get")
    def test_with_api_key(self, mock_get):
        from prompture.drivers.openai_driver import OpenAIDriver

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}
        mock_get.return_value = mock_resp

        result = OpenAIDriver.list_models(api_key="sk-test")
        assert "gpt-4o" in result
        assert "gpt-4o-mini" in result

    def test_no_key_returns_none(self):
        from prompture.drivers.openai_driver import OpenAIDriver

        with patch.dict(os.environ, {}, clear=True):
            with patch("prompture.infra.settings.settings.openai_api_key", None):
                result = OpenAIDriver.list_models()
                assert result is None


# ---------------------------------------------------------------------------
# Groq driver
# ---------------------------------------------------------------------------


class TestGroqDriverListModels:
    @patch("prompture.drivers.base.requests.get")
    def test_with_api_key(self, mock_get):
        from prompture.drivers.groq_driver import GroqDriver

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "llama-3.3-70b"}, {"id": "mixtral-8x7b"}]}
        mock_get.return_value = mock_resp

        result = GroqDriver.list_models(api_key="gsk-test")
        assert "llama-3.3-70b" in result

    def test_no_key_returns_none(self):
        from prompture.drivers.groq_driver import GroqDriver

        with patch.dict(os.environ, {}, clear=True):
            result = GroqDriver.list_models()
            assert result is None


# ---------------------------------------------------------------------------
# Grok driver
# ---------------------------------------------------------------------------


class TestGrokDriverListModels:
    @patch("prompture.drivers.base.requests.get")
    def test_with_api_key(self, mock_get):
        from prompture.drivers.grok_driver import GrokDriver

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "grok-3"}, {"id": "grok-3-mini"}]}
        mock_get.return_value = mock_resp

        result = GrokDriver.list_models(api_key="xai-test")
        assert "grok-3" in result

    def test_no_key_returns_none(self):
        from prompture.drivers.grok_driver import GrokDriver

        with patch.dict(os.environ, {}, clear=True):
            result = GrokDriver.list_models()
            assert result is None


# ---------------------------------------------------------------------------
# OpenRouter driver
# ---------------------------------------------------------------------------


class TestOpenRouterDriverListModels:
    @patch("prompture.drivers.base.requests.get")
    def test_with_api_key(self, mock_get):
        from prompture.drivers.openrouter_driver import OpenRouterDriver

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "openai/gpt-4o"}, {"id": "anthropic/claude-sonnet-4"}]}
        mock_get.return_value = mock_resp

        result = OpenRouterDriver.list_models(api_key="sk-or-test")
        assert "openai/gpt-4o" in result

        # Verify OpenRouter-specific headers
        call_args = mock_get.call_args
        assert "HTTP-Referer" in call_args[1]["headers"]

    def test_no_key_returns_none(self):
        from prompture.drivers.openrouter_driver import OpenRouterDriver

        with patch.dict(os.environ, {}, clear=True):
            result = OpenRouterDriver.list_models()
            assert result is None


# ---------------------------------------------------------------------------
# Moonshot driver
# ---------------------------------------------------------------------------


class TestMoonshotDriverListModels:
    @patch("prompture.drivers.base.requests.get")
    def test_with_api_key(self, mock_get):
        from prompture.drivers.moonshot_driver import MoonshotDriver

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "kimi-k2-0905-preview"}]}
        mock_get.return_value = mock_resp

        result = MoonshotDriver.list_models(api_key="ms-test")
        assert "kimi-k2-0905-preview" in result

    @patch("prompture.drivers.base.requests.get")
    def test_custom_endpoint(self, mock_get):
        from prompture.drivers.moonshot_driver import MoonshotDriver

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "kimi-k2"}]}
        mock_get.return_value = mock_resp

        result = MoonshotDriver.list_models(api_key="ms-test", endpoint="https://api.moonshot.cn/v1")
        assert result == ["kimi-k2"]

        call_args = mock_get.call_args
        assert "moonshot.cn" in call_args[0][0]

    def test_no_key_returns_none(self):
        from prompture.drivers.moonshot_driver import MoonshotDriver

        with patch.dict(os.environ, {}, clear=True):
            result = MoonshotDriver.list_models()
            assert result is None


# ---------------------------------------------------------------------------
# Claude driver
# ---------------------------------------------------------------------------


class TestClaudeDriverListModels:
    @patch("prompture.drivers.claude_driver.requests.get")
    def test_with_api_key(self, mock_get):
        from prompture.drivers.claude_driver import ClaudeDriver

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"id": "claude-sonnet-4-20250514"}, {"id": "claude-3-5-haiku-20241022"}]
        }
        mock_get.return_value = mock_resp

        result = ClaudeDriver.list_models(api_key="sk-ant-test")
        assert "claude-sonnet-4-20250514" in result

        # Verify Anthropic-specific headers
        call_args = mock_get.call_args
        assert call_args[1]["headers"]["x-api-key"] == "sk-ant-test"
        assert "anthropic-version" in call_args[1]["headers"]

    @patch("prompture.drivers.claude_driver.requests.get")
    def test_auth_failure(self, mock_get):
        from prompture.drivers.claude_driver import ClaudeDriver

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        result = ClaudeDriver.list_models(api_key="bad-key")
        assert result is None

    def test_no_key_returns_none(self):
        from prompture.drivers.claude_driver import ClaudeDriver

        with patch.dict(os.environ, {}, clear=True):
            result = ClaudeDriver.list_models()
            assert result is None


# ---------------------------------------------------------------------------
# Google driver
# ---------------------------------------------------------------------------


class TestGoogleDriverListModels:
    @patch("prompture.drivers.google_driver.genai")
    def test_with_api_key(self, mock_genai):
        from prompture.drivers.google_driver import GoogleDriver

        mock_model_1 = MagicMock()
        mock_model_1.name = "models/gemini-2.5-pro"
        mock_model_2 = MagicMock()
        mock_model_2.name = "models/gemini-2.0-flash"

        mock_client = MagicMock()
        mock_client.models.list.return_value = [mock_model_1, mock_model_2]
        mock_genai.Client.return_value = mock_client

        result = GoogleDriver.list_models(api_key="google-test")
        assert "gemini-2.5-pro" in result
        assert "gemini-2.0-flash" in result

    @patch("prompture.drivers.google_driver.genai")
    def test_sdk_failure(self, mock_genai):
        from prompture.drivers.google_driver import GoogleDriver

        mock_genai.Client.side_effect = Exception("SDK error")

        result = GoogleDriver.list_models(api_key="google-test")
        assert result is None

    def test_no_key_returns_none(self):
        from prompture.drivers.google_driver import GoogleDriver

        with patch.dict(os.environ, {}, clear=True):
            result = GoogleDriver.list_models()
            assert result is None


# ---------------------------------------------------------------------------
# Ollama driver
# ---------------------------------------------------------------------------


class TestOllamaDriverListModels:
    @patch("prompture.drivers.ollama_driver.requests.get")
    def test_success(self, mock_get):
        from prompture.drivers.ollama_driver import OllamaDriver

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "llama3:latest"}, {"name": "mistral:latest"}]}
        mock_get.return_value = mock_resp

        result = OllamaDriver.list_models()
        assert result == ["llama3:latest", "mistral:latest"]

    @patch("prompture.drivers.ollama_driver.requests.get")
    def test_custom_endpoint(self, mock_get):
        from prompture.drivers.ollama_driver import OllamaDriver

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "phi3"}]}
        mock_get.return_value = mock_resp

        result = OllamaDriver.list_models(endpoint="http://gpu-server:11434/api/generate")
        assert result == ["phi3"]

        call_args = mock_get.call_args
        assert call_args[0][0] == "http://gpu-server:11434/api/tags"

    @patch("prompture.drivers.ollama_driver.requests.get")
    def test_connection_error(self, mock_get):
        from prompture.drivers.ollama_driver import OllamaDriver

        mock_get.side_effect = ConnectionError("refused")
        result = OllamaDriver.list_models()
        assert result is None


# ---------------------------------------------------------------------------
# LMStudio driver
# ---------------------------------------------------------------------------


class TestLMStudioDriverListModels:
    @patch("prompture.drivers.base.requests.get")
    def test_success(self, mock_get):
        from prompture.drivers.lmstudio_driver import LMStudioDriver

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "deepseek-r1-8b"}, {"id": "qwen-7b"}]}
        mock_get.return_value = mock_resp

        result = LMStudioDriver.list_models()
        assert "deepseek-r1-8b" in result
        assert "qwen-7b" in result

    @patch("prompture.drivers.base.requests.get")
    def test_connection_error(self, mock_get):
        from prompture.drivers.lmstudio_driver import LMStudioDriver

        mock_get.side_effect = ConnectionError("refused")
        result = LMStudioDriver.list_models()
        assert result is None

    @patch("prompture.drivers.base.requests.get")
    def test_renamed_instance_method(self, mock_get):
        """Verify old instance list_models() is now get_loaded_models()."""
        from prompture.drivers.lmstudio_driver import LMStudioDriver

        # The instance method should be get_loaded_models
        assert hasattr(LMStudioDriver, "get_loaded_models")

        # The classmethod should exist and return list[str] | None
        assert isinstance(LMStudioDriver.__dict__["list_models"], classmethod)


# ---------------------------------------------------------------------------
# Discovery integration
# ---------------------------------------------------------------------------


class TestDiscoveryUsesListModels:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    @patch.object(
        __import__("prompture.drivers.openai_driver", fromlist=["OpenAIDriver"]).OpenAIDriver,
        "list_models",
        return_value=["gpt-4o", "gpt-4o-mini", "o1-preview"],
    )
    def test_api_models_included(self, mock_list):
        from prompture.infra.discovery import clear_discovery_cache, get_available_models

        clear_discovery_cache()
        models = get_available_models(force_refresh=True)
        # API-discovered models should be present
        assert "openai/o1-preview" in models

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    @patch.object(
        __import__("prompture.drivers.openai_driver", fromlist=["OpenAIDriver"]).OpenAIDriver,
        "list_models",
        return_value=None,
    )
    def test_api_failure_falls_back_to_static(self, mock_list):
        from prompture.infra.discovery import get_available_models

        models = get_available_models()
        # Static MODEL_PRICING models should still be present
        assert "openai/gpt-4o" in models
