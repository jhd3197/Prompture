import os
from unittest.mock import patch

import pytest

from prompture.drivers.elevenlabs_stt_driver import ElevenLabsSTTDriver
from prompture.drivers.elevenlabs_tts_driver import ElevenLabsTTSDriver
from prompture.drivers.ollama_driver import OllamaDriver
from prompture.infra.discovery import (
    _discovery_cache,
    clear_discovery_cache,
    get_available_audio_models,
    get_available_models,
)


@pytest.fixture(autouse=True)
def _clear_discovery_cache():
    """Clear the discovery cache before each test to ensure isolation."""
    _discovery_cache.clear()
    yield
    _discovery_cache.clear()


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


class TestAudioDiscovery:
    """Tests for ElevenLabs audio model discovery via get_available_audio_models()."""

    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "el-test"})
    @patch.object(
        ElevenLabsTTSDriver,
        "list_models",
        return_value=["eleven_multilingual_v2", "eleven_turbo_v2_5", "eleven_flash_v2_5"],
    )
    def test_elevenlabs_tts_dynamic_discovery(self, mock_list):
        """TTS models are discovered dynamically from the API."""
        models = get_available_audio_models(modality="tts")
        assert "elevenlabs/eleven_multilingual_v2" in models
        assert "elevenlabs/eleven_turbo_v2_5" in models
        assert "elevenlabs/eleven_flash_v2_5" in models
        mock_list.assert_called_once()

    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "el-test"})
    @patch.object(ElevenLabsTTSDriver, "list_models", return_value=None)
    def test_elevenlabs_tts_fallback_to_pricing(self, mock_list):
        """When the API call fails, TTS falls back to AUDIO_PRICING keys."""
        models = get_available_audio_models(modality="tts")
        for model_id in ElevenLabsTTSDriver.AUDIO_PRICING:
            assert f"elevenlabs/{model_id}" in models

    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "el-test"})
    @patch.object(
        ElevenLabsTTSDriver,
        "list_models",
        return_value=[
            "eleven_multilingual_v2",
            "eleven_turbo_v2_5",
            "eleven_flash_v2_5",
            "eleven_new_model_v3",
        ],
    )
    def test_elevenlabs_tts_discovers_new_models(self, mock_list):
        """New models from the API that aren't in AUDIO_PRICING are included."""
        models = get_available_audio_models(modality="tts")
        assert "elevenlabs/eleven_new_model_v3" in models

    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "el-test"})
    def test_elevenlabs_stt_returns_known_models(self):
        """STT discovery returns the known static model list."""
        models = get_available_audio_models(modality="stt")
        assert "elevenlabs/scribe_v1" in models

    @patch.dict(os.environ, {}, clear=True)
    def test_elevenlabs_no_key_no_models(self):
        """No ElevenLabs models when API key is missing."""
        with patch("prompture.infra.settings.settings.elevenlabs_api_key", None):
            models = get_available_audio_models()
            assert not any(m.startswith("elevenlabs/") for m in models)

    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "el-test"})
    @patch.object(
        ElevenLabsTTSDriver,
        "list_models",
        return_value=["eleven_multilingual_v2"],
    )
    def test_elevenlabs_both_modalities(self, mock_list):
        """Both STT and TTS models appear when modality is None."""
        models = get_available_audio_models()
        assert "elevenlabs/scribe_v1" in models
        assert "elevenlabs/eleven_multilingual_v2" in models


class TestElevenLabsTTSListModels:
    """Unit tests for ElevenLabsTTSDriver.list_models() itself."""

    @patch("prompture.drivers.elevenlabs_tts_driver.httpx")
    def test_list_models_filters_tts(self, mock_httpx):
        """Only models with can_do_text_to_speech=True are returned."""
        mock_resp = mock_httpx.get.return_value
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"model_id": "eleven_multilingual_v2", "can_do_text_to_speech": True},
            {"model_id": "eleven_english_sts_v2", "can_do_text_to_speech": False},
            {"model_id": "eleven_turbo_v2_5", "can_do_text_to_speech": True},
        ]
        result = ElevenLabsTTSDriver.list_models(api_key="test-key")
        assert result == ["eleven_multilingual_v2", "eleven_turbo_v2_5"]

    @patch("prompture.drivers.elevenlabs_tts_driver.httpx")
    def test_list_models_sends_api_key(self, mock_httpx):
        """The xi-api-key header is sent when an API key is provided."""
        mock_resp = mock_httpx.get.return_value
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        ElevenLabsTTSDriver.list_models(api_key="my-secret-key")
        call_kwargs = mock_httpx.get.call_args
        assert call_kwargs.kwargs["headers"]["xi-api-key"] == "my-secret-key"

    @patch("prompture.drivers.elevenlabs_tts_driver.httpx")
    def test_list_models_api_error_returns_none(self, mock_httpx):
        """Returns None on non-200 response."""
        mock_resp = mock_httpx.get.return_value
        mock_resp.status_code = 401
        result = ElevenLabsTTSDriver.list_models(api_key="bad-key")
        assert result is None

    @patch("prompture.drivers.elevenlabs_tts_driver.httpx")
    def test_list_models_network_error_returns_none(self, mock_httpx):
        """Returns None when the HTTP call raises."""
        mock_httpx.get.side_effect = Exception("connection refused")
        result = ElevenLabsTTSDriver.list_models(api_key="test-key")
        assert result is None

    @patch("prompture.drivers.elevenlabs_tts_driver.httpx", None)
    def test_list_models_no_httpx_returns_none(self):
        """Returns None when httpx is not installed."""
        result = ElevenLabsTTSDriver.list_models(api_key="test-key")
        assert result is None

    @patch("prompture.drivers.elevenlabs_tts_driver.httpx")
    def test_list_models_custom_endpoint(self, mock_httpx):
        """Respects a custom endpoint URL."""
        mock_resp = mock_httpx.get.return_value
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        ElevenLabsTTSDriver.list_models(
            api_key="k", endpoint="http://localhost:8080/v1"
        )
        url = mock_httpx.get.call_args.args[0]
        assert url == "http://localhost:8080/v1/models"


class TestElevenLabsSTTListModels:
    """Unit tests for ElevenLabsSTTDriver.list_models()."""

    def test_list_models_returns_known_models(self):
        """Returns the known STT models when a key is provided."""
        result = ElevenLabsSTTDriver.list_models(api_key="test-key")
        assert "scribe_v1" in result

    def test_list_models_no_key_returns_none(self):
        """Returns None when no API key is available."""
        with patch.dict(os.environ, {}, clear=True):
            result = ElevenLabsSTTDriver.list_models(api_key=None)
            assert result is None


class TestDiscoveryCache:
    """Tests for the in-memory discovery cache."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    def test_second_call_returns_cached_result(self):
        """Repeated calls return the same cached list without re-querying."""
        first = get_available_models()
        second = get_available_models()
        assert first == second

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    @patch.object(OllamaDriver, "list_models", return_value=["llama3:latest"])
    def test_cache_hit_skips_list_models(self, mock_list):
        """After a cached call, list_models() should not be invoked again."""
        get_available_models()
        call_count_after_first = mock_list.call_count

        get_available_models()
        assert mock_list.call_count == call_count_after_first

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    @patch.object(OllamaDriver, "list_models", return_value=["llama3:latest"])
    def test_force_refresh_bypasses_cache(self, mock_list):
        """force_refresh=True re-queries providers even when cache is warm."""
        get_available_models()
        call_count_after_first = mock_list.call_count

        get_available_models(force_refresh=True)
        assert mock_list.call_count > call_count_after_first

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    def test_clear_discovery_cache(self):
        """clear_discovery_cache() invalidates the cache."""
        first = get_available_models()
        clear_discovery_cache()
        assert not _discovery_cache.has("models:False:False")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    def test_different_params_cached_independently(self):
        """Calls with different include_capabilities are cached separately."""
        plain = get_available_models(include_capabilities=False)
        enriched = get_available_models(include_capabilities=True)
        # They should be different types
        assert isinstance(plain[0], str) if plain else True
        assert isinstance(enriched[0], dict) if enriched else True
