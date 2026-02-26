import os
from unittest.mock import MagicMock, patch

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


class TestProvenanceTracking:
    """Tests for source provenance tracking in enriched discovery results."""

    def _mock_settings(self, mock_settings):
        """Configure mock_settings so only ollama is 'configured'."""
        mock_settings.openai_api_key = None
        mock_settings.azure_api_key = None
        mock_settings.claude_api_key = None
        mock_settings.google_api_key = None
        mock_settings.groq_api_key = None
        mock_settings.openrouter_api_key = None
        mock_settings.grok_api_key = None
        mock_settings.moonshot_api_key = None
        mock_settings.zhipu_api_key = None
        mock_settings.modelscope_api_key = None
        mock_settings.cachibot_api_key = None
        mock_settings.ollama_endpoint = "http://localhost:11434/api/generate"
        mock_settings.lmstudio_endpoint = None
        mock_settings.lmstudio_api_key = None

    @patch.object(OllamaDriver, "list_models", return_value=["llama3:latest"])
    def test_api_source_models(self, mock_list):
        """Models from list_models() are tagged source='api'."""
        mock_ledger = MagicMock()
        mock_ledger.get_verified_models.return_value = set()
        mock_ledger.get_all_stats.return_value = []

        with (
            patch("prompture.infra.ledger._get_ledger", return_value=mock_ledger),
            patch("prompture.infra.discovery.settings") as mock_settings,
            patch("prompture.infra.model_rates.PROVIDER_MAP", {}),
        ):
            self._mock_settings(mock_settings)
            enriched = get_available_models(include_capabilities=True)
            llama = [e for e in enriched if e["model"] == "ollama/llama3:latest"]
            assert len(llama) == 1
            assert llama[0]["source"] == "api"

    @patch.object(OllamaDriver, "list_models", return_value=None)
    def test_static_source_models(self, mock_list):
        """When list_models() returns None, MODEL_PRICING models are tagged source='static'."""
        mock_ledger = MagicMock()
        mock_ledger.get_verified_models.return_value = set()
        mock_ledger.get_all_stats.return_value = []

        with (
            patch("prompture.infra.ledger._get_ledger", return_value=mock_ledger),
            patch("prompture.infra.discovery.settings") as mock_settings,
            patch("prompture.infra.model_rates.PROVIDER_MAP", {}),
        ):
            self._mock_settings(mock_settings)
            enriched = get_available_models(include_capabilities=True)
            ollama_entries = [e for e in enriched if e["provider"] == "ollama"]
            # All should be static since list_models returned None
            for entry in ollama_entries:
                assert entry["source"] == "static", f"{entry['model']} should be static"

    def test_catalog_source_models(self):
        """Models from models.dev catalog (non-authoritative) are tagged source='catalog'."""
        mock_ledger = MagicMock()
        mock_ledger.get_verified_models.return_value = set()
        mock_ledger.get_all_stats.return_value = []

        # Set up: grok is configured, list_models fails, no MODEL_PRICING,
        # but models.dev has grok models
        with (
            patch("prompture.infra.ledger._get_ledger", return_value=mock_ledger),
            patch("prompture.infra.discovery.settings") as mock_settings,
            patch("prompture.infra.model_rates.PROVIDER_MAP", {"grok": "xai"}),
            patch(
                "prompture.infra.model_rates.get_all_provider_models",
                return_value=["grok-2"],
            ),
            patch(
                "prompture.drivers.grok_driver.GrokDriver.list_models",
                side_effect=Exception("no api"),
            ),
        ):
            self._mock_settings(mock_settings)
            mock_settings.grok_api_key = "xai-test"
            # Remove MODEL_PRICING so static path doesn't fire
            with patch.object(
                type(
                    __import__("prompture.drivers.grok_driver", fromlist=["GrokDriver"]).GrokDriver
                ),
                "MODEL_PRICING",
                {},
                create=True,
            ):
                enriched = get_available_models(include_capabilities=True)
                grok_entries = [e for e in enriched if e["model"] == "grok/grok-2"]
                assert len(grok_entries) == 1
                assert grok_entries[0]["source"] == "catalog"

    @patch.object(OllamaDriver, "list_models", return_value=["llama3:latest"])
    def test_source_not_in_plain_mode(self, mock_list):
        """Plain string mode still returns list[str], no source info."""
        models = get_available_models(include_capabilities=False)
        assert isinstance(models, list)
        if models:
            assert isinstance(models[0], str)

    @patch.object(OllamaDriver, "list_models", return_value=["llama3:latest"])
    def test_api_source_wins_over_catalog(self, mock_list):
        """When a model is found via API, catalog doesn't overwrite its source."""
        mock_ledger = MagicMock()
        mock_ledger.get_verified_models.return_value = set()
        mock_ledger.get_all_stats.return_value = []

        with (
            patch("prompture.infra.ledger._get_ledger", return_value=mock_ledger),
            patch("prompture.infra.discovery.settings") as mock_settings,
            # Set up PROVIDER_MAP so catalog also tries to add ollama models
            patch(
                "prompture.infra.model_rates.PROVIDER_MAP",
                {"ollama": "ollama"},
            ),
            patch(
                "prompture.infra.model_rates.get_all_provider_models",
                return_value=["llama3:latest", "mistral:latest"],
            ),
        ):
            self._mock_settings(mock_settings)
            enriched = get_available_models(include_capabilities=True)
            # llama3:latest came from API, catalog shouldn't overwrite
            llama = [e for e in enriched if e["model"] == "ollama/llama3:latest"]
            assert len(llama) == 1
            assert llama[0]["source"] == "api"
            # mistral:latest was only in catalog (not returned by list_models)
            # Actually, list_models returned ["llama3:latest"] and was authoritative,
            # so ollama is in api_authoritative_providers — catalog won't add models.
            # The catalog path skips api_authoritative providers, so mistral won't appear.


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
        get_available_models()
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
