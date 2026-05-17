"""Tests for Google Vertex AI driver registration, import, and discovery."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestVertexAIImport:
    """Verify the driver classes import correctly."""

    def test_sync_driver_import(self):
        from prompture.drivers.google_vertexai_driver import GoogleVertexAIDriver

        assert GoogleVertexAIDriver.supports_json_mode is True
        assert GoogleVertexAIDriver.supports_json_schema is True
        assert GoogleVertexAIDriver.supports_vision is True
        assert GoogleVertexAIDriver.supports_tool_use is True
        assert GoogleVertexAIDriver.supports_streaming is True
        assert GoogleVertexAIDriver.supports_messages is True

    def test_async_driver_import(self):
        from prompture.drivers.async_google_vertexai_driver import (
            AsyncGoogleVertexAIDriver,
        )

        assert AsyncGoogleVertexAIDriver.supports_json_mode is True
        assert AsyncGoogleVertexAIDriver.supports_tool_use is True
        assert AsyncGoogleVertexAIDriver.supports_streaming is True

    def test_async_shares_model_pricing(self):
        from prompture.drivers.async_google_vertexai_driver import (
            AsyncGoogleVertexAIDriver,
        )
        from prompture.drivers.google_vertexai_driver import GoogleVertexAIDriver

        assert AsyncGoogleVertexAIDriver.MODEL_PRICING is GoogleVertexAIDriver.MODEL_PRICING


class TestVertexAIRegistration:
    """Verify Vertex AI is registered in the driver system."""

    def test_registered_in_descriptor_map(self):
        from prompture.drivers.provider_descriptors import PROVIDER_DESCRIPTOR_MAP

        assert "google_vertexai" in PROVIDER_DESCRIPTOR_MAP
        desc = PROVIDER_DESCRIPTOR_MAP["google_vertexai"]
        assert desc.display_name == "Google Vertex AI"
        assert desc.is_configured_fn is not None
        assert desc.llm_sync is not None
        assert desc.llm_async is not None

    def test_alias_vertex(self):
        from prompture.drivers.provider_descriptors import PROVIDER_DESCRIPTOR_MAP

        assert "vertex" in PROVIDER_DESCRIPTOR_MAP
        assert PROVIDER_DESCRIPTOR_MAP["vertex"].alias_for == "google_vertexai"

    def test_alias_vertexai(self):
        from prompture.drivers.provider_descriptors import PROVIDER_DESCRIPTOR_MAP

        assert "vertexai" in PROVIDER_DESCRIPTOR_MAP
        assert PROVIDER_DESCRIPTOR_MAP["vertexai"].alias_for == "google_vertexai"

    def test_registered_in_sync_registry(self):
        from prompture.drivers import DRIVER_REGISTRY

        assert "google_vertexai" in DRIVER_REGISTRY

    def test_registered_in_async_registry(self):
        from prompture.drivers.async_registry import ASYNC_DRIVER_REGISTRY

        assert "google_vertexai" in ASYNC_DRIVER_REGISTRY

    def test_descriptor_list_models_kwargs(self):
        from prompture.drivers.provider_descriptors import PROVIDER_DESCRIPTOR_MAP

        desc = PROVIDER_DESCRIPTOR_MAP["google_vertexai"]
        kwarg_names = [kw[0] for kw in desc.list_models_kwargs]
        assert "api_key" in kwarg_names
        assert "project_id" in kwarg_names
        assert "location" in kwarg_names

    def test_descriptor_models_dev_name(self):
        from prompture.drivers.provider_descriptors import PROVIDER_DESCRIPTOR_MAP

        desc = PROVIDER_DESCRIPTOR_MAP["google_vertexai"]
        # Vertex AI shares pricing with google on models.dev
        assert desc.models_dev_name == "google"


class TestVertexAISettings:
    """Verify settings fields exist."""

    def test_settings_fields(self):
        from prompture.infra.settings import Settings

        fields = Settings.model_fields
        assert "google_vertex_api_key" in fields
        assert "google_vertex_project_id" in fields
        assert "google_vertex_location" in fields
        assert "google_vertex_model" in fields

    def test_settings_defaults(self):
        from prompture.infra.settings import Settings

        s = Settings(
            _env_file=None,
            google_vertex_project_id=None,
            google_vertex_location="us-central1",
            google_vertex_model="gemini-2.0-flash",
        )
        assert s.google_vertex_location == "us-central1"
        assert s.google_vertex_model == "gemini-2.0-flash"


class TestVertexAIRateFile:
    """Verify the JSON rate file exists and has expected models."""

    def test_rate_file_exists(self):
        import json
        from pathlib import Path

        rate_path = Path(__file__).parent.parent / "prompture" / "infra" / "rates" / "google_vertexai.json"
        assert rate_path.exists()
        data = json.loads(rate_path.read_text())
        assert "gemini-2.0-flash" in data
        # Vertex AI also has Claude models
        assert "claude-sonnet-4-20250514" in data

    def test_rate_entries_have_required_keys(self):
        import json
        from pathlib import Path

        rate_path = Path(__file__).parent.parent / "prompture" / "infra" / "rates" / "google_vertexai.json"
        data = json.loads(rate_path.read_text())
        required = {"supports_temperature", "supports_tool_use", "tokens_param"}
        for model_id, entry in data.items():
            for key in required:
                assert key in entry, f"{model_id} missing {key}"


class TestVertexAIDriverInit:
    """Test driver constructor validation (no live API calls)."""

    def test_missing_credentials_raises(self):
        from prompture.drivers.google_vertexai_driver import GoogleVertexAIDriver

        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("GOOGLE_VERTEX_API_KEY", None)
            env.pop("GOOGLE_VERTEX_PROJECT_ID", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="credentials"):
                    GoogleVertexAIDriver(api_key=None, project_id=None)

    def test_missing_genai_raises(self):
        import prompture.drivers.google_vertexai_driver as mod

        original = mod.genai
        try:
            mod.genai = None
            with pytest.raises(RuntimeError, match="google-genai"):
                mod.GoogleVertexAIDriver(project_id="test-project")
        finally:
            mod.genai = original


class TestVertexAIListModels:
    """Test list_models class method with mocked client."""

    def test_list_models_no_credentials(self):
        from prompture.drivers.google_vertexai_driver import GoogleVertexAIDriver

        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("GOOGLE_VERTEX_API_KEY", None)
            env.pop("GOOGLE_VERTEX_PROJECT_ID", None)
            with patch.dict(os.environ, env, clear=True):
                result = GoogleVertexAIDriver.list_models(api_key=None, project_id=None)
                assert result is None

    def test_list_models_returns_ids(self):
        from prompture.drivers.google_vertexai_driver import GoogleVertexAIDriver

        mock_model_1 = MagicMock()
        mock_model_1.name = "publishers/google/models/gemini-2.0-flash"
        mock_model_2 = MagicMock()
        mock_model_2.name = "publishers/anthropic/models/claude-sonnet-4-20250514"

        mock_client = MagicMock()
        mock_client.models.list.return_value = [mock_model_1, mock_model_2]

        with patch("google.genai.Client", return_value=mock_client):
            result = GoogleVertexAIDriver.list_models(api_key="test-key")

        assert result is not None
        assert "gemini-2.0-flash" in result
        assert "claude-sonnet-4-20250514" in result

    def test_list_models_handles_gemini_error_gracefully(self):
        from prompture.drivers.google_vertexai_driver import GoogleVertexAIDriver

        with patch.dict(os.environ, {"GOOGLE_VERTEX_PROJECT_ID": ""}, clear=False):
            with patch("google.genai.Client", side_effect=Exception("Auth failed")):
                # Gemini listing fails, no project_id → no Claude models
                result = GoogleVertexAIDriver.list_models(api_key="test-key", project_id=None)
                assert result is None

    def test_list_models_includes_claude_with_project(self):
        from prompture.drivers.google_vertexai_driver import GoogleVertexAIDriver

        mock_client = MagicMock()
        mock_client.models.list.return_value = []

        with patch("google.genai.Client", return_value=mock_client):
            result = GoogleVertexAIDriver.list_models(project_id="test-proj")

        assert result is not None
        assert any("claude" in m for m in result)


@pytest.mark.integration
class TestVertexAIIntegration:
    """Integration tests that hit the real Vertex AI API.

    Require GOOGLE_VERTEX_PROJECT_ID to be set and valid ADC credentials.
    """

    @pytest.fixture(autouse=True)
    def _require_credentials(self):
        if not os.getenv("GOOGLE_VERTEX_API_KEY") and not os.getenv("GOOGLE_VERTEX_PROJECT_ID"):
            pytest.skip("GOOGLE_VERTEX_API_KEY or GOOGLE_VERTEX_PROJECT_ID not set")

    def test_simple_generation(self):
        from prompture.drivers.google_vertexai_driver import GoogleVertexAIDriver

        driver = GoogleVertexAIDriver()
        result = driver.generate('Return the JSON: {"hello": "world"}')
        assert "text" in result
        assert "meta" in result
        assert result["meta"]["prompt_tokens"] > 0

    def test_list_available_models(self):
        from prompture.drivers.google_vertexai_driver import GoogleVertexAIDriver

        models = GoogleVertexAIDriver.list_models()
        assert models is not None
        assert len(models) > 0
        print(f"\nVertex AI models ({len(models)}):")
        for m in sorted(models):
            print(f"  - {m}")

    def test_json_mode(self):
        from prompture.drivers.google_vertexai_driver import GoogleVertexAIDriver

        driver = GoogleVertexAIDriver()
        result = driver.generate(
            'Extract: name is Alice, age is 30. Return JSON with "name" and "age" keys.',
            options={"json_mode": True},
        )
        assert "text" in result
        import json

        data = json.loads(result["text"])
        assert "name" in data or "Alice" in result["text"]

    def test_messages_generation(self):
        from prompture.drivers.google_vertexai_driver import GoogleVertexAIDriver

        driver = GoogleVertexAIDriver()
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Respond in JSON."},
            {"role": "user", "content": 'Return: {"status": "ok"}'},
        ]
        result = driver.generate_messages(messages, {"json_mode": True})
        assert "text" in result

    def test_discovery_includes_vertex(self):
        from prompture.infra.discovery import get_available_models

        models = get_available_models()
        vertex_models = [m for m in models if m.startswith("google_vertexai/")]
        print(f"\nDiscovered {len(vertex_models)} Vertex AI models")
        for m in sorted(vertex_models)[:10]:
            print(f"  - {m}")
        assert len(vertex_models) > 0
