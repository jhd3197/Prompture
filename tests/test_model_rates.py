"""Tests for the model_rates module (caching, lookup, fallback)."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import prompture.infra.model_rates as mr

# Sample API data mimicking models.dev structure (with capability fields)
SAMPLE_API_DATA = {
    "openai": {
        "gpt-4o": {
            "cost": {"input": 2.5, "output": 10.0},
            "limit": {"context": 128000, "output": 16384},
            "temperature": True,
            "tool_call": True,
            "structured_output": True,
            "reasoning": False,
            "modalities": {"input": ["text", "image"], "output": ["text"]},
        },
        "gpt-4o-mini": {
            "cost": {"input": 0.15, "output": 0.6},
            "limit": {"context": 128000, "output": 16384},
            "temperature": True,
            "tool_call": True,
            "structured_output": True,
            "reasoning": False,
            "modalities": {"input": ["text", "image"], "output": ["text"]},
        },
        "o1": {
            "cost": {"input": 15.0, "output": 60.0},
            "limit": {"context": 200000, "output": 100000},
            "temperature": False,
            "tool_call": True,
            "structured_output": True,
            "reasoning": True,
            "modalities": {"input": ["text", "image"], "output": ["text"]},
        },
    },
    "anthropic": {
        "claude-sonnet-4-20250514": {
            "cost": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
            "limit": {"context": 200000, "output": 8192},
            "temperature": True,
            "tool_call": True,
            "structured_output": True,
            "reasoning": False,
            "modalities": {"input": ["text", "image"], "output": ["text"]},
        },
    },
    "xai": {
        "grok-3": {
            "cost": {"input": 3.0, "output": 15.0},
            "limit": {"context": 131072, "output": 131072},
            "temperature": True,
            "tool_call": True,
            "structured_output": True,
            "reasoning": False,
            "modalities": {"input": ["text"], "output": ["text"]},
        },
    },
    "google": {
        "gemini-2.5-pro": {
            "cost": {"input": 1.25, "output": 10.0},
            "temperature": True,
            "tool_call": True,
            "structured_output": True,
            "reasoning": False,
            "modalities": {"input": ["text", "image", "audio", "video"], "output": ["text"]},
        },
    },
    "groq": {
        "llama2-70b-4096": {
            "cost": {"input": 0.59, "output": 0.79},
            "temperature": True,
            "tool_call": False,
            "structured_output": False,
            "reasoning": False,
            "modalities": {"input": ["text"], "output": ["text"]},
        },
    },
}


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset module-level state before each test."""
    mr._data = None
    mr._loaded = False
    yield
    mr._data = None
    mr._loaded = False


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Override cache dir to a temp directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with (
        patch.object(mr, "_CACHE_DIR", cache_dir),
        patch.object(mr, "_CACHE_FILE", cache_dir / "models_dev.json"),
        patch.object(mr, "_META_FILE", cache_dir / "models_dev_meta.json"),
    ):
        yield cache_dir


class TestGetModelRates:
    def test_returns_rates_when_data_loaded(self):
        """Lookup succeeds when data is already loaded."""
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        rates = mr.get_model_rates("openai", "gpt-4o")
        assert rates is not None
        assert rates["input"] == 2.5
        assert rates["output"] == 10.0

    def test_returns_none_for_unknown_provider(self):
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        assert mr.get_model_rates("unknown_provider", "some-model") is None

    def test_returns_none_for_unknown_model(self):
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        assert mr.get_model_rates("openai", "nonexistent-model") is None

    def test_provider_name_mapping(self):
        """Prompture provider names map to models.dev provider names."""
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        # "claude" maps to "anthropic"
        rates = mr.get_model_rates("claude", "claude-sonnet-4-20250514")
        assert rates is not None
        assert rates["input"] == 3.0
        assert rates["output"] == 15.0

        # "grok" maps to "xai"
        rates = mr.get_model_rates("grok", "grok-3")
        assert rates is not None
        assert rates["input"] == 3.0

    def test_includes_optional_cost_fields(self):
        """Cache-related cost fields are included when present."""
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        rates = mr.get_model_rates("claude", "claude-sonnet-4-20250514")
        assert rates["cache_read"] == 0.3
        assert rates["cache_write"] == 3.75

    def test_returns_none_when_no_data(self):
        """Returns None when data could not be loaded."""
        mr._data = None
        mr._loaded = True

        assert mr.get_model_rates("openai", "gpt-4o") is None

    def test_returns_none_for_missing_cost(self):
        """Returns None if model entry has no cost dict."""
        mr._data = {"openai": {"gpt-x": {"limit": {"context": 128000}}}}
        mr._loaded = True

        assert mr.get_model_rates("openai", "gpt-x") is None


class TestGetModelInfo:
    def test_returns_full_metadata(self):
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        info = mr.get_model_info("openai", "gpt-4o")
        assert info is not None
        assert info["cost"]["input"] == 2.5
        assert info["limit"]["context"] == 128000

    def test_returns_none_for_unknown(self):
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        assert mr.get_model_info("openai", "nope") is None


class TestGetAllProviderModels:
    def test_returns_model_list(self):
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        models = mr.get_all_provider_models("openai")
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models

    def test_provider_name_mapping(self):
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        models = mr.get_all_provider_models("claude")
        assert "claude-sonnet-4-20250514" in models

    def test_returns_empty_for_unknown(self):
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        assert mr.get_all_provider_models("nope") == []

    def test_returns_empty_when_no_data(self):
        mr._data = None
        mr._loaded = True

        assert mr.get_all_provider_models("openai") == []


class TestCacheValidity:
    def test_valid_cache(self, tmp_cache_dir):
        """Cache within TTL is considered valid."""
        cache_file = tmp_cache_dir / "models_dev.json"
        meta_file = tmp_cache_dir / "models_dev_meta.json"

        cache_file.write_text(json.dumps(SAMPLE_API_DATA))
        meta = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "ttl_days": 7,
        }
        meta_file.write_text(json.dumps(meta))

        assert mr._cache_is_valid() is True

    def test_expired_cache(self, tmp_cache_dir):
        """Cache beyond TTL is not valid."""
        cache_file = tmp_cache_dir / "models_dev.json"
        meta_file = tmp_cache_dir / "models_dev_meta.json"

        cache_file.write_text(json.dumps(SAMPLE_API_DATA))
        old_time = datetime.now(timezone.utc) - timedelta(days=10)
        meta = {
            "fetched_at": old_time.isoformat(),
            "ttl_days": 7,
        }
        meta_file.write_text(json.dumps(meta))

        assert mr._cache_is_valid() is False

    def test_missing_cache(self, tmp_cache_dir):
        """No cache files means not valid."""
        assert mr._cache_is_valid() is False


class TestEnsureLoaded:
    @patch.object(mr, "_fetch_from_api", return_value=SAMPLE_API_DATA)
    @patch.object(mr, "_cache_is_valid", return_value=False)
    def test_fetches_when_cache_invalid(self, mock_valid, mock_fetch, tmp_cache_dir):
        """When cache is expired, fetches from API and writes cache."""
        data = mr._ensure_loaded()
        assert data is not None
        assert "openai" in data
        mock_fetch.assert_called_once()

    @patch.object(mr, "_fetch_from_api")
    def test_uses_cache_when_valid(self, mock_fetch, tmp_cache_dir):
        """When cache is valid, reads from disk without fetching."""
        cache_file = tmp_cache_dir / "models_dev.json"
        meta_file = tmp_cache_dir / "models_dev_meta.json"

        cache_file.write_text(json.dumps(SAMPLE_API_DATA))
        meta = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "ttl_days": 7,
        }
        meta_file.write_text(json.dumps(meta))

        data = mr._ensure_loaded()
        assert data is not None
        assert "openai" in data
        mock_fetch.assert_not_called()

    @patch.object(mr, "_fetch_from_api", return_value=None)
    @patch.object(mr, "_cache_is_valid", return_value=False)
    @patch.object(mr, "_read_cache", return_value=SAMPLE_API_DATA)
    def test_falls_back_to_stale_cache(self, mock_read, mock_valid, mock_fetch, tmp_cache_dir):
        """When fetch fails, falls back to stale cache."""
        data = mr._ensure_loaded()
        assert data is not None
        mock_read.assert_called()

    @patch.object(mr, "_fetch_from_api", return_value=None)
    @patch.object(mr, "_cache_is_valid", return_value=False)
    @patch.object(mr, "_read_cache", return_value=None)
    def test_returns_none_when_all_fail(self, mock_read, mock_valid, mock_fetch, tmp_cache_dir):
        """Returns None when fetch fails and no cache exists."""
        data = mr._ensure_loaded()
        assert data is None


class TestRefreshRatesCache:
    @patch.object(mr, "_fetch_from_api", return_value=SAMPLE_API_DATA)
    def test_force_refresh(self, mock_fetch, tmp_cache_dir):
        """Force refresh fetches even when cache is valid."""
        result = mr.refresh_rates_cache(force=True)
        assert result is True
        mock_fetch.assert_called_once()
        assert mr._data is not None

    @patch.object(mr, "_fetch_from_api", return_value=None)
    def test_failed_refresh(self, mock_fetch, tmp_cache_dir):
        """Returns False when fetch fails."""
        result = mr.refresh_rates_cache(force=True)
        assert result is False

    @patch.object(mr, "_cache_is_valid", return_value=True)
    @patch.object(mr, "_fetch_from_api")
    def test_skips_when_cache_valid(self, mock_fetch, mock_valid, tmp_cache_dir):
        """Without force, skips refresh when cache is still valid."""
        result = mr.refresh_rates_cache(force=False)
        assert result is False
        mock_fetch.assert_not_called()


class TestProviderMap:
    def test_all_expected_providers(self):
        """Verify PROVIDER_MAP contains all expected mappings."""
        assert mr.PROVIDER_MAP["openai"] == "openai"
        assert mr.PROVIDER_MAP["claude"] == "anthropic"
        assert mr.PROVIDER_MAP["google"] == "google"
        assert mr.PROVIDER_MAP["groq"] == "groq"
        assert mr.PROVIDER_MAP["grok"] == "xai"
        assert mr.PROVIDER_MAP["azure"] == "azure"
        assert mr.PROVIDER_MAP["openrouter"] == "openrouter"


class TestGetModelCapabilities:
    def test_known_model_with_full_capabilities(self):
        """Returns all capability fields for a well-known model."""
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        caps = mr.get_model_capabilities("openai", "gpt-4o")
        assert caps is not None
        assert caps.supports_temperature is True
        assert caps.supports_tool_use is True
        assert caps.supports_structured_output is True
        assert caps.supports_vision is True
        assert caps.is_reasoning is False
        assert caps.context_window == 128000
        assert caps.max_output_tokens == 16384
        assert caps.modalities_input == ("text", "image")
        assert caps.modalities_output == ("text",)

    def test_reasoning_model(self):
        """Reasoning model has is_reasoning=True and supports_temperature=False."""
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        caps = mr.get_model_capabilities("openai", "o1")
        assert caps is not None
        assert caps.is_reasoning is True
        assert caps.supports_temperature is False
        assert caps.supports_tool_use is True

    def test_model_without_vision(self):
        """Model with text-only input has supports_vision=False."""
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        caps = mr.get_model_capabilities("grok", "grok-3")
        assert caps is not None
        assert caps.supports_vision is False
        assert caps.modalities_input == ("text",)

    def test_model_without_tool_support(self):
        """Model with tool_call=False."""
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        caps = mr.get_model_capabilities("groq", "llama2-70b-4096")
        assert caps is not None
        assert caps.supports_tool_use is False
        assert caps.supports_structured_output is False

    def test_unknown_model_returns_none(self):
        """Returns None for unknown model."""
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        assert mr.get_model_capabilities("openai", "nonexistent") is None

    def test_unknown_provider_returns_none(self):
        """Returns None for unknown provider."""
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        assert mr.get_model_capabilities("unknown", "gpt-4o") is None

    def test_no_data_returns_none(self):
        """Returns None when no API data is loaded."""
        mr._data = None
        mr._loaded = True

        assert mr.get_model_capabilities("openai", "gpt-4o") is None

    def test_model_without_capability_fields(self):
        """Model entry with only cost/limit but no capability fields."""
        mr._data = {"openai": {"gpt-x": {"cost": {"input": 1.0, "output": 2.0}}}}
        mr._loaded = True

        caps = mr.get_model_capabilities("openai", "gpt-x")
        assert caps is not None
        # All capability booleans should be None (unknown)
        assert caps.supports_temperature is None
        assert caps.supports_tool_use is None
        assert caps.supports_structured_output is None
        assert caps.supports_vision is None
        assert caps.is_reasoning is None
        assert caps.context_window is None
        assert caps.max_output_tokens is None
        assert caps.modalities_input == ()
        assert caps.modalities_output == ()

    def test_provider_name_mapping(self):
        """Prompture provider names map correctly to models.dev names."""
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        caps = mr.get_model_capabilities("claude", "claude-sonnet-4-20250514")
        assert caps is not None
        assert caps.supports_tool_use is True
        assert caps.context_window == 200000

    def test_frozen_dataclass(self):
        """ModelCapabilities is immutable."""
        mr._data = SAMPLE_API_DATA
        mr._loaded = True

        caps = mr.get_model_capabilities("openai", "gpt-4o")
        with pytest.raises(AttributeError):
            caps.supports_temperature = False


class TestGetModelConfig:
    """Test _get_model_config() on CostMixin."""

    def test_live_data_overrides_hardcoded_temperature(self):
        """Live models.dev data overrides hardcoded supports_temperature."""
        from prompture.infra.cost_mixin import CostMixin

        mixin = CostMixin()
        mixin.MODEL_PRICING = {
            "test-model": {
                "prompt": 0.01,
                "completion": 0.02,
                "tokens_param": "max_completion_tokens",
                "supports_temperature": True,
            }
        }

        # Mock get_model_capabilities to return supports_temperature=False
        caps = mr.ModelCapabilities(supports_temperature=False, context_window=200000)
        with patch("prompture.infra.model_rates.get_model_capabilities", return_value=caps):
            config = mixin._get_model_config("openai", "test-model")

        assert config["supports_temperature"] is False
        assert config["tokens_param"] == "max_completion_tokens"
        assert config["context_window"] == 200000

    def test_tokens_param_always_from_hardcoded(self):
        """tokens_param is always from MODEL_PRICING, never from models.dev."""
        from prompture.infra.cost_mixin import CostMixin

        mixin = CostMixin()
        mixin.MODEL_PRICING = {
            "test-model": {
                "prompt": 0.01,
                "completion": 0.02,
                "tokens_param": "max_completion_tokens",
                "supports_temperature": True,
            }
        }

        caps = mr.ModelCapabilities(supports_temperature=True)
        with patch("prompture.infra.model_rates.get_model_capabilities", return_value=caps):
            config = mixin._get_model_config("openai", "test-model")

        assert config["tokens_param"] == "max_completion_tokens"

    def test_fallback_when_no_live_data(self):
        """Falls back to hardcoded values when models.dev returns None."""
        from prompture.infra.cost_mixin import CostMixin

        mixin = CostMixin()
        mixin.MODEL_PRICING = {
            "test-model": {
                "prompt": 0.01,
                "completion": 0.02,
                "tokens_param": "max_tokens",
                "supports_temperature": False,
            }
        }

        with patch("prompture.infra.model_rates.get_model_capabilities", return_value=None):
            config = mixin._get_model_config("openai", "test-model")

        assert config["supports_temperature"] is False
        assert config["tokens_param"] == "max_tokens"
        assert config["context_window"] is None
        assert config["max_output_tokens"] is None

    def test_unknown_model_defaults(self):
        """Unknown model gets default values."""
        from prompture.infra.cost_mixin import CostMixin

        mixin = CostMixin()
        mixin.MODEL_PRICING = {}

        with patch("prompture.infra.model_rates.get_model_capabilities", return_value=None):
            config = mixin._get_model_config("openai", "unknown-model")

        assert config["tokens_param"] == "max_tokens"
        assert config["supports_temperature"] is True
        assert config["context_window"] is None
