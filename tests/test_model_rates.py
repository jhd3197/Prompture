"""Tests for the model_rates module (caching, lookup, fallback)."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import prompture.model_rates as mr

# Sample API data mimicking models.dev structure
SAMPLE_API_DATA = {
    "openai": {
        "gpt-4o": {
            "cost": {"input": 2.5, "output": 10.0},
            "limit": {"context": 128000, "output": 16384},
        },
        "gpt-4o-mini": {
            "cost": {"input": 0.15, "output": 0.6},
            "limit": {"context": 128000, "output": 16384},
        },
    },
    "anthropic": {
        "claude-sonnet-4-20250514": {
            "cost": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
            "limit": {"context": 200000, "output": 8192},
        },
    },
    "xai": {
        "grok-3": {
            "cost": {"input": 3.0, "output": 15.0},
            "limit": {"context": 131072, "output": 131072},
        },
    },
    "google": {
        "gemini-2.5-pro": {
            "cost": {"input": 1.25, "output": 10.0},
        },
    },
    "groq": {
        "llama2-70b-4096": {
            "cost": {"input": 0.59, "output": 0.79},
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
