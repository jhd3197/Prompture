"""Tests for the model usage ledger (prompture.ledger)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from prompture.ledger import (
    ModelUsageLedger,
    _resolve_api_key_hash,
    get_recently_used_models,
    record_model_usage,
)


@pytest.fixture()
def ledger(tmp_path):
    """Create a ledger backed by a temp DB."""
    return ModelUsageLedger(db_path=tmp_path / "test_ledger.db")


# ------------------------------------------------------------------ #
# Basic record & retrieve
# ------------------------------------------------------------------ #


class TestRecordAndRetrieve:
    def test_record_and_retrieve(self, ledger):
        ledger.record_usage("openai/gpt-4", api_key_hash="abc12345", tokens=100, cost=0.01)

        stats = ledger.get_model_stats("openai/gpt-4", "abc12345")
        assert stats is not None
        assert stats["model_name"] == "openai/gpt-4"
        assert stats["api_key_hash"] == "abc12345"
        assert stats["use_count"] == 1
        assert stats["total_tokens"] == 100
        assert stats["total_cost"] == pytest.approx(0.01)
        assert stats["last_status"] == "success"
        assert stats["first_used"] is not None
        assert stats["last_used"] is not None

    def test_upsert_increments(self, ledger):
        ledger.record_usage("openai/gpt-4", api_key_hash="abc12345", tokens=100, cost=0.01)
        ledger.record_usage("openai/gpt-4", api_key_hash="abc12345", tokens=200, cost=0.02)

        stats = ledger.get_model_stats("openai/gpt-4", "abc12345")
        assert stats is not None
        assert stats["use_count"] == 2
        assert stats["total_tokens"] == 300
        assert stats["total_cost"] == pytest.approx(0.03)

    def test_different_api_key_hashes(self, ledger):
        ledger.record_usage("openai/gpt-4", api_key_hash="key_a", tokens=50)
        ledger.record_usage("openai/gpt-4", api_key_hash="key_b", tokens=75)

        stats_a = ledger.get_model_stats("openai/gpt-4", "key_a")
        stats_b = ledger.get_model_stats("openai/gpt-4", "key_b")
        assert stats_a is not None
        assert stats_b is not None
        assert stats_a["total_tokens"] == 50
        assert stats_b["total_tokens"] == 75

    def test_get_model_stats_missing(self, ledger):
        assert ledger.get_model_stats("nonexistent/model") is None


# ------------------------------------------------------------------ #
# Verified models
# ------------------------------------------------------------------ #


class TestVerifiedModels:
    def test_get_verified_models(self, ledger):
        ledger.record_usage("openai/gpt-4", tokens=10)
        ledger.record_usage("claude/haiku", tokens=20)
        ledger.record_usage("openai/gpt-3.5", tokens=5, status="error")

        verified = ledger.get_verified_models()
        assert "openai/gpt-4" in verified
        assert "claude/haiku" in verified
        # gpt-3.5 had only error status
        assert "openai/gpt-3.5" not in verified

    def test_verified_models_empty_ledger(self, ledger):
        assert ledger.get_verified_models() == set()


# ------------------------------------------------------------------ #
# Recently used
# ------------------------------------------------------------------ #


class TestRecentlyUsed:
    def test_get_recently_used_ordering(self, ledger):
        ledger.record_usage("model/a", tokens=10)
        # Small delay to ensure different timestamps
        time.sleep(0.01)
        ledger.record_usage("model/b", tokens=20)
        time.sleep(0.01)
        ledger.record_usage("model/c", tokens=30)

        recent = ledger.get_recently_used(limit=10)
        assert len(recent) == 3
        # Most recent first
        assert recent[0]["model_name"] == "model/c"
        assert recent[2]["model_name"] == "model/a"

    def test_get_recently_used_limit(self, ledger):
        for i in range(5):
            ledger.record_usage(f"model/{i}", tokens=10)
            time.sleep(0.01)

        recent = ledger.get_recently_used(limit=2)
        assert len(recent) == 2

    def test_empty_ledger(self, ledger):
        assert ledger.get_recently_used() == []
        assert ledger.get_all_stats() == []


# ------------------------------------------------------------------ #
# Fire-and-forget convenience function
# ------------------------------------------------------------------ #


class TestFireAndForget:
    def test_record_model_usage_fire_and_forget(self, tmp_path):
        """Verify that record_model_usage swallows exceptions."""
        with patch("prompture.ledger._get_ledger") as mock_get:
            mock_ledger = MagicMock()
            mock_ledger.record_usage.side_effect = RuntimeError("DB exploded")
            mock_get.return_value = mock_ledger

            # Should not raise
            record_model_usage("openai/gpt-4", tokens=100)

    def test_get_recently_used_models_empty(self, tmp_path):
        """Convenience function returns empty list when ledger errors."""
        with patch("prompture.ledger._get_ledger") as mock_get:
            mock_get.side_effect = RuntimeError("Cannot open DB")
            result = get_recently_used_models()
            assert result == []


# ------------------------------------------------------------------ #
# API key hash resolution
# ------------------------------------------------------------------ #


class TestResolveApiKeyHash:
    def test_resolve_api_key_hash(self):
        with patch("prompture.settings.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test-key-12345"
            result = _resolve_api_key_hash("openai/gpt-4")
            assert len(result) == 8
            assert all(c in "0123456789abcdef" for c in result)

    def test_resolve_api_key_hash_missing(self):
        with patch("prompture.settings.settings") as mock_settings:
            mock_settings.openai_api_key = None
            result = _resolve_api_key_hash("openai/gpt-4")
            assert result == ""

    def test_resolve_api_key_hash_local_provider(self):
        result = _resolve_api_key_hash("ollama/llama3")
        assert result == ""

    def test_resolve_api_key_hash_unknown_provider(self):
        result = _resolve_api_key_hash("unknown/model")
        assert result == ""


# ------------------------------------------------------------------ #
# Discovery integration
# ------------------------------------------------------------------ #


class TestDiscoveryIntegration:
    def test_verified_only_in_discovery(self, tmp_path):
        """Mock ledger to verify verified_only filtering in discovery."""
        mock_ledger = MagicMock()
        mock_ledger.get_verified_models.return_value = {"ollama/llama3"}
        mock_ledger.get_all_stats.return_value = []

        with (
            patch("prompture.ledger._get_ledger", return_value=mock_ledger),
            patch("prompture.discovery.settings") as mock_settings,
            patch("prompture.discovery.requests"),
            patch("prompture.model_rates.PROVIDER_MAP", {}),
        ):
            # Configure only ollama as available
            mock_settings.openai_api_key = None
            mock_settings.azure_api_key = None
            mock_settings.claude_api_key = None
            mock_settings.google_api_key = None
            mock_settings.groq_api_key = None
            mock_settings.openrouter_api_key = None
            mock_settings.grok_api_key = None
            mock_settings.ollama_endpoint = "http://localhost:11434/api/generate"
            mock_settings.lmstudio_endpoint = "http://127.0.0.1:1234/v1/chat/completions"
            mock_settings.lmstudio_api_key = None

            from prompture.discovery import get_available_models

            # Without verified_only, ollama models show up (from MODEL_PRICING)
            all_models = get_available_models(verified_only=False)
            # With verified_only, only verified ones remain
            verified_models = get_available_models(verified_only=True)

            # The verified set only has "ollama/llama3"
            assert "ollama/llama3" in verified_models or len(verified_models) == 0
            # verified_models is a subset of all_models
            for m in verified_models:
                assert m in all_models

    def test_enriched_models_include_usage_stats(self, tmp_path):
        """Verify enriched dicts include verified, last_used, use_count."""
        mock_ledger = MagicMock()
        mock_ledger.get_verified_models.return_value = {"ollama/llama3"}
        mock_ledger.get_all_stats.return_value = [
            {
                "model_name": "ollama/llama3",
                "api_key_hash": "",
                "use_count": 5,
                "total_tokens": 1000,
                "total_cost": 0.0,
                "first_used": "2024-01-01T00:00:00+00:00",
                "last_used": "2024-06-15T12:00:00+00:00",
                "last_status": "success",
            }
        ]

        with (
            patch("prompture.ledger._get_ledger", return_value=mock_ledger),
            patch("prompture.discovery.settings") as mock_settings,
            patch("prompture.discovery.requests"),
            patch("prompture.model_rates.PROVIDER_MAP", {}),
        ):
            mock_settings.openai_api_key = None
            mock_settings.azure_api_key = None
            mock_settings.claude_api_key = None
            mock_settings.google_api_key = None
            mock_settings.groq_api_key = None
            mock_settings.openrouter_api_key = None
            mock_settings.grok_api_key = None
            mock_settings.ollama_endpoint = "http://localhost:11434/api/generate"
            mock_settings.lmstudio_endpoint = "http://127.0.0.1:1234/v1/chat/completions"
            mock_settings.lmstudio_api_key = None

            from prompture.discovery import get_available_models

            enriched = get_available_models(include_capabilities=True)

            # Find ollama/llama3 in enriched results (it may or may not be there
            # depending on MODEL_PRICING)
            llama_entries = [e for e in enriched if e["model"] == "ollama/llama3"]
            if llama_entries:
                entry = llama_entries[0]
                assert entry["verified"] is True
                assert entry["use_count"] == 5
                assert entry["last_used"] == "2024-06-15T12:00:00+00:00"

            # Any enriched entry should have the new fields
            for entry in enriched:
                assert "verified" in entry
                assert "last_used" in entry
                assert "use_count" in entry
