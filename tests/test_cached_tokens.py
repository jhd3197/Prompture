"""Tests for provider-side prompt-cache token tracking (OpenAI).

Covers:
- Cost calculation discounts cached input tokens at the cache_read rate.
- The OpenAI driver extracts ``prompt_tokens_details.cached_tokens`` into meta.
- Cached tokens flow through the meta dict into the SQLite usage tracker.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prompture.drivers.openai_driver import (
    OpenAIDriver,
    _extract_openai_cached_tokens,
    _extract_openai_meta,
)
from prompture.infra.cost_mixin import CostMixin
from prompture.infra.tracker import UsageEvent, UsageTracker


# ---------------------------------------------------------------------------
# CostMixin._calculate_cost — cache_read discount math
# ---------------------------------------------------------------------------


class TestCostCalcWithCachedTokens:
    """``_calculate_cost`` should bill cached tokens at the cache_read rate."""

    def _bare_mixin(self) -> CostMixin:
        return CostMixin()

    @patch("prompture.infra.model_rates.get_model_rates")
    def test_cached_tokens_use_cache_read_rate(self, mock_rates):
        mock_rates.return_value = {
            "input": 10.0,    # $10 / 1M
            "output": 30.0,   # $30 / 1M
            "cache_read": 1.0,  # $1 / 1M (10% of input)
        }
        mixin = self._bare_mixin()

        # 1000 prompt tokens, 800 of them cached; 500 completion
        cost = mixin._calculate_cost(
            "openai", "gpt-4o", 1000, 500, cached_tokens=800
        )

        # non_cached = 200 → 200/1M * 10 = 0.002
        # cached = 800 → 800/1M * 1 = 0.0008
        # completion = 500 → 500/1M * 30 = 0.015
        # total = 0.0178
        assert cost == pytest.approx(0.0178, abs=1e-9)

    @patch("prompture.infra.model_rates.get_model_rates")
    def test_no_cached_tokens_matches_old_behavior(self, mock_rates):
        mock_rates.return_value = {"input": 10.0, "output": 30.0, "cache_read": 1.0}
        mixin = self._bare_mixin()

        cost_with_zero_cached = mixin._calculate_cost(
            "openai", "gpt-4o", 1000, 500, cached_tokens=0
        )
        cost_default = mixin._calculate_cost("openai", "gpt-4o", 1000, 500)

        assert cost_with_zero_cached == cost_default
        # 1000/1M * 10 + 500/1M * 30 = 0.01 + 0.015 = 0.025
        assert cost_default == pytest.approx(0.025, abs=1e-9)

    @patch("prompture.infra.model_rates.get_model_rates")
    def test_falls_back_to_input_rate_when_no_cache_read_rate(self, mock_rates):
        # Model has no cache_read rate published → cached tokens billed at full input.
        mock_rates.return_value = {"input": 10.0, "output": 30.0}
        mixin = self._bare_mixin()

        cost = mixin._calculate_cost(
            "openai", "some-model", 1000, 500, cached_tokens=800
        )
        # Effectively: 1000/1M * 10 + 500/1M * 30 = 0.025
        assert cost == pytest.approx(0.025, abs=1e-9)

    @patch("prompture.infra.model_rates.get_model_rates")
    def test_zero_rates_returns_zero(self, mock_rates):
        mock_rates.return_value = None
        mixin = self._bare_mixin()
        assert mixin._calculate_cost("openai", "x", 1000, 500, cached_tokens=400) == 0.0


# ---------------------------------------------------------------------------
# _extract_openai_cached_tokens / _extract_openai_meta
# ---------------------------------------------------------------------------


class TestOpenAIMetaExtraction:
    """The OpenAI driver should pull cached_tokens out of ``prompt_tokens_details``."""

    def test_extract_returns_zero_when_usage_missing(self):
        assert _extract_openai_cached_tokens(None) == 0

    def test_extract_returns_zero_when_details_missing(self):
        usage = MagicMock(spec=["prompt_tokens", "completion_tokens", "total_tokens"])
        # spec excludes prompt_tokens_details → getattr returns the default
        usage.prompt_tokens_details = None
        assert _extract_openai_cached_tokens(usage) == 0

    def test_extract_returns_cached_tokens_when_present(self):
        usage = MagicMock()
        usage.prompt_tokens_details.cached_tokens = 512
        assert _extract_openai_cached_tokens(usage) == 512

    def test_meta_includes_cached_prompt_tokens_zero_by_default(self):
        resp = MagicMock()
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50
        resp.usage.total_tokens = 150
        resp.usage.prompt_tokens_details = None
        resp.model_dump.return_value = {}

        meta = _extract_openai_meta(resp, "gpt-4o", 0.01)
        assert meta["cached_prompt_tokens"] == 0

    def test_meta_includes_cached_prompt_tokens_when_set(self):
        resp = MagicMock()
        resp.usage.prompt_tokens = 1000
        resp.usage.completion_tokens = 50
        resp.usage.total_tokens = 1050
        resp.usage.prompt_tokens_details.cached_tokens = 800
        resp.model_dump.return_value = {}

        meta = _extract_openai_meta(resp, "gpt-4o", 0.01)
        assert meta["cached_prompt_tokens"] == 800
        assert meta["prompt_tokens"] == 1000


# ---------------------------------------------------------------------------
# End-to-end: OpenAIDriver.generate with mocked client
# ---------------------------------------------------------------------------


class TestOpenAIDriverEndToEnd:
    """Hit the driver with a mocked OpenAI client and verify meta + cost."""

    @patch("prompture.infra.model_rates.get_model_rates")
    def test_generate_records_cached_prompt_tokens_in_meta(self, mock_rates):
        mock_rates.return_value = {
            "input": 10.0, "output": 30.0, "cache_read": 1.0,
        }

        driver = OpenAIDriver.__new__(OpenAIDriver)
        driver.api_key = "test"
        driver.model = "gpt-4o"

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "hello"
        mock_resp.usage.prompt_tokens = 1000
        mock_resp.usage.completion_tokens = 500
        mock_resp.usage.total_tokens = 1500
        mock_resp.usage.prompt_tokens_details.cached_tokens = 800
        mock_resp.model_dump.return_value = {}

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        driver.client = mock_client

        # Bypass models.dev capability validation noise
        with patch.object(driver, "_validate_model_capabilities"):
            result = driver.generate("test", {})

        meta = result["meta"]
        assert meta["cached_prompt_tokens"] == 800
        assert meta["prompt_tokens"] == 1000
        # Cost should reflect the cache discount: 0.0178 (see TestCostCalc above)
        assert meta["cost"] == pytest.approx(0.0178, abs=1e-9)


# ---------------------------------------------------------------------------
# Tracker persistence
# ---------------------------------------------------------------------------


class TestTrackerCachedPromptTokens:
    """``cached_prompt_tokens`` should round-trip through the SQLite tracker."""

    def test_record_and_query_cached_prompt_tokens(self, tmp_path):
        tracker = UsageTracker(db_path=tmp_path / "usage.db", flush_threshold=1)
        tracker.record(
            UsageEvent(
                model_name="openai/gpt-4o",
                provider="openai",
                prompt_tokens=1000,
                cached_prompt_tokens=800,
                completion_tokens=500,
                total_tokens=1500,
                cost=0.0178,
            )
        )

        rows = tracker.query()
        assert len(rows) == 1
        assert rows[0]["cached_prompt_tokens"] == 800
        assert rows[0]["prompt_tokens"] == 1000

    def test_default_cached_prompt_tokens_is_zero(self, tmp_path):
        tracker = UsageTracker(db_path=tmp_path / "usage.db", flush_threshold=1)
        tracker.record(UsageEvent(model_name="x", provider="x"))

        rows = tracker.query()
        assert rows[0]["cached_prompt_tokens"] == 0

    def test_alter_migration_adds_column_to_old_db(self, tmp_path):
        """A pre-existing DB without cached_prompt_tokens should be migrated."""
        import sqlite3

        db_path = tmp_path / "old_usage.db"
        # Build a minimal "old" schema without the cached_prompt_tokens column.
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                """
                CREATE TABLE usage_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    api_key_hash TEXT DEFAULT '',
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    cost REAL DEFAULT 0.0,
                    elapsed_ms REAL DEFAULT 0.0,
                    session_id TEXT,
                    conversation_id TEXT,
                    agent_id TEXT,
                    tool_name TEXT,
                    operation TEXT,
                    cache_hit INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'success',
                    error_type TEXT,
                    tags TEXT,
                    metadata TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

        tracker = UsageTracker(db_path=db_path, flush_threshold=1)
        # Trigger lazy init + migration
        tracker.record(
            UsageEvent(
                model_name="openai/gpt-4o",
                provider="openai",
                prompt_tokens=200,
                cached_prompt_tokens=150,
            )
        )

        rows = tracker.query()
        assert rows[0]["cached_prompt_tokens"] == 150
