"""Tests for prompture.session module."""

from __future__ import annotations

from prompture.session import UsageSession


class TestUsageSession:
    """Tests for UsageSession dataclass."""

    def test_initial_state(self):
        session = UsageSession()
        assert session.prompt_tokens == 0
        assert session.completion_tokens == 0
        assert session.total_tokens == 0
        assert session.total_cost == 0.0
        assert session.call_count == 0
        assert session.errors == 0

    def test_record_single_response(self):
        session = UsageSession()
        session.record(
            {
                "meta": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "cost": 0.003,
                },
                "driver": "openai/gpt-4",
            }
        )

        assert session.prompt_tokens == 100
        assert session.completion_tokens == 50
        assert session.total_tokens == 150
        assert session.total_cost == 0.003
        assert session.call_count == 1

    def test_record_multiple_responses(self):
        session = UsageSession()
        session.record(
            {
                "meta": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.003},
                "driver": "openai/gpt-4",
            }
        )
        session.record(
            {
                "meta": {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300, "cost": 0.006},
                "driver": "openai/gpt-4",
            }
        )

        assert session.prompt_tokens == 300
        assert session.completion_tokens == 150
        assert session.total_tokens == 450
        assert abs(session.total_cost - 0.009) < 1e-9
        assert session.call_count == 2

    def test_record_error(self):
        session = UsageSession()
        session.record_error({"error": Exception("fail"), "driver": "openai/gpt-4"})
        assert session.errors == 1
        session.record_error({"error": Exception("fail again"), "driver": "openai/gpt-4"})
        assert session.errors == 2

    def test_per_model_breakdown(self):
        session = UsageSession()
        session.record(
            {
                "meta": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.003},
                "driver": "openai/gpt-4",
            }
        )
        session.record(
            {
                "meta": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75, "cost": 0.001},
                "driver": "ollama/llama3",
            }
        )

        summary = session.summary()
        assert "openai/gpt-4" in summary["per_model"]
        assert "ollama/llama3" in summary["per_model"]
        assert summary["per_model"]["openai/gpt-4"]["calls"] == 1
        assert summary["per_model"]["ollama/llama3"]["calls"] == 1

    def test_summary_format(self):
        session = UsageSession()
        session.record(
            {
                "meta": {"prompt_tokens": 1000, "completion_tokens": 234, "total_tokens": 1234, "cost": 0.0123},
                "driver": "openai/gpt-4",
            }
        )

        summary = session.summary()
        assert summary["total_tokens"] == 1234
        assert summary["call_count"] == 1
        assert "1,234 tokens" in summary["formatted"]
        assert "$0.0123" in summary["formatted"]
        assert "1 call(s)" in summary["formatted"]

    def test_summary_with_errors(self):
        session = UsageSession()
        session.record_error({"error": Exception("fail")})
        summary = session.summary()
        assert "1 error(s)" in summary["formatted"]

    def test_reset(self):
        session = UsageSession()
        session.record(
            {
                "meta": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.003},
                "driver": "openai/gpt-4",
            }
        )
        session.record_error({"error": Exception("fail")})

        session.reset()
        assert session.prompt_tokens == 0
        assert session.completion_tokens == 0
        assert session.total_tokens == 0
        assert session.total_cost == 0.0
        assert session.call_count == 0
        assert session.errors == 0
        assert session.summary()["per_model"] == {}

    def test_record_missing_meta(self):
        """Record should handle missing meta gracefully."""
        session = UsageSession()
        session.record({"driver": "test"})
        assert session.call_count == 1
        assert session.total_tokens == 0


class TestTimingMetrics:
    """Tests for timing metrics in UsageSession."""

    def test_initial_timing_state(self):
        session = UsageSession()
        assert session.total_elapsed_ms == 0.0
        assert session.tokens_per_second == 0.0
        assert session.latency_stats == {"min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0, "p95_ms": 0.0}

    def test_record_single_with_timing(self):
        session = UsageSession()
        session.record(
            {
                "meta": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.003},
                "driver": "openai/gpt-4",
                "elapsed_ms": 500.0,
            }
        )

        assert session.total_elapsed_ms == 500.0
        # 50 tokens / 0.5 seconds = 100 tokens/sec
        assert session.tokens_per_second == 100.0
        stats = session.latency_stats
        assert stats["min_ms"] == 500.0
        assert stats["max_ms"] == 500.0
        assert stats["avg_ms"] == 500.0

    def test_record_multiple_with_timing(self):
        session = UsageSession()
        session.record(
            {
                "meta": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.003},
                "driver": "openai/gpt-4",
                "elapsed_ms": 200.0,
            }
        )
        session.record(
            {
                "meta": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200, "cost": 0.004},
                "driver": "openai/gpt-4",
                "elapsed_ms": 400.0,
            }
        )
        session.record(
            {
                "meta": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.003},
                "driver": "openai/gpt-4",
                "elapsed_ms": 300.0,
            }
        )

        assert session.total_elapsed_ms == 900.0
        # 200 completion tokens / 0.9 seconds ≈ 222.22 tokens/sec
        assert abs(session.tokens_per_second - 222.22) < 0.1
        stats = session.latency_stats
        assert stats["min_ms"] == 200.0
        assert stats["max_ms"] == 400.0
        assert stats["avg_ms"] == 300.0

    def test_latency_p95(self):
        """Test p95 latency calculation with multiple samples."""
        session = UsageSession()
        # Add 20 samples with varying latencies
        latencies = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 500]
        for latency in latencies:
            session.record(
                {
                    "meta": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "cost": 0.001},
                    "driver": "test",
                    "elapsed_ms": float(latency),
                }
            )

        stats = session.latency_stats
        assert stats["min_ms"] == 100.0
        assert stats["max_ms"] == 500.0
        # p95 index = int(20 * 0.95) = 19, so 19th element (0-indexed) = 500
        assert stats["p95_ms"] == 500.0

    def test_per_model_timing(self):
        """Test that timing is tracked per model."""
        session = UsageSession()
        session.record(
            {
                "meta": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.003},
                "driver": "openai/gpt-4",
                "elapsed_ms": 300.0,
            }
        )
        session.record(
            {
                "meta": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75, "cost": 0.001},
                "driver": "ollama/llama3",
                "elapsed_ms": 100.0,
            }
        )

        summary = session.summary()
        gpt4 = summary["per_model"]["openai/gpt-4"]
        llama = summary["per_model"]["ollama/llama3"]

        assert gpt4["elapsed_ms"] == 300.0
        assert gpt4["elapsed_samples"] == [300.0]
        assert llama["elapsed_ms"] == 100.0
        assert llama["elapsed_samples"] == [100.0]

    def test_summary_includes_timing(self):
        session = UsageSession()
        session.record(
            {
                "meta": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.003},
                "driver": "openai/gpt-4",
                "elapsed_ms": 500.0,
            }
        )

        summary = session.summary()
        assert summary["total_elapsed_ms"] == 500.0
        assert summary["tokens_per_second"] == 100.0
        assert "latency_stats" in summary
        assert "100.0 tok/s" in summary["formatted"]
        assert "500ms avg latency" in summary["formatted"]

    def test_summary_without_timing(self):
        """Summary should not include timing info when no elapsed_ms provided."""
        session = UsageSession()
        session.record(
            {
                "meta": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.003},
                "driver": "openai/gpt-4",
            }
        )

        summary = session.summary()
        assert summary["total_elapsed_ms"] == 0.0
        assert summary["tokens_per_second"] == 0.0
        # Should not include timing in formatted string
        assert "tok/s" not in summary["formatted"]
        assert "latency" not in summary["formatted"]

    def test_reset_clears_timing(self):
        session = UsageSession()
        session.record(
            {
                "meta": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.003},
                "driver": "openai/gpt-4",
                "elapsed_ms": 500.0,
            }
        )

        session.reset()
        assert session.total_elapsed_ms == 0.0
        assert session.tokens_per_second == 0.0
        assert session.latency_stats == {"min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0, "p95_ms": 0.0}

    def test_zero_elapsed_ms_ignored(self):
        """Zero or negative elapsed_ms should not be recorded."""
        session = UsageSession()
        session.record(
            {
                "meta": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.003},
                "driver": "openai/gpt-4",
                "elapsed_ms": 0.0,
            }
        )

        assert session.total_elapsed_ms == 0.0
        assert len(session._elapsed_samples) == 0
