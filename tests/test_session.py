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
