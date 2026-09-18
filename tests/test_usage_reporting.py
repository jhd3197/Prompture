"""Regression coverage for complete, evidence-based usage reporting."""

import hashlib
import json

import pytest

from prompture.drivers.async_base import AsyncDriver
from prompture.drivers.base import Driver
from prompture.infra.session import UsageSession
from prompture.infra.tracker import UsageEvent, UsageTracker
from prompture.persistence.serialization import export_usage_session, import_usage_session


class TestCompleteReporting:
    def test_summary_ignores_query_limit_and_honors_filters(self, tmp_path):
        tracker = UsageTracker(tmp_path / "usage.db", flush_threshold=2000)
        for index in range(1002):
            tracker.record(
                UsageEvent(
                    model_name="openai/model",
                    provider="openai",
                    cost=1,
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                    elapsed_ms=3,
                    timestamp="2026-01-01T00:00:00+00:00",
                    session_id="s",
                    conversation_id="c",
                    agent_id="a",
                    operation="extract",
                    tool_name="tool",
                    api_key_hash="key",
                    status="error" if index == 1001 else "success",
                )
            )
        assert len(tracker.query()) == 1000
        filters = dict(
            start="2026-01-01",
            end="2026-01-02",
            model="openai/model",
            provider="openai",
            session_id="s",
            conversation_id="c",
            agent_id="a",
            operation="extract",
            tool_name="tool",
            api_key_hash="key",
            status="success",
        )
        summary = tracker.summary(**filters, limit=1)
        assert summary.total_events == 1001
        assert summary.total_cost == 1001
        assert summary.total_tokens == 12012
        assert summary.models == {"openai/model": 1001}
        assert summary.providers == {"openai": 1001}
        assert tracker.efficiency_report(**filters)["total_events"] == 1001
        assert tracker.summary(session_id="missing").total_events == 0

    def test_efficiency_requires_evidence(self, tmp_path):
        tracker = UsageTracker(tmp_path / "usage.db")
        tracker.record(
            UsageEvent(
                cost=2,
                prompt_tokens=100,
                cached_prompt_tokens=80,
                metadata={
                    "cost_status": "estimated",
                    "cost_breakdown": {"output": 2, "total": 2},
                    "cache_savings": -0.1,
                    "retry_attempt": 1,
                    "extraction_success": False,
                },
            )
        )
        tracker.record(
            UsageEvent(
                cost=1,
                metadata={"cost_status": "partial", "cache_savings": 99, "extraction_success": True, "fallback": True},
            )
        )
        tracker.record(UsageEvent(status="error", metadata={"cost_status": "unknown", "usage_complete": False}))
        tracker.record(UsageEvent(cache_hit=True))
        report = tracker.efficiency_report()
        assert report["cost_status_counts"] == {"estimated": 1, "partial": 1, "unknown": 1, "unclassified": 1}
        assert report["cache_savings"] == -0.1
        assert report["cache_savings_events"] == 1
        assert report["provider_cache_token_ratio"] == 0.8
        assert report["local_cache_hits"] == 1
        assert report["incomplete_usage_events"] == 1
        assert report["cost_per_successful_extraction"] == 3
        assert report["retry_cost"] == 2
        assert report["fallback_cost"] == 1
        empty = tracker.efficiency_report(provider="absent")
        assert empty["cost_per_successful_extraction"] is None
        assert empty["retry_cost"] is None
        assert empty["cache_savings"] is None

    @pytest.mark.parametrize("base", [Driver, AsyncDriver])
    def test_base_preserves_metadata_actual_model_and_key(self, base, monkeypatch, tmp_path):
        class OpenAIDriver(base):
            model = "requested"
            api_key = "explicit-key"

            def generate(self, prompt, options):
                return {}

        tracker = UsageTracker(tmp_path / "usage.db")
        monkeypatch.setattr("prompture.infra.tracker.get_tracker", lambda: tracker)
        meta = {
            "model_name": "returned",
            "cost_status": "unknown",
            "usage_complete": False,
            "usage_details": {"reasoning_tokens": 50},
            "request_id": "req",
            "raw_response": "secret text",
        }
        OpenAIDriver()._auto_record_usage({"meta": meta}, 20)
        row = tracker.query()[0]
        assert row["model_name"] == "openai/returned"
        assert row["api_key_hash"] == hashlib.sha256(b"explicit-key").hexdigest()[:8]
        stored = json.loads(row["metadata"])
        assert stored["usage_details"] == {"reasoning_tokens": 50}
        assert stored["request_id"] == "req"
        assert "raw_response" not in stored

    def test_session_serialization_preserves_provenance_and_snapshots(self):
        session = UsageSession()
        meta = {
            "model_name": "returned",
            "prompt_tokens": 10,
            "cost": 1,
            "cost_status": "estimated",
            "cost_breakdown": {"output": 1},
            "pricing": {"source": "test"},
            "usage_details": {"reasoning_tokens": 4},
            "future_provider_field": {"new": 3},
        }
        session.record({"driver": "requested", "meta": meta, "elapsed_ms": 20})
        meta["future_provider_field"]["new"] = 999
        exported = export_usage_session(session)
        restored = import_usage_session(exported)
        assert restored.summary() == session.summary()
        assert restored.usage_records[0]["metadata"]["future_provider_field"] == {"new": 3}
        assert restored.summary()["cost_breakdown"] == {"output": 1}
        assert "returned" in restored.summary()["per_model"]
        exported["usage_records"][0]["metadata"]["pricing"]["source"] = "changed"
        assert restored.usage_records[0]["metadata"]["pricing"]["source"] == "test"
        restored.reset()
        assert restored.usage_records == []


class TestStreamReporting:
    def test_sync_complete_and_interrupted(self, monkeypatch, tmp_path):
        from prompture.agents.conversation import Conversation

        class StreamDriver(Driver):
            supports_messages = True
            supports_streaming = True
            model = "openai/test"

            def generate(self, prompt, options):
                return {}

            def generate_messages_stream(self, messages, options):
                yield {"type": "delta", "text": "hi"}
                yield {
                    "type": "done",
                    "meta": {
                        "cost": 2,
                        "usage_complete": True,
                        "cost_status": "estimated",
                        "usage_details": {"reasoning_tokens": 4},
                    },
                }

        tracker = UsageTracker(tmp_path / "stream.db")
        monkeypatch.setattr("prompture.infra.tracker.get_tracker", lambda: tracker)
        conv = Conversation(driver=StreamDriver())
        assert list(conv.ask_stream("test")) == ["hi"]
        assert tracker.summary().total_events == 1
        assert conv.usage["usage_details"] == {"reasoning_tokens": 4}
        interrupted = conv._ask_stream_raw("interrupt")
        next(interrupted)
        interrupted.close()
        assert tracker.summary().total_events == 2
        assert tracker.efficiency_report()["incomplete_usage_events"] == 1
        assert tracker.query(status="incomplete")[0]["cost"] == 0
        assert conv.usage["usage_complete"] is False

    async def test_async_complete_and_interrupted(self, monkeypatch, tmp_path):
        from prompture.agents.async_conversation import AsyncConversation

        class StreamDriver(AsyncDriver):
            supports_messages = True
            supports_streaming = True
            model = "openai/test"

            async def generate(self, prompt, options):
                return {}

            async def generate_messages_stream(self, messages, options):
                yield {"type": "delta", "text": "hi"}
                yield {"type": "done", "meta": {"cost": 2, "usage_complete": True, "cost_status": "estimated"}}

        tracker = UsageTracker(tmp_path / "async-stream.db")
        monkeypatch.setattr("prompture.infra.tracker.get_tracker", lambda: tracker)
        conv = AsyncConversation(driver=StreamDriver())
        assert [part async for part in conv.ask_stream("test")] == ["hi"]
        assert tracker.summary().total_events == 1
        interrupted = conv._ask_stream_raw("interrupt")
        await anext(interrupted)
        await interrupted.aclose()
        assert tracker.summary().total_events == 2
        assert tracker.efficiency_report()["incomplete_usage_events"] == 1

    def test_live_message_stop_is_recorded_before_consumer_closes(self, monkeypatch, tmp_path):
        from prompture.agents.conversation import Conversation
        from prompture.agents.live_events import MessageStop, TextDelta
        from prompture.agents.tools_schema import ToolRegistry

        class LiveDriver(Driver):
            supports_messages = True
            supports_tool_use = True
            model = "openai/test"

            def generate(self, prompt, options):
                return {}

            def generate_messages_with_tools_stream(self, messages, tools, options):
                yield TextDelta(text="hi")
                yield MessageStop(
                    stop_reason="end_turn", usage={"cost": 3, "cost_status": "estimated", "usage_complete": True}
                )

        def available_tool() -> str:
            """Provide a tool to select the tool streaming path."""
            return "ok"

        registry = ToolRegistry()
        registry.register(available_tool)
        tracker = UsageTracker(tmp_path / "live.db")
        monkeypatch.setattr("prompture.infra.tracker.get_tracker", lambda: tracker)
        conv = Conversation(driver=LiveDriver(), tools=registry)
        stream = conv.ask_live("test")
        for event in stream:
            if isinstance(event, MessageStop):
                stream.close()
                break
        assert tracker.summary().total_events == 1
        assert tracker.summary().total_cost == 3
        assert conv.usage["cost"] == 3
        assert tracker.efficiency_report()["incomplete_usage_events"] == 0
