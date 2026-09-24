"""Reasoning replay, message compression and fusion."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import pytest

from prompture.agents.live_events import MessageStop, TextDelta, ThinkingDelta
from prompture.exceptions import DriverError
from prompture.gateway import ChatOutcome, ReasoningCache, run_chat, stream_anthropic_events
from prompture.groups.fusion import AsyncFusionDriver, FusionDriver, clear_fusions, register_fusion
from prompture.infra.compression import compress_messages
from prompture.resilience import clear_virtual_models, list_virtual_models, resolve_virtual_model

# ---------------------------------------------------------------------------
# Reasoning replay
# ---------------------------------------------------------------------------


class TestReasoningCache:
    def test_restores_by_tool_call_id_and_text(self):
        cache = ReasoningCache()
        cache.remember(ChatOutcome(text="", tool_calls=[{"id": "call_1", "name": "f"}], reasoning="think A"))
        cache.remember(ChatOutcome(text="The answer is 4.", reasoning="think B"))
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "x"},
            {"role": "assistant", "content": "The answer is 4.  "},
            {"role": "assistant", "content": "unknown"},
            {"role": "assistant", "content": "kept", "reasoning_content": "mine"},
        ]
        out = cache.restore(msgs)
        assert out[1]["reasoning_content"] == "think A"
        assert out[3]["reasoning_content"] == "think B"
        assert "reasoning_content" not in out[4]
        assert out[5]["reasoning_content"] == "mine"
        assert "reasoning_content" not in msgs[1]  # input untouched

    def test_lru_and_ttl(self, monkeypatch):
        cache = ReasoningCache(max_entries=1, ttl=10)
        cache.remember(ChatOutcome(text="a", reasoning="r1"))
        cache.remember(ChatOutcome(text="b", reasoning="r2"))
        assert len(cache) == 1
        assert "reasoning_content" not in cache.restore([{"role": "assistant", "content": "a"}])[0]
        now = time.monotonic()
        monkeypatch.setattr(time, "monotonic", lambda: now + 11)
        assert "reasoning_content" not in cache.restore([{"role": "assistant", "content": "b"}])[0]

    def test_outcomes_capture_reasoning(self):
        class D:
            def generate_messages(self, messages, options):
                return {"text": "4", "meta": {}, "reasoning_content": "2+2"}

        assert run_chat(D(), [], {}).reasoning == "2+2"
        done: list[ChatOutcome] = []
        list(
            stream_anthropic_events(
                [
                    ThinkingDelta(text="hmm "),
                    ThinkingDelta(text="ok"),
                    TextDelta(text="4"),
                    MessageStop(stop_reason="end_turn"),
                ],
                model="m",
                on_complete=done.append,
            )
        )
        assert done[0].reasoning == "hmm ok"


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def test_compress_messages():
    big = "line\n" * 5000
    msgs = [
        {"role": "system", "content": "rules  \n\n\n\nmore"},
        {"role": "user", "content": "keep   my\n\n\n\nspacing"},
        {"role": "system", "content": "rules  \n\n\n\nmore"},
        {"role": "tool", "tool_call_id": "a", "content": big},
        {"role": "tool", "tool_call_id": "b", "content": big},
    ]
    out, stats = compress_messages(msgs, max_tool_chars=1000)
    assert out[0]["content"] == "rules\n\nmore"
    assert out[1]["content"] == "keep   my\n\n\n\nspacing"  # prose untouched
    assert len(out) == 4  # duplicate system dropped
    assert "characters of tool output omitted" in out[2]["content"]
    assert out[2]["content"].startswith("line\n") and out[2]["content"].endswith("line\n")
    assert out[3]["content"] == big  # latest tool result kept whole
    assert stats.tool_results_trimmed == 1
    assert stats.duplicate_system_removed == 1
    assert stats.saved_chars > 20000
    assert msgs[3]["content"] == big  # input untouched

    out2, _ = compress_messages(msgs, max_tool_chars=1000, trim_latest_tool=True)
    assert "omitted" in out2[-1]["content"]


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


class Panelist:
    def __init__(self, name: str, text: str | Exception, delay: float = 0.0) -> None:
        self.model = name
        self.text = text
        self.delay = delay
        self.seen: list[Any] = []

    def generate_messages(self, messages, options):
        self.seen.append(messages)
        if self.delay:
            time.sleep(self.delay)
        if isinstance(self.text, Exception):
            raise self.text
        return {
            "text": self.text,
            "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
        }


class TestFusion:
    def test_panel_then_judge(self):
        a = Panelist("p/a", "Answer from A")
        b = Panelist("p/b", "Answer from B")
        broken = Panelist("p/c", RuntimeError("down"))
        judge = Panelist("j/judge", "fused")
        drv = FusionDriver([a, b, broken], judge, rng=random.Random(0))
        resp = drv.generate_messages(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "Explain X"}], {}
        )

        assert resp["text"] == "fused"
        prompt = judge.seen[0][-1]["content"]
        assert "Explain X" in prompt and "Answer A:" in prompt and "Answer B:" in prompt
        assert "p/a" not in prompt  # anonymized
        assert judge.seen[0][0] == {"role": "system", "content": "sys"}
        meta = resp["meta"]
        assert meta["cost"] == pytest.approx(0.03)
        assert meta["total_tokens"] == 45
        fusion = meta["fusion"]
        assert fusion["answers_used"] == 2
        assert {p["model"]: p["ok"] for p in fusion["panel"]} == {"p/a": True, "p/b": True, "p/c": False}

    def test_single_answer_skips_judge(self):
        judge = Panelist("j", "never")
        drv = FusionDriver([Panelist("p/a", "only"), Panelist("p/b", RuntimeError("x"))], judge)
        assert drv.generate("q", {})["text"] == "only"
        assert judge.seen == []

    def test_timeout_and_min_answers(self):
        slow = Panelist("p/slow", "late", delay=0.5)
        drv = FusionDriver([slow, Panelist("p/fast", "fast")], Panelist("j", "x"), timeout=0.1)
        assert drv.generate("q", {})["text"] == "fast"
        assert "timed out" in drv.last_fusion["panel"][-1]["error"]

        strict = FusionDriver([Panelist("p/a", RuntimeError("x"))], Panelist("j", "x"), min_answers=1)
        with pytest.raises(DriverError, match="0 usable"):
            strict.generate("q", {})

    def test_virtual_model_and_async(self):
        clear_fusions()
        clear_virtual_models()
        try:
            register_fusion("council", ["p/a", "p/b"], "j/judge")
            assert "fusion/council" in list_virtual_models()
            _, drv = resolve_virtual_model("fusion/council")
            assert isinstance(drv, FusionDriver) and drv.model == "fusion/council"
            _, adrv = resolve_virtual_model("fusion/council", async_=True)
            assert isinstance(adrv, AsyncFusionDriver)

            adrv._sync._drivers.update(
                {"p/a": Panelist("p/a", "A"), "p/b": Panelist("p/b", "B"), "j/judge": Panelist("j", "J")}
            )
            assert asyncio.run(adrv.generate("q", {}))["text"] == "J"
            with pytest.raises(ValueError, match="Unknown fusion"):
                resolve_virtual_model("fusion/nope")
        finally:
            clear_fusions()
