"""Tests for tool-result offloading, deferred tool loading and tool pair repair."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from prompture.agents.async_conversation import AsyncConversation
from prompture.agents.conversation import Conversation
from prompture.agents.tool_context import (
    INTERRUPTED_TOOL_RESULT,
    LOAD_TOOLS_NAME,
    READ_TOOL_RESULT_NAME,
    SEARCH_TOOLS_NAME,
    ToolLoader,
    ToolResultPolicy,
    make_read_tool_result,
    repair_tool_pairs,
    shape_tool_results,
)
from prompture.agents.tools_schema import ToolRegistry
from prompture.drivers.async_base import AsyncDriver
from prompture.drivers.base import Driver
from prompture.execution.context import ArtifactStore, count_tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta() -> dict[str, Any]:
    return {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0}


def _calls(*calls: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "text": "",
        "meta": _meta(),
        "tool_calls": [{"id": cid, "name": name, "arguments": args} for cid, name, args in calls],
        "stop_reason": "tool_use",
    }


def _text(text: str) -> dict[str, Any]:
    return {"text": text, "meta": _meta(), "tool_calls": [], "stop_reason": "end_turn"}


class RecordingDriver(Driver):
    """Replays canned responses and records the tool names offered each round."""

    supports_messages = True
    supports_tool_use = True

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self.offered: list[list[str]] = []
        self.sent: list[list[dict[str, Any]]] = []

    def generate(self, prompt, options):
        return self._responses.pop(0)

    def generate_messages(self, messages, options):
        self.sent.append([dict(m) for m in messages])
        return self._responses.pop(0)

    def generate_messages_with_tools(self, messages, tools, options):
        self.sent.append([dict(m) for m in messages])
        self.offered.append([t["function"]["name"] for t in tools])
        return self._responses.pop(0)


class AsyncRecordingDriver(AsyncDriver):
    supports_messages = True
    supports_tool_use = True

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self.offered: list[list[str]] = []

    async def generate(self, prompt, options):
        return self._responses.pop(0)

    async def generate_messages(self, messages, options):
        return self._responses.pop(0)

    async def generate_messages_with_tools(self, messages, tools, options):
        self.offered.append([t["function"]["name"] for t in tools])
        return self._responses.pop(0)


def _tool_contents(conv) -> list[str]:
    return [m["content"] for m in conv.messages if m.get("role") == "tool"]


def _big_registry(payload: str) -> ToolRegistry:
    reg = ToolRegistry()

    def fetch_report() -> str:
        """Return the full report."""
        return payload

    def small() -> str:
        """Return a short value."""
        return "ok"

    reg.register(fetch_report)
    reg.register(small)
    return reg


def _many_tools(n: int = 60) -> ToolRegistry:
    reg = ToolRegistry()
    for i in range(n):

        def fn(account_id: str, region: str = "us", verbose: bool = False, _i: int = i) -> str:
            return f"tool {_i} for {account_id}"

        reg.register(
            fn,
            name=f"service_{i}_operation",
            description=(
                f"Operation {i} of the billing service. Looks up invoices, refunds and usage "
                f"records for an account in a region and returns a detailed JSON report."
            ),
        )

    def get_weather(city: str) -> str:
        """Current weather for a city."""
        return f"sunny in {city}"

    reg.register(get_weather)
    return reg


# ---------------------------------------------------------------------------
# shape_tool_results
# ---------------------------------------------------------------------------


class TestShapeToolResults:
    def test_small_batch_is_untouched(self):
        store = ArtifactStore()
        contents, offloaded = shape_tool_results(
            [("a", "short", False), ("b", "also short", False)],
            policy=ToolResultPolicy(),
            store=store,
            scope="s",
        )
        assert contents == ["short", "also short"]
        assert offloaded == 0

    def test_single_large_result_is_offloaded_with_preview(self):
        store = ArtifactStore()
        big = "HEAD" + "x" * 40_000 + "TAIL"
        policy = ToolResultPolicy(per_call_tokens=1000, batch_tokens=2000)
        contents, offloaded = shape_tool_results([("fetch", big, False)], policy=policy, store=store, scope="s")
        assert offloaded == 1
        stub = contents[0]
        assert "artifact://s/" in stub
        assert stub.startswith("[Result of fetch")
        assert "HEAD" in stub and "TAIL" in stub
        assert len(stub) < 2000
        handle = stub.split("stored as ")[1].split("]")[0]
        assert store.text(handle, scope="s") == big

    def test_batch_offloads_largest_first_until_it_fits(self):
        store = ArtifactStore()
        batch = [("a", "alpha " * 900, False), ("b", "beta " * 450, False), ("c", "gamma " * 300, False)]
        sizes = [count_tokens(text) for _n, text, _e in batch]
        # Each result is under the per-call limit; together they are just over the batch limit.
        policy = ToolResultPolicy(per_call_tokens=max(sizes) + 1, batch_tokens=sum(sizes) - 1)
        contents, offloaded = shape_tool_results(batch, policy=policy, store=store, scope="s")
        assert offloaded == 1
        assert "artifact://" in contents[0]
        assert contents[1] == batch[1][1]
        assert contents[2] == batch[2][1]

    def test_errors_are_never_offloaded(self):
        store = ArtifactStore()
        policy = ToolResultPolicy(per_call_tokens=100, batch_tokens=200, error_chars=50)
        contents, offloaded = shape_tool_results(
            [("a", "Error: " + "e" * 5000, True)], policy=policy, store=store, scope="s"
        )
        assert offloaded == 0
        assert "artifact://" not in contents[0]
        assert "result truncated" in contents[0]

    def test_max_chars_ceiling_offloads(self):
        store = ArtifactStore()
        contents, offloaded = shape_tool_results(
            [("a", "y" * 500, False)], policy=ToolResultPolicy(max_chars=100), store=store, scope="s"
        )
        assert offloaded == 1
        assert "y" * 500 not in contents[0]

    def test_offload_disabled_falls_back_to_truncation(self):
        store = ArtifactStore()
        contents, offloaded = shape_tool_results(
            [("a", "y" * 500, False)],
            policy=ToolResultPolicy(max_chars=100, offload=False),
            store=store,
            scope="s",
        )
        assert offloaded == 0
        assert contents[0].startswith("y" * 100)
        assert "result truncated (500 chars total)" in contents[0]

    def test_policy_rejects_inverted_thresholds(self):
        with pytest.raises(ValueError):
            ToolResultPolicy(per_call_tokens=500, batch_tokens=100)


class TestReadToolResult:
    def _setup(self):
        store = ArtifactStore()
        text = "".join(f"line {i}: value {i * 7}\n" for i in range(2000)) + "needle: the secret is 42\n"
        ref = store.put(text, scope="conv", source="dump")
        return store, ref.handle, text

    def test_pages_by_offset(self):
        store, handle, text = self._setup()
        reader = make_read_tool_result(store, "conv").function
        first = reader(handle=handle, max_chars=500)
        assert first.startswith(text[:500])
        assert "next offset=500" in first
        second = reader(handle=handle, offset=500, max_chars=500)
        assert second.startswith(text[500:1000])

    def test_query_returns_matching_section(self):
        store, handle, _text = self._setup()
        reader = make_read_tool_result(store, "conv").function
        out = reader(handle=handle, query="needle secret")
        assert "the secret is 42" in out

    def test_other_scope_and_unknown_handle_are_refused(self):
        store, handle, _text = self._setup()
        assert "Error" in make_read_tool_result(store, "other").function(handle=handle)
        assert "no stored result" in make_read_tool_result(store, "conv").function(handle="artifact://conv/nope")


# ---------------------------------------------------------------------------
# Offloading inside the conversation loop
# ---------------------------------------------------------------------------


class TestConversationOffload:
    def test_large_result_offloaded_and_readable_next_round(self):
        payload = "intro " * 20 + "the answer is 1234 " + "filler " * 20_000
        driver = RecordingDriver(
            [
                _calls(("c1", "fetch_report", {}), ("c2", "small", {})),
                _calls(("c3", READ_TOOL_RESULT_NAME, {"handle": "PLACEHOLDER", "query": "answer"})),
                _text("It is 1234."),
            ]
        )
        conv = Conversation(driver=driver, tools=_big_registry(payload))

        # Patch the placeholder handle once the first round has produced it.
        original = driver.generate_messages_with_tools

        def patched(messages, tools, options):
            stub = next((m["content"] for m in messages if m.get("tool_call_id") == "c1"), None)
            if stub and driver._responses and driver._responses[0]["tool_calls"]:
                handle = stub.split("stored as ")[1].split("]")[0]
                driver._responses[0]["tool_calls"][0]["arguments"]["handle"] = handle
            return original(messages, tools, options)

        driver.generate_messages_with_tools = patched  # type: ignore[method-assign]

        assert conv.ask("What is the answer?") == "It is 1234."
        tool_msgs = _tool_contents(conv)
        assert "artifact://" in tool_msgs[0]
        assert payload not in tool_msgs[0]
        assert tool_msgs[1] == "ok"
        assert "the answer is 1234" in tool_msgs[2]
        # The reader is only offered once something was offloaded.
        assert READ_TOOL_RESULT_NAME not in driver.offered[0]
        assert READ_TOOL_RESULT_NAME in driver.offered[1]
        # Step extraction still sees the full payload.
        assert conv._full_tool_results["c1"] == payload

    def test_explicit_none_length_keeps_results_whole(self):
        payload = "z" * 100_000
        driver = RecordingDriver([_calls(("c1", "fetch_report", {})), _text("done")])
        conv = Conversation(driver=driver, tools=_big_registry(payload), max_tool_result_length=None)
        conv.ask("go")
        assert _tool_contents(conv)[0] == payload

    def test_clear_drops_artifacts(self):
        payload = "q" * 100_000
        driver = RecordingDriver([_calls(("c1", "fetch_report", {})), _text("done")])
        conv = Conversation(driver=driver, tools=_big_registry(payload))
        conv.ask("go")
        assert conv._artifacts.refs_for(conv.conversation_id)
        conv.clear()
        assert not conv._artifacts.refs_for(conv.conversation_id)
        assert conv._active_tools().names == ["fetch_report", "small"]

    def test_async_conversation_offloads(self):
        payload = "w" * 100_000
        driver = AsyncRecordingDriver([_calls(("c1", "fetch_report", {})), _text("done")])
        conv = AsyncConversation(driver=driver, tools=_big_registry(payload))
        assert asyncio.run(conv.ask("go")) == "done"
        assert "artifact://" in _tool_contents(conv)[0]
        assert READ_TOOL_RESULT_NAME in driver.offered[1]


# ---------------------------------------------------------------------------
# Deferred tool loading
# ---------------------------------------------------------------------------


class TestToolLoader:
    def test_auto_defers_only_large_registries(self):
        loader = ToolLoader()
        assert not loader.is_deferring(_big_registry("x"))
        assert loader.is_deferring(_many_tools())

    def test_forced_on_and_off(self):
        assert ToolLoader(defer=True).is_deferring(_big_registry("x"))
        assert not ToolLoader(defer=False).is_deferring(_many_tools())

    def test_search_and_load(self):
        reg = _many_tools()
        loader = ToolLoader(defer=True)
        search, load = (td.function for td in loader.meta_tools(lambda: reg))
        hits = search(query="weather city")
        assert hits.splitlines()[0].startswith("- get_weather")
        out = load(names=["get_weather", "nope"])
        assert "Loaded get_weather" in out
        assert "Unknown tools: nope" in out
        assert loader.visible(reg).names == ["get_weather"]

    def test_meta_tool_description_lists_deferred_names(self):
        reg = _many_tools()
        search_td = ToolLoader(defer=True).meta_tools(lambda: reg)[0]
        assert "get_weather" in search_td.description


class TestConversationDeferral:
    def test_large_registry_sends_only_meta_tools_then_loaded(self):
        driver = RecordingDriver(
            [
                _calls(("c1", SEARCH_TOOLS_NAME, {"query": "weather"})),
                _calls(("c2", LOAD_TOOLS_NAME, {"names": ["get_weather"]})),
                _calls(("c3", "get_weather", {"city": "Lima"})),
                _text("Sunny in Lima."),
            ]
        )
        conv = Conversation(driver=driver, tools=_many_tools())
        assert conv.ask("Weather in Lima?") == "Sunny in Lima."
        assert sorted(driver.offered[0]) == [LOAD_TOOLS_NAME, SEARCH_TOOLS_NAME]
        assert "get_weather" in driver.offered[2]
        assert "service_0_operation" not in driver.offered[2]
        assert _tool_contents(conv)[-1] == "sunny in Lima"

    def test_unloaded_tool_called_by_name_still_runs_and_stays_loaded(self):
        driver = RecordingDriver(
            [
                _calls(("c1", "get_weather", {"city": "Quito"})),
                _text("done"),
            ]
        )
        conv = Conversation(driver=driver, tools=_many_tools())
        conv.ask("go")
        assert _tool_contents(conv)[0] == "sunny in Quito"
        assert "get_weather" in driver.offered[1]

    def test_preload_and_disable(self):
        driver = RecordingDriver([_text("hi")])
        conv = Conversation(driver=driver, tools=_many_tools(), preload_tools=["get_weather"])
        conv.ask("hi")
        assert "get_weather" in driver.offered[0]
        assert SEARCH_TOOLS_NAME in driver.offered[0]

        driver = RecordingDriver([_text("hi")])
        conv = Conversation(driver=driver, tools=_many_tools(), defer_tools=False)
        conv.ask("hi")
        assert len(driver.offered[0]) == 61
        assert SEARCH_TOOLS_NAME not in driver.offered[0]

    def test_user_tool_named_like_a_helper_wins(self):
        reg = _many_tools()

        def search_tools(query: str) -> str:
            """User's own search."""
            return "user search"

        reg.register(search_tools)
        driver = RecordingDriver([_calls(("c1", SEARCH_TOOLS_NAME, {"query": "x"})), _text("done")])
        conv = Conversation(driver=driver, tools=reg)
        conv.ask("go")
        assert _tool_contents(conv)[0] == "user search"


# ---------------------------------------------------------------------------
# Tool pair repair
# ---------------------------------------------------------------------------


def _assistant(*ids: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": i, "type": "function", "function": {"name": "t", "arguments": "{}"}} for i in ids],
    }


def _result(call_id: str, content: str = "r") -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class TestRepairToolPairs:
    def test_complete_history_is_untouched(self):
        msgs = [{"role": "user", "content": "q"}, _assistant("a", "b"), _result("a"), _result("b")]
        before = [dict(m) for m in msgs]
        assert repair_tool_pairs(msgs) == 0
        assert msgs == before

    def test_missing_result_gets_synthetic_one_in_place(self):
        msgs = [_assistant("a", "b"), _result("a"), {"role": "user", "content": "next"}]
        assert repair_tool_pairs(msgs) == 1
        assert [m.get("tool_call_id") for m in msgs] == [None, "a", "b", None]
        assert msgs[2]["content"] == INTERRUPTED_TOOL_RESULT

    def test_orphans_and_duplicates_are_dropped(self):
        msgs = [
            _result("ghost"),
            {"role": "user", "content": "q"},
            _assistant("a"),
            _result("a"),
            _result("a", "dup"),
            _result("zzz"),
        ]
        assert repair_tool_pairs(msgs) == 3
        assert [m.get("tool_call_id") for m in msgs] == [None, None, "a"]

    def test_interrupted_live_run_is_repaired_on_next_ask(self):
        driver = RecordingDriver([_text("recovered")])
        conv = Conversation(driver=driver, tools=_big_registry("x"))
        conv._messages.extend([{"role": "user", "content": "first"}, _assistant("lost")])
        assert conv.ask("again") == "recovered"
        sent = driver.sent[0]
        roles = [m["role"] for m in sent]
        assert roles == ["user", "assistant", "tool", "user"]
        assert sent[2]["content"] == INTERRUPTED_TOOL_RESULT


# ---------------------------------------------------------------------------
# Summarization threshold
# ---------------------------------------------------------------------------


class TestSummarizeThreshold:
    def _caps(self, monkeypatch, window, max_out):
        from types import SimpleNamespace

        import prompture.infra.model_rates as rates

        monkeypatch.setattr(
            rates,
            "get_model_capabilities",
            lambda provider, model_id: SimpleNamespace(context_window=window, max_output_tokens=max_out),
        )

    def test_small_window_uses_output_reservation(self, monkeypatch):
        from prompture.agents.deep.summarizer import resolve_summarize_threshold

        self._caps(monkeypatch, 32_000, 8_000)
        assert resolve_summarize_threshold("x/small") == 24_000

    def test_explicit_max_tokens_overrides_model_output_limit(self, monkeypatch):
        from prompture.agents.deep.summarizer import resolve_summarize_threshold

        self._caps(monkeypatch, 32_000, 16_000)
        assert resolve_summarize_threshold("x/small") == 16_000
        assert resolve_summarize_threshold("x/small", max_output_tokens=2_000) == 24_000

    def test_large_window_is_capped_and_unknown_falls_back(self, monkeypatch):
        from prompture.agents.deep.summarizer import DEFAULT_SUMMARIZE_CEILING, resolve_summarize_threshold

        self._caps(monkeypatch, 1_000_000, 64_000)
        assert resolve_summarize_threshold("x/big") == DEFAULT_SUMMARIZE_CEILING
        self._caps(monkeypatch, None, None)
        assert resolve_summarize_threshold("x/unknown") == DEFAULT_SUMMARIZE_CEILING
        assert resolve_summarize_threshold("no-provider") == DEFAULT_SUMMARIZE_CEILING

    def test_bad_output_reservation_does_not_loop(self, monkeypatch):
        from prompture.agents.deep.summarizer import resolve_summarize_threshold

        self._caps(monkeypatch, 16_000, 50_000)
        assert resolve_summarize_threshold("x/odd") == 12_000

    def test_middleware_resolves_auto_lazily(self, monkeypatch):
        from prompture.agents.deep.summarizer import SummarizationMiddleware
        from prompture.agents.deep_state import DeepAgentState

        calls = []

        def caps(provider, model_id):
            from types import SimpleNamespace

            calls.append(model_id)
            return SimpleNamespace(context_window=10_000, max_output_tokens=None)

        import prompture.infra.model_rates as rates

        monkeypatch.setattr(rates, "get_model_capabilities", caps)
        mw = SummarizationMiddleware("auto", 4, DeepAgentState(), RecordingDriver([]), model="x/m")
        assert calls == []
        assert mw.threshold_tokens == 7_500
        assert mw.threshold_tokens == 7_500
        assert calls == ["m"]
