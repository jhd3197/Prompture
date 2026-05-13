"""Tests for the DeepAgent stack (state, tools, sub-agents, summarization, end-to-end)."""

from __future__ import annotations

from typing import Any

import pytest

from prompture.agents.deep import (
    SummarizationMiddleware,
    make_task_tool,
    make_vfs_tools,
    make_write_todos_tool,
)
from prompture.agents.deep_agent import DeepAgent, create_deep_agent
from prompture.agents.deep_prompts import (
    BASE_DEEP_AGENT_PROMPT,
    PLANNING_PROMPT_FRAGMENT,
    VFS_PROMPT_FRAGMENT,
    assemble_system_prompt,
    format_subagent_section,
)
from prompture.agents.deep_state import (
    DeepAgentResult,
    DeepAgentState,
    SubAgentCallRecord,
    SubAgentSpec,
    SummaryEvent,
    Todo,
)
from prompture.drivers.base import Driver

# ---------------------------------------------------------------------------
# Mock drivers
# ---------------------------------------------------------------------------


class MockToolDriver(Driver):
    """Mock driver supporting tool use; returns scripted responses in order."""

    supports_messages = True
    supports_tool_use = True

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self._idx = 0
        self.model = "mock/tool-driver"

    def generate(self, prompt, options):
        return self._next()

    def generate_messages(self, messages, options):
        return self._next()

    def generate_messages_with_tools(self, messages, tools, options):
        return self._next()

    def _next(self):
        if self._idx >= len(self._responses):
            raise AssertionError(
                f"MockToolDriver ran out of responses after {self._idx} calls"
            )
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


class CountingDriver(Driver):
    """Mock driver reporting a configurable prompt_tokens per call."""

    supports_messages = True
    supports_tool_use = True

    def __init__(self, prompt_tokens_sequence: list[int], final_text: str = "done"):
        self._tokens = list(prompt_tokens_sequence)
        self._idx = 0
        self._final = final_text
        self.model = "mock/counting"

    def generate(self, prompt, options):
        return self._next()

    def generate_messages(self, messages, options):
        return self._next()

    def generate_messages_with_tools(self, messages, tools, options):
        return self._next()

    def _next(self):
        pt = self._tokens[self._idx] if self._idx < len(self._tokens) else 1
        is_last = self._idx == len(self._tokens) - 1
        self._idx += 1
        return {
            "text": self._final if is_last else "",
            "meta": {
                "prompt_tokens": pt,
                "completion_tokens": 5,
                "total_tokens": pt + 5,
                "cost": 0.0,
            },
            "tool_calls": [],
            "stop_reason": "end_turn",
        }


class StaticSummariserDriver(Driver):
    """Mock driver that always returns a fixed summary text."""

    supports_messages = True
    supports_tool_use = False

    def __init__(self, summary: str = "[SUMMARY]"):
        self._summary = summary
        self.calls = 0
        self.model = "mock/summariser"

    def generate(self, prompt, options):
        self.calls += 1
        return self._make()

    def generate_messages(self, messages, options):
        self.calls += 1
        return self._make()

    def _make(self):
        return {
            "text": self._summary,
            "meta": {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70,
                "cost": 0.0,
            },
        }


# ---------------------------------------------------------------------------
# DeepAgentState
# ---------------------------------------------------------------------------


class TestDeepAgentState:
    def test_defaults_empty(self):
        s = DeepAgentState()
        assert s.todos == []
        assert s.files == {}
        assert s.sub_agent_calls == []
        assert s.summary_events == []

    def test_reset_clears_everything(self):
        s = DeepAgentState(files={"a.md": "x"})
        s.todos.append(Todo(content="t1"))
        s.sub_agent_calls.append(
            SubAgentCallRecord(subagent_name="x", description="d", result_summary="r")
        )
        s.summary_events.append(
            SummaryEvent(triggered_at_message_index=1, summary_text="s", summarized_message_count=2)
        )
        s.reset()
        assert s.todos == []
        assert s.files == {}
        assert s.sub_agent_calls == []
        assert s.summary_events == []

    def test_round_trip_serialization(self):
        s = DeepAgentState(files={"x.md": "hi"})
        s.todos.append(Todo(content="step1", status="in_progress", active_form="Doing"))
        s.sub_agent_calls.append(
            SubAgentCallRecord(
                subagent_name="fact_checker",
                description="check this",
                result_summary="ok",
                usage={"cost": 0.5},
                iterations=3,
            )
        )
        s.summary_events.append(
            SummaryEvent(
                triggered_at_message_index=10,
                summary_text="prev convo",
                summarized_message_count=4,
                summarizer_usage={"cost": 0.1},
            )
        )
        data = s.to_dict()
        restored = DeepAgentState.from_dict(data)
        assert restored.files == s.files
        assert len(restored.todos) == 1
        assert restored.todos[0].status == "in_progress"
        assert restored.sub_agent_calls[0].iterations == 3
        assert restored.summary_events[0].summarized_message_count == 4


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class TestPlanner:
    def test_write_todos_replaces_list(self):
        state = DeepAgentState()
        tool = make_write_todos_tool(state)
        tool.function(
            todos=[
                {"content": "step1", "status": "pending"},
                {"content": "step2", "status": "in_progress", "active_form": "doing step2"},
            ]
        )
        assert len(state.todos) == 2
        assert state.todos[0].content == "step1"
        assert state.todos[1].active_form == "doing step2"

    def test_replace_drops_previous(self):
        state = DeepAgentState()
        tool = make_write_todos_tool(state)
        tool.function(todos=[{"content": "old", "status": "pending"}])
        tool.function(todos=[{"content": "new", "status": "completed"}])
        assert len(state.todos) == 1
        assert state.todos[0].content == "new"
        assert state.todos[0].status == "completed"

    def test_invalid_status_normalised(self):
        state = DeepAgentState()
        tool = make_write_todos_tool(state)
        tool.function(todos=[{"content": "x", "status": "bogus"}])
        assert state.todos[0].status == "pending"

    def test_at_most_one_in_progress(self):
        state = DeepAgentState()
        tool = make_write_todos_tool(state)
        tool.function(
            todos=[
                {"content": "a", "status": "in_progress"},
                {"content": "b", "status": "in_progress"},
            ]
        )
        in_progress = [t for t in state.todos if t.status == "in_progress"]
        assert len(in_progress) == 1
        assert in_progress[0].content == "a"

    def test_empty_content_dropped(self):
        state = DeepAgentState()
        tool = make_write_todos_tool(state)
        tool.function(todos=[{"content": "", "status": "pending"}, {"content": "ok"}])
        assert len(state.todos) == 1
        assert state.todos[0].content == "ok"

    def test_returns_status_summary(self):
        state = DeepAgentState()
        tool = make_write_todos_tool(state)
        msg = tool.function(
            todos=[
                {"content": "a", "status": "completed"},
                {"content": "b", "status": "in_progress"},
                {"content": "c", "status": "pending"},
            ]
        )
        assert "3 item" in msg
        assert "1 completed" in msg
        assert "1 in progress" in msg


# ---------------------------------------------------------------------------
# Virtual filesystem
# ---------------------------------------------------------------------------


class TestVFS:
    def _tools(self, state: DeepAgentState) -> dict[str, Any]:
        return {t.name: t for t in make_vfs_tools(state)}

    def test_write_then_read(self):
        s = DeepAgentState()
        tools = self._tools(s)
        tools["write_file"].function(file_path="notes/a.md", content="hello\nworld")
        out = tools["read_file"].function(file_path="notes/a.md")
        assert "hello" in out
        assert "world" in out
        # Line numbers are included
        assert "     1\thello" in out
        assert "     2\tworld" in out

    def test_read_missing_file(self):
        s = DeepAgentState()
        out = self._tools(s)["read_file"].function(file_path="missing.md")
        assert "Error" in out
        assert "missing.md" in out

    def test_read_with_pagination(self):
        s = DeepAgentState()
        tools = self._tools(s)
        content = "\n".join(f"line{i}" for i in range(1, 11))
        tools["write_file"].function(file_path="big.txt", content=content)
        out = tools["read_file"].function(file_path="big.txt", offset=3, limit=3)
        assert "line4" in out
        assert "line5" in out
        assert "line6" in out
        # truncation marker present because we didn't read to end
        assert "truncated" in out

    def test_path_normalisation_strips_leading_slash(self):
        s = DeepAgentState()
        tools = self._tools(s)
        tools["write_file"].function(file_path="/foo.md", content="x")
        # Stored without leading slash
        assert "foo.md" in s.files
        assert "/foo.md" not in s.files
        # Read works with or without leading slash
        assert "x" in tools["read_file"].function(file_path="/foo.md")
        assert "x" in tools["read_file"].function(file_path="foo.md")

    def test_edit_file_unique_match(self):
        s = DeepAgentState()
        tools = self._tools(s)
        tools["write_file"].function(file_path="a.md", content="foo bar baz")
        result = tools["edit_file"].function(
            file_path="a.md", old_string="bar", new_string="QUX"
        )
        assert "Replaced 1" in result
        assert s.files["a.md"] == "foo QUX baz"

    def test_edit_file_ambiguous(self):
        s = DeepAgentState()
        tools = self._tools(s)
        tools["write_file"].function(file_path="a.md", content="foo foo foo")
        result = tools["edit_file"].function(
            file_path="a.md", old_string="foo", new_string="BAR"
        )
        assert "Error" in result
        assert "3 times" in result
        # File is unchanged
        assert s.files["a.md"] == "foo foo foo"

    def test_edit_file_replace_all(self):
        s = DeepAgentState()
        tools = self._tools(s)
        tools["write_file"].function(file_path="a.md", content="foo foo foo")
        result = tools["edit_file"].function(
            file_path="a.md", old_string="foo", new_string="BAR", replace_all=True
        )
        assert "Replaced 3" in result
        assert s.files["a.md"] == "BAR BAR BAR"

    def test_edit_file_not_found(self):
        s = DeepAgentState()
        out = self._tools(s)["edit_file"].function(
            file_path="ghost.md", old_string="x", new_string="y"
        )
        assert "Error" in out
        assert "not found" in out

    def test_ls(self):
        s = DeepAgentState()
        tools = self._tools(s)
        for p in ("notes/a.md", "notes/b.md", "drafts/c.md"):
            tools["write_file"].function(file_path=p, content="x")
        all_files = tools["ls"].function()
        assert "notes/a.md" in all_files
        assert "drafts/c.md" in all_files
        notes_only = tools["ls"].function(path="notes/")
        assert "notes/a.md" in notes_only
        assert "drafts/c.md" not in notes_only

    def test_ls_empty(self):
        s = DeepAgentState()
        assert "no files" in self._tools(s)["ls"].function()

    def test_glob(self):
        s = DeepAgentState()
        tools = self._tools(s)
        for p in ("notes/a.md", "notes/b.txt", "drafts/c.md"):
            tools["write_file"].function(file_path=p, content="x")
        md = tools["glob"].function(pattern="*/*.md")
        assert "notes/a.md" in md
        assert "drafts/c.md" in md
        assert "notes/b.txt" not in md

    def test_grep_files_with_matches(self):
        s = DeepAgentState()
        tools = self._tools(s)
        tools["write_file"].function(file_path="a.md", content="hello world")
        tools["write_file"].function(file_path="b.md", content="goodbye")
        out = tools["grep"].function(pattern="hello")
        assert "a.md" in out
        assert "b.md" not in out

    def test_grep_content_mode(self):
        s = DeepAgentState()
        tools = self._tools(s)
        tools["write_file"].function(file_path="a.md", content="line1 alpha\nline2 beta\nline3 alpha")
        out = tools["grep"].function(pattern="alpha", output_mode="content")
        assert "a.md:1:line1 alpha" in out
        assert "a.md:3:line3 alpha" in out
        assert "line2" not in out

    def test_grep_count(self):
        s = DeepAgentState()
        tools = self._tools(s)
        tools["write_file"].function(file_path="a.md", content="x x y x")
        out = tools["grep"].function(pattern="x", output_mode="count")
        assert "a.md: 3" in out

    def test_grep_invalid_regex(self):
        s = DeepAgentState()
        tools = self._tools(s)
        tools["write_file"].function(file_path="a.md", content="x")
        out = tools["grep"].function(pattern="[")
        assert "invalid regex" in out


# ---------------------------------------------------------------------------
# Sub-agents (uses MockDriver responses end-to-end)
# ---------------------------------------------------------------------------


class TestSubAgents:
    def test_task_tool_unknown_type_errors(self):
        """The task tool returns a friendly error for unknown sub-agent names."""
        state = DeepAgentState()
        spec = SubAgentSpec(
            name="answerer",
            description="Answers numeric questions.",
            system_prompt="You answer numbers.",
        )
        tool = make_task_tool(
            specs=[spec],
            state=state,
            parent_model="mock/parent",
            parent_user_tools=[],
        )
        out = tool.function(description="x", subagent_type="unknown")
        assert "unknown subagent_type" in out

    def test_duplicate_spec_names_raise(self):
        state = DeepAgentState()
        specs = [
            SubAgentSpec(name="x", description="d", system_prompt="p"),
            SubAgentSpec(name="x", description="d", system_prompt="p"),
        ]
        with pytest.raises(ValueError):
            make_task_tool(
                specs=specs,
                state=state,
                parent_model="m",
                parent_user_tools=[],
            )

    def test_task_tool_schema_enum(self):
        state = DeepAgentState()
        specs = [
            SubAgentSpec(name="a", description="d", system_prompt="p"),
            SubAgentSpec(name="b", description="d", system_prompt="p"),
        ]
        tool = make_task_tool(
            specs=specs,
            state=state,
            parent_model="m",
            parent_user_tools=[],
        )
        enum = tool.parameters["properties"]["subagent_type"]["enum"]
        assert set(enum) == {"a", "b"}


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


class TestSummarizer:
    def _make_conv_with_messages(self, n: int):
        from prompture.agents.conversation import Conversation

        # Use a CountingDriver so Conversation can be built without a real model.
        driver = CountingDriver(prompt_tokens_sequence=[1])
        conv = Conversation(driver=driver, system_prompt="sys")
        for i in range(n):
            conv._messages.append({"role": "user", "content": f"msg{i}"})
            conv._messages.append({"role": "assistant", "content": f"reply{i}"})
        return conv

    def test_no_op_below_threshold(self):
        state = DeepAgentState()
        sd = StaticSummariserDriver()
        mw = SummarizationMiddleware(
            threshold_tokens=100,
            keep_last_n=4,
            state=state,
            summariser=sd,
        )
        conv = self._make_conv_with_messages(10)  # 20 msgs
        before = list(conv._messages)
        mw(conv, last_prompt_tokens=50)  # below threshold
        assert conv._messages == before
        assert state.summary_events == []
        assert sd.calls == 0

    def test_collapses_when_above_threshold(self):
        state = DeepAgentState()
        sd = StaticSummariserDriver(summary="The agent did stuff.")
        mw = SummarizationMiddleware(
            threshold_tokens=100,
            keep_last_n=4,
            state=state,
            summariser=sd,
        )
        conv = self._make_conv_with_messages(10)  # 20 msgs

        original_count = len(conv._messages)
        mw(conv, last_prompt_tokens=200)  # well above

        # After: [summary_msg, last 4 messages] = 5 total
        assert len(conv._messages) == 5
        assert "The agent did stuff." in conv._messages[0]["content"]
        # Tail preserved exactly
        original_tail = [{"role": "user", "content": f"msg{8}"} for _ in range(1)] + [
            {"role": "assistant", "content": f"reply{8}"},
            {"role": "user", "content": f"msg{9}"},
            {"role": "assistant", "content": f"reply{9}"},
        ]
        assert conv._messages[1:] == original_tail
        assert len(state.summary_events) == 1
        assert state.summary_events[0].summarized_message_count == original_count - 4

    def test_reentrancy_guard(self):
        """Hook should not recurse if invoked during summarization."""
        state = DeepAgentState()
        sd = StaticSummariserDriver()
        mw = SummarizationMiddleware(
            threshold_tokens=10,
            keep_last_n=2,
            state=state,
            summariser=sd,
        )
        conv = self._make_conv_with_messages(5)
        # Simulate that we are already summarising
        mw._summarising = True
        before = list(conv._messages)
        mw(conv, last_prompt_tokens=999)
        assert conv._messages == before
        mw._summarising = False

    def test_skips_when_history_too_small(self):
        state = DeepAgentState()
        sd = StaticSummariserDriver()
        mw = SummarizationMiddleware(
            threshold_tokens=10,
            keep_last_n=10,
            state=state,
            summariser=sd,
        )
        conv = self._make_conv_with_messages(2)  # only 4 msgs, less than keep_last_n + 1
        before = list(conv._messages)
        mw(conv, last_prompt_tokens=999)
        assert conv._messages == before
        assert sd.calls == 0


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


class TestPromptAssembly:
    def test_base_only(self):
        out = assemble_system_prompt(
            persona_text=None,
            enable_planning=False,
            enable_vfs=False,
            subagent_section=None,
        )
        assert out == BASE_DEEP_AGENT_PROMPT.strip()

    def test_with_planning(self):
        out = assemble_system_prompt(
            persona_text=None,
            enable_planning=True,
            enable_vfs=False,
            subagent_section=None,
        )
        assert BASE_DEEP_AGENT_PROMPT.strip() in out
        assert PLANNING_PROMPT_FRAGMENT.strip() in out
        assert "virtual filesystem" not in out.lower()

    def test_with_vfs(self):
        out = assemble_system_prompt(
            persona_text=None,
            enable_planning=False,
            enable_vfs=True,
            subagent_section=None,
        )
        assert VFS_PROMPT_FRAGMENT.strip() in out
        assert "write_todos" not in out.lower()

    def test_persona_prefixed(self):
        out = assemble_system_prompt(
            persona_text="You are SuperBot.",
            enable_planning=False,
            enable_vfs=False,
            subagent_section=None,
        )
        assert out.startswith("You are SuperBot.")
        assert BASE_DEEP_AGENT_PROMPT.strip() in out

    def test_subagent_section_listing(self):
        specs = [
            SubAgentSpec(name="alpha", description="Does A things.", system_prompt="p"),
            SubAgentSpec(name="beta", description="Does B things.", system_prompt="p"),
        ]
        section = format_subagent_section(specs)
        assert "alpha" in section
        assert "Does A things." in section
        assert "beta" in section

    def test_no_subagents_yields_empty(self):
        assert format_subagent_section([]) == ""


# ---------------------------------------------------------------------------
# End-to-end DeepAgent with mocked driver
# ---------------------------------------------------------------------------


def _resp(text: str, tool_calls: list[dict] | None = None) -> dict[str, Any]:
    return {
        "text": text,
        "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0},
        "tool_calls": tool_calls or [],
        "stop_reason": "tool_use" if tool_calls else "end_turn",
    }


class TestDeepAgentEndToEnd:
    def test_basic_construction_no_subagents(self):
        driver = MockToolDriver([_resp("hello")])
        agent = create_deep_agent(
            model="mock/m",
            driver=driver,
            enable_summarization=False,
        )
        assert isinstance(agent, DeepAgent)
        assert agent.deep_state.files == {}
        # System prompt includes base + planning + vfs by default
        assert "write_todos" in agent._system_prompt.lower() or "todos" in agent._system_prompt.lower()
        assert "virtual filesystem" in agent._system_prompt.lower()

    def test_planning_disabled_omits_tool_and_prompt_section(self):
        driver = MockToolDriver([_resp("ok")])
        agent = create_deep_agent(
            model="mock/m",
            driver=driver,
            enable_planning=False,
            enable_summarization=False,
        )
        tool_names = {t.name for t in agent._tools.definitions}
        assert "write_todos" not in tool_names
        assert "## Planning" not in agent._system_prompt

    def test_vfs_disabled_omits_tools_and_prompt_section(self):
        driver = MockToolDriver([_resp("ok")])
        agent = create_deep_agent(
            model="mock/m",
            driver=driver,
            enable_vfs=False,
            enable_summarization=False,
        )
        tool_names = {t.name for t in agent._tools.definitions}
        for forbidden in ("read_file", "write_file", "edit_file", "ls", "glob", "grep"):
            assert forbidden not in tool_names
        assert "virtual filesystem" not in agent._system_prompt.lower()

    def test_initial_files_seeded(self):
        driver = MockToolDriver([_resp("ok")])
        agent = create_deep_agent(
            model="mock/m",
            driver=driver,
            initial_files={"brief.md": "Research X."},
            enable_summarization=False,
        )
        assert agent.deep_state.files["brief.md"] == "Research X."

    def test_user_tool_name_collision_warns_and_drops(self):
        def write_todos(todos):
            """Shadow built-in."""
            return "user!"

        driver = MockToolDriver([_resp("ok")])
        with pytest.warns(UserWarning, match="shadow DeepAgent built-ins"):
            agent = create_deep_agent(
                model="mock/m",
                driver=driver,
                tools=[write_todos],
                enable_summarization=False,
            )
        # Built-in wins, user's write_todos was dropped.
        write_todos_def = agent._tools.get("write_todos")
        assert write_todos_def is not None
        result_msg = write_todos_def.function(
            todos=[{"content": "t", "status": "pending"}]
        )
        assert "Todo list updated" in result_msg

    def test_subagents_inject_task_tool_and_prompt_section(self):
        driver = MockToolDriver([_resp("ok")])
        spec = SubAgentSpec(
            name="helper",
            description="Helps with helper things.",
            system_prompt="You are a helper.",
        )
        agent = create_deep_agent(
            model="mock/m",
            driver=driver,
            subagents=[spec],
            enable_summarization=False,
        )
        tool_names = {t.name for t in agent._tools.definitions}
        assert "task" in tool_names
        assert "helper" in agent._system_prompt
        assert "Helps with helper things." in agent._system_prompt

    def test_run_with_write_todos_tool_call(self):
        # 1) model calls write_todos
        # 2) model returns final answer
        responses = [
            _resp(
                "",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "write_todos",
                        "arguments": {
                            "todos": [
                                {"content": "Find sources", "status": "in_progress"},
                                {"content": "Draft report", "status": "pending"},
                            ]
                        },
                    }
                ],
            ),
            _resp("Plan written; here is the answer."),
        ]
        driver = MockToolDriver(responses)
        agent = create_deep_agent(
            model="mock/m",
            driver=driver,
            enable_vfs=False,
            enable_summarization=False,
        )
        result = agent.run("Plan and answer.")
        assert isinstance(result, DeepAgentResult)
        assert result.output_text == "Plan written; here is the answer."
        assert len(result.todos) == 2
        assert result.todos[0].status == "in_progress"
        # Convenience aliases match state
        assert result.todos is result.deep_state.todos
        assert result.files == {}

    def test_run_with_vfs_tools(self):
        responses = [
            _resp(
                "",
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "write_file",
                        "arguments": {"file_path": "notes.md", "content": "hello"},
                    }
                ],
            ),
            _resp(
                "",
                tool_calls=[
                    {
                        "id": "c2",
                        "name": "read_file",
                        "arguments": {"file_path": "notes.md"},
                    }
                ],
            ),
            _resp("I read the file."),
        ]
        driver = MockToolDriver(responses)
        agent = create_deep_agent(
            model="mock/m",
            driver=driver,
            enable_planning=False,
            enable_summarization=False,
        )
        result = agent.run("Create and read a note.")
        assert result.files == {"notes.md": "hello"}
        assert "I read the file." in result.output_text

    def test_reset_clears_state(self):
        driver = MockToolDriver([_resp("ok")] * 4)
        agent = create_deep_agent(
            model="mock/m",
            driver=driver,
            initial_files={"x.md": "y"},
            enable_summarization=False,
        )
        agent.deep_state.todos.append(Todo(content="t"))
        agent.reset()
        assert agent.deep_state.files == {}
        assert agent.deep_state.todos == []

    def test_persona_and_system_prompt_conflict_raises(self):
        from prompture.agents.persona import Persona

        driver = MockToolDriver([_resp("ok")])
        with pytest.raises(ValueError, match="not both"):
            create_deep_agent(
                model="mock/m",
                driver=driver,
                persona=Persona(name="p", system_prompt="x"),
                system_prompt="y",
                enable_summarization=False,
            )

    def test_summarization_disabled_means_no_hook(self):
        driver = MockToolDriver([_resp("ok")])
        agent = create_deep_agent(
            model="mock/m",
            driver=driver,
            enable_summarization=False,
        )
        assert agent._before_turn_hook is None
        assert agent._summarizer is None

    def test_summarization_enabled_installs_hook(self):
        driver = MockToolDriver([_resp("ok")])
        agent = create_deep_agent(
            model="mock/m",
            driver=driver,
            enable_summarization=True,
            summarize_at_tokens=1_000,
            summarizer_model=StaticSummariserDriver(),
        )
        assert agent._before_turn_hook is not None
        assert agent._summarizer is not None


# ---------------------------------------------------------------------------
# End-to-end Conversation + before_turn integration
# ---------------------------------------------------------------------------


class TestConversationHook:
    def test_before_turn_invoked(self):
        from prompture.agents.conversation import Conversation

        calls: list[int] = []

        def hook(conv, last_pt):
            calls.append(last_pt)

        responses = [
            _resp(
                "",
                tool_calls=[
                    {"id": "c1", "name": "tool_x", "arguments": {}},
                ],
            ),
            _resp("final"),
        ]
        driver = MockToolDriver(responses)

        def tool_x() -> str:
            """A trivial tool."""
            return "ok"

        from prompture.agents.tools_schema import ToolRegistry

        reg = ToolRegistry()
        reg.register(tool_x)
        conv = Conversation(driver=driver, tools=reg, before_turn=hook)
        conv.ask("hi")
        # Hook fires before each driver call (2 calls total)
        assert len(calls) == 2
        # First call: no prior usage, so last_prompt_tokens=0
        assert calls[0] == 0
        # Second call: after one response with prompt_tokens=10
        assert calls[1] == 10

    def test_before_turn_can_mutate_messages(self):
        from prompture.agents.conversation import Conversation

        responses = [
            _resp(
                "",
                tool_calls=[{"id": "c1", "name": "tool_x", "arguments": {}}],
            ),
            _resp("done"),
        ]
        driver = MockToolDriver(responses)

        def tool_x() -> str:
            """A trivial tool."""
            return "ok"

        from prompture.agents.tools_schema import ToolRegistry

        reg = ToolRegistry()
        reg.register(tool_x)

        mutate_calls = []

        def hook(conv, last_pt):
            mutate_calls.append(len(conv._messages))
            if last_pt > 0 and len(conv._messages) > 1:
                # Replace history with a single user message
                conv._messages.clear()
                conv._messages.append({"role": "user", "content": "[summary]"})

        conv = Conversation(driver=driver, tools=reg, before_turn=hook)
        conv.ask("hello")
        # The mutation should have happened on the second turn
        assert any(c >= 1 for c in mutate_calls)
        # After the run, the synthetic summary message should still be there
        # (the assistant's final response is appended after it).
        assert any(m.get("content") == "[summary]" for m in conv._messages)
