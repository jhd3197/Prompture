"""Tests for the history module."""

import json
import time

import pytest

from prompture.agent_types import AgentResult, AgentState, AgentStep, StepType
from prompture.history import (
    calculate_cost_breakdown,
    export_result_json,
    filter_steps,
    get_tool_call_summary,
    result_to_dict,
    search_messages,
)


@pytest.fixture
def sample_steps():
    """Create sample steps for testing."""
    now = time.time()
    return [
        AgentStep(
            step_type=StepType.think,
            timestamp=now,
            content="Thinking about the problem...",
        ),
        AgentStep(
            step_type=StepType.tool_call,
            timestamp=now + 1,
            content="",
            tool_name="search",
            tool_args={"query": "python"},
        ),
        AgentStep(
            step_type=StepType.tool_result,
            timestamp=now + 2,
            content="Found 10 results",
            tool_name="search",
        ),
        AgentStep(
            step_type=StepType.tool_call,
            timestamp=now + 3,
            content="",
            tool_name="calculate",
            tool_args={"x": 1, "y": 2},
        ),
        AgentStep(
            step_type=StepType.tool_result,
            timestamp=now + 4,
            content="Result: 3",
            tool_name="calculate",
        ),
        AgentStep(
            step_type=StepType.output,
            timestamp=now + 5,
            content="The answer is 3",
        ),
    ]


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Let me calculate that.", "tool_calls": [
            {"id": "call_1", "function": {"name": "calculate", "arguments": '{"x": 2, "y": 2}'}}
        ]},
        {"role": "tool", "content": "4", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "Thanks!"},
        {"role": "assistant", "content": "You're welcome!"},
    ]


@pytest.fixture
def sample_result(sample_steps, sample_messages):
    """Create a sample AgentResult for testing."""
    return AgentResult(
        output="The answer is 3",
        output_text="The answer is 3",
        messages=sample_messages,
        usage={"total_tokens": 100},
        steps=sample_steps,
        all_tool_calls=[
            {"name": "search", "arguments": {"query": "python"}, "id": "call_1"},
            {"name": "calculate", "arguments": {"x": 1, "y": 2}, "id": "call_2"},
        ],
        state=AgentState.idle,
        run_usage={
            "prompt_tokens": 50,
            "completion_tokens": 50,
            "total_tokens": 100,
            "prompt_cost": 0.001,
            "completion_cost": 0.002,
            "total_cost": 0.003,
            "call_count": 2,
            "error_count": 0,
        },
    )


class TestFilterSteps:
    """Tests for filter_steps function."""

    def test_filter_by_type(self, sample_steps):
        """Test filtering by step type."""
        tool_calls = filter_steps(sample_steps, step_type=StepType.tool_call)
        assert len(tool_calls) == 2
        assert all(s.step_type == StepType.tool_call for s in tool_calls)

    def test_filter_by_multiple_types(self, sample_steps):
        """Test filtering by multiple step types."""
        filtered = filter_steps(
            sample_steps,
            step_type=[StepType.tool_call, StepType.tool_result]
        )
        assert len(filtered) == 4

    def test_filter_by_tool_name(self, sample_steps):
        """Test filtering by tool name."""
        search_steps = filter_steps(sample_steps, tool_name="search")
        assert len(search_steps) == 2  # call + result

    def test_filter_by_timestamp(self, sample_steps):
        """Test filtering by timestamp range."""
        now = sample_steps[0].timestamp

        # Get steps after the first one
        after = filter_steps(sample_steps, after_timestamp=now)
        assert len(after) == 5  # All except the first

    def test_combined_filters(self, sample_steps):
        """Test combining multiple filters."""
        now = sample_steps[0].timestamp

        filtered = filter_steps(
            sample_steps,
            step_type=StepType.tool_call,
            after_timestamp=now + 2,
        )
        assert len(filtered) == 1
        assert filtered[0].tool_name == "calculate"


class TestSearchMessages:
    """Tests for search_messages function."""

    def test_filter_by_role(self, sample_messages):
        """Test filtering by role."""
        assistant = search_messages(sample_messages, role="assistant")
        assert len(assistant) == 3

    def test_filter_by_multiple_roles(self, sample_messages):
        """Test filtering by multiple roles."""
        filtered = search_messages(sample_messages, role=["user", "assistant"])
        assert len(filtered) == 5

    def test_filter_by_content(self, sample_messages):
        """Test filtering by content."""
        with_answer = search_messages(sample_messages, content_contains="answer")
        assert len(with_answer) == 1
        assert "4" in with_answer[0]["content"]

    def test_filter_by_tool_calls(self, sample_messages):
        """Test filtering by presence of tool calls."""
        with_tools = search_messages(sample_messages, has_tool_calls=True)
        assert len(with_tools) == 1
        assert "tool_calls" in with_tools[0]

        without_tools = search_messages(sample_messages, has_tool_calls=False)
        assert len(without_tools) == 5

    def test_case_insensitive_content_search(self, sample_messages):
        """Test that content search is case-insensitive."""
        results = search_messages(sample_messages, content_contains="ANSWER")
        assert len(results) == 1


class TestGetToolCallSummary:
    """Tests for get_tool_call_summary function."""

    def test_basic_summary(self, sample_result):
        """Test basic tool call summary."""
        summary = get_tool_call_summary(sample_result)

        assert len(summary) == 2
        assert summary[0]["name"] == "search"
        assert summary[0]["arguments"] == {"query": "python"}
        assert summary[1]["name"] == "calculate"

    def test_includes_result(self, sample_result):
        """Test that results are included when available."""
        summary = get_tool_call_summary(sample_result)

        assert "result" in summary[0]
        assert summary[0]["result"] == "Found 10 results"


class TestCalculateCostBreakdown:
    """Tests for calculate_cost_breakdown function."""

    def test_basic_breakdown(self, sample_result):
        """Test basic cost breakdown."""
        breakdown = calculate_cost_breakdown(sample_result.run_usage)

        assert breakdown["prompt_tokens"] == 50
        assert breakdown["completion_tokens"] == 50
        assert breakdown["total_tokens"] == 100
        assert breakdown["total_cost"] == 0.003
        assert breakdown["call_count"] == 2
        assert breakdown["error_count"] == 0

    def test_empty_usage(self):
        """Test with empty usage dict."""
        breakdown = calculate_cost_breakdown({})

        assert breakdown["prompt_tokens"] == 0
        assert breakdown["total_cost"] == 0.0
        assert breakdown["call_count"] == 0


class TestExportResultJson:
    """Tests for export_result_json function."""

    def test_basic_export(self, sample_result):
        """Test basic JSON export."""
        json_str = export_result_json(sample_result)

        # Verify it's valid JSON
        data = json.loads(json_str)

        assert data["output_text"] == "The answer is 3"
        assert "steps" in data
        assert "messages" in data
        assert "exported_at" in data

    def test_export_without_messages(self, sample_result):
        """Test export without messages."""
        json_str = export_result_json(sample_result, include_messages=False)
        data = json.loads(json_str)

        assert "messages" not in data

    def test_steps_serialization(self, sample_result):
        """Test that steps are properly serialized."""
        json_str = export_result_json(sample_result)
        data = json.loads(json_str)

        assert len(data["steps"]) == 6
        assert data["steps"][0]["step_type"] == "think"


class TestResultToDict:
    """Tests for result_to_dict function."""

    def test_basic_conversion(self, sample_result):
        """Test basic dictionary conversion."""
        d = result_to_dict(sample_result)

        assert d["output_text"] == "The answer is 3"
        assert d["state"] == "idle"
        assert len(d["steps"]) == 6

    def test_without_messages(self, sample_result):
        """Test conversion without messages."""
        d = result_to_dict(sample_result, include_messages=False)

        assert "messages" not in d


class TestAgentResultMethods:
    """Tests for AgentResult.to_dict() and export_json() methods."""

    def test_to_dict_method(self, sample_result):
        """Test AgentResult.to_dict() method."""
        d = sample_result.to_dict()

        assert d["output_text"] == "The answer is 3"
        assert "steps" in d
        assert "messages" in d

    def test_export_json_method(self, sample_result):
        """Test AgentResult.export_json() method."""
        json_str = sample_result.export_json()

        # Verify it's valid JSON
        data = json.loads(json_str)
        assert data["output_text"] == "The answer is 3"

    def test_export_json_without_messages(self, sample_result):
        """Test export_json without messages."""
        json_str = sample_result.export_json(include_messages=False)
        data = json.loads(json_str)

        assert "messages" not in data
