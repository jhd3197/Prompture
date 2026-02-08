"""Tests for ToolRegistry filtering methods (subset, filter, exclude)."""

from __future__ import annotations

import pytest

from prompture.agents.tools_schema import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_registry() -> ToolRegistry:
    """Create a registry with three tools for testing."""
    reg = ToolRegistry()

    @reg.tool
    def file_read(path: str) -> str:
        """Read a file."""
        return path

    @reg.tool
    def file_write(path: str, content: str) -> str:
        """Write a file."""
        return f"{path}: {content}"

    @reg.tool
    def python_execute(code: str) -> str:
        """Execute Python code."""
        return code

    return reg


# ---------------------------------------------------------------------------
# subset()
# ---------------------------------------------------------------------------


class TestSubset:
    def test_subset_returns_correct_tools(self):
        reg = _build_registry()
        sub = reg.subset({"file_read", "python_execute"})
        assert len(sub) == 2
        assert "file_read" in sub
        assert "python_execute" in sub
        assert "file_write" not in sub

    def test_subset_returns_new_registry(self):
        reg = _build_registry()
        sub = reg.subset({"file_read"})
        assert sub is not reg
        assert len(reg) == 3  # original unchanged

    def test_subset_with_list(self):
        reg = _build_registry()
        sub = reg.subset(["file_read"])
        assert len(sub) == 1
        assert "file_read" in sub

    def test_subset_unknown_name_raises_keyerror(self):
        reg = _build_registry()
        with pytest.raises(KeyError, match="Unknown tools"):
            reg.subset({"file_read", "nonexistent"})

    def test_subset_all_unknown_raises_keyerror(self):
        reg = _build_registry()
        with pytest.raises(KeyError, match="Unknown tools"):
            reg.subset({"foo", "bar"})


# ---------------------------------------------------------------------------
# filter()
# ---------------------------------------------------------------------------


class TestFilter:
    def test_filter_with_predicate(self):
        reg = _build_registry()
        sub = reg.filter(lambda td: "file_" in td.name)
        assert len(sub) == 2
        assert "file_read" in sub
        assert "file_write" in sub
        assert "python_execute" not in sub

    def test_filter_returns_new_registry(self):
        reg = _build_registry()
        sub = reg.filter(lambda td: True)
        assert sub is not reg
        assert len(sub) == 3

    def test_filter_no_match(self):
        reg = _build_registry()
        sub = reg.filter(lambda td: False)
        assert len(sub) == 0

    def test_filter_by_description(self):
        reg = _build_registry()
        sub = reg.filter(lambda td: "Execute" in td.description)
        assert len(sub) == 1
        assert "python_execute" in sub


# ---------------------------------------------------------------------------
# exclude()
# ---------------------------------------------------------------------------


class TestExclude:
    def test_exclude_removes_named_tools(self):
        reg = _build_registry()
        sub = reg.exclude({"python_execute"})
        assert len(sub) == 2
        assert "python_execute" not in sub
        assert "file_read" in sub
        assert "file_write" in sub

    def test_exclude_returns_new_registry(self):
        reg = _build_registry()
        sub = reg.exclude({"file_read"})
        assert sub is not reg
        assert len(reg) == 3

    def test_exclude_missing_name_is_silent(self):
        reg = _build_registry()
        sub = reg.exclude({"nonexistent"})
        assert len(sub) == 3

    def test_exclude_all(self):
        reg = _build_registry()
        sub = reg.exclude({"file_read", "file_write", "python_execute"})
        assert len(sub) == 0

    def test_exclude_with_list(self):
        reg = _build_registry()
        sub = reg.exclude(["file_read", "file_write"])
        assert len(sub) == 1
        assert "python_execute" in sub


# ---------------------------------------------------------------------------
# Tool execution on filtered registries
# ---------------------------------------------------------------------------


class TestFilteredExecution:
    def test_subset_tools_are_executable(self):
        reg = _build_registry()
        sub = reg.subset({"file_read"})
        result = sub.execute("file_read", {"path": "/tmp/test"})
        assert result == "/tmp/test"

    def test_filter_tools_are_executable(self):
        reg = _build_registry()
        sub = reg.filter(lambda td: td.name == "python_execute")
        result = sub.execute("python_execute", {"code": "print('hi')"})
        assert result == "print('hi')"

    def test_exclude_tools_are_executable(self):
        reg = _build_registry()
        sub = reg.exclude({"file_read"})
        result = sub.execute("file_write", {"path": "/tmp/f", "content": "data"})
        assert result == "/tmp/f: data"
