"""Tests for ``prompture.tools.code_exec.PythonSandboxTool``.

Requires the ``sandbox`` extra (tukuy).  Tests are skipped at module
load time when tukuy is not importable.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

tukuy = pytest.importorskip("tukuy", reason="sandbox extra not installed")
from tukuy.analysis import RiskLevel  # noqa: E402

from prompture import (  # noqa: E402
    ApprovalRequired,
    PythonSandboxTool,
    ToolRegistry,
    python_execute_tool,
)

# ---------------------------------------------------------------------------
# ToolDefinition shape
# ---------------------------------------------------------------------------


def test_to_tool_definition_shape():
    td = PythonSandboxTool().to_tool_definition()
    assert td.name == "python_execute"
    assert "sandbox" in td.description.lower()
    assert td.parameters["type"] == "object"
    assert "code" in td.parameters["properties"]
    assert td.parameters["required"] == ["code"]


def test_custom_tool_name_and_description():
    td = PythonSandboxTool(
        tool_name="run_py",
        tool_description="My custom desc.",
    ).to_tool_definition()
    assert td.name == "run_py"
    assert td.description == "My custom desc."


def test_python_execute_tool_shortcut_returns_definition():
    td = python_execute_tool()
    assert td.name == "python_execute"
    assert callable(td.function)


def test_registers_on_tool_registry():
    tool = PythonSandboxTool()
    reg = ToolRegistry()
    td = tool.register_on(reg)
    assert td.name in reg
    assert reg.get("python_execute") is td


# ---------------------------------------------------------------------------
# Happy path execution
# ---------------------------------------------------------------------------


def test_execute_simple_print():
    out = PythonSandboxTool()("print('hello')")
    assert "hello" in out


def test_execute_safe_import_math():
    out = PythonSandboxTool().execute("import math; print(math.sqrt(81))")
    assert "9.0" in out


def test_execute_via_registry():
    reg = ToolRegistry()
    PythonSandboxTool().register_on(reg)
    out = reg.execute("python_execute", {"code": "print(2 + 2)"})
    assert "4" in out


def test_no_output_returns_friendly_message():
    out = PythonSandboxTool()("x = 1")
    assert "no output" in out.lower()


def test_output_truncation():
    code = "print('A' * 5000)"
    out = PythonSandboxTool(max_output_chars=100)(code)
    assert "truncated" in out
    # Body trimmed to roughly the cap (plus newline + truncation suffix).
    assert len(out) < 5000


# ---------------------------------------------------------------------------
# AST risk gate
# ---------------------------------------------------------------------------


def test_high_risk_import_raises_approval_required():
    tool = PythonSandboxTool()  # default threshold = HIGH
    with pytest.raises(ApprovalRequired) as exc_info:
        tool("import os; print(os.getcwd())")
    assert exc_info.value.tool_name == "python_execute"
    assert exc_info.value.details["risk_level"] in ("high", "critical")
    assert "import" in " ".join(exc_info.value.details["reasons"]).lower()
    assert "code_hash" in exc_info.value.details


def test_threshold_critical_lets_high_risk_through_to_runtime():
    tool = PythonSandboxTool(risk_threshold=RiskLevel.CRITICAL)
    # AST gate bypassed; tukuy runtime restriction blocks os.
    out = tool("import os; print(os.getcwd())")
    assert out.startswith("Error:")
    assert "not allowed" in out.lower() or "blocked" in out.lower()


def test_auto_approve_bypasses_ast_gate():
    tool = PythonSandboxTool(auto_approve=True)
    # AST gate bypassed; tukuy runtime restriction still blocks os.
    out = tool("import os")
    assert out.startswith("Error:")


def test_mark_approved_is_one_shot():
    tool = PythonSandboxTool()
    code = "import os; print('done')"

    # First call gated.
    with pytest.raises(ApprovalRequired):
        tool(code)

    tool.mark_approved(code)
    # Gate bypassed → runtime block kicks in (os is always-blocked).
    out = tool(code)
    assert out.startswith("Error:")

    # Second call gates again — one-shot consumed.
    with pytest.raises(ApprovalRequired):
        tool(code)


def test_revoke_approval():
    tool = PythonSandboxTool()
    code = "import os"
    tool.mark_approved(code)
    tool.revoke_approval(code)
    with pytest.raises(ApprovalRequired):
        tool(code)


# ---------------------------------------------------------------------------
# Syntax error handling
# ---------------------------------------------------------------------------


def test_syntax_error_returns_error_string_not_approval():
    out = PythonSandboxTool()("def x(:\n  pass")
    assert out.startswith("Error: Syntax error")


# ---------------------------------------------------------------------------
# Runtime restrictions still apply
# ---------------------------------------------------------------------------


def test_open_blocked_when_no_paths_configured():
    tool = PythonSandboxTool(auto_approve=True)
    out = tool("open('/etc/passwd').read()")
    assert "Error" in out
    assert "open" in out


def test_read_path_whitelist_allows_listed_directory():
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "data.txt")
        with open(fp, "w", encoding="utf-8") as f:
            f.write("PAYLOAD")

        tool = PythonSandboxTool(
            allowed_read_paths=[td],
            auto_approve=True,
        )
        out = tool(f"print(open({fp!r}).read())")
        assert "PAYLOAD" in out


def test_read_path_outside_whitelist_denied():
    with tempfile.TemporaryDirectory() as allowed_td:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as outside_file:
            outside_file.write("SECRET")
            outside_path = outside_file.name
        try:
            tool = PythonSandboxTool(
                allowed_read_paths=[allowed_td],
                auto_approve=True,
            )
            out = tool(f"print(open({outside_path!r}).read())")
            assert "Error" in out
        finally:
            os.unlink(outside_path)


# ---------------------------------------------------------------------------
# Missing tukuy
# ---------------------------------------------------------------------------


def test_helpful_error_when_tukuy_missing(monkeypatch):
    """Constructor raises a clear install-hint when tukuy is unavailable."""
    # Pretend tukuy.sandbox is missing by removing it from sys.modules and
    # blocking the import.
    blocked = {"tukuy.sandbox", "tukuy.analysis"}
    monkeypatch.setattr(
        "builtins.__import__",
        _make_blocking_import(blocked, sys.modules.copy()),
    )
    # Drop any cached tukuy submodules so the patched __import__ is consulted.
    for k in list(sys.modules):
        if k.startswith("tukuy"):
            monkeypatch.delitem(sys.modules, k, raising=False)

    with pytest.raises(ImportError) as exc_info:
        PythonSandboxTool()
    assert "prompture[sandbox]" in str(exc_info.value)


def _make_blocking_import(blocked: set[str], baseline: dict):
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in blocked or any(name.startswith(b + ".") for b in blocked):
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, globals, locals, fromlist, level)

    return fake_import
