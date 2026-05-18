import subprocess
from unittest.mock import patch

from prompture.infra import build_coding_agent_command, run_coding_agent
from prompture.infra.discovery import CodingAgentExecutable


def _executable(path: str) -> tuple[CodingAgentExecutable, bool, None]:
    return CodingAgentExecutable(binary=path, argv=(path,), display=path), True, None


class TestCodingAgentCommands:
    """Tests for coding-agent command construction."""

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/codex"),
    )
    def test_codex_auto_approve_command(self, _mock_resolve, tmp_path):
        """Codex auto mode uses non-interactive exec with workspace approvals disabled."""
        command = build_coding_agent_command(
            "codex",
            "fix the tests",
            cwd=tmp_path,
            approval_mode="auto",
        )

        assert command.argv == [
            "/usr/local/bin/codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--ask-for-approval",
            "never",
            "fix the tests",
        ]

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/codex"),
    )
    def test_codex_yolo_command(self, _mock_resolve, tmp_path):
        """Codex yolo mode uses the explicit dangerous bypass flag."""
        command = build_coding_agent_command(
            "codex",
            "refactor this module",
            cwd=tmp_path,
            approval_mode="yolo",
        )

        assert "--dangerously-bypass-approvals-and-sandbox" in command.argv

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/claude"),
    )
    def test_claude_auto_and_model_command(self, _mock_resolve, tmp_path):
        """Claude auto mode runs non-interactively with dontAsk permissions."""
        command = build_coding_agent_command(
            "claude",
            "review this repository",
            cwd=tmp_path,
            approval_mode="auto",
            model="sonnet",
        )

        assert command.argv == [
            "/usr/local/bin/claude",
            "--print",
            "--output-format",
            "text",
            "--model",
            "sonnet",
            "--permission-mode",
            "dontAsk",
            "review this repository",
        ]

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/claude"),
    )
    def test_claude_yolo_command(self, _mock_resolve, tmp_path):
        """Claude yolo mode uses Claude Code's dangerous skip-permissions flag."""
        command = build_coding_agent_command(
            "claude",
            "make the requested edits",
            cwd=tmp_path,
            approval_mode="yolo",
        )

        assert "--dangerously-skip-permissions" in command.argv

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/gemini"),
    )
    def test_gemini_auto_command(self, _mock_resolve, tmp_path):
        """Gemini auto mode passes yes mode through to the CLI."""
        command = build_coding_agent_command(
            "gemini",
            "summarize the repo",
            cwd=tmp_path,
            approval_mode="auto",
        )

        assert command.argv == ["/usr/local/bin/gemini", "-y", "summarize the repo"]

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/qwen"),
    )
    def test_qwen_auto_and_model_command(self, _mock_resolve, tmp_path):
        """Qwen Code is a gemini-cli fork and uses the same arg shape."""
        command = build_coding_agent_command(
            "qwen",
            "explain this file",
            cwd=tmp_path,
            approval_mode="auto",
            model="qwen3-coder-plus",
        )

        assert command.argv == [
            "/usr/local/bin/qwen",
            "--model",
            "qwen3-coder-plus",
            "-y",
            "explain this file",
        ]

    def test_unknown_agent_rejected(self, tmp_path):
        """Unknown agent ids fail fast with a helpful list of valid options."""
        import pytest

        with pytest.raises(ValueError, match="Unsupported coding agent"):
            build_coding_agent_command("aider", "do a thing", cwd=tmp_path)


class TestCodingAgentSpecs:
    """Registry-level sanity checks."""

    def test_registry_includes_qwen(self):
        from prompture.infra import CODING_AGENT_SPECS, get_coding_agent_spec

        assert "qwen" in CODING_AGENT_SPECS
        spec = get_coding_agent_spec("qwen")
        assert spec is not None
        assert spec.display_name == "Qwen Code"
        assert spec.default_binary == "qwen"
        assert spec.npm_packages == ("@qwen-code/qwen-code",)

    def test_get_spec_returns_none_for_unknown(self):
        from prompture.infra import get_coding_agent_spec

        assert get_coding_agent_spec("nonesuch") is None


class TestCodingAgentRunner:
    """Tests for running coding-agent commands without invoking real CLIs."""

    @patch("prompture.infra.coding_agents.subprocess.run")
    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/codex"),
    )
    def test_run_coding_agent_returns_result(self, _mock_resolve, mock_run, tmp_path):
        """Runner captures subprocess output and metadata."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["/usr/local/bin/codex"],
            returncode=0,
            stdout="done",
            stderr="",
        )

        result = run_coding_agent("codex", "fix bug", cwd=tmp_path, approval_mode="auto")

        assert result.ok is True
        assert result.agent == "codex"
        assert result.output == "done"
        assert result.returncode == 0
        mock_run.assert_called_once()

    @patch("prompture.infra.coding_agents.subprocess.run")
    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=(
            None,
            False,
            "node: not found",
        ),
    )
    def test_run_coding_agent_verifies_before_running_task(self, _mock_resolve, mock_run, tmp_path):
        """Runner fails early when PATH resolves to a broken CLI shim."""
        result = run_coding_agent("gemini", "fix bug", cwd=tmp_path, approval_mode="auto")

        assert result.ok is False
        assert result.returncode == -1
        assert result.output == "gemini CLI health check failed: node: not found"
        mock_run.assert_not_called()

    @patch("prompture.infra.coding_agents.subprocess.run")
    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/codex"),
    )
    def test_run_coding_agent_can_skip_verification(self, _mock_resolve, mock_run, tmp_path):
        """Callers can skip the preflight check when they need raw subprocess behavior."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["/usr/local/bin/codex"],
            returncode=0,
            stdout="done",
            stderr="",
        )

        result = run_coding_agent("codex", "fix bug", cwd=tmp_path, verify_binary=False)

        assert result.ok is True
        assert result.output == "done"
        mock_run.assert_called_once()
