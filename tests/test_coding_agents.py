import subprocess
from unittest.mock import patch

from prompture.infra import build_coding_agent_command, run_coding_agent


class TestCodingAgentCommands:
    """Tests for coding-agent command construction."""

    @patch("prompture.infra.coding_agents.resolve_coding_agent_binary", return_value="/usr/local/bin/codex")
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

    @patch("prompture.infra.coding_agents.resolve_coding_agent_binary", return_value="/usr/local/bin/codex")
    def test_codex_yolo_command(self, _mock_resolve, tmp_path):
        """Codex yolo mode uses the explicit dangerous bypass flag."""
        command = build_coding_agent_command(
            "codex",
            "refactor this module",
            cwd=tmp_path,
            approval_mode="yolo",
        )

        assert "--dangerously-bypass-approvals-and-sandbox" in command.argv

    @patch("prompture.infra.coding_agents.resolve_coding_agent_binary", return_value="/usr/local/bin/claude")
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

    @patch("prompture.infra.coding_agents.resolve_coding_agent_binary", return_value="/usr/local/bin/claude")
    def test_claude_yolo_command(self, _mock_resolve, tmp_path):
        """Claude yolo mode uses Claude Code's dangerous skip-permissions flag."""
        command = build_coding_agent_command(
            "claude",
            "make the requested edits",
            cwd=tmp_path,
            approval_mode="yolo",
        )

        assert "--dangerously-skip-permissions" in command.argv

    @patch("prompture.infra.coding_agents.resolve_coding_agent_binary", return_value="/usr/local/bin/gemini")
    def test_gemini_auto_command(self, _mock_resolve, tmp_path):
        """Gemini auto mode passes yes mode through to the CLI."""
        command = build_coding_agent_command(
            "gemini",
            "summarize the repo",
            cwd=tmp_path,
            approval_mode="auto",
        )

        assert command.argv == ["/usr/local/bin/gemini", "-y", "summarize the repo"]


class TestCodingAgentRunner:
    """Tests for running coding-agent commands without invoking real CLIs."""

    @patch("prompture.infra.coding_agents.subprocess.run")
    @patch("prompture.infra.coding_agents.resolve_coding_agent_binary", return_value="/usr/local/bin/codex")
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
