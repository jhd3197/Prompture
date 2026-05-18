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
        """Claude auto mode skips permission prompts via the dangerous flag.

        Claude Code does not expose an intermediate non-interactive mode — the
        only way to suppress approval prompts is --dangerously-skip-permissions.
        """
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
            "--dangerously-skip-permissions",
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
            build_coding_agent_command("nonesuch", "do a thing", cwd=tmp_path)

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/aider"),
    )
    def test_aider_auto_and_model_command(self, _mock_resolve, tmp_path):
        """Aider takes the task via --message and skips prompts with --yes-always."""
        command = build_coding_agent_command(
            "aider",
            "rename foo to bar",
            cwd=tmp_path,
            approval_mode="auto",
            model="gpt-4o",
        )

        assert command.argv == [
            "/usr/local/bin/aider",
            "--model",
            "gpt-4o",
            "--yes-always",
            "--message",
            "rename foo to bar",
        ]

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/opencode"),
    )
    def test_opencode_run_subcommand(self, _mock_resolve, tmp_path):
        """OpenCode uses the `run` subcommand for non-interactive execution."""
        command = build_coding_agent_command(
            "opencode",
            "add a CHANGELOG entry",
            cwd=tmp_path,
            approval_mode="auto",
            model="anthropic/claude-sonnet-4",
        )

        assert command.argv == [
            "/usr/local/bin/opencode",
            "run",
            "--model",
            "anthropic/claude-sonnet-4",
            "add a CHANGELOG entry",
        ]

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/cursor-agent"),
    )
    def test_cursor_agent_print_mode(self, _mock_resolve, tmp_path):
        """Cursor Agent uses -p for non-interactive print mode."""
        command = build_coding_agent_command(
            "cursor-agent",
            "explain main.py",
            cwd=tmp_path,
            approval_mode="auto",
        )

        assert command.argv == [
            "/usr/local/bin/cursor-agent",
            "-p",
            "--force",
            "explain main.py",
        ]

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/crush"),
    )
    def test_crush_yolo_command(self, _mock_resolve, tmp_path):
        """Crush exposes an explicit --yolo flag for full auto mode."""
        command = build_coding_agent_command(
            "crush",
            "tidy this module",
            cwd=tmp_path,
            approval_mode="yolo",
        )

        assert "run" in command.argv
        assert "--yolo" in command.argv
        assert command.argv[-1] == "tidy this module"


class TestCodingAgentSpecs:
    """Registry-level sanity checks."""

    def test_registry_includes_all_quick_wins(self):
        from prompture.infra import CODING_AGENT_SPECS, get_coding_agent_spec

        for agent_id in ("claude", "codex", "gemini", "qwen", "aider", "opencode", "cursor-agent", "crush"):
            assert agent_id in CODING_AGENT_SPECS, f"missing spec: {agent_id}"
            assert get_coding_agent_spec(agent_id) is not None

    def test_qwen_uses_gemini_npm_package_shape(self):
        from prompture.infra import get_coding_agent_spec

        spec = get_coding_agent_spec("qwen")
        assert spec is not None
        assert spec.display_name == "Qwen Code"
        assert spec.npm_packages == ("@qwen-code/qwen-code",)

    def test_pip_only_agents_have_no_npm_package(self):
        """Aider, Cursor Agent, and Crush are not distributed via npm."""
        from prompture.infra import get_coding_agent_spec

        for agent_id in ("aider", "cursor-agent", "crush"):
            spec = get_coding_agent_spec(agent_id)
            assert spec is not None
            assert spec.npm_packages == ()

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
