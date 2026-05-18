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


class TestStructuredOutput:
    """Tests for output_format='json' wiring."""

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/claude"),
    )
    def test_claude_json_mode_adds_stream_json_flags(self, _mock_resolve, tmp_path):
        from prompture.infra import build_coding_agent_command

        command = build_coding_agent_command(
            "claude",
            "summarize this file",
            cwd=tmp_path,
            output_format="json",
        )
        assert "--output-format" in command.argv
        idx = command.argv.index("--output-format")
        assert command.argv[idx + 1] == "stream-json"
        assert "--verbose" in command.argv

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/gemini"),
    )
    def test_json_mode_rejected_for_unsupported_agent(self, _mock_resolve, tmp_path):
        """Gemini has no parse_events yet, so requesting json should fail loudly."""
        import pytest

        from prompture.infra import build_coding_agent_command

        with pytest.raises(ValueError, match="does not support output_format='json'"):
            build_coding_agent_command(
                "gemini",
                "do a thing",
                cwd=tmp_path,
                output_format="json",
            )

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/codex"),
    )
    def test_codex_json_mode_adds_json_flag(self, _mock_resolve, tmp_path):
        from prompture.infra import build_coding_agent_command

        command = build_coding_agent_command(
            "codex",
            "do a thing",
            cwd=tmp_path,
            approval_mode="auto",
            output_format="json",
        )
        assert "--json" in command.argv
        assert command.argv[-1] == "do a thing"

    @patch("prompture.infra.coding_agents.subprocess.run")
    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/claude"),
    )
    def test_claude_json_mode_parses_events_and_cost(self, _mock_resolve, mock_run, tmp_path):
        """run_coding_agent with output_format='json' populates events + cost."""
        import json as _json

        stream_lines = "\n".join(
            [
                _json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}),
                _json.dumps(
                    {
                        "type": "assistant",
                        "message": {"content": [{"type": "text", "text": "Looking at this..."}]},
                    }
                ),
                _json.dumps(
                    {
                        "type": "assistant",
                        "message": {"content": [{"type": "tool_use", "name": "Read", "input": {"path": "a.py"}}]},
                    }
                ),
                _json.dumps(
                    {
                        "type": "result",
                        "is_error": False,
                        "result": "Summarised the file.",
                        "total_cost_usd": 0.0123,
                        "duration_ms": 4500,
                        "usage": {"input_tokens": 1500, "output_tokens": 800},
                    }
                ),
            ]
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=["/usr/local/bin/claude"],
            returncode=0,
            stdout=stream_lines,
            stderr="",
        )

        result = run_coding_agent(
            "claude",
            "summarize a.py",
            cwd=tmp_path,
            approval_mode="auto",
            output_format="json",
        )

        assert result.ok is True
        assert result.cost_usd == 0.0123
        assert result.input_tokens == 1500
        assert result.output_tokens == 800
        types = [e.type for e in result.events]
        assert types == ["system", "message", "tool_call", "done"]

    @patch("prompture.infra.coding_agents.subprocess.run")
    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/claude"),
    )
    def test_text_mode_leaves_events_empty(self, _mock_resolve, mock_run, tmp_path):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["/usr/local/bin/claude"],
            returncode=0,
            stdout="plain text reply",
            stderr="",
        )

        result = run_coding_agent("claude", "hi", cwd=tmp_path, approval_mode="auto")

        assert result.events == ()
        assert result.cost_usd is None
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.output == "plain text reply"


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


class TestStreaming:
    """Tests for astream_coding_agent."""

    @staticmethod
    def _fake_proc(stdout_lines: list[str], returncode: int = 0, stderr: bytes = b""):
        class _FakeStdout:
            def __init__(self, lines):
                self._iter = iter(lines)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration from None

        class _FakeStderr:
            def __init__(self, data: bytes):
                self._data = data

            async def read(self) -> bytes:
                return self._data

        class _FakeProc:
            def __init__(self):
                self.stdout = _FakeStdout([(line + "\n").encode() for line in stdout_lines])
                self.stderr = _FakeStderr(stderr)
                self.returncode = None
                self._rc = returncode

            async def wait(self):
                self.returncode = self._rc
                return self._rc

            def terminate(self):
                pass

            def kill(self):
                pass

        async def _factory(*args, **kwargs):
            return _FakeProc()

        return _factory

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/claude"),
    )
    def test_astream_yields_typed_events(self, _mock_resolve, tmp_path):
        import asyncio
        import json as _json

        from prompture.infra import astream_coding_agent

        lines = [
            _json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}),
            _json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "Hi"}]},
                }
            ),
            _json.dumps(
                {
                    "type": "result",
                    "is_error": False,
                    "result": "ok",
                    "total_cost_usd": 0.002,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            ),
        ]

        async def _run():
            collected = []
            with patch(
                "prompture.infra.coding_agents.asyncio.create_subprocess_exec",
                side_effect=self._fake_proc(lines),
            ):
                async for ev in astream_coding_agent(
                    "claude",
                    "hi",
                    cwd=tmp_path,
                    approval_mode="auto",
                ):
                    collected.append(ev)
            return collected

        events = asyncio.run(_run())

        assert [e.type for e in events] == ["system", "message", "done"]
        assert events[-1].cost_usd == 0.002

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/claude"),
    )
    def test_astream_yields_error_on_nonzero_exit(self, _mock_resolve, tmp_path):
        import asyncio

        from prompture.infra import astream_coding_agent

        async def _run():
            collected = []
            with patch(
                "prompture.infra.coding_agents.asyncio.create_subprocess_exec",
                side_effect=self._fake_proc([], returncode=1, stderr=b"boom"),
            ):
                async for ev in astream_coding_agent(
                    "claude",
                    "x",
                    cwd=tmp_path,
                    approval_mode="auto",
                ):
                    collected.append(ev)
            return collected

        events = asyncio.run(_run())
        assert len(events) == 1
        assert events[0].type == "error"
        assert "exited with code 1" in (events[0].error or "")

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=(None, False, "node: not found"),
    )
    def test_astream_yields_error_on_health_check_failure(self, _mock_resolve, tmp_path):
        import asyncio

        from prompture.infra import astream_coding_agent

        async def _run():
            collected = []
            async for ev in astream_coding_agent(
                "claude",
                "x",
                cwd=tmp_path,
                approval_mode="auto",
            ):
                collected.append(ev)
            return collected

        events = asyncio.run(_run())
        assert [e.type for e in events] == ["error"]
        assert "health check failed" in (events[0].error or "")

    def test_astream_rejects_unsupported_agent(self, tmp_path):
        """Agents without a parser (e.g. gemini today) cannot be streamed."""
        import asyncio

        import pytest

        from prompture.infra import astream_coding_agent

        async def _run():
            async for _ev in astream_coding_agent("gemini", "x", cwd=tmp_path):
                pass

        with pytest.raises(ValueError, match="output_format='json'"):
            asyncio.run(_run())


class TestSettingsAgentPaths:
    """Tests for picking up agent binary paths from settings/env."""

    @patch("prompture.infra.coding_agents.resolve_coding_agent_executable")
    def test_settings_binary_is_used_when_no_explicit_path(self, mock_resolve, tmp_path):
        """A coding_agent_bin_<id> setting is honoured by resolve when no kwarg given."""
        from prompture.infra import build_coding_agent_command
        from prompture.infra.settings import settings as _s

        exe = CodingAgentExecutable(
            binary="/opt/aider/aider",
            argv=("/opt/aider/aider",),
            display="/opt/aider/aider",
        )
        mock_resolve.return_value = (exe, True, None)

        original = _s.coding_agent_bin_aider
        _s.coding_agent_bin_aider = "/opt/aider/aider"
        try:
            command = build_coding_agent_command(
                "aider",
                "do a thing",
                cwd=tmp_path,
                approval_mode="auto",
            )
        finally:
            _s.coding_agent_bin_aider = original

        # Resolver was asked for the path the user configured in settings.
        assert mock_resolve.call_args.args[1] == "/opt/aider/aider"
        assert command.argv[0] == "/opt/aider/aider"

    @patch("prompture.infra.coding_agents.resolve_coding_agent_executable")
    def test_explicit_kwarg_overrides_settings(self, mock_resolve, tmp_path):
        """An explicit agent_paths kwarg wins over the settings default."""
        from prompture.infra import build_coding_agent_command
        from prompture.infra.settings import settings as _s

        exe = CodingAgentExecutable(
            binary="/from/kwarg/aider",
            argv=("/from/kwarg/aider",),
            display="/from/kwarg/aider",
        )
        mock_resolve.return_value = (exe, True, None)

        original = _s.coding_agent_bin_aider
        _s.coding_agent_bin_aider = "/from/settings/aider"
        try:
            build_coding_agent_command(
                "aider",
                "task",
                cwd=tmp_path,
                approval_mode="auto",
                agent_paths={"aider": "/from/kwarg/aider"},
            )
        finally:
            _s.coding_agent_bin_aider = original

        assert mock_resolve.call_args.args[1] == "/from/kwarg/aider"

    @patch("prompture.infra.coding_agents.resolve_coding_agent_executable")
    def test_cursor_agent_hyphen_id_maps_to_underscore_setting(self, mock_resolve, tmp_path):
        from prompture.infra import build_coding_agent_command
        from prompture.infra.settings import settings as _s

        exe = CodingAgentExecutable(
            binary="/opt/cursor-agent",
            argv=("/opt/cursor-agent",),
            display="/opt/cursor-agent",
        )
        mock_resolve.return_value = (exe, True, None)

        original = _s.coding_agent_bin_cursor_agent
        _s.coding_agent_bin_cursor_agent = "/opt/cursor-agent"
        try:
            build_coding_agent_command(
                "cursor-agent",
                "task",
                cwd=tmp_path,
                approval_mode="auto",
            )
        finally:
            _s.coding_agent_bin_cursor_agent = original

        assert mock_resolve.call_args.args[1] == "/opt/cursor-agent"


class TestSessionCapture:
    """Tests for recording coding-agent runs into UsageSession."""

    @patch("prompture.infra.coding_agents.subprocess.run")
    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/claude"),
    )
    def test_session_receives_cost_and_tokens(self, _mock_resolve, mock_run, tmp_path):
        import json as _json

        from prompture.infra import UsageSession

        stream = "\n".join(
            [
                _json.dumps({"type": "system", "subtype": "init"}),
                _json.dumps(
                    {
                        "type": "result",
                        "is_error": False,
                        "result": "ok",
                        "total_cost_usd": 0.05,
                        "duration_ms": 1000,
                        "usage": {"input_tokens": 200, "output_tokens": 80},
                    }
                ),
            ]
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=["/usr/local/bin/claude"],
            returncode=0,
            stdout=stream,
            stderr="",
        )

        sess = UsageSession()
        result = run_coding_agent(
            "claude",
            "do a thing",
            cwd=tmp_path,
            approval_mode="auto",
            output_format="json",
            model="sonnet",
            session=sess,
        )

        assert result.ok is True
        assert sess.call_count == 1
        assert sess.prompt_tokens == 200
        assert sess.completion_tokens == 80
        assert sess.total_tokens == 280
        assert sess.cost == 0.05
        # Per-model bucket uses the namespaced driver key.
        assert any(k.startswith("coding-agent/claude") for k in sess.summary()["per_model"])

    @patch("prompture.infra.coding_agents.subprocess.run")
    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/claude"),
    )
    def test_session_skipped_on_timeout(self, _mock_resolve, mock_run, tmp_path):
        """A timed-out run should not pollute the session counters."""
        import subprocess as _sp

        from prompture.infra import UsageSession

        mock_run.side_effect = _sp.TimeoutExpired(cmd=["claude"], timeout=1)

        sess = UsageSession()
        result = run_coding_agent(
            "claude",
            "x",
            cwd=tmp_path,
            approval_mode="auto",
            session=sess,
        )

        assert result.timed_out is True
        assert sess.call_count == 0
        assert sess.cost == 0.0

    @patch("prompture.infra.coding_agents.subprocess.run")
    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/claude"),
    )
    def test_session_records_zero_for_text_mode(self, _mock_resolve, mock_run, tmp_path):
        """A text-mode call still bumps call_count but with zero tokens/cost."""
        from prompture.infra import UsageSession

        mock_run.return_value = subprocess.CompletedProcess(
            args=["/usr/local/bin/claude"], returncode=0, stdout="hi", stderr=""
        )

        sess = UsageSession()
        run_coding_agent(
            "claude",
            "x",
            cwd=tmp_path,
            approval_mode="auto",
            session=sess,
        )

        assert sess.call_count == 1
        assert sess.cost == 0.0
        assert sess.total_tokens == 0


class TestAskedQuestionProperty:
    @patch("prompture.infra.coding_agents.subprocess.run")
    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/claude"),
    )
    def test_asked_question_surfaces_last_question_event(self, _mock_resolve, mock_run, tmp_path):
        import json as _json

        stream = "\n".join(
            [
                _json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {"type": "text", "text": "Which approach do you want?\n1. Foo\n2. Bar"}
                            ]
                        },
                    }
                ),
                _json.dumps(
                    {
                        "type": "result",
                        "is_error": False,
                        "result": "I need a decision before continuing.",
                    }
                ),
            ]
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=["/usr/local/bin/claude"], returncode=0, stdout=stream, stderr=""
        )

        result = run_coding_agent(
            "claude",
            "do a thing",
            cwd=tmp_path,
            approval_mode="auto",
            output_format="json",
        )

        q = result.asked_question
        assert q is not None
        assert q.type == "question"
        assert q.choices == ("Foo", "Bar")

    @patch("prompture.infra.coding_agents.subprocess.run")
    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/claude"),
    )
    def test_asked_question_is_none_when_no_question(self, _mock_resolve, mock_run, tmp_path):
        import json as _json

        stream = _json.dumps(
            {
                "type": "result",
                "is_error": False,
                "result": "Done.",
            }
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=["/usr/local/bin/claude"], returncode=0, stdout=stream, stderr=""
        )

        result = run_coding_agent(
            "claude",
            "task",
            cwd=tmp_path,
            approval_mode="auto",
            output_format="json",
        )

        assert result.asked_question is None


class TestSessionContinuity:
    """Tests for resuming a previous coding-agent session."""

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/claude"),
    )
    def test_claude_resume_flag_emitted(self, _mock_resolve, tmp_path):
        command = build_coding_agent_command(
            "claude",
            "follow up",
            cwd=tmp_path,
            approval_mode="auto",
            session_id="sess-abc",
        )
        assert "--resume" in command.argv
        idx = command.argv.index("--resume")
        assert command.argv[idx + 1] == "sess-abc"

    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/codex"),
    )
    def test_codex_resume_subcommand(self, _mock_resolve, tmp_path):
        command = build_coding_agent_command(
            "codex",
            "follow up",
            cwd=tmp_path,
            approval_mode="auto",
            session_id="sess-xyz",
        )
        # codex exec resume <id> ...
        assert command.argv[1:4] == ["exec", "resume", "sess-xyz"]

    @patch("prompture.infra.coding_agents.subprocess.run")
    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/claude"),
    )
    def test_result_captures_session_id_from_system_event(self, _mock_resolve, mock_run, tmp_path):
        import json as _json

        stream = "\n".join(
            [
                _json.dumps({"type": "system", "subtype": "init", "session_id": "captured-1"}),
                _json.dumps({"type": "result", "is_error": False, "result": "ok"}),
            ]
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=["/usr/local/bin/claude"], returncode=0, stdout=stream, stderr=""
        )

        result = run_coding_agent(
            "claude",
            "task",
            cwd=tmp_path,
            approval_mode="auto",
            output_format="json",
        )

        assert result.session_id == "captured-1"

    @patch("prompture.infra.coding_agents.subprocess.run")
    @patch(
        "prompture.infra.coding_agents.resolve_coding_agent_executable",
        return_value=_executable("/usr/local/bin/claude"),
    )
    def test_supplied_session_id_preserved_when_no_init_event(self, _mock_resolve, mock_run, tmp_path):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["/usr/local/bin/claude"], returncode=0, stdout="plain text", stderr=""
        )

        result = run_coding_agent(
            "claude",
            "follow up",
            cwd=tmp_path,
            approval_mode="auto",
            session_id="from-caller",
        )

        assert result.session_id == "from-caller"
