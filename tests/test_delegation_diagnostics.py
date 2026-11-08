"""Targeted discovery and native approval event regression coverage."""

import json
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from prompture.cli.cli import cli
from prompture.infra import discovery
from prompture.infra.coding_agent_events import parse_claude_stream_json_lines, parse_codex_json_lines
from prompture.infra.coding_agent_failures import classify_failure


def test_targeted_discovery_never_probes_unselected_tools(monkeypatch):
    resolve = Mock(return_value=(None, False, "not installed"))
    monkeypatch.setattr(discovery, "resolve_coding_agent_executable", resolve)
    result = discovery.get_available_coding_agents(agent_ids=["claude", "codex"], verify=True)
    assert [r.id for r in result] == ["claude", "codex"]
    assert [c.args[0] for c in resolve.call_args_list] == ["claude", "codex"]
    assert all(c.kwargs["verify"] for c in resolve.call_args_list)


def test_cli_agent_filter_and_invalid_id(monkeypatch):
    resolve = Mock(return_value=(None, False, "missing"))
    monkeypatch.setattr(discovery, "resolve_coding_agent_executable", resolve)
    runner = CliRunner()
    result = runner.invoke(cli, ["coding-agents", "--agent", "codex", "--verify", "--json"])
    assert result.exit_code == 0, result.output
    assert [r["id"] for r in json.loads(result.output)] == ["codex"]
    resolve.reset_mock()
    bad = runner.invoke(cli, ["coding-agents", "--agent", "typo"])
    assert bad.exit_code == 2
    resolve.assert_not_called()


def test_claude_permission_denials_survive_success_result():
    events = list(
        parse_claude_stream_json_lines(
            [
                json.dumps(
                    {
                        "type": "result",
                        "is_error": False,
                        "result": "Done",
                        "session_id": "s1",
                        "permission_denials": [{"tool_name": "Bash", "tool_input": {"command": "pytest"}}],
                    }
                )
            ]
        )
    )
    assert events[0].type == "approval"
    assert events[0].tool_input == {"command": "pytest"}
    assert events[1].type == "done"
    assert events[1].session_id == "s1"


def test_codex_native_request_is_not_a_completed_command():
    events = list(
        parse_codex_json_lines(
            [
                json.dumps(
                    {
                        "msg": {"type": "exec_approval_request", "command": ["python", "extract.py"]},
                    }
                )
            ]
        )
    )
    assert events[0].type == "approval"
    assert events[0].tool_input["command"] == ["python", "extract.py"]


@pytest.mark.parametrize(
    "message", ["PermissionError: cannot read /tmp/input", "the test checks permission denied text"]
)
def test_filesystem_errors_are_not_invented_user_approvals(message):
    assert classify_failure(message)["category"] == "task"
