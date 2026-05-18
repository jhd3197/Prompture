import json
import os
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from prompture.cli.cli import cli
from prompture.infra.discovery import CodingAgentExecutable


@pytest.mark.integration
def test_run_command(tmp_path):
    """Test the 'run' command using a mock spec file."""
    # Create temporary directory and files
    spec_file = tmp_path / "spec.json"
    output_file = tmp_path / "output.json"

    # Create a minimal valid spec file
    spec_data = {
        "models": [
            {
                "id": "test-model",
                "driver": os.getenv("AI_PROVIDER", "openai"),
                "options": {
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "model": os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
                },
            }
        ],
        "tests": [
            {
                "id": "test-1",
                "prompt_template": "Extract name from: {text}",
                "inputs": [{"text": "My name is Juan"}],
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
            }
        ],
    }

    # Write the spec data to the spec file
    spec_file.write_text(json.dumps(spec_data), encoding="utf-8")

    # Create a CliRunner instance and run the command
    runner = CliRunner()
    result = runner.invoke(cli, ["run", str(spec_file), str(output_file)])

    # Assert that the command succeeded
    assert result.exit_code == 0
    assert f"Report saved to {output_file}" in result.output

    # Assert that the output file exists and contains valid JSON
    assert output_file.exists()

    # Read and parse the output file
    output_data = json.loads(output_file.read_text(encoding="utf-8"))

    # Assert that the output is a dictionary (valid JSON object)
    assert isinstance(output_data, dict)

    # Additional assertions on the structure of the output data could be added
    # For example:
    assert "results" in output_data


def test_coding_agents_command_lists_agents():
    """The coding-agents command prints discovery results."""
    runner = CliRunner()
    result = runner.invoke(cli, ["coding-agents"])

    assert result.exit_code == 0
    assert "claude" in result.output
    assert "codex" in result.output
    assert "gemini" in result.output


def test_coding_agents_command_can_verify_health():
    """The coding-agents command can show health-check errors."""
    runner = CliRunner()
    with (
        patch(
            "prompture.infra.discovery.shutil.which",
            side_effect=lambda binary: "/usr/local/bin/codex" if binary == "codex" else None,
        ),
        patch("prompture.infra.discovery.verify_coding_agent_executable", return_value=(False, "node: not found")),
    ):
        result = runner.invoke(cli, ["coding-agents", "--verify"])

    assert result.exit_code == 0
    assert "failed" in result.output
    assert "node: not found" in result.output


def test_code_agent_dry_run_auto_approve(tmp_path):
    """Dry-run shows the resolved coding-agent command without executing it."""
    runner = CliRunner()
    executable = CodingAgentExecutable(
        binary="/usr/local/bin/codex",
        argv=("/usr/local/bin/codex",),
        display="/usr/local/bin/codex",
    )
    with patch("prompture.infra.coding_agents.resolve_coding_agent_executable", return_value=(executable, None, None)):
        result = runner.invoke(
            cli,
            [
                "code-agent",
                "codex",
                "--cwd",
                str(tmp_path),
                "--auto-approve",
                "--dry-run",
                "fix",
                "the",
                "tests",
            ],
        )

    assert result.exit_code == 0
    assert "/usr/local/bin/codex exec" in result.output
    assert "--ask-for-approval never" in result.output
