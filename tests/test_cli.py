import json
import pytest
from pathlib import Path
from click.testing import CliRunner

from prompture.cli import cli


def test_run_command(tmp_path):
    """Test the 'run' command using a mock spec file."""
    # Create temporary directory and files
    spec_file = tmp_path / "spec.json"
    output_file = tmp_path / "output.json"
    
    # Create a minimal valid spec file
    spec_data = {
        "models": [{"id": "mock-model", "driver": "mock"}],
        "tests": [{
            "id": "test-1",
            "prompt_template": "test",
            "inputs": [{}],
            "schema": {"type": "object"}
        }]
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