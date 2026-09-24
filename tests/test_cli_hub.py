"""`prompture hub` delegates to the optional prompture-hub package."""

from __future__ import annotations

import os
import sys
import types

from click.testing import CliRunner

from prompture.cli.cli import cli


def test_hub_missing_prints_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "prompture_hub", None)
    monkeypatch.setitem(sys.modules, "prompture_hub.main", None)
    result = CliRunner().invoke(cli, ["hub"])
    assert result.exit_code == 1
    assert 'pip install "prompture[hub]"' in result.output


def test_hub_delegates_and_passes_host_port(monkeypatch):
    calls = []
    pkg = types.ModuleType("prompture_hub")
    main = types.ModuleType("prompture_hub.main")
    main.cli = lambda: calls.append("ran")
    monkeypatch.setitem(sys.modules, "prompture_hub", pkg)
    monkeypatch.setitem(sys.modules, "prompture_hub.main", main)
    # setenv first so monkeypatch restores whatever the CLI writes.
    monkeypatch.setenv("HUB_HOST", "unset")
    monkeypatch.setenv("HUB_PORT", "0")

    result = CliRunner().invoke(cli, ["hub", "--host", "0.0.0.0", "--port", "2026"])

    assert result.exit_code == 0, result.output
    assert calls == ["ran"]
    assert os.environ["HUB_HOST"] == "0.0.0.0"
    assert os.environ["HUB_PORT"] == "2026"
