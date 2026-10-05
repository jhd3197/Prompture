"""Tests for the bundled agent skill and ``prompture skill``."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from prompture.cli.skill_cmd import skill
from prompture.skill import MARKER_FILE, install_skill, skill_files, uninstall_skill


def test_bundle_has_skill_and_references():
    files = {str(f).replace("\\", "/") for f in skill_files()}
    assert "SKILL.md" in files
    for topic in ("search", "web", "media", "models", "mcp", "finance", "dev"):
        assert f"references/{topic}.md" in files


def test_skill_frontmatter_and_routing_table():
    from prompture.skill import skill_source_dir

    text = (skill_source_dir() / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\nname: prompture\n")
    assert "description:" in text.split("---")[1]
    assert "prompture doctor --json" in text
    for topic in ("search", "web", "media", "models", "mcp", "finance", "dev"):
        assert f"references/{topic}.md" in text


def test_install_dry_run_writes_nothing(tmp_path):
    plan = install_skill("path", path=tmp_path, dry_run=True)
    assert plan.dry_run
    assert not (tmp_path / "prompture").exists()


def test_install_upgrade_and_uninstall(tmp_path):
    install_skill("path", path=tmp_path)
    dest = tmp_path / "prompture"
    assert (dest / "SKILL.md").exists()
    assert (dest / MARKER_FILE).exists()
    install_skill("path", path=tmp_path)  # upgrade in place is allowed
    plan = uninstall_skill("path", path=tmp_path, dry_run=True)
    assert dest.exists() and "SKILL.md" in plan.files
    uninstall_skill("path", path=tmp_path)
    assert not dest.exists()


def test_refuses_foreign_directories(tmp_path):
    foreign = tmp_path / "prompture"
    foreign.mkdir()
    (foreign / "mine.txt").write_text("user data")
    with pytest.raises(FileExistsError):
        install_skill("path", path=tmp_path)
    with pytest.raises(PermissionError):
        uninstall_skill("path", path=tmp_path)
    assert (foreign / "mine.txt").exists()


def test_claude_target_uses_home(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    install_skill("claude")
    assert (tmp_path / ".claude" / "skills" / "prompture" / "SKILL.md").exists()


def test_kimi_and_agents_targets_use_home(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    install_skill("kimi")
    assert (tmp_path / ".kimi-code" / "skills" / "prompture" / "SKILL.md").exists()
    install_skill("agents")
    assert (tmp_path / ".agents" / "skills" / "prompture" / "SKILL.md").exists()
    plan = uninstall_skill("agents")
    assert not (tmp_path / ".agents" / "skills" / "prompture").exists()
    assert "SKILL.md" in plan.files


def test_cli_install_and_show(tmp_path):
    runner = CliRunner()
    result = runner.invoke(skill, ["install", "--target", "path", "--path", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "[dry-run]" in result.output
    result = runner.invoke(skill, ["show"])
    assert result.exit_code == 0 and "name: prompture" in result.output
    result = runner.invoke(skill, ["uninstall", "--target", "path", "--path", str(tmp_path)])
    assert result.exit_code == 0 and "Nothing to do" in result.output
