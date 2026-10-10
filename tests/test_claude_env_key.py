"""The Claude drivers find their key under either common environment name."""

from prompture.drivers.claude_driver import ClaudeDriver, env_api_key


def test_anthropic_api_key_is_accepted(monkeypatch):
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-anthropic-name")
    assert env_api_key() == "sk-from-anthropic-name"
    assert ClaudeDriver().api_key == "sk-from-anthropic-name"


def test_claude_api_key_wins_when_both_are_set(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    assert env_api_key() == "sk-claude"
