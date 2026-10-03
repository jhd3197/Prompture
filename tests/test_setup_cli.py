"""Tests for ``prompture setup`` / ``configure`` / ``reset``."""

from __future__ import annotations

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import prompture.infra.credentials as credentials
import prompture.infra.environment as environment
from prompture.cli import setup_cmd
from prompture.cli.setup_cmd import COMMANDS, KeyCheck, configure, reset, setup, validate_key
from prompture.infra.credentials import CredentialStore

SECRET = "sk-test-SECRET-abcdefghijklmnop1234"
OTHER_SECRET = "sk-test-OTHER-zyxwvutsrqponmlk9876"


@pytest.fixture
def home(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    for var in ("PROMPTURE_PROFILE", "PROMPTURE_PROXY", "OPENAI_API_KEY", "CLAUDE_API_KEY", "PROMPTURE_DEFAULT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    credentials._read_cache.clear()
    return fake_home


@pytest.fixture
def checks(monkeypatch):
    """Mock live validation: returns queued results, records calls."""
    calls: list[tuple[str, str]] = []
    queue: list[KeyCheck] = []

    def fake(provider, key, **kwargs):
        calls.append((provider, key))
        return queue.pop(0) if queue else KeyCheck("valid", "accepted by the provider")

    monkeypatch.setattr(setup_cmd, "validate_key", fake)
    fake.calls = calls  # type: ignore[attr-defined]
    fake.queue = queue  # type: ignore[attr-defined]
    return fake


@pytest.fixture
def local_env(monkeypatch):
    info = environment.EnvironmentInfo(kind="local", platform="win32", suggestions=["local tip"])
    monkeypatch.setattr(environment, "detect_environment", lambda *a, **k: info)
    return info


@pytest.fixture
def server_env(monkeypatch):
    info = environment.EnvironmentInfo(
        kind="server",
        platform="linux",
        container=True,
        markers=["container: /.dockerenv"],
        suggestions=["Servers often get blocked by some providers from datacenter IPs; consider PROMPTURE_PROXY"],
    )
    monkeypatch.setattr(environment, "detect_environment", lambda *a, **k: info)
    return info


def _store_path(home: Path) -> Path:
    return home / ".prompture" / "credentials.yaml"


def test_commands_exported():
    assert [c.name for c in COMMANDS] == ["setup", "configure", "reset"]
    assert all(isinstance(c, click.Command) for c in COMMANDS)


def test_catalogue_has_key_providers():
    catalogue = setup_cmd.provider_key_catalogue()
    by_name = {e.provider: e for e in catalogue}
    assert by_name["openai"].env_var == "OPENAI_API_KEY"
    assert by_name["openai"].llm
    assert by_name["claude"].env_var == "CLAUDE_API_KEY"
    first_non_llm = next(i for i, e in enumerate(catalogue) if not e.llm)
    assert all(not e.llm for e in catalogue[first_non_llm:])


# ---------------------------------------------------------------------------
# setup wizard
# ---------------------------------------------------------------------------


class TestSetupWizard:
    def test_happy_path_saves_validated_key(self, home, checks, local_env):
        inputs = [
            "openai",  # providers
            SECRET,  # key (hidden)
            "openai/gpt-4o-mini",  # default model
            "n",  # web search keys
            "n",  # proxy (local default is no)
            "y",  # save
        ]
        result = CliRunner().invoke(setup, [], input="\n".join(inputs) + "\n")
        assert result.exit_code == 0, result.output
        assert checks.calls == [("openai", SECRET)]
        assert SECRET not in result.output
        assert 'pip install "prompture[web]"' in result.output
        store = CredentialStore()
        assert store.get("OPENAI_API_KEY") == SECRET
        assert store.get("PROMPTURE_DEFAULT_MODEL") == "openai/gpt-4o-mini"
        assert store.get("AI_PROVIDER") == "openai"
        assert store.get("OPENAI_MODEL") == "gpt-4o-mini"

    def test_rejected_key_can_be_reentered(self, home, checks, local_env):
        checks.queue.extend([KeyCheck("invalid", "rejected by the provider (HTTP 401)"), KeyCheck("valid", "ok")])
        inputs = ["1", OTHER_SECRET, "r", SECRET, "", "n", "n", "y"]
        result = CliRunner().invoke(setup, [], input="\n".join(inputs) + "\n")
        assert result.exit_code == 0, result.output
        assert "Rejected" in result.output
        assert [c[1] for c in checks.calls] == [OTHER_SECRET, SECRET]
        first = setup_cmd.provider_key_catalogue()[0]
        assert CredentialStore().get(first.env_var) == SECRET
        assert SECRET not in result.output and OTHER_SECRET not in result.output

    def test_unverified_key_kept_by_default(self, home, checks, local_env):
        checks.queue.append(KeyCheck("unverified", "network error (ConnectionError)"))
        inputs = ["claude", SECRET, "", "", "n", "n", "y"]
        result = CliRunner().invoke(setup, ["--profile", "work"], input="\n".join(inputs) + "\n")
        assert result.exit_code == 0, result.output
        assert "Could not verify" in result.output
        assert CredentialStore().get("CLAUDE_API_KEY", profile="work") == SECRET
        assert CredentialStore().list_keys("default") == []

    def test_skip_rejected_key(self, home, checks, local_env):
        checks.queue.append(KeyCheck("invalid", "rejected"))
        inputs = ["openai", SECRET, "s", "", "n", "n"]
        result = CliRunner().invoke(setup, [], input="\n".join(inputs) + "\n")
        assert result.exit_code == 0, result.output
        assert "Nothing to save" in result.output
        assert not _store_path(home).exists()

    def test_dry_run_writes_nothing(self, home, checks, local_env):
        inputs = ["openai", SECRET, "", "n", "n"]
        result = CliRunner().invoke(setup, ["--dry-run"], input="\n".join(inputs) + "\n")
        assert result.exit_code == 0, result.output
        assert "Dry run" in result.output
        assert "OPENAI_API_KEY" in result.output
        assert SECRET not in result.output
        assert not (home / ".prompture").exists()

    def test_server_environment_offers_proxy(self, home, checks, server_env):
        inputs = [
            "",  # no providers
            "",  # no default model
            "n",  # web keys
            "",  # proxy confirm: default yes on servers
            "ftp://nope",  # invalid proxy, re-asked
            "http://proxy.internal:3128",
            "youtube=http://yt-proxy:8080",
            "bad line",
            "",  # finish per-backend
            "y",
        ]
        result = CliRunner().invoke(setup, [], input="\n".join(inputs) + "\n")
        assert result.exit_code == 0, result.output
        assert "Environment: server" in result.output
        assert "datacenter IPs" in result.output
        store = CredentialStore()
        assert store.get("PROMPTURE_PROXY") == "http://proxy.internal:3128"
        assert store.get("PROMPTURE_YOUTUBE_PROXY") == "http://yt-proxy:8080"
        assert checks.calls == []

    def test_web_search_keys(self, home, checks, local_env):
        inputs = ["", "", "y", "tvly-secret-key-123456", "", "", "", "n", "y"]
        result = CliRunner().invoke(setup, [], input="\n".join(inputs) + "\n")
        assert result.exit_code == 0, result.output
        assert CredentialStore().get("TAVILY_API_KEY") == "tvly-secret-key-123456"
        assert "tvly-secret-key-123456" not in result.output

    def test_declining_save_writes_nothing(self, home, checks, local_env):
        inputs = ["openai", SECRET, "", "n", "n", "n"]
        result = CliRunner().invoke(setup, [], input="\n".join(inputs) + "\n")
        assert result.exit_code == 0, result.output
        assert not _store_path(home).exists()


# ---------------------------------------------------------------------------
# configure
# ---------------------------------------------------------------------------


class TestConfigure:
    def test_set_validates_and_masks(self, home, checks):
        result = CliRunner().invoke(configure, ["OPENAI_API_KEY", SECRET])
        assert result.exit_code == 0, result.output
        assert checks.calls == [("openai", SECRET)]
        assert SECRET not in result.output
        assert CredentialStore().get("OPENAI_API_KEY") == SECRET

    def test_invalid_key_is_not_saved(self, home, checks):
        checks.queue.append(KeyCheck("invalid", "rejected by the provider (HTTP 401)"))
        result = CliRunner().invoke(configure, ["OPENAI_API_KEY", SECRET])
        assert result.exit_code == 1
        assert "--no-validate" in result.output
        assert not _store_path(home).exists()

    def test_no_validate_skips_check(self, home, checks):
        result = CliRunner().invoke(configure, ["openai_api_key", SECRET, "--no-validate", "--profile", "work"])
        assert result.exit_code == 0, result.output
        assert checks.calls == []
        assert CredentialStore().get("OPENAI_API_KEY", profile="work") == SECRET

    def test_dry_run_makes_no_changes(self, home, checks):
        result = CliRunner().invoke(configure, ["PROMPTURE_PROXY", "http://user:pw@proxy:1", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "Dry run" in result.output
        assert "pw@proxy" not in result.output
        assert not (home / ".prompture").exists()

    def test_unset_and_unset_dry_run(self, home, checks):
        CredentialStore().set("PROMPTURE_PROXY", "http://proxy:1")
        result = CliRunner().invoke(configure, ["PROMPTURE_PROXY", "--unset", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert CredentialStore().get("PROMPTURE_PROXY") == "http://proxy:1"
        result = CliRunner().invoke(configure, ["PROMPTURE_PROXY", "--unset"])
        assert result.exit_code == 0, result.output
        assert CredentialStore().get("PROMPTURE_PROXY") is None
        result = CliRunner().invoke(configure, ["PROMPTURE_PROXY", "--unset"])
        assert "not stored" in result.output

    def test_list_masks_values(self, home, checks):
        store = CredentialStore()
        store.set("OPENAI_API_KEY", SECRET)
        store.set("GROQ_MODEL", "llama-3.3-70b")
        store.set("CLAUDE_API_KEY", OTHER_SECRET, profile="work")
        result = CliRunner().invoke(configure, ["--list"])
        assert result.exit_code == 0, result.output
        assert "[default] (active)" in result.output
        assert "[work]" in result.output
        assert "OPENAI_API_KEY" in result.output and "CLAUDE_API_KEY" in result.output
        assert SECRET not in result.output and OTHER_SECRET not in result.output
        assert "llama-3.3-70b" in result.output  # non-secret values stay readable

    def test_list_empty_store(self, home):
        result = CliRunner().invoke(configure, ["--list"])
        assert result.exit_code == 0
        assert "does not exist yet" in result.output
        assert not (home / ".prompture").exists()

    def test_value_from_stdin_and_prompt(self, home, checks):
        result = CliRunner().invoke(configure, ["OPENAI_API_KEY", "-"], input=SECRET + "\n")
        assert result.exit_code == 0, result.output
        assert CredentialStore().get("OPENAI_API_KEY") == SECRET
        result = CliRunner().invoke(configure, ["CLAUDE_API_KEY"], input=OTHER_SECRET + "\n")
        assert result.exit_code == 0, result.output
        assert OTHER_SECRET not in result.output
        assert CredentialStore().get("CLAUDE_API_KEY") == OTHER_SECRET

    def test_bad_proxy_and_bad_names(self, home, checks):
        assert CliRunner().invoke(configure, ["PROMPTURE_PROXY", "proxy:1"]).exit_code == 2
        assert CliRunner().invoke(configure, ["BAD-NAME", "x"]).exit_code != 0
        assert CliRunner().invoke(configure, []).exit_code == 2
        assert not _store_path(home).exists()

    def test_env_precedence_note(self, home, checks, monkeypatch):
        monkeypatch.setenv("GROQ_MODEL", "env-model")
        result = CliRunner().invoke(configure, ["GROQ_MODEL", "stored-model"])
        assert "takes precedence" in result.output


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def _make_state(home: Path) -> Path:
    state = home / ".prompture"
    (state / "usage").mkdir(parents=True)
    (state / "usage" / "usage.db").write_bytes(b"db")
    (state / "cache").mkdir()
    (state / "cache" / "models.json").write_text("{}", encoding="utf-8")
    (state / "update_check.json").write_text("{}", encoding="utf-8")
    (state / "companion.json").write_text("{}", encoding="utf-8")
    CredentialStore().set("OPENAI_API_KEY", SECRET)
    return state


class TestReset:
    def test_dry_run_lists_and_keeps_everything(self, home):
        state = _make_state(home)
        before = sorted(p.name for p in state.iterdir())
        result = CliRunner().invoke(reset, ["--dry-run"])
        assert result.exit_code == 0, result.output
        for name in ("credentials.yaml", "usage/", "cache/", "update_check.json", "companion.json"):
            assert name in result.output
        assert "Dry run" in result.output
        assert sorted(p.name for p in state.iterdir()) == before

    def test_confirmation_declined(self, home):
        state = _make_state(home)
        result = CliRunner().invoke(reset, [], input="n\n")
        assert result.exit_code == 0
        assert (state / "credentials.yaml").exists()

    def test_yes_removes_everything(self, home):
        state = _make_state(home)
        result = CliRunner().invoke(reset, ["--yes"])
        assert result.exit_code == 0, result.output
        assert not state.exists()

    def test_keep_credentials(self, home):
        state = _make_state(home)
        result = CliRunner().invoke(reset, ["--keep-credentials"], input="y\n")
        assert result.exit_code == 0, result.output
        assert sorted(p.name for p in state.iterdir()) == ["credentials.yaml"]
        credentials._read_cache.clear()
        assert CredentialStore().get("OPENAI_API_KEY") == SECRET

    def test_missing_state(self, home):
        result = CliRunner().invoke(reset, ["--yes"])
        assert result.exit_code == 0
        assert "Nothing to reset" in result.output

    def test_never_touches_outside_via_junction(self, home, tmp_path):
        import sys

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "precious.txt").write_text("keep me", encoding="utf-8")
        state = _make_state(home)
        link = state / "linked"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            if sys.platform != "win32":
                pytest.skip("symlinks not supported here")
            import _winapi

            _winapi.CreateJunction(str(outside), str(link))
        result = CliRunner().invoke(reset, ["--yes"])
        assert result.exit_code == 1
        assert "points outside" in result.output
        assert (outside / "precious.txt").read_text(encoding="utf-8") == "keep me"
        assert link.exists()
        assert not (state / "usage").exists()

    def test_refuses_dir_containing_outside_link(self, home, tmp_path):
        import sys

        outside = tmp_path / "outside2"
        outside.mkdir()
        (outside / "precious.txt").write_text("keep me", encoding="utf-8")
        state = _make_state(home)
        nested = state / "cache" / "sub"
        try:
            nested.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            if sys.platform != "win32":
                pytest.skip("symlinks not supported here")
            import _winapi

            _winapi.CreateJunction(str(outside), str(nested))
        result = CliRunner().invoke(reset, ["--yes"])
        assert result.exit_code == 1
        assert (state / "cache").exists()
        assert (outside / "precious.txt").exists()

    def test_refuses_symlinked_state_dir(self, home, tmp_path):
        import sys

        real = tmp_path / "real-state"
        real.mkdir()
        (real / "x.json").write_text("{}", encoding="utf-8")
        state = home / ".prompture"
        try:
            state.symlink_to(real, target_is_directory=True)
        except (OSError, NotImplementedError):
            if sys.platform != "win32":
                pytest.skip("symlinks not supported here")
            import _winapi

            _winapi.CreateJunction(str(real), str(state))
        result = CliRunner().invoke(reset, ["--yes"])
        assert result.exit_code == 1
        assert (real / "x.json").exists()


# ---------------------------------------------------------------------------
# validate_key (HTTP mocked)
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status):
        self.status_code = status


class TestValidateKey:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [(200, "valid"), (401, "invalid"), (403, "invalid"), (429, "unverified"), (500, "unverified")],
    )
    def test_status_mapping(self, monkeypatch, status, expected):
        import requests

        seen = {}

        def fake_get(url, headers=None, **kwargs):
            seen.update(url=url, headers=headers, kwargs=kwargs)
            return _Resp(status)

        monkeypatch.setattr(requests, "get", fake_get)
        check = validate_key("openai", SECRET)
        assert check.status == expected
        assert SECRET not in check.detail
        assert seen["headers"]["Authorization"] == f"Bearer {SECRET}"
        assert seen["kwargs"]["allow_redirects"] is False

    def test_anthropic_headers(self, monkeypatch):
        import requests

        seen = {}
        monkeypatch.setattr(requests, "get", lambda url, headers=None, **kw: seen.update(h=headers) or _Resp(200))
        assert validate_key("claude", SECRET).status == "valid"
        assert seen["h"]["x-api-key"] == SECRET

    def test_network_error(self, monkeypatch):
        import requests

        def boom(*a, **k):
            raise requests.ConnectionError(f"failed with {SECRET}")

        monkeypatch.setattr(requests, "get", boom)
        check = validate_key("openai", SECRET)
        assert check.status == "unverified"
        assert SECRET not in check.detail

    def test_unknown_provider_unchecked(self):
        assert validate_key("no-such-provider", SECRET).status == "unchecked"
