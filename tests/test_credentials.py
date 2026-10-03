"""Tests for the private credential store and its Settings / foundation hooks."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import prompture.infra.credentials as credentials
from prompture.infra.credentials import (
    MAX_STORE_BYTES,
    CredentialStore,
    CredentialStoreError,
    credentials_path,
    get_config_value,
    mask_value,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point ``Path.home()`` at a temp dir so the real ~/.prompture is never touched."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    for var in ("PROMPTURE_PROFILE", "PROMPTURE_PROXY", "PROMPTURE_JINA_READER_PROXY"):
        monkeypatch.delenv(var, raising=False)
    credentials._read_cache.clear()
    return fake_home


def _symlinks_supported(tmp_path: Path) -> bool:
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(tmp_path)
    except (OSError, NotImplementedError):
        return False
    probe.unlink()
    return True


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


class TestStoreBasics:
    def test_default_path_under_home(self, home):
        assert credentials_path() == home / ".prompture" / "credentials.yaml"

    def test_not_created_until_first_write(self, home):
        store = CredentialStore()
        assert store.get("OPENAI_API_KEY") is None
        assert store.profiles() == []
        assert not (home / ".prompture").exists()
        store.set("OPENAI_API_KEY", "sk-one")
        assert store.path.is_file()

    def test_roundtrip_is_case_insensitive_and_keeps_special_chars(self, home):
        store = CredentialStore()
        value = "sk-\"quoted\" 'single' : # hash \\ back\nnewline é"
        store.set("openai_api_key", value)
        assert store.get("OPENAI_API_KEY") == value
        assert store.get("openai_api_key") == value
        assert store.list_keys() == ["OPENAI_API_KEY"]

    def test_on_disk_format_is_valid_yaml(self, home):
        yaml = pytest.importorskip("yaml")
        store = CredentialStore()
        store.set("A_KEY", "x: y # z")
        store.set("B_KEY", "true", profile="work")
        loaded = yaml.safe_load(store.path.read_text(encoding="utf-8"))
        assert loaded == {"default": {"A_KEY": "x: y # z"}, "work": {"B_KEY": "true"}}

    def test_unset_and_delete_profile(self, home):
        store = CredentialStore()
        store.set("A", "1")
        store.set("B", "2", profile="work")
        assert store.unset("A") is True
        assert store.unset("A") is False
        assert store.get("A") is None
        assert store.delete_profile("work") is True
        assert store.delete_profile("work") is False
        assert "work" not in store.profiles()

    def test_set_many_single_write(self, home):
        store = CredentialStore()
        store.set_many({"A": "1", "b": "2"}, profile="work")
        assert store.list_keys("work", with_values=True) == {"A": "1", "B": "2"}

    @pytest.mark.parametrize("bad", ["", "1ABC", "A-B", "A B", "x" * 200])
    def test_invalid_key_names(self, home, bad):
        with pytest.raises(CredentialStoreError):
            CredentialStore().set(bad, "v")

    @pytest.mark.parametrize("bad", ["", "-x", "a/b", "a b", "../up"])
    def test_invalid_profile_names(self, home, bad):
        with pytest.raises(CredentialStoreError):
            CredentialStore().set("A", "v", profile=bad or "!")

    def test_values_must_be_strings(self, home):
        with pytest.raises(CredentialStoreError):
            CredentialStore().set("A", 5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


class TestProfiles:
    def test_profile_sections_and_inheritance(self, home):
        store = CredentialStore()
        store.set("OPENAI_API_KEY", "sk-default")
        store.set("PROMPTURE_PROXY", "http://shared:1")
        store.set("OPENAI_API_KEY", "sk-work", profile="work")
        assert store.get("OPENAI_API_KEY", profile="work") == "sk-work"
        assert store.get("PROMPTURE_PROXY", profile="work") == "http://shared:1"
        assert store.get("PROMPTURE_PROXY", profile="work", inherit=False) is None
        assert store.get("OPENAI_API_KEY") == "sk-default"
        assert store.list_keys("work") == ["OPENAI_API_KEY"]
        assert sorted(store.profiles()) == ["default", "work"]

    def test_env_selects_active_profile(self, home, monkeypatch):
        store = CredentialStore()
        store.set("OPENAI_API_KEY", "sk-default")
        store.set("OPENAI_API_KEY", "sk-personal", profile="personal")
        monkeypatch.setenv("PROMPTURE_PROFILE", "personal")
        assert CredentialStore().get("OPENAI_API_KEY") == "sk-personal"
        assert CredentialStore(profile="default").get("OPENAI_API_KEY") == "sk-default"
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert get_config_value("OPENAI_API_KEY") == "sk-personal"

    def test_list_keys_hides_values_by_default(self, home):
        store = CredentialStore()
        store.set("OPENAI_API_KEY", "sk-very-secret-value")
        listed = store.list_keys()
        assert listed == ["OPENAI_API_KEY"]
        assert "sk-very-secret-value" not in repr(listed)


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


class TestSafety:
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    def test_posix_mode_is_0600(self, home):
        store = CredentialStore()
        store.set("A", "1")
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
        store.set("B", "2")
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows ACLs")
    def test_windows_acl_is_user_only(self, home):
        store = CredentialStore()
        store.set("A", "1")
        out = subprocess.run(["icacls", str(store.path)], capture_output=True, text=True, check=True).stdout
        body = out.replace(str(store.path), "")
        aces = [ln.strip() for ln in body.splitlines() if ln.strip() and ":" in ln and "Successfully" not in ln]
        assert len(aces) == 1, out
        assert "(F)" in aces[0]
        assert "(I)" not in aces[0]
        user = os.environ.get("USERNAME", "")
        assert user.lower() in aces[0].lower()

    def test_atomic_write_leaves_no_temp_files(self, home):
        store = CredentialStore()
        for i in range(3):
            store.set(f"K{i}", str(i))
        names = sorted(p.name for p in store.path.parent.iterdir())
        assert names == ["credentials.yaml"]

    def test_failed_replace_keeps_original_and_cleans_temp(self, home, monkeypatch):
        store = CredentialStore()
        store.set("A", "original")

        def boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(credentials.os, "replace", boom)
        with pytest.raises(OSError):
            store.set("A", "new")
        monkeypatch.undo()
        credentials._read_cache.clear()
        assert sorted(p.name for p in store.path.parent.iterdir()) == ["credentials.yaml"]
        assert CredentialStore(store.path).get("A") == "original"

    def test_read_cap(self, home):
        store = CredentialStore()
        store.path.parent.mkdir(parents=True)
        store.path.write_bytes(b"default:\n  A: " + b"x" * MAX_STORE_BYTES + b"\n")
        with pytest.raises(CredentialStoreError, match="larger than"):
            store.load()
        assert get_config_value("A", "fallback") == "fallback"

    def test_rejects_symlinked_file(self, home, tmp_path):
        if not _symlinks_supported(tmp_path):
            pytest.skip("symlinks not supported here")
        real = tmp_path / "elsewhere.yaml"
        real.write_text('default:\n  A: "1"\n', encoding="utf-8")
        target = home / ".prompture" / "credentials.yaml"
        target.parent.mkdir()
        target.symlink_to(real)
        store = CredentialStore()
        with pytest.raises(CredentialStoreError, match="symlink"):
            store.get("A")
        with pytest.raises(CredentialStoreError, match="symlink"):
            store.set("A", "2")
        assert real.read_text(encoding="utf-8") == 'default:\n  A: "1"\n'

    def test_rejects_symlinked_parent(self, home, tmp_path):
        if not _symlinks_supported(tmp_path):
            pytest.skip("symlinks not supported here")
        real_dir = tmp_path / "real-state"
        real_dir.mkdir()
        (home / ".prompture").symlink_to(real_dir, target_is_directory=True)
        with pytest.raises(CredentialStoreError, match="symlink"):
            CredentialStore().set("A", "1")
        assert list(real_dir.iterdir()) == []

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows directory junctions")
    def test_rejects_junctioned_parent(self, home, tmp_path):
        import _winapi

        real_dir = tmp_path / "real-state"
        real_dir.mkdir()
        _winapi.CreateJunction(str(real_dir), str(home / ".prompture"))
        with pytest.raises(CredentialStoreError, match="symlink"):
            CredentialStore().set("A", "1")
        assert list(real_dir.iterdir()) == []

    def test_posix_branch_chmods_0600(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(credentials.sys, "platform", "linux")
        monkeypatch.setattr(credentials.os, "chmod", lambda path, mode: calls.append(mode))
        credentials._restrict_permissions(tmp_path / "f")
        assert calls == [0o600]

    def test_windows_acl_failure_aborts_write(self, home, monkeypatch):
        def fail(path):
            raise CredentialStoreError("icacls failed")

        monkeypatch.setattr(credentials, "_restrict_permissions", fail)
        with pytest.raises(CredentialStoreError):
            CredentialStore().set("A", "1")
        assert [p.name for p in (home / ".prompture").iterdir()] == []

    def test_malformed_store_is_an_error_but_lookup_survives(self, home, monkeypatch):
        store = CredentialStore()
        store.path.parent.mkdir(parents=True)
        store.path.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(CredentialStoreError):
            store.load()
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert get_config_value("OPENAI_API_KEY", "dflt") == "dflt"


# ---------------------------------------------------------------------------
# Fallback (no PyYAML) parser
# ---------------------------------------------------------------------------


class TestFallbackParser:
    def test_roundtrip_without_pyyaml(self, home, monkeypatch):
        monkeypatch.setattr(credentials, "_yaml", None)
        store = CredentialStore()
        value = "sk-\"q\" 'x' : # \\ \n é \u0007"
        store.set("OPENAI_API_KEY", value)
        store.set("OTHER", "", profile="work")
        credentials._read_cache.clear()
        assert store.get("OPENAI_API_KEY") == value
        assert store.list_keys("work", with_values=True) == {"OTHER": ""}

    def test_hand_written_subset(self, home, monkeypatch):
        monkeypatch.setattr(credentials, "_yaml", None)
        text = (
            "# comment\n"
            "---\n"
            "default:\n"
            "  openai_api_key: sk-plain   # trailing comment\n"
            "  CLAUDE_API_KEY: 'it''s'\n"
            '  "GROQ_API_KEY": "g\\"q"\n'
            "  EMPTY:\n"
            "  PROXY: http://u:p@h:1/#frag\n"
            "work: {}\n"
        )
        data = credentials._normalize_loaded(credentials._parse_minimal(text))
        assert data == {
            "default": {
                "OPENAI_API_KEY": "sk-plain",
                "CLAUDE_API_KEY": "it's",
                "GROQ_API_KEY": 'g"q',
                "PROXY": "http://u:p@h:1/#frag",
            },
            "work": {},
        }

    @pytest.mark.parametrize(
        "text",
        [
            "  A: 1\n",  # key outside a profile
            "default:\n  A: 'unterminated\n",
            'default:\n  A: "unterminated\n',
            'default:\n  A: "x" trailing\n',
            "default:\n\tA: 1\n",
            "just text\n",
        ],
    )
    def test_malformed_inputs_raise(self, text):
        with pytest.raises(CredentialStoreError):
            credentials._parse_minimal(text)


# ---------------------------------------------------------------------------
# get_config_value and masking
# ---------------------------------------------------------------------------


class TestGetConfigValue:
    def test_env_wins_then_store_then_default(self, home, monkeypatch):
        CredentialStore().set("PROMPTURE_TEST_SETTING", "from-store")
        monkeypatch.setenv("PROMPTURE_TEST_SETTING", "from-env")
        assert get_config_value("PROMPTURE_TEST_SETTING") == "from-env"
        monkeypatch.setenv("PROMPTURE_TEST_SETTING", "   ")
        assert get_config_value("PROMPTURE_TEST_SETTING") == "from-store"
        monkeypatch.delenv("PROMPTURE_TEST_SETTING")
        assert get_config_value("PROMPTURE_TEST_SETTING") == "from-store"
        assert get_config_value("PROMPTURE_MISSING_SETTING", "dflt") == "dflt"

    def test_missing_store_returns_default(self, home):
        assert get_config_value("PROMPTURE_NOT_THERE") is None

    def test_mask_never_reveals_full_secret(self):
        secret = "sk-proj-abcdefghijklmnopqrstuvwxyz"
        masked = mask_value(secret)
        assert secret not in masked
        assert masked.endswith("wxyz")
        assert mask_value("short") == "****"
        assert "abcdefghij" not in mask_value("abcdefghij")
        assert mask_value("gpt-4o", secret=False) == "gpt-4o"


# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------


class TestSettingsSource:
    FIELD = "GROQ_MODEL"

    def test_precedence_init_env_dotenv_store(self, home, tmp_path, monkeypatch):
        from prompture.infra.settings import Settings

        monkeypatch.delenv(self.FIELD, raising=False)
        CredentialStore().set(self.FIELD, "from-store")
        assert Settings(_env_file=None).groq_model == "from-store"

        dotenv = tmp_path / ".env"
        dotenv.write_text(f"{self.FIELD}=from-dotenv\n", encoding="utf-8")
        assert Settings(_env_file=str(dotenv)).groq_model == "from-dotenv"

        monkeypatch.setenv(self.FIELD, "from-env")
        assert Settings(_env_file=str(dotenv)).groq_model == "from-env"
        assert Settings(_env_file=str(dotenv), groq_model="from-init").groq_model == "from-init"

    def test_store_values_are_coerced(self, home, monkeypatch):
        from prompture.infra.settings import Settings

        monkeypatch.delenv("MODEL_RATES_TTL_DAYS", raising=False)
        CredentialStore().set("MODEL_RATES_TTL_DAYS", "3")
        assert Settings(_env_file=None).model_rates_ttl_days == 3

    def test_active_profile_feeds_settings(self, home, monkeypatch):
        from prompture.infra.settings import Settings

        monkeypatch.delenv(self.FIELD, raising=False)
        CredentialStore().set(self.FIELD, "default-model")
        CredentialStore().set(self.FIELD, "work-model", profile="work")
        monkeypatch.setenv("PROMPTURE_PROFILE", "work")
        assert Settings(_env_file=None).groq_model == "work-model"

    def test_missing_store_does_not_break_settings(self, home):
        from prompture.infra.settings import Settings

        assert not (home / ".prompture").exists()
        assert Settings(_env_file=None).pricing_source

    @pytest.mark.parametrize(
        "body",
        [b"- not a mapping\n", b"default: [1, 2\n", b"\xff\xfe\x00garbage", b"x" * (MAX_STORE_BYTES + 10)],
        ids=["list", "broken-yaml", "not-utf8", "oversized"],
    )
    def test_invalid_store_does_not_break_settings(self, home, body):
        from prompture.infra.settings import Settings

        path = home / ".prompture" / "credentials.yaml"
        path.parent.mkdir()
        path.write_bytes(body)
        assert Settings(_env_file=None).ai_provider

    def test_bad_profile_env_does_not_break_settings(self, home, monkeypatch):
        from prompture.infra.settings import Settings

        monkeypatch.setenv("PROMPTURE_PROFILE", "../bad")
        assert Settings(_env_file=None).ai_provider


# ---------------------------------------------------------------------------
# Foundation hooks
# ---------------------------------------------------------------------------


class TestFoundationHooks:
    def test_resolve_proxy_reads_store_env_wins(self, home, monkeypatch):
        from prompture.capabilities.http import resolve_proxy

        assert resolve_proxy("jina_reader") is None
        store = CredentialStore()
        store.set("PROMPTURE_PROXY", "http://stored:1")
        assert resolve_proxy("jina_reader") == "http://stored:1"
        store.set("PROMPTURE_JINA_READER_PROXY", "http://stored-specific:2")
        assert resolve_proxy("jina_reader") == "http://stored-specific:2"
        monkeypatch.setenv("PROMPTURE_JINA_READER_PROXY", "http://env:3")
        assert resolve_proxy("jina_reader") == "http://env:3"

    def test_backend_chain_override_from_store(self, home, monkeypatch):
        from prompture.capabilities.backends import BackendChain

        class B:
            def __init__(self, name):
                self.name = name

        chain = BackendChain([B("a"), B("b"), B("c")], override_env="PROMPTURE_TEST_ORDER")
        monkeypatch.delenv("PROMPTURE_TEST_ORDER", raising=False)
        assert [b.name for b in chain.ordered()] == ["a", "b", "c"]
        CredentialStore().set("PROMPTURE_TEST_ORDER", "c")
        assert [b.name for b in chain.ordered()] == ["c", "a", "b"]
        monkeypatch.setenv("PROMPTURE_TEST_ORDER", "b")
        assert [b.name for b in chain.ordered()] == ["b", "a", "c"]
