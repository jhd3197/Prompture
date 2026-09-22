"""Regression tests for ``.env`` loading at package import.

Importing ``prompture`` populates ``os.environ`` from a ``.env`` file. Keys with
a blank value must stay *unset* rather than becoming set-but-empty: libraries
that resolve defaults with ``os.getenv(KEY, default)`` take the empty string
over their default, which breaks unrelated tooling in the same process.

The concrete failure this guards against: ``.env.copy`` ships ``HF_ENDPOINT=``,
which made ``huggingface_hub.constants.ENDPOINT`` the empty string, so every
Hub URL lost its scheme and downloads raised ``UnsupportedProtocol``.
"""

from __future__ import annotations

import os

import pytest

import prompture
from prompture import _load_dotenv_skipping_blanks


def _write_env(tmp_path, body: str):
    path = tmp_path / ".env"
    path.write_text(body, encoding="utf-8")
    return path


class TestDotenvBlankHandling:
    def test_blank_values_are_not_exported(self, tmp_path, monkeypatch):
        env = _write_env(tmp_path, "HF_ENDPOINT=\nHF_TOKEN=\nREAL_VALUE=set\n")
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("REAL_VALUE", raising=False)

        _load_dotenv_skipping_blanks(str(env))

        assert "HF_ENDPOINT" not in os.environ
        assert "HF_TOKEN" not in os.environ
        assert os.environ["REAL_VALUE"] == "set"

    def test_whitespace_only_counts_as_blank(self, tmp_path, monkeypatch):
        env = _write_env(tmp_path, 'PADDED="   "\n')
        monkeypatch.delenv("PADDED", raising=False)

        _load_dotenv_skipping_blanks(str(env))

        assert "PADDED" not in os.environ

    def test_does_not_override_real_environment(self, tmp_path, monkeypatch):
        env = _write_env(tmp_path, "OLLAMA_ENDPOINT=http://from-dotenv\n")
        monkeypatch.setenv("OLLAMA_ENDPOINT", "http://from-real-env")

        _load_dotenv_skipping_blanks(str(env))

        assert os.environ["OLLAMA_ENDPOINT"] == "http://from-real-env"

    def test_missing_env_file_is_not_an_error(self, tmp_path):
        _load_dotenv_skipping_blanks(str(tmp_path / "nope.env"))  # must not raise

    def test_importing_prompture_leaves_no_blank_hf_endpoint(self):
        """The end-to-end invariant, using only the standard library.

        ``prompture`` has already been imported by the time this runs, so the
        check is on what that import left behind.
        """
        endpoint = os.environ.get("HF_ENDPOINT")
        assert endpoint is None or endpoint.strip(), "importing prompture must not leave HF_ENDPOINT set-but-empty"

    def test_hub_endpoint_keeps_its_scheme(self):
        """The same invariant seen from the library that the bug actually broke.

        ``huggingface_hub`` is not a Prompture dependency — it arrives with the
        ``laya`` extra — so this only runs where it is installed.
        """
        constants = pytest.importorskip("huggingface_hub.constants")
        assert constants.ENDPOINT.startswith(("http://", "https://"))

    def test_env_copy_template_still_has_blank_placeholders(self):
        """The template is *expected* to ship blanks — that is why the guard exists."""
        template = os.path.join(os.path.dirname(os.path.dirname(prompture.__file__)), ".env.copy")
        if not os.path.isfile(template):
            return
        with open(template, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh]
        assert any(ln.endswith("=") and not ln.startswith("#") for ln in lines)
