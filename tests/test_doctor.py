"""``prompture doctor``: report schema, providers section, CLI, watch exit codes, companion summary.

No test here touches the network: the capability registry is isolated and
filled with fakes, provider descriptors are built in the test and live model
listings are monkeypatched.
"""

from __future__ import annotations

import json
import logging
import time

import pytest
from click.testing import CliRunner

import prompture.doctor.core as core
import prompture.doctor.providers as prov
from prompture.capabilities import health
from prompture.capabilities.health import HealthStatus, register_capability
from prompture.cli.doctor_cmd import COMMANDS, check_update, doctor, watch
from prompture.doctor import (
    EXIT_BROKEN,
    EXIT_OK,
    EXIT_UPDATE,
    SCHEMA,
    capabilities_summary,
    check_all,
    clear_summary_cache,
    run_watch,
    watch_exit_code,
)
from prompture.drivers.openai_driver import OpenAIDriver
from prompture.drivers.provider_descriptors import DriverSpec, ProviderDescriptor
from prompture.infra.settings import settings
from prompture.infra.updates import UpdateInfo

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def registry(monkeypatch):
    """An empty capability registry (built-in modules are not imported)."""
    saved = dict(health._registry)
    monkeypatch.setattr(health, "_loaded_builtins", True)
    health._registry.clear()
    clear_summary_cache()
    yield health._registry
    health._registry.clear()
    health._registry.update(saved)
    clear_summary_cache()


def _fake(name: str, category: str, status: str, **kw) -> None:
    register_capability(name, category, lambda live: HealthStatus(name, status, **kw))


@pytest.fixture
def fake_caps(registry):
    _fake("web_search", "tools", "ok", active_backend="duckduckgo", message="3 backends")
    _fake("transcribe", "media", "missing", message="no STT", fix_hint="set OPENAI_API_KEY")
    _fake("ffmpeg", "binaries", "ok", active_backend="/usr/bin/ffmpeg", message="ffmpeg 7")
    _fake("yt-dlp", "binaries", "broken", message="shim is stale", fix_hint="pip install -U yt-dlp")
    return registry


def _llm(cls_path: str, kwarg_map: dict[str, str]) -> dict:
    return {"llm_sync": DriverSpec(cls_path, kwarg_map, "m")}


def _openai_desc() -> ProviderDescriptor:
    return ProviderDescriptor(
        name="openai",
        display_name="OpenAI",
        is_configured_check="openai_api_key",
        list_models_kwargs=[("api_key", "openai_api_key", "OPENAI_API_KEY")],
        **_llm("openai_driver.OpenAIDriver", {"api_key": "openai_api_key"}),
    )


def _keyed(name: str) -> ProviderDescriptor:
    return ProviderDescriptor(
        name=name,
        display_name=name.title(),
        is_configured_check=f"{name}_api_key",
        **_llm("openai_driver.OpenAIDriver", {"api_key": f"{name}_api_key"}),
    )


@pytest.fixture
def openai_key(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-not-real", raising=False)
    monkeypatch.setitem(prov.PROVIDER_SDKS, "openai", ((), None))  # SDK may be absent in CI
    return "sk-test-not-real"


@pytest.fixture
def no_keys(monkeypatch):
    for name in ("acme", "zeta", "omega"):
        monkeypatch.delenv(f"{name.upper()}_API_KEY", raising=False)


# ── Report schema ─────────────────────────────────────────────────────────


class TestReport:
    def test_schema_is_stable(self, fake_caps):
        data = check_all().to_dict()
        assert data["schema"] == SCHEMA == "prompture.doctor/1"
        assert set(data) == {
            "schema",
            "prompture_version",
            "python",
            "platform",
            "generated_at",
            "live",
            "only",
            "summary",
            "checks",
        }
        assert set(data["summary"]) == {"worst", "ok", "counts", "by_category"}
        assert data["live"] is False
        for row in data["checks"]:
            assert set(row) == {"name", "status", "category", "active_backend", "message", "fix_hint", "details"}
        json.dumps(data)  # serializable

    def test_summary_counts_and_categories(self, fake_caps):
        report = check_all()
        summary = report.to_dict()["summary"]
        assert summary["worst"] == "broken" and report.worst == "broken"
        assert summary["counts"] == {"ok": 2, "missing": 1, "broken": 1}
        assert summary["by_category"]["binaries"] == {"worst": "broken", "counts": {"ok": 1, "broken": 1}}
        assert summary["by_category"]["tools"]["worst"] == "ok"
        assert report.ok is False and [r.name for r in report.failing] == ["yt-dlp"]

    def test_rows_are_grouped_by_category_worst_first(self, fake_caps):
        names = [(r.category, r.name) for r in check_all().rows]
        assert names == [
            ("tools", "web_search"),
            ("media", "transcribe"),
            ("binaries", "yt-dlp"),
            ("binaries", "ffmpeg"),
        ]

    def test_missing_optional_pieces_are_not_failures(self, registry):
        _fake("transcribe", "media", "missing")
        _fake("search", "tools", "unconfigured")
        report = check_all()
        assert report.ok is True and report.worst == "missing"

    def test_only_filters_categories(self, fake_caps):
        assert {r.category for r in check_all(only="binaries").rows} == {"binaries"}
        assert {r.category for r in check_all(only=["tools", "media"]).rows} == {"tools", "media"}
        assert check_all(only="tools").to_dict()["only"] == ["tools"]

    def test_only_rejects_unknown_categories(self, fake_caps):
        with pytest.raises(ValueError, match="unknown doctor categories"):
            check_all(only="nope")

    def test_crashing_check_becomes_an_error_row(self, registry):
        def boom(live):
            raise RuntimeError("kaput")

        register_capability("flaky", "tools", boom)
        [row] = check_all().rows
        assert row.status == "error" and "kaput" in row.message

    def test_live_flag_reaches_checks(self, registry):
        seen = []
        register_capability("t", "tools", lambda live: seen.append(live) or HealthStatus("t", "ok"))
        assert check_all(live=True).live is True
        assert seen == [True]

    def test_table_has_columns_and_fixes(self, fake_caps):
        table = check_all().to_table()
        assert "CAPABILITY" in table and "STATUS" in table and "ACTIVE BACKEND" in table and "FIX" in table
        assert "pip install -U yt-dlp" in table
        assert "duckduckgo" in table
        assert table.splitlines()[-1].startswith("Overall: broken")

    def test_table_without_rows(self, registry):
        assert check_all().to_table() == "No capabilities registered."

    def test_secrets_are_scrubbed_from_rows(self, registry):
        _fake("leaky", "tools", "error", message="failed https://user:hunter2@example.com/x?api_key=abc123")
        row = check_all().to_dict()["checks"][0]
        assert "hunter2" not in row["message"] and "abc123" not in row["message"]


# ── Providers ─────────────────────────────────────────────────────────────


class TestProviders:
    def test_configured_provider_row(self, openai_key):
        [row] = prov.provider_rows(descriptors=[_openai_desc()])
        assert row.category == "providers" and row.status == "ok"
        assert "OPENAI_API_KEY" in row.message
        assert openai_key not in json.dumps(row.to_dict())

    def test_unconfigured_collapse_into_one_summary_row(self, openai_key, no_keys):
        rows = prov.provider_rows(descriptors=[_openai_desc(), _keyed("acme"), _keyed("zeta"), _keyed("omega")])
        assert [r.name for r in rows] == ["openai", "unconfigured providers"]
        summary = rows[1]
        assert summary.status == "unconfigured"
        assert "acme" in summary.message and "zeta" in summary.message
        assert summary.details["providers"] == {
            "acme": "ACME_API_KEY",
            "zeta": "ZETA_API_KEY",
            "omega": "OMEGA_API_KEY",
        }
        assert "ACME_API_KEY" in summary.fix_hint and "--verbose" in summary.fix_hint

    def test_verbose_lists_every_provider(self, openai_key, no_keys):
        rows = prov.provider_rows(verbose=True, descriptors=[_openai_desc(), _keyed("acme")])
        assert [(r.name, r.status) for r in rows] == [("openai", "ok"), ("acme", "unconfigured")]
        assert rows[1].fix_hint == "set ACME_API_KEY"

    def test_nothing_configured_says_so(self, no_keys):
        [row] = prov.provider_rows(descriptors=[_keyed("acme")])
        assert row.message.startswith("no provider configured")

    def test_sdk_missing(self, monkeypatch):
        monkeypatch.setenv("ACME_API_KEY", "k-1")
        monkeypatch.setitem(prov.PROVIDER_SDKS, "acme", (("prompture_no_such_sdk_xyz",), "acme"))
        [row] = prov.provider_rows(descriptors=[_keyed("acme")])
        assert row.status == "missing"
        assert row.fix_hint == "pip install prompture[acme]"

    def test_sdk_missing_without_extra_names_the_package(self, monkeypatch):
        monkeypatch.setitem(prov.PROVIDER_SDKS, "acme", (("prompture_no_such_sdk_xyz",), None))
        assert prov.sdk_status("acme") == (False, "prompture_no_such_sdk_xyz", "pip install prompture_no_such_sdk_xyz")

    def test_broken_driver_import(self, monkeypatch):
        monkeypatch.setenv("ACME_API_KEY", "k-1")
        desc = ProviderDescriptor(
            name="acme", is_configured_check="acme_api_key", **_llm("no_such_driver.Nope", {"api_key": "acme_api_key"})
        )
        [row] = prov.provider_rows(descriptors=[desc])
        assert row.status == "broken" and "pip install" in row.fix_hint

    def test_bad_base_url(self, monkeypatch):
        monkeypatch.setenv("ACME_API_KEY", "k-1")
        monkeypatch.setenv("ACME_ENDPOINT", "localhost:9999/v1")
        desc = ProviderDescriptor(
            name="acme",
            is_configured_check="acme_api_key",
            **_llm("openai_driver.OpenAIDriver", {"api_key": "acme_api_key", "endpoint": "acme_endpoint"}),
        )
        [row] = prov.provider_rows(descriptors=[desc])
        assert row.status == "error"
        assert "ACME_ENDPOINT" in row.message and row.fix_hint.startswith("set ACME_ENDPOINT")

    def test_good_base_url_is_the_active_backend(self, monkeypatch):
        monkeypatch.setenv("ACME_API_KEY", "k-1")
        monkeypatch.setenv("ACME_ENDPOINT", "https://user:pw@api.acme.test/v1")
        desc = ProviderDescriptor(
            name="acme",
            is_configured_check="acme_api_key",
            **_llm("openai_driver.OpenAIDriver", {"api_key": "acme_api_key", "endpoint": "acme_endpoint"}),
        )
        [row] = prov.provider_rows(descriptors=[desc])
        assert row.status == "ok"
        assert "api.acme.test" in row.active_backend and "pw@" not in row.active_backend

    def test_url_problem(self):
        assert prov.url_problem("https://api.example.com/v1") is None
        assert prov.url_problem("http://127.0.0.1:1234") is None
        assert "http" in prov.url_problem("ftp://x")
        assert prov.url_problem("https://") == "has no host"
        assert prov.url_problem("http://h:notaport") == "has an invalid port"

    def test_local_provider_without_endpoint_is_hidden(self, monkeypatch):
        monkeypatch.delenv("SELFHOST_BASE_URL", raising=False)
        desc = ProviderDescriptor(
            name="selfhost",
            always_available=True,
            **_llm("openai_driver.OpenAIDriver", {"base_url": "selfhost_base_url"}),
        )
        [row] = prov.provider_rows(descriptors=[desc])
        assert row.name == "unconfigured providers"

    def test_local_provider_with_endpoint_is_shown_offline(self, monkeypatch):
        monkeypatch.setenv("SELFHOST_BASE_URL", "http://127.0.0.1:47999")
        desc = ProviderDescriptor(
            name="selfhost",
            always_available=True,
            **_llm("openai_driver.OpenAIDriver", {"base_url": "selfhost_base_url"}),
        )
        [row] = prov.provider_rows(descriptors=[desc])
        assert row.status == "ok" and "not contacted" in row.message

    def test_bedrock_offline_never_uses_the_credential_chain(self, monkeypatch):
        for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(settings, "aws_access_key_id", None, raising=False)
        monkeypatch.setattr(settings, "aws_secret_access_key", None, raising=False)

        def chain(env=None):
            raise AssertionError("boto3 credential chain must not run offline")

        desc = ProviderDescriptor(name="bedrock", is_configured_fn=chain, **_llm("bedrock_driver.BedrockDriver", {}))
        [row] = prov.provider_rows(descriptors=[desc])
        assert row.name == "unconfigured providers"
        monkeypatch.setenv("AWS_PROFILE", "dev")
        assert prov._is_configured(desc) is True

    def test_registered_as_a_capability(self, registry, monkeypatch, openai_key):
        register_capability(prov.CAPABILITY_NAME, "providers", prov._check)
        monkeypatch.setattr(prov, "_descriptors", lambda: [_openai_desc()])
        report = check_all(only="providers")
        assert [(r.name, r.category) for r in report.rows] == [("openai", "providers")]


class TestProvidersLive:
    def test_live_models_list_reports_latency(self, openai_key, monkeypatch):
        seen = {}

        def list_models(cls, *, api_key=None, timeout=10, **kw):
            seen.update(api_key=api_key, timeout=timeout)
            return ["gpt-a", "gpt-b"]

        monkeypatch.setattr(OpenAIDriver, "list_models", classmethod(list_models))
        [row] = prov.provider_rows(live=True, descriptors=[_openai_desc()], timeout=3)
        assert row.status == "ok"
        assert "2 models" in row.message and "ms" in row.message
        assert row.details["model_count"] == 2 and row.details["latency_ms"] >= 0
        assert seen == {"api_key": openai_key, "timeout": 3}

    def test_offline_never_calls_list_models(self, openai_key, monkeypatch):
        def list_models(cls, **kw):
            raise AssertionError("network call while offline")

        monkeypatch.setattr(OpenAIDriver, "list_models", classmethod(list_models))
        [row] = prov.provider_rows(live=False, descriptors=[_openai_desc()])
        assert row.status == "ok"

    def test_live_failure_explains_why(self, openai_key, monkeypatch):
        def list_models(cls, *, api_key=None, timeout=10, **kw):
            logging.getLogger("prompture.driver").warning(
                "Model discovery: %s returned HTTP 401 - Invalid API Key", "https://api.openai.com/v1/models"
            )
            return None

        monkeypatch.setattr(OpenAIDriver, "list_models", classmethod(list_models))
        [row] = prov.provider_rows(live=True, descriptors=[_openai_desc()])
        assert row.status == "error"
        assert "HTTP 401" in row.message and "https://" not in row.message
        assert "OPENAI_API_KEY" in row.fix_hint

    def test_live_exception_is_an_error_row(self, openai_key, monkeypatch):
        def list_models(cls, *, timeout=10, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr(OpenAIDriver, "list_models", classmethod(list_models))
        [row] = prov.provider_rows(live=True, descriptors=[_openai_desc()])
        assert row.status == "error" and "refused" in row.message

    def test_live_timeout(self, openai_key, monkeypatch):
        def list_models(cls, *, timeout=10, **kw):
            time.sleep(1.0)
            return ["late"]

        monkeypatch.setattr(OpenAIDriver, "list_models", classmethod(list_models))
        monkeypatch.setattr(prov, "LIVE_GRACE_SECONDS", 0.0)
        [row] = prov.provider_rows(live=True, descriptors=[_openai_desc()], timeout=0.1)
        assert row.status == "timeout"

    def test_static_catalog_is_not_a_live_check(self, openai_key, monkeypatch):
        monkeypatch.setattr(OpenAIDriver, "list_models", classmethod(lambda cls, **kw: ["static"]))
        [row] = prov.provider_rows(live=True, descriptors=[_openai_desc()])
        assert row.status == "ok" and "no free live check" in row.message

    def test_unreachable_default_local_server_is_unconfigured(self, monkeypatch):
        monkeypatch.setattr(settings, "ollama_endpoint", "http://127.0.0.1:47998/api/generate", raising=False)
        monkeypatch.delenv("OLLAMA_ENDPOINT", raising=False)
        monkeypatch.setattr(prov, "_explicitly_set", lambda attr: False)
        from prompture.drivers.ollama_driver import OllamaDriver

        monkeypatch.setattr(OllamaDriver, "list_models", classmethod(lambda cls, *, timeout=5, **kw: None))
        desc = ProviderDescriptor(
            name="ollama",
            always_available=True,
            list_models_kwargs=[("endpoint", "ollama_endpoint", "OLLAMA_ENDPOINT")],
            **_llm("ollama_driver.OllamaDriver", {"endpoint": "ollama_endpoint"}),
        )
        [row] = prov.provider_rows(live=True, descriptors=[desc])
        assert row.status == "unconfigured"
        assert "ollama serve" in row.fix_hint


# ── Watch ─────────────────────────────────────────────────────────────────


def _update(available: bool) -> UpdateInfo:
    return UpdateInfo(installed="1.0.0", latest="1.1.0" if available else "1.0.0", update_available=available)


class TestWatch:
    def test_exit_code_contract(self, registry):
        _fake("a", "tools", "ok")
        healthy = check_all()
        assert watch_exit_code(healthy) == EXIT_OK == 0
        assert watch_exit_code(healthy, _update(True)) == EXIT_OK
        assert watch_exit_code(healthy, _update(True), fail_on_update=True) == EXIT_UPDATE == 2
        assert watch_exit_code(healthy, _update(False), fail_on_update=True) == EXIT_OK
        _fake("b", "binaries", "broken")
        broken = check_all()
        assert watch_exit_code(broken, _update(True), fail_on_update=True) == EXIT_BROKEN == 1
        _fake("b", "binaries", "error")
        assert watch_exit_code(check_all()) == EXIT_BROKEN

    def test_offline_update_check_never_fails(self, registry):
        _fake("a", "tools", "ok")
        offline = UpdateInfo(installed="1.0.0", source="offline", error="ConnectionError")
        assert watch_exit_code(check_all(), offline, fail_on_update=True) == EXIT_OK

    def test_run_watch_is_offline_and_skips_update_on_request(self, registry, monkeypatch):
        seen = []
        register_capability("t", "tools", lambda live: seen.append(live) or HealthStatus("t", "ok"))
        result = run_watch(update_check=False)
        assert seen == [False] and result.update is None and result.exit_code == 0
        assert result.to_dict()["schema"] == "prompture.watch/1"


# ── CLI ───────────────────────────────────────────────────────────────────


class TestCLI:
    def test_commands_exported(self):
        assert [c.name for c in COMMANDS] == ["doctor", "check-update", "watch"]

    def test_doctor_json(self, fake_caps):
        result = CliRunner().invoke(doctor, ["--json"])
        assert result.exit_code == 0  # doctor reports, it doesn't gate
        data = json.loads(result.output)
        assert data["schema"] == "prompture.doctor/1" and data["summary"]["worst"] == "broken"

    def test_doctor_only_and_table(self, fake_caps):
        result = CliRunner().invoke(doctor, ["--only", "binaries"])
        assert result.exit_code == 0
        assert "[binaries]" in result.output and "web_search" not in result.output

    def test_doctor_rejects_unknown_only(self, fake_caps):
        assert CliRunner().invoke(doctor, ["--only", "nope"]).exit_code == 2

    def test_doctor_verbose_reaches_providers(self, registry, monkeypatch):
        register_capability(prov.CAPABILITY_NAME, "providers", prov._check)
        monkeypatch.delenv("ACME_API_KEY", raising=False)
        monkeypatch.setattr(prov, "_descriptors", lambda: [_keyed("acme")])
        compact = json.loads(CliRunner().invoke(doctor, ["--json"]).output)
        verbose = json.loads(CliRunner().invoke(doctor, ["--json", "--verbose"]).output)
        assert [c["name"] for c in compact["checks"]] == ["unconfigured providers"]
        assert [c["name"] for c in verbose["checks"]] == ["acme"]

    @pytest.mark.parametrize(
        ("status", "available", "flags", "code"),
        [
            ("ok", False, [], 0),
            ("ok", True, [], 0),
            ("ok", True, ["--fail-on-update"], 2),
            ("broken", True, ["--fail-on-update"], 1),
            ("error", False, [], 1),
        ],
    )
    def test_watch_exit_codes(self, registry, monkeypatch, status, available, flags, code):
        _fake("thing", "tools", status)
        monkeypatch.setattr("prompture.infra.updates.check_for_update", lambda **kw: _update(available))
        result = CliRunner().invoke(watch, flags)
        assert result.exit_code == code, result.output
        assert "health:" in result.output and "update:" in result.output

    def test_watch_json_and_no_update_check(self, registry, monkeypatch):
        _fake("thing", "tools", "ok")

        def no_network(**kw):
            raise AssertionError("update check should be skipped")

        monkeypatch.setattr("prompture.infra.updates.check_for_update", no_network)
        result = CliRunner().invoke(watch, ["--json", "--no-update-check"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["schema"] == "prompture.watch/1" and data["update"] is None and data["exit_code"] == 0

    def test_check_update_json(self, monkeypatch):
        monkeypatch.setattr("prompture.infra.updates.check_for_update", lambda **kw: _update(True))
        result = CliRunner().invoke(check_update, ["--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["latest"] == "1.1.0" and data["upgrade_command"] == "pip install -U prompture"

    def test_check_update_text_with_highlights(self, monkeypatch):
        info = _update(True)
        info.highlights = [{"version": "1.1.0", "date": "2026-09-01", "url": None, "notes": ["Faster doctor"]}]
        monkeypatch.setattr("prompture.infra.updates.check_for_update", lambda **kw: info)
        result = CliRunner().invoke(check_update, [])
        assert result.exit_code == 0
        assert "1.1.0 is available" in result.output and "Faster doctor" in result.output


# ── Companion summary ─────────────────────────────────────────────────────


class TestCompanionSummary:
    def test_summary_shape_and_cache(self, registry, monkeypatch):
        calls = []
        register_capability("t", "binaries", lambda live: calls.append(live) or HealthStatus("t", "missing"))
        first = capabilities_summary()
        assert first["schema"] == "prompture.doctor.summary/1" and first["state"] == "ready"
        assert first["worst"] == "missing" and first["by_category"] == {"binaries": "missing"}
        assert first["counts"] == {"missing": 1} and first["ok"] is True
        capabilities_summary()
        assert calls == [False]  # cached, and offline
        capabilities_summary(max_age=0)
        assert calls == [False, False]

    def test_non_blocking_mode_returns_pending_then_ready(self, registry):
        register_capability("t", "tools", lambda live: HealthStatus("t", "ok"))
        first = capabilities_summary(wait=False)
        assert first == {"schema": "prompture.doctor.summary/1", "state": "pending"}
        deadline = time.time() + 5
        while time.time() < deadline:
            value = capabilities_summary(wait=False)
            if value.get("state") == "ready":
                break
            time.sleep(0.02)
        assert value["state"] == "ready" and value["worst"] == "ok"

    def test_summary_survives_a_failing_doctor(self, registry, monkeypatch):
        def boom(**kw):
            raise RuntimeError("nope")

        monkeypatch.setattr(core, "check_all", boom)
        assert capabilities_summary()["state"] == "error"
