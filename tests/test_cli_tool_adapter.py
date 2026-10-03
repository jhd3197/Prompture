"""Tests for the CLI tool adapter (prompture.tools.cli)."""

from __future__ import annotations

import json
import logging
import os
import sys

import pytest

from prompture.agents.tools_schema import ToolRegistry
from prompture.capabilities.probe import ProbeResult
from prompture.tools.cli import (
    CLIArg,
    CLICommand,
    CLITool,
    CLIToolError,
    adapter,
    gh_tool,
    load_cli_tools,
    resolve_cli_tools,
    vtt_to_text,
    yt_dlp_tool,
)
from prompture.tools.cli import config as cli_config
from prompture.tools.cli.adapter import ProcessOutput

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class FakeRunner:
    """Stand-in for adapter.run_process that records calls."""

    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", exit_code: int = 0, **extra):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.extra = extra
        self.calls: list[dict] = []
        self.on_call = None

    def __call__(self, argv, *, env, timeout, max_bytes, cwd=None):
        self.calls.append({"argv": list(argv), "env": env, "timeout": timeout, "max_bytes": max_bytes, "cwd": cwd})
        if self.on_call:
            self.on_call(argv, cwd)
        return ProcessOutput(self.exit_code, self.stdout, self.stderr, **self.extra)


@pytest.fixture
def installed(monkeypatch):
    """Pretend every binary is on PATH and probes ok."""
    monkeypatch.setattr(adapter.shutil, "which", lambda cmd: f"/usr/bin/{os.path.basename(cmd)}")
    monkeypatch.setattr(
        adapter,
        "cached_probe",
        lambda cmd, args=("--version",), **kw: ProbeResult("ok", cmd, path=f"/usr/bin/{cmd}", output=f"{cmd} 1.0"),
    )


@pytest.fixture
def runner(monkeypatch, installed):
    fake = FakeRunner()
    monkeypatch.setattr(adapter, "run_process", fake)
    return fake


@pytest.fixture(autouse=True)
def _no_user_config(monkeypatch, tmp_path):
    """Keep the developer's own .prompture/tools.* out of the tests."""
    monkeypatch.setattr(cli_config, "config_paths", lambda cwd=None, include_home=True: [])


def _simple_tool(**kw) -> CLITool:
    return CLITool(
        name="demo",
        command="demo",
        commands=[
            CLICommand(
                "show",
                ("show",),
                args=[
                    CLIArg("target", required=True),
                    CLIArg("verbose", type="boolean", flag="-v"),
                    CLIArg("level", flag="-l"),
                ],
            )
        ],
        **kw,
    )


# ---------------------------------------------------------------------------
# argv building
# ---------------------------------------------------------------------------


class TestArgvBuilding:
    def test_gh_search_puts_query_after_end_of_options(self):
        argv = gh_tool().build_argv("search_repos", {"query": "-label:bug llm", "limit": 3, "sort": "stars"})
        assert argv[:3] == ["gh", "search", "repos"]
        assert "--limit=3" in argv and "--sort=stars" in argv
        assert argv[-2:] == ["--", "-label:bug llm"]
        assert argv.index("--json") < argv.index("--")

    def test_gh_issue_list_flags_are_single_items(self):
        argv = gh_tool().build_argv("issue list", {"repo": "cli/cli", "state": "open", "search": "--evil"})
        assert "--repo=cli/cli" in argv
        # a long-flag value can never become its own option
        assert "--search=--evil" in argv and "--evil" not in argv
        assert "--" not in argv  # no positionals

    def test_defaults_applied_and_integers_rendered(self):
        argv = gh_tool().build_argv("issue_view", {"number": "42", "repo": "a/b"})
        assert argv[-2:] == ["--", "42"]

    def test_yt_dlp_search_template(self):
        argv = yt_dlp_tool().build_argv("search", {"query": "lofi beats", "max_results": 3})
        assert argv[-2:] == ["--", "ytsearch3:lofi beats"]
        assert "--ignore-config" in argv
        argv = yt_dlp_tool().build_argv("search", {"query": "x"})
        assert argv[-1] == "ytsearch5:x"

    def test_yt_dlp_metadata(self):
        argv = yt_dlp_tool().build_argv("metadata", {"url": "https://www.youtube.com/watch?v=abc"})
        assert {"--dump-json", "--skip-download", "--no-playlist"} <= set(argv)
        assert argv[-2:] == ["--", "https://www.youtube.com/watch?v=abc"]

    def test_boolean_and_short_flag(self):
        argv = _simple_tool().build_argv("show", {"target": "x", "verbose": True, "level": "3"})
        assert argv == ["demo", "show", "-v", "-l", "3", "x"]
        assert _simple_tool().build_argv("show", {"target": "x", "verbose": False}) == ["demo", "show", "x"]

    def test_array_argument_repeats_flag(self):
        tool = CLITool(
            name="t",
            command="t",
            commands=[CLICommand("run", args=[CLIArg("tag", type="array", flag="--tag", pattern=r"[a-z]+")])],
        )
        assert tool.build_argv("run", {"tag": ["a", "b"]}) == ["t", "--tag=a", "--tag=b"]

    def test_schema_matches_declaration(self):
        td = gh_tool().to_tool_definition("issue_list")
        props = td.parameters["properties"]
        assert td.name == "gh_issue_list"
        assert props["state"]["enum"] == ["open", "closed", "all"]
        assert props["limit"]["maximum"] == 100
        assert td.parameters["required"] == ["repo"]
        assert td.metadata["is_write"] is False


# ---------------------------------------------------------------------------
# rejection
# ---------------------------------------------------------------------------


class TestRejection:
    def test_positional_flag_injection_rejected(self):
        with pytest.raises(CLIToolError, match="may not start with '-'"):
            _simple_tool().build_argv("show", {"target": "--output=/etc/passwd"})

    def test_short_flag_value_injection_rejected(self):
        with pytest.raises(CLIToolError, match="may not start with '-'"):
            _simple_tool().build_argv("show", {"target": "x", "level": "-rf"})

    def test_pattern_rejects_flag_like_repo(self):
        with pytest.raises(CLIToolError, match="allowed format"):
            gh_tool().build_argv("repo_view", {"repo": "--help"})

    def test_yt_dlp_rejects_non_http(self):
        for bad in ("-o /tmp/x", "file:///etc/passwd", "--exec rm"):
            with pytest.raises(CLIToolError):
                yt_dlp_tool().build_argv("metadata", {"url": bad})

    def test_allow_dash_requires_end_of_options(self):
        with pytest.raises(CLIToolError, match="does not end options"):
            CLICommand("bad", args=[CLIArg("q", allow_dash=True)])

    def test_nul_and_length(self):
        with pytest.raises(CLIToolError, match="NUL"):
            _simple_tool().build_argv("show", {"target": "a\x00b"})
        with pytest.raises(CLIToolError, match="longer than"):
            _simple_tool().build_argv("show", {"target": "a" * 5000})

    def test_unknown_and_missing_arguments(self):
        with pytest.raises(CLIToolError, match="Unknown argument"):
            _simple_tool().build_argv("show", {"target": "x", "shell": "rm"})
        with pytest.raises(CLIToolError, match="missing required"):
            _simple_tool().build_argv("show", {})

    def test_enum_and_bounds(self):
        with pytest.raises(CLIToolError, match="not one of"):
            gh_tool().build_argv("issue_list", {"repo": "a/b", "state": "deleted"})
        with pytest.raises(CLIToolError, match="<= 100"):
            gh_tool().build_argv("issue_list", {"repo": "a/b", "limit": 1000})

    def test_allowlist_rejects_before_execution(self, runner):
        result = gh_tool().run("repo delete", repo="a/b")
        assert not result.ok and "not an allowed command" in result.error
        assert "gh repo view (repo_view)" in result.error
        assert runner.calls == []
        assert gh_tool().call("auth token").startswith("Error: ")

    def test_gh_api_is_get_only(self):
        tool = gh_tool()
        argv = tool.build_argv("api_get", {"endpoint": "repos/cli/cli/releases/latest"})
        assert argv[1:4] == ["api", "--method", "GET"]
        assert not any(a in ("-f", "-F", "--input", "-X", "--field", "--raw-field") for a in argv)
        assert "jq" in tool.to_tool_definition("api_get").parameters["properties"]
        with pytest.raises(CLIToolError, match="GraphQL"):
            tool.build_argv("api_get", {"endpoint": "graphql"})
        with pytest.raises(CLIToolError, match=r"'\.\.'"):
            tool.build_argv("api_get", {"endpoint": "repos/../user"})
        with pytest.raises(CLIToolError):
            tool.build_argv("api_get", {"endpoint": "https://evil.example/x"})
        with pytest.raises(CLIToolError):
            tool.build_argv("api_get", {"endpoint": "-XPOST"})

    def test_shell_programs_refused(self):
        for shell in ("bash", "/bin/sh", "cmd.exe", "powershell", "pwsh"):
            with pytest.raises(CLIToolError, match="shell"):
                CLITool(name="x", command=shell, commands=[CLICommand("run")])

    def test_batch_launcher_refused(self, monkeypatch):
        monkeypatch.setattr(adapter.shutil, "which", lambda cmd: r"C:\tools\demo.cmd")
        monkeypatch.setattr(adapter.os, "name", "nt")
        result = _simple_tool().run("show", target="x")
        assert "batch launcher" in result.error
        assert _simple_tool().check().status == "broken"

    def test_missing_binary(self, monkeypatch):
        monkeypatch.setattr(adapter.shutil, "which", lambda cmd: None)
        result = gh_tool().run("search_repos", query="x")
        assert "not installed" in result.error and "cli.github.com" in result.error


# ---------------------------------------------------------------------------
# execution: real child processes (python itself), no network
# ---------------------------------------------------------------------------


def _py_tool(code: str, **kw) -> CLITool:
    return CLITool(name="py", command=sys.executable, commands=[CLICommand("run", fixed_args=("-c", code))], **kw)


class TestExecution:
    def test_timeout_kills_child(self):
        result = _py_tool("import time; time.sleep(30)", timeout=0.5).run("run")
        assert result.timed_out
        assert "timed out after 0.5s" in result.error

    def test_output_cap(self):
        result = _py_tool("import sys; sys.stdout.write('x' * 500000)", max_output_bytes=1000).run("run")
        assert result.ok and result.truncated
        assert result.output.startswith("x" * 1000)
        assert result.output.endswith("[output truncated at 1000 bytes]")
        assert len(result.output) < 1100

    def test_exit_code_reported(self):
        result = _py_tool("import sys; sys.stderr.write('boom'); sys.exit(3)").run("run")
        assert result.exit_code == 3 and "exited with code 3: boom" in result.error

    def test_utf8_output(self):
        result = _py_tool("print('caf\\u00e9 \\u2713')").run("run")
        assert result.output.strip() == "caf\u00e9 \u2713"

    def test_injected_env_reaches_child(self, monkeypatch):
        monkeypatch.setenv("CLI_TEST_SOURCE", "value-from-parent")
        tool = _py_tool(
            "import os; print(os.environ.get('CHILD_VAR', 'unset'))", env={"CHILD_VAR": "${CLI_TEST_SOURCE}"}
        )
        # the child sees the value, but it is scrubbed from what the model gets
        assert tool.run("run").output.strip() == "***"


# ---------------------------------------------------------------------------
# output parsing
# ---------------------------------------------------------------------------


class TestOutputParsing:
    def test_json_parsed_and_keys_selected(self, runner):
        runner.stdout = json.dumps({"id": "abc", "title": "T", "formats": [1] * 50, "duration": 12}).encode()
        result = yt_dlp_tool().run("metadata", url="https://youtu.be/abc")
        assert result.parsed == {"id": "abc", "title": "T", "duration": 12}
        assert json.loads(result.output) == result.parsed

    def test_jsonl_search(self, runner):
        lines = [{"id": str(i), "title": f"v{i}", "url": f"https://y/{i}", "junk": "x"} for i in range(3)]
        runner.stdout = "\n".join(json.dumps(x) for x in lines).encode()
        result = yt_dlp_tool().run("search", query="cats", max_results=3)
        assert [r["id"] for r in result.parsed] == ["0", "1", "2"]
        assert "junk" not in result.output
        assert runner.calls[0]["argv"][-1] == "ytsearch3:cats"
        assert runner.calls[0]["argv"][0] == "/usr/bin/yt-dlp"

    def test_unparseable_json_falls_back_to_text(self, runner):
        runner.stdout = b"v2.102.0\n"
        assert gh_tool().call("api_get", endpoint="repos/a/b", jq=".tag_name").strip() == "v2.102.0"

    def test_yaml_output(self, runner):
        pytest.importorskip("yaml")
        tool = CLITool(name="k", command="k", commands=[CLICommand("get", output="yaml")])
        runner.stdout = b"items:\n  - name: a\n  - name: b\n"
        assert tool.run("get").parsed == {"items": [{"name": "a"}, {"name": "b"}]}

    def test_subtitles_collected_from_temp_dir(self, runner):
        seen = {}

        def write_subs(argv, cwd):
            seen["cwd"] = cwd
            assert argv[argv.index("--paths") + 1] == cwd
            with open(os.path.join(cwd, "abc.en.vtt"), "w", encoding="utf-8") as fh:
                fh.write("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhello <c>world</c>\n\n")
                fh.write("00:00:02.000 --> 00:00:03.000\nhello world\n\n00:00:03.000 --> 00:00:04.000\nbye\n")

        runner.on_call = write_subs
        out = yt_dlp_tool().call("subtitles", url="https://youtu.be/abc")
        assert out == "## abc.en.vtt\nhello world\nbye"
        assert not os.path.exists(seen["cwd"])  # temp dir removed

    def test_subtitles_none_found(self, runner):
        runner.stderr = b"no subtitles"
        out = yt_dlp_tool().call("subtitles", url="https://youtu.be/abc")
        assert out.startswith("Error:") and "no output files" in out

    def test_vtt_to_text(self):
        assert vtt_to_text("WEBVTT\nKind: captions\n\n1\n00:00.000 --> 00:01.000\nHi &amp; bye\n") == "Hi & bye"


# ---------------------------------------------------------------------------
# secrets
# ---------------------------------------------------------------------------


class TestSecrets:
    SECRET = "s3cr3t-value-123456"

    def test_env_values_never_in_errors_or_logs(self, runner, monkeypatch, caplog):
        monkeypatch.setenv("SOURCE_TOKEN", self.SECRET)
        runner.exit_code = 1
        runner.stderr = f"auth failed for token {self.SECRET}".encode()
        tool = CLITool(
            name="svc",
            command="svc",
            env={"SVC_TOKEN": "$SOURCE_TOKEN", "SVC_MODE": "readonly"},
            commands=[CLICommand("status", ("status",), args=[CLIArg("name")])],
        )
        with caplog.at_level(logging.DEBUG, logger="prompture.tools.cli"):
            result = tool.run("status", name="x")
        assert runner.calls[0]["env"]["SVC_TOKEN"] == self.SECRET
        assert runner.calls[0]["env"]["PYTHONUTF8"] == "1"
        assert self.SECRET not in result.error and "***" in result.error
        assert self.SECRET not in result.stderr
        assert self.SECRET not in caplog.text
        assert "SVC_TOKEN" in caplog.text  # keys may be logged, values never

    def test_secret_in_stdout_scrubbed(self, runner, monkeypatch):
        monkeypatch.setenv("SOURCE_TOKEN", self.SECRET)
        runner.stdout = f"token={self.SECRET}".encode()
        tool = CLITool(name="svc", command="svc", env={"T": "${SOURCE_TOKEN}"}, commands=[CLICommand("run")])
        assert self.SECRET not in tool.call("run")

    def test_missing_env_reference_degrades_health(self, installed, monkeypatch):
        monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
        tool = CLITool(name="svc", command="svc", env={"T": "${NOT_SET_ANYWHERE}"}, commands=[CLICommand("run")])
        row = tool.check()
        assert row.status == "degraded" and "NOT_SET_ANYWHERE" in row.fix_hint


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


class TestHealth:
    @pytest.mark.parametrize("status", ["ok", "missing", "broken", "timeout", "error"])
    def test_check_maps_probe_status(self, monkeypatch, status):
        monkeypatch.setattr(adapter.shutil, "which", lambda cmd: None if status == "missing" else "/usr/bin/gh")
        probe = ProbeResult(
            status, "gh", path="/usr/bin/gh", output="gh version 2.0" if status == "ok" else "", hint="fix it"
        )
        monkeypatch.setattr(adapter, "cached_probe", lambda *a, **k: probe)
        row = gh_tool().check()
        assert row.name == "cli:gh" and row.category == "tools" and row.status == status
        assert (row.fix_hint is None) == (status == "ok")
        assert "gh_search_repos" in row.details["tools"]
        assert gh_tool().is_active() == (status == "ok")

    def test_live_check_runs_extra_probe(self, installed, monkeypatch):
        calls = []

        def fake_probe(cmd, args, **kw):
            calls.append(tuple(args))
            return ProbeResult("error", cmd, output="not logged in", hint="run gh auth login")

        monkeypatch.setattr(adapter, "probe_command", fake_probe)
        assert gh_tool().check(live=False).status == "ok"
        assert calls == []
        row = gh_tool().check(live=True)
        assert calls == [("auth", "status")]
        assert row.status == "degraded" and "live check failed" in row.message

    def test_rows_registered_for_doctor(self):
        from prompture.capabilities.health import list_capabilities
        from prompture.tools.cli import health

        names = health.register_cli_capabilities()
        assert {"cli:gh", "cli:yt-dlp"} <= set(names)
        registered = {c.name for c in list_capabilities(category="tools")}
        assert {"cli:gh", "cli:yt-dlp"} <= registered


# ---------------------------------------------------------------------------
# config files
# ---------------------------------------------------------------------------

_CONFIG = {
    "tools": [
        {
            "name": "kubectl",
            "command": "kubectl",
            "description": "Read-only Kubernetes queries",
            "version_args": ["version", "--client"],
            "env": {"KUBECONFIG": "${MY_KUBECONFIG}"},
            "commands": [
                {
                    "name": "get_pods",
                    "subcommand": "get pods",
                    "fixed_args": ["-o", "json"],
                    "output": "json",
                    "args": [{"name": "namespace", "flag": "--namespace", "pattern": "^[a-z0-9-]{1,63}$"}],
                }
            ],
        },
        {"name": "evil", "command": "bash", "commands": [{"name": "run"}]},
        {"name": "nocommands", "command": "ls"},
    ]
}


class TestConfig:
    def _project(self, tmp_path, name: str, text: str):
        d = tmp_path / ".prompture"
        d.mkdir(exist_ok=True)
        (d / name).write_text(text, encoding="utf-8")
        return tmp_path

    def test_json_config(self, tmp_path, monkeypatch):
        monkeypatch.undo()  # use the real config_paths
        root = self._project(tmp_path, "tools.json", json.dumps(_CONFIG))
        tools = load_cli_tools(cwd=root, include_home=False)
        assert [t.name for t in tools] == ["kubectl"]
        kubectl = tools[0]
        assert kubectl.build_argv("get_pods", {"namespace": "kube-system"}) == [
            "kubectl",
            "get",
            "pods",
            "-o",
            "json",
            "--namespace=kube-system",
        ]
        with pytest.raises(CLIToolError):
            kubectl.build_argv("delete", {})
        errors = cli_config.config_errors()
        assert any("shell" in e for e in errors) and any("at least one" in e for e in errors)

    def test_yaml_config(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        path = tmp_path / "tools.yaml"
        path.write_text(
            yaml.safe_dump({"jq": {"command": "jq", "commands": {"version": {"fixed_args": ["--version"]}}}})
        )
        tools = load_cli_tools(path)
        assert tools[0].name == "jq" and tools[0].tool_names() == ["jq_version"]

    def test_yaml_without_pyyaml(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "yaml", None)
        path = tmp_path / "tools.yaml"
        path.write_text("tools: []\n")
        assert load_cli_tools(path) == []
        assert "PyYAML" in cli_config.config_errors()[0]

    def test_unknown_keys_rejected(self):
        tools, errors = cli_config.parse_cli_tools(
            {"x": {"command": "x", "commands": [{"name": "a", "shell": True}]}}, source="t"
        )
        assert tools == [] and "unknown keys" in errors[0]

    def test_project_overrides_builtin(self, monkeypatch, installed):
        custom = CLITool(name="gh", command="gh", commands=[CLICommand("status", ("status",))])
        monkeypatch.setattr("prompture.tools.cli.load_cli_tools", lambda: [custom])
        assert [t.name for t in resolve_cli_tools("gh")] == ["gh_status"]


# ---------------------------------------------------------------------------
# cli: namespace
# ---------------------------------------------------------------------------


class TestResolve:
    def test_cli_gh_via_named_specs(self, installed):
        from prompture.tools.named import resolve_tool_spec

        names = [t.name for t in resolve_tool_spec("cli:gh")]
        assert "gh_search_repos" in names and "gh_api_get" in names
        assert all(n.startswith("gh_") for n in names)

    def test_aliases_and_all(self, installed):
        assert next(t.name for t in resolve_cli_tools("yt_dlp")).startswith("ytdlp_")
        all_names = {t.name for t in resolve_cli_tools("all")}
        assert "gh_issue_view" in all_names and "ytdlp_metadata" in all_names

    def test_inactive_tool_resolves_empty(self, monkeypatch, caplog):
        monkeypatch.setattr(adapter.shutil, "which", lambda cmd: None)
        monkeypatch.setattr(adapter, "cached_probe", lambda cmd, *a, **k: ProbeResult("missing", cmd, hint="install"))
        with caplog.at_level(logging.WARNING, logger="prompture.tools.cli"):
            assert resolve_cli_tools("gh") == []
        assert "not available" in caplog.text
        assert resolve_cli_tools("all") == []

    def test_unknown_tool(self):
        with pytest.raises(ValueError, match="Unknown CLI tool"):
            resolve_cli_tools("rm")

    def test_registry_execution(self, runner):
        runner.stdout = b'[{"fullName": "a/b"}]'
        registry = ToolRegistry()
        for td in resolve_cli_tools("gh"):
            registry.add(td)
        assert json.loads(registry.execute("gh_search_repos", {"query": "llm"})) == [{"fullName": "a/b"}]
        # schema validation rejects bad enums before the adapter is even reached
        out = registry.execute("gh_issue_list", {"repo": "a/b", "state": "deleted"})
        assert isinstance(out, str) and len(runner.calls) == 1
