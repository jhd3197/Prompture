"""Tests for local-vs-server environment detection."""

from __future__ import annotations

import pytest

from prompture.infra.environment import EnvironmentInfo, detect_environment


@pytest.fixture
def root(tmp_path):
    """An empty fake filesystem root (no /.dockerenv, no /proc)."""
    return tmp_path


def test_plain_desktop_is_local(root):
    info = detect_environment({"DISPLAY": ":0"}, root=root, platform="linux")
    assert isinstance(info, EnvironmentInfo)
    assert info.kind == "local"
    assert not info.is_server
    assert info.markers == []
    assert any("credentials.yaml" in s for s in info.suggestions)


@pytest.mark.parametrize("plat", ["win32", "darwin"])
def test_windows_and_macos_without_markers_are_local(root, plat):
    assert detect_environment({}, root=root, platform=plat).kind == "local"


def test_dockerenv_marker(root):
    (root / ".dockerenv").write_text("", encoding="utf-8")
    info = detect_environment({"DISPLAY": ":0"}, root=root, platform="linux")
    assert info.kind == "server"
    assert info.container
    assert "container: /.dockerenv" in info.markers
    assert any("PROMPTURE_PROXY" in s for s in info.suggestions)
    assert any("ephemeral" in s for s in info.suggestions)


def test_cgroup_marker(root):
    (root / "proc" / "1").mkdir(parents=True)
    (root / "proc" / "1" / "cgroup").write_text("0::/kubepods/besteffort/pod123\n", encoding="utf-8")
    info = detect_environment({"DISPLAY": ":0"}, root=root, platform="linux")
    assert info.container
    assert any("kubepods" in m for m in info.markers)


def test_kubernetes_env_marker(root):
    info = detect_environment({"KUBERNETES_SERVICE_HOST": "10.0.0.1"}, root=root, platform="win32")
    assert info.container and info.kind == "server"


@pytest.mark.parametrize(
    "env",
    [
        {"CI": "true"},
        {"GITHUB_ACTIONS": "true"},
        {"GITLAB_CI": "1"},
        {"JENKINS_URL": "http://ci"},
        {"TF_BUILD": "True"},
    ],
)
def test_ci_markers(root, env):
    info = detect_environment(env, root=root, platform="darwin")
    assert info.ci and info.kind == "server"
    assert any("masked environment secrets" in s for s in info.suggestions)


@pytest.mark.parametrize("value", ["0", "false", "", "no"])
def test_falsy_ci_values_are_ignored(root, value):
    assert not detect_environment({"CI": value}, root=root, platform="darwin").ci


def test_ssh_session(root):
    info = detect_environment({"SSH_CONNECTION": "1.2.3.4 5 6.7.8.9 22"}, root=root, platform="darwin")
    assert info.ssh and info.kind == "server"
    assert any("paste API keys" in s for s in info.suggestions)


def test_headless_linux(root):
    info = detect_environment({}, root=root, platform="linux")
    assert info.headless and info.kind == "server"
    assert detect_environment({"WAYLAND_DISPLAY": "wayland-0"}, root=root, platform="linux").kind == "local"


def test_to_dict_is_serialisable(root):
    import json

    info = detect_environment({"CI": "1"}, root=root, platform="linux")
    data = json.loads(json.dumps(info.to_dict()))
    assert data["kind"] == "server"
    assert data["ci"] is True


def test_real_environment_does_not_raise():
    info = detect_environment()
    assert info.kind in ("local", "server")
