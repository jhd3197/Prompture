"""probe_local_servers: which local model servers are up, and what they serve."""

from __future__ import annotations

import requests

from prompture.infra import local_servers


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def test_running_servers_list_their_models(monkeypatch):
    seen = []

    def get(url, timeout, headers):
        seen.append(url)
        if url.endswith("/api/tags"):
            return FakeResponse(200, {"models": [{"name": "qwen3:8b"}, {"name": "llama3.2:3b"}]})
        return FakeResponse(200, {"data": [{"id": "gemma-3-12b"}]})

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(local_servers.settings, "ollama_endpoint", "http://localhost:11434/api/generate")
    servers = {s.provider: s for s in local_servers.probe_local_servers()}
    assert servers["ollama"].running and servers["ollama"].models == ("qwen3:8b", "llama3.2:3b")
    assert servers["ollama"].url == "http://localhost:11434"
    assert servers["lmstudio"].models == ("gemma-3-12b",)
    assert sorted(seen) == ["http://127.0.0.1:1234/v1/models", "http://localhost:11434/api/tags"]


def test_a_server_that_is_down_or_odd_is_not_running(monkeypatch):
    def get(url, timeout, headers):
        if url.endswith("/api/tags"):
            raise requests.ConnectionError("refused")
        return FakeResponse(200, ValueError("not json"))

    monkeypatch.setattr(requests, "get", get)
    servers = {s.provider: s for s in local_servers.probe_local_servers()}
    assert not servers["ollama"].running and "not running" in servers["ollama"].detail
    assert not servers["lmstudio"].running and servers["lmstudio"].models == ()


def test_only_probes_the_ones_asked_for(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, timeout, headers: FakeResponse(503, {}))
    servers = local_servers.probe_local_servers(only=("lmstudio",))
    assert [(s.provider, s.running, s.detail) for s in servers] == [("lmstudio", False, "answered HTTP 503")]
