"""The conformance suite never talks to a provider: outbound sockets are refused."""

from __future__ import annotations

import socket

import pytest

_LOOPBACK = ("127.0.0.1", "::1", "localhost")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect

    def guarded(self: socket.socket, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        # asyncio's own self-pipe connects over loopback on Windows.
        if host not in _LOOPBACK:
            raise AssertionError(f"conformance test tried to reach {address!r}; a transport is not faked")
        return real_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded)

    # A pricing lookup made behind the guard cannot fetch models.dev and would
    # mark the rates as loaded-but-empty for every later test in the process.
    # Restoring the load state afterwards lets the next test fetch normally.
    from prompture.infra import model_rates

    monkeypatch.setattr(model_rates, "_data", model_rates._data)
    monkeypatch.setattr(model_rates, "_loaded", model_rates._loaded)
