"""Tests for prompture.infra.callbacks module."""

from __future__ import annotations

from prompture.infra.callbacks import DriverCallbacks


class TestDriverCallbacks:
    """Tests for DriverCallbacks dataclass."""

    def test_defaults_are_none(self):
        cb = DriverCallbacks()
        assert cb.on_request is None
        assert cb.on_response is None
        assert cb.on_error is None

    def test_with_callbacks(self):
        captured = []

        def on_req(info):
            captured.append(("request", info))

        def on_resp(info):
            captured.append(("response", info))

        def on_err(info):
            captured.append(("error", info))

        cb = DriverCallbacks(on_request=on_req, on_response=on_resp, on_error=on_err)
        assert cb.on_request is on_req
        assert cb.on_response is on_resp
        assert cb.on_error is on_err

        # Verify callables work
        cb.on_request({"prompt": "test"})
        cb.on_response({"text": "result"})
        cb.on_error({"error": Exception("fail")})

        assert len(captured) == 3
        assert captured[0][0] == "request"
        assert captured[1][0] == "response"
        assert captured[2][0] == "error"

    def test_partial_callbacks(self):
        """Only some callbacks provided."""
        cb = DriverCallbacks(on_response=lambda info: None)
        assert cb.on_request is None
        assert cb.on_response is not None
        assert cb.on_error is None
