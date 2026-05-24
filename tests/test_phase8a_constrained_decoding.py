"""Phase 8A-lite: vLLM-style ``guided_json`` + ``extra_body`` pass-through.

Two layers:

1. **Helper unit tests** — exercise
   ``prompture.drivers._openai_compat`` directly. Stdlib-only, fast.
2. **Driver wire-format tests** — stub the HTTP layer and assert that
   ``guided_json`` / ``guided_decoding_backend`` / extra-body fields
   land in the request payload in the right places for every
   OpenAI-compatible driver pair (sync + async).
"""

from __future__ import annotations

from typing import Any

import pytest

from prompture.drivers._openai_compat import (
    apply_guided_decoding,
    apply_structured_output,
    build_response_format,
    merge_extra_body,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


def _empty() -> dict[str, Any]:
    return {}


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestBuildResponseFormat:
    def test_shape(self) -> None:
        rf = build_response_format(SCHEMA)
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "extraction"
        assert rf["json_schema"]["strict"] is True
        # The schema must round-trip the user input (post-strict-patching).
        assert "properties" in rf["json_schema"]["schema"]


class TestApplyGuidedDecoding:
    def test_off_is_noop(self) -> None:
        payload = _empty()
        apply_guided_decoding(payload, json_schema=SCHEMA, guided_decoding=False)
        assert payload == {}

    def test_no_schema_is_noop(self) -> None:
        payload = _empty()
        apply_guided_decoding(payload, json_schema=None, guided_decoding=True)
        assert payload == {}

    def test_true_writes_guided_json_without_backend(self) -> None:
        payload = _empty()
        apply_guided_decoding(payload, json_schema=SCHEMA, guided_decoding=True)
        assert payload == {"guided_json": SCHEMA}

    def test_outlines_string_writes_backend(self) -> None:
        payload = _empty()
        apply_guided_decoding(payload, json_schema=SCHEMA, guided_decoding="outlines")
        assert payload["guided_json"] == SCHEMA
        assert payload["guided_decoding_backend"] == "outlines"

    def test_xgrammar_backend(self) -> None:
        payload = _empty()
        apply_guided_decoding(payload, json_schema=SCHEMA, guided_decoding="xgrammar")
        assert payload["guided_decoding_backend"] == "xgrammar"

    def test_unknown_backend_passes_through(self) -> None:
        # Don't reject — vLLM ships new backends sometimes.
        payload = _empty()
        apply_guided_decoding(payload, json_schema=SCHEMA, guided_decoding="future-backend")
        assert payload["guided_decoding_backend"] == "future-backend"


class TestMergeExtraBody:
    def test_no_extra_is_noop(self) -> None:
        payload = {"a": 1}
        merge_extra_body(payload, {})
        assert payload == {"a": 1}

    def test_non_dict_extra_ignored(self) -> None:
        payload = {"a": 1}
        merge_extra_body(payload, {"extra_body": "not-a-dict"})
        assert payload == {"a": 1}

    def test_extra_keys_overwrite(self) -> None:
        payload = {"a": 1, "model": "default"}
        merge_extra_body(payload, {"extra_body": {"a": 2, "min_p": 0.1}})
        # User extras WIN — that's the whole point of the escape hatch.
        assert payload["a"] == 2
        assert payload["min_p"] == 0.1
        assert payload["model"] == "default"


class TestApplyStructuredOutput:
    def test_full_pipeline(self) -> None:
        payload = {"model": "fake", "messages": []}
        apply_structured_output(
            payload,
            options={
                "json_mode": True,
                "json_schema": SCHEMA,
                "guided_decoding": "outlines",
                "extra_body": {"min_p": 0.05},
            },
        )
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["guided_json"] == SCHEMA
        assert payload["guided_decoding_backend"] == "outlines"
        assert payload["min_p"] == 0.05

    def test_loose_json_mode_without_schema(self) -> None:
        payload = {"model": "fake"}
        apply_structured_output(payload, options={"json_mode": True})
        assert payload["response_format"] == {"type": "json_object"}
        # No guided_json without a schema.
        assert "guided_json" not in payload


# ---------------------------------------------------------------------------
# Driver wire-format tests
# ---------------------------------------------------------------------------


class _StubResponse:
    """Minimal requests-Response stand-in."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


def _stub_chat_response() -> dict[str, Any]:
    return {
        "id": "stub",
        "model": "test",
        "choices": [{"message": {"role": "assistant", "content": '{"x": 1}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class TestOpenAICompatibleWireFormat:
    """Sync OpenAICompatibleDriver."""

    def _driver(self, monkeypatch: pytest.MonkeyPatch):
        from prompture.drivers import openai_compatible_driver as ocd

        captured: list[dict[str, Any]] = []

        def _post(url: str, *, headers: Any = None, json: dict[str, Any], **kw: Any) -> _StubResponse:
            captured.append(json)
            return _StubResponse(_stub_chat_response())

        monkeypatch.setattr(ocd.requests, "post", _post)
        driver = ocd.OpenAICompatibleDriver(
            endpoint="http://fake.example.com/v1",
            api_key="test",
            model="fake-model",
        )
        return driver, captured

    def test_guided_decoding_writes_vllm_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, captured = self._driver(monkeypatch)
        driver.generate(
            "anything",
            options={
                "json_mode": True,
                "json_schema": SCHEMA,
                "guided_decoding": "outlines",
            },
        )
        body = captured[0]
        assert body["guided_json"] == SCHEMA
        assert body["guided_decoding_backend"] == "outlines"
        assert body["response_format"]["type"] == "json_schema"

    def test_extra_body_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, captured = self._driver(monkeypatch)
        driver.generate(
            "anything",
            options={
                "json_mode": True,
                "json_schema": SCHEMA,
                "extra_body": {"min_p": 0.1, "guided_grammar": "S -> 'yes'"},
            },
        )
        body = captured[0]
        assert body["min_p"] == 0.1
        assert body["guided_grammar"] == "S -> 'yes'"

    def test_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, captured = self._driver(monkeypatch)
        driver.generate(
            "anything",
            options={"json_mode": True, "json_schema": SCHEMA},
        )
        # No guided_decoding flag → no vLLM fields.
        body = captured[0]
        assert "guided_json" not in body
        assert "guided_decoding_backend" not in body


class TestOpenRouterWireFormat:
    def _driver(self, monkeypatch: pytest.MonkeyPatch):
        import os

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from prompture.drivers import openrouter_driver as ord_

        captured: list[dict[str, Any]] = []

        def _post(url: str, *, headers: Any = None, json: dict[str, Any], **kw: Any) -> _StubResponse:
            captured.append(json)
            return _StubResponse(_stub_chat_response())

        monkeypatch.setattr(ord_.requests, "post", _post)
        driver = ord_.OpenRouterDriver(model="meta-llama/llama-3-8b-instruct")
        # Defensive: zero out os.getenv side-effects.
        _ = os.environ.get("OPENROUTER_API_KEY")
        return driver, captured

    def test_guided_decoding_in_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, captured = self._driver(monkeypatch)
        driver.generate(
            "anything",
            options={
                "json_mode": True,
                "json_schema": SCHEMA,
                "guided_decoding": True,
            },
        )
        body = captured[0]
        assert body["guided_json"] == SCHEMA
        # ``True`` without a backend string → no guided_decoding_backend key.
        assert "guided_decoding_backend" not in body

    def test_extra_body_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, captured = self._driver(monkeypatch)
        driver.generate(
            "anything",
            options={
                "json_mode": True,
                "json_schema": SCHEMA,
                "extra_body": {"provider": {"order": ["Hyperbolic", "Together"]}},
            },
        )
        body = captured[0]
        # OpenRouter-specific provider preference field rides through unmolested.
        assert body["provider"] == {"order": ["Hyperbolic", "Together"]}


class TestLMStudioWireFormat:
    def _driver(self, monkeypatch: pytest.MonkeyPatch):
        from prompture.drivers import lmstudio_driver as lmd

        captured: list[dict[str, Any]] = []

        def _post(url: str, *, headers: Any = None, json: dict[str, Any], **kw: Any) -> _StubResponse:
            captured.append(json)
            return _StubResponse(_stub_chat_response())

        monkeypatch.setattr(lmd.requests, "post", _post)
        driver = lmd.LMStudioDriver(endpoint="http://localhost:1234/v1/chat/completions")
        return driver, captured

    def test_guided_decoding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, captured = self._driver(monkeypatch)
        driver.generate(
            "x",
            options={
                "json_mode": True,
                "json_schema": SCHEMA,
                "guided_decoding": "lm-format-enforcer",
            },
        )
        body = captured[0]
        assert body["guided_json"] == SCHEMA
        assert body["guided_decoding_backend"] == "lm-format-enforcer"

    def test_response_format_still_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # LM Studio's native schema enforcement should still be in place
        # alongside the vLLM-style fields.
        driver, captured = self._driver(monkeypatch)
        driver.generate(
            "x",
            options={"json_mode": True, "json_schema": SCHEMA, "guided_decoding": True},
        )
        body = captured[0]
        assert body["response_format"]["type"] == "json_schema"
        assert body["guided_json"] == SCHEMA


# ---------------------------------------------------------------------------
# Async sibling smoke tests
# ---------------------------------------------------------------------------


class _AsyncStubResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


class _StubAsyncClient:
    """Capturing httpx.AsyncClient stand-in."""

    captured: list[dict[str, Any]] = []

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    async def __aenter__(self) -> _StubAsyncClient:
        return self

    async def __aexit__(self, *a: Any) -> None:
        return None

    async def post(self, url: str, *, headers: Any = None, json: dict[str, Any], **kw: Any) -> _AsyncStubResponse:
        type(self).captured.append(json)
        return _AsyncStubResponse(_stub_chat_response())


class TestAsyncDriversWireFormat:
    @pytest.mark.asyncio
    async def test_async_openai_compatible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from prompture.drivers import async_openai_compatible_driver as a_ocd

        _StubAsyncClient.captured = []
        monkeypatch.setattr(a_ocd.httpx, "AsyncClient", _StubAsyncClient)
        driver = a_ocd.AsyncOpenAICompatibleDriver(
            endpoint="http://fake/v1",
            api_key="t",
            model="m",
        )
        await driver.generate(
            "x",
            options={"json_mode": True, "json_schema": SCHEMA, "guided_decoding": "outlines"},
        )
        body = _StubAsyncClient.captured[0]
        assert body["guided_json"] == SCHEMA
        assert body["guided_decoding_backend"] == "outlines"

    @pytest.mark.asyncio
    async def test_async_openrouter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        from prompture.drivers import async_openrouter_driver as a_ord

        _StubAsyncClient.captured = []
        monkeypatch.setattr(a_ord.httpx, "AsyncClient", _StubAsyncClient)
        driver = a_ord.AsyncOpenRouterDriver(model="meta-llama/llama-3-8b-instruct")
        await driver.generate(
            "x",
            options={
                "json_mode": True,
                "json_schema": SCHEMA,
                "guided_decoding": True,
                "extra_body": {"provider": {"order": ["X"]}},
            },
        )
        body = _StubAsyncClient.captured[0]
        assert body["guided_json"] == SCHEMA
        assert body["provider"] == {"order": ["X"]}

    @pytest.mark.asyncio
    async def test_async_lmstudio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from prompture.drivers import async_lmstudio_driver as a_lmd

        _StubAsyncClient.captured = []
        monkeypatch.setattr(a_lmd.httpx, "AsyncClient", _StubAsyncClient)
        driver = a_lmd.AsyncLMStudioDriver(endpoint="http://localhost:1234/v1/chat/completions")
        await driver.generate(
            "x",
            options={"json_mode": True, "json_schema": SCHEMA, "guided_decoding": "xgrammar"},
        )
        body = _StubAsyncClient.captured[0]
        assert body["guided_json"] == SCHEMA
        assert body["guided_decoding_backend"] == "xgrammar"
