"""Rate-limit snapshots from provider response headers."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import httpx
import pytest

from prompture.infra.rate_limits import (
    LimitSnapshot,
    LimitWindow,
    add_rate_limits,
    attach_rate_limit_hook,
    capture_rate_limits,
    limits_from_response,
    parse_rate_limit_headers,
)

NOW = 1_800_000_000.0

# Header sets mirror the examples in each vendor's rate-limit documentation.
OPENAI_HEADERS = {
    "x-ratelimit-limit-requests": "60",
    "x-ratelimit-limit-tokens": "150000",
    "x-ratelimit-remaining-requests": "59",
    "x-ratelimit-remaining-tokens": "149984",
    "x-ratelimit-reset-requests": "1s",
    "x-ratelimit-reset-tokens": "6m0s",
}

GROQ_HEADERS = {
    "x-ratelimit-limit-requests": "14400",
    "x-ratelimit-limit-tokens": "18000",
    "x-ratelimit-remaining-requests": "14370",
    "x-ratelimit-remaining-tokens": "17997",
    "x-ratelimit-reset-requests": "2m59.56s",
    "x-ratelimit-reset-tokens": "7.66s",
}

ANTHROPIC_HEADERS = {
    "anthropic-ratelimit-requests-limit": "1000",
    "anthropic-ratelimit-requests-remaining": "999",
    "anthropic-ratelimit-requests-reset": "2027-01-15T08:00:30Z",
    "anthropic-ratelimit-input-tokens-limit": "2000000",
    "anthropic-ratelimit-input-tokens-remaining": "500000",
    "anthropic-ratelimit-input-tokens-reset": "2027-01-15T08:00:05+00:00",
    "anthropic-ratelimit-output-tokens-limit": "400000",
    "anthropic-ratelimit-output-tokens-remaining": "399000",
    "anthropic-ratelimit-output-tokens-reset": "2027-01-15T08:00:01Z",
}


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()


def _sdk_http(sdk, client_name: str):
    """The HTTP package an SDK builds on: ``httpx``, or ``httpx2`` in newer releases."""
    import importlib

    client = getattr(sdk, client_name)(api_key="test")
    for cls in type(client._client).__mro__:
        root = cls.__module__.partition(".")[0]
        if root in ("httpx", "httpx2"):
            return importlib.import_module(root)
    return httpx


class TestParseHeaders:
    def test_openai_windows_and_duration_resets(self):
        snap = parse_rate_limit_headers(OPENAI_HEADERS, now=NOW)
        assert snap is not None
        assert snap.windows["requests"] == LimitWindow(limit=60, remaining=59, resets_at=NOW + 1)
        assert snap.windows["tokens"] == LimitWindow(limit=150000, remaining=149984, resets_at=NOW + 360)
        assert snap.observed_at == NOW
        assert snap.source == "headers"

    def test_openai_project_token_window(self):
        headers = {
            "x-ratelimit-limit-project-tokens": "60000",
            "x-ratelimit-remaining-project-tokens": "57000",
            "x-ratelimit-reset-project-tokens": "3s",
        }
        snap = parse_rate_limit_headers(headers, now=NOW)
        assert snap is not None
        assert snap.windows["project_tokens"] == LimitWindow(limit=60000, remaining=57000, resets_at=NOW + 3)

    def test_groq_fractional_durations(self):
        snap = parse_rate_limit_headers(GROQ_HEADERS, now=NOW)
        assert snap is not None
        assert snap.windows["requests"].resets_at == pytest.approx(NOW + 179.56)
        assert snap.windows["tokens"].resets_at == pytest.approx(NOW + 7.66)

    def test_anthropic_rfc3339_resets_and_split_token_windows(self):
        snap = parse_rate_limit_headers(ANTHROPIC_HEADERS, now=NOW)
        assert snap is not None
        assert set(snap.windows) == {"requests", "input_tokens", "output_tokens"}
        assert snap.windows["requests"].resets_at == _ts("2027-01-15T08:00:30Z")
        assert snap.windows["input_tokens"] == LimitWindow(
            limit=2_000_000, remaining=500_000, resets_at=_ts("2027-01-15T08:00:05+00:00")
        )

    def test_anthropic_priority_tier_is_its_own_window(self):
        headers = {
            "anthropic-priority-input-tokens-limit": "100000",
            "anthropic-priority-input-tokens-remaining": "90000",
        }
        snap = parse_rate_limit_headers(headers, now=NOW)
        assert snap is not None
        assert snap.windows == {"priority_input_tokens": LimitWindow(limit=100000, remaining=90000)}

    def test_header_names_are_case_insensitive(self):
        snap = parse_rate_limit_headers({"X-RateLimit-Remaining-Requests": "5"}, now=NOW)
        assert snap is not None
        assert snap.windows["requests"].remaining == 5

    def test_httpx_headers_are_accepted(self):
        snap = parse_rate_limit_headers(httpx.Headers(OPENAI_HEADERS), now=NOW)
        assert snap is not None
        assert snap.windows["tokens"].limit == 150000

    @pytest.mark.parametrize(
        "headers",
        [
            None,
            {},
            {"content-type": "application/json"},
            {"retry-after": "20"},
            {"x-ratelimit-reset": "20"},  # bare reset has no window name
            {"x-ratelimit-remaining-requests": "not-a-number"},
            {"x-ratelimit-reset-requests": "soon"},
        ],
    )
    def test_nothing_recognized_returns_none(self, headers):
        assert parse_rate_limit_headers(headers, now=NOW) is None

    def test_numeric_reset_is_seconds_from_now(self):
        snap = parse_rate_limit_headers({"x-ratelimit-reset-requests": "12"}, now=NOW)
        assert snap is not None
        assert snap.windows["requests"].resets_at == NOW + 12


class TestSnapshot:
    def test_headroom_is_the_tightest_window(self):
        snap = parse_rate_limit_headers(ANTHROPIC_HEADERS, now=NOW)
        assert snap is not None
        assert snap.headroom == pytest.approx(0.25)
        assert snap.tightest_window == "input_tokens"

    def test_headroom_unknown_without_limit_and_remaining(self):
        snap = LimitSnapshot(windows={"requests": LimitWindow(remaining=3)}, observed_at=NOW)
        assert snap.headroom is None
        assert snap.tightest_window is None

    def test_fraction_is_clamped(self):
        assert LimitWindow(limit=10, remaining=15).fraction_remaining == 1.0
        assert LimitWindow(limit=0, remaining=0).fraction_remaining is None

    def test_dict_round_trip(self):
        snap = parse_rate_limit_headers(OPENAI_HEADERS, now=NOW)
        assert snap is not None
        data = snap.to_dict()
        assert data["headroom"] == pytest.approx(59 / 60)
        assert data["tightest_window"] == "requests"
        assert LimitSnapshot.from_dict(json.loads(json.dumps(data))) == snap

    def test_add_rate_limits_only_sets_key_when_known(self):
        meta: dict = {}
        add_rate_limits(meta, None)
        assert meta == {}
        add_rate_limits(meta, parse_rate_limit_headers(OPENAI_HEADERS, now=NOW))
        assert meta["rate_limits"]["windows"]["requests"]["remaining"] == 59

    def test_limits_from_response_ignores_test_doubles(self):
        assert limits_from_response(MagicMock()) is None
        assert limits_from_response(None) is None
        assert limits_from_response(httpx.Response(200, headers=OPENAI_HEADERS)) is not None


class TestCapture:
    def _client(self, headers: dict[str, str]) -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, headers=headers)))
        holder = MagicMock()
        holder._client = client
        attach_rate_limit_hook(holder)
        return client

    def test_hook_records_inside_capture_only(self):
        client = self._client(OPENAI_HEADERS)
        client.get("https://example.test/")  # no active capture: nothing to record into
        with capture_rate_limits() as limits:
            client.get("https://example.test/")
        assert limits.snapshot is not None
        assert limits.snapshot.windows["requests"].remaining == 59

    def test_last_response_wins(self):
        responses = iter([OPENAI_HEADERS, {**OPENAI_HEADERS, "x-ratelimit-remaining-requests": "40"}])
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, headers=next(responses))))
        holder = MagicMock()
        holder._client = client
        attach_rate_limit_hook(holder)
        with capture_rate_limits() as limits:
            client.get("https://example.test/")
            client.get("https://example.test/")
        assert limits.snapshot is not None
        assert limits.snapshot.windows["requests"].remaining == 40

    def test_attaching_twice_adds_one_hook(self):
        holder = MagicMock()
        holder._client = httpx.Client()
        attach_rate_limit_hook(holder)
        attach_rate_limit_hook(holder)
        assert len(holder._client.event_hooks["response"]) == 1

    def test_httpx2_clients_are_hooked(self):
        httpx2 = pytest.importorskip("httpx2")
        client = httpx2.Client(transport=httpx2.MockTransport(lambda r: httpx2.Response(200, headers=OPENAI_HEADERS)))
        holder = MagicMock()
        holder._client = client
        attach_rate_limit_hook(holder)
        with capture_rate_limits() as limits:
            client.get("https://example.test/")
        assert limits.snapshot is not None
        assert limits.snapshot.windows["requests"].remaining == 59

    def test_non_httpx_clients_are_left_alone(self):
        holder = MagicMock()
        attach_rate_limit_hook(holder)  # must not raise
        attach_rate_limit_hook(object())

    def test_async_hook_records_in_the_calling_task(self):
        async def run() -> LimitSnapshot | None:
            client = httpx.AsyncClient(
                transport=httpx.MockTransport(lambda r: httpx.Response(200, headers=ANTHROPIC_HEADERS))
            )
            holder = MagicMock()
            holder._client = client
            attach_rate_limit_hook(holder)
            with capture_rate_limits() as limits:
                await client.get("https://example.test/")
            await client.aclose()
            return limits.snapshot

        snap = asyncio.run(run())
        assert snap is not None
        assert snap.windows["output_tokens"].remaining == 399000


# ---------------------------------------------------------------------------
# End to end through the real SDKs, with the network replaced by MockTransport
# ---------------------------------------------------------------------------

CHAT_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}

ANTHROPIC_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5-20251001",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 1},
}


def _sse(chunks: list[dict]) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode()


def _openai_stream_body() -> bytes:
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o-mini"}
    return _sse(
        [
            {**base, "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {**base, "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}},
        ]
    )


class TestOpenAIDriver:
    @pytest.fixture
    def driver(self):
        openai = pytest.importorskip("openai")
        from prompture.drivers.openai_driver import OpenAIDriver

        h = _sdk_http(openai, "OpenAI")

        def handler(request: h.Request) -> h.Response:
            if json.loads(request.content).get("stream"):
                return h.Response(
                    200,
                    headers={**OPENAI_HEADERS, "content-type": "text/event-stream"},
                    content=_openai_stream_body(),
                )
            return h.Response(200, headers=OPENAI_HEADERS, json=CHAT_COMPLETION)

        driver = OpenAIDriver(api_key="sk-test", model="gpt-4o-mini")
        driver.client = openai.OpenAI(
            api_key="sk-test", http_client=h.Client(transport=h.MockTransport(handler)), max_retries=0
        )
        attach_rate_limit_hook(driver.client)
        return driver

    def test_constructor_attaches_hook(self):
        pytest.importorskip("openai")
        from prompture.drivers.openai_driver import OpenAIDriver

        driver = OpenAIDriver(api_key="sk-test")
        assert len(driver.client._client.event_hooks["response"]) == 1

    def test_generate_reports_rate_limits(self, driver):
        result = driver.generate("hello", {})
        limits = result["meta"]["rate_limits"]
        assert limits["windows"]["requests"] == {
            "limit": 60,
            "remaining": 59,
            "resets_at": pytest.approx(limits["observed_at"] + 1),
        }
        assert limits["tightest_window"] == "requests"

    def test_stream_reports_rate_limits(self, driver):
        events = list(driver.generate_messages_stream([{"role": "user", "content": "hello"}], {}))
        done = events[-1]
        assert done["type"] == "done"
        assert done["meta"]["rate_limits"]["windows"]["tokens"]["remaining"] == 149984

    def test_tool_stream_reports_rate_limits(self, driver):
        tools = [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}]
        events = list(driver.generate_messages_with_tools_stream([{"role": "user", "content": "hi"}], tools, {}))
        assert events[-1].usage["rate_limits"]["windows"]["requests"]["remaining"] == 59

    def test_mocked_sdk_client_adds_no_key(self):
        pytest.importorskip("openai")
        from prompture.drivers.openai_driver import OpenAIDriver

        driver = OpenAIDriver(api_key="sk-test", model="gpt-4o-mini")
        driver.client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="hi"))]
        resp.usage = MagicMock(prompt_tokens=3, completion_tokens=1, total_tokens=4)
        resp.model_dump.return_value = {}
        driver.client.chat.completions.create.return_value = resp
        assert "rate_limits" not in driver.generate("hello", {})["meta"]


class TestAsyncOpenAIDriver:
    def test_generate_reports_rate_limits(self):
        openai = pytest.importorskip("openai")
        from prompture.drivers.async_openai_driver import AsyncOpenAIDriver

        h = _sdk_http(openai, "AsyncOpenAI")

        async def run() -> dict:
            driver = AsyncOpenAIDriver(api_key="sk-test", model="gpt-4o-mini")
            driver.client = openai.AsyncOpenAI(
                api_key="sk-test",
                http_client=h.AsyncClient(
                    transport=h.MockTransport(lambda r: h.Response(200, headers=OPENAI_HEADERS, json=CHAT_COMPLETION))
                ),
                max_retries=0,
            )
            attach_rate_limit_hook(driver.client)
            return await driver.generate("hello", {})

        result = asyncio.run(run())
        assert result["meta"]["rate_limits"]["windows"]["tokens"]["limit"] == 150000


class TestClaudeDriver:
    @pytest.fixture
    def patched_anthropic(self, monkeypatch):
        anthropic = pytest.importorskip("anthropic")
        real = anthropic.Anthropic
        h = _sdk_http(anthropic, "Anthropic")

        def factory(**kwargs):
            transport = h.MockTransport(lambda r: h.Response(200, headers=ANTHROPIC_HEADERS, json=ANTHROPIC_MESSAGE))
            return real(**kwargs, http_client=h.Client(transport=transport), max_retries=0)

        monkeypatch.setattr(anthropic, "Anthropic", factory)
        return anthropic

    def test_generate_reports_rate_limits(self, patched_anthropic):
        from prompture.drivers.claude_driver import ClaudeDriver

        result = ClaudeDriver(api_key="sk-ant-test").generate("hello", {})
        limits = result["meta"]["rate_limits"]
        assert limits["tightest_window"] == "input_tokens"
        assert limits["headroom"] == pytest.approx(0.25)

    def test_async_generate_reports_rate_limits(self):
        anthropic = pytest.importorskip("anthropic")
        from prompture.drivers.async_claude_driver import AsyncClaudeDriver

        h = _sdk_http(anthropic, "AsyncAnthropic")

        async def run() -> dict:
            driver = AsyncClaudeDriver(api_key="sk-ant-test")
            driver.client = anthropic.AsyncAnthropic(
                api_key="sk-ant-test",
                http_client=h.AsyncClient(
                    transport=h.MockTransport(
                        lambda r: h.Response(200, headers=ANTHROPIC_HEADERS, json=ANTHROPIC_MESSAGE)
                    )
                ),
                max_retries=0,
            )
            attach_rate_limit_hook(driver.client)
            return await driver.generate("hello", {})

        result = asyncio.run(run())
        assert result["meta"]["rate_limits"]["windows"]["requests"]["remaining"] == 999


class TestGroqDriver:
    def test_generate_reports_rate_limits(self):
        groq = pytest.importorskip("groq")
        h = _sdk_http(groq, "Groq")
        from prompture.drivers.groq_driver import GroqDriver

        driver = GroqDriver(api_key="gsk-test", model="llama-3.1-8b-instant")
        driver.client = groq.Client(
            api_key="gsk-test",
            http_client=h.Client(
                transport=h.MockTransport(
                    lambda r: h.Response(
                        200, headers=GROQ_HEADERS, json={**CHAT_COMPLETION, "model": "llama-3.1-8b-instant"}
                    )
                )
            ),
            max_retries=0,
        )
        attach_rate_limit_hook(driver.client)
        result = driver.generate("hello", {})
        assert result["meta"]["rate_limits"]["windows"]["requests"]["limit"] == 14400


def _requests_response(headers: dict[str, str], body: bytes, content_type: str = "application/json"):
    import requests

    resp = requests.Response()
    resp.status_code = 200
    resp.headers.update({**headers, "content-type": content_type})
    resp._content = body
    resp._content_consumed = True
    resp.encoding = "utf-8"
    resp.url = "https://api.example.test/v1/chat/completions"
    return resp


class TestOpenAICompatibleDriver:
    def _driver(self):
        from prompture.drivers.openai_compatible_driver import OpenAICompatibleDriver

        return OpenAICompatibleDriver(api_key="k", model="some-model", endpoint="https://api.example.test/v1")

    def test_generate_reports_rate_limits(self, monkeypatch):
        import requests

        body = json.dumps({**CHAT_COMPLETION, "model": "some-model"}).encode()
        monkeypatch.setattr(requests, "post", lambda *a, **kw: _requests_response(OPENAI_HEADERS, body))
        result = self._driver().generate("hello", {})
        assert result["meta"]["rate_limits"]["windows"]["requests"]["remaining"] == 59

    def test_endpoint_without_headers_adds_no_key(self, monkeypatch):
        import requests

        body = json.dumps({**CHAT_COMPLETION, "model": "some-model"}).encode()
        monkeypatch.setattr(requests, "post", lambda *a, **kw: _requests_response({}, body))
        assert "rate_limits" not in self._driver().generate("hello", {})["meta"]

    def test_raw_http_tool_stream_reports_rate_limits(self, monkeypatch):
        import requests

        monkeypatch.setattr(
            requests,
            "post",
            lambda *a, **kw: _requests_response(GROQ_HEADERS, _openai_stream_body(), "text/event-stream"),
        )
        tools = [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}]
        events = list(
            self._driver().generate_messages_with_tools_stream([{"role": "user", "content": "hi"}], tools, {})
        )
        assert events[-1].usage["rate_limits"]["windows"]["requests"]["limit"] == 14400

    def test_async_generate_reports_rate_limits(self, monkeypatch):
        from prompture.drivers.async_openai_compatible_driver import AsyncOpenAICompatibleDriver

        real = httpx.AsyncClient
        transport = httpx.MockTransport(
            lambda r: httpx.Response(200, headers=OPENAI_HEADERS, json={**CHAT_COMPLETION, "model": "some-model"})
        )
        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real(*a, transport=transport, **kw))

        async def run() -> dict:
            driver = AsyncOpenAICompatibleDriver(
                api_key="k", model="some-model", endpoint="https://api.example.test/v1"
            )
            return await driver.generate("hello", {})

        result = asyncio.run(run())
        assert result["meta"]["rate_limits"]["windows"]["tokens"]["remaining"] == 149984


class TestAzureDriver:
    def test_generate_reports_rate_limits(self, monkeypatch):
        openai = pytest.importorskip("openai")
        h = _sdk_http(openai, "OpenAI")
        from prompture.drivers import azure_driver

        transport = h.MockTransport(lambda r: h.Response(200, headers=OPENAI_HEADERS, json=CHAT_COMPLETION))
        monkeypatch.setattr(
            azure_driver,
            "AzureOpenAI",
            lambda **kw: openai.AzureOpenAI(**kw, http_client=h.Client(transport=transport), max_retries=0),
        )
        driver = azure_driver.AzureDriver(
            api_key="k", endpoint="https://example.openai.azure.com", deployment_id="gpt-4o-mini", model="gpt-4o-mini"
        )
        result = driver.generate("hello", {})
        assert result["meta"]["rate_limits"]["windows"]["requests"]["remaining"] == 59
