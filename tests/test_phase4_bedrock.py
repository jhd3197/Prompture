"""Phase 4: AWS Bedrock driver tests.

All tests fully mock boto3 — the driver runs without boto3 installed in
the environment (we inject a fake module into ``sys.modules`` before the
driver is imported). The shape of the mocked responses mirrors the real
Bedrock InvokeModel return type so the driver's payload parsing is
exercised end-to-end.
"""

from __future__ import annotations

import importlib
import json
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

# ----------------------------------------------------------------------
# Fake boto3 injection — happens at module import time, before
# ``prompture.drivers.bedrock_driver`` is imported by any test.
# ----------------------------------------------------------------------


class _FakeClientError(Exception):
    pass


class _FakeBotoCoreError(Exception):
    pass


def _ensure_fake_boto3() -> MagicMock:
    """Install a fake ``boto3`` + ``botocore.exceptions`` module pair."""
    if "boto3" in sys.modules and getattr(sys.modules["boto3"], "_prompture_fake", False):
        return sys.modules["boto3"]

    fake_boto3 = MagicMock()
    fake_boto3._prompture_fake = True
    fake_boto3.client = MagicMock(return_value=MagicMock())
    sys.modules["boto3"] = fake_boto3

    fake_botocore = MagicMock()
    fake_botocore_exceptions = MagicMock()
    fake_botocore_exceptions.ClientError = _FakeClientError
    fake_botocore_exceptions.BotoCoreError = _FakeBotoCoreError
    sys.modules.setdefault("botocore", fake_botocore)
    sys.modules["botocore.exceptions"] = fake_botocore_exceptions
    fake_botocore.exceptions = fake_botocore_exceptions
    return fake_boto3


_FAKE_BOTO3 = _ensure_fake_boto3()

# Force a fresh import of the bedrock driver so it picks up the fake boto3.
for _mod in [
    "prompture.drivers.bedrock_driver",
    "prompture.drivers.async_bedrock_driver",
]:
    if _mod in sys.modules:
        del sys.modules[_mod]

bedrock_driver = importlib.import_module("prompture.drivers.bedrock_driver")
async_bedrock_driver = importlib.import_module("prompture.drivers.async_bedrock_driver")

BedrockDriver = bedrock_driver.BedrockDriver
AsyncBedrockDriver = async_bedrock_driver.AsyncBedrockDriver

# The driver registry caches the resolved BedrockDriver class at import time
# (before our fake boto3 was installed). Replace the cached entry with our
# fresh class so registry lookups (``get_driver_for_model``) work correctly.
from prompture.drivers import ASYNC_PROVIDER_DRIVER_MAP, PROVIDER_DRIVER_MAP  # noqa: E402
from prompture.drivers import provider_descriptors as _pd  # noqa: E402

_pd._cls_cache["bedrock_driver.BedrockDriver"] = BedrockDriver
_pd._cls_cache["async_bedrock_driver.AsyncBedrockDriver"] = AsyncBedrockDriver
if "bedrock" in PROVIDER_DRIVER_MAP:
    _kw, _dm = PROVIDER_DRIVER_MAP["bedrock"][1], PROVIDER_DRIVER_MAP["bedrock"][2]
    PROVIDER_DRIVER_MAP["bedrock"] = (BedrockDriver, _kw, _dm)
if "bedrock" in ASYNC_PROVIDER_DRIVER_MAP:
    _kw, _dm = ASYNC_PROVIDER_DRIVER_MAP["bedrock"][1], ASYNC_PROVIDER_DRIVER_MAP["bedrock"][2]
    ASYNC_PROVIDER_DRIVER_MAP["bedrock"] = (AsyncBedrockDriver, _kw, _dm)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _mock_response(payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    """Construct a Bedrock InvokeModel response shape with a mock body."""
    body = MagicMock()
    body.read.return_value = json.dumps(payload).encode("utf-8")
    return {
        "body": body,
        "ResponseMetadata": {
            "HTTPHeaders": headers or {},
        },
    }


@pytest.fixture
def fake_client():
    """Reset and return a fresh fake boto3 client for each test."""
    client = MagicMock()
    _FAKE_BOTO3.client.reset_mock(return_value=True, side_effect=True)
    _FAKE_BOTO3.client.return_value = client
    return client


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_anthropic_generate_basic(fake_client):
    fake_client.invoke_model.return_value = _mock_response(
        {
            "content": [{"type": "text", "text": "hello world"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }
    )
    drv = BedrockDriver(
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region="us-east-1",
        model="anthropic.claude-3-5-haiku-20241022-v1:0",
    )
    result = drv.generate("hello", {"max_tokens": 100})
    assert result["text"] == "hello world"
    assert result["meta"]["prompt_tokens"] == 10
    assert result["meta"]["completion_tokens"] == 5
    assert result["meta"]["total_tokens"] == 15
    assert result["meta"]["cost"] > 0
    # Verify body shape
    call_kwargs = fake_client.invoke_model.call_args.kwargs
    body = json.loads(call_kwargs["body"])
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["max_tokens"] == 100
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_anthropic_tool_use(fake_client):
    fake_client.invoke_model.return_value = _mock_response(
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                }
            ],
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "stop_reason": "tool_use",
        }
    )
    drv = BedrockDriver(
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region="us-east-1",
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    result = drv.generate_messages_with_tools(
        [{"role": "user", "content": "Weather in NYC?"}], tools, {"max_tokens": 200}
    )
    assert result["tool_calls"][0]["name"] == "get_weather"
    assert result["tool_calls"][0]["arguments"] == {"city": "NYC"}
    body = json.loads(fake_client.invoke_model.call_args.kwargs["body"])
    assert body["tools"][0]["name"] == "get_weather"
    assert body["tools"][0]["input_schema"]["type"] == "object"


def test_meta_llama_request_shape(fake_client):
    fake_client.invoke_model.return_value = _mock_response(
        {
            "generation": "The answer is 42.",
            "prompt_token_count": 8,
            "generation_token_count": 6,
            "stop_reason": "stop",
        }
    )
    drv = BedrockDriver(
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region="us-east-1",
        model="meta.llama3-1-8b-instruct-v1:0",
    )
    result = drv.generate("What is the meaning of life?", {"max_tokens": 50})
    assert result["text"] == "The answer is 42."
    assert result["meta"]["prompt_tokens"] == 8
    assert result["meta"]["completion_tokens"] == 6
    body = json.loads(fake_client.invoke_model.call_args.kwargs["body"])
    assert "prompt" in body
    assert body["max_gen_len"] == 50
    assert "user:" in body["prompt"]


def test_mistral_request_and_header_tokens(fake_client):
    fake_client.invoke_model.return_value = _mock_response(
        {"outputs": [{"text": "Bonjour!", "stop_reason": "stop"}]},
        headers={
            "x-amzn-bedrock-input-token-count": "12",
            "x-amzn-bedrock-output-token-count": "3",
        },
    )
    drv = BedrockDriver(
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region="us-east-1",
        model="mistral.mistral-large-2407-v1:0",
    )
    result = drv.generate("Hello", {"max_tokens": 80})
    assert result["text"] == "Bonjour!"
    assert result["meta"]["prompt_tokens"] == 12
    assert result["meta"]["completion_tokens"] == 3
    body = json.loads(fake_client.invoke_model.call_args.kwargs["body"])
    assert body["prompt"].startswith("<s>[INST]")
    assert body["max_tokens"] == 80


def test_cohere_request_shape(fake_client):
    fake_client.invoke_model.return_value = _mock_response(
        {"text": "Hi there.", "finish_reason": "COMPLETE"},
        headers={
            "x-amzn-bedrock-input-token-count": "7",
            "x-amzn-bedrock-output-token-count": "4",
        },
    )
    drv = BedrockDriver(
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region="us-east-1",
        model="cohere.command-r-plus-v1:0",
    )
    result = drv.generate("Hello", {"max_tokens": 40})
    assert result["text"] == "Hi there."
    assert result["meta"]["prompt_tokens"] == 7
    assert result["meta"]["completion_tokens"] == 4
    body = json.loads(fake_client.invoke_model.call_args.kwargs["body"])
    assert body["message"] == "Hello"
    assert body["max_tokens"] == 40


def test_titan_request_shape(fake_client):
    fake_client.invoke_model.return_value = _mock_response(
        {
            "results": [{"outputText": "Titan reply.", "completionReason": "FINISH", "tokenCount": 3}],
            "inputTextTokenCount": 5,
        }
    )
    drv = BedrockDriver(
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region="us-east-1",
        model="amazon.titan-text-express-v1",
    )
    result = drv.generate("Hello Titan", {"max_tokens": 60, "temperature": 0.5})
    assert result["text"] == "Titan reply."
    assert result["meta"]["prompt_tokens"] == 5
    assert result["meta"]["completion_tokens"] == 3
    body = json.loads(fake_client.invoke_model.call_args.kwargs["body"])
    assert "inputText" in body
    assert body["textGenerationConfig"]["maxTokenCount"] == 60
    assert body["textGenerationConfig"]["temperature"] == 0.5


def test_non_anthropic_tool_use_raises(fake_client):
    drv = BedrockDriver(
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region="us-east-1",
        model="meta.llama3-1-70b-instruct-v1:0",
    )
    with pytest.raises(NotImplementedError, match="only supported for Anthropic"):
        drv.generate_messages_with_tools(
            [{"role": "user", "content": "x"}], [{"type": "function", "function": {"name": "f"}}], {}
        )


def test_boto3_missing_raises(monkeypatch):
    """When boto3 is unavailable, instantiating the driver yields a clear error."""
    monkeypatch.setattr(bedrock_driver, "_BOTO3_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"pip install prompture\[bedrock\]"):
        BedrockDriver(
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region="us-east-1",
            model="anthropic.claude-3-haiku-20240307-v1:0",
        )


def test_registry_resolves_bedrock(fake_client):
    from prompture.drivers import get_driver_for_model

    drv = get_driver_for_model(
        "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region="us-east-1",
    )
    assert isinstance(drv, BedrockDriver)
    assert drv.model == "anthropic.claude-3-5-haiku-20241022-v1:0"


def test_unknown_vendor_raises(fake_client):
    drv = BedrockDriver(
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region="us-east-1",
        model="weirdvendor.foo-v1",
    )
    with pytest.raises(ValueError, match="Unsupported Bedrock vendor"):
        drv.generate("hi", {})


def test_async_driver_smoke(fake_client):
    import asyncio

    fake_client.invoke_model.return_value = _mock_response(
        {
            "content": [{"type": "text", "text": "async hi"}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
            "stop_reason": "end_turn",
        }
    )
    drv = AsyncBedrockDriver(
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region="us-east-1",
        model="anthropic.claude-3-haiku-20240307-v1:0",
    )
    result = asyncio.run(drv.generate("hi", {"max_tokens": 16}))
    assert result["text"] == "async hi"
    assert result["meta"]["prompt_tokens"] == 3


def test_request_body_error_includes_body(fake_client):
    """Mirror openrouter pattern: surface request body in errors."""
    fake_client.invoke_model.side_effect = _FakeClientError("boom")
    drv = BedrockDriver(
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region="us-east-1",
        model="anthropic.claude-3-haiku-20240307-v1:0",
    )
    with pytest.raises(RuntimeError, match="Request body"):
        drv.generate("hi", {"max_tokens": 5})
