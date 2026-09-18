"""Provider token counting makes only explicit requests, using complete inputs."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from prompture.infra.token_counting import (
    acount_request_tokens,
    aestimate_request_cost,
    count_request_tokens,
    estimate_request_cost,
)


def openai_driver(endpoint):
    return SimpleNamespace(
        model="gpt-5.5", client=SimpleNamespace(responses=SimpleNamespace(input_tokens=SimpleNamespace(count=endpoint)))
    )


class TestProviderTokenCounting:
    def test_openai_counts_system_tools_images_and_schema(self):
        endpoint = Mock(return_value=SimpleNamespace(input_tokens=321, _request_id="req_count"))
        messages = [
            {"role": "system", "content": "Be precise"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read it"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            },
        ]
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
        result = count_request_tokens(
            "openai/gpt-5.5",
            messages,
            tools=tools,
            options={"json_mode": True, "json_schema": schema, "max_tokens": 100},
            driver=openai_driver(endpoint),
        )
        request = endpoint.call_args.kwargs
        assert result.input_tokens == 321
        assert result.request_id == "req_count"
        assert request["input"][0]["role"] == "system"
        assert request["input"][1]["content"][1]["type"] == "input_image"
        assert request["tools"][0]["name"] == "lookup"
        assert request["text"]["format"]["schema"]["required"] == ["name"]
        assert "max_tokens" not in request
        assert "required" not in schema

    def test_claude_json_extraction_and_system(self):
        endpoint = Mock(return_value={"input_tokens": 99})
        driver = SimpleNamespace(client=SimpleNamespace(messages=SimpleNamespace(count_tokens=endpoint)))
        result = count_request_tokens(
            "claude/claude-sonnet-4-6",
            [{"role": "system", "content": "Extract"}, {"role": "user", "content": "Alice"}],
            options={"json_mode": True, "json_schema": {"type": "object"}},
            driver=driver,
        )
        assert result.input_tokens == 99
        request = endpoint.call_args.kwargs
        assert request["system"] == "Extract"
        assert request["tools"][0]["name"] == "extract_json"
        assert request["tool_choice"] == {"type": "tool", "name": "extract_json"}

    def test_claude_on_demand_client_closed_without_generation(self):
        endpoint = Mock(return_value={"input_tokens": 8})
        client = Mock()
        client.messages.count_tokens = endpoint
        with patch("anthropic.Anthropic", return_value=client):
            result = count_request_tokens(
                "claude/test", [{"role": "user", "content": "hi"}], driver=SimpleNamespace(api_key="test")
            )
        assert result.input_tokens == 8
        client.close.assert_called_once()
        client.messages.create.assert_not_called()

    def test_estimate_uses_count_and_explicit_pricing(self):
        endpoint = Mock(return_value={"input_tokens": 300000})
        with patch("prompture.infra.model_rates.get_model_rates", return_value={"input": 5, "output": 30}):
            result = estimate_request_cost(
                "openai/gpt-5.5",
                [],
                driver=openai_driver(endpoint),
                expected_completion_tokens=1000,
                options={"service_tier": "flex"},
            )
        assert result.total_cost == pytest.approx(3.045 / 2)
        assert result.token_counter == "provider"
        endpoint.assert_called_once()

    def test_unsupported_provider_fails_before_driver_creation(self):
        with patch("prompture.drivers.get_driver_for_model") as factory:
            with pytest.raises(ValueError, match="requires"):
                count_request_tokens("other/model", [])
        factory.assert_not_called()

    def test_older_sdk_clear_error(self):
        with pytest.raises(RuntimeError, match="upgrade"):
            count_request_tokens("openai/test", [], driver=SimpleNamespace(model="test", client=object()))

    def test_malformed_provider_count_rejected(self):
        with pytest.raises(ValueError, match="invalid"):
            count_request_tokens("openai/test", [], driver=openai_driver(Mock(return_value={"input_tokens": -1})))

    @pytest.mark.asyncio
    async def test_async_openai_and_claude(self):
        endpoint = AsyncMock(return_value={"input_tokens": 42})
        result = await acount_request_tokens("openai/gpt-5.5", [], driver=openai_driver(endpoint))
        assert result.input_tokens == 42
        endpoint.assert_awaited_once()
        claude_count = AsyncMock(return_value={"input_tokens": 55})
        driver = SimpleNamespace(client=SimpleNamespace(messages=SimpleNamespace(count_tokens=claude_count)))
        with patch("prompture.infra.model_rates.get_model_rates", return_value={"input": 3, "output": 15}):
            estimate = await aestimate_request_cost("claude/test", [], driver=driver, expected_completion_tokens=10)
        assert estimate.input_tokens == 55
        assert estimate.token_counter == "provider"
        claude_count.assert_awaited_once()


class TestOwnedClientCleanup:
    def test_missing_openai_client_preserves_endpoint_error(self):
        with patch("prompture.drivers.get_driver_for_model", return_value=SimpleNamespace(model="test", client=None)):
            with pytest.raises(RuntimeError, match="upgrade"):
                count_request_tokens("openai/test", [])

    def test_sync_conversion_error_closes_owned_client(self):
        client = SimpleNamespace(close=Mock())
        driver = SimpleNamespace(model="test", client=client)
        with patch("prompture.drivers.get_driver_for_model", return_value=driver):
            with patch("prompture.infra.token_counting._request", side_effect=ValueError("bad request")):
                with pytest.raises(ValueError, match="bad request"):
                    count_request_tokens("openai/test", [])
        client.close.assert_called_once()

    def test_conversion_error_preserves_supplied_client(self):
        client = SimpleNamespace(close=Mock())
        with patch("prompture.infra.token_counting._request", side_effect=ValueError("bad request")):
            with pytest.raises(ValueError, match="bad request"):
                count_request_tokens("openai/test", [], driver=SimpleNamespace(client=client))
        client.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_conversion_error_closes_owned_client(self):
        client = SimpleNamespace(close=AsyncMock())
        driver = SimpleNamespace(model="test", client=client)
        with patch("prompture.drivers.async_registry.get_async_driver_for_model", return_value=driver):
            with patch("prompture.infra.token_counting._request", side_effect=ValueError("bad request")):
                with pytest.raises(ValueError, match="bad request"):
                    await acount_request_tokens("openai/test", [])
        client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_missing_client_preserves_endpoint_error(self):
        with patch(
            "prompture.drivers.async_registry.get_async_driver_for_model",
            return_value=SimpleNamespace(model="test", client=None),
        ):
            with pytest.raises(RuntimeError, match="upgrade"):
                await acount_request_tokens("openai/test", [])


def test_openai_count_preserves_tool_selection():
    endpoint = Mock(return_value={"input_tokens": 20})
    count_request_tokens(
        "openai/gpt-5.5",
        [],
        driver=openai_driver(endpoint),
        options={
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            "parallel_tool_calls": False,
            "conversation": "conv_example",
            "timeout": 10,
        },
    )
    request = endpoint.call_args.kwargs
    assert request["tool_choice"] == {"type": "function", "name": "lookup"}
    assert request["parallel_tool_calls"] is False
    assert request["conversation"] == "conv_example"
    assert request["timeout"] == 10
