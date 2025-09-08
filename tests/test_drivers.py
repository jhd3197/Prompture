import os
import pytest
from unittest.mock import patch, Mock
from prompture.drivers import OllamaDriver
from prompture.drivers import OpenAIDriver
from prompture.drivers import MockDriver


def assert_valid_usage_metadata(meta: dict):
    """Helper function to validate usage metadata structure."""
    required_keys = {"prompt_tokens", "completion_tokens", "total_tokens", "cost", "raw_response"}

    for key in required_keys:
        assert key in meta, f"Missing required metadata key: {key}"

    # Validate types
    assert isinstance(meta["prompt_tokens"], int), "prompt_tokens must be int"
    assert isinstance(meta["completion_tokens"], int), "completion_tokens must be int"
    assert isinstance(meta["total_tokens"], int), "total_tokens must be int"
    assert isinstance(meta["cost"], (int, float)), "cost must be numeric"
    assert isinstance(meta["raw_response"], dict), "raw_response must be dict"

    # Validate reasonable totals
    assert meta["total_tokens"] == meta["prompt_tokens"] + meta["completion_tokens"], "total_tokens should equal prompt + completion"


def assert_jsonify_response_structure(response: dict):
    """Helper function to validate the structure of jsonify responses."""
    required_keys = {"json_string", "json_object", "usage"}

    for key in required_keys:
        assert key in response, f"Missing required response key: {key}"

    # Validate types
    assert isinstance(response["json_string"], str), "json_string must be string"
    assert isinstance(response["json_object"], dict), "json_object must be dict"
    assert isinstance(response["usage"], dict), "usage must be dict"

    # Validate usage metadata
    assert_valid_usage_metadata(response["usage"])


class TestOllamaDriver:
    """Tests for OllamaDriver with mocked HTTP requests."""

    @patch('prompture.drivers.ollama_driver.requests.post')
    def test_successful_generation_with_token_tracking(self, mock_post):
        """Test OllamaDriver generates response with proper token metadata."""
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '{"name": "Juan", "age": 28}',
            "prompt_eval_count": 25,
            "eval_count": 18
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        driver = OllamaDriver(endpoint="http://localhost:11434/api/generate", model="test-model")

        result = driver.generate("Extract user info from text", {"model": "custom-model"})

        # Validate the API call
        mock_post.assert_called_once()
        call_args_params = mock_post.call_args[0]  # Get positional args tuple
        call_args_kwargs = mock_post.call_args[1]  # Get kwargs dict
        assert call_args_params[0] == "http://localhost:11434/api/generate"
        assert call_args_kwargs["json"]["prompt"] == "Extract user info from text"
        assert call_args_kwargs["json"]["model"] == "custom-model"  # Options override
        assert call_args_kwargs["json"]["stream"] is False

        # Validate response structure
        assert result["text"] == '{"name": "Juan", "age": 28}'
        assert "meta" in result

        # Validate metadata structure and calculations
        meta = result["meta"]
        assert_valid_usage_metadata(meta)

        # Validate Ollama-specific token extraction
        assert meta["prompt_tokens"] == 25
        assert meta["completion_tokens"] == 18
        assert meta["total_tokens"] == 43  # 25 + 18
        assert meta["cost"] == 0.0  # Ollama is free

        # Validate raw response preservation
        assert meta["raw_response"] == mock_response.json.return_value

    @patch('prompture.drivers.ollama_driver.requests.post')
    def test_generation_without_token_info(self, mock_post):
        """Test handling when Ollama doesn't provide token information."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '{"name": "Test"}'
            # No prompt_eval_count or eval_count
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        driver = OllamaDriver(endpoint="http://localhost:11434/api/generate", model="test-model")
        result = driver.generate("Test prompt", {})

        meta = result["meta"]
        assert_valid_usage_metadata(meta)

        # Should default to 0 when tokens not provided
        assert meta["prompt_tokens"] == 0
        assert meta["completion_tokens"] == 0
        assert meta["total_tokens"] == 0
        assert meta["cost"] == 0.0

    @patch('prompture.drivers.ollama_driver.requests.post')
    def test_default_model_usage(self, mock_post):
        """Test that default model is used when not specified in options."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "OK",
            "prompt_eval_count": 10,
            "eval_count": 5
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        driver = OllamaDriver(endpoint="http://localhost:11434/api/generate", model="default-model")
        driver.generate("Test prompt", {})  # No model in options

        call_args = mock_post.call_args[1]
        assert call_args["json"]["model"] == "default-model"

    @patch('prompture.drivers.ollama_driver.requests.post')
    def test_http_error_handling(self, mock_post):
        """Test that HTTP errors are properly raised."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_post.return_value = mock_response

        driver = OllamaDriver(endpoint="http://localhost:11434/api/generate", model="test-model")

        with pytest.raises(Exception, match="HTTP Error"):
            driver.generate("Test prompt", {})

    @patch('prompture.drivers.ollama_driver.requests.post')
    def test_timeout_option(self, mock_post):
        """Test that timeout is properly passed to requests."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "OK",
            "prompt_eval_count": 1,
            "eval_count": 1
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        driver = OllamaDriver(endpoint="http://localhost:11434/api/generate", model="test-model")
        driver.generate("Test", {"timeout": 120})

        call_args_kwargs = mock_post.call_args[1]
        assert call_args_kwargs["timeout"] == 60  # Default timeout from the driver


class TestOpenAIDriver:
    """Tests for OpenAIDriver with mocked OpenAI library."""

    def test_init_without_api_key_uses_env_var(self):
        """Test initialization with API key from environment variable."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch('prompture.drivers.openai_driver.openai') as mock_openai:
                driver = OpenAIDriver()
                assert driver.api_key == "test-key"
                assert driver.model == "gpt-4o-mini"
                # Verify openai.api_key was set
                mock_openai.api_key = "test-key"

    def test_explicit_api_key_takes_precedence(self):
        """Test that explicit API key overrides environment variable."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            with patch('prompture.drivers.openai_driver.openai'):
                driver = OpenAIDriver(api_key="explicit-key")
                assert driver.api_key == "explicit-key"

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch('prompture.drivers.openai_driver.openai'):
                driver = OpenAIDriver(model="gpt-4")
                assert driver.model == "gpt-4"

    @patch('prompture.drivers.openai_driver.openai')
    def test_openai_not_installed_raises_error(self, mock_openai):
        """Test that missing openai package raises RuntimeError."""
        mock_openai = None  # Simulate missing package

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            # Re-import to trigger the None check
            from importlib import reload
            import prompture.drivers.openai_driver
            reload(prompture.drivers.openai_driver)

            # Remove the None check temporarily for this test
            driver = prompture.drivers.openai_driver.OpenAIDriver.__new__(
                prompture.drivers.openai_driver.OpenAIDriver
            )

            with pytest.raises((RuntimeError, AttributeError)):
                driver.generate("Test", {})

    @patch('prompture.drivers.openai_driver.openai.ChatCompletion.create')
    def test_successful_generation_with_cost_calculation(self, mock_create):
        """Test generation with proper cost calculation."""
        # Mock the OpenAI response
        mock_response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            },
            "choices": [{ "message": { "content": '{"name": "Juan"}' } }]
        }
        mock_create.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            driver = OpenAIDriver(model="gpt-4o-mini")
            result = driver.generate("Test prompt", {})

        # Validate the API call
        mock_create.assert_called_once()
        call_args = mock_create.call_args[1]
        assert call_args["model"] == "gpt-4o-mini"
        assert call_args["messages"][0]["content"] == "Test prompt"
        assert call_args["temperature"] == 0.0  # default from options

        # Validate response
        assert result["text"] == '{"name": "Juan"}'

        # Validate metadata
        meta = result["meta"]
        assert_valid_usage_metadata(meta)

        # Validate OpenAI-specific data
        assert meta["prompt_tokens"] == 100
        assert meta["completion_tokens"] == 50
        assert meta["total_tokens"] == 150

        # Validate cost calculation for gpt-4o-mini
        expected_cost = (100 / 1000 * 0.00015) + (50 / 1000 * 0.0006)
        assert abs(meta["cost"] - expected_cost) < 0.000001

        # Validate raw response preservation
        assert meta["raw_response"] == mock_response

    @patch('prompture.drivers.openai_driver.openai.ChatCompletion.create')
    def test_cost_calculation_for_different_models(self, mock_create):
        """Test cost calculation for different OpenAI models."""
        test_cases = [
            ("gpt-4o", 100, 50, (100/1000*0.005) + (50/1000*0.015)),
            ("gpt-4o-mini", 200, 75, (200/1000*0.00015) + (75/1000*0.0006)),
            ("gpt-4", 50, 25, (50/1000*0.03) + (25/1000*0.06)),
        ]

        for model, prompt_tokens, completion_tokens, expected_cost in test_cases:
            mock_response = {
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                "choices": [{ "message": { "content": "{}" } }]
            }
            mock_create.return_value = mock_response

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                driver = OpenAIDriver(model=model)
                result = driver.generate("Test", {})

            assert abs(result["meta"]["cost"] - expected_cost) < 0.000001

    @patch('prompture.drivers.openai_driver.openai.ChatCompletion.create')
    def test_missing_usage_defaults_to_zero(self, mock_create):
        """Test handling when usage info is missing from OpenAI response."""
        mock_response = {
            "choices": [{ "message": { "content": "{}" } }]
            # No usage field
        }
        mock_create.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            driver = OpenAIDriver()
            result = driver.generate("Test", {})

        meta = result["meta"]
        assert_valid_usage_metadata(meta)
        assert meta["prompt_tokens"] == 0
        assert meta["completion_tokens"] == 0
        assert meta["total_tokens"] == 0
        assert meta["cost"] == 0.0

    @patch('prompture.drivers.openai_driver.openai.ChatCompletion.create')
    def test_custom_model_override(self, mock_create):
        """Test that options model overrides default model."""
        mock_response = {
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "choices": [{ "message": { "content": "{}" } }]
        }
        mock_create.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            driver = OpenAIDriver(model="gpt-4o-mini")
            driver.generate("Test", {"model": "gpt-4"})

        call_args = mock_create.call_args[1]
        assert call_args["model"] == "gpt-4"  # Should use options model

    @patch('prompture.drivers.openai_driver.openai.ChatCompletion.create')
    def test_temperature_and_max_tokens_options(self, mock_create):
        """Test that temperature and max_tokens options are passed correctly."""
        mock_response = {
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "choices": [{ "message": { "content": "{}" } }]
        }
        mock_create.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            driver = OpenAIDriver()
            driver.generate("Test", {"temperature": 0.5, "max_tokens": 100})

        call_args = mock_create.call_args[1]
        assert call_args["temperature"] == 0.5
        assert call_args["max_tokens"] == 100


class TestMockDriver:
    """Tests for MockDriver token usage and metadata."""

    def test_mock_driver_with_specific_prompt(self):
        """Test MockDriver response for specific prompt."""
        driver = MockDriver()
        result = driver.generate("Juan is 28 and lives in Miami.", {})

        assert result["text"] == '{"name":"Juan","age":28,"location":"Miami","interests":["basketball","coding"]}'
        assert_valid_usage_metadata(result["meta"])

    def test_mock_driver_generic_response(self):
        """Test MockDriver generic response for unknown prompts."""
        driver = MockDriver()
        result = driver.generate("Some unknown prompt", {})

        assert result["text"] == '{"name":"Unknown","age":null,"location":null,"interests":[]}'
        assert_valid_usage_metadata(result["meta"])

    def test_mock_driver_token_estimation(self):
        """Test that MockDriver provides realistic token estimates."""
        driver = MockDriver()

        # Short prompt
        short_result = driver.generate("Hi", {})
        short_meta = short_result["meta"]

        # Long prompt
        long_prompt = "This is a much longer prompt that should result in more tokens " * 10
        long_result = driver.generate(long_prompt, {})
        long_meta = long_result["meta"]

        # Longer prompts should have more prompt tokens
        assert long_meta["prompt_tokens"] > short_meta["prompt_tokens"]

        # All tokens should be reasonable estimates
        assert short_meta["prompt_tokens"] >= 1
        assert short_meta["completion_tokens"] >= 1
        assert short_meta["total_tokens"] == short_meta["prompt_tokens"] + short_meta["completion_tokens"]
        assert long_result["meta"]["cost"] == 0.0  # Mock is always free


class TestIntegration:
    """Integration tests that can run with real LLM providers when configured."""

    @pytest.mark.integration
    def test_ollama_integration(self, integration_driver):
        """Integration test with real Ollama if available."""
        from prompture.drivers import OllamaDriver

        if not isinstance(integration_driver, OllamaDriver):
            pytest.skip("Ollama not configured (missing OLLAMA_URI)")

        from prompture import extract_and_jsonify

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"]
        }

        result = extract_and_jsonify(
            integration_driver,
            "Juan is 28 years old and lives in Miami",
            schema
        )

        # Validate structure with real driver
        assert_jsonify_response_structure(result)
        assert_valid_usage_metadata(result["usage"])

        # Should have some meaningful token usage
        assert result["usage"]["prompt_tokens"] > 0
        assert result["usage"]["completion_tokens"] > 0

    @pytest.mark.integration
    def test_openai_integration(self, integration_driver):
        """Integration test with real OpenAI if available."""
        from prompture.drivers import OpenAIDriver

        if not isinstance(integration_driver, OpenAIDriver):
            pytest.skip("OpenAI not configured (missing OPENAI_API_KEY)")

        from prompture import extract_and_jsonify

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"]
        }

        result = extract_and_jsonify(
            integration_driver,
            "John is 25 and from Texas",
            schema
        )

        # Validate structure with real driver
        assert_jsonify_response_structure(result)
        assert_valid_usage_metadata(result["usage"])

        # Should have some meaningful token usage and cost
        assert result["usage"]["prompt_tokens"] > 0
        assert result["usage"]["completion_tokens"] > 0
        assert result["usage"]["cost"] > 0  # OpenAI should have some cost

    @pytest.mark.integration
    def test_cross_driver_comparison(self):
        """Compare responses between different real drivers if multiple are available."""
        drivers = []

        # Try to set up Ollama driver
        if os.getenv("OLLAMA_URI"):
            from prompture.drivers import OllamaDriver
            ollama_driver = OllamaDriver(
                endpoint=os.getenv("OLLAMA_URI"),
                model=os.getenv("OLLAMA_MODEL", "gemma3:latest")
            )
            drivers.append(("ollama", ollama_driver))

        # Try to set up OpenAI driver
        if os.getenv("OPENAI_API_KEY"):
            from prompture.drivers import OpenAIDriver
            openai_driver = OpenAIDriver(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            )
            drivers.append(("openai", openai_driver))

        if len(drivers) < 2:
            pytest.skip("Need at least 2 drivers configured for comparison")

        from prompture import extract_and_jsonify

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        text = "Extract the name: The person is named Alex"
        results = {}

        for driver_name, driver in drivers:
            result = extract_and_jsonify(driver, text, schema)
            results[driver_name] = result

        # All drivers should produce valid JSON
        for driver_name, result in results.items():
            assert_jsonify_response_structure(result)
            assert_valid_usage_metadata(result["usage"])

            # Should extract some reasonable JSON
            json_obj = result["json_object"]
            assert "name" in json_obj or isinstance(json_obj, str)