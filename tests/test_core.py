import json
import os
from unittest.mock import Mock

import pytest
from prompture import (
    clean_json_text,
    clean_json_text_with_ai,
    ask_for_json,
    extract_and_jsonify
)


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


def assert_jsonify_response_structure(response: dict, schema_type: str = "object"):
    """
    Helper function to validate the structure of jsonify responses.
    
    Args:
        response: The response dictionary to validate
        schema_type: The expected schema type (default: "object")
    """
    required_keys = {"json_string", "json_object", "usage"}

    for key in required_keys:
        assert key in response, f"Missing required response key: {key}"

    # Validate types
    assert isinstance(response["json_string"], str), "json_string must be string"
    
    # Check json_object type based on schema_type
    if schema_type == "string":
        assert isinstance(response["json_object"], str), "json_object must be string when schema type is string"
    else:
        assert isinstance(response["json_object"], dict), "json_object must be dict"
        
    assert isinstance(response["usage"], dict), "usage must be dict"

    # Validate usage metadata
    assert_valid_usage_metadata(response["usage"])


class TestCleanJsonText:
    """Tests for clean_json_text function."""

    def test_clean_simple_json_string(self):
        """Test cleaning a simple JSON string without any markdown formatting."""
        input_text = '{"name": "Juan", "age": 28}'
        result = clean_json_text(input_text)
        assert result == input_text

    def test_clean_json_with_code_fence_basic(self):
        """Test cleaning JSON with basic markdown code fence."""
        input_text = '''```json
{"name": "Juan", "age": 28}
```'''
        expected = '{"name": "Juan", "age": 28}'
        result = clean_json_text(input_text)
        assert result == expected

    def test_clean_json_with_code_fence_with_language(self):
        """Test cleaning JSON with markdown code fence including language tag."""
        input_text = '''```json
{"name": "Juan", "age": 28}
```'''
        expected = '{"name": "Juan", "age": 28}'
        result = clean_json_text(input_text)
        assert result == expected

    def test_clean_json_with_partial_match(self):
        """Test extracting JSON from text that contains both JSON and other content."""
        input_text = 'Here is some data: {"name": "Juan", "age": 28} and some other text'
        expected = '{"name": "Juan", "age": 28}'
        result = clean_json_text(input_text)
        assert result == expected

    def test_clean_json_with_nested_objects(self):
        """Test cleaning complex JSON with nested objects and arrays."""
        input_text = '''```json
{
    "name": "Juan",
    "profile": {
        "age": 28,
        "location": "Miami",
        "interests": ["basketball", "coding"]
    }
}
```'''
        expected = '''{
    "name": "Juan",
    "profile": {
        "age": 28,
        "location": "Miami",
        "interests": ["basketball", "coding"]
    }
}'''
        result = clean_json_text(input_text)
        assert result == expected.strip()

    def test_clean_json_with_explanation_text(self):
        """Test cleaning JSON that comes after explanatory text."""
        input_text = '''I have extracted the following information:
The person's name is Juan and they are 28 years old.

```json
{"name": "Juan", "age": 28}
```

This information was extracted from the given text.'''
        expected = '{"name": "Juan", "age": 28}'
        result = clean_json_text(input_text)
        assert result == expected

    def test_clean_json_with_multiple_code_fences(self):
        """Test cleaning JSON when multiple code fences are present (should extract first)."""
        input_text = '''```json
{"name": "First"}
```

And here is the second:
```json
{"name": "Second"}
```'''
        expected = '{"name": "First"}'
        result = clean_json_text(input_text)
        assert result == expected

    def test_clean_json_fallback_to_partial_extraction(self):
        """Test fallback to partial JSON extraction when no code fence is found."""
        input_text = 'Some text before {"name": "Juan", "age": 28} and some text after'
        expected = '{"name": "Juan", "age": 28}'
        result = clean_json_text(input_text)
        assert result == expected

    def test_clean_empty_input(self):
        """Test cleaning empty input."""
        result = clean_json_text("")
        assert result == ""

    def test_clean_whitespace_only(self):
        """Test cleaning whitespace-only input."""
        result = clean_json_text("   ")
        assert result == ""  # Should strip to empty string

    def test_clean_no_json_found(self):
        """Test input with no JSON content."""
        input_text = "This is just plain text with no JSON"
        result = clean_json_text(input_text)
        assert result == input_text


class TestCleanJsonTextWithAi:
    """Tests for clean_json_text_with_ai function."""

    @pytest.mark.integration
    def test_clean_malformed_json_with_ai_help(self, integration_driver):
        """Test using AI to clean malformed JSON."""
        malformed_json = '{"name": "Juan", "age": 28, "interests": ["basketball"]'  # Missing closing brace
        from tests.conftest import DEFAULT_MODEL
        result = clean_json_text_with_ai(integration_driver, malformed_json, model_name=DEFAULT_MODEL)
        
        # Verify the result is valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert "age" in parsed
        assert "interests" in parsed

    @pytest.mark.integration
    def test_clean_already_valid_json(self, integration_driver):
        """Test that valid JSON is returned unchanged."""
        valid_json = '{"name": "Juan", "age": 30}'
        from tests.conftest import DEFAULT_MODEL
        result = clean_json_text_with_ai(integration_driver, valid_json, model_name=DEFAULT_MODEL)
        assert json.loads(result) == json.loads(valid_json)


class TestAskForJson:
    """Tests for ask_for_json function."""

    @pytest.mark.integration
    def test_successful_json_response(self, integration_driver, sample_json_schema):
        """Test successful conversion of prompt to JSON."""
        content_prompt = "Extract user profile: Juan is 28 years old from Miami"
        from tests.conftest import DEFAULT_MODEL
        result = ask_for_json(integration_driver, content_prompt, sample_json_schema, model_name=DEFAULT_MODEL)

        # Validate response structure
        assert "json_string" in result
        assert "json_object" in result
        assert "usage" in result

        # Validate JSON structure
        assert isinstance(result["json_string"], str)
        json_obj = result["json_object"]
        assert isinstance(json_obj, dict)

        # Should parse successfully without error
        assert json.loads(result["json_string"])

        # Validate usage metadata
        assert_valid_usage_metadata(result["usage"])

    @pytest.mark.integration
    def test_json_schema_inclusion_in_prompt(self, integration_driver, sample_json_schema):
        """Test that JSON schema is properly included in the generated prompt."""
        content_prompt = "Extract user info: Juan is 28 and lives in Miami"
        from tests.conftest import DEFAULT_MODEL
        result = ask_for_json(integration_driver, content_prompt, sample_json_schema, model_name=DEFAULT_MODEL)
        assert "json_string" in result
        parsed = json.loads(result["json_string"])
        assert isinstance(parsed, dict)

    @pytest.mark.integration
    def test_ai_cleanup_enabled(self, integration_driver, sample_json_schema):
        """Test AI cleanup when JSON parsing fails initially."""
        content_prompt = "Extract user info"
        from tests.conftest import DEFAULT_MODEL
        result = ask_for_json(integration_driver, content_prompt, sample_json_schema, ai_cleanup=True, model_name=DEFAULT_MODEL)
        assert "json_object" in result
        assert isinstance(result["json_object"], dict)

    def test_ai_cleanup_disabled_raises_error(self, sample_json_schema):
        """Test that invalid JSON raises exception when AI cleanup is disabled."""
        mock_driver = Mock()
        mock_driver.generate.return_value = {
            "text": '{"name": "Test", "incomplete": true'  # Missing closing brace keeps JSON invalid
        }
        
        content_prompt = "Extract user info"
        with pytest.raises(json.JSONDecodeError):
            ask_for_json(mock_driver, content_prompt, sample_json_schema, ai_cleanup=False)


class TestExtractAndJsonify:
    """Tests for extract_and_jsonify function."""

    @pytest.mark.integration
    def test_successful_extraction_with_template(self, integration_driver, sample_json_schema):
        """Test successful extraction with custom instruction template."""
        text = "Juan is 28 years old and lives in Miami."
        instruction_template = "Please extract the following information:"
        from tests.conftest import DEFAULT_MODEL
        result = extract_and_jsonify(
            text=text,
            json_schema=sample_json_schema,
            model_name=DEFAULT_MODEL,
            instruction_template=instruction_template,
            options={"driver": integration_driver}
        )

        # Validate response structure
        assert_jsonify_response_structure(result)
        assert "json_string" in result
        assert result["json_string"]  # Should contain some response

    @pytest.mark.integration
    def test_default_template_usage(self, integration_driver, sample_json_schema):
        """Test using the default instruction template."""
        text = "John is 25 and from Texas."
        from tests.conftest import DEFAULT_MODEL
        result = extract_and_jsonify(
            text=text,
            json_schema=sample_json_schema,
            model_name=DEFAULT_MODEL,
            options={"driver": integration_driver}
        )
        assert_jsonify_response_structure(result)

    def test_empty_text_raises_error(self, integration_driver, sample_json_schema):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Text input cannot be empty"):
            extract_and_jsonify(
                text="",
                json_schema=sample_json_schema,
                options={"driver": integration_driver}
            )

    def test_whitespace_only_text_raises_error(self, integration_driver, sample_json_schema):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="Text input cannot be empty"):
            extract_and_jsonify(
                text="   ",
                json_schema=sample_json_schema,
                options={"driver": integration_driver}
            )

    @pytest.mark.integration
    def test_with_ai_cleanup(self, integration_driver, sample_json_schema):
        """Test extraction with AI cleanup enabled."""
        text = "Juan has information to extract"
        from tests.conftest import DEFAULT_MODEL
        result = extract_and_jsonify(
            text=text,
            json_schema=sample_json_schema,
            model_name=DEFAULT_MODEL,
            ai_cleanup=True,
            options={"driver": integration_driver}
        )
        assert_jsonify_response_structure(result)

class TestModelNameParameter:
    """Tests specifically for model_name parameter functionality."""

    @pytest.mark.integration
    def test_model_name_in_core_functions(self, integration_driver):
        """Test model_name parameter in all core functions."""
        from tests.conftest import DEFAULT_MODEL
        schema = {"type": "object", "properties": {"test": {"type": "string"}}}
        
        # Test in clean_json_text_with_ai
        json_text = '{"test": "value"'  # Intentionally malformed
        result = clean_json_text_with_ai(integration_driver, json_text, model_name=DEFAULT_MODEL)
        assert isinstance(result, str)
        
        # Test in ask_for_json
        result = ask_for_json(integration_driver, "Extract test info", schema, model_name=DEFAULT_MODEL)
        assert_jsonify_response_structure(result)
        
        # Test in extract_and_jsonify
        result = extract_and_jsonify(
            text="Test data",
            json_schema=schema,
            model_name=DEFAULT_MODEL,
            options={"driver": integration_driver}
        )
        assert_jsonify_response_structure(result)

    @pytest.mark.integration
    def test_model_name_inheritance(self, integration_driver):
        """Test that model_name is properly inherited through nested function calls."""
        from tests.conftest import DEFAULT_MODEL
        schema = {"type": "object", "properties": {"test": {"type": "string"}}}
        
        # Use extract_and_jsonify which internally calls ask_for_json
        result = extract_and_jsonify(
            text="Test data",
            json_schema=schema,
            model_name=DEFAULT_MODEL,
            ai_cleanup=True,  # This will trigger clean_json_text_with_ai
            options={"driver": integration_driver}
        )
        assert_jsonify_response_structure(result)

    @pytest.mark.integration
    def test_model_name_override_precedence(self, integration_driver):
        """Test that explicitly passed model_name takes precedence."""
        from tests.conftest import DEFAULT_MODEL
        override_model = "ollama/mistral:latest"  # Example override
        schema = {"type": "object", "properties": {"test": {"type": "string"}}}
        
        try:
            result = ask_for_json(
                integration_driver,
                "Test data",
                schema,
                model_name=override_model
            )
            assert_jsonify_response_structure(result)
        except ValueError as e:
            if "Unsupported provider" in str(e):
                pytest.skip("Override model not supported in test environment")


class TestErrorHandlingAndEdgeCases:
    """Tests for error handling and edge cases."""

    @pytest.mark.integration
    def test_custom_model_name_override(self, integration_driver):
        """Test using a custom model name that overrides the default."""
        schema = {"type": "object", "properties": {"test": {"type": "string"}}}
        custom_model = "ollama/mistral"  # Example model
        result = ask_for_json(integration_driver, "Simple test", schema, model_name=custom_model)
        assert_jsonify_response_structure(result)

    @pytest.mark.integration
    def test_invalid_model_name_format(self, integration_driver):
        """Test handling of invalid model name format."""
        schema = {"type": "object", "properties": {"test": {"type": "string"}}}
        with pytest.raises(ValueError, match="Invalid model string format. Expected format: 'provider/model'"):
            result = extract_and_jsonify(
                text="Test data",
                json_schema=schema,
                model_name="invalid-model-format",  # Invalid format without provider/model separator
                options={"driver": None}  # Don't pass driver to force model name validation
            )

    @pytest.mark.integration
    def test_unsupported_provider_in_model_name(self, integration_driver):
        """Test handling of unsupported provider in model name."""
        schema = {"type": "object", "properties": {"test": {"type": "string"}}}
        with pytest.raises(ValueError, match="Unsupported provider in model name: nonexistent/model"):
            result = extract_and_jsonify(
                text="Test data",
                json_schema=schema,
                model_name="nonexistent/model",
                options={"driver": None}  # Don't pass driver to force provider validation
            )

    @pytest.mark.integration
    def test_model_name_propagation(self, integration_driver):
        """Test that model_name is properly propagated through nested calls."""
        text = "Test data"
        from tests.conftest import DEFAULT_MODEL
        result = extract_and_jsonify(
            text=text,
            json_schema={"type": "object"},
            model_name=DEFAULT_MODEL,
            ai_cleanup=True,
            options={"driver": integration_driver}
        )
        assert_jsonify_response_structure(result)
        # The model should be used both for extraction and cleanup



    @pytest.mark.integration
    def test_very_large_json_schema(self, integration_driver):
        """Test handling of very large/complex JSON schemas."""
        from tests.conftest import DEFAULT_MODEL
        # Create a very large schema
        large_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        # Add many properties
        for i in range(50):
            large_schema["properties"][f"field_{i}"] = {"type": "string"}

        result = ask_for_json(integration_driver, "Extract info", large_schema, model_name=DEFAULT_MODEL)
        assert_jsonify_response_structure(result)
        assert isinstance(result["json_string"], str)

    @pytest.mark.integration
    def test_nested_schema_validation(self, integration_driver):
        """Test schema with deeply nested structures."""
        nested_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "object",
                            "properties": {
                                "details": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        }

        from tests.conftest import DEFAULT_MODEL
        result = ask_for_json(integration_driver, "Extract user profile", nested_schema, model_name=DEFAULT_MODEL)
        assert_jsonify_response_structure(result)

    @pytest.mark.integration
    def test_empty_content_prompt(self, integration_driver, sample_json_schema):
        """Test with empty content prompt."""
        from tests.conftest import DEFAULT_MODEL
        result = ask_for_json(integration_driver, "", sample_json_schema, model_name=DEFAULT_MODEL)
        assert_jsonify_response_structure(result)

    @pytest.mark.integration
    def test_very_long_text_extraction(self, integration_driver, sample_json_schema):
        """Test extraction from very long text."""
        long_text = "Some information about Juan. " * 1000 + "He is 28 years old."
        from tests.conftest import DEFAULT_MODEL
        result = extract_and_jsonify(
            text=long_text,
            json_schema=sample_json_schema,
            model_name=DEFAULT_MODEL,
            options={"driver": integration_driver}
        )
        assert_jsonify_response_structure(result)

    @pytest.mark.integration
    def test_special_characters_in_prompt(self, integration_driver):
        """Test handling of special characters and unicode in prompts."""
        special_prompt = "Extract from: Juan's info → José María & François"
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        from tests.conftest import DEFAULT_MODEL
        result = ask_for_json(integration_driver, special_prompt, schema, model_name=DEFAULT_MODEL)
        assert_jsonify_response_structure(result)



class TestRenderOutput:
    """Tests for render_output function."""

    @pytest.mark.integration
    def test_render_output_text(self, integration_driver):
        """Test requesting raw text output."""
        from prompture import render_output
        from tests.conftest import DEFAULT_MODEL
        
        prompt = "Say 'Hello World' and nothing else."
        result = render_output(
            driver=integration_driver,
            content_prompt=prompt,
            output_format="text",
            model_name=DEFAULT_MODEL
        )
        
        assert "text" in result
        assert "usage" in result
        assert result["output_format"] == "text"
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0
        # Should not contain markdown fences
        assert "```" not in result["text"]

    @pytest.mark.integration
    def test_render_output_html(self, integration_driver):
        """Test requesting HTML output."""
        from prompture import render_output
        from tests.conftest import DEFAULT_MODEL
        
        prompt = "Create a <div> with text 'Hello'."
        result = render_output(
            driver=integration_driver,
            content_prompt=prompt,
            output_format="html",
            model_name=DEFAULT_MODEL
        )
        
        assert "text" in result
        assert result["output_format"] == "html"
        # Should contain HTML tags
        assert "<div" in result["text"] or "<DIV" in result["text"]
        # Should not contain markdown fences (function cleans them)
        assert "```" not in result["text"]

    @pytest.mark.integration
    def test_render_output_markdown(self, integration_driver):
        """Test requesting markdown output."""
        from prompture import render_output
        from tests.conftest import DEFAULT_MODEL
        
        prompt = "Write a list of 3 items."
        result = render_output(
            driver=integration_driver,
            content_prompt=prompt,
            output_format="markdown",
            model_name=DEFAULT_MODEL
        )
        
        assert "text" in result
        assert result["output_format"] == "markdown"
        # Markdown is allowed, so fences or bullets are fine
        assert "-" in result["text"] or "*" in result["text"] or "1." in result["text"]

    def test_invalid_format_raises_error(self):
        """Test that invalid format raises ValueError."""
        from prompture import render_output
        mock_driver = Mock()
        
        with pytest.raises(ValueError, match="Unsupported output_format"):
            render_output(mock_driver, "prompt", output_format="xml")
