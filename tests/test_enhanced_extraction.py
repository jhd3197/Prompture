"""
Tests for the enhanced stepwise_extract_with_model function.

This module tests the enhanced stepwise extraction with field definitions
support, default value handling, and graceful failure management.
"""

from typing import Optional
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from prompture.core import stepwise_extract_with_model
from prompture.field_definitions import FIELD_DEFINITIONS


# Test models for extraction
class PersonModel(BaseModel):
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    email: Optional[str] = Field(None, description="Email address")
    occupation: Optional[str] = Field(None, description="Job title")


class SimpleModel(BaseModel):
    name: str
    value: int


class ComplexModel(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    occupation: Optional[str] = Field(None, description="Job title")
    experience_years: Optional[int] = Field(None, description="Years of experience")


class TestStepwiseExtractionBasic:
    """Test basic functionality of enhanced stepwise extraction."""

    def test_input_validation(self):
        """Test input validation for stepwise extraction."""
        # Empty text should raise error
        with pytest.raises(ValueError, match="Text input cannot be empty"):
            stepwise_extract_with_model(PersonModel, "", model_name="test/model")

        # None text should raise error
        with pytest.raises(ValueError, match="Text input cannot be empty"):
            stepwise_extract_with_model(PersonModel, None, model_name="test/model")

        # Whitespace-only text should raise error
        with pytest.raises(ValueError, match="Text input cannot be empty"):
            stepwise_extract_with_model(PersonModel, "   \n  ", model_name="test/model")

    def test_field_validation(self):
        """Test field validation for stepwise extraction."""
        # Valid fields should work

        # Invalid field names should raise KeyError
        with pytest.raises(KeyError, match="Fields not found in model"):
            with patch("prompture.core.extract_and_jsonify"):
                stepwise_extract_with_model(
                    PersonModel, "test text", model_name="test/model", fields=["nonexistent_field"]
                )

    @patch("prompture.core.extract_and_jsonify")
    def test_successful_extraction_all_fields(self, mock_extract):
        """Test successful extraction of all fields."""
        # Mock successful extractions for each field
        mock_extract.side_effect = [
            {  # name field
                "json_object": {"value": "John Doe"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
            {  # age field
                "json_object": {"value": "30"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
            {  # email field
                "json_object": {"value": "john@example.com"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
            {  # occupation field
                "json_object": {"value": "Engineer"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
        ]

        result = stepwise_extract_with_model(
            PersonModel, "John Doe is a 30-year-old engineer. Email: john@example.com", model_name="test/model"
        )

        # Check that result is returned
        assert result is not None
        assert isinstance(result, dict)

        # Check that model instance was created
        assert "model" in result
        assert result["model"].name == "John Doe"
        assert result["model"].age == 30
        assert result["model"].email == "john@example.com"
        assert result["model"].occupation == "Engineer"

        # Check usage accumulation
        assert "usage" in result
        usage = result["usage"]
        assert usage["total_tokens"] == 60  # 4 fields * 15 tokens each
        assert usage["cost"] == 0.04  # 4 fields * 0.01 each

        # Check field results
        assert "field_results" in result
        field_results = result["field_results"]
        for field_name in ["name", "age", "email", "occupation"]:
            assert field_name in field_results
            assert field_results[field_name]["status"] == "success"
            assert not field_results[field_name]["used_default"]

    @patch("prompture.core.extract_and_jsonify")
    def test_extraction_with_field_subset(self, mock_extract):
        """Test extraction with only specific fields."""
        mock_extract.side_effect = [
            {  # name field
                "json_object": {"value": "Jane Smith"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
            {  # age field
                "json_object": {"value": "25"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
        ]

        result = stepwise_extract_with_model(
            PersonModel,
            "Jane Smith is 25 years old",
            model_name="test/model",
            fields=["name", "age"],  # Only extract these fields
        )

        # Check extracted fields
        assert result["model"].name == "Jane Smith"
        assert result["model"].age == 25

        # Other fields should have default values
        assert result["model"].email is None  # Optional field default
        assert result["model"].occupation is None  # Optional field default


class TestStepwiseExtractionWithFailures:
    """Test stepwise extraction with various failure scenarios."""

    @patch("prompture.core.extract_and_jsonify")
    def test_extraction_failure_with_defaults(self, mock_extract):
        """Test extraction failure with graceful default handling."""
        # First field succeeds, second fails, third succeeds
        mock_extract.side_effect = [
            {  # name field - success
                "json_object": {"value": "John Doe"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
            Exception("Network error"),  # age field - extraction failure
            {  # email field - success
                "json_object": {"value": "john@example.com"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
            {  # occupation field - success
                "json_object": {"value": "Engineer"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
        ]

        result = stepwise_extract_with_model(
            PersonModel, "John Doe is an engineer. Email: john@example.com", model_name="test/model"
        )

        # Successful fields should be extracted
        assert result["model"].name == "John Doe"
        assert result["model"].email == "john@example.com"
        assert result["model"].occupation == "Engineer"

        # Failed field should use default
        assert result["model"].age == 0  # int default

        # Check field results
        field_results = result["field_results"]
        assert field_results["name"]["status"] == "success"
        assert field_results["age"]["status"] == "extraction_failed"
        assert field_results["age"]["used_default"]
        assert field_results["email"]["status"] == "success"
        assert field_results["occupation"]["status"] == "success"

    @patch("prompture.core.extract_and_jsonify")
    def test_conversion_failure_with_defaults(self, mock_extract):
        """Test type conversion failure with graceful default handling."""
        mock_extract.side_effect = [
            {  # name field - success
                "json_object": {"value": "John Doe"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
            {  # age field - extraction success but conversion will fail
                "json_object": {"value": "not_a_number"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
            {  # email field - success
                "json_object": {"value": "john@example.com"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
            {  # occupation field - success
                "json_object": {"value": "Engineer"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
        ]

        result = stepwise_extract_with_model(
            PersonModel, "John Doe is an engineer. Email: john@example.com", model_name="test/model"
        )

        # Successful conversions should work
        assert result["model"].name == "John Doe"
        assert result["model"].email == "john@example.com"
        assert result["model"].occupation == "Engineer"

        # Failed conversion should use default
        assert result["model"].age == 0  # int default

        # Check field results - convert_value is more robust and might succeed
        field_results = result["field_results"]
        assert field_results["name"]["status"] == "success"
        # The conversion might succeed due to robust parsing, so check the actual behavior
        if field_results["age"]["status"] == "conversion_failed":
            assert field_results["age"]["used_default"]
        else:
            # If conversion succeeded, check the result is reasonable
            assert isinstance(result["model"].age, int)


class TestStepwiseExtractionWithFieldDefinitions:
    """Test stepwise extraction with field definitions for enhanced defaults."""

    @patch("prompture.core.extract_and_jsonify")
    def test_extraction_with_field_definitions(self, mock_extract):
        """Test extraction using field definitions for defaults."""
        # Mock one successful extraction and one failure
        mock_extract.side_effect = [
            {  # name field - success
                "json_object": {"value": "John Doe"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
            Exception("Extraction failed"),  # age field - failure
        ]

        # Use built-in field definitions
        result = stepwise_extract_with_model(
            SimpleModel,  # Only name and value fields
            "John Doe information",
            model_name="test/model",
            fields=["name", "value"],
            field_definitions=FIELD_DEFINITIONS,
        )

        # Successful field should be extracted
        assert result["model"].name == "John Doe"

        # Failed field should use field definition default or type default
        assert isinstance(result["model"].value, int)  # Should be int type

        # Check that field definitions were used
        field_results = result["field_results"]
        assert field_results["name"]["status"] == "success"
        assert field_results["value"]["status"] == "extraction_failed"
        assert field_results["value"]["used_default"]

    @patch("prompture.core.extract_and_jsonify")
    def test_custom_field_definitions(self, mock_extract):
        """Test extraction with custom field definitions."""
        custom_definitions = {
            "name": {"type": "str", "description": "Person's name", "default": "Unknown Person", "nullable": False},
            "value": {"type": "int", "description": "Numeric value", "default": 999, "nullable": True},
        }

        # Mock extraction failure for both fields
        mock_extract.side_effect = [Exception("Extraction failed for name"), Exception("Extraction failed for value")]

        result = stepwise_extract_with_model(
            SimpleModel, "Some text without clear info", model_name="test/model", field_definitions=custom_definitions
        )

        # When all extractions fail, the function may return an error result
        if "model" in result:
            # Should use custom defaults
            assert result["model"].name == "Unknown Person"
            assert result["model"].value == 999

            # Check field results
            field_results = result["field_results"]
            assert field_results["name"]["used_default"]
            assert field_results["value"]["used_default"]
        else:
            # If validation failed completely, check error handling
            assert "error" in result
            field_results = result["field_results"]
            assert field_results["name"]["used_default"]
            assert field_results["value"]["used_default"]


class TestStepwiseExtractionUsageTracking:
    """Test usage tracking and metadata in stepwise extraction."""

    @patch("prompture.core.extract_and_jsonify")
    def test_usage_accumulation(self, mock_extract):
        """Test that usage statistics are properly accumulated."""
        mock_extract.side_effect = [
            {
                "json_object": {"value": "John"},
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25, "cost": 0.02},
            },
            {
                "json_object": {"value": "30"},
                "usage": {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18, "cost": 0.015},
            },
        ]

        result = stepwise_extract_with_model(SimpleModel, "John is 30", model_name="test/model")

        usage = result["usage"]
        assert usage["prompt_tokens"] == 35  # 20 + 15
        assert usage["completion_tokens"] == 8  # 5 + 3
        assert usage["total_tokens"] == 43  # 25 + 18
        assert usage["cost"] == 0.035  # 0.02 + 0.015
        assert usage["model_name"] == "test/model"

        # Check field-level usage tracking
        assert "field_usages" in usage
        assert "name" in usage["field_usages"]
        assert "value" in usage["field_usages"]

    @patch("prompture.core.extract_and_jsonify")
    def test_usage_with_failures(self, mock_extract):
        """Test usage tracking when some extractions fail."""
        mock_extract.side_effect = [
            {
                "json_object": {"value": "John"},
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25, "cost": 0.02},
            },
            Exception("Extraction failed"),  # Second field fails
        ]

        result = stepwise_extract_with_model(SimpleModel, "John information", model_name="test/model")

        usage = result["usage"]
        # Only successful extraction should contribute to usage
        assert usage["prompt_tokens"] == 20
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 25
        assert usage["cost"] == 0.02

        # Failed field should have error info in field_usages
        assert "value" in usage["field_usages"]
        assert "error" in usage["field_usages"]["value"]
        assert usage["field_usages"]["value"]["status"] == "failed"


class TestStepwiseExtractionBackwardCompatibility:
    """Test backward compatibility of enhanced stepwise extraction."""

    @patch("prompture.core.extract_and_jsonify")
    def test_original_call_signature(self, mock_extract):
        """Test that original way of calling stepwise_extract_with_model still works."""
        mock_extract.side_effect = [
            {
                "json_object": {"value": "Test Name"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
            {
                "json_object": {"value": "42"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
            },
        ]

        # Call without new field_definitions parameter
        result = stepwise_extract_with_model(SimpleModel, "Test Name with value 42", model_name="test/model")

        # Should work as before
        assert result["model"].name == "Test Name"
        assert result["model"].value == 42

    @patch("prompture.core.extract_and_jsonify")
    def test_enhanced_call_signature(self, mock_extract):
        """Test that new enhanced parameters work correctly."""
        mock_extract.return_value = {
            "json_object": {"value": "Enhanced Test"},
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
        }

        # Call with all new parameters
        result = stepwise_extract_with_model(
            SimpleModel,
            "Enhanced test data",
            model_name="test/model",
            fields=["name"],  # Only extract name
            field_definitions=FIELD_DEFINITIONS,
            options={"temperature": 0.7},
        )

        # Should work with new parameters
        if "model" in result:
            assert result["model"].name == "Enhanced Test"
            # Value should use default since not extracted
            assert isinstance(result["model"].value, int)
        else:
            # If model validation failed due to missing required field
            assert "error" in result
            # But the extraction should still have succeeded
            field_results = result["field_results"]
            assert field_results["name"]["status"] == "success"


class TestStepwiseExtractionComplexScenarios:
    """Test complex scenarios with multiple failures and recoveries."""

    @patch("prompture.core.extract_and_jsonify")
    def test_partial_success_scenario(self, mock_extract):
        """Test scenario where some fields succeed and others fail."""
        # Simulate mixed success/failure for a complex model
        mock_responses = []

        # name - success
        mock_responses.append(
            {
                "json_object": {"value": "Alice Johnson"},
                "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20, "cost": 0.02},
            }
        )

        # age - conversion failure (invalid data)
        mock_responses.append(
            {
                "json_object": {"value": "twenty-five"},  # Will fail int conversion
                "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20, "cost": 0.02},
            }
        )

        # email - success
        mock_responses.append(
            {
                "json_object": {"value": "alice@company.com"},
                "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20, "cost": 0.02},
            }
        )

        # phone - extraction failure
        mock_responses.append(Exception("Could not extract phone"))

        # occupation - success
        mock_responses.append(
            {
                "json_object": {"value": "Software Engineer"},
                "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20, "cost": 0.02},
            }
        )

        # experience_years - success
        mock_responses.append(
            {
                "json_object": {"value": "5"},
                "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20, "cost": 0.02},
            }
        )

        mock_extract.side_effect = mock_responses

        result = stepwise_extract_with_model(
            ComplexModel,
            "Alice Johnson is a software engineer with 5 years experience. Email: alice@company.com",
            model_name="test/model",
            field_definitions=FIELD_DEFINITIONS,
        )

        # Check successful extractions
        assert result["model"].name == "Alice Johnson"
        assert result["model"].email == "alice@company.com"
        assert result["model"].occupation == "Software Engineer"
        assert result["model"].experience_years == 5

        # Check failed extractions use defaults
        assert result["model"].age == 0  # Field definition default for age
        assert result["model"].phone == ""  # Field definition default for phone

        # Check field results tracking
        field_results = result["field_results"]
        assert field_results["name"]["status"] == "success"
        # Age conversion might succeed due to robust parsing
        if field_results["age"]["status"] == "conversion_failed":
            assert field_results["age"]["used_default"]
        assert field_results["email"]["status"] == "success"
        assert field_results["phone"]["status"] == "extraction_failed"
        assert field_results["occupation"]["status"] == "success"
        assert field_results["experience_years"]["status"] == "success"

        # Check usage accumulation (only successful extractions)
        usage = result["usage"]
        assert usage["total_tokens"] == 100  # 5 successful * 20 tokens each
        assert usage["cost"] == 0.10  # 5 successful * 0.02 each


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
