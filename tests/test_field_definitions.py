"""
Tests for the enhanced field definitions system.

This module tests the field definitions in prompture/field_definitions.py
and the utility functions in prompture/tools.py for loading, merging,
and validating field definitions.
"""

import json
import os
import tempfile

import pytest

from prompture.field_definitions import (
    FIELD_DEFINITIONS,
    get_field_definition,
    get_field_names,
    get_required_fields,
)
from prompture.tools import get_field_default, get_type_default, load_field_definitions, validate_field_definition


class TestFieldDefinitionsModule:
    """Test the core field definitions module."""

    def test_field_definitions_structure(self):
        """Test that FIELD_DEFINITIONS has the expected structure."""
        assert isinstance(FIELD_DEFINITIONS, dict)
        assert len(FIELD_DEFINITIONS) > 0

        # Test a few key fields exist
        expected_fields = ["name", "age", "email", "phone", "occupation"]
        for field in expected_fields:
            assert field in FIELD_DEFINITIONS

    def test_field_definition_structure(self):
        """Test that each field definition has the required structure."""
        for field_name, definition in FIELD_DEFINITIONS.items():
            assert isinstance(definition, dict), f"Field {field_name} definition must be dict"

            # Required keys
            required_keys = ["type", "description", "instructions", "default", "nullable"]
            for key in required_keys:
                assert key in definition, f"Field {field_name} missing key: {key}"

            # Type validation
            assert definition["type"] in [str, int, float, bool], f"Field {field_name} has invalid type"
            assert isinstance(definition["description"], str), f"Field {field_name} description must be string"
            assert isinstance(definition["instructions"], str), f"Field {field_name} instructions must be string"
            assert isinstance(definition["nullable"], bool), f"Field {field_name} nullable must be bool"

    def test_get_field_definition(self):
        """Test get_field_definition function."""
        # Test existing field
        name_def = get_field_definition("name")
        assert name_def is not None
        assert name_def["type"] is str
        assert not name_def["nullable"]

        # Test non-existent field
        assert get_field_definition("nonexistent") is None

    def test_get_required_fields(self):
        """Test get_required_fields function."""
        required = get_required_fields()
        assert isinstance(required, list)
        assert "name" in required  # name is non-nullable
        assert "age" in required  # age is non-nullable

        # Check that nullable fields are not in required
        nullable_fields = [name for name, def_ in FIELD_DEFINITIONS.items() if def_.get("nullable", True)]
        for field in nullable_fields:
            assert field not in required

    def test_get_field_names(self):
        """Test get_field_names function."""
        names = get_field_names()
        assert isinstance(names, list)
        assert len(names) == len(FIELD_DEFINITIONS)
        assert set(names) == set(FIELD_DEFINITIONS.keys())


class TestFieldDefinitionUtilities:
    """Test utility functions for field definitions."""

    def test_load_field_definitions_yaml(self):
        """Test loading field definitions from YAML file."""
        yaml_content = """
name:
  type: str
  description: "Person's name"
  instructions: "Extract full name"
  default: ""
  nullable: false

age:
  type: int
  description: "Person's age"
  instructions: "Extract age in years"
  default: 0
  nullable: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name

        try:
            definitions = load_field_definitions(temp_file)
            assert isinstance(definitions, dict)
            assert "name" in definitions
            assert "age" in definitions
            assert definitions["name"]["type"] == "str"
            assert definitions["age"]["default"] == 0
        finally:
            try:
                os.unlink(temp_file)
            except (PermissionError, FileNotFoundError):
                pass  # Ignore cleanup errors on Windows

    def test_load_field_definitions_json(self):
        """Test loading field definitions from JSON file."""
        json_content = {
            "name": {
                "type": "str",
                "description": "Person's name",
                "instructions": "Extract full name",
                "default": "",
                "nullable": False,
            },
            "email": {
                "type": "str",
                "description": "Email address",
                "instructions": "Extract email",
                "default": "",
                "nullable": True,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_content, f)
            temp_file = f.name

        try:
            definitions = load_field_definitions(temp_file)
            assert isinstance(definitions, dict)
            assert "name" in definitions
            assert "email" in definitions
            assert definitions["name"]["type"] == "str"
            assert definitions["email"]["nullable"]
        finally:
            try:
                os.unlink(temp_file)
            except (PermissionError, FileNotFoundError):
                pass  # Ignore cleanup errors on Windows

    def test_load_field_definitions_file_not_found(self):
        """Test loading field definitions from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_field_definitions("nonexistent.yaml")

    def test_load_field_definitions_invalid_format(self):
        """Test loading field definitions from invalid file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("invalid content")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_field_definitions(temp_file)
        finally:
            try:
                os.unlink(temp_file)
            except (PermissionError, FileNotFoundError):
                pass  # Ignore cleanup errors on Windows

    def test_validate_field_definition_valid(self):
        """Test validation of valid field definitions."""
        valid_def = {
            "type": "str",
            "description": "Test field",
            "instructions": "Extract test data",
            "default": "",
            "nullable": True,
        }

        assert validate_field_definition(valid_def)

    def test_validate_field_definition_invalid(self):
        """Test validation of invalid field definitions."""
        # Missing required keys
        invalid_def1 = {"type": "str"}
        assert not validate_field_definition(invalid_def1)

        # Invalid type - the validation function accepts both type objects and strings
        # so this test needs to be updated to reflect that string types are valid
        invalid_def2 = {
            "type": 123,  # Neither type object nor string
            "description": "Test",
            "instructions": "Test",
            "default": "",
            "nullable": True,
        }
        assert not validate_field_definition(invalid_def2)

        # Wrong value types
        invalid_def3 = {
            "type": "str",
            "description": 123,  # Should be string
            "instructions": "Test",
            "default": "",
            "nullable": True,
        }
        assert not validate_field_definition(invalid_def3)

    def test_get_type_default(self):
        """Test getting default values for different types."""
        assert get_type_default(str) == ""
        assert get_type_default(int) == 0
        assert get_type_default(float) == 0.0
        assert not get_type_default(bool)
        assert get_type_default(list) == []
        assert get_type_default(dict) == {}

        # Test with type strings - the function doesn't handle string types
        # Only handles actual type objects, so these should return None
        assert get_type_default("str") is None
        assert get_type_default("int") is None
        assert get_type_default("float") is None
        assert get_type_default("bool") is None

    def test_get_field_default_priority_system(self):
        """Test the priority system for field defaults."""

        # Create a mock field info object that mimics Pydantic FieldInfo
        class MockFieldInfo:
            def __init__(self, default=..., annotation=str):
                self.default = default
                self.annotation = annotation

        field_definitions = {"test_field": {"type": "str", "default": "field_def_default", "nullable": True}}

        # Test priority: field_definitions > model default > type default

        # 1. Field definitions should win
        field_info = MockFieldInfo(default="model_default")
        result = get_field_default("test_field", field_info, field_definitions)
        assert result == "field_def_default"

        # 2. Model default should be used if no field definition
        field_info = MockFieldInfo(default="model_default")
        result = get_field_default("unknown_field", field_info, field_definitions)
        assert result == "model_default"

        # 3. Type default should be used if no other defaults
        field_info = MockFieldInfo(default=..., annotation=int)  # Ellipsis means no default
        result = get_field_default("unknown_field", field_info, None)
        assert result == 0


class TestFieldDefinitionsExamples:
    """Test with example field definition files."""

    def test_load_example_yaml(self):
        """Test loading the example YAML field definitions."""
        try:
            definitions = load_field_definitions("examples/field_definitions.yaml")

            # Should load successfully
            assert isinstance(definitions, dict)
            assert len(definitions) > 0

            # Check some expected fields
            expected_fields = ["name", "age", "email", "occupation"]
            for field in expected_fields:
                assert field in definitions

            # Validate structure
            for field_name, definition in definitions.items():
                assert validate_field_definition(definition), f"Invalid definition for {field_name}"

        except FileNotFoundError:
            pytest.skip("Example YAML file not found")

    def test_load_example_json(self):
        """Test loading the example JSON field definitions."""
        try:
            definitions = load_field_definitions("examples/field_definitions.json")

            # Should load successfully
            assert isinstance(definitions, dict)
            assert len(definitions) > 0

            # Validate structure
            for field_name, definition in definitions.items():
                assert validate_field_definition(definition), f"Invalid definition for {field_name}"

        except FileNotFoundError:
            pytest.skip("Example JSON file not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
