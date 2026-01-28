"""
Field definitions module for Prompture.

This module contains the centralized field definitions system that supports
structured data extraction with specific type hints, descriptions, and instructions.

Features:
- Global field registry with auto-loading
- Template engine for dynamic values ({{current_year}}, {{current_date}}, etc.)
- Pydantic integration via field_from_registry()
- Clean registration API with register_field() and add_field_definition()
"""

import collections.abc
import threading
from datetime import date, datetime
from typing import Any, Optional, Union

from pydantic import Field


# Template variable providers
def _get_template_variables() -> dict[str, Any]:
    """Get current template variables for field definitions."""
    now = datetime.now()
    today = date.today()

    return {
        "current_year": now.year,
        "current_date": today.isoformat(),
        "current_datetime": now.isoformat(),
        "current_timestamp": int(now.timestamp()),
        "current_month": now.month,
        "current_day": now.day,
        "current_weekday": now.strftime("%A"),  # e.g. "Monday"
        "current_iso_week": now.isocalendar().week,  # ISO week number
    }


def _apply_templates(text: str, custom_vars: Optional[dict[str, Any]] = None) -> str:
    """Apply template variable substitution to a text string."""
    if not isinstance(text, str):
        return text

    variables = _get_template_variables()
    if custom_vars:
        variables.update(custom_vars)

    # Simple template replacement
    result = text
    for key, value in variables.items():
        placeholder = f"{{{{{key}}}}}"
        result = result.replace(placeholder, str(value))

    return result


# Thread-safe global registry
_registry_lock = threading.Lock()
_global_registry: dict[str, dict[str, Any]] = {}

# Base field definitions dictionary containing all supported fields
BASE_FIELD_DEFINITIONS = {
    # Person/Identity Fields
    "name": {
        "type": str,
        "description": "Full legal name of the person.",
        "instructions": "Extract as-is, no modifications.",
        "default": "",
        "nullable": False,
    },
    "age": {
        "type": int,
        "description": "The age of the person in number of years.",
        "instructions": "Calculate as {{current_year}} - birth_year if needed.",
        "default": 0,
        "nullable": False,
    },
    "birth_year": {
        "type": int,
        "description": "The year the person was born (YYYY).",
        "instructions": "Extract as a 4-digit year number.",
        "default": None,
        "nullable": True,
    },
    # Contact Information
    "email": {
        "type": str,
        "description": "Primary email address.",
        "instructions": "Extract in lowercase, verify basic email format.",
        "default": "",
        "nullable": True,
    },
    "phone": {
        "type": str,
        "description": "Primary phone number.",
        "instructions": "Extract digits only, standardize to E.164 format if possible.",
        "default": "",
        "nullable": True,
    },
    "address": {
        "type": str,
        "description": "Full mailing address.",
        "instructions": "Combine all address components into a single string.",
        "default": "",
        "nullable": True,
    },
    # Professional Information
    "occupation": {
        "type": str,
        "description": "Current job title or profession.",
        "instructions": "Extract primary occupation, standardize common titles.",
        "default": "",
        "nullable": True,
    },
    "company": {
        "type": str,
        "description": "Current employer or company name.",
        "instructions": "Extract organization name, remove legal suffixes.",
        "default": "",
        "nullable": True,
    },
    "experience_years": {
        "type": int,
        "description": "Years of professional experience.",
        "instructions": "Calculate total years of relevant experience.",
        "default": 0,
        "nullable": True,
    },
    # Metadata Fields
    "source": {
        "type": str,
        "description": "Source of the extracted information.",
        "instructions": "Record origin of data (e.g., 'resume', 'linkedin').",
        "default": "unknown",
        "nullable": False,
    },
    "last_updated": {
        "type": str,
        "description": "Last update timestamp (ISO format).",
        "instructions": "Use ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ), default to {{current_datetime}}.",
        "default": "{{current_datetime}}",
        "nullable": False,
    },
    "confidence_score": {
        "type": float,
        "description": "Confidence score of extraction (0.0-1.0).",
        "instructions": "Calculate based on extraction certainty.",
        "default": 0.0,
        "nullable": False,
    },
    # Location Fields
    "city": {
        "type": str,
        "description": "City name",
        "instructions": "Extract city name, standardize capitalization",
        "default": "",
        "nullable": True,
    },
    "state": {
        "type": str,
        "description": "State or province name",
        "instructions": "Extract state/province, use full name or abbreviation",
        "default": "",
        "nullable": True,
    },
    "postal_code": {
        "type": str,
        "description": "Postal or ZIP code",
        "instructions": "Extract postal code, maintain original format",
        "default": "",
        "nullable": True,
    },
    "country": {
        "type": str,
        "description": "Country name",
        "instructions": "Extract country name, use full English name",
        "default": "",
        "nullable": True,
    },
    "coordinates": {
        "type": str,
        "description": "Geographic coordinates (lat, long)",
        "instructions": "Extract as 'latitude,longitude' format if available",
        "default": "",
        "nullable": True,
    },
    # Demographic Fields
    "gender": {
        "type": str,
        "description": "Gender identification",
        "instructions": "Extract gender if explicitly stated, otherwise leave empty",
        "default": "",
        "nullable": True,
    },
    "nationality": {
        "type": str,
        "description": "Nationality or citizenship",
        "instructions": "Extract nationality, use country demonym",
        "default": "",
        "nullable": True,
    },
    "marital_status": {
        "type": str,
        "description": "Marital status",
        "instructions": "Extract marital status (single, married, divorced, etc.)",
        "default": "",
        "nullable": True,
    },
    "language": {
        "type": str,
        "description": "Primary language spoken",
        "instructions": "Extract primary or native language",
        "default": "",
        "nullable": True,
    },
    # Education Fields
    "education_level": {
        "type": str,
        "description": "Highest education level",
        "instructions": "Extract highest degree (High School, Bachelor's, Master's, PhD, etc.)",
        "default": "",
        "nullable": True,
    },
    "graduation_year": {
        "type": int,
        "description": "Year of graduation",
        "instructions": "Extract graduation year as 4-digit number",
        "default": None,
        "nullable": True,
    },
    "gpa": {
        "type": float,
        "description": "Grade point average",
        "instructions": "Extract GPA, convert to 4.0 scale if needed",
        "default": None,
        "nullable": True,
    },
    # Financial Fields
    "salary": {
        "type": float,
        "description": "Annual salary amount",
        "instructions": "Extract salary as numeric value, remove currency symbols",
        "default": None,
        "nullable": True,
    },
    "currency": {
        "type": str,
        "description": "Currency code",
        "instructions": "Extract or infer currency code (USD, EUR, GBP, etc.)",
        "default": "USD",
        "nullable": True,
    },
    "bonus": {
        "type": float,
        "description": "Bonus amount",
        "instructions": "Extract bonus as numeric value",
        "default": None,
        "nullable": True,
    },
    # Social Media Fields
    "sentiment": {
        "type": str,
        "description": "Sentiment classification",
        "instructions": "Classify the sentiment of the content",
        "enum": ["positive", "negative", "neutral"],
        "default": "neutral",
        "nullable": True,
    },
    "hashtags": {
        "type": str,
        "description": "Hashtags from content",
        "instructions": "Extract all hashtags as comma-separated list",
        "default": "",
        "nullable": True,
    },
    "mentions": {
        "type": str,
        "description": "User mentions from content",
        "instructions": "Extract all @mentions as comma-separated list",
        "default": "",
        "nullable": True,
    },
    "topic": {
        "type": str,
        "description": "Main topic or subject",
        "instructions": "Identify primary topic or theme of content",
        "default": "",
        "nullable": True,
    },
    # Enum Fields for Task Management
    "priority": {
        "type": str,
        "description": "Priority level",
        "instructions": "Determine the priority level",
        "enum": ["low", "medium", "high", "urgent"],
        "default": "medium",
        "nullable": True,
    },
    "status": {
        "type": str,
        "description": "Status of the item",
        "instructions": "Identify the current status",
        "enum": ["pending", "in_progress", "completed", "cancelled"],
        "default": "pending",
        "nullable": True,
    },
    "risk_level": {
        "type": str,
        "description": "Risk assessment level",
        "instructions": "Assess the risk level",
        "enum": ["minimal", "low", "moderate", "high", "critical"],
        "default": "low",
        "nullable": True,
    },
    "tone": {
        "type": str,
        "description": "Tone of the text",
        "instructions": "Classify the tone of the text",
        "enum": ["formal", "informal", "optimistic", "pessimistic"],
        "default": "formal",
        "nullable": True,
    },
}


def _initialize_registry() -> None:
    """Initialize the global registry with base field definitions."""
    with _registry_lock:
        if not _global_registry:
            _global_registry.update(BASE_FIELD_DEFINITIONS)


# Initialize registry on import
_initialize_registry()

# Type hints for field definition structure
FieldType = Union[type, str]
FieldDefinition = dict[str, Union[FieldType, str, Any, bool]]
FieldDefinitions = dict[str, FieldDefinition]

# Maintain backward compatibility
FIELD_DEFINITIONS = _global_registry


def get_field_definition(
    field_name: str, apply_templates: bool = True, custom_template_vars: Optional[dict[str, Any]] = None
) -> Optional[FieldDefinition]:
    """
    Retrieve the definition for a specific field from the global registry.

    Args:
        field_name (str): Name of the field to retrieve
        apply_templates (bool): Whether to apply template variable substitution
        custom_template_vars (Optional[Dict[str, Any]]): Custom template variables

    Returns:
        Optional[FieldDefinition]: Field definition dictionary if found, None otherwise
    """
    with _registry_lock:
        field_def = _global_registry.get(field_name)

    if field_def is None:
        return None

    # Make a copy to avoid modifying the original
    result = field_def.copy()

    if apply_templates:
        # Apply templates to string values
        for key, value in result.items():
            if isinstance(value, str):
                result[key] = _apply_templates(value, custom_template_vars)

    return result


def get_required_fields() -> list[str]:
    """
    Get a list of all required (non-nullable) fields.

    Returns:
        List[str]: List of field names that are marked as non-nullable
    """
    with _registry_lock:
        return [
            field_name for field_name, definition in _global_registry.items() if not definition.get("nullable", True)
        ]


def get_field_names() -> list[str]:
    """
    Get a list of all defined field names.

    Returns:
        List[str]: List of all field names in the registry
    """
    with _registry_lock:
        return list(_global_registry.keys())


def register_field(field_name: str, field_definition: FieldDefinition) -> None:
    """
    Register a single field definition in the global registry.

    Args:
        field_name (str): Name of the field
        field_definition (FieldDefinition): Field definition dictionary

    Raises:
        ValueError: If field definition is invalid
    """
    from .tools import validate_field_definition

    if not validate_field_definition(field_definition):
        raise ValueError(f"Invalid field definition for '{field_name}'")

    with _registry_lock:
        _global_registry[field_name] = field_definition.copy()


def add_field_definition(field_name: str, field_definition: FieldDefinition) -> None:
    """
    Add a field definition to the global registry (alias for register_field).

    Args:
        field_name (str): Name of the field
        field_definition (FieldDefinition): Field definition dictionary
    """
    register_field(field_name, field_definition)


def add_field_definitions(field_definitions: dict[str, FieldDefinition]) -> None:
    """
    Add multiple field definitions to the global registry.

    Args:
        field_definitions (Dict[str, FieldDefinition]): Dictionary of field definitions
    """
    for field_name, field_def in field_definitions.items():
        register_field(field_name, field_def)


def field_from_registry(
    field_name: str, apply_templates: bool = True, custom_template_vars: Optional[dict[str, Any]] = None
) -> Field:
    """
    Create a Pydantic Field from a field definition in the global registry.

    Args:
        field_name (str): Name of the field in the registry
        apply_templates (bool): Whether to apply template variable substitution
        custom_template_vars (Optional[Dict[str, Any]]): Custom template variables

    Returns:
        pydantic.Field: Configured Pydantic Field object

    Raises:
        KeyError: If field_name is not found in the registry
    """
    field_def = get_field_definition(field_name, apply_templates, custom_template_vars)

    if field_def is None:
        raise KeyError(f"Field '{field_name}' not found in registry. Available fields: {', '.join(get_field_names())}")

    # Extract Pydantic Field parameters
    default_value = field_def.get("default")
    description = field_def.get("description", f"Extract the {field_name} from the text.")
    instructions = field_def.get("instructions", "")

    # Handle enum fields
    enum_values = field_def.get("enum")
    if enum_values:
        # Enhance description with enum constraint information
        enum_str = "', '".join(str(v) for v in enum_values)
        enhanced_instructions = f"{instructions}. Must be one of: '{enum_str}'"
        enhanced_description = f"{description}. Allowed values: {enum_str}"

        # Create json_schema_extra with enum constraint
        json_schema_extra = {"enum": enum_values, "instructions": enhanced_instructions}

        # Handle nullable/required logic with enum
        if field_def.get("nullable", True) and default_value is not None:
            return Field(default=default_value, description=enhanced_description, json_schema_extra=json_schema_extra)
        elif field_def.get("nullable", True):
            return Field(default=None, description=enhanced_description, json_schema_extra=json_schema_extra)
        else:
            return Field(description=enhanced_description, json_schema_extra=json_schema_extra)

    # Handle non-enum fields (original logic)
    if field_def.get("nullable", True) and default_value is not None:
        # Optional field with default
        return Field(default=default_value, description=description)
    elif field_def.get("nullable", True):
        # Optional field without default (None)
        return Field(default=None, description=description)
    else:
        # Required field
        return Field(description=description)


def validate_enum_value(field_name: str, value: Any) -> bool:
    """
    Validate that a value is in the allowed enum list for a field.

    Args:
        field_name (str): Name of the field in the registry
        value (Any): Value to validate

    Returns:
        bool: True if value is valid for the enum field, False otherwise
    """
    field_def = get_field_definition(field_name, apply_templates=False)

    if field_def is None:
        return False

    enum_values = field_def.get("enum")
    if not enum_values:
        # Not an enum field, so any value is valid
        return True

    # Check if value is in the allowed enum list
    return value in enum_values


def normalize_enum_value(field_name: str, value: Any, case_sensitive: bool = True) -> Any:
    """
    Normalize and validate an enum value for a field.

    Args:
        field_name (str): Name of the field in the registry
        value (Any): Value to normalize
        case_sensitive (bool): Whether to perform case-sensitive matching

    Returns:
        Any: Normalized value if valid, original value otherwise

    Raises:
        ValueError: If value is not in the allowed enum list
    """
    field_def = get_field_definition(field_name, apply_templates=False)

    if field_def is None:
        raise KeyError(f"Field '{field_name}' not found in registry")

    enum_values = field_def.get("enum")
    if not enum_values:
        # Not an enum field, return as-is
        return value

    # Convert value to string for comparison
    str_value = str(value) if value is not None else None

    if str_value is None:
        # Handle nullable fields
        if field_def.get("nullable", True):
            return None
        else:
            raise ValueError(f"Field '{field_name}' does not allow null values")

    # Case-sensitive matching
    if case_sensitive:
        if str_value in enum_values:
            return str_value
        raise ValueError(
            f"Invalid value '{str_value}' for field '{field_name}'. "
            f"Must be one of: {', '.join(repr(v) for v in enum_values)}"
        )

    # Case-insensitive matching
    lower_value = str_value.lower()
    for enum_val in enum_values:
        if str(enum_val).lower() == lower_value:
            return enum_val

    raise ValueError(
        f"Invalid value '{str_value}' for field '{field_name}'. "
        f"Must be one of: {', '.join(repr(v) for v in enum_values)}"
    )


def get_registry_snapshot() -> dict[str, FieldDefinition]:
    """
    Get a snapshot of the current global registry.

    Returns:
        Dict[str, FieldDefinition]: Copy of the current registry
    """
    with _registry_lock:
        return _global_registry.copy()


def clear_registry() -> None:
    """
    Clear all field definitions from the global registry.
    Warning: This will remove all registered fields including base definitions.
    """
    with _registry_lock:
        _global_registry.clear()


def reset_registry() -> None:
    """
    Reset the global registry to contain only the base field definitions.
    """
    with _registry_lock:
        _global_registry.clear()
        _global_registry.update(BASE_FIELD_DEFINITIONS)


# For backward compatibility, keep the old FIELD_DEFINITIONS reference
# but make it point to the global registry
def _get_field_definitions():
    """Backward compatibility getter for FIELD_DEFINITIONS."""
    return get_registry_snapshot()


# Create a property-like access to maintain backward compatibility
class _FieldDefinitionsProxy(dict, collections.abc.MutableMapping):
    """Proxy class to maintain backward compatibility with FIELD_DEFINITIONS."""

    def __getitem__(self, key):
        return get_field_definition(key)

    def __setitem__(self, key, value):
        register_field(key, value)

    def __delitem__(self, key):
        """Remove a field from the registry."""
        with _registry_lock:
            if key in _global_registry:
                del _global_registry[key]
            else:
                raise KeyError(f"Field '{key}' not found in registry")

    def __contains__(self, key):
        return key in get_field_names()

    def __iter__(self):
        return iter(get_field_names())

    def keys(self):
        return get_field_names()

    def values(self):
        with _registry_lock:
            return list(_global_registry.values())

    def items(self):
        with _registry_lock:
            return list(_global_registry.items())

    def __len__(self):
        with _registry_lock:
            return len(_global_registry)

    def get(self, key, default=None):
        field_def = get_field_definition(key)
        return field_def if field_def is not None else default

    def update(self, other):
        if hasattr(other, "items"):
            add_field_definitions(dict(other.items()))
        else:
            add_field_definitions(dict(other))


# Replace FIELD_DEFINITIONS with the proxy for backward compatibility
FIELD_DEFINITIONS = _FieldDefinitionsProxy()
