"""Tools for enhanced type validation and field extraction.

This module provides utilities for:
1. Type determination and JSON schema creation
2. Value conversion with support for human-readable formats
3. Exclusive field extraction
"""
from __future__ import annotations
import re
import decimal
from typing import Any, Dict, List, Optional, Type, Union, get_origin, get_args
from decimal import Decimal
from datetime import datetime
import dateutil.parser
    

from pydantic import BaseModel


def create_field_schema(
    field_name: str,
    field_type: Type,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """Creates a JSON schema for a field based on its type and metadata.
    
    Supports enhanced type validation for:
    - Basic types (int, float, str, bool)
    - Date/time fields (detected by field name)
    - Lists and arrays
    - Optional/Union types
    - Custom types via __schema__ attribute
    
    Args:
        field_name: Name of the field
        field_type: Type annotation of the field
        description: Optional field description
        
    Returns:
        A dictionary containing the JSON schema for the field
        
    Raises:
        ValueError: If field_type is not supported
    """
    from typing import get_origin, get_args
    
    # Initialize schema with description
    schema = {
        "description": description or f"Extract the {field_name} from the text."
    }
    
    # Handle Optional types
    origin = get_origin(field_type)
    if origin is Union:
        args = get_args(field_type)
        # Check if it's Optional (Union with NoneType)
        if type(None) in args:
            # Use the other type
            field_type = next(arg for arg in args if arg is not type(None))
            schema["nullable"] = True
    
    # Handle basic types
    if field_type == int:
        schema["type"] = "integer"
    elif field_type == float or field_type == Decimal:
        schema["type"] = "number"
    elif field_type == bool:
        schema["type"] = "boolean"
    elif field_type == str:
        schema["type"] = "string"
        # Enhanced datetime detection
        if any(term in field_name.lower() for term in ["date", "time", "when", "timestamp"]):
            schema["format"] = "date-time"
    # Handle list types
    elif origin in (list, List):
        schema["type"] = "array"
        if args := get_args(field_type):
            item_schema = create_field_schema(f"{field_name}_item", args[0], None)
            schema["items"] = {k: v for k, v in item_schema.items() if k != "description"}
    # Handle custom types with schema
    elif hasattr(field_type, "__schema__"):
        schema.update(field_type.__schema__)
    else:
        schema["type"] = "string"
        
    return schema


def convert_value(
    value: Any,
    target_type: Type,
    allow_shorthand: bool = True
) -> Any:
    """Converts a value to the target type with support for shorthand notations.
    
    Supports human-readable formats like:
    - Numbers: "1k" (1,000), "2.5m" (2,500,000)
    - Booleans: "yes"/"no", "true"/"false", "1"/"0"
    - Dates: Various common formats
    
    Args:
        value: The value to convert
        target_type: The type to convert to
        allow_shorthand: Whether to support shorthand notations
        
    Returns:
        The converted value
        
    Raises:
        ValueError: If conversion fails
    """
    # Handle None
    if value is None:
        return None
        
    # Handle strings that need conversion
    if isinstance(value, str):
        value = value.strip()
        
        # Try shorthand number notation for numeric types
        if allow_shorthand and target_type in (int, float, Decimal):
            try:
                parsed = parse_shorthand_number(value)
                if target_type == Decimal:
                    return Decimal(str(parsed))
                return target_type(parsed)
            except ValueError:
                pass
        
        # Handle boolean values
        if target_type == bool:
            value = value.lower()
            if value in ('true', 'yes', '1', 'on'):
                return True
            if value in ('false', 'no', '0', 'off'):
                return False
            raise ValueError(f"Cannot convert '{value}' to boolean")
            
        # Handle date/time values
        if target_type == datetime:
            try:
                return dateutil.parser.parse(value)
            except (ValueError, TypeError):
                raise ValueError(f"Cannot parse '{value}' as datetime")
    
    # Handle lists/arrays
    origin = get_origin(target_type)
    if origin in (list, List):
        if not isinstance(value, (list, tuple)):
            value = [value]
        item_type = get_args(target_type)[0]
        return [convert_value(item, item_type, allow_shorthand) for item in value]
    
    # Direct type conversion for basic types
    try:
        if target_type in (int, float, Decimal, str):
            return target_type(value)
    except (ValueError, TypeError, decimal.InvalidOperation) as e:
        raise ValueError(f"Cannot convert '{value}' to {target_type.__name__}: {str(e)}")
        
    return value


def extract_fields(
    model_cls: Type[BaseModel],
    data: Dict[str, Any],
    fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Extracts only specified fields from data based on a Pydantic model.
    
    Args:
        model_cls: The Pydantic model class defining the schema
        data: Dictionary containing the data to extract from
        fields: Optional list of field names to extract. If None, extracts all fields.
        
    Returns:
        Dictionary containing only the specified fields
        
    Raises:
        KeyError: If a requested field doesn't exist in the model
        ValueError: If field validation fails
    """
    # Get all valid field names from the model
    valid_fields = set(model_cls.model_fields.keys())
    
    # If no fields specified, use all fields
    if fields is None:
        fields = list(valid_fields)
    
    # Validate requested fields exist in model
    invalid_fields = set(fields) - valid_fields
    if invalid_fields:
        raise KeyError(f"Fields not found in model: {', '.join(invalid_fields)}")
    
    # Extract and validate each field
    result = {}
    for field_name in fields:
        if field_name not in data:
            continue
            
        field_info = model_cls.model_fields[field_name]
        field_value = data[field_name]
        
        try:
            # Convert value using field's type
            converted_value = convert_value(
                field_value,
                field_info.annotation,
                allow_shorthand=True
            )
            result[field_name] = converted_value
        except ValueError as e:
            raise ValueError(f"Validation failed for field '{field_name}': {str(e)}")
    
    return result


def parse_shorthand_number(value: str) -> Union[int, float]:
    """Parses a number with shorthand notation (e.g., '1k', '2.5m').
    
    Supported suffixes:
    - k/K: thousands (1k = 1,000)
    - m/M: millions (1m = 1,000,000)
    - b/B: billions (1b = 1,000,000,000)
    
    Args:
        value: String containing a number with optional shorthand suffix
        
    Returns:
        The parsed number as int or float
        
    Raises:
        ValueError: If the string cannot be parsed
    """
    if not isinstance(value, str):
        raise ValueError(f"Expected string, got {type(value)}")
        
    value = value.strip().lower()
    if not value:
        raise ValueError("Empty string")
    
    # Extract number and suffix
    match = re.match(r'^(-?\d*\.?\d+)([kmb])?$', value)
    if not match:
        raise ValueError(f"Invalid number format: {value}")
        
    number, suffix = match.groups()
    number = float(number)
    
    # Apply multiplier based on suffix
    multipliers = {
        'k': 1_000,
        'm': 1_000_000,
        'b': 1_000_000_000
    }
    
    if suffix:
        number *= multipliers[suffix]
    
    # Return int if number is whole, float otherwise
    return int(number) if number.is_integer() else number

def clean_json_text(text: str) -> str:
    """Attempts to extract a valid JSON object string from text.

    Handles multiple possible formatting issues:
    - Removes <think>...</think> blocks.
    - Strips markdown code fences (```json ... ```).
    - Falls back to first {...} block found.

    Args:
        text: Raw string that may contain JSON plus extra formatting.

    Returns:
        A string that best resembles valid JSON content.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    if text.startswith("```"):
        start_fence = text.find("```")
        if start_fence != -1:
            start_content = text.find("\n", start_fence)
            if start_content != -1:
                end_fence = text.find("```", start_content)
                if end_fence != -1:
                    return text[start_content + 1:end_fence].strip()
                else:
                    return text[start_content + 1 :].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text

def log_debug(level: int, current_level: int, msg: str):
    if current_level >= level:
        print(msg)