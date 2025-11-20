Tools Module
============

.. automodule:: prompture.tools
   :members:
   :undoc-members:
   :show-inheritance:

The tools module provides utility functions for data conversion, validation, field schema generation, and debugging support used throughout Prompture's extraction pipeline.

Overview
--------

The tools module contains:

- **Data Conversion**: Robust type conversion with fallback handling
- **Schema Generation**: JSON schema creation from Python types and field definitions
- **Parsing Utilities**: Specialized parsers for dates, numbers, and boolean values
- **Validation**: Field definition validation and type checking
- **Debugging**: Comprehensive logging system with configurable levels
- **Text Processing**: JSON cleanup and text manipulation utilities

Logging and Debugging
---------------------

LogLevel
~~~~~~~~

.. autoclass:: LogLevel
   :members:
   :show-inheritance:

Enumeration defining logging levels for debug output throughout Prompture.

**Available Levels:**

- ``LogLevel.OFF`` (0) - No logging output
- ``LogLevel.ERROR`` (1) - Error messages only
- ``LogLevel.WARN`` (2) - Warnings and errors
- ``LogLevel.INFO`` (3) - Informational messages
- ``LogLevel.DEBUG`` (4) - Detailed debugging information
- ``LogLevel.TRACE`` (5) - Maximum verbosity with full data dumps

log_debug()
~~~~~~~~~~~

.. autofunction:: log_debug

Conditional logging function that outputs debug information based on current and target log levels.

**Example:**

.. code-block:: python

   from prompture.tools import log_debug, LogLevel
   
   # Log at different levels
   log_debug(LogLevel.INFO, LogLevel.DEBUG, "Processing started", prefix="[main]")
   log_debug(LogLevel.DEBUG, LogLevel.DEBUG, {"field": "name", "value": "John"})
   log_debug(LogLevel.ERROR, LogLevel.INFO, "Validation failed")

Data Parsing and Conversion
---------------------------

parse_boolean()
~~~~~~~~~~~~~~

.. autofunction:: parse_boolean

Robustly parse various boolean representations into Python boolean values.

**Supported Formats:**

- **Strings**: "true", "false", "yes", "no", "1", "0", "on", "off"
- **Numbers**: 1 (True), 0 (False), any non-zero number (True)
- **Booleans**: Direct pass-through
- **Case-insensitive**: "TRUE", "True", "tRuE" all work

**Example:**

.. code-block:: python

   from prompture.tools import parse_boolean
   
   assert parse_boolean("yes") == True
   assert parse_boolean("FALSE") == False
   assert parse_boolean(1) == True
   assert parse_boolean("0") == False

as_list()
~~~~~~~~~

.. autofunction:: as_list

Convert various input types to lists with intelligent parsing.

**Features:**

- **String splitting**: Automatic delimiter detection or custom separators
- **Single values**: Wrap non-list values in lists
- **List pass-through**: Return lists unchanged
- **Empty handling**: Proper handling of None and empty strings

**Example:**

.. code-block:: python

   from prompture.tools import as_list
   
   # Automatic delimiter detection
   assert as_list("apple,banana,cherry") == ["apple", "banana", "cherry"]
   assert as_list("red; blue; green") == ["red", "blue", "green"]
   
   # Custom separator
   assert as_list("a|b|c", sep="|") == ["a", "b", "c"]
   
   # Single value wrapping
   assert as_list("single") == ["single"]
   assert as_list(42) == [42]

parse_datetime()
~~~~~~~~~~~~~~~

.. autofunction:: parse_datetime

Parse datetime strings in various formats into Python datetime objects.

**Supported Formats:**

- ISO 8601: "2024-03-15T14:30:00Z"
- Date only: "2024-03-15", "03/15/2024" 
- Relative: "today", "yesterday", "tomorrow"
- Timestamps: Unix timestamps (integers)

**Example:**

.. code-block:: python

   from prompture.tools import parse_datetime
   
   dt1 = parse_datetime("2024-03-15T14:30:00")
   dt2 = parse_datetime("03/15/2024")
   dt3 = parse_datetime("today")
   dt4 = parse_datetime(1710512400)  # Unix timestamp

parse_shorthand_number()
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: parse_shorthand_number

Parse numbers with shorthand suffixes like "1K", "2.5M", "1.2B".

**Supported Suffixes:**

- **K/k**: Thousands (×1,000)
- **M/m**: Millions (×1,000,000) 
- **B/b**: Billions (×1,000,000,000)
- **T/t**: Trillions (×1,000,000,000,000)

**Example:**

.. code-block:: python

   from prompture.tools import parse_shorthand_number
   
   assert parse_shorthand_number("1.5K") == 1500
   assert parse_shorthand_number("2M") == 2000000
   assert parse_shorthand_number("1.2B") == 1200000000
   assert parse_shorthand_number("500") == 500  # No suffix

Schema Generation and Validation
-------------------------------

create_field_schema()
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: create_field_schema

Generate JSON schema definitions from field names, types, and field definitions.

**Features:**

- **Type mapping**: Python types to JSON schema types
- **Field definitions**: Integration with Prompture's field definition system
- **Validation rules**: Automatic constraint generation
- **Default values**: Schema default value handling

**Example:**

.. code-block:: python

   from prompture.tools import create_field_schema
   
   # Basic type schema
   schema = create_field_schema("age", int)
   # Returns: {"type": "integer", "description": "Value for age"}
   
   # With field definitions
   field_defs = {"name": {"type": str, "description": "Person's name"}}
   schema = create_field_schema("name", str, field_definitions=field_defs)

validate_field_definition()
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: validate_field_definition

Validate that a field definition dictionary contains all required properties and valid values.

**Required Properties:**

- ``type``: Valid Python type (str, int, float, list, dict, bool)
- ``description``: Non-empty string description
- ``instructions``: Extraction instructions for LLMs
- ``default``: Default value matching the specified type
- ``nullable``: Boolean indicating if None values are allowed

**Example:**

.. code-block:: python

   from prompture.tools import validate_field_definition
   
   valid_def = {
       "type": str,
       "description": "Person's full name",
       "instructions": "Extract complete name as written",
       "default": "",
       "nullable": False
   }
   
   assert validate_field_definition(valid_def) == True

Data Conversion and Type Handling
---------------------------------

convert_value()
~~~~~~~~~~~~~~

.. autofunction:: convert_value

Robust value conversion with intelligent type coercion and fallback handling.

**Key Features:**

- **Smart Type Conversion**: Handles strings, numbers, lists, dictionaries
- **Pydantic Integration**: Works with Pydantic models and field types
- **Graceful Fallbacks**: Returns appropriate defaults when conversion fails
- **Recursive Processing**: Deep conversion for nested data structures
- **Special Type Handling**: Decimal, datetime, custom classes

**Conversion Examples:**

.. code-block:: python

   from prompture.tools import convert_value
   from typing import List
   
   # String to integer
   result = convert_value("42", int)  # Returns: 42
   
   # String to list
   result = convert_value("apple,banana,cherry", List[str])
   # Returns: ["apple", "banana", "cherry"]
   
   # Failed conversion with fallback
   result = convert_value("invalid", int, fallback_value=0)  # Returns: 0
   
   # Dictionary with type conversion
   data = {"age": "25", "scores": "80,90,95"}
   types = {"age": int, "scores": List[int]}
   result = convert_value(data, dict, value_types=types)
   # Returns: {"age": 25, "scores": [80, 90, 95]}

extract_fields()
~~~~~~~~~~~~~~~

.. autofunction:: extract_fields

Extract and validate specific fields from data dictionaries with type conversion and default value handling.

**Features:**

- **Field Selection**: Extract only specified fields from larger datasets
- **Type Conversion**: Automatic conversion using [`convert_value()`](#convert_value)
- **Default Handling**: Intelligent defaults from field definitions or type defaults
- **Validation**: Field presence and type validation
- **Error Recovery**: Graceful handling of missing or invalid fields

**Example:**

.. code-block:: python

   from prompture.tools import extract_fields
   
   raw_data = {
       "name": "John Doe",
       "age": "25",
       "scores": "85,92,78",
       "extra": "ignored"
   }
   
   field_types = {
       "name": str,
       "age": int, 
       "scores": List[int]
   }
   
   field_definitions = {
       "name": {"default": "Unknown", "nullable": False},
       "age": {"default": 0, "nullable": False}
   }
   
   result = extract_fields(
       data=raw_data,
       field_names=["name", "age", "scores"],
       field_types=field_types,
       field_definitions=field_definitions
   )
   # Returns: {"name": "John Doe", "age": 25, "scores": [85, 92, 78]}

File and Data Loading
--------------------

load_field_definitions()
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: load_field_definitions

Load field definitions from JSON or YAML files with automatic format detection.

**Supported Formats:**

- **JSON**: Standard JSON field definition files
- **YAML**: YAML format for more readable configuration files
- **Auto-detection**: Based on file extension (.json, .yaml, .yml)

**Example:**

.. code-block:: python

   from prompture.tools import load_field_definitions
   
   # Load from JSON
   fields = load_field_definitions("custom_fields.json")
   
   # Load from YAML  
   fields = load_field_definitions("fields.yaml")
   
   # Use loaded definitions
   from prompture.field_definitions import add_field_definitions
   add_field_definitions(fields)

Default Value Management
-----------------------

get_type_default()
~~~~~~~~~~~~~~~~~

.. autofunction:: get_type_default

Get appropriate default values for Python types.

**Type Defaults:**

- ``str`` → ``""`` (empty string)
- ``int`` → ``0``
- ``float`` → ``0.0``
- ``bool`` → ``False``
- ``list`` → ``[]`` (empty list)
- ``dict`` → ``{}`` (empty dictionary)
- ``None`` → ``None``

**Example:**

.. code-block:: python

   from prompture.tools import get_type_default
   
   assert get_type_default(str) == ""
   assert get_type_default(int) == 0
   assert get_type_default(list) == []

get_field_default()
~~~~~~~~~~~~~~~~~~

.. autofunction:: get_field_default

Get default values for fields using field definitions, Pydantic field info, or type defaults.

**Priority Order:**

1. Field definition default value
2. Pydantic Field default value  
3. Type-based default from [`get_type_default()`](#get_type_default)

**Example:**

.. code-block:: python

   from prompture.tools import get_field_default
   from pydantic import Field
   
   # With field definition
   field_defs = {"name": {"default": "Anonymous", "type": str}}
   default = get_field_default("name", None, field_defs)  # "Anonymous"
   
   # With Pydantic Field
   field_info = Field(default="Unknown")
   default = get_field_default("name", field_info)  # "Unknown"

Text Processing Utilities
------------------------

clean_json_text()
~~~~~~~~~~~~~~~~

.. autofunction:: clean_json_text

Clean and normalize JSON text by removing markdown formatting, extra whitespace, and common text artifacts.

**Cleaning Operations:**

- Remove markdown code block markers (```json, ```)
- Strip extra whitespace and newlines
- Remove common prefixes ("Here's the JSON:", "Result:")
- Normalize quote characters
- Fix common JSON syntax issues

**Example:**

.. code-block:: python

   from prompture.tools import clean_json_text
   
   messy_json = '''
   Here's your JSON:
   ```json
   {
     "name": "John",
     "age": 25
   }
   ```
   '''
   
   clean = clean_json_text(messy_json)
   # Returns: '{"name": "John", "age": 25}'

Internal Utility Functions
-------------------------

The tools module also includes several internal utility functions used by other Prompture modules:

- **_base_schema_for_type()**: Generate base JSON schemas for Python types
- **_strip_desc()**: Remove description fields from schemas
- **_to_decimal()**: Convert values to Decimal objects safely
- **_safe_convert_recursive()**: Recursive type conversion with error handling

Integration with Other Modules
------------------------------

The tools module provides essential utilities used throughout Prompture:

**Core Module Integration:**

- [`convert_value()`](#convert_value) used in [`stepwise_extract_with_model()`](../api/core.rst#stepwise_extract_with_model)
- [`clean_json_text()`](#clean_json_text) used in [`ask_for_json()`](../api/core.rst#ask_for_json)
- [`log_debug()`](#log_debug) used for debugging throughout extraction functions

**Field Definitions Integration:**

- [`validate_field_definition()`](#validate_field_definition) used in field registration
- [`get_field_default()`](#get_field_default) used in [`field_from_registry()`](../api/field_definitions.rst#field_from_registry)

**Driver Integration:**

- Logging utilities used across all driver implementations
- Type conversion used in response processing

Best Practices
--------------

1. **Use Appropriate Log Levels**: Set log levels based on your debugging needs
2. **Handle Conversion Failures**: Always provide sensible fallback values
3. **Validate Field Definitions**: Use [`validate_field_definition()`](#validate_field_definition) before registering custom fields
4. **Leverage Smart Conversion**: Use [`convert_value()`](#convert_value) for robust type handling
5. **Clean External Data**: Use parsing utilities for user input and external data sources
6. **Load Definitions from Files**: Use [`load_field_definitions()`](#load_field_definitions) for maintainable configuration