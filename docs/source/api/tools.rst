Tools Module
============

.. automodule:: prompture.tools
   :members:
   :undoc-members:
   :show-inheritance:

The tools module provides utility functions for data conversion, validation, field schema generation, and text processing used throughout Prompture's extraction pipeline.

Overview
--------

The tools module contains:

- **Data Conversion**: Robust type conversion with fallback handling
- **Schema Generation**: JSON schema creation from Python types and field definitions
- **Parsing Utilities**: Specialized parsers for dates, numbers, and boolean values
- **Validation**: Field definition validation and type checking
- **Text Processing**: JSON cleanup and text manipulation utilities

.. note::

   The custom ``LogLevel`` enum and ``log_debug()`` function have been removed.
   Prompture now uses Python's standard ``logging`` module.  Use
   :func:`prompture.configure_logging` to enable library-level logging output.

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

- **K/k**: Thousands (x1,000)
- **M/m**: Millions (x1,000,000)
- **B/b**: Billions (x1,000,000,000)
- **T/t**: Trillions (x1,000,000,000,000)

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

validate_field_definition()
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: validate_field_definition

Validate that a field definition dictionary contains all required properties and valid values.

Data Conversion and Type Handling
---------------------------------

convert_value()
~~~~~~~~~~~~~~

.. autofunction:: convert_value

Robust value conversion with intelligent type coercion and fallback handling.

extract_fields()
~~~~~~~~~~~~~~~

.. autofunction:: extract_fields

Extract and validate specific fields from data dictionaries with type conversion and default value handling.

File and Data Loading
--------------------

load_field_definitions()
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: load_field_definitions

Load field definitions from JSON or YAML files with automatic format detection.

Default Value Management
-----------------------

get_type_default()
~~~~~~~~~~~~~~~~~

.. autofunction:: get_type_default

Get appropriate default values for Python types.

get_field_default()
~~~~~~~~~~~~~~~~~~

.. autofunction:: get_field_default

Get default values for fields using field definitions, Pydantic field info, or type defaults.

Text Processing Utilities
------------------------

clean_json_text()
~~~~~~~~~~~~~~~~

.. autofunction:: clean_json_text

Clean and normalize JSON text by removing markdown formatting, extra whitespace, and common text artifacts.

Best Practices
--------------

1. **Use configure_logging()**: Call ``configure_logging(logging.DEBUG)`` to see internal debug output
2. **Handle Conversion Failures**: Always provide sensible fallback values
3. **Validate Field Definitions**: Use ``validate_field_definition()`` before registering custom fields
4. **Leverage Smart Conversion**: Use ``convert_value()`` for robust type handling
5. **Clean External Data**: Use parsing utilities for user input and external data sources
6. **Load Definitions from Files**: Use ``load_field_definitions()`` for maintainable configuration
