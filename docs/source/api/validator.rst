Validator Module
================

.. automodule:: prompture.validator
   :members:
   :undoc-members:
   :show-inheritance:

The validator module provides JSON schema validation functionality for ensuring that extracted data conforms to expected structures and constraints.

Overview
--------

The validator module enables:

- **Schema Validation**: Validate JSON data against JSON Schema specifications
- **Type Checking**: Ensure extracted values match expected data types
- **Constraint Validation**: Verify that data meets defined constraints (ranges, formats, etc.)
- **Error Reporting**: Detailed validation error messages for debugging

Main Functions
--------------

validate_against_schema()
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: validate_against_schema

Validate a JSON string against a JSON Schema specification and return detailed validation results.

**Features:**

- **JSON Schema Compliance**: Full support for JSON Schema Draft 7 specification
- **Comprehensive Validation**: Type checking, format validation, constraint verification
- **Detailed Error Reporting**: Specific error messages with location information
- **Graceful Error Handling**: Returns structured error information instead of raising exceptions

**Parameters:**

- ``instance_json`` (str): JSON string to validate
- ``schema`` (Dict[str, Any]): JSON Schema specification as a dictionary

**Returns:**

Dictionary containing validation results:

.. code-block:: python

   {
       "valid": True,           # Boolean indicating if validation passed
       "errors": [],            # List of validation errors (empty if valid)
       "data": {...},          # Parsed JSON data (if valid)
       "error_count": 0,        # Number of validation errors
       "error_summary": "..."   # Human-readable error summary
   }

**Example Usage:**

.. code-block:: python

   from prompture.validator import validate_against_schema
   
   # Define JSON Schema
   person_schema = {
       "type": "object",
       "properties": {
           "name": {
               "type": "string",
               "minLength": 1,
               "maxLength": 100
           },
           "age": {
               "type": "integer", 
               "minimum": 0,
               "maximum": 150
           },
           "email": {
               "type": "string",
               "format": "email"
           }
       },
       "required": ["name", "age"],
       "additionalProperties": False
   }
   
   # Valid JSON
   valid_json = '{"name": "John Doe", "age": 25, "email": "john@example.com"}'
   result = validate_against_schema(valid_json, person_schema)
   
   if result["valid"]:
       print("Validation passed!")
       print(f"Data: {result['data']}")
   else:
       print(f"Validation failed: {result['error_summary']}")
       for error in result["errors"]:
           print(f"- {error}")

**Validation Error Examples:**

.. code-block:: python

   # Invalid JSON - missing required field
   invalid_json = '{"name": "John Doe"}'  # Missing required 'age' field
   result = validate_against_schema(invalid_json, person_schema)
   # result["valid"] == False
   # result["errors"] contains details about missing 'age' field
   
   # Invalid JSON - type mismatch
   invalid_json = '{"name": "John Doe", "age": "twenty-five"}'  # Age as string
   result = validate_against_schema(invalid_json, person_schema)
   # result["valid"] == False
   # result["errors"] contains type mismatch error for 'age' field
   
   # Invalid JSON - constraint violation
   invalid_json = '{"name": "", "age": -5}'  # Empty name, negative age
   result = validate_against_schema(invalid_json, person_schema)
   # result["valid"] == False
   # result["errors"] contains constraint violation errors

Supported JSON Schema Features
------------------------------

The validator supports comprehensive JSON Schema Draft 7 features:

Type Validation
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   
   * - Schema Type
     - Python Type
     - Description
   * - ``string``
     - str
     - Text values with optional format constraints
   * - ``integer``  
     - int
     - Whole numbers with optional range constraints
   * - ``number``
     - int, float
     - Numeric values including decimals
   * - ``boolean``
     - bool
     - True/False values
   * - ``array``
     - list
     - Ordered collections with item type constraints
   * - ``object``
     - dict
     - Key-value structures with property constraints
   * - ``null``
     - None
     - Null/None values

String Constraints and Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Length Constraints:**

- ``minLength``: Minimum string length
- ``maxLength``: Maximum string length

**Pattern Matching:**

- ``pattern``: Regular expression pattern matching

**Built-in Formats:**

- ``email``: Email address validation
- ``date``: Date format (YYYY-MM-DD)
- ``date-time``: ISO 8601 datetime format
- ``uri``: URI/URL format validation
- ``uuid``: UUID format validation

**Example:**

.. code-block:: python

   string_schema = {
       "type": "string",
       "minLength": 3,
       "maxLength": 50,
       "pattern": "^[A-Za-z ]+$",  # Letters and spaces only
       "format": "email"           # Email format validation
   }

Numeric Constraints
~~~~~~~~~~~~~~~~~~

**Range Constraints:**

- ``minimum``: Minimum value (inclusive)
- ``maximum``: Maximum value (inclusive)
- ``exclusiveMinimum``: Minimum value (exclusive)
- ``exclusiveMaximum``: Maximum value (exclusive)

**Multiple Constraints:**

- ``multipleOf``: Value must be multiple of specified number

**Example:**

.. code-block:: python

   numeric_schema = {
       "type": "integer",
       "minimum": 18,
       "maximum": 65,
       "multipleOf": 5  # Age in 5-year increments
   }

Array Constraints
~~~~~~~~~~~~~~~~

**Length Constraints:**

- ``minItems``: Minimum array length
- ``maxItems``: Maximum array length

**Item Constraints:**

- ``items``: Schema for array items
- ``uniqueItems``: Require unique items

**Example:**

.. code-block:: python

   array_schema = {
       "type": "array",
       "minItems": 1,
       "maxItems": 10,
       "uniqueItems": True,
       "items": {
           "type": "string",
           "minLength": 1
       }
   }

Object Constraints
~~~~~~~~~~~~~~~~~

**Property Constraints:**

- ``properties``: Schema for specific properties
- ``required``: List of required property names
- ``additionalProperties``: Allow/disallow extra properties

**Property Count:**

- ``minProperties``: Minimum number of properties
- ``maxProperties``: Maximum number of properties

**Example:**

.. code-block:: python

   object_schema = {
       "type": "object",
       "properties": {
           "name": {"type": "string"},
           "age": {"type": "integer"}
       },
       "required": ["name"],
       "additionalProperties": False,
       "minProperties": 1,
       "maxProperties": 5
   }

Advanced Schema Features
-----------------------

Conditional Validation
~~~~~~~~~~~~~~~~~~~~~

Use ``if``/``then``/``else`` for conditional constraints:

.. code-block:: python

   conditional_schema = {
       "type": "object",
       "properties": {
           "age": {"type": "integer"},
           "driver_license": {"type": "boolean"}
       },
       "if": {
           "properties": {"age": {"minimum": 18}}
       },
       "then": {
           "properties": {"driver_license": {"type": "boolean"}}
       },
       "else": {
           "properties": {"driver_license": {"const": False}}
       }
   }

Composition Keywords
~~~~~~~~~~~~~~~~~~~

Combine schemas with logical operators:

- ``allOf``: Must match all sub-schemas
- ``anyOf``: Must match at least one sub-schema  
- ``oneOf``: Must match exactly one sub-schema
- ``not``: Must not match the sub-schema

**Example:**

.. code-block:: python

   composition_schema = {
       "anyOf": [
           {"type": "string", "format": "email"},
           {"type": "string", "pattern": "^\\+[1-9]\\d{1,14}$"}  # Phone number
       ]
   }

Integration with Prompture
--------------------------

The validator integrates with core Prompture functionality:

**Automatic Validation in Extraction:**

.. code-block:: python

   from prompture.core import extract_and_jsonify
   from prompture.validator import validate_against_schema
   
   # Extract data
   result = extract_and_jsonify(
       text="John Doe is 25 years old",
       json_schema=person_schema,
       model_name="openai/gpt-4"
   )
   
   # Validate extracted data
   validation = validate_against_schema(result["json_string"], person_schema)
   
   if validation["valid"]:
       print("Extraction and validation successful!")
   else:
       print(f"Validation errors: {validation['error_summary']}")

**Integration with Test Suites:**

.. code-block:: python

   from prompture.runner import run_suite_from_spec
   
   # Test specifications automatically use validation
   test_spec = {
       "tests": [{
           "schema": person_schema,  # Automatically validated
           "expected": {...}
       }]
   }

Error Handling and Debugging
---------------------------

The validator provides comprehensive error information:

**Error Structure:**

.. code-block:: python

   {
       "message": "Detailed error description",
       "path": ["property", "name"],  # Location in data structure
       "schema_path": ["properties", "name"],  # Location in schema
       "instance": "invalid_value",  # The invalid value
       "validator": "minLength",     # Which validation rule failed
       "constraint": 1              # The constraint value
   }

**Common Error Types:**

- **Type Errors**: Data type doesn't match schema expectation
- **Constraint Errors**: Value violates defined constraints (min/max, length, etc.)
- **Format Errors**: String doesn't match required format (email, date, etc.)
- **Required Errors**: Missing required properties in objects
- **Additional Property Errors**: Extra properties when ``additionalProperties`` is False

**Debugging Tips:**

1. **Check Error Paths**: Use the ``path`` field to locate problematic data
2. **Examine Constraints**: Review the ``constraint`` field to understand requirements
3. **Validate Incrementally**: Test individual properties before complex schemas
4. **Use Simple Schemas First**: Start with basic validation and add complexity gradually

Best Practices
--------------

1. **Define Comprehensive Schemas**: Include all relevant constraints and formats
2. **Use Appropriate Data Types**: Choose the most specific type for each field
3. **Validate Early**: Validate data as soon as possible after extraction
4. **Handle Validation Errors**: Always check validation results before using data
5. **Provide Clear Error Messages**: Use schema descriptions for better error reporting
6. **Test Edge Cases**: Validate schemas with boundary values and invalid inputs
7. **Version Your Schemas**: Keep track of schema changes for backward compatibility

Dependencies
------------

The validator module requires:

- **jsonschema**: JSON Schema validation library (``pip install jsonschema``)

If the jsonschema library is not available, validation functions will return appropriate error messages indicating the missing dependency.

**Installation:**

.. code-block:: bash

   pip install jsonschema

The library is automatically included when installing Prompture with validation extras:

.. code-block:: bash

   pip install prompture[validation]