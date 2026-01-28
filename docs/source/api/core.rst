Core Module
===========

.. automodule:: prompture.core
   :members:
   :undoc-members:
   :show-inheritance:

The core module provides the main extraction functions for converting unstructured text into structured JSON data using various LLM providers.

Overview
--------

The core module contains the primary extraction functions that handle the interaction between text input, field definitions, and LLM drivers to produce structured data output. These functions support multiple extraction strategies, from simple single-step extraction to advanced multi-step validation processes.

Main Functions
--------------

extract_and_jsonify()
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: extract_and_jsonify

The primary function for extracting structured data from text using field definitions and automatic driver selection.

**Key Features:**

- Automatic driver selection based on model name format (``provider/model``)
- Support for custom JSON schemas
- Built-in error handling and validation
- Optional AI-powered cleanup for malformed JSON responses

**Example:**

.. code-block:: python

   result = extract_and_jsonify(
       text="John Smith is 25 years old and works at OpenAI",
       json_schema={
           "type": "object",
           "properties": {
               "name": {"type": "string"},
               "age": {"type": "integer"},
               "company": {"type": "string"}
           }
       },
       model_name="openai/gpt-4"
   )

extract_with_model()
~~~~~~~~~~~~~~~~~~~

.. autofunction:: extract_with_model

Extract structured data using Pydantic models with integration to the field registry system.

**Key Features:**

- Automatic JSON schema generation from Pydantic models  
- Integration with [`field_from_registry()`](../api/field_definitions.rst#field_from_registry) for enhanced field definitions
- Model validation and type conversion
- Backward compatibility with legacy function signatures

**Example:**

.. code-block:: python

   from pydantic import BaseModel
   from prompture import field_from_registry, extract_with_model
   
   class Person(BaseModel):
       name: str = field_from_registry("name")
       age: int = field_from_registry("age")
       
   result = extract_with_model(
       model_cls=Person,
       text="Alice Johnson, 32 years old", 
       model_name="openai/gpt-4"
   )
   person = result.model  # Access the Pydantic model instance

stepwise_extract_with_model()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: stepwise_extract_with_model

Advanced multi-step extraction process that processes each model field individually with enhanced validation and error handling.

**Key Features:**

- Field-by-field extraction for improved accuracy
- Graceful handling of partial failures with default values
- Detailed usage tracking per field
- Integration with global field registry for enhanced defaults
- Comprehensive logging and debugging support

**Example:**

.. code-block:: python

   result = stepwise_extract_with_model(
       model_cls=Person,
       text="Complex document with multiple data points...",
       model_name="openai/gpt-4",
       instruction_template="Extract the {field_name} from this document:",
   )
   
   person = result["model"]
   usage_info = result["usage"]
   field_results = result["field_results"]

manual_extract_and_jsonify()
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: manual_extract_and_jsonify

Low-level extraction function that uses an explicitly provided driver instance for direct control over the LLM interaction.

**Key Features:**

- Direct driver control (OpenAI, Claude, Ollama, etc.)
- Optional model override per call
- Detailed logging and debugging capabilities
- Full control over driver configuration and options

**Example:**

.. code-block:: python

   from prompture.drivers import OpenAIDriver
   
   driver = OpenAIDriver(api_key="your-key", model="gpt-4")
   result = manual_extract_and_jsonify(
       driver=driver,
       text="Extract from this text...",
       json_schema=schema,
       model_name="gpt-4-turbo",  # Override driver's default model
   )

Utility Functions
-----------------

ask_for_json()
~~~~~~~~~~~~~

.. autofunction:: ask_for_json

Core function that handles the actual LLM communication and JSON parsing with robust error handling.

**Key Features:**

- Schema-enforced LLM prompting
- Automatic JSON parsing and validation
- Usage tracking (tokens, cost, etc.)
- AI-powered cleanup for malformed JSON responses

clean_json_text_with_ai()
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: clean_json_text_with_ai

Uses an LLM to repair malformed JSON strings when standard parsing fails.

**Features:**

- Intelligent JSON repair using AI
- Fallback mechanism for [`ask_for_json()`](#ask_for_json)
- Preserves data integrity while fixing syntax issues

Error Handling
--------------

The core module implements comprehensive error handling:

**ValueError**: Raised for invalid inputs, empty text, or malformed model names
**ConnectionError**: Raised for network issues (automatically skipped in pytest environments)
**json.JSONDecodeError**: Raised when JSON parsing fails and AI cleanup is disabled

All functions support graceful degradation and provide detailed error messages to help with debugging.

Return Value Structure
----------------------

Most extraction functions return a dictionary with the following structure:

.. code-block:: python

   {
       "json_string": "Raw JSON string from LLM",
       "json_object": {...},  # Parsed JSON object
       "usage": {
           "prompt_tokens": 150,
           "completion_tokens": 80, 
           "total_tokens": 230,
           "cost": 0.00046,
           "model_name": "gpt-4",
           "raw_response": {...}  # Full driver response
       }
   }

Integration with Other Modules
------------------------------

The core module integrates tightly with:

- **Field Definitions**: Uses [`get_registry_snapshot()`](../api/field_definitions.rst#get_registry_snapshot) for field defaults
- **Drivers**: Uses [`get_driver_for_model()`](../api/drivers.rst#get_driver_for_model) for automatic driver selection  
- **Tools**: Uses utility functions like [`convert_value()`](../api/tools.rst#convert_value) and [`clean_json_text()`](../api/tools.rst#clean_json_text)

Best Practices
--------------

1. **Use extract_and_jsonify()** for most simple extraction tasks
2. **Use extract_with_model()** when working with Pydantic models and field registry
3. **Use stepwise_extract_with_model()** for complex documents or when you need detailed per-field control
4. **Use manual_extract_and_jsonify()** when you need direct control over the driver configuration
5. **Always handle ConnectionError** in production code for network resilience
6. **Enable ai_cleanup=True** (default) for better handling of imperfect LLM responses