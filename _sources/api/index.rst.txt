API Reference
=============

This section provides detailed documentation for all Prompture modules, classes, and functions.

Overview
--------

Prompture is organized into several key modules:

- **Core Module** (:doc:`core`) - Main extraction functions and utilities
- **Field Definitions** (:doc:`field_definitions`) - Field registry and validation system  
- **Drivers** (:doc:`drivers`) - LLM provider interfaces
- **Runner** (:doc:`runner`) - Test suite and batch processing
- **Validator** (:doc:`validator`) - Data validation utilities

Quick Reference
---------------

**Main Extraction Functions**

.. code-block:: python

    from prompture import (
        extract_and_jsonify,        # Basic extraction with field definitions
        extract_with_model,         # Extract using Pydantic models  
        stepwise_extract_with_model # Multi-step extraction process
    )

**Field Registry System**

.. code-block:: python

    from prompture import (
        field_from_registry,        # Get field for Pydantic models
        register_field,             # Register custom field definition
        get_registry_snapshot,      # View all registered fields
        clear_registry             # Clear custom fields
    )

**Driver and Utilities**

.. code-block:: python

    from prompture import (
        Driver,                     # Base driver class
        validate_against_schema,    # JSON schema validation
        run_suite_from_spec        # Run test suites
    )

Core Functions
--------------

extract_and_jsonify()
~~~~~~~~~~~~~~~~~~~~~

Main function for extracting structured data from text using field definitions.

.. code-block:: python

    def extract_and_jsonify(
        prompt: str,
        fields: dict,
        model_name: str = "auto",
        **kwargs
    ) -> dict:
        """
        Extract structured JSON data from text using field definitions.
        
        Args:
            prompt: The input text to extract data from
            fields: Dictionary mapping field names to field types or definitions
            model_name: LLM model to use (e.g., "openai/gpt-4")
            **kwargs: Additional parameters passed to the driver
            
        Returns:
            Dictionary containing extracted structured data
            
        Raises:
            ValueError: If extraction fails or data is invalid
            RuntimeError: If model or driver is not available
        """

**Example:**

.. code-block:: python

    result = extract_and_jsonify(
        prompt="John Smith is 25 years old",
        fields={"name": "name", "age": "age"},
        model_name="openai/gpt-4"
    )
    # Returns: {"name": "John Smith", "age": 25}

extract_with_model()
~~~~~~~~~~~~~~~~~~~~

Extract data using Pydantic models with the field registry system.

.. code-block:: python

    def extract_with_model(
        model_class: Type[BaseModel],
        prompt: str,
        model_name: str = "auto",
        **kwargs
    ) -> BaseModel:
        """
        Extract structured data using a Pydantic model.
        
        Args:
            model_class: Pydantic model class defining the output structure
            prompt: The input text to extract data from
            model_name: LLM model to use
            **kwargs: Additional parameters
            
        Returns:
            Instance of the Pydantic model with extracted data
        """

**Example:**

.. code-block:: python

    class Person(BaseModel):
        name: str = field_from_registry("name")
        age: int = field_from_registry("age")
    
    person = extract_with_model(
        model_class=Person,
        prompt="Alice Johnson, 32 years old",
        model_name="openai/gpt-4"
    )

stepwise_extract_with_model()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-step extraction process with enhanced validation and error handling.

.. code-block:: python

    def stepwise_extract_with_model(
        model_class: Type[BaseModel],
        prompt: str,
        model_name: str = "auto",
        **kwargs
    ) -> BaseModel:
        """
        Extract data using a multi-step validation process.
        
        This function performs extraction in multiple phases with
        validation at each step for improved accuracy.
        
        Args:
            model_class: Pydantic model class
            prompt: Input text
            model_name: LLM model to use
            **kwargs: Additional parameters
            
        Returns:
            Validated Pydantic model instance
        """

Field Registry System
---------------------

field_from_registry()
~~~~~~~~~~~~~~~~~~~~~

Get a field definition from the registry for use in Pydantic models.

.. code-block:: python

    def field_from_registry(field_name: str) -> Any:
        """
        Retrieve a field definition from the registry.
        
        Args:
            field_name: Name of the registered field
            
        Returns:
            Pydantic Field object with the registered definition
            
        Raises:
            KeyError: If field_name is not registered
        """

register_field()
~~~~~~~~~~~~~~~~

Register a custom field definition in the global registry.

.. code-block:: python

    def register_field(name: str, definition: dict) -> None:
        """
        Register a custom field definition.
        
        Args:
            name: Field name identifier
            definition: Dictionary containing field specification
            
        Definition format:
            {
                "type": "str|int|float|list|dict|bool",
                "description": "Human readable description",
                "instructions": "Instructions for LLM extraction", 
                "default": "Default value or template variable",
                "nullable": True/False,
                "validation": {...}  # Optional validation rules
            }
        """

**Example:**

.. code-block:: python

    register_field("skills", {
        "type": "list",
        "description": "List of professional skills",
        "instructions": "Extract skills as list of strings",
        "default": [],
        "nullable": True
    })

Built-in Field Types
--------------------

Prompture includes many built-in field definitions:

**Personal Information**
  - ``name`` - Person's full name
  - ``age`` - Age in years (0-150)
  - ``email`` - Email address with validation
  - ``phone`` - Phone number
  - ``address`` - Physical address

**Professional Fields**
  - ``occupation`` - Job title or profession
  - ``company`` - Company or organization name
  - ``experience_years`` - Years of experience

**Temporal Fields**
  - ``date`` - Date in various formats
  - ``year`` - Year (1900-current)
  - ``last_updated`` - Timestamp field

**Content Fields**
  - ``title`` - Title or heading text
  - ``description`` - Longer descriptive text
  - ``category`` - Classification or category
  - ``content`` - General content field

Driver System
-------------

The driver system provides a unified interface for different LLM providers.

Supported Models
~~~~~~~~~~~~~~~~

**OpenAI**
  - ``openai/gpt-4`` - GPT-4 (recommended for complex tasks)
  - ``openai/gpt-3.5-turbo`` - GPT-3.5 Turbo (fast and cost-effective)

**Anthropic**
  - ``anthropic/claude-3-opus-20240229`` - Claude 3 Opus (most capable)
  - ``anthropic/claude-3-sonnet-20240229`` - Claude 3 Sonnet (balanced)
  - ``anthropic/claude-3-haiku-20240307`` - Claude 3 Haiku (fast)

**Google**
  - ``google/gemini-pro`` - Gemini Pro
  - ``google/gemini-pro-vision`` - Gemini Pro with vision

**Groq**
  - ``groq/llama2-70b-4096`` - Llama 2 70B (fast inference)
  - ``groq/mixtral-8x7b-32768`` - Mixtral 8x7B

**Local Models**
  - ``ollama/llama2`` - Local Llama 2 via Ollama
  - ``ollama/mistral`` - Local Mistral via Ollama

Driver Base Class
~~~~~~~~~~~~~~~~~

.. code-block:: python

    class Driver:
        """Base class for LLM drivers."""
        
        def __init__(self, model_name: str, **kwargs):
            """Initialize the driver with model configuration."""
            
        def ask_for_json(self, prompt: str, **kwargs) -> dict:
            """Send prompt to LLM and return JSON response."""
            
        def validate_response(self, response: dict) -> bool:
            """Validate the LLM response format."""

Validation and Utilities
------------------------

validate_against_schema()
~~~~~~~~~~~~~~~~~~~~~~~~~

Validate extracted data against a JSON schema.

.. code-block:: python

    def validate_against_schema(data: dict, schema: dict) -> bool:
        """
        Validate data against a JSON schema.
        
        Args:
            data: Dictionary to validate
            schema: JSON schema specification
            
        Returns:
            True if data matches schema, False otherwise
        """

**Example:**

.. code-block:: python

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0, "maximum": 150}
        },
        "required": ["name", "age"]
    }
    
    is_valid = validate_against_schema(result, schema)

Error Handling
--------------

Prompture defines several custom exceptions:

.. code-block:: python

    class PromptureError(Exception):
        """Base exception for Prompture errors."""
        
    class ExtractionError(PromptureError):
        """Raised when data extraction fails."""
        
    class ValidationError(PromptureError):
        """Raised when data validation fails."""
        
    class DriverError(PromptureError):
        """Raised when driver operations fail."""

Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Prompture uses environment variables for configuration:

.. code-block:: bash

    # API Keys
    OPENAI_API_KEY=your_openai_key
    ANTHROPIC_API_KEY=your_anthropic_key
    GOOGLE_API_KEY=your_google_key
    GROQ_API_KEY=your_groq_key
    
    # Custom Endpoints
    OPENAI_BASE_URL=https://api.openai.com/v1
    LOCAL_API_BASE_URL=http://localhost:8000
    OLLAMA_BASE_URL=http://localhost:11434

Template Variables
~~~~~~~~~~~~~~~~~~

Field definitions support template variables that are automatically resolved:

- ``{{current_year}}`` - Current year (e.g., 2024)
- ``{{current_date}}`` - Current date (YYYY-MM-DD format)
- ``{{current_datetime}}`` - Current datetime (ISO format)

**Example:**

.. code-block:: python

    register_field("processed_at", {
        "type": "str",
        "description": "Processing timestamp",
        "default": "{{current_datetime}}",
        "nullable": False
    })

Module Reference
----------------

.. toctree::
   :maxdepth: 2
   
   core
   field_definitions
   drivers
   tools
   runner
   validator

For detailed module documentation, select a module from the list above.

The following API documentation files have been generated using Sphinx autodoc:

Core Modules
~~~~~~~~~~~~

- **Core Module** (:doc:`core`) - Main extraction functions: [`extract_and_jsonify()`](core.rst#extract_and_jsonify), [`extract_with_model()`](core.rst#extract_with_model), [`stepwise_extract_with_model()`](core.rst#stepwise_extract_with_model)
- **Field Definitions** (:doc:`field_definitions`) - Field registry system: [`field_from_registry()`](field_definitions.rst#field_from_registry), [`register_field()`](field_definitions.rst#register_field), [`get_registry_snapshot()`](field_definitions.rst#get_registry_snapshot)
- **Drivers** (:doc:`drivers`) - LLM provider interfaces: [`get_driver_for_model()`](drivers.rst#get_driver_for_model), [`OpenAIDriver`](drivers.rst#openaidriver), [`ClaudeDriver`](drivers.rst#claudedriver), and more

Utility Modules
~~~~~~~~~~~~~~~

- **Tools** (:doc:`tools`) - Utility functions: [`convert_value()`](tools.rst#convert_value), [`clean_json_text()`](tools.rst#clean_json_text)
- **Runner** (:doc:`runner`) - Test suite and batch processing utilities
- **Validator** (:doc:`validator`) - Data validation and schema checking utilities