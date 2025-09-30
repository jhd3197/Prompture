Quick Start Guide
=================

This guide will get you up and running with Prompture in just a few minutes. After completing the :doc:`installation` guide, you're ready to start extracting structured data from text using LLMs.

Basic Setup
-----------

First, make sure you have your API keys configured in a ``.env`` file:

.. code-block:: bash

    # .env file
    OPENAI_API_KEY=your_openai_api_key_here
    ANTHROPIC_API_KEY=your_anthropic_api_key_here

Import Prompture in your Python script:

.. code-block:: python

    from prompture import extract_and_jsonify, extract_with_model, field_from_registry
    from pydantic import BaseModel
    from typing import Optional

Your First Extraction
---------------------

Let's start with a simple example extracting person information from text:

.. code-block:: python

    from prompture import extract_and_jsonify

    # Define what data you want to extract
    fields = {
        "name": "string",
        "age": "integer", 
        "occupation": "string"
    }

    # Text containing the information
    text = "Sarah Johnson is a 32-year-old software engineer at TechCorp."

    # Extract structured data
    result = extract_and_jsonify(
        prompt=text,
        fields=fields,
        model_name="openai/gpt-3.5-turbo"
    )

    print(result)
    # Output: {"name": "Sarah Johnson", "age": 32, "occupation": "software engineer"}

Using Field Definitions
-----------------------

Prompture provides a powerful field definitions system with built-in validation and descriptions. Here's how to use pre-defined fields:

.. code-block:: python

    from prompture import extract_and_jsonify

    # Use built-in field definitions
    fields = {
        "name": "name",           # Built-in name field with validation
        "age": "age",             # Built-in age field (integer, 0-150)
        "email": "email",         # Built-in email field with format validation
        "phone": "phone"          # Built-in phone field
    }

    text = """
    Contact: John Doe, 28 years old
    Email: john.doe@company.com
    Phone: +1-555-123-4567
    """

    result = extract_and_jsonify(
        prompt=text,
        fields=fields,
        model_name="openai/gpt-4"
    )

    print(result)

Using Pydantic Models (Recommended)
-----------------------------------

The modern approach uses Pydantic models with the field registry system. This provides better type safety and validation:

.. code-block:: python

    from pydantic import BaseModel
    from typing import Optional
    from prompture import field_from_registry, extract_with_model

    # Define your data model
    class Person(BaseModel):
        name: str = field_from_registry("name")
        age: int = field_from_registry("age")
        email: Optional[str] = field_from_registry("email")
        occupation: Optional[str] = field_from_registry("occupation")

    # Extract using the model
    text = "Dr. Alice Smith, 45, is a cardiologist. Email: alice@hospital.com"

    result = extract_with_model(
        model_class=Person,
        prompt=text,
        model_name="openai/gpt-4"
    )

    print(result)
    print(f"Name: {result.name}, Age: {result.age}")

Custom Field Definitions
------------------------

You can register your own field definitions for reusable, validated fields:

.. code-block:: python

    from prompture import register_field, field_from_registry, extract_with_model
    from pydantic import BaseModel
    from typing import List, Optional

    # Register custom fields
    register_field("skills", {
        "type": "list",
        "description": "List of professional skills and competencies", 
        "instructions": "Extract skills as a list of strings",
        "default": [],
        "nullable": True
    })

    register_field("experience_years", {
        "type": "int",
        "description": "Years of professional experience",
        "instructions": "Extract total years of work experience",
        "default": 0,
        "nullable": True
    })

    # Use custom fields in a model
    class Professional(BaseModel):
        name: str = field_from_registry("name")
        skills: Optional[List[str]] = field_from_registry("skills")
        experience_years: Optional[int] = field_from_registry("experience_years")
        occupation: Optional[str] = field_from_registry("occupation")

    # Extract professional profile
    text = """
    Michael Chen has 8 years of experience as a data scientist.
    His skills include Python, machine learning, SQL, and data visualization.
    """

    result = extract_with_model(
        model_class=Professional,
        prompt=text,
        model_name="openai/gpt-4"
    )

    print(f"Professional: {result.name}")
    print(f"Skills: {', '.join(result.skills)}")
    print(f"Experience: {result.experience_years} years")

Different LLM Providers
-----------------------

Prompture supports multiple LLM providers. Simply change the ``model_name`` parameter:

.. code-block:: python

    from prompture import extract_and_jsonify

    fields = {"name": "name", "age": "age"}
    text = "Emma Watson, 33 years old"

    # OpenAI GPT models
    result1 = extract_and_jsonify(text, fields, model_name="openai/gpt-4")
    result2 = extract_and_jsonify(text, fields, model_name="openai/gpt-3.5-turbo")

    # Anthropic Claude models  
    result3 = extract_and_jsonify(text, fields, model_name="anthropic/claude-3-haiku-20240307")
    result4 = extract_and_jsonify(text, fields, model_name="anthropic/claude-3-sonnet-20240229")

    # Google Gemini models
    result5 = extract_and_jsonify(text, fields, model_name="google/gemini-pro")

    # Groq models (fast inference)
    result6 = extract_and_jsonify(text, fields, model_name="groq/llama2-70b-4096")

    # Local models via Ollama
    result7 = extract_and_jsonify(text, fields, model_name="ollama/llama2")

Template Variables
------------------

Prompture supports template variables in field definitions for dynamic defaults:

.. code-block:: python

    from prompture import register_field, field_from_registry, extract_with_model
    from pydantic import BaseModel

    # Register field with template variables
    register_field("processed_at", {
        "type": "str",
        "description": "When this data was processed",
        "instructions": "Use {{current_datetime}} for processing timestamp",
        "default": "{{current_datetime}}",
        "nullable": False
    })

    register_field("document_year", {
        "type": "int", 
        "description": "Year of the document",
        "instructions": "Extract year, use {{current_year}} if not specified",
        "default": "{{current_year}}",
        "nullable": False
    })

    class Document(BaseModel):
        title: str = field_from_registry("title")
        document_year: int = field_from_registry("document_year")
        processed_at: str = field_from_registry("processed_at")

    text = "Annual Report: Company Performance Review"

    result = extract_with_model(
        model_class=Document,
        prompt=text,
        model_name="openai/gpt-4"
    )

    print(f"Document: {result.title}")
    print(f"Year: {result.document_year}")  # Will use current year if not found
    print(f"Processed: {result.processed_at}")  # Current datetime

Error Handling
--------------

Prompture provides built-in error handling and validation:

.. code-block:: python

    from prompture import extract_and_jsonify, validate_against_schema
    import json

    try:
        result = extract_and_jsonify(
            prompt="Invalid text with no clear data",
            fields={"name": "name", "age": "age"}, 
            model_name="openai/gpt-4"
        )
        
        # Validate the result
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0, "maximum": 150}
            },
            "required": ["name", "age"]
        }
        
        is_valid = validate_against_schema(result, schema)
        if is_valid:
            print("✅ Valid result:", result)
        else:
            print("❌ Invalid result format")
            
    except Exception as e:
        print(f"❌ Extraction failed: {e}")

Batch Processing
----------------

For processing multiple texts, you can use a loop or batch approach:

.. code-block:: python

    from prompture import extract_with_model
    from pydantic import BaseModel
    from typing import Optional

    class Contact(BaseModel):
        name: str = field_from_registry("name")
        email: Optional[str] = field_from_registry("email")
        phone: Optional[str] = field_from_registry("phone")

    # Multiple text samples
    texts = [
        "John Smith - john@company.com - (555) 123-4567",
        "Alice Johnson, email: alice.j@startup.io, phone: +1-555-987-6543", 
        "Bob Wilson | bwilson@corp.com | 555.111.2222"
    ]

    results = []
    for text in texts:
        try:
            contact = extract_with_model(
                model_class=Contact,
                prompt=text,
                model_name="openai/gpt-3.5-turbo"
            )
            results.append(contact)
        except Exception as e:
            print(f"Failed to extract from '{text}': {e}")

    for contact in results:
        print(f"Name: {contact.name}, Email: {contact.email}")

Configuration Tips
------------------

**Environment Variables**
  Keep API keys in ``.env`` files and never commit them to version control.

**Model Selection**
  - Use ``gpt-3.5-turbo`` for fast, cost-effective extraction
  - Use ``gpt-4`` for complex or nuanced extraction tasks
  - Use ``claude-3-haiku`` for fast Anthropic processing
  - Use local models (Ollama) for privacy or offline use

**Field Definitions**
  - Use built-in fields when possible for consistency
  - Register custom fields for domain-specific data
  - Include clear descriptions and instructions in field definitions

**Error Handling**
  - Always wrap extraction calls in try-catch blocks
  - Validate results when data quality is critical
  - Use nullable fields for optional data

Next Steps
----------

Now that you've learned the basics, explore:

- :doc:`examples` - More comprehensive examples and use cases
- :doc:`field_definitions` - Advanced field definition techniques  
- :doc:`drivers` - Working with different LLM providers
- :doc:`api/index` - Complete API reference

For practical examples with different LLM providers and complex extraction scenarios, see the :doc:`examples` section.