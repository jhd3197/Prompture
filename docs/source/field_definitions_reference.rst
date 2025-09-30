===========================
Field Definitions Reference
===========================

Overview
========

The Prompture field definitions system provides a centralized registry of structured data extraction fields. Each field definition specifies the data type, description, extraction instructions, default values, and validation rules. This system enables consistent, reusable field configurations across your data extraction workflows.

**Key Features:**

- **Centralized Registry**: All field definitions stored in a global registry with thread-safe access
- **Template Variables**: Dynamic defaults using ``{{current_year}}``, ``{{current_date}}``, etc.
- **Pydantic Integration**: Seamless integration with Pydantic models via :func:`field_from_registry()`
- **Custom Fields**: Easy registration of domain-specific fields with :func:`register_field()`
- **Type Safety**: Full type hints and validation support

Quick Start
===========

Basic Usage with Built-in Fields
---------------------------------

.. code-block:: python

    from pydantic import BaseModel
    from prompture import field_from_registry, stepwise_extract_with_model
    
    class Person(BaseModel):
        name: str = field_from_registry("name")
        age: int = field_from_registry("age")
        email: str = field_from_registry("email")
    
    # Use with extraction
    result = stepwise_extract_with_model(
        Person,
        "John Smith is 25 years old, email: john@example.com",
        model_name="openai/gpt-4"
    )

Registering Custom Fields
-------------------------

.. code-block:: python

    from prompture import register_field, field_from_registry
    
    # Register a custom field with template variables
    register_field("document_date", {
        "type": "str",
        "description": "Document creation or processing date",
        "instructions": "Use {{current_date}} if not specified in document",
        "default": "{{current_date}}",
        "nullable": False
    })
    
    # Use in Pydantic model
    class Document(BaseModel):
        title: str = field_from_registry("name")  # Reuse built-in field
        created_date: str = field_from_registry("document_date")  # Custom field

Built-in Field Definitions
===========================

The following field definitions are available by default in the ``BASE_FIELD_DEFINITIONS`` registry:

Person/Identity Fields
----------------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 30 30 10 10 15

   * - Field Name
     - Type
     - Description
     - Instructions
     - Default
     - Nullable
     - Notes
   * - ``name``
     - ``str``
     - Full legal name of the person
     - Extract as-is, no modifications
     - ``""``
     - ``False``
     - Required field
   * - ``age``
     - ``int``
     - The age of the person in number of years
     - Calculate as ``{{current_year}} - birth_year`` if needed
     - ``0``
     - ``False``
     - Uses ``{{current_year}}``
   * - ``birth_year``
     - ``int``
     - The year the person was born (YYYY)
     - Extract as a 4-digit year number
     - ``None``
     - ``True``
     - Optional field

**Usage Example:**

.. code-block:: python

    class Person(BaseModel):
        name: str = field_from_registry("name")
        age: int = field_from_registry("age")
        birth_year: int = field_from_registry("birth_year")

Contact Information Fields
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 30 30 10 10 15

   * - Field Name
     - Type
     - Description
     - Instructions
     - Default
     - Nullable
     - Notes
   * - ``email``
     - ``str``
     - Primary email address
     - Extract in lowercase, verify basic email format
     - ``""``
     - ``True``
     - Optional field
   * - ``phone``
     - ``str``
     - Primary phone number
     - Extract digits only, standardize to E.164 format if possible
     - ``""``
     - ``True``
     - Optional field
   * - ``address``
     - ``str``
     - Full mailing address
     - Combine all address components into a single string
     - ``""``
     - ``True``
     - Optional field

**Usage Example:**

.. code-block:: python

    class ContactInfo(BaseModel):
        email: str = field_from_registry("email")
        phone: str = field_from_registry("phone")
        address: str = field_from_registry("address")

Professional Information Fields
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 30 30 10 10 15

   * - Field Name
     - Type
     - Description
     - Instructions
     - Default
     - Nullable
     - Notes
   * - ``occupation``
     - ``str``
     - Current job title or profession
     - Extract primary occupation, standardize common titles
     - ``""``
     - ``True``
     - Optional field
   * - ``company``
     - ``str``
     - Current employer or company name
     - Extract organization name, remove legal suffixes
     - ``""``
     - ``True``
     - Optional field
   * - ``experience_years``
     - ``int``
     - Years of professional experience
     - Calculate total years of relevant experience
     - ``0``
     - ``True``
     - Optional field

**Usage Example:**

.. code-block:: python

    class Professional(BaseModel):
        occupation: str = field_from_registry("occupation")
        company: str = field_from_registry("company")
        experience_years: int = field_from_registry("experience_years")

Metadata Fields
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 30 30 10 10 15

   * - Field Name
     - Type
     - Description
     - Instructions
     - Default
     - Nullable
     - Notes
   * - ``source``
     - ``str``
     - Source of the extracted information
     - Record origin of data (e.g., 'resume', 'linkedin')
     - ``"unknown"``
     - ``False``
     - Required field
   * - ``last_updated``
     - ``str``
     - Last update timestamp (ISO format)
     - Use ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ), default to ``{{current_datetime}}``
     - ``"{{current_datetime}}"``
     - ``False``
     - Uses ``{{current_datetime}}``
   * - ``confidence_score``
     - ``float``
     - Confidence score of extraction (0.0-1.0)
     - Calculate based on extraction certainty
     - ``0.0``
     - ``False``
     - Required field

**Usage Example:**

.. code-block:: python

    class DataRecord(BaseModel):
        source: str = field_from_registry("source")
        last_updated: str = field_from_registry("last_updated")
        confidence_score: float = field_from_registry("confidence_score")


Location Fields
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 30 30 10 10 15

   * - Field Name
     - Type
     - Description
     - Instructions
     - Default
     - Nullable
     - Notes
   * - ``city``
     - ``str``
     - City name
     - Extract city name, standardize capitalization
     - ``""``
     - ``True``
     - Optional field
   * - ``state``
     - ``str``
     - State or province name
     - Extract state/province, use full name or abbreviation
     - ``""``
     - ``True``
     - Optional field
   * - ``postal_code``
     - ``str``
     - Postal or ZIP code
     - Extract postal code, maintain original format
     - ``""``
     - ``True``
     - Optional field
   * - ``country``
     - ``str``
     - Country name
     - Extract country name, use full English name
     - ``""``
     - ``True``
     - Optional field
   * - ``coordinates``
     - ``str``
     - Geographic coordinates (lat, long)
     - Extract as 'latitude,longitude' format if available
     - ``""``
     - ``True``
     - Optional field

**Usage Example:**

.. code-block:: python

    class LocationData(BaseModel):
        city: str = field_from_registry("city")
        country: str = field_from_registry("country")
        postal_code: str = field_from_registry("postal_code")

Demographic Fields
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 30 30 10 10 15

   * - Field Name
     - Type
     - Description
     - Instructions
     - Default
     - Nullable
     - Notes
   * - ``gender``
     - ``str``
     - Gender identification
     - Extract gender if explicitly stated, otherwise leave empty
     - ``""``
     - ``True``
     - Optional field
   * - ``nationality``
     - ``str``
     - Nationality or citizenship
     - Extract nationality, use country demonym
     - ``""``
     - ``True``
     - Optional field
   * - ``marital_status``
     - ``str``
     - Marital status
     - Extract marital status (single, married, divorced, etc.)
     - ``""``
     - ``True``
     - Optional field
   * - ``language``
     - ``str``
     - Primary language spoken
     - Extract primary or native language
     - ``""``
     - ``True``
     - Optional field

**Usage Example:**

.. code-block:: python

    class DemographicData(BaseModel):
        nationality: str = field_from_registry("nationality")
        language: str = field_from_registry("language")

Education Fields
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 30 30 10 10 15

   * - Field Name
     - Type
     - Description
     - Instructions
     - Default
     - Nullable
     - Notes
   * - ``education_level``
     - ``str``
     - Highest education level
     - Extract highest degree (High School, Bachelor's, Master's, PhD, etc.)
     - ``""``
     - ``True``
     - Optional field
   * - ``graduation_year``
     - ``int``
     - Year of graduation
     - Extract graduation year as 4-digit number
     - ``None``
     - ``True``
     - Optional field
   * - ``gpa``
     - ``float``
     - Grade point average
     - Extract GPA, convert to 4.0 scale if needed
     - ``None``
     - ``True``
     - Optional field

**Usage Example:**

.. code-block:: python

    class EducationData(BaseModel):
        education_level: str = field_from_registry("education_level")
        graduation_year: int = field_from_registry("graduation_year")
        gpa: float = field_from_registry("gpa")

Financial Fields
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 30 30 10 10 15

   * - Field Name
     - Type
     - Description
     - Instructions
     - Default
     - Nullable
     - Notes
   * - ``salary``
     - ``float``
     - Annual salary amount
     - Extract salary as numeric value, remove currency symbols
     - ``None``
     - ``True``
     - Optional field
   * - ``currency``
     - ``str``
     - Currency code
     - Extract or infer currency code (USD, EUR, GBP, etc.)
     - ``"USD"``
     - ``True``
     - Optional field
   * - ``bonus``
     - ``float``
     - Bonus amount
     - Extract bonus as numeric value
     - ``None``
     - ``True``
     - Optional field

**Usage Example:**

.. code-block:: python

    class FinancialData(BaseModel):
        salary: float = field_from_registry("salary")
        currency: str = field_from_registry("currency")

Social Media Fields
-------------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 30 30 10 10 15

   * - Field Name
     - Type
     - Description
     - Instructions
     - Default
     - Nullable
     - Notes
   * - ``sentiment``
     - ``str``
     - Sentiment classification
     - Classify as positive, negative, or neutral
     - ``"neutral"``
     - ``True``
     - Optional field
   * - ``hashtags``
     - ``str``
     - Hashtags from content
     - Extract all hashtags as comma-separated list
     - ``""``
     - ``True``
     - Optional field
   * - ``mentions``
     - ``str``
     - User mentions from content
     - Extract all @mentions as comma-separated list
     - ``""``
     - ``True``
     - Optional field
   * - ``topic``
     - ``str``
     - Main topic or subject
     - Identify primary topic or theme of content
     - ``""``
     - ``True``
     - Optional field

**Usage Example:**

.. code-block:: python

    class SocialMediaData(BaseModel):
        sentiment: str = field_from_registry("sentiment")
        hashtags: str = field_from_registry("hashtags")
        topic: str = field_from_registry("topic")

Template Variable System
========================

Template variables provide dynamic default values that are resolved at runtime. They're especially useful for timestamps, dates, and calculated values.

Available Template Variables
----------------------------

The following template variables are available for use in field definitions:

``{{current_year}}``
  Current year as 4-digit integer (e.g., 2024)

``{{current_date}}``
  Current date in ISO format (YYYY-MM-DD)

``{{current_datetime}}``
  Current datetime in ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ)

``{{current_timestamp}}``
  Current Unix timestamp as integer

``{{current_month}}``
  Current month as integer (1-12)

``{{current_day}}``
  Current day of month as integer (1-31)

``{{current_weekday}}``
  Current day name as string (e.g., "Monday", "Tuesday")

``{{current_iso_week}}``
  Current ISO week number as integer (1-53)

Using Template Variables
------------------------

Template variables can be used in any string field within a field definition:

.. code-block:: python

    register_field("processing_date", {
        "type": "str",
        "description": "Date when document was processed",
        "instructions": "Use {{current_date}} if processing date not available",
        "default": "{{current_date}}",
        "nullable": False
    })
    
    register_field("academic_year", {
        "type": "str", 
        "description": "Academic year for enrollment",
        "instructions": "Use {{current_year}} for current enrollment",
        "default": "{{current_year}}-{{current_year}}",  # e.g. "2024-2024"
        "nullable": True
    })

Custom Template Variables
-------------------------

You can provide custom template variables when retrieving field definitions:

.. code-block:: python

    from prompture import get_field_definition
    
    # Custom variables for specific use cases
    custom_vars = {
        "report_year": 2023,
        "department": "Engineering"
    }
    
    # Register field with custom template
    register_field("report_title", {
        "type": "str",
        "description": "Report title",
        "instructions": "Use format: {{department}} Report {{report_year}}",
        "default": "{{department}} Report {{report_year}}",
        "nullable": False
    })
    
    # Retrieve with custom variables
    field_def = get_field_definition("report_title", 
                                   apply_templates=True, 
                                   custom_template_vars=custom_vars)

Custom Field Registration
=========================

Creating Custom Fields
-----------------------

Register custom fields using :func:`register_field()` to extend the built-in definitions:

.. code-block:: python

    from prompture import register_field, field_from_registry
    
    # Define field structure
    register_field("product_price", {
        "type": "str",
        "description": "Product price with currency symbol",
        "instructions": "Extract price including currency, handle ranges like $10-$15",
        "default": "Price not available", 
        "nullable": True
    })
    
    register_field("skills", {
        "type": "list",
        "description": "List of professional skills",
        "instructions": "Extract skills as comma-separated list, normalize tech names",
        "default": [],
        "nullable": True
    })

Field Definition Structure
--------------------------

Each field definition must include these required properties:

``type`` (required)
  Python type or string representation (``str``, ``int``, ``float``, ``bool``, ``list``, ``dict``)

``description`` (required)
  Human-readable description of the field purpose

``instructions`` (required)
  Specific extraction instructions for LLM processing

``default`` (required)
  Default value when field is not extracted (supports template variables)

``nullable`` (required)
  Boolean indicating if field accepts None/null values

**Example:**

.. code-block:: python

    field_definition = {
        "type": "str",
        "description": "Product category classification",
        "instructions": "Classify into: Electronics, Clothing, Books, Home, Other",
        "default": "Other",
        "nullable": True
    }

Validation
----------

Field definitions are automatically validated when registered:

.. code-block:: python

    from prompture.tools import validate_field_definition
    
    # Validate before registering
    field_def = {
        "type": "str",
        "description": "Valid field",
        "instructions": "Extract text value",
        "default": "",
        "nullable": True
    }
    
    if validate_field_definition(field_def):
        register_field("my_field", field_def)
    else:
        print("Invalid field definition")

Integration Examples
====================

Complete Extraction Workflow
-----------------------------

Here's a complete example showing field definitions in a real extraction scenario:

.. code-block:: python

    from pydantic import BaseModel
    from prompture import (
        field_from_registry, 
        register_field, 
        stepwise_extract_with_model
    )
    
    # Register custom business fields
    register_field("industry", {
        "type": "str",
        "description": "Business industry classification",
        "instructions": "Classify into standard industry categories",
        "default": "Unknown",
        "nullable": True
    })
    
    register_field("founded_year", {
        "type": "int",
        "description": "Year company was founded",
        "instructions": "Extract founding year, use {{current_year}} if recent",
        "default": None,
        "nullable": True
    })
    
    # Create comprehensive model
    class BusinessProfile(BaseModel):
        # Built-in fields
        name: str = field_from_registry("name")
        email: str = field_from_registry("email")
        phone: str = field_from_registry("phone")
        address: str = field_from_registry("address")
        
        # Professional fields
        company: str = field_from_registry("company")
        
        # Custom fields
        industry: str = field_from_registry("industry")
        founded_year: int = field_from_registry("founded_year")
        
        # Metadata
        source: str = field_from_registry("source")
        last_updated: str = field_from_registry("last_updated")
        confidence_score: float = field_from_registry("confidence_score")
    
    # Sample business text
    business_text = """
    TechStart Solutions is a cloud computing company founded in 2019.
    Contact: Sarah Johnson, CEO
    Email: sarah@techstart.com
    Phone: (555) 123-4567
    Address: 123 Innovation Drive, San Francisco, CA 94105
    Industry: Software as a Service (SaaS)
    """
    
    # Extract structured data
    result = stepwise_extract_with_model(
        BusinessProfile,
        business_text,
        model_name="openai/gpt-4"
    )
    
    print(result.model_dump())

Multi-Domain Field Sets
-----------------------

Organize fields by domain for better maintainability:

.. code-block:: python

    # E-commerce fields
    ecommerce_fields = {
        "product_name": {
            "type": "str",
            "description": "Product name or title",
            "instructions": "Extract main product name, exclude brand",
            "default": "Unknown Product",
            "nullable": False
        },
        "sku": {
            "type": "str", 
            "description": "Product SKU or model number",
            "instructions": "Extract alphanumeric SKU code",
            "default": "",
            "nullable": True
        },
        "category": {
            "type": "str",
            "description": "Product category",
            "instructions": "Classify into Electronics, Clothing, Books, etc.",
            "default": "Other",
            "nullable": True
        }
    }
    
    # Medical fields
    medical_fields = {
        "patient_id": {
            "type": "str",
            "description": "Patient identification number",
            "instructions": "Extract patient ID, mask if sensitive",
            "default": "",
            "nullable": True
        },
        "diagnosis": {
            "type": "str",
            "description": "Primary diagnosis or condition",
            "instructions": "Extract main diagnosis, use medical terminology",
            "default": "",
            "nullable": True
        },
        "treatment_date": {
            "type": "str",
            "description": "Date of treatment or visit",
            "instructions": "Extract date, use {{current_date}} if not specified",
            "default": "{{current_date}}",
            "nullable": False
        }
    }
    
    # Register field sets
    from prompture import add_field_definitions
    
    add_field_definitions(ecommerce_fields)
    add_field_definitions(medical_fields)

External Configuration Files
----------------------------

Load field definitions from external YAML or JSON files:

**field_definitions.yaml:**

.. code-block:: yaml

    document_fields:
      title:
        type: str
        description: "Document title or heading"
        instructions: "Extract main document title"
        default: "Untitled Document"
        nullable: false
      
      author:
        type: str
        description: "Document author or creator"
        instructions: "Extract author name, handle multiple authors"
        default: "Unknown Author"
        nullable: true
      
      created_date:
        type: str
        description: "Document creation date"
        instructions: "Use {{current_date}} if date not found"
        default: "{{current_date}}"
        nullable: false

**Python integration:**

.. code-block:: python

    from prompture.tools import load_field_definitions
    from prompture import add_field_definitions
    
    # Load from external file
    external_fields = load_field_definitions("field_definitions.yaml")
    
    # Register all fields from the file
    add_field_definitions(external_fields)
    
    # Use in models
    class Document(BaseModel):
        title: str = field_from_registry("title")
        author: str = field_from_registry("author") 
        created_date: str = field_from_registry("created_date")

Registry Management
===================

The field definitions registry provides several utility functions for managing field definitions:

Inspecting the Registry
-----------------------

.. code-block:: python

    from prompture import (
        get_field_names,
        get_required_fields,
        get_field_definition,
        get_registry_snapshot
    )
    
    # List all available fields
    all_fields = get_field_names()
    print(f"Available fields: {all_fields}")
    
    # Get required fields only
    required_fields = get_required_fields()
    print(f"Required fields: {required_fields}")
    
    # Inspect specific field
    name_field = get_field_definition("name")
    print(f"Name field: {name_field}")
    
    # Get full registry snapshot
    registry = get_registry_snapshot()
    print(f"Registry contains {len(registry)} fields")

Registry Maintenance
--------------------

.. code-block:: python

    from prompture import reset_registry, clear_registry
    
    # Reset to base definitions only
    reset_registry()  # Keeps built-in fields, removes custom ones
    
    # Clear everything (use with caution)
    clear_registry()  # Removes ALL fields including built-ins

Best Practices
==============

Field Naming Conventions
-------------------------

- Use descriptive, lowercase names with underscores: ``first_name``, ``created_date``
- Group related fields with prefixes: ``contact_email``, ``contact_phone``  
- Avoid abbreviations: use ``experience_years`` not ``exp_yrs``
- Be consistent across your domain

Type Selection Guidelines
-------------------------

- Use ``str`` for text, IDs, formatted data (dates, phone numbers)
- Use ``int`` for counts, years, numeric IDs
- Use ``float`` for scores, percentages, monetary values  
- Use ``list`` for multiple values of same type
- Use ``dict`` for nested structured data

Template Variable Usage
-----------------------

- Use ``{{current_date}}`` for document dates and timestamps
- Use ``{{current_year}}`` for age calculations and academic years
- Use ``{{current_datetime}}`` for precise processing timestamps
- Provide fallback values when templates might not resolve

Validation and Testing
----------------------

.. code-block:: python

    from prompture.tools import validate_field_definition
    
    # Always validate custom fields
    def create_safe_field(name, definition):
        if validate_field_definition(definition):
            register_field(name, definition)
            return True
        else:
            print(f"Invalid field definition for '{name}'")
            return False
    
    # Test field definitions with sample data
    def test_field_extraction(field_name, sample_text):
        class TestModel(BaseModel):
            test_field: str = field_from_registry(field_name)
        
        # Test extraction (requires API key)
        # result = stepwise_extract_with_model(TestModel, sample_text)
        # return result.test_field

Performance Considerations
--------------------------

- Register fields once at application startup
- Use :func:`get_registry_snapshot()` for bulk operations
- Cache field definitions for frequently used fields
- Validate definitions before registration to avoid runtime errors

API Reference
=============

Core Functions
--------------

.. py:function:: field_from_registry(field_name: str, apply_templates: bool = True, custom_template_vars: Optional[Dict[str, Any]] = None) -> Field

   Create a Pydantic Field from a registered field definition.
   
   :param field_name: Name of field in the registry
   :param apply_templates: Whether to apply template variable substitution
   :param custom_template_vars: Custom template variables for substitution
   :return: Configured Pydantic Field object
   :raises KeyError: If field_name not found in registry

.. py:function:: register_field(field_name: str, field_definition: FieldDefinition) -> None

   Register a single field definition in the global registry.
   
   :param field_name: Name of the field
   :param field_definition: Dictionary containing field configuration
   :raises ValueError: If field definition is invalid

.. py:function:: get_field_definition(field_name: str, apply_templates: bool = True, custom_template_vars: Optional[Dict[str, Any]] = None) -> Optional[FieldDefinition]

   Retrieve a field definition from the registry.
   
   :param field_name: Name of field to retrieve
   :param apply_templates: Whether to apply template substitution
   :param custom_template_vars: Custom variables for templates
   :return: Field definition dictionary or None if not found

Registry Management Functions
-----------------------------

.. py:function:: get_field_names() -> List[str]

   Get list of all registered field names.
   
   :return: List of field names in the registry

.. py:function:: get_required_fields() -> List[str]

   Get list of required (non-nullable) field names.
   
   :return: List of required field names

.. py:function:: add_field_definitions(field_definitions: Dict[str, FieldDefinition]) -> None

   Register multiple field definitions at once.
   
   :param field_definitions: Dictionary mapping field names to definitions

.. py:function:: get_registry_snapshot() -> Dict[str, FieldDefinition]

   Get a copy of the current registry state.
   
   :return: Dictionary of all registered field definitions

.. py:function:: reset_registry() -> None

   Reset registry to contain only base field definitions.

.. py:function:: clear_registry() -> None

   Remove all field definitions from registry.
   
   .. warning:: This removes all fields including built-in definitions.

See Also
========

- :doc:`api/field_definitions` - API documentation for field definitions module
- :doc:`examples` - Usage examples and tutorials
- :doc:`quickstart` - Getting started guide
- :doc:`api/core` - Core extraction functions