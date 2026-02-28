Field Definitions Module
========================

.. automodule:: prompture.field_definitions
   :members:
   :undoc-members:
   :show-inheritance:

The field definitions module provides a centralized registry system for defining reusable field specifications that enhance structured data extraction with type hints, descriptions, validation rules, and LLM-specific extraction instructions.

Overview
--------

The field definitions system allows you to:

- **Define reusable field specifications** with type information, descriptions, and extraction instructions
- **Register custom fields** that can be used across multiple extraction operations
- **Leverage built-in field definitions** for common data types (names, ages, emails, etc.)
- **Use template variables** for dynamic values like ``{{current_year}}`` and ``{{current_date}}``
- **Integrate with Pydantic models** through the [`field_from_registry()`](#field_from_registry) function

Registry Management Functions
-----------------------------

get_field_definition()
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: get_field_definition

Retrieve a specific field definition from the registry with optional template variable substitution.

**Example:**

.. code-block:: python

   # Get the built-in "age" field definition
   age_def = get_field_definition("age")
   print(age_def)
   # {
   #     "type": int,
   #     "description": "The age of the person in number of years.",
   #     "instructions": "Calculate as 2024 - birth_year if needed.",
   #     "default": 0,
   #     "nullable": False
   # }

register_field()
~~~~~~~~~~~~~~~

.. autofunction:: register_field

Register a new field definition or update an existing one in the global registry.

**Example:**

.. code-block:: python

   register_field("skills", {
       "type": list,
       "description": "List of professional skills and competencies",
       "instructions": "Extract as a list of strings, one skill per item",
       "default": [],
       "nullable": True
   })

add_field_definition()
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: add_field_definition

Alias for [`register_field()`](#register_field) - adds or updates a field definition in the registry.

add_field_definitions()
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: add_field_definitions

Register multiple field definitions at once from a dictionary.

**Example:**

.. code-block:: python

   new_fields = {
       "salary": {
           "type": float,
           "description": "Annual salary in USD",
           "instructions": "Extract numeric value, convert K/M suffixes",
           "default": 0.0,
           "nullable": True
       },
       "department": {
           "type": str,
           "description": "Department or division name",
           "instructions": "Extract official department name",
           "default": "",
           "nullable": True
       }
   }
   add_field_definitions(new_fields)

field_from_registry()
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: field_from_registry

Create a Pydantic Field object from a registered field definition for use in Pydantic models.

**Key Features:**

- Automatic conversion to Pydantic Field objects
- Template variable substitution in defaults and descriptions
- Type annotation integration
- Custom field configuration support

**Example:**

.. code-block:: python

   from pydantic import BaseModel
   from prompture import field_from_registry
   
   class Employee(BaseModel):
       name: str = field_from_registry("name")
       age: int = field_from_registry("age")
       email: str = field_from_registry("email")
       department: str = field_from_registry("department")

Registry Inspection Functions
-----------------------------

get_registry_snapshot()
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: get_registry_snapshot

Get a complete copy of the current field registry for inspection or backup.

**Example:**

.. code-block:: python

   registry = get_registry_snapshot()
   print(f"Available fields: {list(registry.keys())}")
   
   # Use in stepwise extraction for enhanced defaults
   from prompture.core import stepwise_extract_with_model
   result = stepwise_extract_with_model(
       model_cls=Person,
       text="...",
       model_name="openai/gpt-4",
       field_definitions=registry  # Explicit registry usage
   )

get_field_names()
~~~~~~~~~~~~~~~~

.. autofunction:: get_field_names

Get a list of all currently registered field names.

**Example:**

.. code-block:: python

   available_fields = get_field_names()
   print("Available field types:", available_fields)

get_required_fields()
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: get_required_fields

Get a list of field names that are marked as non-nullable (required fields).

**Example:**

.. code-block:: python

   required = get_required_fields()
   print("Required fields:", required)

Registry Maintenance Functions  
-----------------------------

clear_registry()
~~~~~~~~~~~~~~~

.. autofunction:: clear_registry

Remove all custom field definitions from the registry, keeping only built-in fields.

**Example:**

.. code-block:: python

   # Clear custom fields while preserving built-ins
   clear_registry()
   
   # Only built-in fields like "name", "age", "email" remain
   print(get_field_names())

reset_registry()
~~~~~~~~~~~~~~~

.. autofunction:: reset_registry

Completely reset the registry to its initial state with only built-in field definitions.

**Example:**

.. code-block:: python

   # Add some custom fields
   register_field("custom_field", {...})
   
   # Reset to built-in fields only
   reset_registry()
   
   # Custom field is now gone
   assert "custom_field" not in get_field_names()

Built-in Field Definitions
---------------------------

Prompture includes extensive built-in field definitions organized by category:

Personal Information Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   
   * - Field Name
     - Type
     - Description
   * - ``name``
     - str
     - Full legal name of the person
   * - ``age``
     - int
     - Age in years (0-150)
   * - ``birth_year``
     - int
     - Year born (YYYY format)

Contact Information Fields
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   
   * - Field Name  
     - Type
     - Description
   * - ``email``
     - str
     - Primary email address with validation
   * - ``phone``
     - str
     - Phone number in standardized format
   * - ``address``
     - str
     - Full mailing address

Professional Fields
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   
   * - Field Name
     - Type 
     - Description
   * - ``occupation``
     - str
     - Job title or profession
   * - ``company``
     - str
     - Company or organization name
   * - ``experience_years``
     - int
     - Years of professional experience

Temporal Fields
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   
   * - Field Name
     - Type
     - Description  
   * - ``date``
     - str
     - Date in various formats
   * - ``year``
     - int
     - Year (1900-current)
   * - ``last_updated``
     - str
     - Timestamp with template support

Content and Classification Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   
   * - Field Name
     - Type
     - Description
   * - ``title``
     - str
     - Title or heading text
   * - ``description``
     - str
     - Longer descriptive text
   * - ``category``
     - str
     - Classification or category
   * - ``content``
     - str
     - General content field

Template Variable System
------------------------

Field definitions support template variables that are automatically resolved:

Available Template Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   
   * - Variable
     - Description
     - Example Value
   * - ``{{current_year}}``
     - Current year
     - ``2024``
   * - ``{{current_date}}``
     - Current date (ISO format)
     - ``2024-03-15``
   * - ``{{current_datetime}}``
     - Current datetime (ISO format)
     - ``2024-03-15T14:30:00``
   * - ``{{current_timestamp}}``
     - Unix timestamp
     - ``1710512400``

**Example with Template Variables:**

.. code-block:: python

   register_field("processed_date", {
       "type": str,
       "description": "Date when this record was processed",
       "instructions": "Use the current date",
       "default": "{{current_date}}",  # Resolves to actual date
       "nullable": False
   })
   
   # When used, "{{current_date}}" becomes "2024-03-15"
   field_def = get_field_definition("processed_date")
   print(field_def["default"])  # "2024-03-15"

Field Definition Structure
--------------------------

Each field definition is a dictionary with the following structure:

.. code-block:: python

   {
       "type": str | int | float | list | dict | bool,  # Python type
       "description": "Human-readable field description",
       "instructions": "LLM extraction instructions", 
       "default": "Default value or template variable",
       "nullable": True | False,  # Whether field can be None
       "validation": {...}  # Optional validation rules
   }

**Field Definition Properties:**

- **type**: The Python type for this field (str, int, float, list, dict, bool)
- **description**: Human-readable description of what this field represents
- **instructions**: Specific instructions for the LLM on how to extract this field
- **default**: Default value to use when extraction fails or field is missing
- **nullable**: Whether the field can accept None/null values
- **validation**: Optional dictionary containing validation rules and constraints

Advanced Usage Patterns
------------------------

Loading Field Definitions from Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can load field definitions from JSON or YAML files:

.. code-block:: python

   from prompture.tools import load_field_definitions
   
   # Load from JSON file
   custom_fields = load_field_definitions("my_fields.json")
   add_field_definitions(custom_fields)
   
   # Load from YAML file  
   yaml_fields = load_field_definitions("fields.yaml")
   add_field_definitions(yaml_fields)

Thread Safety
~~~~~~~~~~~~~

The field registry is thread-safe and can be used safely in multi-threaded applications:

.. code-block:: python

   import threading
   
   def worker_thread():
       # Safe to call from multiple threads
       register_field("thread_field", {...})
       field_def = get_field_definition("name")
   
   threads = [threading.Thread(target=worker_thread) for _ in range(10)]
   for t in threads:
       t.start()

Integration with Core Functions
-------------------------------

The field definitions system integrates seamlessly with core extraction functions:

**With extract_with_model():**

.. code-block:: python

   from prompture import extract_with_model, field_from_registry
   
   class Person(BaseModel):
       name: str = field_from_registry("name")
       email: str = field_from_registry("email")
   
   result = extract_with_model(Person, text, "openai/gpt-4")

**With stepwise_extract_with_model():**

.. code-block:: python

   from prompture.core import stepwise_extract_with_model
   
   # Automatically uses global registry for enhanced defaults
   result = stepwise_extract_with_model(
       model_cls=Person,
       text="...",
       model_name="openai/gpt-4"
   )

Best Practices
--------------

1. **Use built-in fields** when possible before creating custom ones
2. **Provide clear instructions** that help LLMs extract the field correctly  
3. **Set appropriate defaults** that make sense when extraction fails
4. **Use template variables** for dynamic values like dates and timestamps
5. **Group related fields** logically when registering multiple definitions
6. **Document custom fields** with descriptive names and comprehensive instructions
7. **Test field definitions** with representative text samples to ensure accuracy