"""
Modern Field Definitions API Examples for Prompture

This example demonstrates the clean, modern API for field definitions:
- Using field_from_registry() with Pydantic models
- Registering custom fields with register_field()
- Template variable usage for dynamic defaults
- Loading external field definitions
- Automatic extraction without manual parameters

The new API reduces boilerplate and provides better integration with modern Python practices.
"""

from pydantic import BaseModel

from prompture import (
    field_from_registry,
    get_field_definition,
    get_field_names,
    get_required_fields,
    register_field,
)
from prompture.tools import (
    load_field_definitions,  # Still works for loading external files
    validate_field_definition,  # Still useful for validation
)

# =============================================================================
# BASIC FIELD REGISTRY USAGE
# =============================================================================


def demonstrate_basic_usage():
    """Show basic field_from_registry usage with built-in fields."""
    print("=" * 60)
    print("BASIC FIELD REGISTRY USAGE")
    print("=" * 60)

    # Show available built-in fields
    print(f"📋 Available built-in fields: {get_required_fields()}")
    print(f"📊 Total fields in registry: {len(get_field_names())}")

    # Define a simple model using built-in fields
    class BasicPerson(BaseModel):
        name: str = field_from_registry("name")
        age: int = field_from_registry("age")
        email: str = field_from_registry("email")

    print("✅ BasicPerson model created using built-in fields")
    print(f"   Fields: {list(BasicPerson.model_fields.keys())}")

    # Show field definition details
    name_field = get_field_definition("name")
    if name_field:
        print(f"📝 'name' field description: {name_field.get('description', 'N/A')}")


# =============================================================================
# CUSTOM FIELD REGISTRATION WITH TEMPLATES
# =============================================================================


def demonstrate_custom_fields():
    """Show register_field with templates and custom fields."""
    print("\n" + "=" * 60)
    print("CUSTOM FIELD REGISTRATION")
    print("=" * 60)

    # Register fields with template variables
    register_field(
        "education",
        {
            "type": "str",
            "description": "Education level and graduation details",
            "instructions": "Extract education level. Use {{current_year}} for recent graduates if year not specified.",
            "default": "{{current_date}} - Not specified",
            "nullable": True,
        },
    )

    register_field(
        "profile_created",
        {
            "type": "str",
            "description": "Profile creation timestamp",
            "instructions": "Use {{current_datetime}} if creation time not available",
            "default": "{{current_datetime}}",
            "nullable": False,
        },
    )

    register_field(
        "salary_range",
        {
            "type": "str",
            "description": "Salary or income range",
            "instructions": "Extract salary information, handle currency symbols and ranges like $50k-$75k",
            "default": "Not specified",
            "nullable": True,
        },
    )

    register_field(
        "skills",
        {
            "type": "list",
            "description": "List of professional skills or competencies",
            "instructions": "Extract skills as a comma-separated list, normalize technology names",
            "default": [],
            "nullable": True,
        },
    )

    print("✅ Registered custom fields with templates:")
    print("   - education (with {{current_year}} template)")
    print("   - profile_created (with {{current_datetime}} template)")
    print("   - salary_range (for financial data)")
    print("   - skills (list type for multiple values)")


# =============================================================================
# PYDANTIC MODEL INTEGRATION
# =============================================================================


def demonstrate_pydantic_integration():
    """Show clean model definitions with mixed built-in and custom fields."""
    print("\n" + "=" * 60)
    print("PYDANTIC MODEL INTEGRATION")
    print("=" * 60)

    # Create comprehensive model mixing built-in and custom fields
    class ComprehensiveProfile(BaseModel):
        # Built-in fields
        name: str = field_from_registry("name")
        age: int = field_from_registry("age")
        email: str = field_from_registry("email")

        # Custom fields
        education: str = field_from_registry("education")
        profile_created: str = field_from_registry("profile_created")
        salary_range: str = field_from_registry("salary_range")
        skills: list = field_from_registry("skills")

    print("✅ ComprehensiveProfile model created")
    print(f"   Combined fields: {list(ComprehensiveProfile.model_fields.keys())}")

    # Show example usage (commented as it requires API key)
    sample_text = """
    Sarah Johnson is a 32-year-old software architect at TechCorp.
    She can be reached at sarah.j@techcorp.com. Sarah graduated with
    a Master's in Computer Science from MIT in 2015. She specializes
    in Python, JavaScript, cloud architecture, and machine learning.
    Her current salary range is $120,000 - $150,000 annually.
    """

    print("\n📄 Sample text for extraction:")
    print(f'"{sample_text.strip()}"')

    print("\n💡 Example extraction usage (requires API key):")
    print("# result = stepwise_extract_with_model(")
    print("#     ComprehensiveProfile, ")
    print("#     sample_text, ")
    print("#     model_name='openai/gpt-4'")
    print("# )")
    print("# print(result.model_dump())")


# =============================================================================
# TEMPLATE VARIABLES DEMONSTRATION
# =============================================================================


def demonstrate_template_variables():
    """Show template variable capabilities and use cases."""
    print("\n" + "=" * 60)
    print("TEMPLATE VARIABLES DEMONSTRATION")
    print("=" * 60)

    print("🏷️  Available template variables:")
    print("   {{current_date}} - Current date (YYYY-MM-DD)")
    print("   {{current_datetime}} - Current datetime (ISO 8601)")
    print("   {{current_year}} - Current year (YYYY)")

    # Register more fields showing different template use cases
    template_examples = {
        "document_date": {
            "type": "str",
            "description": "Document creation or update date",
            "instructions": "Use {{current_date}} if date not specified in document",
            "default": "{{current_date}}",
            "nullable": False,
        },
        "processing_timestamp": {
            "type": "str",
            "description": "When this document was processed",
            "instructions": "Always use {{current_datetime}} for processing time",
            "default": "{{current_datetime}}",
            "nullable": False,
        },
        "graduation_year": {
            "type": "int",
            "description": "Year of graduation",
            "instructions": "Extract graduation year, use {{current_year}} for recent graduates if unspecified",
            "default": "{{current_year}}",
            "nullable": True,
        },
        "contract_expires": {
            "type": "str",
            "description": "Contract expiration date",
            "instructions": "Extract contract end date, assume {{current_date}} + 1 year if not specified",
            "default": "{{current_date}}",
            "nullable": True,
        },
    }

    print("\n📝 Template field examples:")
    for field_name, definition in template_examples.items():
        register_field(field_name, definition)
        template_var = definition["default"]
        print(f"   ✓ {field_name}: uses {template_var}")

    # Create a model using template fields
    class DocumentProfile(BaseModel):
        document_date: str = field_from_registry("document_date")
        processing_timestamp: str = field_from_registry("processing_timestamp")
        graduation_year: int = field_from_registry("graduation_year")
        contract_expires: str = field_from_registry("contract_expires")

    print("\n✅ DocumentProfile model with template fields created")
    print(f"   Fields: {list(DocumentProfile.model_fields.keys())}")


# =============================================================================
# EXTERNAL FILE LOADING
# =============================================================================


def demonstrate_external_loading():
    """Show loading field definitions from external files (still valid)."""
    print("\n" + "=" * 60)
    print("EXTERNAL FILE LOADING")
    print("=" * 60)

    try:
        # Load from YAML file (if exists)
        yaml_fields = load_field_definitions("examples/field_definitions.yaml")
        print(f"📁 Loaded {len(yaml_fields)} fields from YAML file")
        print(f"   Sample fields: {list(yaml_fields.keys())[:5]}...")

        # Validate some loaded fields
        valid_count = 0
        for field_name, definition in list(yaml_fields.items())[:3]:
            is_valid = validate_field_definition(definition)
            if is_valid:
                valid_count += 1
            print(f"   ✓ '{field_name}' is {'valid' if is_valid else 'invalid'}")

        print(f"📊 Validation: {valid_count}/3 sample fields are valid")

    except FileNotFoundError:
        print("⚠️  YAML file not found - skipping YAML loading demo")
    except Exception as e:
        print(f"⚠️  Error loading YAML: {e}")

    try:
        # Load from JSON file (if exists)
        json_fields = load_field_definitions("examples/field_definitions.json")
        print(f"📁 Loaded {len(json_fields)} fields from JSON file")
        print(f"   Sample fields: {list(json_fields.keys())[:5]}...")

    except FileNotFoundError:
        print("⚠️  JSON file not found - skipping JSON loading demo")
    except Exception as e:
        print(f"⚠️  Error loading JSON: {e}")

    print("\n💡 External loading tips:")
    print("   - Use YAML/JSON for large field definition sets")
    print("   - Validate definitions after loading")
    print("   - Register loaded fields individually with register_field()")


# =============================================================================
# PRACTICAL EXTRACTION SCENARIOS
# =============================================================================


def demonstrate_practical_scenarios():
    """Show real-world usage patterns and scenarios."""
    print("\n" + "=" * 60)
    print("PRACTICAL EXTRACTION SCENARIOS")
    print("=" * 60)

    # Scenario 1: Resume/CV extraction
    print("📋 Scenario 1: Resume/CV Processing")

    class ResumeProfile(BaseModel):
        name: str = field_from_registry("name")
        email: str = field_from_registry("email")
        education: str = field_from_registry("education")
        skills: list = field_from_registry("skills")
        salary_range: str = field_from_registry("salary_range")

    resume_text = """
    John Smith, Software Engineer
    Contact: john.smith@email.com
    Education: BS Computer Science, University of California (2019)
    Skills: Python, React, AWS, Docker, PostgreSQL
    Expected salary: $80,000 - $100,000
    """

    print("   ✓ ResumeProfile model ready")
    print(f"   📄 Sample resume text prepared ({len(resume_text)} chars)")

    # Scenario 2: Product information extraction
    print("\n🛍️  Scenario 2: Product Information")

    # Register product-specific fields
    register_field(
        "product_name",
        {
            "type": "str",
            "description": "Product name or title",
            "instructions": "Extract the main product name, excluding brand prefixes",
            "default": "Unknown Product",
            "nullable": False,
        },
    )

    register_field(
        "price",
        {
            "type": "str",
            "description": "Product price with currency",
            "instructions": "Extract price including currency symbol, handle ranges like $10-$15",
            "default": "Price not listed",
            "nullable": True,
        },
    )

    register_field(
        "category",
        {
            "type": "str",
            "description": "Product category or department",
            "instructions": "Classify product into appropriate category",
            "default": "Miscellaneous",
            "nullable": True,
        },
    )

    class ProductInfo(BaseModel):
        product_name: str = field_from_registry("product_name")
        price: str = field_from_registry("price")
        category: str = field_from_registry("category")

    print("   ✓ ProductInfo model with custom product fields")
    print("   ✓ Registered: product_name, price, category")

    # Scenario 3: Event information
    print("\n📅 Scenario 3: Event Processing")

    register_field(
        "event_name",
        {
            "type": "str",
            "description": "Name or title of the event",
            "instructions": "Extract the main event title",
            "default": "Untitled Event",
            "nullable": False,
        },
    )

    register_field(
        "event_date",
        {
            "type": "str",
            "description": "Event date and time",
            "instructions": "Extract event date, use {{current_date}} if not specified",
            "default": "{{current_date}}",
            "nullable": False,
        },
    )

    class EventInfo(BaseModel):
        event_name: str = field_from_registry("event_name")
        event_date: str = field_from_registry("event_date")
        processing_timestamp: str = field_from_registry("processing_timestamp")

    print("   ✓ EventInfo model with date templates")
    print("   ✓ Uses {{current_date}} and {{current_datetime}} templates")


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Run all examples to demonstrate the modern field definitions API."""
    print("🚀 Prompture Modern Field Definitions API Examples")
    print("This example showcases the clean, modern API without deprecated functions.\n")

    # Core API demonstrations
    demonstrate_basic_usage()
    demonstrate_custom_fields()
    demonstrate_pydantic_integration()
    demonstrate_template_variables()
    demonstrate_external_loading()
    demonstrate_practical_scenarios()

    print("\n" + "=" * 60)
    print("MODERN API BENEFITS")
    print("=" * 60)
    print("✅ CLEAN & SIMPLE:")
    print("  • Less boilerplate code")
    print("  • Better Pydantic integration")
    print("  • Template variable support")
    print("  • Automatic field resolution")
    print("  • Type safety with modern Python")

    print("\n🎯 KEY FUNCTIONS:")
    print("  • field_from_registry() - Clean Pydantic field creation")
    print("  • register_field() - Add custom fields with templates")
    print("  • stepwise_extract_with_model() - Automatic extraction")
    print("  • get_field_definition() - Inspect field properties")
    print("  • load_field_definitions() - Import from YAML/JSON")

    print("\n🏷️  TEMPLATE VARIABLES:")
    print("  • {{current_date}} - Dynamic date defaults")
    print("  • {{current_datetime}} - Timestamp defaults")
    print("  • {{current_year}} - Year-based defaults")

    print("\n📚 USAGE PATTERNS:")
    print("  1. Use built-in fields for common data (name, age, email)")
    print("  2. Register custom fields for domain-specific needs")
    print("  3. Leverage templates for dynamic defaults")
    print("  4. Combine in Pydantic models for type safety")
    print("  5. Use stepwise_extract_with_model() for extraction")


if __name__ == "__main__":
    main()
