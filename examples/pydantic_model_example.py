"""
pydantic_model_example.py

Example: Using extract_with_model() and stepwise_extract_with_model() with Pydantic models.

This script demonstrates:
1. Defining a Pydantic model with Field attributes for rich schema information
2. Using extract_with_model() with Ollama driver
3. Using stepwise_extract_with_model() with debug output to show field-by-field extraction
4. Default value handling when fields cannot be extracted from incomplete text

The example shows how Pydantic models provide a type-safe way to extract
structured information from text, with field-level descriptions that help
guide the LLM's extraction process.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from prompture import extract_with_model, stepwise_extract_with_model


# Define an enhanced Person model to demonstrate new features
class Person(BaseModel):
    name: str = Field(..., description="The full name of the person.")
    age: int = Field(..., description="The age of the person, expressed as a number.", gt=0, lt=150)
    birth_date: datetime = Field(..., description="The person's date of birth in ISO format (YYYY-MM-DD).")
    profession: str = Field(..., description="Their main job or role.")
    is_employed: bool = Field(..., description="Whether the person is currently employed. True or False.")
    salary: Optional[float] = Field(None, description="Annual salary in USD, if available. Numbers only e.g. 75000.50")


# Define a Person model with default values to demonstrate default handling
class PersonWithDefaults(BaseModel):
    name: str = Field(..., description="The full name of the person.")
    age: int = Field(0, description="The age of the person in years. Default to 0 if not found.", ge=0, lt=150)
    profession: str = Field("unknown", description="Their main job or role. Default to 'unknown' if not mentioned.")
    is_employed: bool = Field(
        False, description="Whether the person is currently employed. Default to False if unclear."
    )
    salary: Optional[float] = Field(None, description="Annual salary in USD, if available. Numbers only e.g. 75000.50")


# Example text to extract information from
text = """
Maria Garcia was born on April 15, 1991. She is 32 years old and works as a senior
software developer at Tech Corp. She is currently employed and earns $120,000 per year.
"""

print("\n=== EXAMPLE 1: Using extract_with_model ===")
try:
    # Use the Ollama driver explicitly
    result = extract_with_model(
        model_cls=Person,
        text=text,
        model_name="ollama/gpt-oss:20b",
        instruction_template="Extract biographical information into structured data:",
    )
    print("\nExtracted Person object:")
    print(result["model"])

except Exception as e:
    print(f"Error with Ollama driver: {e!s}")


print("\n=== EXAMPLE 2: Using stepwise_extract_with_model with debug mode ===")
try:
    # Extract each field individually with debug output enabled
    print("\nExtracting fields with debug mode:")

    result = stepwise_extract_with_model(
        model_cls=Person,
        text=text,
        model_name="ollama/gpt-oss:20b",
    )
    print("\nExtracted and validated data:")
    print(result["model"])

except Exception as e:
    print(f"Error with stepwise extraction: {e!s}")

print("\n=== EXAMPLE 3: Demonstrating validation features ===")
try:
    # Test with invalid data to show validation
    invalid_text = """
    Bob Smith was born yesterday. He is 15 years old and
    works as a time traveler. His salary is 'one million' dollars.
    """

    print("\nAttempting extraction with invalid data:")
    result = stepwise_extract_with_model(
        model_cls=Person,
        text=invalid_text,
        model_name="ollama/gpt-oss:20b",
    )
    print(result["error"])

except Exception as e:
    print(f"\nValidation caught the following issues:\n{e!s}")


print("\n=== EXAMPLE 4: Demonstrating type handling ===")
try:
    # Test various data types
    mixed_text = """
    Alice Johnson was born on 2000-01-01. She is 23.5 years old and
    works as a data scientist. She is employed and makes approximately 95.5k per year.
    """

    print("\nExtracting data with various types:")
    result = stepwise_extract_with_model(
        model_cls=Person,
        text=mixed_text,
        model_name="ollama/gpt-oss:20b",
    )

    print("\nExtracted and validated data:")
    print(result["model"])

except Exception as e:
    print(f"\nError handling types: {e!s}")


print("\n=== EXAMPLE 5: Demonstrating default value handling ===")
try:
    # Test with incomplete/invalid text where some fields cannot be extracted
    incomplete_text = """
    John Doe is mentioned here but with very limited information available.
    The document doesn't specify much else about him.
    """

    print("\nExtracting from incomplete text (should use defaults for missing fields):")
    result = stepwise_extract_with_model(
        model_cls=PersonWithDefaults,
        text=incomplete_text,
        model_name="ollama/gpt-oss:20b",
    )

    print("\nExtracted data with defaults applied:")
    print(f"  Name: {result['model'].name}")
    print(f"  Age: {result['model'].age} (default used)")
    print(f"  Profession: {result['model'].profession} (default used)")
    print(f"  Is Employed: {result['model'].is_employed} (default used)")
    print(f"  Salary: {result['model'].salary} (default used)")
    print("\nFull model:")
    print(result["model"])

except Exception as e:
    print(f"\nError with default handling: {e!s}")
