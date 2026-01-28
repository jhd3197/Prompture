"""
Text Classification Example with Multiple Enum Fields

This example shows how to use Prompture for classifying text into both tone
and topic categories using enum fields. Enum fields ensure that LLM outputs
stay within predefined valid options, improving reliability and consistency.
"""

from typing import Optional

from pydantic import BaseModel

from prompture import extract_with_model, field_from_registry, get_field_definition, validate_enum_value


# Define the Pydantic model for text classification using enum fields
class TextClassification(BaseModel):
    tone: str = field_from_registry("tone")  # e.g. ["formal", "informal", "optimistic", "pessimistic"]
    topic: Optional[str] = field_from_registry("topic")  # General subject/topic of the text


# Example texts to classify
sample_texts = [
    """
    We are delighted to announce the grand opening of our new office space
    in downtown Miami. This expansion reflects our commitment to innovation
    and growth in the region. We look forward to welcoming our clients to
    this modern and vibrant workspace.
    """,
    """
    Honestly, I don’t think the new update did much to fix the app. It’s still
    laggy, crashes often, and the support team keeps giving canned responses.
    I’m getting really tired of this.
    """,
    """
    The company will host a quarterly town hall meeting next week. Employees
    are encouraged to submit questions in advance. The agenda includes a
    review of financial performance, upcoming projects, and a Q&A session.
    """,
]

# Display enum info for tone
print("=" * 70)
print("ENUM FIELD INFORMATION")
print("=" * 70)
tone_def = get_field_definition("tone")
if tone_def and "enum" in tone_def:
    print("Field: tone")
    print(f"Allowed values: {tone_def['enum']}")
    print(f"Description: {tone_def['description']}")
    print(f"Instructions: {tone_def['instructions']}")
print("=" * 70)
print()

# Analyze each text
for i, text in enumerate(sample_texts, 1):
    print(f"Classifying Text {i}...")
    print("-" * 70)

    # Extract classification from text
    classification = extract_with_model(TextClassification, text, "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b")

    # Print results
    print(f"Tone: {classification.model.tone}")
    print(f"Topic: {classification.model.topic or 'N/A'}")

    # Validate enum value for tone
    is_valid = validate_enum_value("tone", classification.model.tone)
    print(f"Valid tone value: {is_valid}")

    print("-" * 70)
    print()

# Manual validation demo
print("=" * 70)
print("ENUM VALIDATION EXAMPLES")
print("=" * 70)

test_values = ["formal", "casual", "optimistic", "angry", "PESSIMISTIC"]
for value in test_values:
    is_valid = validate_enum_value("tone", value)
    print(f"Value '{value}': {'✓ Valid' if is_valid else '✗ Invalid'}")

print("=" * 70)
