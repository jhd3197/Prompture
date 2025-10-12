"""
Text Analysis Example with Enum Field Support

This example demonstrates how to use Prompture for analyzing general text and extracting
sentiment using enum fields. It shows how enum fields restrict LLM output to specific
predefined values (positive, negative, neutral) for sentiment analysis.
"""

from pydantic import BaseModel
from typing import Optional, Literal
from prompture import field_from_registry, extract_with_model, get_field_definition, validate_enum_value

# Define the Pydantic model for text analysis using the sentiment enum field
class TextAnalysis(BaseModel):
    sentiment: str = field_from_registry("sentiment")
    topic: Optional[str] = field_from_registry("topic")

# Sample texts to analyze
sample_texts = [
    """
    I recently purchased the TechPro Wireless Headphones and I'm absolutely thrilled with my purchase!
    The sound quality is exceptional, delivering crisp highs and deep bass that brings my music to life.
    
    The battery lasts for an impressive 30 hours on a single charge, and the quick-charge feature gives
    you 5 hours of playback in just 10 minutes. The noise cancellation technology is top-notch, blocking
    out up to 95% of ambient noise according to the manufacturer's specifications.
    
    At $149.99, these headphones offer incredible value for money. If you're in the market for premium
    wireless headphones without breaking the bank, I highly recommend giving these a try. You won't be
    disappointed! Check them out on the TechPro website today.
    """,
    """
    I had a terrible experience with customer service today. I waited on hold for over an hour
    only to be transferred three times to different departments. Nobody seemed to know how to help me,
    and my issue remains unresolved. Very frustrating and disappointing.
    """,
    """
    The weather forecast for tomorrow shows partly cloudy skies with temperatures ranging from
    68 to 75 degrees Fahrenheit. There is a 20% chance of precipitation in the afternoon. Wind
    speeds will be moderate at 10-15 mph from the northwest.
    """
]

# Display enum information
print("=" * 70)
print("ENUM FIELD INFORMATION")
print("=" * 70)
sentiment_def = get_field_definition("sentiment")
if sentiment_def and 'enum' in sentiment_def:
    print(f"Field: sentiment")
    print(f"Allowed values: {sentiment_def['enum']}")
    print(f"Description: {sentiment_def['description']}")
    print(f"Instructions: {sentiment_def['instructions']}")
print("=" * 70)
print()

# Analyze each text
for i, text in enumerate(sample_texts, 1):
    print(f"Analyzing Text {i}...")
    print("-" * 70)
    
    # Extract sentiment analysis from the text
    analysis = extract_with_model(
        TextAnalysis,
        text,
        "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b"
    )
    
    # Print the analysis results
    print(f"Sentiment: {analysis.model.sentiment}")
    print(f"Topic: {analysis.model.topic or 'N/A'}")
    
    # Validate the enum value
    is_valid = validate_enum_value("sentiment", analysis.model.sentiment)
    print(f"Valid sentiment value: {is_valid}")
    
    print("-" * 70)
    print()

# Demonstrate manual enum validation
print("=" * 70)
print("ENUM VALIDATION EXAMPLES")
print("=" * 70)

test_values = ["positive", "negative", "neutral", "happy", "POSITIVE"]
for value in test_values:
    is_valid = validate_enum_value("sentiment", value)
    print(f"Value '{value}': {'✓ Valid' if is_valid else '✗ Invalid'}")

print("=" * 70)