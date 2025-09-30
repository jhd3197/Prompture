"""
Text Analysis Example

This example demonstrates how to use Prompture for analyzing general text and extracting
boolean (true/false) values. It shows how to register boolean fields for sentiment analysis,
content type detection, and writing style assessment.
"""

from pydantic import BaseModel
from typing import Optional
from prompture import register_field, field_from_registry, extract_with_model

# Register boolean fields for text analysis
register_field("is_positive_sentiment", {
    "type": "bool",
    "description": "Whether the text expresses positive sentiment",
    "instructions": "Analyze the overall tone and determine if the sentiment is predominantly positive. Look for positive words, optimistic language, and favorable opinions.",
    "default": False,
    "nullable": False
})

register_field("contains_facts", {
    "type": "bool",
    "description": "Whether the text contains factual information or data",
    "instructions": "Determine if the text includes verifiable facts, statistics, dates, or objective information rather than just opinions.",
    "default": False,
    "nullable": False
})

register_field("is_formal_tone", {
    "type": "bool",
    "description": "Whether the text uses formal language and professional tone",
    "instructions": "Check if the writing style is formal, professional, and uses proper grammar. Informal language, slang, or casual expressions indicate false.",
    "default": False,
    "nullable": False
})

register_field("has_call_to_action", {
    "type": "bool",
    "description": "Whether the text includes a call to action",
    "instructions": "Look for explicit requests or suggestions for the reader to take action, such as 'buy now', 'sign up', 'learn more', or similar directives.",
    "default": False,
    "nullable": False
})

register_field("is_persuasive", {
    "type": "bool",
    "description": "Whether the text attempts to persuade or convince the reader",
    "instructions": "Determine if the text uses persuasive techniques, arguments, or tries to influence the reader's opinion or behavior.",
    "default": False,
    "nullable": False
})

# Define the Pydantic model for text analysis
class TextAnalysis(BaseModel):
    is_positive_sentiment: bool = field_from_registry("is_positive_sentiment")
    contains_facts: bool = field_from_registry("contains_facts")
    is_formal_tone: bool = field_from_registry("is_formal_tone")
    has_call_to_action: bool = field_from_registry("has_call_to_action")
    is_persuasive: bool = field_from_registry("is_persuasive")

# Sample text - a product review
sample_text = """
I recently purchased the TechPro Wireless Headphones and I'm absolutely thrilled with my purchase!
The sound quality is exceptional, delivering crisp highs and deep bass that brings my music to life.

The battery lasts for an impressive 30 hours on a single charge, and the quick-charge feature gives
you 5 hours of playback in just 10 minutes. The noise cancellation technology is top-notch, blocking
out up to 95% of ambient noise according to the manufacturer's specifications.

At $149.99, these headphones offer incredible value for money. If you're in the market for premium
wireless headphones without breaking the bank, I highly recommend giving these a try. You won't be
disappointed! Check them out on the TechPro website today.
"""

# Extract boolean analysis from the text
analysis = extract_with_model(
    TextAnalysis,
    sample_text,
    "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b"
)

# Print the analysis results
print("=" * 60)
print("TEXT ANALYSIS RESULTS")
print("=" * 60)
print(f"Positive Sentiment:     {analysis.model.is_positive_sentiment}")
print(f"Contains Facts:         {analysis.model.contains_facts}")
print(f"Formal Tone:            {analysis.model.is_formal_tone}")
print(f"Has Call to Action:     {analysis.model.has_call_to_action}")
print(f"Is Persuasive:          {analysis.model.is_persuasive}")
print("=" * 60)