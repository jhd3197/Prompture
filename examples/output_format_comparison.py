"""Example script demonstrating TOON output format."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompture import extract_and_jsonify

SAMPLE_TEXT = """
Alice Johnson is a 30-year-old data scientist from Denver who specializes in
machine learning. She has 8 years of experience, currently works at QuantumSight,
and previously led automation projects that reduced costs by 18%. Alice mentors
a team of five engineers and recently gave a talk on applying TOON serialization
to reduce LLM output tokens.

Outside of work she enjoys trail running and volunteers at a local coding bootcamp.
She is available for freelance consulting. She is reachable at alice@example.com and
lists Python, SQL, and prompt engineering as core skills.
"""

EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "location": {"type": "string"},
        "experience_years": {"type": "integer"},
        "current_company": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "recent_achievement": {"type": "string"},
        "availability": {"type": "string"},
        "email": {"type": "string"},
    },
    "required": ["name", "age", "skills"],
}

MODEL_TO_TEST = os.getenv(
    "PROMPTURE_TEST_MODEL",
    "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b"
)


def demonstrate_toon_output():
    """Demonstrate TOON output format."""
    print("TOON OUTPUT FORMAT DEMONSTRATION")
    print("=" * 50)
    print(f"Model: {MODEL_TO_TEST}")
    print()
    
    try:
        print("Running extraction with TOON output format...")
        result = extract_and_jsonify(
            text=SAMPLE_TEXT,
            json_schema=EXTRACTION_SCHEMA,
            model_name=MODEL_TO_TEST,
            output_format="toon"
        )
        
        print("✓ Extraction successful!")
        print()
        
        # Show the extracted data
        print("Extracted Data (JSON object):")
        print("-" * 30)
        import json
        print(json.dumps(result["json_object"], indent=2))
        print()
        
        # Show the TOON format output
        if "toon_string" in result:
            print("TOON Format Output:")
            print("-" * 30)
            print(result["toon_string"])
            print()
        
        # Show usage information
        usage = result.get("usage", {})
        if usage:
            print("Token Usage:")
            print("-" * 30)
            print(f"Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"Total tokens: {usage.get('total_tokens', 'N/A')}")
            if usage.get('cost'):
                print(f"Cost: ${usage.get('cost', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def main() -> None:
    """Run the TOON output demonstration."""
    success = demonstrate_toon_output()
    
    if success:
        print()
        print("🎉 TOON output format demonstration completed!")
        print()
        print("📝 About TOON format:")
        print("   • TOON (Token-Oriented Object Notation) is a compact serialization format")
        print("   • Designed for efficient token usage when LLMs need to output structured data")
        print("   • Particularly effective for tabular data and uniform object arrays")
        print("   • Can reduce token usage by 45-60% compared to standard JSON")
    else:
        print()
        print("❌ Demonstration failed. Please check your model configuration.")


if __name__ == "__main__":
    main()
