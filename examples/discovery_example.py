"""
Example script demonstrating how to auto-detect available models using prompture.

This script will:
1. Load environment variables (if any)
2. Call get_available_models()
3. Print the list of detected models
"""

import os
import sys

# Add parent directory to path so we can import prompture if running from source
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompture import get_available_models


def main():
    print("🔍 Auto-detecting available models...")

    try:
        models = get_available_models()

        if not models:
            print("❌ No models detected.")
            print("Make sure you have set your API keys in .env or environment variables.")
            print("Or ensure Ollama is running if you expect local models.")
            return

        print(f"✅ Found {len(models)} models:")
        print("-" * 40)

        # Group by provider for nicer output
        by_provider = {}
        for model in models:
            provider = model.split("/")[0]
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(model)

        for provider, provider_models in sorted(by_provider.items()):
            print(f"\n[{provider.upper()}]")
            for m in sorted(provider_models):
                print(f"  - {m}")

        print("-" * 40)

    except Exception as e:
        print(f"❌ Error during discovery: {e}")


if __name__ == "__main__":
    main()
