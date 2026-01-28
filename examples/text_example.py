"""Example of using render_output for text and HTML generation."""

import os
import sys

from dotenv import load_dotenv

# Add parent directory to path to import prompture if running from examples folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompture import get_driver_for_model, render_output


def main():
    load_dotenv()

    # Use default model or override with env var
    model_name = os.getenv("MODEL", "openai/gpt-3.5-turbo")
    print(f"Using model: {model_name}")

    try:
        driver = get_driver_for_model(model_name)
    except Exception as e:
        print(f"Error initializing driver: {e}")
        print("Please ensure you have set up your .env file with API keys.")
        return

    print("\n--- Text Example ---")
    prompt = "Write a haiku about coding."
    print(f"Prompt: {prompt}")

    result = render_output(driver=driver, content_prompt=prompt, output_format="text", model_name=model_name)

    print("\nResult:")
    print(result["text"])
    print(f"\nUsage: {result['usage']['total_tokens']} tokens")

    print("\n--- HTML Example ---")
    prompt = "Create a simple button that alerts 'Hello World' when clicked."
    print(f"Prompt: {prompt}")

    result = render_output(driver=driver, content_prompt=prompt, output_format="html", model_name=model_name)

    print("\nResult:")
    print(result["text"])
    print(f"\nUsage: {result['usage']['total_tokens']} tokens")


if __name__ == "__main__":
    main()
