# packages/llm_to_toon/llm_to_toon/__init__.py
try:
    from prompture import extract_and_jsonify as _extract_and_jsonify
    from prompture.drivers import OllamaDriver
except Exception:
    from prompture.core import extract_and_jsonify as _extract_and_jsonify
    from prompture.drivers import OllamaDriver


def from_llm_text(text: str, schema: dict, driver: dict | None = None):
    """Extract TOON from text using an LLM and Prompture helpers.

    Args:
        text: Text to extract structured data from.
        schema: JSON schema describing the expected structure.
        driver: Optional LLM driver (defaults to local Ollama if not provided).

    Returns:
        TOON string encoded from the extracted data.
    """
    if driver is None:
        driver = OllamaDriver(endpoint="http://localhost:11434", model="gemma:latest")
    result = _extract_and_jsonify(driver, text, schema, output_format="toon")
    return result["toon_string"]
