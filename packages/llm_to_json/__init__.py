# packages/llm_to_json/__init__.py
try:
    from prompture import extract_and_jsonify as _extract_and_jsonify
    from prompture.drivers import OllamaDriver
except Exception:
    from prompture.core import extract_and_jsonify as _extract_and_jsonify
    from prompture.drivers import OllamaDriver

def from_llm_text(text: str, schema: dict, driver: dict | None = None):
    """Extract JSON from text using LLM.
    
    Args:
        text: Text to extract JSON from
        schema: JSON schema to validate against
        driver: Optional LLM driver (defaults to local Ollama if not provided)
    
    Returns:
        Extracted and validated JSON object
    """
    if driver is None:
        driver = OllamaDriver(endpoint="http://localhost:11434", model="gemma:latest")
    result = _extract_and_jsonify(driver, text, schema)
    return result["json_object"]
