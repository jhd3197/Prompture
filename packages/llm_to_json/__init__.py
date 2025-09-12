# packages/llm_to_json/__init__.py
try:
    from prompture import extract_and_jsonify as _extract_and_jsonify
    from prompture.drivers.mock_driver import MockDriver
except Exception:
    from prompture.core import extract_and_jsonify as _extract_and_jsonify
    from prompture.drivers.mock_driver import MockDriver

def from_llm_text(text: str, schema: dict, driver: dict | None = None):
    if driver is None:
        driver = MockDriver()
    result = _extract_and_jsonify(driver, text, schema)
    return result["json_object"]
