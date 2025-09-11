# packages/llm_to_json/__init__.py
try:
    from prompture import extract_and_jsonify as _extract_and_jsonify
except Exception:
    from prompture.core import extract_and_jsonify as _extract_and_jsonify

def from_llm_text(text: str, schema: dict, driver: dict | None = None):
    return _extract_and_jsonify(driver or {}, text, schema)
