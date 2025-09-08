import json
from typing import Any, Dict

try:
    import jsonschema
except Exception:
    jsonschema = None


def validate_against_schema(instance_json: str, schema: Dict[str,Any]) -> Dict[str,Any]:
    """Valida el JSON (string) contra un JSON Schema.
    Devuelve dict con ok: bool y detalles.
    """
    try:
        data = json.loads(instance_json)
    except Exception as e:
        return {"ok": False, "error": f"invalid_json: {e}"}

    if jsonschema is None:
        # fallback ligero: solo comprobar campos requeridos
        required = schema.get("required", [])
        missing = [k for k in required if k not in data]
        if missing:
            return {"ok": False, "error": f"missing_required: {missing}", "data": data}
        return {"ok": True, "data": data}

    try:
        jsonschema.validate(instance=data, schema=schema)
        return {"ok": True, "data": data}
    except jsonschema.ValidationError as e:
        return {"ok": False, "error": str(e), "data": data}