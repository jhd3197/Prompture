"""prompture - API package to convert LLM outputs into JSON + test harness."""

from dotenv import load_dotenv
from .core import (
    ask_for_json,
    extract_and_jsonify,
    manual_extract_and_jsonify,
    Driver,
    clean_json_text,
    clean_toon_text,
    clean_json_text_with_ai,
    extract_with_model,
    stepwise_extract_with_model,
)
from .field_definitions import (
    FIELD_DEFINITIONS, get_field_definition, get_required_fields, get_field_names,
    field_from_registry, register_field, add_field_definition, add_field_definitions,
    get_registry_snapshot, clear_registry, reset_registry, validate_enum_value,
    normalize_enum_value
)
from .runner import run_suite_from_spec
from .validator import validate_against_schema

# Load environment variables from .env file
load_dotenv()

# runtime package version (from installed metadata)
try:
    # Python 3.8+
    from importlib.metadata import version as _get_version
except Exception:
    # older python using importlib-metadata backport (if you include it)
    from importlib_metadata import version as _get_version

try:
    __version__ = _get_version("prompture")
except Exception:
    # fallback during local editable development
    __version__ = "0.0.0"

__all__ = [
    "ask_for_json",
    "extract_and_jsonify",
    "manual_extract_and_jsonify",
    "run_suite_from_spec",
    "validate_against_schema",
    "Driver",
    "clean_json_text",
    "clean_toon_text",
    "clean_json_text_with_ai",
    "extract_with_model",
    "stepwise_extract_with_model",
    # Field Definitions
    "FIELD_DEFINITIONS",
    "get_field_definition",
    "get_required_fields",
    "get_field_names",
    # New Field Registry API
    "field_from_registry",
    "register_field",
    "add_field_definition",
    "add_field_definitions",
    "get_registry_snapshot",
    "clear_registry",
    "reset_registry",
    # Enum Field Support
    "validate_enum_value",
    "normalize_enum_value",
]
