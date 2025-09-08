"""prompture - API package to convert LLM outputs into JSON + test harness."""

from dotenv import load_dotenv
from .core import ask_for_json, extract_and_jsonify, Driver, clean_json_text, clean_json_text_with_ai
from .runner import run_suite_from_spec
from .validator import validate_against_schema

# Load environment variables from .env file
load_dotenv()

__all__ = ["ask_for_json", "extract_and_jsonify", "run_suite_from_spec", "validate_against_schema", "Driver", "clean_json_text", "clean_json_text_with_ai"]
__version__ = "0.0.1"