"""Test suite runner for executing JSON validation tests across multiple models."""
from typing import Dict, Any, List

from .core import ask_for_json, Driver
from prompture.validator import validate_against_schema


def run_suite_from_spec(spec: Dict[str, Any], drivers: Dict[str, Driver]) -> Dict[str, Any]:
    """Run a test suite specified by a spec dictionary across multiple models.
    
    Args:
        spec: A dictionary containing the test suite specification with the structure:
            {
                "meta": {"project": str, ...},
                "models": [{"id": str, "driver": str, "options": dict}, ...],
                "tests": [{
                    "id": str,
                    "prompt_template": str,
                    "inputs": [{"text": str, ...}, ...],
                    "schema": dict
                }, ...]
            }
        drivers: A dictionary mapping driver names to driver instances
        
    Returns:
        A dictionary containing test results with the structure:
            {
                "meta": dict,  # Original meta info
                "results": [{
                    "test_id": str,
                    "model_id": str,
                    "input": dict,
                    "prompt": str,
                    "response": dict,
                    "validation": {
                        "ok": bool,
                        "error": str | None,
                        "data": dict
                    },
                    "usage": dict
                }, ...]
            }
    """
    results = []
    
    for test in spec["tests"]:
        for model in spec["models"]:
            driver = drivers.get(model["driver"])
            if not driver:
                continue
                
            # Run test for each input
            for input_data in test["inputs"]:
                # Format prompt template with input data
                try:
                    prompt = test["prompt_template"].format(**input_data)
                except KeyError as e:
                    results.append({
                        "test_id": test["id"],
                        "model_id": model["id"],
                        "input": input_data,
                        "prompt": test["prompt_template"],
                        "error": f"Template formatting error: missing key {e}",
                        "validation": {"ok": False, "error": "Prompt formatting failed", "data": None},
                        "usage": {"total_tokens": 0, "cost": 0}
                    })
                    continue
                    
                # Get JSON response from model
                try:
                    response = ask_for_json(
                        driver=driver,
                        content_prompt=prompt,
                        json_schema=test["schema"],
                        options=model.get("options", {})
                    )
                    
                    # Validate response against schema
                    validation = validate_against_schema(
                        response["json_string"],
                        test["schema"]
                    )
                    
                    results.append({
                        "test_id": test["id"],
                        "model_id": model["id"],
                        "input": input_data,
                        "prompt": prompt,
                        "response": response["json_object"],
                        "validation": validation,
                        "usage": response["usage"]
                    })
                    
                except Exception as e:
                    results.append({
                        "test_id": test["id"],
                        "model_id": model["id"],
                        "input": input_data,
                        "prompt": prompt,
                        "error": str(e),
                        "validation": {"ok": False, "error": "Model response error", "data": None},
                        "usage": {"total_tokens": 0, "cost": 0}
                    })
    
    return {
        "meta": spec.get("meta", {}),
        "results": results
    }