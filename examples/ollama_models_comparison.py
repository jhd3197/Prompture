"""
Example script to compare multiple Ollama models on a complex parsing task.

This script demonstrates different Ollama models by extracting structured information
from a smartphone description using a complex JSON schema. It uses the
`extract_and_jsonify` function to explicitly provide the Ollama driver,
ignoring environment defaults.
"""

from prompture import extract_and_jsonify

# Define the complex text for parsing - 4-paragraph smartphone description
COMPLEX_TEXT = """
Product Name: GalaxyFold Ultra Pro Max

The GalaxyFold Ultra Pro Max is Samsung's latest flagship smartphone, priced at $1299.99.
It features a sleek titanium frame and comes in multiple color variants including Space Black,
Midnight Blue, and Titanium Gray. Available storage options are 256GB, 512GB, and 1TB.
This premium device combines cutting-edge technology with luxurious design.

The design of the GalaxyFold Ultra Pro Max is innovative, featuring a 6.9-inch foldable AMOLED
display with 1440 x 3120 resolution. The phone weighs 280 grams and has a thickness of 6.9mm
when unfolded. The hinge mechanism allows seamless folding into a compact 4.6-inch screen
for pocket convenience. The device includes IP68 water and dust resistance rating.

Photography is powered by a quad-camera system with 50MP main sensor, 48MP ultra-wide,
12MP periscope telephoto lenses, and a 10MP front-facing camera. The cameras support
8K video recording up to 60fps and include professional photography modes like HDR+,
night mode, and computational photography. Additional features include optical image
stabilization and laser autofocus for superior image quality.

The device comes with a comprehensive 3-year warranty covering hardware defects and
accidental damage. Customers can add extended coverage options for up to 5 years total.
The GalaxyFold Ultra Pro Max is a new device released this year and includes premium
packaging with accessories like wireless earbuds and a fast charger. Additional perks
include 1-year subscription to premium services.
"""

# Define the complex JSON schema with nested objects, arrays, and conditional logic
COMPLEX_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "brand": {"type": "string"},
        "price": {"type": "number"},
        "variants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"color": {"type": "string"}, "storage": {"type": "string"}},
                "required": ["color", "storage"],
            },
        },
        "design": {
            "type": "object",
            "properties": {
                "screen_size": {"type": "number"},
                "weight": {"type": "number"},
                "thickness_mm": {"type": "number"},
                "unfolded_screen_inches": {"type": "number"},
            },
            "required": ["screen_size", "weight"],
        },
        "cameras": {
            "type": "array",
            "items": {"type": "object", "properties": {"type": {"type": "string"}, "megapixels": {"type": "number"}}},
        },
        "warranty_years": {"type": "integer", "minimum": 1, "maximum": 5},
        "is_new": {"type": "boolean"},
    },
    "required": ["name", "price"],
}

# List of Ollama models to test
MODELS_TO_TEST = [
    "ollama/gpt-oss:20b",
    "ollama/llama3.1:8b",
    "ollama/gemma3:latest",
    "ollama/qwen2.5:1.5b",
    "ollama/qwen2.5:3b",
    "ollama/mistral:latest",
]


def compare_ollama_models():
    """
    Demonstrate comparing multiple Ollama models on extracting smartphone specs.

    Returns:
        dict: A mapping from model names to their results.
              Each entry contains:
                - success (bool)
                - json_object, usage, json_string (on success)
                - error (str) on failure
    """
    results = {}
    failed_models = []

    for model in MODELS_TO_TEST:
        print(f"Testing model: {model}")
        try:
            result = extract_and_jsonify(text=COMPLEX_TEXT, json_schema=COMPLEX_SCHEMA, model_name=model)
            results[model] = {
                "success": True,
                "json_object": result["json_object"],
                "usage": result["usage"],
                "json_string": result["json_string"],
            }
            print(f"  Success: Extracted {len(result['json_object'])} fields")
        except Exception as e:
            print(f"  Failed: {e!s}")
            results[model] = {"success": False, "error": str(e)}
            failed_models.append(model)

    return results


def print_comparison_table(results_dict):
    """
    Print a detailed comparison table of model performance.
    """
    print("\n" + "=" * 160)
    print("OLLAMA MODEL COMPARISON REPORT")
    print("=" * 160)
    row_format = "{:<15} {:<7} {:<8} {:<11} {:<6} {:<7} {:<11} {:<15} {:<8} {:<9} {:<13} {:<11} {:<8} {:<20}"
    headers = [
        "Model",
        "Success",
        "Prompt",
        "Completion",
        "Total",
        "Fields",
        "Validation",
        "Name",
        "Price",
        "Variants",
        "Screen Size",
        "Warranty",
        "Is New",
        "Error",
    ]
    print(row_format.format(*headers))
    print("-" * 130)

    successful_models = []
    failed_models = []

    for model, result in results_dict.items():
        if result["success"]:
            success = "True"
            json_obj = result["json_object"]
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            field_count = len(json_obj)
            # Required fields check
            has_required = (
                "name" in json_obj
                and json_obj["name"] is not None
                and "price" in json_obj
                and json_obj["price"] is not None
                and "variants" in json_obj
                and isinstance(json_obj["variants"], list)
                and "design" in json_obj
                and isinstance(json_obj["design"], dict)
                and "screen_size" in json_obj["design"]
                and json_obj["design"]["screen_size"] is not None
                and "warranty_years" in json_obj
                and json_obj["warranty_years"] is not None
                and "is_new" in json_obj
                and json_obj["is_new"] is not None
            )
            validation = "✓" if has_required else "✗"
            name = str(json_obj.get("name", "N/A"))[:15]
            price = str(json_obj.get("price", "N/A"))[:8]
            variants = len(json_obj.get("variants", []))
            screen_size = str(json_obj.get("design", {}).get("screen_size", "N/A"))[:13]
            warranty = str(json_obj.get("warranty_years", "N/A"))[:11]
            is_new = str(json_obj.get("is_new", "N/A"))[:8]
            error = ""
            successful_models.append(model)
        else:
            success = "False"
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            field_count = 0
            validation = "N/A"
            name = "N/A"
            price = "N/A"
            variants = 0
            screen_size = "N/A"
            warranty = "N/A"
            is_new = "N/A"
            error = str(result.get("error", "N/A"))[:20]
            failed_models.append(model)

        print(
            row_format.format(
                model[:15],
                success,
                str(prompt_tokens)[:8],
                str(completion_tokens)[:11],
                str(total_tokens)[:6],
                str(field_count)[:7],
                validation,
                name,
                price,
                str(variants)[:9],
                screen_size,
                warranty,
                is_new,
                error,
            )
        )

    print("\n" + "=" * 160)
    print("SUMMARY")
    print("=" * 160)
    print(f"Successful models ({len(successful_models)}): {', '.join(successful_models)}")
    print(f"Failed models ({len(failed_models)}): {', '.join(failed_models)}")


def main():
    """Run the Ollama model comparison example."""
    print("Starting Ollama Model Comparison Example...")
    results = compare_ollama_models()
    print_comparison_table(results)


if __name__ == "__main__":
    main()
