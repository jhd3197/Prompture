"""Example script comparing JSON vs TOON token usage for a single model."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompture import extract_and_jsonify

COMPARISON_TEXT = """
Alice Johnson is a 30-year-old data scientist from Denver who specializes in
machine learning. She has 8 years of experience, currently works at QuantumSight,
and previously led automation projects that reduced costs by 18%. Alice mentors
a team of five engineers and recently gave a talk on applying TOON serialization
to reduce LLM output tokens.

Outside of work she enjoys trail running and volunteers at a local coding bootcamp.
She is available for freelance consulting. She is reachable at alice@example.com and
lists Python, SQL, and prompt engineering as core skills.
"""

COMPARISON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "location": {"type": "string"},
        "experience_years": {"type": "integer"},
        "current_company": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "recent_achievement": {"type": "string"},
        "availability": {"type": "string"},
        "email": {"type": "string"},
    },
    "required": ["name", "age", "skills"],
}

MODEL_TO_TEST = os.getenv(
    "PROMPTURE_COMPARISON_MODEL",
    "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b",
)
OUTPUT_FORMATS = ["json", "toon"]
REFERENCE_FORMAT = "json"


def run_extraction(fmt: str) -> Dict[str, Any]:
    result = extract_and_jsonify(
        text=COMPARISON_TEXT,
        json_schema=COMPARISON_SCHEMA,
        model_name=MODEL_TO_TEST,
        output_format=fmt,
    )
    return {
        "success": True,
        "json_object": result["json_object"],
        "usage": result.get("usage", {}),
        "json_string": result["json_string"],
    }


def compare_formats() -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for fmt in OUTPUT_FORMATS:
        print(f"Running extraction with output_format='{fmt}'...")
        try:
            results[fmt] = run_extraction(fmt)
            usage = results[fmt].get("usage", {})
            print(
                f"  ✓ success — total_tokens={usage.get('total_tokens', 'N/A')} "
                f"(prompt={usage.get('prompt_tokens', 'N/A')}, completion={usage.get('completion_tokens', 'N/A')})"
            )
        except Exception as exc:
            print(f"  ✗ failed — {exc}")
            results[fmt] = {"success": False, "error": str(exc)}
    return results


def _delta_tokens(current: int | float, baseline: int | float) -> tuple[str, str]:
    if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
        return ("N/A", "N/A")
    if baseline == 0:
        return ("N/A", "N/A")
    diff = baseline - current
    pct = (diff / baseline) * 100 if baseline else 0
    return (f"{diff:.0f}", f"{pct:.1f}%")


def print_comparison_table(results: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "=" * 120)
    print("JSON VS TOON OUTPUT COMPARISON")
    print("=" * 120)
    headers = [
        "Format",
        "Success",
        "Prompt",
        "Completion",
        "Total",
        "Δ Tokens",
        "Δ %",
        "Fields",
        "Valid",
        "Error",
    ]
    row_fmt = "{:<10} {:<7} {:<10} {:<12} {:<10} {:<10} {:<8} {:<8} {:<6} {:<20}"
    print(row_fmt.format(*headers))
    print("-" * 120)

    baseline_total = results.get(REFERENCE_FORMAT, {}).get("usage", {}).get("total_tokens", 0)

    for fmt in OUTPUT_FORMATS:
        entry = results.get(fmt, {})
        success = entry.get("success", False)
        if success:
            json_obj = entry["json_object"]
            usage = entry.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            fields = len(json_obj)
            has_required = all(json_obj.get(field) not in (None, "") for field in ["name", "age"])
            delta_tokens, delta_pct = (
                _delta_tokens(total_tokens, baseline_total)
                if fmt != REFERENCE_FORMAT
                else ("-", "-")
            )
            error = ""
        else:
            prompt_tokens = completion_tokens = total_tokens = 0
            fields = 0
            has_required = False
            delta_tokens = delta_pct = "-"
            error = str(entry.get("error", ""))[:20]

        print(
            row_fmt.format(
                fmt,
                str(success),
                str(prompt_tokens),
                str(completion_tokens),
                str(total_tokens),
                delta_tokens,
                delta_pct,
                str(fields),
                "✓" if has_required else "✗",
                error,
            )
        )

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    for fmt in OUTPUT_FORMATS:
        entry = results.get(fmt)
        if not entry:
            print(f"{fmt}: no data")
            continue
        if not entry.get("success"):
            print(f"{fmt}: failed ({entry.get('error')})")
            continue
        usage = entry.get("usage", {})
        print(
            f"{fmt}: total={usage.get('total_tokens', 'N/A')}, prompt={usage.get('prompt_tokens', 'N/A')}, "
            f"completion={usage.get('completion_tokens', 'N/A')}"
        )


def main() -> None:
    print(f"Model: {MODEL_TO_TEST}")
    results = compare_formats()
    print_comparison_table(results)


if __name__ == "__main__":
    main()
