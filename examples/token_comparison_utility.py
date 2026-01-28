"""Token comparison utility for previewing TOON vs JSON savings without LLM calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Union

# Add project root to path
EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import toon

    TOON_AVAILABLE = True
except ImportError:
    TOON_AVAILABLE = False


def estimate_token_count(text: str, chars_per_token: int = 4) -> int:
    """Estimate token count from character count.

    Args:
        text: Text to count tokens for
        chars_per_token: Rough ratio of characters to tokens (varies by model)

    Returns:
        Estimated token count
    """
    return max(1, len(text) // chars_per_token)


def compare_formats(
    data: Union[list[dict[str, Any]], dict[str, Any]], data_key: str | None = None, chars_per_token: int = 4
) -> dict[str, Any]:
    """Compare JSON vs TOON format for token efficiency.

    Args:
        data: List of dicts or dict containing array
        data_key: If data is dict, key containing the array
        chars_per_token: Character to token ratio for estimation

    Returns:
        Dictionary with comparison statistics
    """
    # Extract array data
    if isinstance(data, list):
        array_data = data
        data_label = "Array"
    elif isinstance(data, dict):
        if data_key and data_key in data:
            array_data = data[data_key]
            data_label = f"data['{data_key}']"
        else:
            # Find first array in dict
            array_data = None
            for key, value in data.items():
                if isinstance(value, list) and value:
                    array_data = value
                    data_label = f"data['{key}']"
                    break
            if array_data is None:
                raise ValueError("No array found in data")
    else:
        raise ValueError("Data must be list or dict")

    if not array_data:
        raise ValueError("Array data is empty")

    # Generate JSON format
    json_text = json.dumps(array_data, indent=2)
    json_compact = json.dumps(array_data, separators=(",", ":"))

    # Generate TOON format (if available)
    toon_text = None
    toon_error = None

    if TOON_AVAILABLE:
        try:
            toon_text = toon.encode(array_data)
        except Exception as e:
            toon_error = str(e)
    else:
        toon_error = "python-toon not installed"

    # Calculate statistics
    stats = {
        "data_info": {
            "label": data_label,
            "array_length": len(array_data),
            "sample_keys": list(array_data[0].keys()) if array_data else [],
            "uniform_structure": all(set(item.keys()) == set(array_data[0].keys()) for item in array_data)
            if array_data
            else False,
        },
        "json_formatted": {
            "text": json_text,
            "character_count": len(json_text),
            "estimated_tokens": estimate_token_count(json_text, chars_per_token),
        },
        "json_compact": {
            "text": json_compact,
            "character_count": len(json_compact),
            "estimated_tokens": estimate_token_count(json_compact, chars_per_token),
        },
    }

    if toon_text:
        stats["toon"] = {
            "text": toon_text,
            "character_count": len(toon_text),
            "estimated_tokens": estimate_token_count(toon_text, chars_per_token),
        }

        # Calculate savings vs formatted JSON
        json_chars = stats["json_formatted"]["character_count"]
        toon_chars = stats["toon"]["character_count"]
        char_savings = json_chars - toon_chars
        char_savings_pct = (char_savings / json_chars * 100) if json_chars > 0 else 0

        json_tokens = stats["json_formatted"]["estimated_tokens"]
        toon_tokens = stats["toon"]["estimated_tokens"]
        token_savings = json_tokens - toon_tokens
        token_savings_pct = (token_savings / json_tokens * 100) if json_tokens > 0 else 0

        stats["savings_vs_formatted"] = {
            "character_savings": char_savings,
            "character_savings_percent": round(char_savings_pct, 1),
            "token_savings": token_savings,
            "token_savings_percent": round(token_savings_pct, 1),
        }

        # Calculate savings vs compact JSON
        json_compact_chars = stats["json_compact"]["character_count"]
        char_savings_compact = json_compact_chars - toon_chars
        char_savings_pct_compact = (char_savings_compact / json_compact_chars * 100) if json_compact_chars > 0 else 0

        json_compact_tokens = stats["json_compact"]["estimated_tokens"]
        token_savings_compact = json_compact_tokens - toon_tokens
        token_savings_pct_compact = (
            (token_savings_compact / json_compact_tokens * 100) if json_compact_tokens > 0 else 0
        )

        stats["savings_vs_compact"] = {
            "character_savings": char_savings_compact,
            "character_savings_percent": round(char_savings_pct_compact, 1),
            "token_savings": token_savings_compact,
            "token_savings_percent": round(token_savings_pct_compact, 1),
        }
    else:
        stats["toon_error"] = toon_error

    return stats


def print_comparison_report(stats: dict[str, Any]) -> None:
    """Print a formatted comparison report.

    Args:
        stats: Statistics from compare_formats()
    """
    info = stats["data_info"]
    print("=" * 80)
    print("TOKEN EFFICIENCY COMPARISON: JSON vs TOON")
    print("=" * 80)

    print(f"Data: {info['label']}")
    print(f"Array Length: {info['array_length']} items")
    print(f"Sample Keys: {', '.join(info['sample_keys'][:5])}")
    if len(info["sample_keys"]) > 5:
        print(f"             ...and {len(info['sample_keys']) - 5} more")
    print(f"Uniform Structure: {'Yes' if info['uniform_structure'] else 'No'}")

    if not info["uniform_structure"]:
        print("\n⚠️  WARNING: Non-uniform structure may reduce TOON efficiency")

    print("\n" + "-" * 80)
    print("FORMAT COMPARISON")
    print("-" * 80)

    # Table headers
    headers = ["Format", "Characters", "Est. Tokens", "Savings (chars)", "Savings (%)"]
    print(f"{headers[0]:<15} {headers[1]:>12} {headers[2]:>12} {headers[3]:>15} {headers[4]:>12}")
    print("-" * 80)

    # JSON formatted
    json_fmt = stats["json_formatted"]
    print(
        f"{'JSON (formatted)':<15} {json_fmt['character_count']:>12} {json_fmt['estimated_tokens']:>12} {'-':>15} {'-':>12}"
    )

    # JSON compact
    json_cmp = stats["json_compact"]
    json_fmt_chars = json_fmt["character_count"]
    compact_savings_chars = json_fmt_chars - json_cmp["character_count"]
    compact_savings_pct = (compact_savings_chars / json_fmt_chars * 100) if json_fmt_chars > 0 else 0
    print(
        f"{'JSON (compact)':<15} {json_cmp['character_count']:>12} {json_cmp['estimated_tokens']:>12} {compact_savings_chars:>15} {compact_savings_pct:>11.1f}%"
    )

    # TOON
    if "toon" in stats:
        toon_data = stats["toon"]
        savings_fmt = stats["savings_vs_formatted"]
        print(
            f"{'TOON':<15} {toon_data['character_count']:>12} {toon_data['estimated_tokens']:>12} {savings_fmt['character_savings']:>15} {savings_fmt['character_savings_percent']:>11.1f}%"
        )
    else:
        print(f"{'TOON':<15} {'N/A':>12} {'N/A':>12} {'N/A':>15} {'N/A':>12}")
        print(f"\nTOON Error: {stats.get('toon_error', 'Unknown error')}")

    # Summary
    if "savings_vs_formatted" in stats:
        savings = stats["savings_vs_formatted"]
        print("\n📊 SUMMARY (TOON vs Formatted JSON):")
        print(f"   Character reduction: {savings['character_savings']} chars ({savings['character_savings_percent']}%)")
        print(f"   Estimated token savings: ~{savings['token_savings']} tokens ({savings['token_savings_percent']}%)")

        if savings["token_savings_percent"] > 50:
            print("   🎉 Excellent savings! TOON is highly effective for this data.")
        elif savings["token_savings_percent"] > 25:
            print("   ✅ Good savings! TOON provides meaningful token reduction.")
        elif savings["token_savings_percent"] > 0:
            print("   📈 Modest savings. TOON provides some benefit.")
        else:
            print("   ⚠️  No savings. This data may not benefit from TOON format.")

    # Show sample data formats
    print("\n" + "-" * 80)
    print("SAMPLE OUTPUT FORMATS")
    print("-" * 80)

    print("\n📄 JSON (formatted):")
    json_preview = stats["json_formatted"]["text"]
    if len(json_preview) > 300:
        print(json_preview[:300] + "...")
    else:
        print(json_preview)

    if "toon" in stats:
        print("\n📋 TOON:")
        toon_preview = stats["toon"]["text"]
        if len(toon_preview) > 300:
            print(toon_preview[:300] + "...")
        else:
            print(toon_preview)


def demo_with_sample_data():
    """Demonstrate the utility with sample data sets."""
    print("TOKEN COMPARISON UTILITY DEMONSTRATION")
    print("Comparing different data structures for TOON efficiency")

    # Sample 1: Uniform product data (ideal for TOON)
    print("\n" + "🛍️" * 3 + " SAMPLE 1: E-COMMERCE PRODUCTS (UNIFORM STRUCTURE)")
    products = [
        {"id": 1, "name": "Laptop", "price": 999.99, "category": "electronics", "rating": 4.5},
        {"id": 2, "name": "Book", "price": 19.99, "category": "books", "rating": 4.2},
        {"id": 3, "name": "Headphones", "price": 149.99, "category": "electronics", "rating": 4.7},
        {"id": 4, "name": "Coffee Mug", "price": 12.99, "category": "home", "rating": 4.0},
    ]

    try:
        stats = compare_formats(products)
        print_comparison_report(stats)
    except Exception as e:
        print(f"Error analyzing products: {e}")

    # Sample 2: API response structure
    print("\n\n" + "🌐" * 3 + " SAMPLE 2: API RESPONSE WITH NESTED DATA")
    api_response = {
        "status": "success",
        "page": 1,
        "total_pages": 5,
        "results": [
            {"user_id": 101, "username": "alice", "score": 85, "level": "advanced"},
            {"user_id": 102, "username": "bob", "score": 72, "level": "intermediate"},
            {"user_id": 103, "username": "charlie", "score": 94, "level": "expert"},
        ],
    }

    try:
        stats = compare_formats(api_response, data_key="results")
        print_comparison_report(stats)
    except Exception as e:
        print(f"Error analyzing API response: {e}")

    # Sample 3: Financial data
    print("\n\n" + "💰" * 3 + " SAMPLE 3: FINANCIAL TRANSACTIONS")
    transactions = [
        {"date": "2024-01-15", "amount": -45.67, "merchant": "Coffee Shop", "category": "food"},
        {"date": "2024-01-16", "amount": -1200.00, "merchant": "Rent Payment", "category": "housing"},
        {"date": "2024-01-17", "amount": 3500.00, "merchant": "Salary Deposit", "category": "income"},
        {"date": "2024-01-18", "amount": -89.32, "merchant": "Gas Station", "category": "transport"},
    ]

    try:
        stats = compare_formats(transactions)
        print_comparison_report(stats)
    except Exception as e:
        print(f"Error analyzing transactions: {e}")


def interactive_mode():
    """Interactive mode for testing user's own data."""
    print("\n" + "🔧" * 3 + " INTERACTIVE MODE")
    print("Enter your JSON data to see TOON efficiency analysis")
    print("(Type 'quit' to exit)")

    while True:
        try:
            user_input = input("\nEnter JSON array or paste data: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                break

            if not user_input:
                continue

            # Try to parse as JSON
            data = json.loads(user_input)

            if isinstance(data, list):
                stats = compare_formats(data)
            elif isinstance(data, dict):
                # Ask for key if it's a dict
                print("Dict detected. Available keys:", list(data.keys()))
                key = input("Which key contains the array? (or press Enter for auto-detect): ").strip()
                stats = compare_formats(data, data_key=key if key else None)
            else:
                print("Please provide a JSON array or object containing an array")
                continue

            print_comparison_report(stats)

        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function with command-line options."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        demo_with_sample_data()

        print(f"\n{'=' * 80}")
        print("💡 TIP: Use extract_from_data() or extract_from_pandas() in your code")
        print("   to automatically get these token savings when querying LLMs!")
        print(f"{'=' * 80}")

        if not TOON_AVAILABLE:
            print("\n⚠️  To see actual TOON output, install python-toon:")
            print("   pip install python-toon")


if __name__ == "__main__":
    main()
