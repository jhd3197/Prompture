"""
Example script demonstrating how to auto-detect available models using prompture.

This script will:
1. Load environment variables (if any)
2. Call get_available_models() with capability enrichment
3. Print the list of detected models with details (context window, features, etc.)

Usage:
    python discovery_example.py              # default: compact view
    python discovery_example.py --simple     # plain list without capabilities
    python discovery_example.py --verified   # only show models that have been used
"""

import argparse
import os
import sys

# Add parent directory to path so we can import prompture if running from source
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompture import get_available_models
from prompture.model_rates import get_model_rates


def _format_tokens(n: int | None) -> str:
    """Format a token count as a human-readable string (e.g. 128K, 1M)."""
    if n is None:
        return "?"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M" if n % 1_000_000 == 0 else f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K" if n % 1_000 == 0 else f"{n / 1_000:.1f}K"
    return str(n)


def _format_price(rate: float) -> str:
    """Format a per-1M-token rate as a compact dollar string."""
    if rate == 0:
        return "free"
    if rate >= 1:
        return f"${rate:.2f}"
    return f"${rate:.4f}".rstrip("0")


def _capability_badges(caps: dict) -> str:
    """Build a compact badge string from a capabilities dict."""
    badges: list[str] = []
    if caps.get("supports_tool_use"):
        badges.append("tools")
    if caps.get("supports_structured_output"):
        badges.append("json")
    if caps.get("supports_vision"):
        badges.append("vision")
    if caps.get("is_reasoning"):
        badges.append("reasoning")
    if not caps.get("supports_temperature", True):
        badges.append("no-temp")
    return ", ".join(badges) if badges else ""


def main():
    parser = argparse.ArgumentParser(description="Auto-detect available LLM models")
    parser.add_argument("--simple", action="store_true", help="Plain list without capabilities")
    parser.add_argument("--verified", action="store_true", help="Only show models that have been used successfully")
    args = parser.parse_args()

    print("Auto-detecting available models...")

    try:
        if args.simple:
            models = get_available_models(verified_only=args.verified)
            if not models:
                print("No models detected. Check your API keys or .env file.")
                return
            print(f"Found {len(models)} models:")
            print("-" * 40)
            by_provider: dict[str, list[str]] = {}
            for model in models:
                provider = model.split("/")[0]
                by_provider.setdefault(provider, []).append(model)
            for provider, provider_models in sorted(by_provider.items()):
                print(f"\n[{provider.upper()}]")
                for m in sorted(provider_models):
                    print(f"  - {m}")
            print("-" * 40)
            return

        # Enriched mode — show capabilities
        models = get_available_models(include_capabilities=True, verified_only=args.verified)

        if not models:
            print("No models detected. Check your API keys or .env file.")
            return

        print(f"Found {len(models)} models:\n")

        # Group by provider
        by_provider_enriched: dict[str, list[dict]] = {}
        for entry in models:
            by_provider_enriched.setdefault(entry["provider"], []).append(entry)

        for provider, entries in sorted(by_provider_enriched.items()):
            print(f"[{provider.upper()}]")
            for entry in sorted(entries, key=lambda e: e["model_id"]):
                caps = entry.get("capabilities")
                verified_badge = " [verified]" if entry.get("verified") else ""
                last_used = entry.get("last_used")
                use_info = ""
                if last_used:
                    use_info = f"  used={entry.get('use_count', 0)}x last={last_used[:10]}"

                # Fetch pricing from models.dev cache
                rates = get_model_rates(entry["provider"], entry["model_id"])
                price_str = ""
                if rates:
                    inp = _format_price(rates["input"])
                    outp = _format_price(rates["output"])
                    price_str = f"  in={inp}  out={outp}/1M"

                if caps:
                    ctx = _format_tokens(caps.get("context_window"))
                    out = _format_tokens(caps.get("max_output_tokens"))
                    badges = _capability_badges(caps)
                    detail = f"ctx={ctx}  out={out}"
                    if badges:
                        detail += f"  [{badges}]"
                    if price_str:
                        detail += price_str
                    print(f"  {entry['model_id']:<40s} {detail}{verified_badge}{use_info}")
                else:
                    line = f"  {entry['model_id']}"
                    if price_str:
                        line += f"  {price_str.strip()}"
                    print(f"{line}{verified_badge}{use_info}")
            print()

    except Exception as e:
        print(f"Error during discovery: {e}")


if __name__ == "__main__":
    main()
