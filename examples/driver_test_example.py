#!/usr/bin/env python3
"""
driver_test_example.py — Comprehensive test suite for Prompture API drivers.

Run this script to verify that your configured providers are working correctly.
It tests each driver across multiple Prompture features:

1. Basic JSON extraction (extract_and_jsonify)
2. Pydantic model extraction (extract_with_model)
3. Stepwise field-by-field extraction (stepwise_extract_with_model)
4. Conversations (multi-turn chat)
5. Agent with tool calling
6. TOON input (token-saving structured data extraction)
7. Driver discovery (list available models)

Usage:
    # Test all configured providers
    python examples/driver_test_example.py

    # Test a specific provider
    python examples/driver_test_example.py --provider openai

    # Test a specific model
    python examples/driver_test_example.py --model openai/gpt-4o-mini

    # Run only specific test sections
    python examples/driver_test_example.py --tests extract pydantic agent

    # Quick mode: run only the basic extraction test
    python examples/driver_test_example.py --quick

Environment:
    Copy .env.copy to .env and fill in your API keys before running.
    See .env.copy for all supported providers and their required variables.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from prompture import (
    Agent,
    Conversation,
    extract_and_jsonify,
    extract_from_data,
    extract_with_model,
    get_available_models,
    stepwise_extract_with_model,
)

# ── Default models to test per provider ──────────────────────────────────────
# These are reasonable defaults; override with --model if needed.
PROVIDER_MODELS = {
    "openai": "openai/gpt-4o-mini",
    "claude": "claude/claude-3-5-haiku-20241022",
    "ollama": "ollama/llama3.1:8b",
    "google": "google/gemini-2.5-flash",
    "groq": "groq/llama-3.3-70b-versatile",
    "grok": "grok/grok-3-mini-fast",
    "openrouter": "openrouter/openai/gpt-4o-mini",
    "azure": "azure/gpt-4o-mini",
    "lmstudio": "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b",
    "moonshot": "moonshot/kimi-k2-0905-preview",
    "zai": "zai/glm-4.7",
    "modelscope": "modelscope/Qwen/Qwen3-235B-A22B-Instruct-2507",
}


# ── Shared test data ────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "Maria Garcia is 32 years old and works as a senior software developer "
    "at TechCorp in New York. She was born on April 15, 1993. She loves "
    "hiking and photography, and earns $120,000 per year."
)

BASIC_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "profession": {"type": "string"},
        "city": {"type": "string"},
        "hobbies": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age", "profession", "city", "hobbies"],
}


class PersonModel(BaseModel):
    name: str = Field(..., description="Full name of the person.")
    age: int = Field(..., description="Age in years.", gt=0, lt=150)
    profession: str = Field(..., description="Job title or role.")
    city: str = Field(..., description="City where they live.")
    salary: float = Field(..., description="Annual salary in USD. Use 0 if unknown.")


PRODUCT_DATA = [
    {"id": 1, "name": "Laptop Pro", "price": 999.99, "rating": 4.7, "in_stock": True},
    {"id": 2, "name": "Wireless Mouse", "price": 29.99, "rating": 4.3, "in_stock": True},
    {"id": 3, "name": "USB-C Hub", "price": 49.99, "rating": 4.1, "in_stock": False},
    {"id": 4, "name": "Mechanical Keyboard", "price": 149.99, "rating": 4.8, "in_stock": True},
    {"id": 5, "name": "Monitor 27in", "price": 399.99, "rating": 4.6, "in_stock": True},
]

PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "total_products": {"type": "integer"},
        "average_price": {"type": "number"},
        "highest_rated": {"type": "string"},
        "out_of_stock_count": {"type": "integer"},
    },
    "required": ["total_products", "average_price", "highest_rated", "out_of_stock_count"],
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _subheader(title: str) -> None:
    print(f"\n--- {title} ---")


def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def _usage(usage: dict) -> None:
    tokens = usage.get("total_tokens", "?")
    cost = usage.get("cost", 0)
    model = usage.get("model_name", "?")
    print(f"         tokens={tokens}  cost=${cost:.6f}  model={model}")


# ── Test 1: Basic JSON Extraction ────────────────────────────────────────────

def test_extract(model_name: str) -> bool:
    """Test extract_and_jsonify with a basic schema."""
    _subheader(f"extract_and_jsonify  [{model_name}]")

    try:
        start = time.time()
        result = extract_and_jsonify(
            text=SAMPLE_TEXT,
            json_schema=BASIC_SCHEMA,
            model_name=model_name,
        )
        elapsed = time.time() - start

        obj = result["json_object"]
        usage = result["usage"]

        # Validate key fields exist
        assert "name" in obj, "Missing 'name' in output"
        assert "age" in obj, "Missing 'age' in output"
        assert isinstance(obj["age"], int), f"'age' should be int, got {type(obj['age'])}"

        _pass(f"Extracted: name={obj['name']}, age={obj['age']}, profession={obj.get('profession', '?')}")
        _usage(usage)
        _info(f"Elapsed: {elapsed:.2f}s")
        return True

    except Exception as e:
        _fail(f"{e}")
        return False


# ── Test 2: Pydantic Model Extraction ────────────────────────────────────────

def test_pydantic(model_name: str) -> bool:
    """Test extract_with_model using a Pydantic BaseModel."""
    _subheader(f"extract_with_model (Pydantic)  [{model_name}]")

    try:
        start = time.time()
        result = extract_with_model(
            model_cls=PersonModel,
            text=SAMPLE_TEXT,
            model_name=model_name,
        )
        elapsed = time.time() - start

        person: PersonModel = result["model"]

        assert isinstance(person, PersonModel), f"Expected PersonModel, got {type(person)}"
        assert person.age > 0, f"Age should be > 0, got {person.age}"

        _pass(f"Extracted: {person.name}, age={person.age}, city={person.city}, salary=${person.salary:,.2f}")
        _usage(result["usage"])
        _info(f"Elapsed: {elapsed:.2f}s")
        return True

    except Exception as e:
        _fail(f"{e}")
        return False


# ── Test 3: Stepwise Extraction ──────────────────────────────────────────────

def test_stepwise(model_name: str) -> bool:
    """Test stepwise_extract_with_model (field-by-field extraction)."""
    _subheader(f"stepwise_extract_with_model  [{model_name}]")

    try:
        start = time.time()
        result = stepwise_extract_with_model(
            model_cls=PersonModel,
            text=SAMPLE_TEXT,
            model_name=model_name,
        )
        elapsed = time.time() - start

        if "model" not in result:
            error = result.get("error", "Unknown error")
            field_results = result.get("field_results", {})
            _fail(f"Stepwise returned no model: {error}")
            for fname, fres in field_results.items():
                _info(f"  {fname}: {fres.get('status', '?')}")
            return False

        person: PersonModel = result["model"]

        assert isinstance(person, PersonModel), f"Expected PersonModel, got {type(person)}"

        _pass(f"Stepwise extracted: {person.name}, age={person.age}, salary=${person.salary:,.2f}")
        _info(f"Fields extracted individually ({elapsed:.2f}s total)")
        return True

    except Exception as e:
        _fail(f"{e}")
        return False


# ── Test 4: Conversation ─────────────────────────────────────────────────────

def test_conversation(model_name: str) -> bool:
    """Test multi-turn Conversation with message history."""
    _subheader(f"Conversation (multi-turn)  [{model_name}]")

    try:
        conv = Conversation(
            model_name=model_name,
            system_prompt="You are a concise assistant. Reply in one sentence.",
        )

        start = time.time()
        response1 = conv.ask("What is the capital of France?")
        elapsed1 = time.time() - start

        assert response1 and len(response1) > 0, "Empty first response"
        _pass(f"Turn 1: {response1[:100]}...")
        _info(f"Elapsed: {elapsed1:.2f}s")

        start = time.time()
        response2 = conv.ask("And what country is it in?")
        elapsed2 = time.time() - start

        assert response2 and len(response2) > 0, "Empty follow-up response"
        _pass(f"Turn 2: {response2[:100]}...")
        _info(f"Elapsed: {elapsed2:.2f}s")
        _info(f"Message history: {len(conv.messages)} messages")

        return True

    except Exception as e:
        _fail(f"{e}")
        return False


# ── Test 5: Agent with Tools ─────────────────────────────────────────────────

def test_agent(model_name: str) -> bool:
    """Test Agent with tool calling (function calling)."""
    _subheader(f"Agent with tools  [{model_name}]")

    def get_price(product: str) -> str:
        """Look up the price of a product."""
        prices = {
            "laptop": "$999.99",
            "mouse": "$29.99",
            "keyboard": "$149.99",
        }
        return prices.get(product.lower(), "Product not found")

    def get_stock(product: str) -> str:
        """Check if a product is in stock."""
        stock = {
            "laptop": "In stock",
            "mouse": "In stock",
            "keyboard": "In stock",
        }
        return stock.get(product.lower(), "Out of stock")

    try:
        agent = Agent(
            model_name,
            system_prompt="You are a store assistant. Use the tools to answer questions about products. Be concise.",
            tools=[get_price, get_stock],
        )

        start = time.time()
        result = agent.run("What is the price of the laptop and is it in stock?")
        elapsed = time.time() - start

        assert result.output and len(result.output) > 0, "Empty agent output"

        _pass(f"Output: {result.output[:120]}...")
        _info(f"Tool calls: {len(result.all_tool_calls)}")
        for tc in result.all_tool_calls:
            _info(f"  -> {tc['name']}({tc['arguments']})")
        _info(f"Steps: {len(result.steps)}, Elapsed: {elapsed:.2f}s")

        return True

    except Exception as e:
        _fail(f"{e}")
        return False


# ── Test 6: TOON Input (Token Savings) ───────────────────────────────────────

def test_toon(model_name: str) -> bool:
    """Test extract_from_data with TOON token compression."""
    _subheader(f"extract_from_data (TOON)  [{model_name}]")

    try:
        start = time.time()
        result = extract_from_data(
            data=PRODUCT_DATA,
            question="Analyze the products: count them, calculate average price, find the highest rated, and count out-of-stock items.",
            json_schema=PRODUCT_SCHEMA,
            model_name=model_name,
        )
        elapsed = time.time() - start

        obj = result["json_object"]
        savings = result["token_savings"]

        assert "total_products" in obj, "Missing 'total_products'"

        _pass(f"Extracted: total={obj.get('total_products')}, avg_price={obj.get('average_price')}, top={obj.get('highest_rated')}")
        _info(f"Token savings: {savings['percentage_saved']}% "
              f"(JSON: ~{savings['estimated_json_tokens']} -> TOON: ~{savings['estimated_toon_tokens']} tokens)")
        _usage(result["usage"])
        _info(f"Elapsed: {elapsed:.2f}s")

        return True

    except Exception as e:
        _fail(f"{e}")
        return False


# ── Test 7: Discovery ────────────────────────────────────────────────────────

def test_discovery() -> bool:
    """Test get_available_models to list detected providers and models."""
    _subheader("Model Discovery (get_available_models)")

    try:
        models = get_available_models()

        if not models:
            _fail("No models detected. Check your .env file and API keys.")
            return False

        by_provider: dict[str, list[str]] = {}
        for m in models:
            provider = m.split("/")[0]
            by_provider.setdefault(provider, []).append(m)

        _pass(f"Found {len(models)} models across {len(by_provider)} providers")
        for provider, provider_models in sorted(by_provider.items()):
            _info(f"  [{provider}] {len(provider_models)} model(s): {', '.join(provider_models[:3])}{'...' if len(provider_models) > 3 else ''}")

        return True

    except Exception as e:
        _fail(f"{e}")
        return False


# ── Test runner ──────────────────────────────────────────────────────────────

ALL_TESTS = {
    "extract": ("Basic JSON Extraction", test_extract),
    "pydantic": ("Pydantic Model Extraction", test_pydantic),
    "stepwise": ("Stepwise Field Extraction", test_stepwise),
    "conversation": ("Multi-turn Conversation", test_conversation),
    "agent": ("Agent with Tool Calling", test_agent),
    "toon": ("TOON Input (Token Savings)", test_toon),
}


def run_tests(model_name: str, test_names: list[str] | None = None) -> dict[str, bool]:
    """Run selected tests against a model. Returns {test_name: passed}."""
    tests_to_run = test_names or list(ALL_TESTS.keys())
    results: dict[str, bool] = {}

    for name in tests_to_run:
        if name not in ALL_TESTS:
            print(f"  [WARN] Unknown test '{name}', skipping")
            continue

        label, fn = ALL_TESTS[name]
        passed = fn(model_name)
        results[name] = passed

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test Prompture API drivers across features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/driver_test_example.py                          # test all configured providers
  python examples/driver_test_example.py --provider openai        # test OpenAI only
  python examples/driver_test_example.py --model ollama/gemma:2b  # test a specific model
  python examples/driver_test_example.py --tests extract pydantic # run specific tests only
  python examples/driver_test_example.py --quick                  # basic extraction test only
        """,
    )
    parser.add_argument(
        "--provider", "-p",
        help="Test a single provider (e.g., openai, claude, ollama, google)",
    )
    parser.add_argument(
        "--model", "-m",
        help="Test a specific model (e.g., openai/gpt-4o-mini, claude/claude-3-5-haiku-20241022)",
    )
    parser.add_argument(
        "--tests", "-t",
        nargs="+",
        choices=list(ALL_TESTS.keys()) + ["discovery"],
        help="Run only specific tests (default: all)",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: run only basic extraction test",
    )

    args = parser.parse_args()

    _header("Prompture Driver Test Suite")
    print(f"  Timestamp: {datetime.now().isoformat()}")

    # Determine which models to test
    models_to_test: list[str] = []

    if args.model:
        models_to_test = [args.model]
    elif args.provider:
        provider = args.provider.lower()
        if provider in PROVIDER_MODELS:
            models_to_test = [PROVIDER_MODELS[provider]]
        else:
            models_to_test = [f"{provider}/default"]
    else:
        # Auto-detect: try all providers
        models_to_test = list(PROVIDER_MODELS.values())

    # Determine which tests to run
    test_names = None
    if args.quick:
        test_names = ["extract"]
    elif args.tests:
        test_names = [t for t in args.tests if t != "discovery"]

    # Run discovery test first if requested
    if args.tests and "discovery" in args.tests:
        _header("Discovery Test")
        test_discovery()

    # Run tests per model
    all_results: dict[str, dict[str, bool]] = {}

    for model in models_to_test:
        provider = model.split("/")[0]
        _header(f"Testing: {model}")

        results = run_tests(model, test_names)
        all_results[model] = results

    # Summary
    _header("TEST SUMMARY")
    total_passed = 0
    total_run = 0

    for model, results in all_results.items():
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        total_passed += passed
        total_run += total

        status = "ALL PASSED" if passed == total else f"{passed}/{total} passed"
        marker = "[OK]" if passed == total else "[!!]"
        print(f"  {marker} {model}: {status}")

        for test_name, test_passed in results.items():
            icon = "[PASS]" if test_passed else "[FAIL]"
            label = ALL_TESTS[test_name][0]
            print(f"        {icon} {label}")

    print(f"\n  Total: {total_passed}/{total_run} tests passed across {len(all_results)} model(s)")

    if total_passed == total_run:
        print("\n  All tests passed!")
    else:
        print(f"\n  {total_run - total_passed} test(s) failed. Check provider keys in .env")

    sys.exit(0 if total_passed == total_run else 1)


if __name__ == "__main__":
    main()
