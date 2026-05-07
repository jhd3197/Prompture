"""
Example: Using extract_with_models for multi-model fallback with per-attempt accounting.

This example demonstrates how to:
1. Define a Pydantic model that describes the desired structured output.
2. Pass an ordered list of models to ``extract_with_models``; the first model
   that succeeds wins, while every attempt (success, failure, or skip) is
   recorded with full cost and operational tracking.
3. Force a fallback by placing a deliberately invalid model first in the list,
   so the per-attempt accounting array surfaces both a failed attempt and a
   subsequent successful one.
4. Inspect the returned ``attempts`` array — status, strategy, capabilities,
   tokens, cost, and duration are reported per model, which makes failure
   modes and routing decisions auditable.

The result returned by ``extract_with_models`` includes:
- model: the validated Pydantic instance produced by the winning model.
- selected_model: the model string that succeeded.
- strategy_used: "single" or "stepwise" (depends on driver capabilities).
- attempts: list of per-attempt dicts (model, status, reason, strategy,
  cost, prompt_tokens, completion_tokens, duration_ms, capabilities).
- total_cost: sum of costs across all attempts (not just the winner).
- total_tokens: sum of tokens across all attempts.
- total_attempts: number of models actually called.
- usage: the winning attempt's usage dict (backward compatible).
"""

import json

from pydantic import BaseModel, Field

from prompture import extract_with_models


# 1. Define the Pydantic model describing the structured output
class Person(BaseModel):
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    profession: str = Field(description="Job title or profession")
    city: str = Field(description="City where the person lives")
    hobbies: list[str] = Field(default_factory=list, description="List of hobbies")


# 2. Define the raw text to extract from
text = (
    "Maria is 32 years old and works as a software developer in New York. "
    "She loves hiking and photography."
)

# 3. Build a model fallback chain
# The first entry is intentionally a non-existent model — its driver will
# either fail to resolve (recorded as "skipped") or fail at request time
# (recorded as "failed"). The second entry is a real local Ollama model
# that should succeed and become the winner.
models = [
    "openai/this-model-does-not-exist",
    "ollama/gpt-oss:20b",
]

# === FIRST EXAMPLE: Multi-model fallback with forced first-attempt failure ===
print("Extracting structured Person info using a fallback chain...")
print(f"Models (in priority order): {models}")

result = extract_with_models(
    Person,
    text,
    models=models,
    instruction_template="Extract the biographical details from this text:",
)

# 4. Inspect the winning extraction
print("\n=== WINNING EXTRACTION ===")
print(f"Selected model:  {result['selected_model']}")
print(f"Strategy used:   {result['strategy_used']}")
print(f"Total attempts:  {result['total_attempts']}")

person = result["model"]
print("\nValidated Pydantic instance:")
print(f"  name:       {person.name}")
print(f"  age:        {person.age}")
print(f"  profession: {person.profession}")
print(f"  city:       {person.city}")
print(f"  hobbies:    {person.hobbies}")

print("\nRaw JSON output from winning model:")
print(json.dumps(result["json_object"], indent=2))


# === SECOND EXAMPLE: Per-attempt accounting array ===
# This is what makes extract_with_models more transparent than other extraction
# libraries: every model attempted is recorded — including failures and skips —
# with its own cost, token usage, duration, and capability flags.
print("\n\n=== PER-ATTEMPT ACCOUNTING ===")
print(f"Total attempts recorded: {len(result['attempts'])}")

for i, attempt in enumerate(result["attempts"], start=1):
    print(f"\n--- Attempt #{i} ---")
    print(f"  Model:             {attempt['model']}")
    print(f"  Status:            {attempt['status']}")
    print(f"  Reason:            {attempt['reason']}")
    print(f"  Strategy:          {attempt['strategy']}")
    print(f"  Capabilities:      {attempt['capabilities']}")
    print(f"  Prompt tokens:     {attempt['prompt_tokens']}")
    print(f"  Completion tokens: {attempt['completion_tokens']}")
    print(f"  Cost (USD):        {attempt['cost']}")
    print(f"  Duration (ms):     {attempt['duration_ms']}")


# === THIRD EXAMPLE: Aggregate cost & token tracking across all attempts ===
# Note: total_cost and total_tokens span ALL attempts, not just the winner.
# This matters because failed attempts can still incur tokens (e.g., a model
# that produced output but failed to validate). The winning attempt's usage
# is also exposed via result["usage"] for backward compatibility.
print("\n=== AGGREGATE TOTALS (across all attempts) ===")
print(f"Total cost (USD): {result['total_cost']}")
print(f"Total tokens:     {result['total_tokens']}")

winning_usage = result["usage"]
print("\n=== WINNING ATTEMPT USAGE (backward-compatible view) ===")
print(f"Prompt tokens:     {winning_usage.get('prompt_tokens')}")
print(f"Completion tokens: {winning_usage.get('completion_tokens')}")
print(f"Total tokens:      {winning_usage.get('total_tokens')}")
print(f"Cost (USD):        {winning_usage.get('cost')}")
print(f"Model name:        {winning_usage.get('model_name')}")
