"""Example: Inspect OpenAI/Claude estimates and provider usage reports.

The default run estimates known token counts without generating output.
Use --count for a real provider token-count request (OPENAI_API_KEY or
CLAUDE_API_KEY), or --billing for yesterday's organization cost report
(OPENAI_ADMIN_KEY or ANTHROPIC_ADMIN_KEY). No inference is generated.

    python examples/cost_reporting_example.py
    python examples/cost_reporting_example.py --count --model openai/gpt-5.5
    python examples/cost_reporting_example.py --billing anthropic
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from prompture import BillingClient, estimate_call_cost, estimate_request_cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gpt-5.5")
    parser.add_argument("--count", action="store_true", help="Make an explicit provider token-count request")
    parser.add_argument("--billing", choices=("openai", "anthropic"), help="Fetch yesterday's admin cost report")
    args = parser.parse_args()

    # === FIRST EXAMPLE: Detailed estimate, including provenance ===
    if args.count:
        estimate = estimate_request_cost(
            args.model,
            [{"role": "user", "content": "Extract the delivery date: Order 1042 arrives October 12."}],
            expected_completion_tokens=100,
        )
    else:
        estimate = estimate_call_cost(args.model, 12000, expected_completion_tokens=500, cached_tokens=8000)
    print("Estimated usage (output length and cache hits are forecasts):")
    print(json.dumps(asdict(estimate), indent=2, default=str))

    # === SECOND EXAMPLE: Separate provider-reported organization costs ===
    if args.billing:
        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        with BillingClient(args.billing) as client:
            report = client.get_costs(end - timedelta(days=1), end)
        print("\nProvider-reported cost snapshot:")
        print(
            json.dumps(
                {"totals": report.totals, "complete": report.complete, "warnings": report.warnings},
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
