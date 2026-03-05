"""Output formatters for the test suite runner."""

from __future__ import annotations

from typing import Any


def format_table(report: dict[str, Any]) -> str:
    """Format a test suite report as a human-readable summary table.

    Groups results by test_id and shows per-model pass rates, token usage,
    and cost.
    """
    results = report.get("results", [])
    if not results:
        return "No results."

    # Group by test_id -> model_id
    tests: dict[str, dict[str, dict[str, Any]]] = {}
    for r in results:
        tid = r["test_id"]
        mid = r["model_id"]
        if tid not in tests:
            tests[tid] = {}
        if mid not in tests[tid]:
            tests[tid][mid] = {"passed": 0, "total": 0, "tokens": 0, "cost": 0.0}
        entry = tests[tid][mid]
        entry["total"] += 1
        if r.get("validation", {}).get("ok"):
            entry["passed"] += 1
        usage = r.get("usage", {})
        entry["tokens"] += usage.get("total_tokens", 0)
        entry["cost"] += usage.get("cost", 0.0)

    lines: list[str] = []
    meta = report.get("meta", {})
    suite_name = meta.get("suite", meta.get("project", "Test Suite"))
    lines.append(f"Cross-Model Test Results: {suite_name}")
    lines.append("=" * len(lines[0]))
    lines.append("")

    for tid, models in tests.items():
        lines.append(f"Test: {tid}")

        # Calculate column widths
        model_names = list(models.keys())
        col_model = max(len("Model"), max(len(m) for m in model_names))
        col_pass = max(len("Pass"), 5)
        col_tokens = max(len("Tokens"), 6)
        col_cost = max(len("Cost"), 8)

        header = (
            f"  {'Model':<{col_model}}  "
            f"{'Pass':>{col_pass}}  "
            f"{'Tokens':>{col_tokens}}  "
            f"{'Cost':>{col_cost}}"
        )
        sep = f"  {'-' * col_model}  {'-' * col_pass}  {'-' * col_tokens}  {'-' * col_cost}"

        lines.append(header)
        lines.append(sep)

        for mid, stats in models.items():
            pass_str = f"{stats['passed']}/{stats['total']}"
            cost_str = f"${stats['cost']:.4f}"
            lines.append(
                f"  {mid:<{col_model}}  "
                f"{pass_str:>{col_pass}}  "
                f"{stats['tokens']:>{col_tokens}}  "
                f"{cost_str:>{col_cost}}"
            )

        lines.append("")

    # Summary
    total_pass = sum(
        s["passed"] for m in tests.values() for s in m.values()
    )
    total_tests = sum(
        s["total"] for m in tests.values() for s in m.values()
    )
    total_cost = sum(
        s["cost"] for m in tests.values() for s in m.values()
    )
    lines.append(f"Overall: {total_pass}/{total_tests} passed, total cost: ${total_cost:.4f}")

    return "\n".join(lines)
