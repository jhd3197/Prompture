"""Detect drift between local rate files and models.dev.

The local files in ``prompture/infra/rates/*.json`` are hand-maintained
capability/limit metadata for the LLM models prompture supports — they are
the primary source of truth; models.dev is a fallback. This script compares
each entry against the live models.dev catalog and reports stale fields plus
models that exist upstream but aren't tracked locally.

By default the script is informational and always exits 0 (drift is reported
but not treated as a failure, since local data may intentionally diverge from
models.dev). Pass ``--strict`` to exit non-zero on any drift or missing model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RATES_DIR = REPO_ROOT / "prompture" / "infra" / "rates"
sys.path.insert(0, str(REPO_ROOT))

from prompture.infra.model_rates import PROVIDER_MAP, _fetch_from_api  # noqa: E402

# Local-field → (models.dev path) for fields we can meaningfully compare.
_LIMIT_FIELDS = {
    "context_window": ("limit", "context"),
    "max_output_tokens": ("limit", "output"),
}
_BOOL_FIELDS = {
    "supports_tool_use": "tool_call",
    "supports_structured_output": "structured_output",
    "is_reasoning": "reasoning",
    "supports_temperature": "temperature",
}


def _remote_models(data: dict[str, Any], api_provider: str) -> dict[str, Any]:
    p = data.get(api_provider)
    if not isinstance(p, dict):
        return {}
    models = p.get("models", p)
    return models if isinstance(models, dict) else {}


def _remote_limit(entry: dict[str, Any], path: tuple[str, str]) -> Any:
    node = entry
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _compare_model(
    provider: str, model_id: str, local: dict[str, Any], remote: dict[str, Any]
) -> list[str]:
    drifts: list[str] = []
    for local_field, path in _LIMIT_FIELDS.items():
        if local_field not in local:
            continue
        remote_val = _remote_limit(remote, path)
        if remote_val is None:
            continue
        try:
            if int(local[local_field]) != int(remote_val):
                drifts.append(
                    f"{provider}/{model_id}: {local_field} local={local[local_field]} "
                    f"remote={remote_val} (stale)"
                )
        except (TypeError, ValueError):
            continue
    for local_field, remote_key in _BOOL_FIELDS.items():
        if local_field not in local or remote_key not in remote:
            continue
        if bool(local[local_field]) != bool(remote[remote_key]):
            drifts.append(
                f"{provider}/{model_id}: {local_field} local={local[local_field]} "
                f"remote={remote[remote_key]} (stale)"
            )
    return drifts


def _file_age_days(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 86400.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-stale-days",
        type=int,
        default=0,
        help="Skip drift checks for files modified within this many days (default: 0).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on any drift or missing model. Default is informational (always exit 0).",
    )
    args = parser.parse_args()

    print("Checking pricing/capability freshness against models.dev...")
    data = _fetch_from_api()
    if data is None:
        print("could not reach models.dev, skipping check")
        return 0

    rate_files = sorted(RATES_DIR.glob("*.json"))
    if not rate_files:
        print(f"No rate files found in {RATES_DIR}", file=sys.stderr)
        return 1

    all_drifts: list[str] = []
    all_missing: list[str] = []
    skipped: list[str] = []

    for rf in rate_files:
        provider = rf.stem
        api_provider = PROVIDER_MAP.get(provider, provider)
        if args.allow_stale_days > 0 and _file_age_days(rf) <= args.allow_stale_days:
            skipped.append(f"{provider} (modified {_file_age_days(rf):.1f}d ago)")
            continue

        try:
            local = json.loads(rf.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"{provider}: failed to parse {rf.name}: {exc}", file=sys.stderr)
            return 1

        remote_models = _remote_models(data, api_provider)
        if not remote_models:
            print(f"{provider}: no models found upstream as '{api_provider}', skipping")
            continue

        for model_id, local_entry in local.items():
            remote_entry = remote_models.get(model_id)
            if remote_entry is None:
                continue  # local-only model (e.g. private/unreleased) — not flagged
            if not isinstance(remote_entry, dict):
                continue
            all_drifts.extend(_compare_model(provider, model_id, local_entry, remote_entry))

        for remote_id in remote_models:
            if remote_id not in local:
                all_missing.append(
                    f"{provider}/{remote_id}: in models.dev but not in local rates"
                )

    print("\n=== Freshness Report ===")
    print(f"Stale fields: {len(all_drifts)}")
    print(f"Untracked upstream models: {len(all_missing)}")
    if skipped:
        print(f"Skipped (within grace period): {', '.join(skipped)}")

    if all_drifts:
        print("\n-- Drift --")
        for line in all_drifts:
            print(line)
    if all_missing:
        print("\n-- Missing --")
        for line in all_missing[:50]:
            print(line)
        if len(all_missing) > 50:
            print(f"... and {len(all_missing) - 50} more")

    if all_drifts or all_missing:
        if args.strict:
            return 1
        print("\n(informational mode: drift reported but not failing — pass --strict to enforce)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
