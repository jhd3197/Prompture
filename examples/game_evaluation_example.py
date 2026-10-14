"""Evaluate a game harness setting on fixtures: hint checks, a scorecard and a paired comparison.

This script demonstrates:
1. Deterministic hint checks on the built-in chess (or Connect Four) fixtures (no model calls).
2. ``evaluate`` over decision fixtures in a seeded, randomized order with raw per-sample records.
3. A readable scorecard and a JSON export from the same records.
4. ``compare`` refusing a quality verdict for scripted evidence.
5. ``--strategy reconsider|proposals``: opt-in experiments through the same harness path.

The default is a dry run: a local scripted driver picks the first listed move,
so nothing is billed and the numbers say nothing about any model. A live run
needs ``--model`` and ``--max-calls`` (a hard cap enforced by ``GameBudget``);
it may cost money with paid providers.

Run it::

    python examples/game_evaluation_example.py
    python examples/game_evaluation_example.py --game connect4
    python examples/game_evaluation_example.py --strategy reconsider
    python examples/game_evaluation_example.py --model ollama/qwen3:8b --max-calls 40
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompture.drivers.base import Driver
from prompture.games import Connect4, GameBudget, GameHarness, Proposals, Reconsider, experiment_status
from prompture.games.evaluation import Preregistration, check_hints, compare, evaluate, scorecard, select, summary_text
from prompture.games.fixtures import builtin_fixtures


class FirstListed(Driver):
    """Local stand-in for a model: answers with the first move on the menu."""

    supports_messages = True
    model = "local/first-listed"

    def generate(self, prompt, options):
        move = re.search(r"^- ([^:\n]+)", prompt, re.M).group(1).strip()
        text = json.dumps({"reason": "first listed", "plan": "", "move": move})
        return {
            "text": text,
            "meta": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 12, "total_tokens": len(prompt) // 4 + 12},
        }


def setup(name: str):
    if name == "chess":
        from prompture.games.chess import ChessGame, ChessHarness, chess_fixtures, chess_hint_report

        return ChessGame(), chess_fixtures(), chess_hint_report, lambda **kw: ChessHarness(**kw)
    game = Connect4()
    return (
        game,
        builtin_fixtures(game.title),
        lambda: check_hints(game, builtin_fixtures(game.title)),
        lambda **kw: GameHarness(game, **kw),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", choices=("chess", "connect4"), default="chess")
    parser.add_argument("--model", help="live model, e.g. ollama/qwen3:8b (requires --max-calls)")
    parser.add_argument("--max-calls", type=int, help="hard cap on model calls for a live run")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", help="write the run as JSON here")
    parser.add_argument("--strategy", choices=("none", "reconsider", "proposals"), default="none")
    args = parser.parse_args()
    if args.model and not args.max_calls:
        parser.error("a live run needs --max-calls so spending is capped")

    game, fixtures, hints, make = setup(args.game)
    report = hints()
    print(f"Hint checks: {report.summary()}")
    for check in report.known_gaps:
        print(f"  known gap ({check.gap}): {check.fixture} {check.move}: {check.problems}")

    pool = select(fixtures)  # dev + validation; the held-out test split stays untouched
    budget = GameBudget(max_calls=args.max_calls) if args.model else None
    driver = None if args.model else FirstListed()

    strategy = {"none": None, "reconsider": Reconsider(), "proposals": Proposals()}[args.strategy]

    def player(concepts):
        return lambda: make(
            model=args.model, driver=driver, level=1, concepts=concepts, budget=budget, strategy=strategy
        )

    evidence = "live" if args.model else "scripted"
    base = evaluate(player(()), pool, game=game, repeats=args.repeats, seed=7, evidence=evidence, label="no concepts")
    taught = evaluate(
        player("all"), pool, game=game, repeats=args.repeats, seed=7, evidence=evidence, label="all concepts"
    )
    print()
    print(summary_text(scorecard(base), title=f"{base.label} ({evidence})"))
    print()
    print(summary_text(scorecard(taught), title=f"{taught.label} ({evidence})"))
    if "experiments" in scorecard(taught):
        print()
        print(f"Experiment counts: {scorecard(taught)['experiments']}")
        print(f"Experiment status: {experiment_status(None)} (no live comparison was run)")
    result = compare(base, taught, Preregistration(factor="concepts", min_pairs=30))
    print(f"\nComparison: {result.verdict}: {'; '.join(result.reasons)}")
    if args.out:
        Path(args.out).write_text(
            json.dumps({"baseline": base.to_dict(), "candidate": taught.to_dict()}, indent=2, default=str)
        )
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
