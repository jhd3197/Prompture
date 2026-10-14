"""A resumable, side-balanced Connect Four tournament between local players.

This script demonstrates:
1. A double round robin over frozen entrants (4 entrants -> 12 games).
2. A JSONL batch store: stop part-way, run again, and finished games are skipped.
3. Standings over decided games and leaderboard rows with eligibility flags.
4. Exploratory calibration against ordered engine profiles (not a rating).

All players are local engines or seeded random movers: no model calls, no cost.
Swap an entrant's factory for ``GameHarness(ctx.game, "provider/model", budget=ctx.budget,
cancel=ctx.cancel)`` to enter a model (and pass a ``GameBudget`` to ``run_batch``).

Run it::

    python examples/game_tournament_example.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompture.games import Connect4, Connect4Engine, EnginePlayer, RandomPlayer
from prompture.games.control import CancelToken
from prompture.games.tournament import BatchStore, Entrant, calibrate, double_round_robin, run_batch, standings


def engine(depth: int) -> Entrant:
    return Entrant(
        f"engine-d{depth}",
        lambda ctx: EnginePlayer(ctx.game, Connect4Engine(depth)),
        {"engine": "Connect4Engine", "depth": depth},
    )


def random_player(name: str, seed: int) -> Entrant:
    return Entrant(name, lambda ctx: RandomPlayer(ctx.game, seed=seed + int(ctx.game_id[-4:])), {"random_seed": seed})


def main() -> None:
    entrants = [engine(1), engine(3), random_player("random-a", 1), random_player("random-b", 2)]
    schedule = double_round_robin([e.name for e in entrants], Connect4.sides)
    store = Path(tempfile.mkdtemp()) / "connect4-batch.jsonl"
    print(f"{len(schedule)} games scheduled; results stored in {store}")

    stop = CancelToken()
    finished: list[str] = []

    def stop_after_four(outcome) -> None:
        finished.append(outcome.game_id)
        if len(finished) == 4:
            stop.cancel("operator pressed Stop")

    first = run_batch(Connect4, entrants, schedule, store=store, cancel=stop, on_outcome=stop_after_four)
    print(
        f"first run: {len(first.outcomes)} played, {len(first.not_started)} not started (cancelled={first.cancelled})"
    )
    second = run_batch(Connect4, entrants, schedule, store=store, max_concurrency=2)
    print(f"resumed:   {len(second.skipped)} skipped as finished, {len(second.outcomes)} played")

    everything = BatchStore(store).outcomes()
    table = standings(everything)
    print(f"\nStandings over all {len(everything)} stored results (decided games only):")
    for name, row in sorted(table.items(), key=lambda kv: -kv[1]["score"]):
        print(f"  {name:<10} {row['score']:>4} / {row['games']}  (W{row['wins']} D{row['draws']} L{row['losses']})")
    sample = second.export()[0]
    print(
        f"\nLeaderboard row example: {sample['entrants']} -> {sample['terminal_reason']}, eligibility {sample['eligibility']}"
    )

    result = calibrate(
        engine(2),
        [random_player("random", 7), engine(1), engine(3), engine(5)],
        Connect4,
        games_per_step=4,
        max_games=24,
    )
    print(f"\nCalibration ({result.status}): {result.reason}; {result.games} games")
    for step in result.steps:
        print(f"  vs {step.profile}: score {step.score} over {step.games} games, 95% interval {step.interval}")
    print(f"  {result.note}")


if __name__ == "__main__":
    main()
