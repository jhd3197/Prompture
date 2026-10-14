# Games: Reliable Turns, Records, Evaluation and Batches

`prompture.games` lets a language model play rules-bound games without illegal moves: each turn the model picks from a menu of legal candidates (an enum-constrained JSON reply), and every reply is checked against the rules. Chess, Connect Four and Othello ship with the library. Chess needs `pip install "prompture[chess]"`; nothing else does, and `import prompture.games` never loads python-chess.

This page covers the harness's records, turn control, evaluation, memory, batches and experiments, and the contract a host application builds on.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Turn Records and Failure Policies](#turn-records-and-failure-policies)
- [Deadlines, Cancellation, Budgets and Progress](#deadlines-cancellation-budgets-and-progress)
- [Evaluation and Scorecards](#evaluation-and-scorecards)
- [Memory, Context Budgets and Checkpoints](#memory-context-budgets-and-checkpoints)
- [Batches, Schedules and Recovery](#batches-schedules-and-recovery)
- [Experiments (Off by Default)](#experiments-off-by-default)
- [Host Integration Contract](#host-integration-contract)
- [Writing a Game](#writing-a-game)

---

## Quick Start

```python
from prompture.games import Connect4, GameHarness, RandomPlayer, play_match

game = Connect4()
harness = GameHarness(game, "ollama/qwen3:8b", level=1, concepts="all")
result = play_match(game, {"Red": harness, "Yellow": RandomPlayer(game, seed=1)})
print(result.winner, result.terminal_reason, result.rescues)
```

Runnable, no-cost examples: `examples/game_progress_example.py`, `examples/game_evaluation_example.py`, `examples/game_tournament_example.py`. Each uses a local scripted driver or local engines unless you pass `--model`.

## Turn Records and Failure Policies

Every `MoveDecision` carries `decision.record` (a `TurnRecord`): run, game and turn IDs, side, ply, a state digest, the configuration fingerprint, and one `Attempt` per model call. Each attempt records its purpose, rendered prompt and schema, public response (partial for a broken stream), validation result, classified failure, timing, finish reason and its own usage.

- **Failure kinds** (`FailureKind`): `invalid_output`, `auth`, `configuration`, `rate_limit`, `upstream`, `transport`, `timeout`, `cancelled`, `budget`. Status codes, `Retry-After` and cause chains are read through the resilience error classifier.
- **Accounting** (`record.accounting`): calls, tokens, cost, and how many calls had partial or unknown usage or price. A call whose price is unknown makes the cost a lower bound (`cost_label` shows `>= $…`); it is never reported as zero.
- **Failure policy**: `failure_policy="rescue"` (default) plays a fallback marked `rescued=True`, `assistance="rescue"`. `"forfeit"` and `"stop"` raise `TurnFailed`, and `play_match` ends the game as `forfeit` (the other side wins) or `stopped` (no winner, not a draw). Cancellation, budget exhaustion and auth/configuration failures always raise, so they never become a played move.
- **Fingerprint** (`harness.fingerprint()`): a digest of the behavior configuration (game and rule versions, prompt and hint versions, level, concepts, memory, retry and failure policy, allowlisted options such as effort and output limits, engine assistance, strategy, representation). Model IDs, player names and secrets are excluded.
- **Traces**: `trace=JsonlTraceSink(path)` (or `MemoryTraceSink`) receives `run`, `turn_started`, `attempt` and `turn_finished` records built from allowlisted fields, redacted and size-bounded. `trace_errors="warn"` logs sink failures; `"raise"` fails the turn before its move is returned.

## Deadlines, Cancellation, Budgets and Progress

```python
from prompture.games import CancelToken, EventBuffer, GameBudget, RetryPolicy

stop = CancelToken()
events = EventBuffer(maxsize=256)
harness = GameHarness(game, model, deadline=60, cancel=stop, retry=RetryPolicy(),
                      budget=GameBudget(max_calls=200, max_cost=2.0), on_event=events.put)
```

- `deadline` is a total per-turn limit covering requests, retries and backoff. An expired turn is handled by the failure policy.
- `cancel` stops a turn. A streaming request is closed between chunks. A blocking request cannot be interrupted from Python, so it is marked `abandoned`, its late reply is logged (`attempt.late`) and never applied, and no request is sent on its behalf. The provider may still finish and bill it.
- `retry=RetryPolicy(...)` gives transient failures (rate limits, 5xx, timeouts, dropped connections) their own allowance with exponential backoff, jitter and `Retry-After`. Without it, the legacy behavior holds: they share `max_retries` and retry immediately.
- `GameBudget` reserves before every request and settles afterwards. It can be shared across harnesses and threads. With `max_cost` set, an unknown price stops spending unless `max_calls` or `max_tokens` still bounds it (always, with `strict=True`).
- `on_event` receives sequenced `TurnEvent`s (`turn_started`, `attempt_started`, `output_delta`, `attempt_finished`, `retry_scheduled`, `turn_completed`, `turn_failed`, `turn_cancelled`, `late_response`). Events may arrive on a worker thread. `EventBuffer` bounds memory for slow consumers by merging deltas.
- `await harness.achoose(state)`: cancelling the task cancels the turn.

## Evaluation and Scorecards

```python
from prompture.games import Preregistration, compare, evaluate, scorecard, summary_text
from prompture.games.chess import ChessGame, ChessHarness, chess_fixtures, chess_hint_report

print(chess_hint_report().summary())        # deterministic hint checks, no model calls
run = evaluate(lambda: ChessHarness(model, level=1, concepts="all"), chess_fixtures(),
               game=ChessGame(), repeats=3, seed=7, evidence="live")
print(summary_text(scorecard(run)))
```

- Fixtures carry provenance and a split (`dev`, `validation`, `test`). `select(..., ["test"])` needs `include_test=True`.
- Scorecards keep strength (unassisted success), reliability (first-answer validity, invalid replies per 100 attempts, rescues per turn, failed turns, clean games) and efficiency (tokens, cost and latency per accepted move) apart. Every metric shows its numerator and denominator. Blunders need an independent `judge` and are otherwise `not_assessed`.
- `compare(baseline, candidate, Preregistration(factor=...))` returns `not_evidence` for scripted runs, `not_comparable` when more than the declared factor changed, and `inconclusive` for small or mixed samples. `improved` requires a paired interval above zero with the reliability and cost gates satisfied.

## Memory, Context Budgets and Checkpoints

- `memory="none" | "previous" | n`: the side's own accepted reasons and plans, tied to the exact game line. Chess defaults to `"previous"`; other games default to `"none"`. Memory advances when the move is applied (seen in the game line, or via `harness.commit(state, decision)`). Rejected attempts, opponents' explanations and rescues never become remembered plans.
- `harness.memory.checkpoint(state)` / `.restore(checkpoint, state)` validate version, game, initial position and move prefix. `.restore(None, state)` resumes a legacy game without memory, explicitly.
- `context_budget=n` (estimated tokens): the position, every legal move and the instructions always stay; hints are added urgent-first, then memory. If the required content cannot fit, the turn fails with a configuration error.

## Batches, Schedules and Recovery

```python
from prompture.games import Entrant, double_round_robin, run_batch

entrants = [Entrant("a", lambda ctx: GameHarness(ctx.game, model_a, budget=ctx.budget, cancel=ctx.cancel)), ...]
schedule = double_round_robin([e.name for e in entrants], Connect4.sides)   # 4 entrants -> 12 games
report = run_batch(Connect4, entrants, schedule, store="runs/c4.jsonl", max_concurrency=2,
                   provider_limits={"openai": 1}, budget=GameBudget(max_cost=5.0))
```

- Game IDs derive from the schedule. Re-running with the same store skips finished games. A game that was running when the process stopped is recorded as `interrupted` with unknown usage before it is replayed. This prevents duplicate results; it cannot undo provider billing.
- Outcomes are `game_over`, `forfeit`, `move_cap`, `stopped` or `error` (final), and `cancelled` or `budget_exhausted` (replayed on resume). A crashing entrant ends only its own game, as `error` with no winner.
- `report.export()` rows include eligibility flags (`assisted`, `forfeit`, `failed`, `incompatible`) for a host leaderboard. No rating system is included.
- `calibrate(...)` is exploratory: it searches ordered opponent profiles for a 35–65% score with minimum evidence and game, time and budget ceilings. Engine settings are not human ratings.

## Experiments (Off by Default)

`strategy=Reconsider()` (one review when the chosen move carries a warning), `strategy=Proposals()` (two proposals plus a constrained selector) and `representation="text+image"` (a deterministic board picture alongside the unchanged text) are experiments. They use the same validation, deadlines, budgets, events and accounting, and their settings enter the fingerprint. Scorecards count review fixes *and* harms, proposal agreement, and image turns versus text fallbacks. None is known to improve play; `experiment_status` reports `live_pending` until a preregistered live comparison exists. PNG chess pictures need `cairosvg`; without it the turn records a text fallback, which is not an image trial.

## Host Integration Contract

A host application (UI, server, ladder) builds on these pieces:

| Host feature | Library hooks |
| --- | --- |
| Turn inspector, saved traces | `decision.record.to_dict()`, `JsonlTraceSink`, `read_jsonl_trace` |
| Live progress, reconnect, Stop button | `on_event` + `EventBuffer` (sequenced `seq` lets a client resume after a gap), `CancelToken` |
| Continuing a saved game with its plans | `game.dump_state`/`load_state` + `harness.memory.checkpoint`/`restore` |
| Rating pools and eligibility | `harness.fingerprint()`, `GameOutcome.eligibility()`, `failure_policy` |
| Tournaments, standings, calibration screens | `double_round_robin`, `run_batch` + `BatchStore`, `standings`, `calibrate` |
| Spending controls | a shared `GameBudget` passed through `MatchContext` |

Transport (SSE, WebSockets), storage, translations and screens are the host's choice; the library does not prescribe them.

## Writing a Game

Implement the `Game` protocol (`new_state`, `player`, `legal_moves`, `move_name`, `parse_move`, `describe_move`, `render`, `apply`, `is_over`, `winner`, `default_engine`). Optional capabilities unlock more:

- `sides = ("A", "B")` for schedules.
- `history(state) -> GameLine` for state identity and memory.
- `dump_state` / `load_state` for fixtures, checkpoints and batch start positions.
- `render_image(state) -> RenderedImage` for the image experiment.

A game without these still plays; features that need them say so explicitly.
