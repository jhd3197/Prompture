# Updating Prompture — a guide for coding agents

Hand this to an agent with:

> Update Prompture by following https://github.com/jhd3197/prompture/blob/main/docs/agent-update.md

## 1. Check

```bash
prompture check-update --json
```

One PyPI request, cached for 24 hours (`--force` skips the cache). Read
`installed`, `latest`, `update_available` and `highlights` (release notes
newer than the installed version). Offline is fine: it says so and exits 0.

If `update_available` is false, tell the user they're current and stop.

## 2. Tell the user what changes

Summarize `highlights` in a few bullets. Call out anything that sounds like a
breaking change and point to `BREAKING_CHANGES.md` in the repository.

## 3. Upgrade in the same environment

```bash
pip install -U prompture
```

Keep the extras the user already has (e.g. `pip install -U "prompture[web,mcp]"`).
Use the project's environment manager if it has one.

## 4. Re-check health and the skill

```bash
prompture doctor
prompture skill install --target project   # refreshes an installed skill in place
```

`skill install` only replaces a directory it created itself. If the user
installed it with `--target claude`, refresh that target instead.

## 5. Report

Two lines: the version change, and anything doctor now reports as not ok
(with its fix).

## Scheduled checks

For a cron job or scheduled task:

```bash
prompture watch            # offline doctor + update check
# exit 0 healthy, 1 something broken, 2 update available (only with --fail-on-update)
```

## The "mention once" rule

When working in a session, mention an available update once after a
substantial task, never mid-task, and never twice for the same version. In
Python: `prompture.infra.updates.should_announce(version)` and
`mark_announced(version)`.
