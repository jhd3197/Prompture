---
name: run-tests
description: Run Prompture's test suite with the correct pytest flags. Supports unit-only, integration, single file, single test, verbose, and credential-skip modes. Use when running or debugging tests.
metadata:
  author: prompture
  version: "1.0"
---

# Run Tests

## Commands

| Intent | Command |
|--------|---------|
| All unit tests | `pytest tests/ -x -q` |
| Include integration tests | `pytest tests/ --run-integration -x -q` |
| Specific file | `pytest tests/{file}.py -x -q` |
| Specific class | `pytest tests/{file}.py::{Class} -x -q` |
| Specific test | `pytest tests/{file}.py::{Class}::{test} -x -q` |
| Verbose output | Replace `-q` with `-v` |
| Show print output | Add `-s` |
| Pattern match | Add `-k "pattern"` |
| Skip missing credentials | `TEST_SKIP_NO_CREDENTIALS=true pytest tests/ --run-integration -x -q` |
| Legacy runner | `python test.py` |

## Flags

- `-x` stop on first failure
- `-q` quiet (dots + summary)
- `-v` verbose (each test name)
- `-s` show stdout/stderr
- `--run-integration` include `@pytest.mark.integration` tests

## After Running

- **Pass**: report count (e.g. "137 passed, 1 skipped")
- **Fail**: read the failure output, identify root cause, fix it
- Always run after modifying any file under `prompture/`
