# Skill: Run Tests

When the user asks to run tests, use the appropriate command based on what they want.

## Commands

| Intent | Command |
|--------|---------|
| All unit tests (default) | `pytest tests/ -x -q` |
| All tests including integration | `pytest tests/ --run-integration -x -q` |
| Specific test file | `pytest tests/{file}.py -x -q` |
| Specific test class | `pytest tests/{file}.py::{ClassName} -x -q` |
| Specific test function | `pytest tests/{file}.py::{ClassName}::{test_name} -x -q` |
| With verbose output | Add `-v` instead of `-q` |
| Show print output | Add `-s` |
| Skip integration if no creds | `TEST_SKIP_NO_CREDENTIALS=true pytest tests/ --run-integration -x -q` |
| Using test.py runner | `python test.py` |

## Flags Reference

- `-x` — Stop on first failure
- `-q` — Quiet output (just dots and summary)
- `-v` — Verbose (show each test name)
- `-s` — Show stdout/stderr (print statements)
- `--run-integration` — Include `@pytest.mark.integration` tests
- `-k "pattern"` — Run tests matching name pattern

## After Running

- If tests pass: Report the count (e.g. "137 passed, 1 skipped")
- If tests fail: Read the failure output, identify the root cause, and fix it
- Always run tests after making changes to any source file under `prompture/`
