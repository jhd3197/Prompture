#!/usr/bin/env bash
# Prompture dev launcher.
#
# Usage:
#   bash dev.sh              # run all checks once (default)
#   bash dev.sh watch-lint   # watch Python files and lint on changes
#   bash dev.sh check        # run all checks once (alias)

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-check}"
PY_PATH="$ROOT_DIR/prompture"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
DIM='\033[2m'
RESET='\033[0m'

run_all_checks() {
    local ts
    ts=$(date +%H:%M:%S)
    echo -e "${DIM}[$ts]${RESET} ${CYAN}Running checks...${RESET}"
    echo ""

    # --- ruff check ---
    echo -ne "  ${CYAN}ruff check      ${RESET}"
    if ruff check "$PY_PATH" 2>&1; then
        echo -e "  ${GREEN}ok${RESET}"
    else
        echo -e "  ${RED}fail${RESET}"
    fi

    # --- ruff format (auto-fix) ---
    echo -ne "  ${CYAN}ruff format     ${RESET}"
    if ruff format --check "$PY_PATH" >/dev/null 2>&1; then
        echo -e "${GREEN}ok${RESET}"
    else
        ruff format "$PY_PATH" >/dev/null 2>&1
        echo -e "${YELLOW}reformatted${RESET}"
    fi

    # --- mypy type check ---
    echo -ne "  ${CYAN}mypy            ${RESET}"
    local mypy_out
    mypy_out=$(mypy "$PY_PATH" 2>&1) && echo -e "${GREEN}ok${RESET}" || {
        local summary
        summary=$(echo "$mypy_out" | grep "^Found [0-9]" | head -1)
        if [ -n "$summary" ]; then
            echo -e "${RED}fail ($summary)${RESET}"
        else
            echo -e "${RED}fail${RESET}"
        fi
        echo "$mypy_out" | grep "error:" | while IFS= read -r line; do
            echo -e "    ${DIM}$line${RESET}"
        done
    }

    # --- pytest ---
    echo -ne "  ${CYAN}pytest          ${RESET}"
    local pytest_out
    pytest_out=$(pytest --tb=short -q 2>&1) && {
        local pass_line
        pass_line=$(echo "$pytest_out" | grep "passed" | head -1)
        if [ -n "$pass_line" ]; then
            echo -e "${GREEN}ok ($pass_line)${RESET}"
        else
            echo -e "${GREEN}ok${RESET}"
        fi
    } || {
        local fail_line
        fail_line=$(echo "$pytest_out" | grep -E "failed|error" | head -1)
        if [ -n "$fail_line" ]; then
            echo -e "${RED}fail ($fail_line)${RESET}"
        else
            echo -e "${RED}fail${RESET}"
        fi
        echo "$pytest_out" | grep -E "FAILED|ERROR" | while IFS= read -r line; do
            echo -e "    ${DIM}$line${RESET}"
        done
    }

    echo ""
}

# --- Single run for "check" mode ---
if [ "$MODE" = "check" ]; then
    echo -e "${CYAN}[dev]${RESET} Running all checks for Prompture"
    echo -e "${DIM}[dev]   Source: $PY_PATH${RESET}"
    echo ""
    run_all_checks
    exit 0
fi

# --- watch-lint mode ---
if [ "$MODE" != "watch-lint" ]; then
    echo "Usage: bash dev.sh [check|watch-lint]"
    exit 1
fi

echo -e "${CYAN}[dev]${RESET} Watching for lint + type + test errors"
echo -e "${DIM}[dev]   Source: $PY_PATH${RESET}"
echo -e "${DIM}[dev] Press Ctrl+C to stop.${RESET}"
echo ""

run_all_checks

# Try inotifywait (Linux/WSL), then fswatch (macOS), then poll
if command -v inotifywait &>/dev/null; then
    while inotifywait -r -e modify,create,delete \
          --include '\.py$' \
          "$PY_PATH" 2>/dev/null; do
        sleep 0.5
        run_all_checks
    done
elif command -v fswatch &>/dev/null; then
    fswatch -e ".*" -i "\\.py$" -r "$PY_PATH" | while read -r _; do
        sleep 0.5
        run_all_checks
    done
else
    echo -e "${YELLOW}[dev] No watcher found. Install inotify-tools or fswatch for live updates.${RESET}"
    echo -e "${YELLOW}[dev] Falling back to polling every 3s...${RESET}"
    LAST_HASH=""
    while true; do
        HASH=$(find "$PY_PATH" -name '*.py' -printf '%T@ %p\n' 2>/dev/null | md5sum | cut -d' ' -f1)
        if [ "$HASH" != "$LAST_HASH" ] && [ -n "$LAST_HASH" ]; then
            run_all_checks
        fi
        LAST_HASH="$HASH"
        sleep 3
    done
fi
