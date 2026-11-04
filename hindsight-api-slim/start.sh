#!/usr/bin/env bash
# Start Hindsight API (Agent Memory System).
# Safe to call repeatedly — checks if already running OR still starting.
#
# Usage:
#   ./start.sh          # foreground (logs to stdout)
#   ./start.sh --daemon # background (logs to $LOG_FILE)
#
# Exit codes of --daemon: 0 healthy (or already running), 1 the process died,
# 3 still starting when the wait ended — NOT a failure, the start goes on.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HOME/.local/share/hindsight/logs"
LOG_FILE="$LOG_DIR/hindsight.log"
PID_FILE="$LOG_DIR/hindsight.pid"
# Overridable for tests only (tests/test_start_script.py runs on a free port).
PORT="${HINDSIGHT_PORT:-8888}"

# Ensure uv is in PATH (crontab @reboot has minimal PATH)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

mkdir -p "$LOG_DIR"

# Already running?
if curl -sf --max-time 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "Hindsight already running on port $PORT"
    exit 0
fi

# Supply-chain protection: no auto-downloads from HuggingFace
export HF_HUB_OFFLINE=1

cd "$SCRIPT_DIR"

# A process start time identifies a PID beyond reuse: after a reboot the PID in
# a stale file may belong to something else, its start time will not match.
start_time() { ps -o lstart= -p "$1" 2>/dev/null || true; }

if [[ "${1:-}" == "--daemon" ]]; then
    # Still starting from an earlier call? /health answers only after the port is
    # bound — measured 2026-11-01 that was 150 s after launch on a cold restart —
    # so the check above cannot see a start in progress. The @reboot autostart
    # calls this script a second time when its own polling runs out; without this
    # guard that launched a second instance competing for the port and PostgreSQL.
    PID=""
    if [[ -f "$PID_FILE" ]]; then
        read -r OLD_PID OLD_START < <(tr '|' ' ' < "$PID_FILE") || true
        if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null \
            && [[ "$(start_time "$OLD_PID" | tr -s ' ' '_')" == "${OLD_START:-}" ]]; then
            PID="$OLD_PID"
            echo "Hindsight: already starting (PID $PID) — not starting a second instance"
        fi
    fi

    if [[ -z "$PID" ]]; then
        # APPEND, never truncate: the previous `> "$LOG_FILE"` wiped the log on
        # every restart, so the 2026-09-16 08:48 @reboot crash ("CRASHED after
        # 23s") left no trace once a later start.sh overwrote it — the cause is
        # unrecoverable. One-generation rotation keeps the file bounded: measured
        # 2026-09-16 ~162 KB per 2 h (~2 MB/day), so 20 MB ≈ 10 days of evidence.
        LOG_ROTATE_BYTES=$((20 * 1024 * 1024))
        if [[ -f "$LOG_FILE" ]] && [[ "$(stat -c %s "$LOG_FILE")" -gt "$LOG_ROTATE_BYTES" ]]; then
            mv -f "$LOG_FILE" "$LOG_FILE.1"
        fi
        echo "=== start.sh --daemon $(date '+%F %T') ===" >> "$LOG_FILE"
        nohup uv run hindsight-api >> "$LOG_FILE" 2>&1 &
        PID=$!
        printf '%s|%s\n' "$PID" "$(start_time "$PID" | tr -s ' ' '_')" > "$PID_FILE"
        echo "Hindsight starting (PID $PID, log: $LOG_FILE)"
    fi

    # How long to wait before reporting "still starting". Measured: 14 s on a warm
    # start (2026-09-05), 150 s on a cold restart (2026-11-01: embedded PostgreSQL
    # 31.6 s + embedding model 93.8 s, n=1). Running past it is friction, not loss:
    # the script then says STILL STARTING (exit 3) and the start goes on. The old
    # 15 s and then 90 s sat inside the measured spread and printed FAILED with
    # exit 1 for starts that were fine — the ready signal is the log line
    # "Uvicorn running on http://127.0.0.1:8888", equivalently /health = 200.
    WAIT_SECONDS="${HINDSIGHT_START_WAIT_SECONDS:-150}"
    for i in $(seq 1 "$WAIT_SECONDS"); do
        sleep 1
        if curl -sf --max-time 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then
            echo "Hindsight: OK (${i}s)"
            exit 0
        fi
        # Real crash: stop waiting instead of burning the full timeout.
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "Hindsight: CRASHED after ${i}s — check $LOG_FILE"
            exit 1
        fi
    done
    echo "Hindsight: STILL STARTING after ${WAIT_SECONDS}s (PID $PID alive) — not failed, check $LOG_FILE"
    exit 3
else
    exec uv run hindsight-api
fi
