#!/usr/bin/env bash
# Start Hindsight API (Agent Memory System).
# Safe to call repeatedly — checks if already running.
#
# Usage:
#   ./start.sh          # foreground (logs to stdout)
#   ./start.sh --daemon # background (logs to $LOG_FILE)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HOME/.local/share/hindsight/logs"
LOG_FILE="$LOG_DIR/hindsight.log"
PORT=8888

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

if [[ "${1:-}" == "--daemon" ]]; then
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
    echo "Hindsight starting (PID $PID, log: $LOG_FILE)"

    # Wait up to 90s for health.
    # Measured 2026-09-05: 14 s on a warm start, but MORE than 15 s on a cold
    # @reboot (embedded PostgreSQL comes up first, then the local reranker and
    # embedding models load). The old 15 s sat exactly in that spread, so the
    # script reported "FAILED" for a service that was starting fine — a false
    # alarm that misleads every later diagnosis.
    WAIT_SECONDS=90
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
    echo "Hindsight: FAILED after ${WAIT_SECONDS}s — check $LOG_FILE"
    exit 1
else
    exec uv run hindsight-api
fi
