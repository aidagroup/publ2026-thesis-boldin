#!/usr/bin/env bash
# Auto-restart loop for IPPO training on preemptible cloud VMs (Nebius).
# Usage: bash scripts/ops/cloud_resume_loop.sh <run_name> [ippo args...]
#
# Start the run once with the normal trainer command (see
# docs/server-runbook.md) so that runs/<run_name>/latest.pt exists, then
# keep it going under this loop (e.g. inside tmux/screen).
#
# Behavior:
#   trainer exit 0   -> clean finish (total timesteps reached), loop exits
#   non-zero exit    -> log with timestamp, sleep 30 s, restart with --resume
#
# SIGTERM/SIGINT are forwarded to the child so the trainer can finish the
# current iteration and flush latest.pt before the VM goes away.

set -u

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_name> [ippo args...]" >&2
    exit 2
fi

RUN_NAME="$1"
shift

# Everything is relative to the repo root (uv run, runs/).
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_DIR="runs/${RUN_NAME}"
LOG_FILE="${RUN_DIR}/resume_loop.log"

mkdir -p "$RUN_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

CHILD_PID=""

on_signal() {
    log "signal received, forwarding to trainer (pid ${CHILD_PID:-none})"
    if [ -n "$CHILD_PID" ]; then
        # Direct PID first: in non-interactive bash a background job is not
        # a process-group leader, so a group kill alone may not reach it.
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        kill -TERM "-$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    exit 0
}
trap on_signal TERM INT

log "resume loop started: run=${RUN_NAME} args=$*"

while true; do
    log "starting trainer"
    uv run python -m callosum.experiments.arm.trainer --resume "runs/${RUN_NAME}" "$@" &
    CHILD_PID=$!
    wait "$CHILD_PID"
    CODE=$?
    CHILD_PID=""
    if [ "$CODE" -eq 0 ]; then
        log "trainer finished cleanly (exit 0); leaving loop"
        exit 0
    fi
    log "trainer exited with code ${CODE}; restarting in 30 s"
    sleep 30
done
