#!/usr/bin/env bash
# Local wrapper around the pod-side jobctl: a status/log channel without polling loops.
#
# Instead of `sleep N && ssh ...`, launch a job once and then block on its status:
#
#   scripts/ops/pod_job.sh install                       # copy jobctl.py to the pod
#   scripts/ops/pod_job.sh start my-run -- python -m callosum...train ...
#   scripts/ops/pod_job.sh status my-run                  # one SSH round-trip
#   scripts/ops/pod_job.sh wait my-run --follow           # blocks until finished, streams log
#   scripts/ops/pod_job.sh wait-file /root/x.log --pattern DONE
#
# `wait` is meant to be started as a DSH background job: it keeps one SSH stream open and
# exits exactly when the pod records `finished`, so the harness notifies us on completion
# and no agent-side sleep/polling is needed.
#
# Connection comes from POD_HOST / POD_PORT / POD_KEY (defaults: the current A40 pod).
set -euo pipefail

HOST="${POD_HOST:-194.68.245.206}"
PORT="${POD_PORT:-22009}"
KEY="${POD_KEY:-$HOME/.ssh/runpod_gpu}"
REMOTE="root@${HOST}"
JOBCTL="/workspace/callosum-jobs/jobctl.py"
LOCAL_JOBCTL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/jobctl.py"

SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no
          -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -p "$PORT")
SCP_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no
          -o UserKnownHostsFile=/dev/null -P "$PORT")

usage() {
    sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

remote_jobctl() {
    # Quote each argument for the remote shell (macOS bash has printf %q).
    local quoted=""
    local arg
    for arg in "$@"; do
        quoted+=" $(printf '%q' "$arg")"
    done
    ssh "${SSH_OPTS[@]}" "$REMOTE" "python3 ${JOBCTL}${quoted}"
}

cmd="${1:-}"
[ -z "$cmd" ] && usage 1
shift || true

case "$cmd" in
    install)
        ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p /workspace/callosum-jobs"
        scp "${SCP_OPTS[@]}" "$LOCAL_JOBCTL" "${REMOTE}:${JOBCTL}"
        echo "installed ${JOBCTL}"
        ;;
    start)
        name="${1:?start needs a job name}"
        shift
        # Accept an optional literal `--` before the command.
        [ "${1:-}" = "--" ] && shift
        remote_jobctl start "$name" -- "$@"
        ;;
    status|logs)
        remote_jobctl "$cmd" "$@"
        ;;
    wait|wait-file)
        # Long waits may outlive an SSH session (RunPod NAT/sshd drops). Reconnect and
        # re-run the blocking remote wait; the job state lives on the volume, so no work
        # is lost. Only SSH failures (255) are retried; a finished/failed job status is
        # returned to the caller.
        if [ "$cmd" = "wait-file" ]; then
            path="${1:?wait-file needs a path}"
            shift
            remote_args=(wait-file "$path" "$@")
        else
            remote_args=(wait "$@")
        fi
        attempt=0
        while [ "$attempt" -lt 240 ]; do
            if output=$(remote_jobctl "${remote_args[@]}" 2>&1); then
                printf '%s\n' "$output"
                exit 0
            else
                code=$?
                printf '%s\n' "$output"
                if [ "$code" -ne 255 ]; then
                    exit "$code"
                fi
                attempt=$((attempt + 1))
                sleep 5
            fi
        done
        echo "wait: too many SSH reconnects" >&2
        exit 1
        ;;
    *)
        usage 1
        ;;
esac
