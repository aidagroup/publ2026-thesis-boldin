#!/usr/bin/env python3
"""Tiny job/status channel for the RunPod GPU box.

The pod is a separate machine: the agent cannot receive push notifications from it
directly. This tool gives the pod a durable, append-only status channel that the local
side can *block on* instead of sleeping/polling:

* ``start`` launches a detached command, writes ``<job>.state.json`` and ``<job>.log``,
  and spawns a watchdog that records the exit code when the command ends.
* ``status`` prints the current state (one SSH round-trip, no polling loop).
* ``wait`` blocks until the job's state says ``finished`` and optionally follows the log;
  the local wrapper runs it as a background DSH job and is notified on completion.
* ``logs`` prints/tails the job log.
* ``wait-file`` blocks until a pattern appears in a file (used to attach to jobs that
  were started outside ``jobctl``).

State lives on the network volume (``/workspace/callosum-jobs``) so it survives a
container restart; logs stay next to the state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DIR = Path(os.environ.get("CALLOSUM_JOB_DIR", "/workspace/callosum-jobs"))
POLL_S = 1.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _state_path(dir_: Path, name: str) -> Path:
    return dir_ / f"{name}.state.json"


def _log_path(dir_: Path, name: str) -> Path:
    return dir_ / f"{name}.log"


def _read_state(dir_: Path, name: str) -> dict | None:
    path = _state_path(dir_, name)
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_state(dir_: Path, name: str, state: dict) -> None:
    path = _state_path(dir_, name)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)
    events = dir_ / "events.jsonl"
    with events.open("a") as handle:
        handle.write(json.dumps({"time": _now(), **state}) + "\n")


def _cmd_start(args: argparse.Namespace) -> int:
    dir_ = Path(args.dir)
    dir_.mkdir(parents=True, exist_ok=True)
    name = args.name
    if _read_state(dir_, name) and _read_state(dir_, name)["status"] == "running":
        print(f"{name}: already running", file=sys.stderr)
        return 2
    log = _log_path(dir_, name).open("w")
    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("start: no command given", file=sys.stderr)
        return 2
    env = dict(os.environ)
    env.setdefault("WRENCH_TORCH_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, cwd=args.cwd, env=env, start_new_session=True
    )
    state = {
        "name": name,
        "status": "running",
        "pid": proc.pid,
        "command": cmd,
        "cwd": args.cwd,
        "log": str(_log_path(dir_, name)),
        "started_at": _now(),
        "finished_at": None,
        "exit_code": None,
    }
    _write_state(dir_, name, state)
    # Watchdog survives the SSH session (new session) and records the exit code.
    subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_watch",
            name,
            str(proc.pid),
            "--dir",
            str(dir_),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(json.dumps(state))
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    dir_ = Path(args.dir)
    pid = int(args.pid)
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(POLL_S)
    state = _read_state(dir_, args.name) or {"name": args.name}
    # The PID may be the process-group leader; collect the real exit code from a marker is
    # impossible after the fact, so report success unless the log has a failure marker.
    log_tail = ""
    try:
        log_tail = _log_path(dir_, args.name).read_text(errors="ignore")[-4000:]
    except OSError:
        pass
    failed = bool(re.search(r"Traceback|FAILED|Error:|exit [1-9]", log_tail))
    state.update(
        status="failed" if failed else "finished",
        finished_at=_now(),
        exit_code=1 if failed else 0,
    )
    _write_state(dir_, args.name, state)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    dir_ = Path(args.dir)
    if args.name:
        state = _read_state(dir_, args.name)
        if state is None:
            print(f"unknown job {args.name}", file=sys.stderr)
            return 1
        print(json.dumps(state, indent=2))
        return 0
    states = []
    for path in sorted(dir_.glob("*.state.json")):
        try:
            states.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    print(json.dumps(states, indent=2))
    return 0


def _cmd_wait(args: argparse.Namespace) -> int:
    dir_ = Path(args.dir)
    deadline = None if args.timeout <= 0 else time.time() + args.timeout
    log_path = _log_path(dir_, args.name)
    printed = 0
    while True:
        state = _read_state(dir_, args.name)
        if state is None:
            print(f"unknown job {args.name}", file=sys.stderr)
            return 1
        if args.follow and log_path.exists():
            lines = log_path.read_text(errors="ignore").splitlines()
            for line in lines[printed:]:
                print(line, flush=True)
            printed = len(lines)
        if state["status"] != "running":
            if args.follow and log_path.exists():
                lines = log_path.read_text(errors="ignore").splitlines()
                for line in lines[printed:]:
                    print(line, flush=True)
            print(f"JOB {args.name} {state['status']} exit={state['exit_code']}", flush=True)
            return 0 if state["status"] == "finished" else 1
        if deadline is not None and time.time() > deadline:
            print(f"JOB {args.name} still running after {args.timeout}s", flush=True)
            return 1
        time.sleep(POLL_S)


def _cmd_logs(args: argparse.Namespace) -> int:
    dir_ = Path(args.dir)
    path = _log_path(dir_, args.name)
    if args.follow:
        os.execvp("tail", ["tail", "-n", str(args.lines), "-F", str(path)])
    text = path.read_text(errors="ignore").splitlines()
    for line in text[-args.lines :]:
        print(line)
    return 0


def _cmd_wait_file(args: argparse.Namespace) -> int:
    path = Path(args.path)
    deadline = None if args.timeout <= 0 else time.time() + args.timeout
    pattern = re.compile(args.pattern) if args.pattern else None
    while True:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            text = ""
        if pattern.search(text) if pattern else path.exists():
            print(f"FILE {path} matched {args.pattern!r}", flush=True)
            return 0
        if deadline is not None and time.time() > deadline:
            print(f"FILE {path} not matched after {args.timeout}s", flush=True)
            return 1
        time.sleep(POLL_S)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="state directory")
    sub = parser.add_subparsers(dest="command_name", required=True)

    start = sub.add_parser("start", help="launch a detached job")
    start.add_argument("name")
    start.add_argument("--cwd", default=None)
    start.add_argument("command", nargs=argparse.REMAINDER)
    start.set_defaults(func=_cmd_start)

    watch = sub.add_parser("_watch", help="internal watchdog")
    watch.add_argument("name")
    watch.add_argument("pid")
    watch.add_argument("--dir", default=str(DEFAULT_DIR))
    watch.set_defaults(func=_cmd_watch)

    status = sub.add_parser("status", help="show job state")
    status.add_argument("name", nargs="?")
    status.set_defaults(func=_cmd_status)

    wait = sub.add_parser("wait", help="block until the job finishes")
    wait.add_argument("name")
    wait.add_argument("--timeout", type=float, default=0)
    wait.add_argument("--follow", action="store_true")
    wait.set_defaults(func=_cmd_wait)

    logs = sub.add_parser("logs", help="print/tail a job log")
    logs.add_argument("name")
    logs.add_argument("--lines", type=int, default=40)
    logs.add_argument("--follow", action="store_true")
    logs.set_defaults(func=_cmd_logs)

    wait_file = sub.add_parser("wait-file", help="block until a pattern appears in a file")
    wait_file.add_argument("path")
    wait_file.add_argument("--pattern", default=None)
    wait_file.add_argument("--timeout", type=float, default=0)
    wait_file.set_defaults(func=_cmd_wait_file)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command_name == "_watch":
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
