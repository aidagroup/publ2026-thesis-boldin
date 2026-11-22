#!/usr/bin/env bash
# Update the server checkout to the latest pushed code, then re-run setup.
# One command for the lab JupyterHub terminal (no SSH, no root):
#
#   bash scripts/update_server.sh                       # current branch, then setup_server.sh
#   bash scripts/update_server.sh --branch main         # switch to another branch
#   bash scripts/update_server.sh --force               # discard local changes to tracked files
#   bash scripts/update_server.sh --no-setup            # only update the code
#   bash scripts/update_server.sh -- --smoke            # args after `--` go to setup_server.sh
#
# The server checkout is disposable (code is never edited there), and the remote
# history can be rewritten by force-pushes. So this is a HARD sync, never a
# `git pull`/merge: fetch, then point the local branch exactly at origin/<branch>.
# Tracked-file changes are refused unless --force. Untracked and ignored files
# (the `runs` symlink, logs, ...) are never touched: there is no `git clean`.
# Works from any checkout path; the recommended one on the lab server is ~/callosum
# (persistent $HOME), see docs/server-runbook.md.
set -euo pipefail

say() { printf '\n\033[1m>> %s\033[0m\n' "$*"; }
ok()  { printf '   \033[32m✓\033[0m %s\n' "$*"; }
warn(){ printf '   \033[33m!\033[0m %s\n' "$*"; }
die() { printf '   \033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
usage: bash scripts/update_server.sh [--branch <name>] [--force] [--no-setup] [-- <setup_server.sh args>]

  --branch <name>  switch to <name> (default: the currently checked-out branch)
  --force          discard uncommitted changes to tracked files instead of refusing
  --no-setup       only update the code, do not run scripts/setup_server.sh
  --               everything after this is passed to scripts/setup_server.sh (e.g. --smoke)
EOF
}

# Whole script lives in main(), called on the last line. Bash reads a script
# incrementally, and the checkout below may replace this very file; with a function
# the entire body is parsed before anything runs. If this file did change, we also
# re-exec the new version (guarded by CALLOSUM_UPDATE_REEXEC) so the rest of the
# run (setup) is driven by the new code, without fetching a second time.
main() {
  local script_dir repo branch="" force=0 run_setup=1
  local -a setup_args=() orig_args=("$@")

  while [ $# -gt 0 ]; do
    case "$1" in
      --branch)   [ $# -ge 2 ] && [ -n "$2" ] || die "--branch needs a name"; branch="$2"; shift 2 ;;
      --branch=*) branch="${1#--branch=}"; [ -n "$branch" ] || die "--branch needs a name"; shift ;;
      --force)    force=1; shift ;;
      --no-setup) run_setup=0; shift ;;
      -h|--help)  usage; exit 0 ;;
      --)         shift; setup_args=("$@"); break ;;
      *)          usage >&2; die "unknown argument: $1 (put setup_server.sh args after --)" ;;
    esac
  done

  # The repo is the git toplevel of this script's own directory, so it works from
  # any cwd and via an absolute path.
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null)" \
    || die "$script_dir is not inside a git checkout"
  cd "$repo"

  if [ "${CALLOSUM_UPDATE_REEXEC:-0}" != "1" ]; then
    sync_checkout "$branch" "$force" "$script_dir" "${orig_args[@]+"${orig_args[@]}"}"
  fi

  if [ "$run_setup" = 1 ]; then
    say "Running scripts/setup_server.sh${setup_args[*]:+ ${setup_args[*]}}"
    [ -f scripts/setup_server.sh ] || die "scripts/setup_server.sh not found in $repo"
    # Not `exec`: keep the exit code visible; `|| rc=$?` also survives `set -e`.
    local rc=0
    bash scripts/setup_server.sh ${setup_args[@]+"${setup_args[@]}"} || rc=$?
    exit "$rc"
  fi
  ok "Code updated; setup skipped (--no-setup)"
}

# Hard-sync the working tree to origin/<branch>; re-exec this script if it changed.
sync_checkout() {
  local branch="$1" force="$2" script_dir="$3"; shift 3
  local cur old new target remote_list dirty
  local -a tmo=() fetch_cfg=(-c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30)

  cur="$(git symbolic-ref --short -q HEAD || true)"
  if [ -z "$branch" ]; then
    [ -n "$cur" ] || die "detached HEAD: pass --branch <name> (e.g. --branch main)"
    branch="$cur"
  fi
  target="origin/$branch"
  old="$(git rev-parse HEAD 2>/dev/null || true)"

  # Refuse to lose tracked-file changes (staged or not) unless --force.
  dirty="$(git status --porcelain --untracked-files=no)"
  if [ -n "$dirty" ]; then
    if [ "$force" = 1 ]; then
      warn "--force: discarding uncommitted changes to tracked files:"
      printf '%s\n' "$dirty" | sed 's/^/     /'
    else
      printf '%s\n' "$dirty" | sed 's/^/     /' >&2
      die "uncommitted changes to tracked files (above). The server checkout is disposable; rerun with --force to discard them."
    fi
  fi

  # Time-bounded fetch: a stuck connection must not hang the terminal. `timeout`
  # may be missing (macOS); --foreground keeps credential prompts usable.
  if command -v timeout >/dev/null 2>&1; then
    if timeout --foreground 5 true >/dev/null 2>&1; then tmo=(timeout --foreground 120); else tmo=(timeout 120); fi
  elif command -v gtimeout >/dev/null 2>&1; then
    tmo=(gtimeout 120)
  fi

  say "Fetching origin"
  "${tmo[@]+"${tmo[@]}"}" git "${fetch_cfg[@]}" fetch --prune origin \
    || die "git fetch origin failed (network, allowlist or credentials?)"

  if ! git rev-parse --verify -q "refs/remotes/$target^{commit}" >/dev/null; then
    # A single-branch clone only fetches its own branch; ask for this one explicitly.
    "${tmo[@]+"${tmo[@]}"}" git "${fetch_cfg[@]}" fetch origin \
      "+refs/heads/$branch:refs/remotes/$target" >/dev/null 2>&1 || true
  fi
  if ! git rev-parse --verify -q "refs/remotes/$target^{commit}" >/dev/null; then
    remote_list="$(git for-each-ref --format='     %(refname:short)' refs/remotes/origin | grep -vE '^ +origin(/HEAD)?$' || true)"
    printf 'remote branches on origin:\n%s\n' "$remote_list" >&2
    die "origin/$branch does not exist (branch deleted or misspelled?)"
  fi

  say "Syncing to $target"
  if [ "$force" = 1 ]; then
    git checkout -q -f -B "$branch" "$target" || die "git checkout -B $branch $target failed"
  else
    git checkout -q -B "$branch" "$target" || die "git checkout -B $branch $target failed"
  fi
  new="$(git rev-parse HEAD)"

  if [ -n "$old" ] && [ "$old" != "$new" ]; then
    printf '   %s -> %s\n' "$(git log -1 --format='%h %s' "$old")" "$(git log -1 --format='%h %s' "$new")"
    if [ "$cur" = "$branch" ] && ! git merge-base --is-ancestor "$old" "$new"; then
      warn "history was rewritten upstream: the old commit is not an ancestor; hard-synced anyway"
    fi
    git --no-pager diff --stat "$old" "$new" | sed 's/^/   /'
  elif [ "$old" = "$new" ]; then
    ok "already up to date on $branch: $(git log -1 --format='%h %s' "$new")"
  else
    ok "now on $branch at $(git log -1 --format='%h %s' "$new")"
  fi

  # Self-update: if this script changed, run the new version for the remaining steps.
  if [ -n "$old" ] && [ "$old" != "$new" ] \
     && ! git diff --quiet "$old" "$new" -- scripts/update_server.sh; then
    say "update_server.sh itself changed; restarting the new version"
    export CALLOSUM_UPDATE_REEXEC=1
    exec bash "$script_dir/update_server.sh" "$@"
  fi
}

main "$@"
