#!/usr/bin/env bash
# Bootstrap callosum on a Linux + NVIDIA (CUDA) machine: the lab training server
# (JupyterHub container, no SSH, no root, small $HOME) or any CUDA box.
# Idempotent and cheap to re-run: after every `git pull`, and again from scratch
# after the server wipes /tmp on restart.
#
#   bash scripts/setup_server.sh            # set up + verify
#   bash scripts/setup_server.sh --smoke    # ... and run the GPU smoke scripts
#
# Everything the project needs is pinned in uv.lock, so this never resolves
# dependencies; it installs the exact locked versions.
#
# Two layouts, chosen automatically:
#
#   scratch mode  ($HOME has < 30 GB free, or $JUPYTERHUB_USER / $CALLOSUM_SCRATCH is set)
#       Everything heavy goes to $CALLOSUM_SCRATCH (default /tmp/$USER-callosum):
#       venv, uv cache, uv-managed Python, HF cache, ManiSkill/SAPIEN assets, library
#       shims. This directory is assumed to be WIPED on server restart. Only results
#       survive: runs/ is a symlink to $CALLOSUM_RUNS (default ~/callosum-runs) when
#       the checkout itself is outside $HOME. A sourceable env file is written to
#       ~/.callosum-env.sh and hooked into ~/.bashrc.
#   normal mode   (plenty of space in $HOME)
#       uv defaults: .venv in the checkout, ~/.cache/uv. Nothing is moved.
#
# Network needs (allowlisted server): github.com (SAPIEN downloads its PhysX GPU
# library from a GitHub release on first use; uv downloads Python from GitHub),
# astral.sh, pypi.org, files.pythonhosted.org, download.pytorch.org.
# shellcheck disable=SC2016,SC2030,SC2031  # single quotes in printf are intentional; subshell probes
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_SMOKE=0
[ "${1:-}" = "--smoke" ] && RUN_SMOKE=1

say() { printf '\n\033[1m>> %s\033[0m\n' "$*"; }
ok()  { printf '   \033[32m✓\033[0m %s\n' "$*"; }
warn(){ printf '   \033[33m!\033[0m %s\n' "$*"; }
die() { printf '   \033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# NOTE: under `set -e` + `pipefail`, `X=$(a | head -1)` kills the script silently
# when `a` takes SIGPIPE or hits an unreadable path. Every such substitution below
# ends in `|| true` and is checked explicitly afterwards.

REPO_ROOT="$(pwd)"
USER_NAME="${USER:-$(id -un)}"
NEED_GB=30   # venv with torch + CUDA libs (~8 GB), uv cache, HF cache, assets: with margin

# Free space in whole GB on the filesystem holding $1 (empty if unknown).
free_gb() { df -Pk "$1" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}' || true; }

# Join non-empty arguments with ':' (an empty LD_LIBRARY_PATH entry means "cwd").
join_path() {
  local out="" p
  for p in "$@"; do
    if [ -n "$p" ]; then out="${out:+$out:}$p"; fi
  done
  printf '%s' "$out"
}

# ---------------------------------------------------------------- 1. platform
say "Platform"
[ "$(uname -s)" = "Linux" ] || die "Linux required (ManiSkill/SAPIEN GPU sim). On macOS use 'make dev' instead."
ok "$(uname -s) $(uname -m)"

if command -v nvidia-smi >/dev/null 2>&1; then
  ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  ok "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
  die "nvidia-smi not found — no NVIDIA driver visible. GPU sim will not work."
fi

# ----------------------------------------------------------------- 2. layout
say "Layout"
home_free="$(free_gb "$HOME")"
SCRATCH_MODE=0
if [ -n "${CALLOSUM_SCRATCH:-}" ] || [ -n "${JUPYTERHUB_USER:-}" ] \
   || { [ -n "$home_free" ] && [ "$home_free" -lt "$NEED_GB" ]; }; then
  SCRATCH_MODE=1
fi

ENV_FILE="$HOME/.callosum-env.sh"
if [ "$SCRATCH_MODE" = "1" ]; then
  SCRATCH="${CALLOSUM_SCRATCH:-/tmp/$USER_NAME-callosum}"
  mkdir -p "$SCRATCH" || die "cannot create $SCRATCH — set CALLOSUM_SCRATCH to a writable directory with >= ${NEED_GB} GB free"
  scratch_free="$(free_gb "$SCRATCH")"
  if [ -n "$scratch_free" ] && [ "$scratch_free" -lt "$NEED_GB" ]; then
    die "$SCRATCH has only ${scratch_free} GB free (< ${NEED_GB}); set CALLOSUM_SCRATCH elsewhere"
  fi
  WORK="$SCRATCH"
  ok "scratch mode: \$HOME has ${home_free:-?} GB free, heavy files go to $SCRATCH (${scratch_free:-?} GB free)"
  warn "assuming $SCRATCH is wiped on server restart: re-run this script then"

  export UV_PROJECT_ENVIRONMENT="$SCRATCH/venv"
  export UV_CACHE_DIR="$SCRATCH/uv-cache"
  export UV_PYTHON_INSTALL_DIR="$SCRATCH/python"
  export HF_HOME="$SCRATCH/hf"
  # ManiSkill v3.0.1 reads MS_ASSET_DIR (mani_skill/__init__.py); default ~/.maniskill.
  export MS_ASSET_DIR="$SCRATCH/maniskill"
  # pip, matplotlib, torch hub, ...: anything that honours XDG leaves $HOME alone.
  export XDG_CACHE_HOME="$SCRATCH/cache"
  mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$MS_ASSET_DIR" "$XDG_CACHE_HOME"
  VENV_DIR="$UV_PROJECT_ENVIRONMENT"
  ok "venv         -> $VENV_DIR"
  ok "uv cache     -> $UV_CACHE_DIR"
  ok "HF_HOME      -> $HF_HOME"
  ok "MS_ASSET_DIR -> $MS_ASSET_DIR"

  # SAPIEN hardcodes Path.home()/.sapien (PhysX GPU library, ~240 MB) with no env
  # override, so redirect it with a symlink.
  link="$HOME/.sapien"; target="$SCRATCH/sapien"
  mkdir -p "$target"
  if [ -L "$link" ]; then
    ln -sfn "$target" "$link"          # also repairs a link left dangling by a wipe
  else
    if [ -e "$link" ]; then cp -a "$link/." "$target/" && rm -rf "$link"; fi
    ln -s "$target" "$link"
  fi
  ok "\$HOME/.sapien -> $target"
else
  WORK="$HOME/.local/share/callosum"   # only used for library shims, if any are needed
  mkdir -p "$WORK"
  VENV_DIR="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"
  ok "normal mode: \$HOME has ${home_free:-?} GB free, uv defaults (venv at $VENV_DIR)"
fi
VENV_PY="$VENV_DIR/bin/python"

# --------------------------------------------------------------------- 3. uv
# No root: uv goes to ~/.local/bin (~50 MB). It survives a scratch wipe, so the
# installer only runs the first time.
say "uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  if ! { curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh; }; then
    warn "astral.sh installer failed; falling back to 'pip install --user uv' (PyPI)"
    python3 -m pip install --user --quiet uv || die "could not install uv (astral.sh and PyPI both failed)"
  fi
  command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH — check ~/.local/bin"
fi
ok "$(uv --version)"

# ------------------------------------------------------------- 4. environment
# Python 3.12: SAPIEN publishes no wheels for 3.13+.
say "Environment (Python 3.12 + locked deps)"
uv python install 3.12 \
  || warn "uv python install failed (github.com unreachable?); uv will use a system 3.12 if it finds one"
# --frozen: install exactly uv.lock, never re-resolve. If this errors, the lock
# is out of sync with pyproject.toml — fix that on the dev machine, not here.
if ! uv sync --frozen --extra sim --extra train --extra dev; then
  warn "if the failure names download-r2.pytorch.org (the CDN behind download.pytorch.org),"
  warn "ask the server owner to open it; do NOT install torch from PyPI instead:"
  warn "its default Linux build targets CUDA 13, which driver 570 (CUDA <= 12.8) cannot run"
  die "uv sync failed"
fi
ok "synced from uv.lock"
[ -x "$VENV_PY" ] || die "expected venv python at $VENV_PY"

# ------------------------------------------------- 4a. libcuda.so (PhysX GPU)
# SAPIEN's physx.enable_gpu() does ctypes.CDLL("libcuda.so"): the UNVERSIONED name.
# Container runtimes inject only libcuda.so.1 from the host driver; the bare name
# normally comes from a driver -devel package. torch links libcuda.so.1 directly,
# so torch can see the GPU while PhysX cannot. Only create the shim if needed.
say "libcuda.so"
SHIM_DIR=""
if "$VENV_PY" -c 'import ctypes; ctypes.CDLL("libcuda.so")' 2>/dev/null; then
  ok "libcuda.so resolves"
else
  LIBCUDA=""
  if command -v ldconfig >/dev/null 2>&1; then
    LIBCUDA="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/ {print $NF; exit}' || true)"
  fi
  if [ -z "$LIBCUDA" ] || [ ! -e "$LIBCUDA" ]; then
    for c in /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib64/libcuda.so.1 \
             /usr/local/nvidia/lib64/libcuda.so.1 /usr/local/cuda/compat/libcuda.so.1; do
      if [ -e "$c" ]; then LIBCUDA="$c"; break; fi
    done
  fi
  [ -n "$LIBCUDA" ] || die "libcuda.so.1 not found — is this container GPU-enabled? (nvidia-smi works?)"
  SHIM_DIR="$WORK/lib"
  mkdir -p "$SHIM_DIR"
  ln -sfn "$LIBCUDA" "$SHIM_DIR/libcuda.so"
  export LD_LIBRARY_PATH
  LD_LIBRARY_PATH="$(join_path "$SHIM_DIR" "${LD_LIBRARY_PATH:-}")"
  ok "linked $SHIM_DIR/libcuda.so -> $LIBCUDA"
  warn "LD_LIBRARY_PATH is read at process start: run from a terminal that sourced $ENV_FILE;"
  warn "a notebook kernel started earlier will not see it"
fi

# ------------------------------------------------ 4b. PhysX GPU library (SAPIEN)
# On first enable_gpu() SAPIEN downloads libPhysXGpu_64.so from a GitHub release
# into ~/.sapien/physx/<version>/. Do it now so a network problem shows up here
# and not three frames inside the first gym.make(). Best effort: not fatal.
say "PhysX GPU library"
if "$VENV_PY" -c 'import sapien; sapien.physx.enable_gpu()' 2>"$WORK/physx.log"; then
  ok "physx.enable_gpu() works ($(du -sh "$HOME/.sapien" 2>/dev/null | cut -f1) in ~/.sapien)"
else
  warn "physx.enable_gpu() failed (last lines of $WORK/physx.log):"
  tail -n 5 "$WORK/physx.log" | sed 's/^/     /'
  warn "if the download from github.com failed: fetch linux-so.zip from the URL in that message"
  warn "elsewhere, upload it, and unzip it into ~/.sapien/physx/<version>/"
fi

# ---------------------------------------------------------- 4c. Vulkan device
# ManiSkill cannot build ANY environment without a Vulkan render device, even
# with obs_mode="state": SAPIEN's URDF loader constructs RenderMaterial()
# unconditionally (render_backend="none" removes the renderer from the scene but
# not the global device). Try the stock configuration first; only if it fails,
# search for a working ICD manifest + loader pair by actually constructing a
# RenderMaterial with each. Known failure on this image (probed 2026-08-30): the
# system libvulkan (1.3.275) is too old for the 570.x NVIDIA ICD, so
# vkCreateInstance returns NULL; a current loader from conda-forge fixes it.
say "Vulkan render device"
ICD=""; VKLIB=""
render_ok() {  # $1 = ICD manifest ("" = leave default), $2 = extra loader dir ("" = none)
  (
    if [ -n "$1" ]; then export VK_ICD_FILENAMES="$1"; fi
    ld="$(join_path "$2" "${LD_LIBRARY_PATH:-}")"
    if [ -n "$ld" ]; then export LD_LIBRARY_PATH="$ld"; fi
    "$VENV_PY" -c 'from sapien.render import RenderMaterial; RenderMaterial()' >/dev/null 2>&1
  )
}
pick() {  # $1 manifest, $2 loader dir, $3 label -> sets ICD / VKLIB on success
  [ -e "$1" ] || return 1
  render_ok "$1" "$2" || return 1
  ICD="$1"; VKLIB="$2"; ok "render device via $3"
}

if render_ok "" ""; then
  ok "stock configuration works (no Vulkan override needed)"
else
  warn "stock Vulkan configuration has no render device; searching for a working one"
  MESA="$WORK/mesa"
  found=0
  for icd in /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json; do
    if pick "$icd" "" "$(basename "$icd") + system loader"; then found=1; break; fi
  done
  if [ "$found" = "0" ]; then
    if [ ! -e "$MESA/lib/libvulkan.so.1" ] && command -v conda >/dev/null 2>&1; then
      warn "installing a current Vulkan loader (+ lavapipe) with conda into $MESA (needs conda-forge)"
      # conda's package cache defaults to ~/.conda/pkgs: keep it off the small $HOME.
      CONDA_PKGS_DIRS="$WORK/conda-pkgs" \
        conda create -y -q -p "$MESA" -c conda-forge mesalib vulkan-tools >/dev/null 2>&1 \
        || warn "conda install failed (conda-forge not reachable from the server?)"
    fi
    if [ -e "$MESA/lib/libvulkan.so.1" ]; then
      # A directory holding ONLY the loader: prepending conda's whole lib/ would
      # also shadow libstdc++/libgcc and can break torch.
      mkdir -p "$WORK/vklib"
      ln -sfn "$MESA/lib/libvulkan.so.1" "$WORK/vklib/libvulkan.so.1"
      for icd in /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json; do
        if pick "$icd" "$WORK/vklib" "$(basename "$icd") + current loader"; then found=1; break; fi
      done
      if [ "$found" = "0" ]; then
        for icd in "$MESA"/share/vulkan/icd.d/*.json; do
          if pick "$icd" "$WORK/vklib" "$(basename "$icd") (software rasteriser: slow, fine for state training)"; then
            found=1; break
          fi
        done
      fi
    fi
  fi
  [ "$found" = "1" ] || die "no Vulkan ICD gives a render device — see docs/server-runbook.md (Vulkan)"
  export VK_ICD_FILENAMES="$ICD"
  export LD_LIBRARY_PATH
  LD_LIBRARY_PATH="$(join_path "$VKLIB" "${LD_LIBRARY_PATH:-}")"
fi

# ------------------------------------------------------------ 5. env file, runs/
# One file that any NEW terminal sources. ~/.bashrc only reaches shells started
# after setup (not `bash script.sh`, not an already-running kernel).
say "Environment file"
{
  echo "# Generated by scripts/setup_server.sh. Source it in every new terminal:"
  echo "#     source ~/.callosum-env.sh"
  echo "# Without it, uv would not know where the venv lives and would build a second one."
  printf 'export PATH="$HOME/.local/bin:$PATH"\n'
  if [ "$SCRATCH_MODE" = "1" ]; then
    printf 'export CALLOSUM_SCRATCH=%q\n' "$SCRATCH"
    printf 'export CALLOSUM_REPO=%q\n' "$REPO_ROOT"
    printf 'export UV_PROJECT_ENVIRONMENT=%q\n' "$UV_PROJECT_ENVIRONMENT"
    printf 'export UV_CACHE_DIR=%q\n' "$UV_CACHE_DIR"
    printf 'export UV_PYTHON_INSTALL_DIR=%q\n' "$UV_PYTHON_INSTALL_DIR"
    printf 'export HF_HOME=%q\n' "$HF_HOME"
    printf 'export MS_ASSET_DIR=%q\n' "$MS_ASSET_DIR"
    printf 'export XDG_CACHE_HOME=%q\n' "$XDG_CACHE_HOME"
  fi
  if [ -n "$SHIM_DIR" ]; then
    printf 'export LD_LIBRARY_PATH=%q"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n' "$SHIM_DIR"
  fi
  if [ -n "$VKLIB" ]; then
    printf 'export LD_LIBRARY_PATH=%q"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n' "$VKLIB"
  fi
  if [ -n "$ICD" ]; then
    printf 'export VK_ICD_FILENAMES=%q\n' "$ICD"
  fi
  if [ "$SCRATCH_MODE" = "1" ]; then
    printf '[ -d %q ] || echo "callosum: scratch was wiped; re-run: bash scripts/setup_server.sh (see docs/server-runbook.md)" >&2\n' "$VENV_DIR"
  fi
} > "$ENV_FILE"
ok "wrote $ENV_FILE"

if [ "$SCRATCH_MODE" = "1" ]; then
  # Hook it into new interactive terminals (idempotent; the file may be stale or
  # missing after a wipe, hence the guard).
  if ! grep -q 'callosum-env.sh' "$HOME/.bashrc" 2>/dev/null; then
    printf '\n# callosum (added by scripts/setup_server.sh)\n[ -f "$HOME/.callosum-env.sh" ] && . "$HOME/.callosum-env.sh"\n' >> "$HOME/.bashrc"
    ok "hooked into ~/.bashrc"
  else
    ok "already hooked into ~/.bashrc"
  fi

  # An accidental in-repo .venv (uv run without the env file) wastes scratch space.
  if [ -d "$REPO_ROOT/.venv" ] && [ "$VENV_DIR" != "$REPO_ROOT/.venv" ]; then
    warn "removing stray $REPO_ROOT/.venv (created by uv run without the env file)"
    rm -rf "$REPO_ROOT/.venv"
  fi

  # Results must survive a restart, and scratch does not. If the checkout lives
  # outside $HOME, keep runs/ in $HOME and symlink it in. If the checkout is in
  # $HOME already, runs/ is as persistent as it can get.
  say "runs/ (results)"
  case "$REPO_ROOT" in
    "$HOME"/*) ok "checkout is under \$HOME; runs/ stays a plain directory" ;;
    *)
      RUNS_STORE="${CALLOSUM_RUNS:-$HOME/callosum-runs}"
      mkdir -p "$RUNS_STORE"
      if [ -L "$REPO_ROOT/runs" ]; then
        ln -sfn "$RUNS_STORE" "$REPO_ROOT/runs"
      else
        # A real runs/ from before the symlink: keep its contents, then link.
        if [ -d "$REPO_ROOT/runs" ]; then
          cp -a "$REPO_ROOT/runs/." "$RUNS_STORE/" && rm -rf "$REPO_ROOT/runs"
        fi
        ln -s "$RUNS_STORE" "$REPO_ROOT/runs"
      fi
      ok "runs -> $RUNS_STORE ($(du -sh "$RUNS_STORE" 2>/dev/null | cut -f1) used, $(free_gb "$RUNS_STORE") GB free)"
      warn "\$HOME is small: prune checkpoints you do not need and download results (JupyterHub file browser)"
      ;;
  esac
fi

# ----------------------------------------------------------- 6. verification
say "Verification"
uv run python - <<'PY'
import sys

fail = []

# --- torch / CUDA -----------------------------------------------------------
import torch
print(f"   torch {torch.__version__} (CUDA build {torch.version.cuda})")
if not torch.cuda.is_available():
    fail.append("torch.cuda.is_available() is False — driver/CUDA mismatch")
else:
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   \033[32m✓\033[0m {name} | compute {major}.{minor} | {total:.1f} GiB")

    # Blackwell (sm_120) needs a CUDA >= 12.8 build. Not triggered on the lab
    # A100 (sm_80); kept so the script stays correct on any newer GPU.
    cuda_ver = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
    if major >= 12 and cuda_ver < (12, 8):
        fail.append(f"GPU is compute {major}.{minor} but torch is a CUDA {torch.version.cuda} build (<12.8)")

    # Real kernel launch — is_available() alone can pass on a broken setup.
    try:
        x = torch.randn(1024, 1024, device="cuda")
        torch.cuda.synchronize()
        assert torch.isfinite(x @ x).all().item()
        print("   \033[32m✓\033[0m CUDA matmul works")
    except Exception as e:  # noqa: BLE001
        fail.append(f"CUDA kernel launch failed: {e}")

# --- ManiSkill / SAPIEN -----------------------------------------------------
import mani_skill
print(f"   mani_skill {getattr(mani_skill, '__version__', '?')}")
import callosum.robots.so101_parallel_gripper  # noqa: F401  (registers the so101_pg agent)
from mani_skill.agents.registration import REGISTERED_AGENTS

if "so101_pg" not in REGISTERED_AGENTS:
    fail.append("so101_pg agent did not register")
else:
    print("   \033[32m✓\033[0m SO-ARM101 + parallel gripper agent registered (uid='so101_pg')")

# --- our environments (need mani_skill, so untestable on macOS) --------------
import callosum
import callosum.envs.face_turn  # noqa: F401  (registers FaceTurn-v0)
import callosum.envs.two_so101_base  # noqa: F401  (registers TwoSO101-v0)
from mani_skill.utils.registration import REGISTERED_ENVS

for env_id in ("TwoSO101-v0", "FaceTurn-v0"):
    if env_id not in REGISTERED_ENVS:
        fail.append(f"{env_id} did not register")
    else:
        print(f"   \033[32m✓\033[0m {env_id} registered")
print(f"   callosum {callosum.__version__}")

if fail:
    print("\n\033[31mFAILED:\033[0m")
    for f in fail:
        print(f"   ✗ {f}")
    sys.exit(1)
PY

# ------------------------------------------------------------- 7. smoke tests
if [ "$RUN_SMOKE" = "1" ]; then
  say "GPU smoke tests (the checks that could not run on macOS)"
  echo "--- scripts/smoke_env.py: two SO-ARM101 arms, per-agent obs/actions, arm reach ---"
  uv run python scripts/smoke_env.py
  echo
  echo "--- scripts/smoke_face_turn.py: turntable articulation, success detection ---"
  uv run python scripts/smoke_face_turn.py
else
  say "Next"
  echo "   Run the pending GPU checks:  bash scripts/setup_server.sh --smoke"
  echo "   (or individually: uv run python scripts/smoke_env.py)"
fi

say "Done"
if [ "$SCRATCH_MODE" = "1" ]; then
  echo "   Checkout : $REPO_ROOT"
  echo "   Scratch  : $SCRATCH   (assumed wiped on server restart: re-run this script)"
  echo "   Results  : $REPO_ROOT/runs"
  echo "   In a new terminal 'source ~/.callosum-env.sh' (automatic via ~/.bashrc), then use"
  echo "   'uv run <cmd>'. Long runs: see docs/server-runbook.md (detached with setsid nohup)."
else
  echo "   Use 'uv run <cmd>' — or activate with: source $VENV_DIR/bin/activate"
fi
