#!/usr/bin/env bash
# Bootstrap callosum on a Linux + NVIDIA (CUDA) machine: the lab training server
# (JupyterHub container, no SSH, no root, persistent 100 GB $HOME) or any CUDA box.
# Idempotent and cheap to re-run: after every `git pull`, and again after a
# container restart (nothing heavy is lost then, see the layout below).
#
#   bash scripts/setup_server.sh            # set up + verify
#   bash scripts/setup_server.sh --smoke    # ... and run the GPU smoke scripts
#
# Everything the project needs is pinned in uv.lock, so this never resolves
# dependencies; it installs the exact locked versions.
#
# Two layouts, chosen automatically (same sub-directories in both):
#
#   home layout   (default)
#       Everything heavy is PERSISTENT, under $CALLOSUM_DATA (default ~/.callosum):
#         venv/  uv-cache/  python/  hf/  maniskill/  cache/ (XDG)  mesa/ (conda Vulkan
#         loader)  lib/ vklib/ vulkan/ (library shims, ICD manifest)  conda-pkgs/
#       plus ~/.sapien as a real directory (SAPIEN hardcodes that path). The recommended
#       checkout is ~/callosum, but the scripts work from any path. The venv, uv cache and
#       uv-managed Python are on one filesystem on purpose (uv hardlinks wheels).
#       A container restart loses nothing: no re-clone, no re-download of torch.
#   scratch layout  (opt-in, or automatic when the data root has < 30 GB free)
#       The same sub-directories under $CALLOSUM_SCRATCH (default /tmp/$USER-callosum)
#       and ~/.sapien as a symlink into it. Assumed WIPED on restart: re-run this
#       script then. Opt in with `CALLOSUM_SCRATCH=/big/disk/dir bash scripts/setup_server.sh`
#       (or CALLOSUM_LAYOUT=scratch; CALLOSUM_LAYOUT=home never falls back to scratch).
#
# In both layouts runs/ in the checkout is a symlink to $CALLOSUM_RUNS (default
# ~/callosum-runs): results survive re-cloning the checkout. A sourceable env file is
# written to ~/.callosum-env.sh (and hooked into ~/.bashrc); it is rewritten from scratch
# on every run, so exports of an earlier layout do not linger.
#
# Coming from the old /tmp-based layout needs nothing special: run this script from a fresh
# clone (an old ~/.callosum-env.sh, a dangling ~/.sapien link into /tmp and stale exports are
# handled below). Temporary files (TMPDIR) stay in /tmp.
#
# Network needs (allowlisted server): github.com (SAPIEN downloads its PhysX GPU
# library from a GitHub release on first use; uv downloads Python from GitHub),
# pypi.org (uv itself, via pip), astral.sh (installer fallback), files.pythonhosted.org,
# download.pytorch.org. Downloads are time-bounded: a non-allowlisted host may hang.
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
# Warn when the persistent runs/ store has less than this many MB free. Full-state checkpoints
# are ~7 MB each, a 30M-step run ~15 MB, so this only trips when $HOME is nearly full.
RUNS_LOW_MB=5120

# Free space in whole GB on the filesystem holding $1 (empty if unknown).
free_gb() { df -Pk "$1" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}' || true; }

# Free space in whole MB on the filesystem holding $1 (empty if unknown).
free_mb() { df -Pk "$1" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024}' || true; }

# Join non-empty arguments with ':' (an empty LD_LIBRARY_PATH entry means "cwd").
join_path() {
  local out="" p
  for p in "$@"; do
    if [ -n "$p" ]; then out="${out:+$out:}$p"; fi
  done
  printf '%s' "$out"
}

# Nearest existing ancestor of $1 (for df on a directory that is not created yet).
existing_ancestor() {
  local d="$1"
  while [ ! -e "$d" ] && [ "$d" != "/" ] && [ -n "$d" ]; do d="$(dirname "$d")"; done
  printf '%s' "$d"
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
ENV_FILE="$HOME/.callosum-env.sh"

# Value a variable has after sourcing the PREVIOUS env file, read in a clean process
# (nothing is exported into this one). Empty when the file or the variable is absent.
env_file_value() {
  [ -f "$ENV_FILE" ] || return 0
  env -i HOME="$HOME" bash -c '. "$1" >/dev/null 2>&1; printf %s "${!2:-}"' _ "$ENV_FILE" "$1" 2>/dev/null || true
}
PREV_SCRATCH="$(env_file_value CALLOSUM_SCRATCH)"   # old scratch layout, if any
PREV_REPO="$(env_file_value CALLOSUM_REPO)"

# An env file from before the home layout exported CALLOSUM_SCRATCH into every new
# terminal. That inherited value is not a request for scratch mode (the new file has a
# "# callosum-layout:" marker; the old one does not): ignore it, unless the user also
# says CALLOSUM_LAYOUT=scratch.
if [ -n "${CALLOSUM_SCRATCH:-}" ] && [ "${CALLOSUM_LAYOUT:-auto}" != "scratch" ] \
   && [ -f "$ENV_FILE" ] && ! grep -q '^# callosum-layout:' "$ENV_FILE" \
   && [ "$CALLOSUM_SCRATCH" = "$PREV_SCRATCH" ]; then
  warn "ignoring CALLOSUM_SCRATCH=$CALLOSUM_SCRATCH inherited from the old ~/.callosum-env.sh (home layout is the default now)"
  warn "to really use scratch: CALLOSUM_LAYOUT=scratch CALLOSUM_SCRATCH=<dir> bash scripts/setup_server.sh"
  unset CALLOSUM_SCRATCH
fi

LAYOUT="${CALLOSUM_LAYOUT:-auto}"
case "$LAYOUT" in auto|home|scratch) ;; *) die "CALLOSUM_LAYOUT must be auto, home or scratch (got '$LAYOUT')" ;; esac
DATA_ROOT="${CALLOSUM_DATA:-$HOME/.callosum}"
data_free="$(free_gb "$(existing_ancestor "$DATA_ROOT")")"
SCRATCH_MODE=0
if [ "$LAYOUT" = "scratch" ]; then
  SCRATCH_MODE=1
elif [ "$LAYOUT" = "auto" ]; then
  if [ -n "${CALLOSUM_SCRATCH:-}" ]; then
    SCRATCH_MODE=1
  elif [ -n "$data_free" ] && [ "$data_free" -lt "$NEED_GB" ]; then
    warn "only ${data_free} GB free for $DATA_ROOT (< ${NEED_GB}): falling back to the scratch layout"
    SCRATCH_MODE=1
  fi
fi

if [ "$SCRATCH_MODE" = "1" ]; then
  SCRATCH="${CALLOSUM_SCRATCH:-/tmp/$USER_NAME-callosum}"
  mkdir -p "$SCRATCH" || die "cannot create $SCRATCH — set CALLOSUM_SCRATCH to a writable directory with >= ${NEED_GB} GB free"
  scratch_free="$(free_gb "$SCRATCH")"
  if [ -n "$scratch_free" ] && [ "$scratch_free" -lt "$NEED_GB" ]; then
    die "$SCRATCH has only ${scratch_free} GB free (< ${NEED_GB}); set CALLOSUM_SCRATCH elsewhere"
  fi
  WORK="$SCRATCH"
  ok "scratch layout: heavy files go to $SCRATCH (${scratch_free:-?} GB free)"
  warn "assuming $SCRATCH is wiped on server restart: re-run this script then"
else
  mkdir -p "$DATA_ROOT" || die "cannot create $DATA_ROOT — set CALLOSUM_DATA to a writable directory"
  WORK="$DATA_ROOT"
  ok "home layout (persistent): heavy files go to $WORK (${data_free:-?} GB free)"
fi

export UV_PROJECT_ENVIRONMENT="$WORK/venv"
export UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/python"
export HF_HOME="$WORK/hf"
# ManiSkill v3.0.1 reads MS_ASSET_DIR (mani_skill/__init__.py); default ~/.maniskill.
export MS_ASSET_DIR="$WORK/maniskill"
# pip, matplotlib, torch hub, ...: anything that honours XDG goes to the same root.
export XDG_CACHE_HOME="$WORK/cache"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$MS_ASSET_DIR" "$XDG_CACHE_HOME"
VENV_DIR="$UV_PROJECT_ENVIRONMENT"
ok "venv         -> $VENV_DIR"
ok "uv cache     -> $UV_CACHE_DIR"
ok "HF_HOME      -> $HF_HOME"
ok "MS_ASSET_DIR -> $MS_ASSET_DIR"

# Library dirs of ANY callosum layout (the old scratch, ~/.callosum, this run's) must
# not leak from an inherited LD_LIBRARY_PATH into the probes below: a stale
# libcuda.so shim would make "libcuda.so resolves" true for the wrong reason.
STALE_ROOT="${PREV_SCRATCH:-/nonexistent-callosum}"
ld_clean=""
IFS=: read -ra _ld_parts <<< "${LD_LIBRARY_PATH:-}"
for _p in ${_ld_parts[@]+"${_ld_parts[@]}"}; do
  case "$_p" in
    ""|*/.callosum/lib|*/.callosum/vklib|*-callosum/lib|*-callosum/vklib) ;;
    "$WORK/lib"|"$WORK/vklib"|"$STALE_ROOT/lib"|"$STALE_ROOT/vklib") ;;
    *) ld_clean="${ld_clean:+$ld_clean:}$_p" ;;
  esac
done
unset _ld_parts _p
if [ -n "$ld_clean" ]; then export LD_LIBRARY_PATH="$ld_clean"; else unset LD_LIBRARY_PATH; fi

# SAPIEN hardcodes Path.home()/.sapien (PhysX GPU library, ~240 MB) with no env override.
link="$HOME/.sapien"
if [ "$SCRATCH_MODE" = "1" ]; then
  # scratch layout: redirect it with a symlink into scratch.
  target="$SCRATCH/sapien"
  mkdir -p "$target"
  if [ -L "$link" ]; then
    ln -sfn "$target" "$link"          # also repairs a link left dangling by a wipe
  else
    if [ -e "$link" ]; then cp -a "$link/." "$target/" && rm -rf "$link"; fi
    ln -s "$target" "$link"
  fi
  ok "\$HOME/.sapien -> $target"
else
  # home layout: ~/.sapien is a real directory. An earlier scratch layout left a symlink
  # into /tmp there: copy what its target still holds (never delete the target, a running
  # process may have the PhysX library mapped), then put the copy in place of the link.
  tmp_sapien="$HOME/.sapien.migrating"
  if [ -L "$link" ]; then
    old_target="$(readlink -f "$link" 2>/dev/null || true)"
    rm -rf "$tmp_sapien"; mkdir -p "$tmp_sapien"
    if [ -n "$old_target" ] && [ -d "$old_target" ]; then
      cp -a "$old_target/." "$tmp_sapien/" || { rm -rf "$tmp_sapien"; die "could not copy $old_target to $tmp_sapien"; }
      ok "copied ~/.sapien from $old_target"
    fi
    rm -f "$link"; mv "$tmp_sapien" "$link"
  elif [ ! -e "$link" ]; then
    if [ -d "$tmp_sapien" ]; then mv "$tmp_sapien" "$link"; else mkdir -p "$link"; fi   # tmp: an interrupted copy
  fi
  ok "\$HOME/.sapien is a real directory (persistent)"
fi

if [ "$SCRATCH_MODE" != "1" ]; then
  case "$REPO_ROOT" in
    /tmp/*|/var/tmp/*)
      warn "this checkout ($REPO_ROOT) is under /tmp and is lost on a container restart;"
      warn "the persistent place is ~/callosum: git clone there and run setup from it (docs/server-runbook.md)" ;;
  esac
fi
VENV_PY="$VENV_DIR/bin/python"

# --------------------------------------------------------------------- 3. uv
# No root: uv goes to ~/.local/bin (~50 MB). It survives a scratch wipe, so this
# only installs the first time. The server's outbound access is an allowlist and
# a blocked host can HANG instead of refusing, so every download here is bounded.
#   1. pip --user (PyPI is allowlisted; pip has its own timeout/retries)
#   2. the astral.sh installer, pinned to github.com release downloads. By default
#      the installer tries releases.astral.sh first with a curl that has no
#      timeout, which hangs when that host is not allowlisted.
say "uv"
export PATH="$HOME/.local/bin:$PATH"

# Run "$@" under `timeout $1` seconds when coreutils timeout exists, else unbounded.
with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then timeout "$secs" "$@"; else "$@"; fi
}

install_uv_pip() {
  command -v python3 >/dev/null 2>&1 || return 1
  python3 -m pip install --user --quiet --timeout 30 --retries 1 uv || return 1
  # pip --user puts the script in <user base>/bin (usually ~/.local/bin).
  local ub
  ub="$(python3 -m site --user-base 2>/dev/null || true)"
  if [ -n "$ub" ]; then export PATH="$ub/bin:$PATH"; fi
}

install_uv_astral() {
  command -v curl >/dev/null 2>&1 || return 1
  (
    export UV_NO_MODIFY_PATH=1
    export UV_INSTALLER_GITHUB_BASE_URL="https://github.com"   # skip releases.astral.sh
    set -o pipefail
    curl -LsSf --connect-timeout 10 --max-time 120 https://astral.sh/uv/install.sh \
      | with_timeout 300 sh
  )
}

if ! command -v uv >/dev/null 2>&1; then
  if install_uv_pip; then
    ok "uv installed via pip --user (PyPI)"
  else
    warn "pip install --user uv failed; trying the astral.sh installer (github.com release)"
    if install_uv_astral; then
      ok "uv installed via astral.sh installer"
    else
      die "could not install uv (PyPI and astral.sh both failed or timed out); install it manually (pip install --user uv) and re-run"
    fi
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
  # (Library dirs of earlier layouts were stripped from LD_LIBRARY_PATH above, so this is
  # not a stale shim.) Keep a shim from an earlier run of THIS layout in the env file.
  if [ -e "$WORK/lib/libcuda.so" ]; then SHIM_DIR="$WORK/lib"; fi
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
if with_timeout 600 "$VENV_PY" -c 'import sapien; sapien.physx.enable_gpu()' 2>"$WORK/physx.log"; then
  ok "physx.enable_gpu() works ($(du -sh "$HOME/.sapien/" 2>/dev/null | cut -f1) in ~/.sapien)"
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
# not the global device). Camera rendering (vision phase, wrist-camera videos) in
# addition needs the device to be the GPU: SAPIEN matches a Vulkan physical device
# to "cuda:0", which lavapipe (CPU) can never satisfy.
#
# What worked on this very server in Aug 2026 (archived branch step/2.1-ippo, see
# docs/server-runbook.md): the container ships the NVIDIA user-space libs
# (libGLX_nvidia.so.0, ...) despite NVIDIA_DRIVER_CAPABILITIES=compute,utility, but
# (a) the system libvulkan (1.3.275) is too old for the 570.x ICD (vkCreateInstance
# returns NULL) and (b) a usable NVIDIA ICD manifest may be missing or hidden:
# SAPIEN replaces the loader's search with its own bundled manifest when
# /usr/share/vulkan/icd.d/nvidia_icd.json is absent. So: write OUR manifest next to
# the driver library, pair it with a current loader (conda-forge) on
# LD_LIBRARY_PATH, and PROVE the result instead of assuming it. Nothing is trusted
# until a probe process has constructed RenderMaterial() and, for "hardware",
# RenderSystem("cuda:0"). Candidates, best first:
#   our NVIDIA manifest + current loader, ... + system loader, stock setup,
#   other system manifests (both loaders), lavapipe (software, last resort).
# The result is re-evaluated on EVERY run (a lavapipe pick from an earlier run is
# not remembered) and the env file below is rewritten from it.
say "Vulkan render device"

# Start from a clean slate: a terminal that sourced ~/.callosum-env.sh after an
# earlier run carries that run's VK_ICD_FILENAMES (e.g. lavapipe) and loader dir,
# which would make every probe below test the OLD answer.
if [ -n "${VK_ICD_FILENAMES:-}" ]; then
  warn "ignoring inherited VK_ICD_FILENAMES=$VK_ICD_FILENAMES (re-evaluating from scratch)"
fi
unset VK_ICD_FILENAMES VK_DRIVER_FILES VK_ADD_DRIVER_FILES
# Strip $WORK/vklib from LD_LIBRARY_PATH for a clean Vulkan probe (same logic as at startup).
vklib_clean=""
IFS=: read -ra _ld_parts <<< "${LD_LIBRARY_PATH:-}"
for _p in ${_ld_parts[@]+"${_ld_parts[@]}"}; do
  case "$_p" in
    ""|"$WORK/vklib") ;;
    *) vklib_clean="${vklib_clean:+$vklib_clean:}$_p" ;;
  esac
done
unset _ld_parts _p
if [ -n "$vklib_clean" ]; then export LD_LIBRARY_PATH="$vklib_clean"; else unset LD_LIBRARY_PATH; fi

# --- NVIDIA user-space libraries (ldconfig, then well-known directories) ---
LDCONFIG="$(command -v ldconfig 2>/dev/null || true)"
[ -n "$LDCONFIG" ] || LDCONFIG=/sbin/ldconfig
find_nv_lib() {  # $1 = file name -> prints the path, returns 1 when not found
  local p="" d
  if [ -x "$LDCONFIG" ]; then
    p="$("$LDCONFIG" -p 2>/dev/null | awk -v n="$1" '$1 == n {print $NF; exit}' || true)"
  fi
  if [ -n "$p" ] && [ -e "$p" ]; then printf '%s' "$p"; return 0; fi
  for d in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib /usr/local/nvidia/lib64 \
           /usr/local/nvidia/lib /usr/lib/nvidia; do
    if [ -e "$d/$1" ]; then printf '%s' "$d/$1"; return 0; fi
  done
  return 1
}

LIBGLX="$(find_nv_lib libGLX_nvidia.so.0 || true)"
NV_ICD=""
if [ -n "$LIBGLX" ]; then
  ok "NVIDIA Vulkan driver library: $LIBGLX"
  # libGLX_nvidia dlopen()s its Vulkan back-end at init, so ldd cannot show a gap.
  if ! ls "$(dirname "$LIBGLX")"/libnvidia-glvkspirv.so.* >/dev/null 2>&1; then
    warn "libnvidia-glvkspirv.so.* is missing next to it: the NVIDIA Vulkan path will likely fail"
  fi
  # Our own manifest, with an ABSOLUTE library path (independent of the loader's
  # search path inside child processes). Format and api_version as in NVIDIA's
  # stock nvidia_icd.json for the 570.x driver (the lab server had api 1.4.303).
  mkdir -p "$WORK/vulkan/icd.d"
  NV_ICD="$WORK/vulkan/icd.d/nvidia_icd.json"
  cat > "$NV_ICD" <<JSON
{
    "file_format_version": "1.0.1",
    "ICD": {
        "library_path": "$LIBGLX",
        "api_version": "1.4.303"
    }
}
JSON
  ok "wrote $NV_ICD"
else
  warn "libGLX_nvidia.so.0 not found (ldconfig, /usr/lib*, /usr/local/nvidia): no hardware Vulkan possible"
  warn "if it is absent the pod lacks NVIDIA_DRIVER_CAPABILITIES=graphics; see docs/server-runbook.md"
fi

# --- a current Vulkan loader (+ lavapipe, vulkaninfo) from conda-forge ---
MESA="$WORK/mesa"
if [ ! -e "$MESA/lib/libvulkan.so.1" ] && command -v conda >/dev/null 2>&1; then
  warn "installing a current Vulkan loader (+ lavapipe, vulkaninfo) with conda into $MESA (needs conda-forge)"
  # conda's package cache defaults to ~/.conda/pkgs: keep it with the other heavy files.
  CONDA_PKGS_DIRS="$WORK/conda-pkgs" \
    with_timeout 900 conda create -y -q -p "$MESA" -c conda-forge mesalib vulkan-tools >/dev/null 2>&1 \
    || warn "conda install failed (conda-forge not reachable from the server?)"
fi
VKLIB_DIR=""
if [ -e "$MESA/lib/libvulkan.so.1" ]; then
  # A directory holding ONLY the loader: prepending conda's whole lib/ would
  # also shadow libstdc++/libgcc and can break torch.
  VKLIB_DIR="$WORK/vklib"
  mkdir -p "$VKLIB_DIR"
  ln -sfn "$MESA/lib/libvulkan.so.1" "$VKLIB_DIR/libvulkan.so.1"
  ok "current Vulkan loader: $(readlink -f "$MESA/lib/libvulkan.so.1" 2>/dev/null || echo "$MESA/lib/libvulkan.so.1")"
else
  warn "no current Vulkan loader (conda unavailable or failed): only the system loader can be tried"
fi

# --- candidate list: "manifest|extra loader dir|label" ("" manifest = stock) ---
CANDS=()
add_cand() { CANDS+=("$1|$2|$3"); }
if [ -n "$NV_ICD" ]; then
  if [ -n "$VKLIB_DIR" ]; then add_cand "$NV_ICD" "$VKLIB_DIR" "generated NVIDIA manifest + current loader"; fi
  add_cand "$NV_ICD" "" "generated NVIDIA manifest + system loader"
fi
add_cand "" "" "stock configuration (SAPIEN default)"
for icd in /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json; do
  [ -e "$icd" ] || continue
  if [ -n "$VKLIB_DIR" ]; then add_cand "$icd" "$VKLIB_DIR" "$icd + current loader"; fi
  add_cand "$icd" "" "$icd + system loader"
done
for icd in "$MESA"/share/vulkan/icd.d/*.json; do
  [ -e "$icd" ] || continue
  if [ -n "$VKLIB_DIR" ]; then add_cand "$icd" "$VKLIB_DIR" "$(basename "$icd") (software rasteriser) + current loader"; fi
  add_cand "$icd" "" "$(basename "$icd") (software rasteriser) + system loader"
done

# Probe = a fresh process (the loader and ICD are chosen at process start). It
# prints "L1" once RenderMaterial() works (any Vulkan device suffices for state
# training) and "HW <name>" once RenderSystem("cuda:0") works, which only a GPU
# Vulkan device that SAPIEN can pair with the CUDA device passes.
PROBE_PY='
import sys
from sapien.render import RenderMaterial, RenderSystem
RenderMaterial()
print("L1", flush=True)
try:
    import sapien
    RenderSystem("cuda:0")
    try:
        name = sapien.Device("cuda:0").name
    except Exception:
        name = "cuda:0"
    print("HW", name, flush=True)
except Exception as e:
    print("NOHW", str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__, flush=True)
'
VK_LOG="$WORK/vulkan-probe.log"
: > "$VK_LOG"
probe() {  # $1 manifest ("" = leave unset), $2 extra loader dir -> prints hw|<name>, soft|<why> or none
  local out
  printf '### manifest=%s loader_dir=%s\n' "${1:-<stock>}" "${2:-<system>}" >> "$VK_LOG"
  out="$(
    if [ -n "$1" ]; then export VK_ICD_FILENAMES="$1"; fi
    ld="$(join_path "$2" "${LD_LIBRARY_PATH:-}")"
    if [ -n "$ld" ]; then export LD_LIBRARY_PATH="$ld"; fi
    with_timeout 120 "$VENV_PY" -c "$PROBE_PY" 2>>"$VK_LOG" || true
  )"
  local hw nohw
  hw="$(printf '%s\n' "$out" | sed -n 's/^HW //p' | head -n 1)"
  nohw="$(printf '%s\n' "$out" | sed -n 's/^NOHW //p' | head -n 1)"
  if [ -n "$hw" ]; then printf 'hw|%s' "$hw"
  elif printf '%s\n' "$out" | grep -q '^L1$'; then printf 'soft|%s' "$nohw"
  else printf 'none'
  fi
}

ICD=""; VKLIB=""; VK_KIND=""; VK_DEVICE=""; VK_LABEL=""
SOFT_ICD=""; SOFT_LIB=""; SOFT_LABEL=""
echo "   probing ${#CANDS[@]} candidate(s), best first (details: $VK_LOG)"
for c in "${CANDS[@]}"; do
  IFS='|' read -r c_icd c_lib c_label <<< "$c"
  if [ -n "$c_icd" ] && [ ! -e "$c_icd" ]; then continue; fi
  r="$(probe "$c_icd" "$c_lib")"
  case "$r" in
    hw\|*)
      printf '     \033[32m✓\033[0m %s: GPU render device (%s)\n' "$c_label" "${r#hw|}"
      ICD="$c_icd"; VKLIB="$c_lib"; VK_KIND="hardware"; VK_DEVICE="${r#hw|}"; VK_LABEL="$c_label"
      break ;;
    soft\|*)
      why="${r#soft|}"
      printf '     ~ %s: Vulkan device works, but not the CUDA GPU%s\n' "$c_label" "${why:+ ($why)}"
      if [ -z "$SOFT_LABEL" ]; then SOFT_ICD="$c_icd"; SOFT_LIB="$c_lib"; SOFT_LABEL="$c_label"; fi ;;
    *)
      printf '     - %s: no render device\n' "$c_label" ;;
  esac
done

if [ "$VK_KIND" != "hardware" ]; then
  if [ -z "$SOFT_LABEL" ]; then
    tail -n 6 "$VK_LOG" | sed 's/^/     /'
    die "no Vulkan ICD gives a render device; run: bash scripts/diagnose_vulkan.sh (see docs/server-runbook.md, Vulkan)"
  fi
  ICD="$SOFT_ICD"; VKLIB="$SOFT_LIB"; VK_KIND="software"; VK_LABEL="$SOFT_LABEL"
  # The first lines of loader/driver complaints explain WHY hardware failed.
  if grep -qiE 'error|fail|warn' "$VK_LOG" 2>/dev/null; then
    warn "loader/driver messages from the probes (full log: $VK_LOG):"
    grep -iE 'error|fail|warn' "$VK_LOG" | sort -u | head -n 6 | sed 's/^/     /'
  fi
fi

if [ -n "$ICD" ]; then export VK_ICD_FILENAMES="$ICD"; fi
if [ -n "$VKLIB" ]; then
  export LD_LIBRARY_PATH
  LD_LIBRARY_PATH="$(join_path "$VKLIB" "${LD_LIBRARY_PATH:-}")"
fi

if [ "$VK_KIND" = "hardware" ]; then
  ok "Vulkan: HARDWARE NVIDIA Vulkan on $VK_DEVICE via $VK_LABEL"
  ok "camera rendering works: use render_backend=\"gpu\" for vision / wrist-camera videos"
  ok "state training: render_backend=\"none\" stays the default (no renderer needed)"
else
  warn "Vulkan: SOFTWARE / no CUDA-matched GPU device ($VK_LABEL): RenderMaterial() works, which is enough to build envs"
  warn "state-based training must pass render_backend=\"none\" to gym.make (the smoke scripts and the trainer do);"
  warn "camera rendering (vision phase) needs hardware Vulkan: run bash scripts/diagnose_vulkan.sh and"
  warn "see docs/server-runbook.md (Vulkan)"
fi

# ------------------------------------------------------------ 5. env file, runs/
# One file that any NEW terminal sources. ~/.bashrc only reaches shells started
# after setup (not `bash script.sh`, not an already-running kernel). It is rewritten
# from scratch on every run, so exports of an earlier layout (e.g. the old /tmp
# scratch paths) are gone from new terminals afterwards.
say "Environment file"
{
  echo "# callosum-layout: $([ "$SCRATCH_MODE" = "1" ] && echo scratch || echo home)"
  echo "# Generated by scripts/setup_server.sh. Source it in every new terminal:"
  echo "#     source ~/.callosum-env.sh"
  echo "# Without it, uv would not know where the venv lives and would build a second one."
  printf 'export PATH="$HOME/.local/bin:$PATH"\n'
  if [ "$SCRATCH_MODE" = "1" ]; then
    printf 'export CALLOSUM_SCRATCH=%q\n' "$SCRATCH"
    echo 'unset CALLOSUM_DATA'
  else
    printf 'export CALLOSUM_DATA=%q\n' "$WORK"
    echo 'unset CALLOSUM_SCRATCH'
  fi
  printf 'export CALLOSUM_REPO=%q\n' "$REPO_ROOT"
  printf 'export UV_PROJECT_ENVIRONMENT=%q\n' "$UV_PROJECT_ENVIRONMENT"
  printf 'export UV_CACHE_DIR=%q\n' "$UV_CACHE_DIR"
  printf 'export UV_PYTHON_INSTALL_DIR=%q\n' "$UV_PYTHON_INSTALL_DIR"
  printf 'export HF_HOME=%q\n' "$HF_HOME"
  printf 'export MS_ASSET_DIR=%q\n' "$MS_ASSET_DIR"
  printf 'export XDG_CACHE_HOME=%q\n' "$XDG_CACHE_HOME"
  # Library dirs of any earlier layout, or of an earlier `source` of this file, must not
  # pile up in LD_LIBRARY_PATH: drop them before the current ones are prepended.
  echo '__cl_keep=""'
  echo 'if [ -n "${LD_LIBRARY_PATH:-}" ]; then'
  echo '  IFS=: read -ra __cl_parts <<< "$LD_LIBRARY_PATH"'
  echo '  for __cl_p in ${__cl_parts[@]+"${__cl_parts[@]}"}; do'
  echo '    case "$__cl_p" in'
  echo '      ""|*/.callosum/lib|*/.callosum/vklib|*-callosum/lib|*-callosum/vklib) ;;'
  printf '      %q|%q) ;;\n' "$WORK/lib" "$WORK/vklib"
  echo '      *) __cl_keep="${__cl_keep:+$__cl_keep:}$__cl_p" ;;'
  echo '    esac'
  echo '  done'
  echo 'fi'
  echo 'if [ -n "$__cl_keep" ]; then export LD_LIBRARY_PATH="$__cl_keep"; else unset LD_LIBRARY_PATH; fi'
  echo 'unset __cl_keep __cl_parts __cl_p'
  if [ -n "$SHIM_DIR" ]; then
    printf 'export LD_LIBRARY_PATH=%q"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n' "$SHIM_DIR"
  fi
  echo "# Vulkan: $VK_KIND (${VK_DEVICE:-no GPU device}) via $VK_LABEL"
  if [ -n "$VKLIB" ]; then
    printf 'export LD_LIBRARY_PATH=%q"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n' "$VKLIB"
  fi
  if [ -n "$ICD" ]; then
    printf 'export VK_ICD_FILENAMES=%q\n' "$ICD"
  else
    # Stock config won: drop any VK_ICD_FILENAMES left over from an earlier run.
    echo 'unset VK_ICD_FILENAMES'
  fi
  printf '[ -d %q ] || echo "callosum: venv missing; run: bash %q/scripts/setup_server.sh (docs/server-runbook.md)" >&2\n' "$VENV_DIR" "$REPO_ROOT"
} > "$ENV_FILE"
ok "wrote $ENV_FILE"

# Hook it into new interactive terminals (idempotent; the file may be stale or
# missing, hence the guard).
if ! grep -q 'callosum-env.sh' "$HOME/.bashrc" 2>/dev/null; then
  printf '\n# callosum (added by scripts/setup_server.sh)\n[ -f "$HOME/.callosum-env.sh" ] && . "$HOME/.callosum-env.sh"\n' >> "$HOME/.bashrc"
  ok "hooked into ~/.bashrc"
else
  ok "already hooked into ~/.bashrc"
fi

# An accidental in-repo .venv (uv run without the env file) wastes space.
if [ -d "$REPO_ROOT/.venv" ] && [ "$VENV_DIR" != "$REPO_ROOT/.venv" ]; then
  warn "removing stray $REPO_ROOT/.venv (created by uv run without the env file)"
  rm -rf "$REPO_ROOT/.venv"
fi

# Results must survive re-cloning the checkout (and a restart, if the checkout is in
# /tmp): runs/ is a symlink to a store in $HOME, whatever the checkout path.
say "runs/ (results)"
RUNS_STORE="${CALLOSUM_RUNS:-$HOME/callosum-runs}"
mkdir -p "$RUNS_STORE"
if [ "$(readlink -f "$RUNS_STORE")" = "$(readlink -f "$REPO_ROOT/runs" 2>/dev/null || true)" ]; then
  ok "runs/ is the store itself ($RUNS_STORE)"
else
  if [ -L "$REPO_ROOT/runs" ]; then
    ln -sfn "$RUNS_STORE" "$REPO_ROOT/runs"
  else
    # A real runs/ from before the symlink: keep its contents, then link.
    if [ -d "$REPO_ROOT/runs" ]; then
      cp -a "$REPO_ROOT/runs/." "$RUNS_STORE/" && rm -rf "$REPO_ROOT/runs"
    fi
    ln -s "$RUNS_STORE" "$REPO_ROOT/runs"
  fi
  runs_free="$(free_mb "$RUNS_STORE")"
  ok "runs -> $RUNS_STORE ($(du -sh "$RUNS_STORE" 2>/dev/null | cut -f1) used, ${runs_free:-?} MB free)"
  if [ -n "$runs_free" ] && [ "$runs_free" -lt "$RUNS_LOW_MB" ]; then
    warn "\$HOME (runs/) has only ${runs_free} MB free: prune checkpoints you do not need and download results (JupyterHub file browser)"
  fi
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
echo "   Checkout : $REPO_ROOT"
if [ "$SCRATCH_MODE" = "1" ]; then
  echo "   Scratch  : $SCRATCH   (assumed wiped on server restart: re-run this script)"
else
  echo "   Data     : $WORK   (venv, caches, Vulkan loader; persistent)"
fi
echo "   Results  : $REPO_ROOT/runs -> $RUNS_STORE"
echo "   Open a NEW terminal (or 'source ~/.callosum-env.sh') so the variables are set, then use"
echo "   'uv run <cmd>'. Long runs: see docs/server-runbook.md (detached with setsid nohup)."
if [ "$SCRATCH_MODE" != "1" ]; then
  # Leftovers of the old /tmp layout (only present if /tmp was not wiped): never removed here.
  for old in "$PREV_SCRATCH" "$PREV_REPO"; do
    if [ -n "$old" ] && [ -d "$old" ] && [ "$old" != "$WORK" ] && [ "$old" != "$REPO_ROOT" ]; then
      warn "old layout still on disk: $old"
      warn "  once no run uses it: rm -rf $old   (a runs symlink inside it is removed, not its target)"
    fi
  done
fi
