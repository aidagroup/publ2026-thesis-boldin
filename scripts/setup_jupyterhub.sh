#!/usr/bin/env bash
# Bootstrap callosum inside a JupyterHub single-user container (no SSH, no sudo).
#
#   bash scripts/setup_jupyterhub.sh            # set up + verify
#   bash scripts/setup_jupyterhub.sh --smoke    # ... and run the GPU smoke scripts
#
# Differs from setup_server.sh (RunPod + SSH) in three ways this environment forces:
#   * $HOME is tiny (a few GB) -> venv, caches and assets all go to a scratch dir.
#   * github.com is unreachable -> uv is installed from PyPI, not astral.sh
#     (the astral.sh installer pulls its binary from GitHub releases).
#   * /tmp is usually wiped when the pod restarts -> this script is idempotent
#     and cheap to re-run; only the sources in $HOME survive.
set -euo pipefail
# NOTE: with `set -e` + `pipefail`, an assignment like `X=$(find ... | head -1)`
# KILLS the script silently when find hits an unreadable directory, or when head
# closes the pipe early and find takes SIGPIPE. Every such substitution below
# therefore ends in `|| true` and is checked explicitly afterwards. This is not
# defensive noise: it cost one round trip when the Vulkan section exited without
# printing a single line.

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '   \033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── 1. Pick a scratch directory with real space ───────────────────────────────
say "Scratch directory"
need_gb=40
WORK=""
for d in /tmp /scratch /data /workspace; do
    [ -d "$d" ] && [ -w "$d" ] || continue
    free_gb=$(df -BG --output=avail "$d" 2>/dev/null | tail -1 | tr -dc '0-9' || true)
    [ -n "${free_gb:-}" ] || continue
    printf '   %-12s %s GB free\n' "$d" "$free_gb"
    if [ -z "$WORK" ] && [ "$free_gb" -ge "$need_gb" ]; then
        WORK="$d/callosum-work"
    fi
done
[ -n "$WORK" ] || die "no directory with >= ${need_gb} GB free; ask the lab admin for scratch space"
mkdir -p "$WORK"/{venv,uv-cache,hf,maniskill}
ok "using $WORK"

home_free=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9' || true)
[ "${home_free:-99}" -lt 5 ] && warn "\$HOME has only ${home_free} GB free — keep runs/ small, checkpoints add up"

# ── 2. Environment: everything heavy points at scratch ────────────────────────
say "Environment variables"
export UV_PROJECT_ENVIRONMENT="$WORK/venv"   # venv OUTSIDE the repo ($HOME is tiny)
export UV_CACHE_DIR="$WORK/uv-cache"
export HF_HOME="$WORK/hf"
export PATH="$HOME/.local/bin:$PATH"

# ManiSkill downloads assets to ~/.maniskill; symlink rather than trusting an
# env-var name, so this holds regardless of the version's config knob.
for d in maniskill sapien; do
    if [ ! -L "$HOME/.$d" ]; then
        [ -d "$HOME/.$d" ] && mv "$HOME/.$d" "$WORK/$d-existing"
        mkdir -p "$WORK/$d"
        ln -sfn "$WORK/$d" "$HOME/.$d"
    fi
done
ok "UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
ok "UV_CACHE_DIR=$UV_CACHE_DIR"
ok "~/.maniskill and ~/.sapien -> $WORK/"

# Persist for future terminals / kernels. Idempotent.
PROFILE="$HOME/.bashrc"
if ! grep -q 'callosum-work' "$PROFILE" 2>/dev/null; then
    {
        echo ""
        echo "# callosum (added by scripts/setup_jupyterhub.sh)"
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo "export UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
        echo "export UV_CACHE_DIR=$UV_CACHE_DIR"
        echo "export HF_HOME=$HF_HOME"
    } >> "$PROFILE"
    ok "exported into $PROFILE"
fi

# ── 3. uv, from PyPI (astral.sh installer needs GitHub, which is blocked) ─────
say "uv"
if command -v uv > /dev/null 2>&1; then
    ok "already present: $(uv --version)"
else
    pip install --user --quiet uv || pip install --quiet uv || die "could not install uv from PyPI"
    command -v uv > /dev/null 2>&1 || die "uv installed but not on PATH — check ~/.local/bin"
    ok "installed: $(uv --version)"
fi

# ── 4. Dependencies, exactly as locked ────────────────────────────────────────
say "Dependencies (uv sync --frozen)"
warn "first run downloads several GB of torch + CUDA wheels; expect a few minutes"
uv sync --frozen --extra sim --extra train --extra dev
ok "synced into $UV_PROJECT_ENVIRONMENT"

# ── 4b. PhysX GPU library, staged by hand (SAPIEN fetches it from GitHub) ─────
# On first import SAPIEN downloads libPhysXGpu_64.so from a GitHub release. This
# network cannot reach github.com, so the archive is carried in by hand and
# unpacked here, before anything imports sapien.
say "PhysX GPU library"
PHYSX_VER="105.1-physx-5.3.1.patch0"
PHYSX_DIR="$HOME/.sapien/physx/$PHYSX_VER"
if [ -f "$PHYSX_DIR/libPhysXGpu_64.so" ]; then
    ok "already installed ($(du -h "$PHYSX_DIR/libPhysXGpu_64.so" | cut -f1))"
else
    STAGED=""
    for z in "$REPO_ROOT/vendor/physx-linux-so.zip" "$HOME/physx-linux-so.zip" \
             "$REPO_ROOT/physx-linux-so.zip" "$HOME/linux-so.zip"; do
        [ -f "$z" ] && { STAGED="$z"; break; }
    done
    if [ -n "$STAGED" ]; then
        mkdir -p "$PHYSX_DIR"
        unzip -o -q "$STAGED" -d "$PHYSX_DIR"
        [ -f "$PHYSX_DIR/libPhysXGpu_64.so" ] || die "archive did not contain libPhysXGpu_64.so"
        ok "installed from $STAGED -> $PHYSX_DIR"
    else
        warn "not found, and this network cannot reach github.com"
        cat <<TXT

   On a machine WITH GitHub access, download:
     https://github.com/sapien-sim/physx-precompiled/releases/download/$PHYSX_VER/linux-so.zip
   Upload it here as ~/physx-linux-so.zip and re-run this script.
   (81 MB compressed, 237 MB unpacked; it lands on scratch via the ~/.sapien symlink.)

TXT
        die "PhysX GPU library missing — sapien cannot start without it"
    fi
fi

# ── 4c. libcuda.so shim ───────────────────────────────────────────────────────
# SAPIEN's physx.enable_gpu() does ctypes.CDLL("libcuda.so") — the UNVERSIONED
# name. Container runtimes inject only the versioned libcuda.so.1 from the host
# driver; the bare symlink normally ships in a driver -devel package that is not
# installed here. torch is unaffected because it links libcuda.so.1 directly.
# Without this shim: OSError: libcuda.so: cannot open shared object file.
say "libcuda.so shim"
SHIM_DIR="$WORK/lib"
mkdir -p "$SHIM_DIR"
if [ -e "$SHIM_DIR/libcuda.so" ]; then
    ok "already present: $SHIM_DIR/libcuda.so"
else
    LIBCUDA=""
    # ldconfig knows where the runtime actually put it; fall back to the usual spots.
    if command -v ldconfig > /dev/null 2>&1; then
        LIBCUDA=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/ {print $NF; exit}' || true)
    fi
    if [ -z "$LIBCUDA" ] || [ ! -e "$LIBCUDA" ]; then
        for c in /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
                 /usr/lib64/libcuda.so.1 \
                 /usr/local/nvidia/lib64/libcuda.so.1 \
                 /usr/local/cuda/compat/libcuda.so.1; do
            [ -e "$c" ] && { LIBCUDA="$c"; break; }
        done
    fi
    [ -n "$LIBCUDA" ] || die "libcuda.so.1 not found — is this container GPU-enabled? (nvidia-smi works?)"
    ln -sfn "$LIBCUDA" "$SHIM_DIR/libcuda.so"
    ok "linked $SHIM_DIR/libcuda.so -> $LIBCUDA"
fi
export LD_LIBRARY_PATH="$SHIM_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if ! grep -q 'callosum-work/lib' "$PROFILE" 2>/dev/null; then
    echo "export LD_LIBRARY_PATH=\"$SHIM_DIR\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"" >> "$PROFILE"
    ok "exported into $PROFILE"
fi
warn "LD_LIBRARY_PATH is read by the loader at process start: run training from a"
warn "TERMINAL (it sources ~/.bashrc). A notebook kernel started earlier will NOT"
warn "see it, and setting os.environ inside the kernel is too late to help."

# ── 4d. Vulkan ICD ───────────────────────────────────────────────────────────
# SAPIEN's _ensure_vulkan_icd() looks ONLY at /usr/share/vulkan/icd.d/nvidia_icd.json.
# When that path is absent it points VK_ICD_FILENAMES at its own bundled manifest --
# and setting that variable REPLACES the loader's entire default search, so a
# perfectly good driver manifest elsewhere (here /etc/vulkan/icd.d/nvidia_icd.json,
# api_version 1.4.303) is ignored in favour of the bundled one (api_version 1.2.140).
# The loader then enumerates zero devices and SAPIEN reports "failed to find a
# rendering device". Fix: point the variable at the driver's own manifest ourselves,
# which also stops SAPIEN from substituting its bundled copy.
say "Vulkan ICD"
ICD=""
for cand in /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json; do
    [ -e "$cand" ] || continue
    lib=$(grep -o '"library_path"[^,}]*' "$cand" | sed 's/.*:[[:space:]]*"//; s/"$//' || true)
    [ -n "$lib" ] || continue
    case "$lib" in
        /*) [ -e "$lib" ] || continue ;;
        *)  ldconfig -p 2>/dev/null | grep -q "$lib" || continue ;;
    esac
    ICD="$cand"
    ok "using the driver's own manifest: $ICD -> $lib"
    break
done

if [ -z "$ICD" ]; then
    LIBGLX=$(ldconfig -p 2>/dev/null | awk '/libGLX_nvidia\.so\.0/ {print $NF; exit}' || true)
    if [ -z "$LIBGLX" ] || [ ! -e "$LIBGLX" ]; then
        LIBGLX=$(find /usr/lib /usr/lib64 /usr/local -maxdepth 4 \
                      -name 'libGLX_nvidia.so.0' 2>/dev/null | head -1 || true)
    fi
    [ -n "$LIBGLX" ] || die "no usable Vulkan ICD and no libGLX_nvidia.so.0 — run scripts/diagnose_vulkan.sh"
    mkdir -p "$WORK/vulkan/icd.d"
    ICD="$WORK/vulkan/icd.d/nvidia_icd.json"
    cat > "$ICD" <<JSON
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "$LIBGLX",
        "api_version": "1.3.242"
    }
}
JSON
    ok "no system manifest — wrote $ICD -> $LIBGLX"
fi

export VK_ICD_FILENAMES="$ICD"
if grep -q '^export VK_ICD_FILENAMES=' "$PROFILE" 2>/dev/null; then
    sed -i "s|^export VK_ICD_FILENAMES=.*|export VK_ICD_FILENAMES=$ICD|" "$PROFILE"
else
    echo "export VK_ICD_FILENAMES=$ICD" >> "$PROFILE"
fi
ok "VK_ICD_FILENAMES=$ICD"

# ── 4e. A sourceable env file ────────────────────────────────────────────────
# ~/.bashrc only helps shells started AFTER setup ran, and not at all for
# `bash script.sh` or a notebook kernel. Without the right UV_PROJECT_ENVIRONMENT,
# `uv run` silently builds a SECOND venv inside the repo -- on a 2 GB $HOME that
# is both wrong and dangerous. So write the exact resolved values to a file that
# any shell can source in one line.
say "Environment file"
ENV_FILE="$REPO_ROOT/.callosum-env.sh"
cat > "$ENV_FILE" <<ENVEOF
# Generated by scripts/setup_jupyterhub.sh — source this before any command:
#     source .callosum-env.sh
export PATH="\$HOME/.local/bin:\$PATH"
export UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"
export UV_CACHE_DIR="$UV_CACHE_DIR"
export HF_HOME="$HF_HOME"
export LD_LIBRARY_PATH="$SHIM_DIR\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES="$VK_ICD_FILENAMES"
ENVEOF
ok "wrote $ENV_FILE"

# A stray in-repo .venv means someone ran uv without the env file; it shadows
# nothing but wastes the little space $HOME has.
if [ -d "$REPO_ROOT/.venv" ]; then
    warn "found $REPO_ROOT/.venv — created by a uv call without the env file; removing"
    rm -rf "$REPO_ROOT/.venv"
fi

# ── 5. Verification — behaviour, not imports ──────────────────────────────────
say "GPU"
uv run python - <<'PY'
import re
import subprocess
import sys

import torch

print(f"   torch         : {torch.__version__} (built against CUDA {torch.version.cuda})")

if not torch.cuda.is_available():
    # The usual cause is a CUDA build newer than the driver supports. Say so
    # explicitly with both numbers rather than leaving a bare assertion: PyPI
    # ships CUDA 13 builds from torch 2.11.0 onward, and a 5xx driver caps at 12.8.
    drv_cuda = "unknown"
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"CUDA Version:\s*([0-9.]+)", out)
        if m:
            drv_cuda = m.group(1)
    except Exception:
        pass
    print(f"   driver supports CUDA up to : {drv_cuda}")
    print()
    print("   ✗ torch cannot use the GPU.")
    if drv_cuda != "unknown" and torch.version.cuda:
        if tuple(map(int, torch.version.cuda.split(".")[:2])) > tuple(map(int, drv_cuda.split(".")[:2])):
            print(f"     torch is built for CUDA {torch.version.cuda}, but the driver only")
            print(f"     supports up to CUDA {drv_cuda}. Lower the torch pin in pyproject.toml")
            print("     ('torch<2.11' already caps it at the newest CUDA 12 build on PyPI)")
            print("     and re-run `uv lock` on the dev machine, then re-upload.")
    sys.exit(1)

name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
built = torch.cuda.get_arch_list()
print(f"   device        : {name}")
print(f"   capability    : sm_{cc[0]}{cc[1]}")
print(f"   built for     : {', '.join(built)}")
if f"sm_{cc[0]}{cc[1]}" not in built:
    sys.exit(f"   ✗ this torch build has no kernels for sm_{cc[0]}{cc[1]}")
# is_available() can be True on a build that still cannot run on this card.
x = torch.randn(2048, 2048, device="cuda")
torch.cuda.synchronize()
print(f"   real matmul   : ok ({float((x @ x).sum()):.1f})")
PY
ok "CUDA verified by a real matmul, not just is_available()"

say "Simulator"
uv run python - <<'PY'
# SAPIEN historically needs libvulkan present even when nothing is rendered.
# State-based training does no rendering, so an import failure here is about
# missing system libraries, not about the GPU.
import sapien
print(f"   sapien        : {sapien.__version__}")
# A render device is required even for state-only observations (SAPIEN's URDF
# loader builds RenderMaterial unconditionally), so prove one exists now
# rather than three stack frames deep inside the first gym.make().
from sapien.render import RenderMaterial
RenderMaterial()
print("   render device : ok (RenderMaterial constructed)")
import mani_skill
print(f"   mani_skill    : {mani_skill.__version__}")
from mani_skill.envs.sapien_env import BaseEnv  # noqa: F401
from mani_skill.utils.registration import REGISTERED_ENVS
import callosum.envs.face_turn      # noqa: F401  (registers FaceTurn-v0)
import callosum.envs.two_so100_base # noqa: F401  (registers TwoSO100-v0)
for env_id in ("TwoSO100-v0", "FaceTurn-v0"):
    assert env_id in REGISTERED_ENVS, f"{env_id} did not register"
    print(f"   registered    : {env_id}")
PY
ok "simulator and both environments are importable and registered"

# ── 6. Optional smoke ─────────────────────────────────────────────────────────
if [ "${1:-}" = "--smoke" ]; then
    say "Phase-1 smoke (the checks that cannot run on macOS)"
    uv run python scripts/smoke_env.py
    uv run python scripts/smoke_face_turn.py
    ok "smoke scripts finished — read the output against docs/server-runbook.md section 2"
fi

say "Ready"
cat <<TXT
   Scratch : $WORK   (wiped on pod restart -> re-run this script)
   Sources : $REPO_ROOT   (survives; keep runs/ here)

   Next, in a TERMINAL (File -> New -> Terminal), not a notebook cell:

     cd $REPO_ROOT
     bash scripts/setup_jupyterhub.sh --smoke      # if not done yet

     nohup uv run python -m callosum.training.ippo \\
         --env-id TwoSO100-v0 --total-timesteps 50000 \\
         > runs/sanity.log 2>&1 &

   nohup detaches the run so it survives closing the browser tab.
   Watch it with:  tail -f runs/sanity.log
TXT
