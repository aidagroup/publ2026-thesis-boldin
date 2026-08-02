#!/usr/bin/env bash
# Bootstrap AIDA on a Linux + NVIDIA (CUDA) machine — a RunPod pod, or any
# CUDA box. Idempotent: safe to re-run on an existing pod.
#
#   bash scripts/setup_server.sh            # set up + verify
#   bash scripts/setup_server.sh --smoke    # ... and run the GPU smoke scripts
#
# Everything the project needs is pinned in uv.lock, so this never resolves
# dependencies — it installs the exact locked versions.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_SMOKE=0
[ "${1:-}" = "--smoke" ] && RUN_SMOKE=1

say() { printf '\n\033[1m>> %s\033[0m\n' "$*"; }
ok()  { printf '   \033[32m✓\033[0m %s\n' "$*"; }
warn(){ printf '   \033[33m!\033[0m %s\n' "$*"; }
die() { printf '   \033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

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

# ------------------------------------------------------- 2. persistent caches
# On RunPod, /workspace survives pod restarts while / does not. Keeping the uv
# cache (and HF cache, for V-JEPA weights later) there avoids re-downloading
# multi-GB torch/CUDA wheels on every fresh pod.
say "Caches"
PERSIST=""
for d in /workspace /runpod-volume; do
  [ -d "$d" ] && [ -w "$d" ] && { PERSIST="$d"; break; }
done
if [ -n "$PERSIST" ]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$PERSIST/.cache/uv}"
  export HF_HOME="${HF_HOME:-$PERSIST/.cache/huggingface}"
  mkdir -p "$UV_CACHE_DIR" "$HF_HOME"
  ok "persistent volume: $PERSIST"
  ok "UV_CACHE_DIR=$UV_CACHE_DIR"
  # Persist for future shells on this pod.
  PROFILE="$HOME/.bashrc"
  if ! grep -q 'UV_CACHE_DIR' "$PROFILE" 2>/dev/null; then
    {
      echo "export UV_CACHE_DIR=$UV_CACHE_DIR"
      echo "export HF_HOME=$HF_HOME"
    } >> "$PROFILE"
    ok "exported into $PROFILE for future shells"
  fi
else
  warn "no persistent volume found — wheels will be re-downloaded on a fresh pod"
fi

# --------------------------------------------------------------------- 3. uv
say "uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "$(uv --version)"

# ------------------------------------------------------------- 4. environment
# Python 3.12: SAPIEN publishes no wheels for 3.13+.
say "Environment (Python 3.12 + locked deps)"
uv python install 3.12
# --frozen: install exactly uv.lock, never re-resolve. If this errors, the lock
# is out of sync with pyproject.toml — fix that on the dev machine, not here.
uv sync --frozen --extra sim --extra train --extra dev
ok "synced from uv.lock"

# ----------------------------------------------------------- 5. verification
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

    # Blackwell (sm_120, e.g. RTX 5090) needs a CUDA >= 12.8 build.
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
from mani_skill.agents.robots.so100 import SO100
assert SO100.uid == "so100"
print(f"   \033[32m✓\033[0m SO-100 agent available (uid={SO100.uid!r})")

# --- our environments (need mani_skill, so untestable on macOS) --------------
import callosum
import callosum.envs.face_turn  # noqa: F401  (registers FaceTurn-v0)
import callosum.envs.two_so100_base  # noqa: F401  (registers TwoSO100-v0)
from mani_skill.utils.registration import REGISTERED_ENVS

for env_id in ("TwoSO100-v0", "FaceTurn-v0"):
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

# ------------------------------------------------------------- 6. smoke tests
if [ "$RUN_SMOKE" = "1" ]; then
  say "GPU smoke tests (the checks that could not run on macOS)"
  echo "--- scripts/smoke_env.py: two SO-100 arms, per-agent obs/actions, arm reach ---"
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
echo "   Use 'uv run <cmd>' — or activate with: source .venv/bin/activate"
