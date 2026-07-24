#!/usr/bin/env bash
# Bootstrap AIDA on a Linux + NVIDIA (CUDA) server — e.g. an RTX 5090 (Blackwell) box.
# Usage:  git clone git@github.com:aidagroup/callosum.git && cd callosum
#         bash scripts/setup_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. uv (Python/dependency manager)
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Python 3.12 + full env (sim + train + dev).
#    torch pulls the cu128 build (Blackwell / RTX 50xx) per pyproject [tool.uv.sources].
uv python install 3.12
uv sync --extra sim --extra train --extra dev

# 3. Sanity checks
echo ">> sanity checks:"
uv run python - <<'PY'
import torch
print("  torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device:", torch.cuda.get_device_name(0))
else:
    print("  WARNING: CUDA not available — check driver / torch build")
import mani_skill
print("  mani_skill", getattr(mani_skill, "__version__", "?"))
try:
    import importlib
    importlib.import_module("mani_skill.agents.robots.so100")
    print("  SO-100 agent module: OK")
except Exception as e:  # noqa: BLE001
    print("  SO-100 import check skipped:", e)
PY
echo ">> done. Activate with: source .venv/bin/activate  (or prefix commands with 'uv run')"
