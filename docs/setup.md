# Dev & deploy workflow

Two machine roles:

| Role | Machine | Does |
|------|---------|------|
| **Dev / authoring** | this macOS box | edit code, lint, type-check, light CPU checks. **No GPU sim** — ManiSkill/SAPIEN GPU needs Linux+CUDA. |
| **Training** | Linux + NVIDIA server (RTX 5090 / Blackwell) | run ManiSkill sim + MARL training. Deployed via `git pull`. |

## Environment (uv)

- Python is pinned to **3.12** (`.python-version`). SAPIEN has no wheels for 3.13+.
- PyTorch: `cu128` build on Linux (Blackwell / RTX 50xx needs CUDA ≥12.8), CPU/MPS on macOS — handled automatically by `pyproject.toml` (`[tool.uv.sources]`).
- Dependencies are split into extras so the Mac doesn't try to install the Linux-only sim stack:
  - `sim`  — ManiSkill (Linux-only), PettingZoo
  - `train` — torch, tensordict, torchrl, benchmarl
  - `dev`  — ruff, pytest

### Local (macOS)

```bash
uv sync                       # base only (numpy, gymnasium)
make dev                      # + train + dev  (installs torch MPS build)
```

### Server (Linux + CUDA) — first time

```bash
git clone git@github.com:aidagroup/callosum.git
cd callosum
bash scripts/setup_server.sh   # uv + Python 3.12 + full env + GPU sanity checks
```

## Deploy loop (git-based)

```
edit locally  →  git commit  →  git push        (dev / macOS)
                                     │
                                     ▼
        git pull  →  uv sync --extra sim --extra train  →  run training   (server)
```

- **Code** flows dev → server via git push/pull (this repo).
- **Results** (checkpoints, logs) stay on the server; pull metrics via Weights & Biases or copy artifacts back with `rsync`/`scp`. Large artifacts are git-ignored (see `.gitignore`).
- The state-based baseline needs **no rendering**, so the server only needs PhysX/CUDA — no Vulkan/EGL display setup. Add that later when vision observations come in.

> Note: `uv.lock` is committed for reproducibility — the same resolved versions install on both machines.
