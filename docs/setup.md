# Dev & deploy workflow

Two machine roles:

| Role | Machine | Does |
|------|---------|------|
| **Dev / authoring** | this macOS box | edit code, lint, type-check, light CPU checks. **No GPU sim** – ManiSkill/SAPIEN GPU needs Linux+CUDA. |
| **Training** | Linux + NVIDIA server (RTX 5090 / Blackwell) | run ManiSkill sim + MARL training. Deployed via `git pull`. |

## Environment (uv)

- Python is pinned to **3.12** (`.python-version`). SAPIEN has no wheels for 3.13+.
- PyTorch: `cu128` build on Linux (Blackwell / RTX 50xx needs CUDA ≥12.8), CPU/MPS on macOS – handled automatically by `pyproject.toml` (`[tool.uv.sources]`).
- Dependencies are split into extras so the Mac doesn't try to install the Linux-only sim stack:
  - `sim`  – ManiSkill (Linux-only), PettingZoo
  - `train` – torch, tensordict, torchrl, benchmarl
  - `dev`  – ruff, pytest

### Local (macOS)

```bash
uv sync                       # base only (numpy, gymnasium)
make dev                      # + train + dev  (installs torch MPS build)
```

### Server (Linux + CUDA) – first time

```bash
git clone git@github.com:aidagroup/callosum.git
cd callosum
bash scripts/setup_server.sh   # uv + Python 3.12 + full env + GPU sanity checks
```

## Packaging

`callosum` is a real (hatchling-built) package, installed **editable** into the venv by
`uv sync`. That makes `import callosum` behave identically from pytest, `scripts/*.py`,
notebooks, and on the server — no `sys.path`/cwd juggling — while edits still take effect
immediately without reinstalling.

## CI

`.github/workflows/ci.yml` runs on every PR and on pushes to `main`: `ruff check`,
`ruff format --check`, an import smoke check, and `pytest`. It installs only the `dev`
extra — the `sim`/`train` extras pull ManiSkill and a multi-GB CUDA torch build, so
anything touching the simulator is verified on the training server instead.

## Deploy loop (git-based)

```
edit locally  →  git commit  →  git push        (dev / macOS)
                                     │
                                     ▼
        git pull  →  uv sync --extra sim --extra train  →  run training   (server)
```

- **Code** flows dev → server via git push/pull (this repo).
- **Results** (checkpoints, logs) stay on the server; pull metrics via Weights & Biases or copy artifacts back with `rsync`/`scp`. Large artifacts are git-ignored (see `.gitignore`).
- The state-based baseline needs **no rendering**, so the server only needs PhysX/CUDA – no Vulkan/EGL display setup. Add that later when vision observations come in.

> Note: `uv.lock` is committed for reproducibility – the same resolved versions install on both machines.

## CUDA index: why `cu128`, and when to revisit

**Finding (audited 2026-07-28):** the `cu128` wheel index tops out at **torch 2.11.0**. Newer
CUDA indexes carry newer torch:

| Index | Newest torch |
|-------|--------------|
| `cu128` (current) | 2.11.0 ← our ceiling |
| `cu129` | 2.13.0 |
| `cu130` | 2.13.0 |

`cu128` is the **oldest CUDA that supports Blackwell / RTX 5090** (sm_120), so it has the
**lowest NVIDIA driver requirement**. We deliberately stay on it until the training server
exists and its driver version is known — a newer CUDA build installs fine but fails at
runtime on an older driver.

**TODO when server access lands:** run `nvidia-smi`, check the driver version, and if it
supports CUDA 12.9/13.0, consider moving to `cu129`/`cu130` for torch ≥2.13. That upgrade
must move `torch` + `torchrl` + `tensordict` **together** (they are released in lockstep),
and be verified against ManiSkill/SAPIEN on the server.

Bonus: torch ≥2.13 also clears GHSA-rrmf-rvhw-rf47 (`torch.jit.script` memory corruption —
local-only, negligible for our use since we never script untrusted input).
