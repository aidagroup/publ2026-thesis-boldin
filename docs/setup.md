# Dev & deploy workflow

Two machine roles:

| Role | Machine | Does |
|------|---------|------|
| **Dev / authoring** | this macOS box | edit code, lint, type-check, light CPU checks, **CPU simulation** in a throwaway env (see [Local](#local-macos)). **No GPU sim** – ManiSkill/SAPIEN GPU needs Linux+CUDA. |
| **Training** | lab GPU server (`culab.ru`): 1× NVIDIA A100-SXM4-80GB, driver 570.172.08 (CUDA ≤ 12.8) | run ManiSkill sim + MARL training. Deployed via `git pull`. |

The training server is a persistent machine (not a rented pod): the checkout,
`.venv`, uv cache and `runs/` survive between sessions.

## Environment (uv)

- Python is pinned to **3.12** (`.python-version`). SAPIEN has no wheels for 3.13+.
- PyTorch: `cu128` build on Linux, CPU/MPS on macOS – handled automatically by `pyproject.toml` (`[tool.uv.sources]`). See [CUDA index](#cuda-index-why-cu128) for why `cu128`.
- Dependencies are split into extras so the Mac doesn't try to install the Linux-only sim stack:
  - `sim`  – ManiSkill (Linux-only), PettingZoo
  - `train` – torch, tensordict, torchrl, benchmarl
  - `dev`  – ruff, pytest

### Local (macOS)

```bash
uv sync                       # base only (numpy, gymnasium)
make dev                      # + train + dev  (installs torch MPS build)
```

The project venv does **not** include ManiSkill on macOS (`pyproject.toml` marks
it Linux-only). ManiSkill 3.0.1 does install and run **CPU simulation** on the
Mac in a throwaway env, which is the first check for anything sim-related
(scene loads, rest pose is stable, both arms reach the cube, gripper grasp):

```bash
PYTHONPATH=. uv run --no-project --python 3.12 --with mani-skill==3.0.1 --with torch \
    python scripts/smoke_env.py --sim-backend cpu --num-envs 2
```

The smoke scripts take `--sim-backend` / `--num-envs`. GPU simulation and
training still need the server.

### Server (Linux + CUDA)

```bash
git clone git@github.com:aidagroup/callosum.git   # first time only
cd callosum
bash scripts/setup_server.sh --smoke
```

The script is idempotent (safe to re-run after every `git pull`) and never
resolves dependencies — `uv sync --frozen` installs exactly what `uv.lock` pins.
It:

1. checks Linux + `nvidia-smi`, prints GPU and driver;
2. installs uv (if missing) + Python 3.12 and syncs `sim` + `train` + `dev`;
3. **verifies**: torch CUDA build, a real CUDA matmul (not just
   `is_available()`), the GPU's compute capability against the torch CUDA
   version, `mani_skill` import, our custom `so101_pg` agent (SO-ARM101 + parallel gripper), and that `TwoSO101-v0` /
   `FaceTurn-v0` actually register;
4. with `--smoke`, runs the GPU-backend checks that cannot run on macOS
   (`smoke_env.py`, `smoke_face_turn.py`).

Rendering (needed later, for the vision phase, not for state-based training)
additionally requires GL/EGL libs, e.g.
`apt-get install -y libgl1 libglvnd0 libegl1-mesa libgles2-mesa libopengl0`
(ask the server owner if there is no root access).

## Network access on the server

Outbound internet on the server is **allowlisted**, not open. Currently reachable:

| Host | Used for |
|------|----------|
| `github.com` | `git clone` / `git pull` |
| `pypi.org`, `files.pythonhosted.org` | `uv sync` (regular packages) |
| `download.pytorch.org` | `uv sync` (torch `cu128` wheels) |
| `astral.sh` | uv installer in `setup_server.sh` |
| `huggingface.co` | model weights (V-JEPA, phase 2) |
| `llm-proxy.spirit.culab.ru` | LLM access via the lab proxy |
| `ultralytics.com`, `kaggle.com` | not used by this project |

Anything else (notably `wandb.ai`) is blocked; the server owner can open
additional hosts on request. Before adding a dependency or service that
downloads from a new host, check it is reachable from the server:

```bash
curl -sI https://<host> | head -1
```

If `git@github.com` (SSH, port 22) is blocked while HTTPS works, clone via
`https://github.com/aidagroup/callosum.git` instead.

## CI

`.github/workflows/ci.yml` runs on every PR and on pushes to `main`: `ruff check`,
`ruff format --check`, an import smoke check, and `pytest`. It installs only the `dev`
extra — the `sim`/`train` extras pull ManiSkill and a multi-GB CUDA torch build, so
anything touching the simulator is verified on the training server instead.

## Packaging

`callosum` is a real (hatchling-built) package, installed **editable** into the venv by
`uv sync`. That makes `import callosum` behave identically from pytest, `scripts/*.py`,
notebooks, and on the server — no `sys.path`/cwd juggling — while edits still take effect
immediately without reinstalling.

## Deploy loop (git-based)

```
edit locally  →  git commit  →  git push        (dev / macOS)
                                     │
                                     ▼
        git pull  →  bash scripts/setup_server.sh  →  run training   (server)
```

- **Code** flows dev → server via git push/pull (this repo).
- **Results** (checkpoints, TensorBoard logs) are written to `runs/` on the server and are git-ignored. View them live through an SSH tunnel (see [server-runbook.md](server-runbook.md)) or copy them back with `rsync`. There is no hosted experiment tracker: `wandb.ai` is not on the server's allowlist.
- The state-based baseline needs **no rendering**, so the server only needs PhysX/CUDA – no Vulkan/EGL display setup. Add that later when vision observations come in.

> Note: `uv.lock` is committed for reproducibility – the same resolved versions install on both machines.

## CUDA index: why `cu128`

The server's driver (570.172.08) supports CUDA **up to 12.8**. A torch build for a
newer CUDA installs fine but fails at runtime on this driver, so `cu128` is the
ceiling. The A100 (Ampere, compute capability 8.0) is supported by every current
CUDA build, so the GPU itself imposes no lower bound.

The `cu128` wheel index tops out at **torch 2.11.0** (audited 2026-07-28):

| Index | Newest torch |
|-------|--------------|
| `cu128` (current) | 2.11.0 ← our ceiling |
| `cu129` | 2.13.0 |
| `cu130` | 2.13.0 |

Moving to `cu129`/`cu130` (torch ≥2.13) requires the server owner to upgrade the
NVIDIA driver first. If that happens, `torch` + `torchrl` + `tensordict` must move
**together** (they are released in lockstep), and the result must be verified against
ManiSkill/SAPIEN on the server.

Bonus of a future upgrade: torch ≥2.13 also clears GHSA-rrmf-rvhw-rf47
(`torch.jit.script` memory corruption — local-only, negligible for our use since we
never script untrusted input).
