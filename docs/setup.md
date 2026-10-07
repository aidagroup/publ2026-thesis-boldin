# Dev & deploy workflow

Two machine roles:

| Role | Machine | Does |
|------|---------|------|
| **Dev / authoring** | this macOS box | edit code, lint, type-check, light CPU checks, **CPU simulation** in a throwaway env (see [Local](#local-macos)). **No GPU sim** – ManiSkill/SAPIEN GPU needs Linux+CUDA. |
| **Training** | lab GPU server (`culab.ru`), reached **only through the JupyterHub web UI and its terminal** (no SSH): 1× NVIDIA A100-SXM4-80GB, driver 570.172.08 (CUDA ≤ 12.8) | run ManiSkill sim + MARL training. Code arrives via `git clone` / `git pull` from GitHub. |

The training server is a JupyterHub single-user **container**, not a persistent
shell machine (image `jupyter/singleuser-gpu_570`, Ubuntu 24.04, user `jovyan`,
no root; no `tmux`/`screen`; `git`, `curl`, `pip`, `conda`, `gcc`, `nohup`,
`setsid` are there; probed 2026-08-30). What that means in practice:

- **`$HOME` (`/home/jovyan`) is a persistent 100 GB disk** (2026-10-07; it was 4.0 GB
  with 888 MB free on 2026-10-04). Results live there; the venv and caches still go to
  scratch (below), which is faster to recreate than to keep in sync with `$HOME`.
- **The overlay filesystem (`/`, so `/tmp`) is big**: 291 GB, 76 GB free
  (2026-10-04). It is assumed to be **wiped when the container restarts**
  (unconfirmed, so plan for it).
- **Only `$HOME` is assumed to survive a restart.** Hence the layout below: code,
  venv and caches in scratch on the overlay, results in `$HOME`.
- **Closing the browser tab kills the notebook kernel**, and with it anything
  started from a notebook cell. Long runs must be detached processes started
  from the terminal (File → New → Terminal), see
  [server-runbook.md](server-runbook.md#long-runs-detached).

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

In a JupyterHub terminal (File → New → Terminal), first time and after every
container restart:

```bash
S=/tmp/$(id -un)-callosum          # scratch; override with CALLOSUM_SCRATCH (export it first)
mkdir -p "$S" && cd "$S"
git clone https://github.com/aidagroup/callosum.git && cd callosum
bash scripts/setup_server.sh --smoke
```

Later: `cd "$CALLOSUM_REPO" && bash scripts/update_server.sh` (hard-syncs the checkout to the
latest pushed code, safe after force-pushes, then re-runs setup; options in
[server-runbook.md](server-runbook.md#1-environment-clone-or-update-then-setup)).
(`CALLOSUM_REPO` comes from `~/.callosum-env.sh`, below). The repo is cloned over
HTTPS because there is no SSH key on the server; if the repository is private,
git will ask for a username and a read-only token (check on the server).

The script is idempotent and never resolves dependencies — `uv sync --frozen`
installs exactly what `uv.lock` pins. It:

1. checks Linux + `nvidia-smi`, prints GPU and driver;
2. **picks a layout**. *Scratch mode* is used when `$HOME` has < 30 GB free (or
   `CALLOSUM_SCRATCH` / `JUPYTERHUB_USER` is set); on an ordinary Linux box with
   a big home it falls back to plain uv defaults (`.venv`, `~/.cache/uv`);
3. installs uv into `~/.local/bin` if missing (no root; installer from
   `astral.sh`, fallback `pip install --user uv`), Python 3.12, and syncs
   `sim` + `train` + `dev`;
4. fixes the two server quirks SAPIEN trips over — the unversioned `libcuda.so`
   and the Vulkan render device — only if the stock setup does not work, and
   pre-fetches SAPIEN's PhysX GPU library from GitHub (details in
   [server-runbook.md](server-runbook.md#server-quirks-sapien-physx-vulkan));
5. **verifies**: torch CUDA build, a real CUDA matmul (not just
   `is_available()`), the GPU's compute capability against the torch CUDA
   version, `mani_skill` import, our custom `so101_pg` agent (SO-ARM101 + parallel gripper), and that `TwoSO101-v0` /
   `FaceTurn-v0` actually register;
6. with `--smoke`, runs the GPU-backend checks that cannot run on macOS
   (`smoke_env.py`, `smoke_face_turn.py`).

### Disk layout on the server (scratch mode)

Scratch (`$CALLOSUM_SCRATCH`, default `/tmp/<user>-callosum`, on the big overlay
filesystem) is assumed to be wiped on restart; `$HOME` is small but persistent.

| What | Where | Survives restart? |
|---|---|---|
| repo checkout | `$CALLOSUM_SCRATCH/callosum` (cloned there) | no — `git clone` again |
| venv (`UV_PROJECT_ENVIRONMENT`) | `$CALLOSUM_SCRATCH/venv` | no |
| uv cache, uv-managed Python | `$CALLOSUM_SCRATCH/uv-cache`, `…/python` | no |
| Hugging Face cache (`HF_HOME`) | `$CALLOSUM_SCRATCH/hf` | no |
| ManiSkill assets (`MS_ASSET_DIR`; default is `~/.maniskill`) | `$CALLOSUM_SCRATCH/maniskill` | no |
| SAPIEN PhysX GPU lib (`~/.sapien`, hardcoded by SAPIEN) | symlink `~/.sapien` → `$CALLOSUM_SCRATCH/sapien` | link yes, target no |
| other caches (pip, torch hub, matplotlib via `XDG_CACHE_HOME`) | `$CALLOSUM_SCRATCH/cache` | no |
| uv binary | `~/.local/bin` (~50 MB) | yes |
| **results** `runs/` | symlink `<checkout>/runs` → `~/callosum-runs` (`CALLOSUM_RUNS`) | **yes** |
| env file for new terminals | `~/.callosum-env.sh`, sourced from `~/.bashrc` | yes (but stale after a wipe) |

Consequences: nothing under the checkout is precious (never edit code on the
server; push from the Mac); re-running setup after a wipe re-downloads torch +
CUDA wheels (several GB; time not measured yet), so keep that in mind when
planning a session. `~/callosum-runs` is on the persistent `$HOME`, so run
directories (full-state checkpoints, ~7 MB each) survive a restart and
`python -m callosum.training.ippo --resume runs/<name>` continues a killed run.

Without `~/.callosum-env.sh` sourced, `uv run` does not know about the venv in
scratch and silently builds a second one inside the checkout. The setup script
removes such a stray `.venv`; new terminals source the file via `~/.bashrc`.

Rendering: SAPIEN needs a **Vulkan render device even for state-only
observations** (its URDF loader builds `RenderMaterial()` unconditionally;
`render_backend="none"` only removes the renderer from the scene). The setup
script checks this and, if needed, installs a newer Vulkan loader with conda
(see the runbook).

## Network access on the server

Outbound internet on the server is **allowlisted**, not open. Currently reachable:

| Host | Used for |
|------|----------|
| `github.com` | `git clone` / `git pull` (HTTPS); uv's Python download; SAPIEN's PhysX GPU library |
| `pypi.org`, `files.pythonhosted.org` | `uv sync` (regular packages) |
| `download.pytorch.org` | `uv sync` (torch `cu128` wheels) |
| `astral.sh` | uv installer in `setup_server.sh` |
| `huggingface.co` | model weights (V-JEPA, phase 2) |
| `llm-proxy.spirit.culab.ru` | LLM API proxy, usable from the server |
| `ultralytics.com`, `kaggle.com` | not used by this project |

Use `llm-proxy.spirit.culab.ru` from the server for any LLM tooling; ask the
server owner for the API format.

Anything else is blocked, notably `wandb.ai` (so there is no hosted experiment
tracker) and, **unverified**, `conda.anaconda.org` (conda-forge, used only by the
Vulkan-loader fallback in the setup script). The server owner can open
additional hosts on request. Before adding a dependency or service that
downloads from a new host, check it is reachable from the server:

```bash
curl -sI https://<host> | head -1
```

Measured from the server's terminal on 2026-10-04: `download.pytorch.org/whl/cu128/`
→ `HTTP/2 200`; `download-r2.pytorch.org` → `404` on `/` (the host is reachable).
That matters: `download.pytorch.org` redirects the actual wheel download to
`download-r2.pytorch.org`, which an earlier probe (2026-08-30) could not reach, so
the first `uv sync` on the server is the real confirmation that the `cu128` wheels
from `uv.lock` install. If it fails on that host, ask the owner to open it; do not
install torch from PyPI instead (its Linux default targets CUDA 13, beyond what
driver 570 can run).

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
edit locally  →  git commit  →  git push                              (dev / macOS)
                                    │
                                    ▼
JupyterHub terminal:  bash scripts/update_server.sh  →  detached run
                                    │
                                    ▼
JupyterHub file browser: download results                              (→ dev / macOS)
```

- **Code** flows dev → server via git push/pull (this repo). There is no rsync/scp/SSH to the server; never edit code on the server (its checkout is disposable).
- **Results** (checkpoints, TensorBoard logs) are written to `runs/`, which on the server is a symlink into `$HOME` so it survives a container restart (git-ignored). Pull them back with the JupyterHub file browser (right-click → Download; tar up a directory first) and view TensorBoard either through JupyterHub or locally (see [server-runbook.md](server-runbook.md#4-watching-and-collecting-results)). `$HOME` is a persistent 100 GB disk, so checkpoints may stay there. There is no hosted experiment tracker: `wandb.ai` is not on the server's allowlist.
- The state-based baseline needs no camera rendering, but SAPIEN still needs a Vulkan device (see the Server section). Camera/vision observations come in later.

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
