# Server runbook (training session)

The ordered checklist for a GPU session on the lab server (see
[setup.md](setup.md) for the machine, network allowlist, disk layout and
environment). Write and verify everything possible on macOS first (including
CPU-sim smoke checks, see [setup.md](setup.md#local-macos)); use server time
for things that genuinely need a GPU.

## Before the session

- [ ] All code for the session is merged into `main` and pushed.
- [ ] CI green on `main`; `make dev && uv run pytest -q` passes locally.
- [ ] `uv.lock` committed and in sync with `pyproject.toml` (the server installs
      with `--frozen` and will refuse to re-resolve).
- [ ] Any new dependency or download comes from a host on the server's
      allowlist (see [setup.md](setup.md#network-access-on-the-server)); if not,
      ask the server owner to open it first.
- [ ] You know which experiments you intend to run (below).

## Server

| Setting | Value |
|---|---|
| GPU | 1× NVIDIA A100-SXM4-80GB |
| Driver | 570.172.08 (CUDA ≤ 12.8 → torch `cu128`) |
| Access | JupyterHub web UI + its terminal only; **no SSH**; user `jovyan`, no root |
| Internet | allowlist only (GitHub, PyPI, PyTorch, Hugging Face, lab LLM proxy, …) |
| `$HOME` | 4.0 GB total, 888 MB free (2026-10-04); persistent (assumed) |
| `/` overlay (`/tmp`) | 291 GB, 76 GB free (2026-10-04); assumed wiped on restart |
| Missing tools | `uv` (setup installs it), `tmux`, `screen` |

## Access (JupyterHub, no SSH)

Everything runs in a **terminal** opened from the JupyterHub UI
(File → New → Terminal), not in notebook cells: the notebook kernel dies when the
browser tab closes, and so does anything it started. Files move only through the
JupyterHub file browser (upload / right-click → Download); code moves through
`git` from GitHub. There is no SSH alias, `scp` or `rsync` to the server.

Layout (details in [setup.md](setup.md#disk-layout-on-the-server-scratch-mode)):
checkout, venv and caches in scratch `/tmp/<user>-callosum` (big, assumed wiped on
restart), results in `~/callosum-runs` (small, persistent) reached through the
`runs/` symlink in the checkout.

## Session order

### 1. Environment: clone or update, then setup

Open a terminal and look at what survived:

```bash
ls ~/.callosum-env.sh ~/callosum-runs      # these persist across restarts
ls -d /tmp/$(id -un)-callosum/callosum    # scratch checkout: gone after a restart
```

**Scratch is gone (first session, or after a restart)** — start from a fresh clone:

```bash
S=/tmp/$(id -un)-callosum                  # or export CALLOSUM_SCRATCH=<dir> first
mkdir -p "$S" && cd "$S"
git clone https://github.com/aidagroup/callosum.git && cd callosum
# git checkout <branch>                    # if the session is not on main
bash scripts/setup_server.sh
```

**Scratch is still there** — update in place:

```bash
source ~/.callosum-env.sh
cd "$CALLOSUM_REPO"
bash scripts/update_server.sh              # fetch + hard-sync, then setup_server.sh
```

`update_server.sh` updates the checkout to the latest pushed code and then re-runs
`scripts/setup_server.sh` (idempotent; quick when nothing changed). It is a hard
sync, not a `git pull`: `git fetch --prune origin` (time-bounded), then
`git checkout -B <branch> origin/<branch>`, so a force-pushed (rewritten) remote
history is no problem. It prints old → new commit and a `git diff --stat`.

```bash
bash scripts/update_server.sh                        # current branch
bash scripts/update_server.sh --branch main          # switch branch (needed on a detached HEAD)
bash scripts/update_server.sh --force                # discard uncommitted changes to tracked files
bash scripts/update_server.sh --no-setup             # only update the code
bash scripts/update_server.sh -- --smoke             # args after `--` go to setup_server.sh
```

Uncommitted changes to tracked files are refused (printed) unless `--force`; code is
never edited on the server. Untracked and ignored files (the `runs` symlink, logs)
are kept: there is no `git clean`. If `update_server.sh` itself changed in the
update, the new version is re-executed automatically. The exit code is that of
`setup_server.sh`.

**One-time, for a checkout that predates this script or still tracks the rewritten
history** (the script cannot run, or `git pull` complains about divergent branches):

```bash
cd "$CALLOSUM_REPO"
git fetch origin && git checkout -B step/1.5-parallel-gripper origin/step/1.5-parallel-gripper
bash scripts/update_server.sh              # from now on, use the script
```

(Add `-f` to `git checkout` if it refuses because of local changes to tracked files;
they are disposable.)

The script verifies driver, a real CUDA matmul, compute capability vs the torch
CUDA build, ManiSkill + our `so101_pg` agent, and that both envs register. Stop and
fix if anything here fails — everything below depends on it. The very first
`uv sync` is also the confirmation that the `cu128` torch wheels download from
`download.pytorch.org` (see the network notes in [setup.md](setup.md#network-access-on-the-server)).
After a restart the uv cache is gone, so setup re-downloads torch and the CUDA
libraries (several GB); expect it to take minutes, not seconds.

If the repository turns out to be private, `git clone` over HTTPS asks for a
username and a read-only token (verify on the server).

In every **new** terminal, `~/.callosum-env.sh` is sourced automatically via
`~/.bashrc` (verify that terminals read it; otherwise `source` it by hand). It
sets `UV_PROJECT_ENVIRONMENT`, the caches, `MS_ASSET_DIR` and the Vulkan/libcuda
variables. A notebook kernel that was started before setup does not have them;
restart it or work in the terminal.

### 2. Phase-1 debt: the GPU-backend checks

```bash
bash scripts/setup_server.sh --smoke       # or, once set up:
uv run python scripts/smoke_env.py
uv run python scripts/smoke_face_turn.py
uv run python scripts/probe_face_turn.py
```

Pre-check the same scripts on the Mac CPU sim first (`--sim-backend cpu`,
one env, command in [setup.md](setup.md#local-macos)); the arm-reach and
gripper questions below can already be answered there. The server run confirms
them on the GPU backend. `probe_face_turn.py` is a scripted two-arm expert (no
teleporting): the holder clamps the cube body, the rotator clamps the face
layer top-down and rolls its wrist by 90 degrees. It prints, per phase, the
grasp flags, face angle, body drift and `success`, and exits non-zero unless
every env ends with `success=True`. On the Mac CPU sim it succeeds (20/20 reset
seeds); on the GPU backend it is still unverified.

**Known open questions these answer** (all flagged in code as `TODO(review)`):

| Question | Where | If it fails |
|---|---|---|
| Do both arms reach the cube in the 90-degree layout (holder 0.32 m on -y, rotator 0.24 m on +x, `ArmLayout` in `callosum/configs/layout.py`)? Can check on the Mac CPU sim. | `smoke_env.py`, `probe_face_turn.py` | adjust the radii/azimuths in `ArmLayout`; SO-101 top-down reach at cube height is ~0.27 m |
| Can a holder and a rotator arm do the face turn together without colliding (holder clamps the body, rotator clamps the face and rolls 90 degrees, body stays put)? Can check on the Mac CPU sim. | `probe_face_turn.py` | check which arm links collide (the wrist housings are 12.8 cm wide); adjust the layout or the probe's grasp parameters |
| Does the face articulation look/behave right (no jitter, face sits on the body)? | `smoke_face_turn.py` | check `disable_self_collisions`; review the joint pose |
| Are the joint friction/damping sane at cube scale? | `smoke_face_turn.py` | tune `friction`/`damping` in `_turntable_cube.py` |
| Does the parallel gripper actually close on the layer's side faces (`is_grasping`)? Can check on the Mac CPU sim. | `smoke_face_turn.py` | revisit the grasp target in `face_turn.compute_dense_reward` |
| Does the scripted turn flip `success` on, and body displacement flip it off? | `smoke_face_turn.py` | success logic bug — fix before any training |

Record the actual printed output; it is the evidence that phase 1 works.

### 3. First training run — the "does it learn at all" gate

The trainer is `callosum/training/ippo.py` (step 2.1): two independent PPO learners, one per
arm, on one shared GPU simulation (details in its module docstring; every flag is a field of
`callosum/configs/ippo.py`, `--help` lists them). Both arms train on the env's shared
*normalised dense* reward; each arm's policy input is cut out of the flat state observation
(own joint state, task state, TCP poses according to `--partner-obs`), see
`callosum/training/_agent_obs.py`. It prints, at start, the exact input fields of each arm: check
that the lines look right before trusting a long run.

Defaults worth knowing: `--num-envs 256` (raise on the A100 once it runs), `--num-steps 100`
(batch 25 600), `--gamma 0.99` (not ManiSkill's 0.8: 5-step horizon), `--max-episode-steps 300`
(the registered 100 is too short for the face turn: the scripted probe needs ~330 steps, and
`--max-episode-steps none` keeps the registered value), evaluation every 20 iterations with
16 envs over a full episode, `--partner-obs full`.

Start with the **easy** env, not the hard one: `TwoSO101-v0` has a pure reach reward, so if
IPPO can't improve there, the problem is the trainer, not the task. Launch both detached (see
"Long runs" below for the details and how to stop a run):

```bash
source ~/.callosum-env.sh && cd "$CALLOSUM_REPO"

# 1. Sanity: reach only. 1M steps = 39 iterations (a few minutes on the A100).
NAME=twoso101_sanity; mkdir -p runs/$NAME
setsid nohup uv run python -m callosum.training.ippo \
    --env-id TwoSO101-v0 --total-timesteps 1000000 --exp-name $NAME \
    > runs/$NAME/stdout.log 2>&1 < /dev/null &
echo $! > runs/$NAME/pid

# 2. The real target, after the sanity run shows a rising reward.
NAME=faceturn_s1; mkdir -p runs/$NAME
setsid nohup uv run python -m callosum.training.ippo \
    --env-id FaceTurn-v0 --total-timesteps 10000000 --seed 1 --exp-name $NAME \
    > runs/$NAME/stdout.log 2>&1 < /dev/null &
echo $! > runs/$NAME/pid
```

`--exp-name` is the run directory name under `runs/`; the trainer reuses an existing directory
(the shell created it for `stdout.log`). Watch it:

```bash
tail -f runs/$NAME/stdout.log        # Ctrl-C leaves the run alone
```

One line per iteration: `iter 12/390 | step 307200 | sps 3700 | ret 31.2 | succ 0.04 (n=35) |
ent a/b 5.1/5.2 | kl a/b 0.01/0.02 | eval_succ 0.00 | eval_ret 28.1`. `ret`/`succ` are the
training episodes that finished during the iteration (`n` = how many; `ret -` / `n=0` until the
first one ends, and `ret`/`succ` then keep their last value); `eval_*` are from the latest
evaluation (an `eval @ iter ...` line is printed each time, also once before training starts
as the untrained baseline). TensorBoard tags (section 4): `train/{return,success_once,...}`,
`eval/{return,success_once,success_at_end,...}`, `losses/agent_{a,b}/{policy_loss,value_loss,
entropy,approx_kl,clipfrac,explained_variance}`, `policy/agent_{a,b}/action_std`, `charts/SPS`.
Files in `runs/$NAME/`: `config.json`, `events.out.tfevents.*`, `latest.pt` (every 20
iterations and at the end), `best.pt` (best evaluation success, then return): about 2.3 MB each,
both agents' weights plus the config, no optimizer state (`--checkpoint <file>` warm-starts from
the weights).

Local pre-check on the Mac CPU sim (one env, a few seconds; also what the trainer was
verified with before the server run):

```bash
PYTHONPATH=. uv run -q --no-project --python 3.12 --with mani-skill==3.0.1 --with torch --with tensorboard \
    python -m callosum.training.ippo --env-id FaceTurn-v0 --sim-backend cpu --num-envs 1 \
    --total-timesteps 600 --num-minibatches 4 --eval-freq 3 --exp-name smoke   # then: rm -r runs/smoke
```

Success criterion (from `docs/implementation-plan.md`, step 2.1 / the experiment design):

* `TwoSO101-v0`: `train/return` and `eval/return` rise clearly over the run (reach reward
  only, no success flag; an early run on the old SO-100 went from -75 to -40 per 300-step
  episode in 1M steps; for comparison the untrained policy scores about -26 per 100 steps on
  the Mac CPU sim), `explained_variance` goes up towards 1, `approx_kl` stays below
  `target_kl` (0.1).
* `FaceTurn-v0`: reward rises and `eval/success_once` is non-trivial (clearly above 0).
  This is the gate that decides whether the whole approach is viable. If the reward rises
  but success stays 0, look at the reward terms before touching the trainer (see the notes in
  the step-2.1 report: holder reward, body-drift penalty).

Expected throughput: the archived SO-100 trainer measured about 3700 steps/s on the A100 with
256 envs (2300 -> 3700 over the first iterations), i.e. 10M steps in under an hour; unmeasured
for the SO-101 envs and for this trainer, so record the real `sps` here after the first run.

**Rule: every state-based env must be created with `render_backend="none"`**
(`gym.make(..., obs_mode="state", render_backend="none")`); the trainer
(`ippo.make_envs`) passes it. ManiSkill v3.0.1 defaults to `"gpu"`, which makes `BaseEnv._setup_scene`
build `sapien.render.RenderSystem(<cuda device>)`; with `"none"` the render device is
`None`, so no `RenderSystem`, lighting or sensors are created. Without it, `gym.make`
fails with `RuntimeError: Failed to find a supported physical device "cuda:0"` on a
machine that only has the lavapipe software ICD (as the lab A100 does, see
[Server quirks](#server-quirks-sapien-physx-vulkan)). The smoke/probe scripts pass it
via `--render-backend` (default `none`); use `--render-backend gpu` only for
rendering/vision, which needs a working hardware Vulkan device.

#### Long runs (detached)

A run started from a notebook cell dies with the kernel when the tab closes. In a
**terminal**, detach it from the terminal with `setsid nohup` (the exact commands for the
two runs are in section 3; the pattern is):

```bash
source ~/.callosum-env.sh && cd "$CALLOSUM_REPO"
NAME=<run name>                               # one directory per run
mkdir -p runs/$NAME
setsid nohup uv run python -m callosum.training.ippo \
    --env-id <env> --exp-name $NAME [flags] \
    > runs/$NAME/stdout.log 2>&1 < /dev/null &
echo $! > runs/$NAME/pid                      # best effort; see pgrep below
```

Check and stop it:

```bash
tail -f runs/$NAME/stdout.log                 # Ctrl-C leaves the run alone
pgrep -af callosum.training                   # alive? (pid, command line)
nvidia-smi                                    # GPU utilisation and memory
pkill -TERM -f "python -m callosum.training.ippo"   # polite stop (SIGTERM), see below
kill <pid>                                    # same, if the pid file is right
```

SIGTERM makes the trainer finish the current iteration, run an evaluation and write
`latest.pt` before exiting (`kill -9` loses everything since the last periodic save).

`stdout.log` lives in `$HOME` via the `runs/` symlink, so it survives a restart
(the run itself does not). Do not run `update_server.sh` / `git pull` / switch branches mid-run: `callosum`
is installed editable, so later imports would see the new code.

**Verify on the server first** that detaching really works: `setsid nohup sleep 600 &`,
close the browser tab, reopen a terminal, `pgrep sleep`. It must still be there. Also
unknown: whether the lab kills long-running processes or idle containers; make
trainers save a checkpoint periodically rather than only at the end.

If a notebook is more convenient than a terminal for reading, `!tail -40
runs/<name>/stdout.log` in a cell is fine (reading only; never start the run there).

### 4. Watching and collecting results

Checkpoints and TensorBoard logs live under `runs/` (→ `~/callosum-runs`) and are
git-ignored, so they do **not** come back via git. `wandb.ai` is not reachable
from the server, so metrics stay local. **`$HOME` has < 1 GB free**: check
`du -sh runs/*` after each run, delete checkpoints you will not use, and download
the keepers promptly (the quota is shared with everything else in `$HOME`).

TensorBoard has no SSH tunnel to ride on. Two options:

1. **Through JupyterHub** — only if `jupyter-server-proxy` is installed (verify:
   `pip list 2>/dev/null | grep -i jupyter-server-proxy` and
   `jupyter server extension list`). `tensorboard` is in the `train` extra
   (installed by `setup_server.sh`):

   ```bash
   cd "$CALLOSUM_REPO"
   uv run tensorboard --logdir runs --port 6006 --host 127.0.0.1 &
   ```

   then open `<your JupyterHub URL>/user/<you>/proxy/6006/` (the standard
   jupyter-server-proxy path; verify). Stop it with `pkill -f tensorboard`.
2. **Copy the logs out and view them on the Mac** — works regardless of any proxy.
   Event files are small; leave the weights out:

   ```bash
   tar czf ~/runs-logs.tar.gz --exclude='*.pt' -C "$CALLOSUM_REPO" runs
   du -sh ~/runs-logs.tar.gz                   # must fit in the 888 MB quota
   ```

   Download `runs-logs.tar.gz` from the file browser (right-click → Download),
   `rm` it on the server, then locally: `tar xzf runs-logs.tar.gz && uvx tensorboard --logdir runs`.

Weights go the same way, one checkpoint at a time (`tar czf` or direct Download).
`runs/<name>/stdout.log` is the always-available fallback for watching a run.

Metrics worth writing into `docs/thesis/` while fresh: success-rate, steps to
converge, and any layout/tuning constants that had to change.

## Server quirks (SAPIEN, PhysX, Vulkan)

Carried over from the earlier SO-100-era setup of this same server (an archived
branch, probed 2026-08-30). `scripts/setup_server.sh` applies them only when the
stock configuration fails; **none has been re-verified yet on the current
environment** — treat the first setup run as the verification.

| Symptom | Cause | What setup does |
|---|---|---|
| `OSError: libcuda.so: cannot open shared object file` from `sapien/physx/__init__.py` (`enable_gpu`), while torch sees the GPU | SAPIEN loads the *unversioned* `libcuda.so`; the container runtime only injects `libcuda.so.1` | symlinks `libcuda.so.1` into `<scratch>/lib/libcuda.so`, prepends it to `LD_LIBRARY_PATH` |
| `vkCreateInstance: Found no drivers!` / `Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr'` when creating *any* env, even with `obs_mode="state"` | SAPIEN's URDF loader builds `RenderMaterial()` unconditionally, so a Vulkan device is mandatory; the system `libvulkan.so.1` (1.3.275) was too old for the 570.x NVIDIA ICD | tries the stock setup, then each ICD manifest with the system loader, then installs a current loader with `conda create -p <scratch>/mesa -c conda-forge mesalib vulkan-tools` (conda package cache moved off `$HOME`), symlinks **only** `libvulkan.so.1` into `<scratch>/vklib` (conda's whole `lib/` would shadow `libstdc++` and break torch), and picks the first working manifest by actually constructing a `RenderMaterial`; lavapipe (software) is the last resort — fine for state training with `render_backend="none"`, not for vision. **Observed 2026-10-04 on the lab A100: hardware Vulkan did not work, setup fell back to lavapipe (`VK_ICD_FILENAMES=lvp_icd...`)**; `RenderMaterial()` works with it, but `gym.make(..., sim_backend="gpu")` with the default `render_backend` fails in `_setup_scene` (`Failed to find a supported physical device "cuda:0"`), hence the `render_backend="none"` rule above |
| First `gym.make(..., sim_backend="gpu")` hangs/fails while downloading | SAPIEN fetches `libPhysXGpu_64.so` (~240 MB unpacked) from a `github.com` release into `~/.sapien/physx/<version>/` | `~/.sapien` is a symlink into scratch; setup pre-fetches via `physx.enable_gpu()`. Manual fallback: download the `linux-so.zip` named in SAPIEN's message elsewhere, upload, unzip into that directory |

Notes:

- `SAPIEN_VULKAN_LIBRARY_PATH` looked like the knob for a custom loader but was
  ignored when a system `libvulkan` exists (observed earlier); the loader is
  swapped through `LD_LIBRARY_PATH` instead. The loader is chosen at process
  start, so run from a terminal that sourced `~/.callosum-env.sh`.
- The conda fallback needs `conda.anaconda.org` (conda-forge) — **not** on the
  owner's list of open hosts; verify on the server, or ask the owner to open it.
- ManiSkill's own asset directory is controlled by `MS_ASSET_DIR` (v3.0.1;
  default `~/.maniskill`). The table, robot URDFs/meshes and our procedural cube
  ship inside the `mani_skill` wheel (checked for 3.0.1 earlier), so nothing
  extra should be downloaded for these envs; verify no `~/.maniskill` appears.
- `torch` is **not** to be installed from PyPI as a workaround for a blocked
  `download-r2.pytorch.org`: PyPI's default Linux build targets CUDA 13, which
  driver 570 cannot run. Ask the owner to open the host instead.

## Verify on the server first

Everything above that is not a measured fact, in the order it will bite:

1. `bash scripts/setup_server.sh` completes: `uv sync --frozen` pulls the `cu128`
   wheels (torch, CUDA libs) through `download-r2.pytorch.org`; `uv` installs via
   `pip --user` from PyPI (fallback: the astral.sh installer, which fetches from github.com); uv's Python 3.12 download from GitHub works.
2. Which of the quirk fixes (libcuda shim, Vulkan loader, PhysX download) the script
   actually had to apply; whether conda-forge is reachable.
3. `~/.bashrc` is read by new JupyterHub terminals (so `~/.callosum-env.sh` loads).
4. `setsid nohup` runs survive closing the tab; whether the lab kills long processes.
5. `jupyter-server-proxy` present? (for TensorBoard)
6. Is `/tmp` really wiped on restart, and is `$HOME` really kept? Is the repo public
   (HTTPS clone without credentials)?
7. How long a from-scratch setup takes after a wipe (add the number here).
