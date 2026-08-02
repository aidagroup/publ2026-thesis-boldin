# Server runbook (rented GPU session)

The GPU is rented **by the hour**, so the goal is: write and verify everything
possible beforehand, and spend paid time only on things that genuinely need a
GPU. This file is the ordered checklist for a session — follow it top to bottom.

## Before starting the pod

- [ ] All code for the session is merged into `main` and pushed.
- [ ] CI green on `main`; `make dev && uv run pytest -q` passes locally.
- [ ] `uv.lock` committed and in sync with `pyproject.toml` (the server installs
      with `--frozen` and will refuse to re-resolve).
- [ ] You know which experiments you intend to run (below), so the pod isn't
      idle while we decide.

## Pod spec

| Setting | Value |
|---|---|
| Image | any CUDA ≥ 12.8 Linux image, e.g. `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| GPU | RTX 5090 (32 GB) or any ≥24 GB NVIDIA card |
| Volume | persistent volume mounted at `/workspace` (keeps the uv cache between pods) |

The image's own PyTorch is unused — `uv sync` installs our locked build.

## SSH access

Add the pod to `~/.ssh/config` on the dev machine under a stable alias, so the
key never has to be passed around and every session uses the same command:

```
Host aida-gpu
    HostName <pod-host>
    Port <pod-port>
    User root
    IdentityFile ~/.ssh/<your-key>
    ServerAliveInterval 30
```

Then everything runs as `ssh aida-gpu '<command>'`. Update `HostName`/`Port`
when a new pod is created — the alias stays the same.

## Session order

### 1. Environment (target: a few minutes, mostly download)

```bash
ssh aida-gpu
git clone git@github.com:aidagroup/callosum.git   # first pod only
cd callosum && git pull
bash scripts/setup_server.sh
```

Verifies driver, a real CUDA matmul, compute capability vs the torch CUDA
build, ManiSkill + the SO-100 agent, and that both envs register. Stop and fix
if anything here fails — everything below depends on it.

### 2. Phase-1 debt: the checks that cannot run on macOS

```bash
uv run python scripts/smoke_env.py
uv run python scripts/smoke_face_turn.py
```

**Known open questions these answer** (all flagged in code as `TODO(review)`):

| Question | Where | If it fails |
|---|---|---|
| Do both arms actually reach the cube at `y = ±0.3`? | `smoke_env.py` | shrink the spacing (`two_so100_base._load_agent`); reach is ~0.5 m max, so 0.25 m is the obvious next try |
| Does the face articulation look/behave right (no jitter, face sits on the body)? | `smoke_face_turn.py` | check `disable_self_collisions`; review the joint pose |
| Are the joint friction/damping sane at cube scale? | `smoke_face_turn.py` | tune `friction`/`damping` in `_turntable_cube.py` |
| Does the gripper actually close on the layer's side faces? | `smoke_face_turn.py` | revisit the grasp target in `face_turn.compute_dense_reward` |
| Does the scripted turn flip `success` on, and body displacement flip it off? | `smoke_face_turn.py` | success logic bug — fix before any training |

Record the actual printed output; it is the evidence that phase 1 works.

### 3. First training run — the "does it learn at all" gate

Start with the **easy** env, not the hard one: `TwoSO100-v0` has a pure reach
reward, so if IPPO can't improve there, the problem is the trainer, not the
task.

```bash
uv run python -m callosum.training.ippo --env-id TwoSO100-v0 --total-timesteps <short>
```

Then the real target:

```bash
uv run python -m callosum.training.ippo --env-id FaceTurn-v0
```

Success criterion (from `docs/implementation-plan.md`, step 2.1 / the
experiment design): reward curve rises and success-rate becomes non-trivial.
This is the gate that decides whether the whole approach is viable.

### 4. Capture results before killing the pod

Checkpoints and logs live under `runs/` and are git-ignored, so they do **not**
come back via git. Pull anything worth keeping:

```bash
rsync -avz aida-gpu:~/callosum/runs/ ./runs/
```

Metrics worth writing into `docs/thesis/` while fresh: success-rate, steps to
converge, and any layout/tuning constants that had to change.

## Cost control

- Stop the pod as soon as the session's questions are answered.
- Long training runs: use `nohup`/`tmux` so an SSH drop doesn't kill them.
- Keep `/workspace` as the volume so the next pod skips the multi-GB download.
