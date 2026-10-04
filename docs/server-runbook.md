# Server runbook (training session)

The ordered checklist for a GPU session on the lab server (see
[setup.md](setup.md) for the machine, network allowlist and environment).
Write and verify everything possible on macOS first (including CPU-sim smoke
checks, see [setup.md](setup.md#local-macos)); use server time for things that
genuinely need a GPU.

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
| Internet | allowlist only (GitHub, PyPI, PyTorch, Hugging Face, …) |

## SSH access

Add the server to `~/.ssh/config` on the dev machine under a stable alias:

```
Host aida-gpu
    HostName <server-host>
    Port <port>
    User <user>
    IdentityFile ~/.ssh/<your-key>
    ServerAliveInterval 30
```

Then everything runs as `ssh aida-gpu '<command>'`.

## Session order

### 1. Environment

```bash
ssh aida-gpu
git clone git@github.com:aidagroup/callosum.git   # first time only
cd callosum && git pull
bash scripts/setup_server.sh
```

Verifies driver, a real CUDA matmul, compute capability vs the torch CUDA
build, ManiSkill + our `so101_pg` agent, and that both envs register. Stop and fix
if anything here fails — everything below depends on it. After the first run
the uv cache is warm, so re-running it after a `git pull` is quick.

### 2. Phase-1 debt: the GPU-backend checks

```bash
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

Start with the **easy** env, not the hard one: `TwoSO101-v0` has a pure reach
reward, so if IPPO can't improve there, the problem is the trainer, not the
task.

```bash
uv run python -m callosum.training.ippo --env-id TwoSO101-v0 --total-timesteps <short>
```

Then the real target:

```bash
uv run python -m callosum.training.ippo --env-id FaceTurn-v0
```

Success criterion (from `docs/implementation-plan.md`, step 2.1 / the
experiment design): reward curve rises and success-rate becomes non-trivial.
This is the gate that decides whether the whole approach is viable.

Run long trainings inside `tmux` (or `nohup`) so an SSH drop doesn't kill them.

### 4. Watching and collecting results

Checkpoints and TensorBoard logs live under `runs/` on the server and are
git-ignored, so they do **not** come back via git. `wandb.ai` is not reachable
from the server, so metrics stay local.

Watch training live through an SSH tunnel:

```bash
ssh aida-gpu 'cd callosum && uv run tensorboard --logdir runs --port 6006'   # on the server
ssh -N -L 6006:localhost:6006 aida-gpu                                        # on the Mac
# then open http://localhost:6006
```

Copy anything worth keeping back to the Mac:

```bash
rsync -avz aida-gpu:~/callosum/runs/ ./runs/
```

Metrics worth writing into `docs/thesis/` while fresh: success-rate, steps to
converge, and any layout/tuning constants that had to change.
