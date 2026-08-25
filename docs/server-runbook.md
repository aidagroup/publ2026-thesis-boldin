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

### 3. Training runs — ordered gates (GPU hours are $)

Every `uv run python -m callosum.training.ippo …` writes TensorBoard logs to
`runs/<exp-name>/` (the name comes from `--exp-name`, falling back to
`<env>__ippo__<seed>__<ts>`) and checkpoints (`agent_<uid>_ckpt_<it>.pt`,
`bijepa_ckpt_<it>.pt`) into the same directory.

**3.1 Fail-fast: short Bi-JEPA wiring validation** — run this BEFORE the full
2M ablation. `setup_server.sh` already verified imports + registrations
crashed-free; this catches the remaining blind bugs (sign errors, detached-slot
shape mismatches from step 3.2, EMA no-op regressions). Budget is **50k
timesteps** (~4 iterations at num_envs=256) per review criterion 2.4 (С2) --
small enough to be a wiring gate in minutes. One run per `partner_input` mode:

```bash
uv run python -m callosum.training.ippo --env-id FaceTurn-v0 --partner-input none      --total-timesteps 50000
uv run python -m callosum.training.ippo --env-id FaceTurn-v0 --partner-input oracle    --total-timesteps 50000
uv run python -m callosum.training.ippo --env-id FaceTurn-v0 --partner-input predicted --total-timesteps 50000
```

Inspect in TensorBoard. The **non-negotiable** gates (review Б3):

- `losses/jepa_loss` exists, is finite, and is logged every iteration **in
  all three** modes (encoder trains even in `none`, via the aux loss alone).
- `bijepa/<uid>/z_i_std` and `bijepa/<uid>/z_j_std` (logged every iteration
  since Б3) must stay bounded and **not collapse toward 0** -- a `jepa_loss`
  that "decreases" while `z_std -> 0` is encoder collapse, not partner
  modelling. A downward blip across the ~4 iterations is expected; a
  *meaningful converged* trend needs the 2M §3.3 budget, not this gate.
- `bijepa/<uid>/z_hat_abs` must stay O(latent scale); no explosion.

If any mode crashes, `jepa_loss` is NaN/missing, or `z_std` collapses,
**stop the pod immediately** -- that's a code bug, not a training-time issue.

Bi-JEPA EMA (`bijepa.ema_target`, Б6): the toggle is now wired, not a silent
no-op -- with `ema_target=true`, `z_j` routes through the EMA `target_encoder`
(fed into `bijepa_step`) and `bijepa.ema_update()` runs after each
`bijepa_optimizer.step()`. Phase-1 default (`ema_target=false`) is unchanged:
`z_j` comes from the online encoder and is detached inside `jepa_loss`.

**3.2 Decentralized baseline (2.1 gate)** — `partner_input=none` *is* the
decentralized baseline (the slot is zeros):

```bash
uv run python -m callosum.training.ippo --env-id FaceTurn-v0
```

Gate (from `docs/implementation-plan.md` step 2.1): reward rises, eval
success-rate becomes non-trivial. If it can't learn at all, the problem is the
task/trainer, not Bi-JEPA — do not reach for the ablation until this passes.

**3.3 The ablation sweep (step 3.3)** — runs all three modes as isolated
subprocesses and writes a machine-readable summary:

```bash
uv run python scripts/run_ablation.py --env-id FaceTurn-v0 --total-timesteps 2000000 --seed 1
```

Compare the three `partner_input` curves side by side; the summary JSONL tells
you which runs finished and how long each took.

**3.4 BenchMARL harness (step 2.2)** — confirm IPPO/MAPPO train our task
through the standard BenchMARL flow:

```bash
uv run python - <<'PY'
from callosum.envs.benchmarl_task import CallosumTask
from benchmarl.experiment import Experiment
from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.models import MlpModelConfig

t = CallosumTask.FACE_TURN.get_task({"task":"FaceTurn-v0","max_steps":100,"partner_obs":"full"})
# Build an Experiment(task=t, algorithm_config=IppoConfig(...), ...) and run a
# short sweep. The Task wraps the mani_skill envs registered in step 1.
PY
```

Gate (run `torchrl.envs.utils.check_env_specs` on the server -- see the
`TODO(review)` table below; the specific dims are server-verified because
ManiSkill can't run on macOS). It must pass, confirming:

- `observation_spec["agents"]["observation"]` is `(B, n_agents=2, D)` where
  ``D`` is the **decentralized** base (partner TCP dropped, review Б4); both
  agents share ``D`` (`_per_agent_obs_dim` computes agent 0's dim, asserted
  symmetric). This is the IPPO actor input -- it must NOT contain the partner
  pose even when `partner_obs="full"`.
- `state_spec["state"]` is `(B, D_state)` = the **full** state incl. partner
  (centralized MAPPO critic) -- the actor/critic asymmetry is the CTDE contract.
- `("next","reward")`, `("next","done")`, `("next","terminated")`,
  `("next","truncated")` are each `(B,1)`; the done-family keys must be BOOL
  (review С4 hypothesis 3: they're currently `Unbounded` float in the spec).

**TODO(review) marks in `callosum/envs/benchmarl_task.py`** -- each is
server-only (`mani_skill` is Linux+CUDA only); verify in order:

| # | Mark (file:line) | What to verify on server | Verified by |
|---|---|---|---|
| 1 | L29 (class docstring) | `benchmarl_task` imports without mani_skill on dev; env materialisation only in `_build` | `uv run python -c "import callosum.envs.benchmarl_task"` (dev) |
| 2 | L208 (`_build_specs` docstring) | spec builders match the actual `_agent_obs`/`_global_state` tensors (decentralized actor, full state) | `check_env_specs` (3.4 gate) |
| 3 | L305 (`_reset`) | `ManiSkillVectorEnv.reset()` returns a torch obs dict; `reset_td` None-shape handling; partial-reset mask honored | short `Experiment.run()` + inspect `next` batch |
| 4 | L322 (`_step`) | `step` returns `rew (B,)`, `term/trunc (B,)`; reward/done dtype | `check_env_specs` + 10-step rollout |
| 5 | L332 (`_step` done dtype) | `done`/`terminated`/`truncated` are BOOL, not float `Unbounded` | `check_env_specs` (hypothesis 3); fix spec if FAIL |
| 6 | L338 (`_step` final_observation) | `final_observation` + `RewardSum` align with the GAE `next_done` bootstrap in `ippo.py` | compare `losses/{kl,pg_loss}` vs `losses/jepa_loss` curves |
| 7 | L344 (`_set_seed`) | underlying vector env RNGs seed from `seed` | deterministic rerun at fixed seed ↔ |
| 8 | L597 (hydra config group) | `task=callosum/face_turn` resolves on the hydra CLI path | `python -m benchmarl.train task=callosum/face_turn ...` (or use the programmatic path, §3.4) |

### 4. Where to find results + pull back

All outputs live under `runs/` (git-ignored):

- `runs/<run-name>/events.out.tfevents.*` — TensorBoard scalars:
  `losses/jepa_loss`, `losses/<uid>/{kl,entropy,vf_loss,pg_loss}`,
  `charts/SPS`, `time/{rollout_time,update_time}`, `eval/{reward,success_rate,…}`,
  `train/{reward,success_rate,…}`.
- `runs/<run-name>/agent_<uid>_ckpt_<it>.pt`, `bijepa_ckpt_<it>.pt`,
  `agent_<uid>_final_ckpt.pt`, `bijepa_final_ckpt.pt` — reload with
  `torch.load(..., weights_only=True)` + `Agent.load_state_dict` /
  `BiJEPA.load_state_dict`.
- `runs/ablation-<ts>/ablation_summary.jsonl` + `runs/ablation-<ts>/{none,oracle,predicted}.log`
  — the 3.3 run records (mode, run_name, return-code, elapsed) and per-mode
  stdout.

Pull back to the dev machine before killing the pod:

```bash
rsync -avz aida-gpu:~/callosum/runs/ ./runs/
```

Then `tensorboard --logdir runs` locally to compare the three `partner_input`
curves. Write down metrics worth keeping into `docs/thesis/` while fresh:
success-rate, steps to converge, layout/tuning constants.

## Cost control

- Stop the pod as soon as the session's questions are answered.
- Long training runs: use `nohup`/`tmux` so an SSH drop doesn't kill them.
- Keep `/workspace` as the volume so the next pod skips the multi-GB download.
