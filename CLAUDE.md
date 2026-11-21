# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

University thesis project (package `callosum`): two SO-ARM101 robot arms with Robonine parallel grippers, each a decentralized agent, cooperatively turn a Rubik's-cube-like object in ManiSkill GPU simulation. The coordination mechanism is **Bi-JEPA** (each agent predicts its partner's latent state, no message passing), with sim2real as a later goal. The README and `docs/` are written in Russian; **code, comments and docstrings must be in English**.

Phase 1 (ManiSkill envs) is implemented. Step 2.1 added the joint-level IPPO trainer `callosum/experiments/arm/trainer.py` (`python -m callosum.experiments.arm.trainer`, config in `callosum/experiments/arm/ppo_config.py`, per-agent policy inputs cut from the flat state obs in `callosum/experiments/arm/agent_obs.py`). IPPO from scratch never grasps, so the baseline is pretrained by behaviour cloning on demos of the scripted expert (`callosum/experiments/arm/`, `scripts/experiments/arm/collect_demos.py`, `python -m callosum.experiments.arm.bc`; pipeline in the runbook). The Bi-JEPA module lives in `callosum/common/bijepa.py` (ported from `step/3.1-bijepa-module`) and is integrated into the reduced-cube trainer; BenchMARL (step 2.2) remains deferred. [docs/implementation-plan.md](docs/implementation-plan.md) is the step-by-step spec (steps 1.1 to 3.3); thesis background lives in [docs/thesis/](docs/thesis/README.md) (glossary in `docs/thesis/glossary.md`).

## Current state and standing rules (read first in a new session)

Short "what next" list: [PLAN.md](PLAN.md) (full spec: [docs/implementation-plan.md](docs/implementation-plan.md)).

- **Thesis main line (decided 2026-10-07/08): skill-level MARL**, new step 2.3 `FaceTurnSkills-v0` (branch `step/2.3-skill-level-marl`, not started): each arm's policy picks a discrete scripted primitive (holder: wait/approach/grasp/release/lift/lower/retreat; rotator: wait/approach/grasp/turn90/turn180/release/retreat; v2 adds holder flips and a rotator side approach). Both agents know the goal and the cube-solving algorithm; planning is not learned, only coordinated execution. Partner modes `none` (control) / `oracle` / `predicted` (Bi-JEPA). Order: 2.3, merge 3.1, 3.2, 3.3. The joint-level IPPO + BC line (2.1) is secondary, and its BC result (holder ever-hold 1.00, rotator 0.02, success 0: the rotator cannot see the holder's jaw close) is motivation for Bi-JEPA, not a baseline to tune forever.
- **Server status:** demos collected (364/512 with DART noise 0.1), BC trained, fine-tune `ft_full_s1` was started and may be resumable from `~/callosum-runs/ft_full_s1` (`--resume runs/ft_full_s1`). Persistent-storage move to `~/callosum` (scripts + docs) is in progress, so check `git status` before assuming the server docs are final.
- **Never run simulations, probes, demo collection, BC or IPPO training on the Mac**, neither directly nor through subagents (the user's laptop overheated). Locally only `ruff`, small `pytest` unit tests and code reading; give the user the server commands instead.
- **Model roles:** Haiku subagents (`model: "haiku"`) write the code, Sonnet reviews the diffs (and tests/lint) and commits/pushes, Opus plans and does the global review after a whole stage (step). Briefs are self-contained with disjoint file ownership and "no local sim, don't commit". Docs-only edits can be done directly.
- **Commits:** subject `type: msg` (the hook adds the `step/...` prefix); no `Co-Authored-By` or other attribution lines. Push routinely to the step branch; merge into `main` only server-verified, stable work; confirm other outward or hard-to-reverse actions first.
- **Simulation only** for now (no real hardware). **Reply to the user in English** (global settings `language: english`); docs stay in Russian, code and comments in English.

## Two-machine workflow (drives most constraints)

- **macOS (dev):** ManiSkill/SAPIEN is marked Linux-only (`sys_platform == 'linux'` in `pyproject.toml`), so it is **not in the project venv**. Code that imports `mani_skill` cannot run in the project venv or in CI; lint, formatting, and pure-Python/pure-torch tests run there. ManiSkill 3.0.1 can run CPU simulation in a throwaway env (`PYTHONPATH=. uv run --no-project --python 3.12 --with mani-skill==3.0.1 --with torch python scripts/experiments/arm/smoke_env.py --sim-backend cpu --num-envs 2`), but **do not run simulations, probes, demo collection or training on the Mac** (it overheats the user's laptop): all sim and training checks run on the GPU server, and reports list the server commands to run. At most a single short import/scene smoke if the user asks for it.
- **Linux GPU server (lab server, 1× A100 80GB, driver 570 → CUDA ≤ 12.8):** all simulation runs here, but only through the **JupyterHub web UI and its terminal (no SSH, no root)**: `git clone` from GitHub once into `~/callosum`, then `bash scripts/ops/update_server.sh [-- --smoke]` (hard-syncs to `origin/<branch>`, never `git pull`, then runs `scripts/ops/setup_server.sh`). The persistent `$HOME` (100 GB) holds everything: checkout `~/callosum`, venv/uv cache/Python/HF/ManiSkill caches/conda Vulkan loader in `~/.callosum` (override `CALLOSUM_DATA`), `~/.sapien`, and `runs/` (symlink into `~/callosum-runs`, for any checkout path), so a container restart loses nothing; only `/tmp` is wiped (temporary files only). The old `/tmp` scratch layout is an explicit opt-in (`CALLOSUM_SCRATCH=<dir>` or `CALLOSUM_LAYOUT=scratch`, or automatic when `$HOME` has < 30 GB free). A killed training run is continued with `python -m callosum.experiments.arm.trainer --resume runs/<name>` (full-state checkpoints, same directory). New terminals source `~/.callosum-env.sh`; long runs must be detached (`setsid nohup`) because the kernel dies with the tab. Setup probes for hardware NVIDIA Vulkan (generated ICD manifest + current conda loader, see the runbook; `bash scripts/ops/diagnose_vulkan.sh` to inspect) and falls back to lavapipe; envs for state training are created with `render_backend="none"` either way (the smoke scripts and any trainer), `"gpu"` only for camera rendering and only with hardware Vulkan. See [docs/setup.md](docs/setup.md) and [docs/server-runbook.md](docs/server-runbook.md). Outbound internet is an **allowlist** (GitHub, PyPI, download.pytorch.org, astral.sh, Hugging Face, LLM proxy `llm-proxy.spirit.culab.ru`); `wandb.ai` and other hosts are blocked, so any new dependency or download must come from an allowed host or the owner must open it first.
- When reading ManiSkill API for reference, read it from GitHub at tag **`v3.0.1`** (matches `uv.lock`), not `main`: `gh api "repos/haosulab/ManiSkill/contents/<path>?ref=v3.0.1" --jq .content | base64 -d`. Don't use APIs missing from `v3.0.1`.

## Commands

```bash
make dev                      # macOS: uv sync --extra train --extra dev (no sim)
make server                   # Linux: --extra sim --extra train --extra dev
uv run ruff check .           # lint (also `make lint`)
uv run ruff format --check .  # CI enforces formatting too; use `ruff format .` to fix
uv run pytest -q              # all tests (also `make test`)
uv run pytest tests/common/test_partner_obs.py::<test_name> -q   # single test

# Smoke scripts take --sim-backend / --num-envs. Run them on the server, not on the Mac.
# Server (GPU sim, the default):
uv run python scripts/experiments/arm/smoke_env.py
uv run python scripts/experiments/arm/smoke_face_turn.py
uv run python scripts/experiments/arm/probe_face_turn.py   # scripted two-arm face turn (needs scipy, which ManiSkill brings)
uv run python -m callosum.experiments.arm.trainer --env-id TwoSO101-v0   # IPPO trainer (GPU server)
uv run python scripts/experiments/arm/collect_demos.py --num-envs 256 --num-batches 2 --action-noise 0.1   # expert demos (GPU; CPU: --sim-backend cpu --num-batches 20)
uv run python -m callosum.experiments.arm.bc --demos runs/demos/X.pt --exp-name bc1               # BC pretraining (CPU ok, no mani_skill)
uv run python -m callosum.experiments.arm.trainer --eval-only --checkpoint runs/bc1/bc.pt            # judge a checkpoint; fine-tune: --checkpoint ... --critic-warmup-iters 10 --demos ... --bc-coef 1
```

CI (`.github/workflows/ci.yml`) installs only the `dev` extra and runs ruff check, ruff format check, an import smoke check of all subpackages, and pytest. Run these locally before pushing; red CI means the PR isn't ready.

Ruff: line length 100, target py312. Python is pinned to 3.12 (`<3.13` because SAPIEN has no newer wheels).

## Dependencies

`uv.lock` is committed and the server installs with `--frozen`; run `make lock` after changing `pyproject.toml`. Extras: `sim` (ManiSkill, PettingZoo), `train` (torch, tensordict, torchrl, benchmarl), `dev` (ruff, pytest). Do **not** add a `setuptools>=83` constraint or merge a Dependabot bump of it: torch 2.11 needs `setuptools<82`, and forcing it silently downgrades torch (rationale in `pyproject.toml`). Torch is pinned to the `cu128` index on Linux because the server's driver caps CUDA at 12.8; `cu129`/`cu130` need a driver upgrade first, and then `torch`, `torchrl` and `tensordict` must move together (see docs/setup.md).

## Architecture

Code is organised by experiment under `callosum/experiments/`; reusable building blocks live in `callosum/common/`; the robot model and vendored assets stay shared.

- `callosum/common/`: experiment-agnostic code, no simulator import at module level — `cli.py` (dataclass→argparse), `ppo_core.py` (actor-critic/GAE/clipped update), `categorical.py`, `checkpoint.py`, `nonfinite.py`, `logging.py` (MetricLogger), `diag.py`+`diag_names.py`, `metrics.py` (episode metrics), `visibility.py` (generic partner-observation rule), `bijepa.py`+`bijepa_config.py`+`bijepa_probe.py`, `render_utils.py` (MuJoCo/PIL render helpers).
- `callosum/experiments/cube_wrench/`: reduced MuJoCo cube (main preddefence line) — `env.py` (`CubeWrench-v0`), `env_gpu.py` (batched `mujoco_warp` backend), `logic.py` (pure obs/visibility/reward/randomisation), `config.py`, `ppo_config.py`, `train.py` (`python -m callosum.experiments.cube_wrench.train`), `vec_env.py`, `eval.py` (`python -m ...eval`), `render.py` (`python -m ...render`), `oracle.py`.
- `callosum/experiments/coop_tasks/`: two-robot suite — `base.py` (shared `CoopTaskEnv`), `cartpole.py` (`CartPoleCoop-v0`), `pendulum.py` (`PendulumPump-v0`), `logic.py`, `config.py`, `oracle.py`, `vec_env.py`, `eval.py`; `TossCatch-v0` planned.
- `callosum/experiments/arm/`: both SO-ARM101 lines. Shared: `scene.py` (`TwoSO101-v0` base), `layout.py`, `turntable.py`, `face_lock.py`, `sim_compat.py`/`sim_utils.py`, the scripted expert (`expert.py`, `phases.py`, `program.py`, `tracking.py`, `noise.py`, `expert_config.py`), `trainer.py` (`python -m callosum.experiments.arm.trainer`), `ppo_config.py`, `agent_obs.py`, `skills.py`/`skills_obs.py`, `skills_agent_obs.py`, `demos.py`, `bc.py`+`bc_config.py`. Skill-level line: `skills_env.py` (`FaceTurnSkills-v0`), `skills_cube.py`, `skills_plans.py`, `skills_runtime.py`, `skills_conductor.py`, `skills_config.py`, `skills_cube_config.py`, `skills_eval.py`. Joint-level legacy: `face_turn_env.py` (`FaceTurn-v0`), `face_turn_config.py`, BC/demos (kept as thesis motivation).
- `callosum/robots/so101_parallel_gripper.py`: custom SO-ARM101 + Robonine parallel gripper agent (`so101_pg`: 5 joint deltas + gripper position). URDF/meshes in `callosum/assets/so101_parallel_gripper/`.
- `scripts/` is grouped the same way: `experiments/{cube_wrench,coop_tasks,arm}/`, `analysis/` (plots/summaries), `ops/` (server setup/update, cloud resume loop); `tests/` mirrors it: `tests/common/`, `tests/experiments/{cube_wrench,coop_tasks,arm}/`, plus root `tests/test_import.py`/`test_assets.py`.

Code that touches the simulator carries `# TODO(review):` markers where behavior is unverified on GPU. Keep that convention when unsure rather than guessing.

## Conventions from the implementation plan

- One step = one branch (`step/<phase>.<step>-<slug>`) = one PR into `main`; don't commit directly to `main`. Branch each step from an up-to-date `main`.
- Public functions/classes get docstrings and type hints.
- Don't commit artifacts (weights, logs, videos); they are git-ignored. Results stay on the server under `runs/` (copy out via the JupyterHub file browser; there is no SSH/`rsync`).
- Steps that need the simulator must state in their report what was verified (unit tests) and list the server commands that verify the rest on the GPU.
