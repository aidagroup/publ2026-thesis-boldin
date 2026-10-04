# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

University thesis project (package `callosum`): two SO-ARM101 robot arms with Robonine parallel grippers, each a decentralized agent, cooperatively turn a Rubik's-cube-like object in ManiSkill GPU simulation. The coordination mechanism is **Bi-JEPA** (each agent predicts its partner's latent state, no message passing), with sim2real as a later goal. The README and `docs/` are written in Russian; **code, comments and docstrings must be in English**.

Phase 1 (ManiSkill envs) is implemented, and step 2.1 adds the IPPO trainer `callosum/training/ippo.py` (`python -m callosum.training.ippo`, config in `callosum/configs/ippo.py`, per-agent policy inputs cut from the flat state obs in `callosum/training/_agent_obs.py`); the server run is still pending. `callosum/agents/` (Bi-JEPA) and BenchMARL (step 2.2) are still empty stubs. [docs/implementation-plan.md](docs/implementation-plan.md) is the step-by-step spec (steps 1.1 to 3.3); thesis background lives in [docs/thesis/](docs/thesis/README.md) (glossary in `docs/thesis/glossary.md`).

## Two-machine workflow (drives most constraints)

- **macOS (dev):** ManiSkill/SAPIEN is marked Linux-only (`sys_platform == 'linux'` in `pyproject.toml`), so it is **not in the project venv**. Code that imports `mani_skill` cannot run in the project venv or in CI; lint, formatting, and pure-Python/pure-torch tests run there. ManiSkill 3.0.1 does install and run **CPU simulation** in a throwaway env, which is the first check for anything sim-related (scene loads, robot rest pose, reach, grasp): `PYTHONPATH=. uv run --no-project --python 3.12 --with mani-skill==3.0.1 --with torch python scripts/smoke_env.py --sim-backend cpu --num-envs 2`. GPU sim and training still need the server.
- **Linux GPU server (lab server, 1× A100 80GB, driver 570 → CUDA ≤ 12.8):** all simulation runs here, but only through the **JupyterHub web UI and its terminal (no SSH, no root)**: `git clone` from GitHub once, then `bash scripts/update_server.sh [-- --smoke]` (hard-syncs to `origin/<branch>`, never `git pull`, then runs `scripts/setup_server.sh`). `$HOME` is tiny (< 1 GB free) and `/tmp` is assumed wiped on restart, so checkout, venv and caches live in scratch (`/tmp/$USER-callosum`, override `CALLOSUM_SCRATCH`); only `runs/` (symlink into `~/callosum-runs`) persists. New terminals source `~/.callosum-env.sh`; long runs must be detached (`setsid nohup`) because the kernel dies with the tab. Hardware Vulkan does not work there (lavapipe fallback), so envs must be created with `render_backend="none"` (the smoke scripts and any trainer). See [docs/setup.md](docs/setup.md) and [docs/server-runbook.md](docs/server-runbook.md). Outbound internet is an **allowlist** (GitHub, PyPI, download.pytorch.org, astral.sh, Hugging Face, LLM proxy `llm-proxy.spirit.culab.ru`); `wandb.ai` and other hosts are blocked, so any new dependency or download must come from an allowed host or the owner must open it first.
- When reading ManiSkill API for reference, read it from GitHub at tag **`v3.0.1`** (matches `uv.lock`), not `main`: `gh api "repos/haosulab/ManiSkill/contents/<path>?ref=v3.0.1" --jq .content | base64 -d`. Don't use APIs missing from `v3.0.1`.

## Commands

```bash
make dev                      # macOS: uv sync --extra train --extra dev (no sim)
make server                   # Linux: --extra sim --extra train --extra dev
uv run ruff check .           # lint (also `make lint`)
uv run ruff format --check .  # CI enforces formatting too; use `ruff format .` to fix
uv run pytest -q              # all tests (also `make test`)
uv run pytest tests/test_partner_obs.py::<test_name> -q   # single test

# Smoke scripts take --sim-backend / --num-envs. Mac CPU sim: see the uv command above.
# Server (GPU sim, the default):
uv run python scripts/smoke_env.py
uv run python scripts/smoke_face_turn.py
uv run python scripts/probe_face_turn.py   # scripted two-arm face turn (needs scipy, which ManiSkill brings)
uv run python -m callosum.training.ippo --env-id TwoSO101-v0   # IPPO trainer (GPU; Mac CPU smoke: see docs/server-runbook.md)
```

CI (`.github/workflows/ci.yml`) installs only the `dev` extra and runs ruff check, ruff format check, an import smoke check of all subpackages, and pytest. Run these locally before pushing; red CI means the PR isn't ready.

Ruff: line length 100, target py312. Python is pinned to 3.12 (`<3.13` because SAPIEN has no newer wheels).

## Dependencies

`uv.lock` is committed and the server installs with `--frozen`; run `make lock` after changing `pyproject.toml`. Extras: `sim` (ManiSkill, PettingZoo), `train` (torch, tensordict, torchrl, benchmarl), `dev` (ruff, pytest). Do **not** add a `setuptools>=83` constraint or merge a Dependabot bump of it: torch 2.11 needs `setuptools<82`, and forcing it silently downgrades torch (rationale in `pyproject.toml`). Torch is pinned to the `cu128` index on Linux because the server's driver caps CUDA at 12.8; `cu129`/`cu130` need a driver upgrade first, and then `torch`, `torchrl` and `tensordict` must move together (see docs/setup.md).

## Architecture

Environments live in `callosum/envs/` and layer as follows:

- `callosum/robots/so101_parallel_gripper.py`: custom ManiSkill agent `SO101ParallelGripper` (uid `so101_pg`): SO-ARM101 (5-DOF arm) plus the Robonine parallel gripper (two prismatic jaws; `right_clamp` is driven, `left_clamp` mirrors it via a mimic controller, ~75 mm stroke in sim). Per arm the action is 5 joint deltas + 1 gripper position (6-D). The URDF and meshes are vendored in `callosum/assets/so101_parallel_gripper/` (flattened from the upstream xacro; jaw collisions are replaced by boxes because the upstream convex hulls fill the jaw gap; see the README there for licences). ManiSkill's built-in `so100` is no longer used.
- `two_so101_base.py`: `TwoSO101Base` (`TwoSO101-v0`), a `BaseEnv` with two `so101_pg` agents (`MultiAgent`; the bases stand 90 degrees apart around the cube, both facing the table centre, placement from `ArmLayout` in `callosum/configs/layout.py`: holder `agent_a` at -y, 0.32 m, rotator `agent_b` at +x, 0.24 m; arms facing each other do not work, see that docstring), a loose cube, `pd_joint_delta_pos` control (the only sensible mode, since the agent has no IK controller), and a reach-only reward. `agent_a`/`agent_b` are the arms. `TableSceneBuilder.initialize` does not place these robots, so the env resets both arms' qpos itself (reset noise only on the arm joints, not the gripper).
- `face_turn.py`: `FaceTurn` (`FaceTurn-v0`) subclasses the base and replaces the cube with the articulated turntable cube from `_turntable_cube.py` (body + one revolute `face` link, limits `[0, pi/2]`). **Roles are fixed:** `agent_a` is the holder (keeps the body still), `agent_b` is the rotator (grasps and turns the face). Success = face within tolerance of 90° **and** body drift within tolerance. Reward weights and tolerances come from the dataclass in `callosum/configs/face_turn.py`; per the plan, tunable parameters belong in `callosum/configs/`, not hardcoded.
- `_partner_obs.py`: the `partner_obs` flag (`"full"` / `"none"`; `"predicted"` comes with Bi-JEPA in phase 3) as **pure logic with no mani_skill/torch import**, so it is unit-testable on macOS. Keep decision logic that can be tested locally in dependency-free modules like this one. Note that with `obs_mode="state"` the whole extra-obs dict is flattened into one tensor, so `partner_obs` is an env-level toggle, not a true per-agent observation split. Per-agent decentralized inputs are a later training/policy-side concern.

`evaluate()` and `compute_dense_reward()` read TCP poses directly from the agents (privileged, CTDE-style), so they are unaffected by `partner_obs`.

Code that touches the simulator carries `# TODO(review):` markers where behavior is unverified on GPU. Keep that convention when unsure rather than guessing.

## Conventions from the implementation plan

- One step = one branch (`step/<phase>.<step>-<slug>`) = one PR into `main`; don't commit directly to `main`. Branch each step from an up-to-date `main`.
- Public functions/classes get docstrings and type hints.
- Don't commit artifacts (weights, logs, videos); they are git-ignored. Results stay on the server under `runs/` (copy out via the JupyterHub file browser; there is no SSH/`rsync`).
- Steps that need the simulator must state in their report what was verified on the Mac CPU sim and what was not verified on the GPU server.
