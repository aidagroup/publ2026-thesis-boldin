# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

University thesis project (package `callosum`): two SO-100 robot arms, each a decentralized agent, cooperatively turn a Rubik's-cube-like object in ManiSkill GPU simulation. The coordination mechanism is **Bi-JEPA** (each agent predicts its partner's latent state, no message passing), with sim2real as a later goal. The README and `docs/` are written in Russian; **code, comments and docstrings must be in English**.

Phase 1 (ManiSkill envs) is implemented. The `callosum/agents/` (Bi-JEPA) and `callosum/training/` (IPPO/BenchMARL) packages are still empty stubs. [docs/implementation-plan.md](docs/implementation-plan.md) is the step-by-step spec (steps 1.1 to 3.3); thesis background lives in [docs/thesis/](docs/thesis/README.md) (glossary in `docs/thesis/glossary.md`).

## Two-machine workflow (drives most constraints)

- **macOS (dev):** ManiSkill/SAPIEN is Linux+CUDA only (`sys_platform == 'linux'` in `pyproject.toml`), so it is **not installed locally**. Anything that imports `mani_skill` cannot be imported or run on the Mac or in CI. Only lint, formatting, and pure-Python/pure-torch tests run locally.
- **Linux GPU server (lab server, 1× A100 80GB, driver 570 → CUDA ≤ 12.8):** all simulation runs here, but only through the **JupyterHub web UI and its terminal (no SSH, no root)**: `git clone`/`git pull` from GitHub, then `bash scripts/setup_server.sh [--smoke]`. `$HOME` is tiny (< 1 GB free) and `/tmp` is assumed wiped on restart, so checkout, venv and caches live in scratch (`/tmp/$USER-callosum`, override `CALLOSUM_SCRATCH`); only `runs/` (symlink into `~/callosum-runs`) persists. New terminals source `~/.callosum-env.sh`; long runs must be detached (`setsid nohup`) because the kernel dies with the tab. See [docs/setup.md](docs/setup.md) and [docs/server-runbook.md](docs/server-runbook.md). Outbound internet is an **allowlist** (GitHub, PyPI, download.pytorch.org, astral.sh, Hugging Face, LLM proxy `llm-proxy.spirit.culab.ru`); `wandb.ai` and other hosts are blocked, so any new dependency or download must come from an allowed host or the owner must open it first.
- When reading ManiSkill API for reference, read it from GitHub at tag **`v3.0.1`** (matches `uv.lock`), not `main`: `gh api "repos/haosulab/ManiSkill/contents/<path>?ref=v3.0.1" --jq .content | base64 -d`. Don't use APIs missing from `v3.0.1`.

## Commands

```bash
make dev                      # macOS: uv sync --extra train --extra dev (no sim)
make server                   # Linux: --extra sim --extra train --extra dev
uv run ruff check .           # lint (also `make lint`)
uv run ruff format --check .  # CI enforces formatting too; use `ruff format .` to fix
uv run pytest -q              # all tests (also `make test`)
uv run pytest tests/test_partner_obs.py::<test_name> -q   # single test

# Server only (GPU sim):
uv run python scripts/smoke_env.py
uv run python scripts/smoke_face_turn.py
```

CI (`.github/workflows/ci.yml`) installs only the `dev` extra and runs ruff check, ruff format check, an import smoke check of all subpackages, and pytest. Run these locally before pushing; red CI means the PR isn't ready.

Ruff: line length 100, target py312. Python is pinned to 3.12 (`<3.13` because SAPIEN has no newer wheels).

## Dependencies

`uv.lock` is committed and the server installs with `--frozen`; run `make lock` after changing `pyproject.toml`. Extras: `sim` (ManiSkill, PettingZoo), `train` (torch, tensordict, torchrl, benchmarl), `dev` (ruff, pytest). Do **not** add a `setuptools>=83` constraint or merge a Dependabot bump of it: torch 2.11 needs `setuptools<82`, and forcing it silently downgrades torch (rationale in `pyproject.toml`). Torch is pinned to the `cu128` index on Linux because the server's driver caps CUDA at 12.8; `cu129`/`cu130` need a driver upgrade first, and then `torch`, `torchrl` and `tensordict` must move together (see docs/setup.md).

## Architecture

Environments live in `callosum/envs/` and layer as follows:

- `two_so100_base.py`: `TwoSO100Base` (`TwoSO100-v0`), a `BaseEnv` with two `so100` agents (`MultiAgent`, mirrored yaws on opposite sides of the table), a loose cube, `pd_joint_delta_pos` control (the only sensible mode, since SO-100 has no IK controller), and a reach-only reward. `agent_a`/`agent_b` are the arms. `TableSceneBuilder.initialize` does not place `("so100","so100")` robots, so the env resets both arms' qpos itself.
- `face_turn.py`: `FaceTurn` (`FaceTurn-v0`) subclasses the base and replaces the cube with the articulated turntable cube from `_turntable_cube.py` (body + one revolute `face` link, limits `[0, pi/2]`). **Roles are fixed:** `agent_a` is the holder (keeps the body still), `agent_b` is the rotator (grasps and turns the face). Success = face within tolerance of 90° **and** body drift within tolerance. Reward weights and tolerances come from the dataclass in `callosum/configs/face_turn.py`; per the plan, tunable parameters belong in `callosum/configs/`, not hardcoded.
- `_partner_obs.py`: the `partner_obs` flag (`"full"` / `"none"`; `"predicted"` comes with Bi-JEPA in phase 3) as **pure logic with no mani_skill/torch import**, so it is unit-testable on macOS. Keep decision logic that can be tested locally in dependency-free modules like this one. Note that with `obs_mode="state"` the whole extra-obs dict is flattened into one tensor, so `partner_obs` is an env-level toggle, not a true per-agent observation split. Per-agent decentralized inputs are a later training/policy-side concern.

`evaluate()` and `compute_dense_reward()` read TCP poses directly from the agents (privileged, CTDE-style), so they are unaffected by `partner_obs`.

Code that touches the simulator carries `# TODO(review):` markers where behavior is unverified on GPU. Keep that convention when unsure rather than guessing.

## Conventions from the implementation plan

- One step = one branch (`step/<phase>.<step>-<slug>`) = one PR into `main`; don't commit directly to `main`. Branch each step from an up-to-date `main`.
- Public functions/classes get docstrings and type hints.
- Don't commit artifacts (weights, logs, videos); they are git-ignored. Results stay on the server under `runs/` (copy out via the JupyterHub file browser; there is no SSH/`rsync`).
- Steps that need the simulator must state in their report that they were not verified locally.
