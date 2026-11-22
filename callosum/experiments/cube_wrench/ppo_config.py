"""Config and CLI of the continuous-wrench IPPO trainer (PLAN.md steps 2 and 3, `wrench_ppo`).

The reduced models (`CubeWrench-v0` and, via `--task`, the two-agent coop suite of
`callosum.experiments.coop_tasks`) are CPU-only MuJoCo, so their trainer is a small CPU-side IPPO: two
independent Gaussian policies on one shared team reward, a Python-loop vectorisation of the
single-instance env, clipped PPO updates, TensorBoard logs, greedy evaluation and
`latest.pt` checkpoints with `--resume`. `task` selects the environment and therefore its
observation width, action width and agent keys; `"cube"` is the default and keeps the
original path unchanged.

Step 3 adds the `predicted` ablation arm on top of the env's `none`/`oracle` partner
visibility: with `use_bijepa=True` the env stays at `partner_mode="none"` (no privileged
partner block) and each policy additionally receives the Bi-JEPA predicted partner latent
`ẑ_j`, so its input width is `obs_size("none") + latent_dim` (28 + 32 = 60 by default)
instead of 26/35. The Bi-JEPA module itself is trained by an auxiliary latent loss in
`callosum.experiments.cube_wrench.train`; its hyperparameters are flattened into this dataclass (one
CLI flag each) because the shared argparse helper only supports scalar fields.

This module is pure stdlib on purpose (same rule as `callosum.experiments.cube_wrench.config` and
`callosum.experiments.cube_wrench.logic`): the config must import and validate on macOS/CI, where the
sandbox extra (mujoco) is absent, so every trainer flag is testable without the simulator.
Torch is imported only by `callosum.experiments.cube_wrench.train`.

The field groups follow the plan's step-2/3 description: the run identity, the environment /
rollout shape, the Bi-JEPA block, the PPO hyperparameters, and the logging / evaluation /
checkpointing cadence. The recommended ranges of the plan are 64-256 envs and 32-128 rollout
control steps; the validation below only enforces the hard sanity conditions, so a small
`--num-envs 8` server smoke run stays possible.

    backend="cpu"  -- the single-instance `WrenchCubeEnv` in a Python loop (macOS/CI-safe
                      default; only the `sandbox` extra is needed);
    backend="warp" -- the batched `WrenchCubeBatchGPU` (mujoco_warp) stepping all worlds on
                      one CUDA device (`--backend warp --device cuda --num-envs 1024`). It is
                      an OPTIONAL GPU backend of the same env contract: the weights of a run
                      are interchangeable between the backends (the policy is the same MLP),
                      only the rollout speed differs. It needs the OPTIONAL `sandbox-gpu`
                      extra (Linux/CUDA only, linux-marked in `uv.lock`); the pod's
                      `uv sync --frozen` (without that extra) removes it from the venv, so
                      install it before a run, e.g.
                      `uv pip install --python /root/.callosum/venv/bin/python mujoco-warp==4.0.1`
                      or sync with `--extra sandbox-gpu`.

`randomization` selects the episode-randomness level of the *environment* the run uses
(`callosum.experiments.cube_wrench.config.RANDOMIZATION_PRESETS`): `"default"` keeps the learnable
ranges, `"harsh"` restores the harder private randomness (gain/authority spread, slips,
start delays). The trainer and the evaluator both build their env from
`WrenchPPOConfig.env_config()`, so a checkpoint's run preset is applied on every evaluation;
the preset is stored in the checkpoint config (an older checkpoint without the field falls
back to `"default"`).
"""

import argparse
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from callosum.common.bijepa_config import BiJEPAConfig
from callosum.common.cli import build_dataclass_parser
from callosum.experiments.coop_tasks.config import (
    TASKS as COOP_TASKS,
)
from callosum.experiments.coop_tasks.config import (
    CartPoleConfig,
    PendulumConfig,
    TossConfig,
    coop_action_size,
    coop_config,
    coop_obs_size,
)
from callosum.experiments.coop_tasks.logic import (
    AGENT_KEYS as COOP_AGENT_KEYS,
)
from callosum.experiments.coop_tasks.logic import OCCLUSION_MODES, validate_occlusion
from callosum.experiments.cube_wrench.config import (
    RANDOMIZATION_PRESETS,
    WrenchCubeConfig,
    wrench_cube_config,
)
from callosum.experiments.cube_wrench.logic import (
    AGENT_KEYS as CUBE_AGENT_KEYS,
)
from callosum.experiments.cube_wrench.logic import (
    PARTNER_MODES,
    obs_size,
    validate_partner_mode,
)

TASKS = ("cube",) + COOP_TASKS
"""Environments of a run: the default `CubeWrench-v0` (`"cube"`, the unchanged path) or one
of the reduced two-agent coop tasks (`"cartpole"`, `"pendulum"`, `"toss"`).

The coop suite has the same two backends as the cube: the batched GPU `mujoco_warp` backend
(`callosum.experiments.coop_tasks.env_gpu.CoopBatchGPU`, the DEFAULT on a CUDA host) and the
CPU `callosum.experiments.coop_tasks.vec_env.VecCoopEnv` loop (parity checks, CI, macOS unit
tests). `default_backend()`/`parse_args` resolve the `"auto"` backend of a run: the cube keeps
its historical CPU default, a coop task picks `"warp"` when a CUDA device is visible and falls
back to `"cpu"` otherwise (with a warning). The cube keeps its original backend selection."""

INPUT_MODES = ("none", "oracle", "predicted")
"""Policy-input modes: the env's partner visibility (`none`, `oracle`) plus the Bi-JEPA
`predicted` mode, which keeps the env at `partner_mode="none"` and appends the predicted
partner latent to the agent's own observation."""

BACKENDS = ("cpu", "warp")
"""Environment backends: the CPU MuJoCo loop (`callosum.experiments.cube_wrench.vec_env` /
`callosum.experiments.coop_tasks.vec_env`) or the batched mujoco_warp GPU env
(`callosum.experiments.cube_wrench.env_gpu` / `callosum.experiments.coop_tasks.env_gpu`)."""


def cuda_visible() -> bool:
    """Whether a CUDA device looks available, without importing torch (this module is stdlib).

    The check is only used to resolve the task-dependent default of :attr:`WrenchPPOConfig.backend`
    at CLI parse time: the GPU stack itself is imported lazily by the trainer, so a wrong guess
    falls back to the CPU path and never breaks a pure config/CI run.

    Returns:
        True on Linux when `/dev/nvidia0` exists or `nvidia-smi` is on `PATH`.
    """
    if not sys.platform.startswith("linux"):
        return False
    return Path("/dev/nvidia0").exists() or shutil.which("nvidia-smi") is not None


def default_backend(task: str) -> str:
    """The backend a run of `task` uses when `--backend` is not given.

    The cube keeps its historical CPU default (the old path is unchanged); the coop suite
    defaults to the batched GPU backend, because the pod's CPU quota makes CPU training of the
    suite hours instead of minutes (`--backend cpu` remains for parity checks and CI).
    """
    if task == "cube":
        return "cpu"
    return "warp" if cuda_visible() else "cpu"


def _resolve_runtime_defaults(values: dict, explicit: set[str]) -> tuple[dict, set[str]]:
    """Fill `backend`/`device` of a parsed CLI dict from the task-dependent default.

    Mutates and returns `values` (the parsed field dict) and the explicit-flag set: the
    resolved backend/device count as explicit so a `--resume` takes them from the CLI instead
    of the checkpoint's runtime resources.
    """
    if "backend" not in explicit:
        values["backend"] = default_backend(values.get("task", "cube"))
        explicit = explicit | {"backend"}
        if values["backend"] == "warp" and "device" not in explicit:
            values["device"] = "cuda"
            explicit = explicit | {"device"}
    return values, explicit


def task_obs_size(task: str, partner_mode: str, occlusion: str = "none") -> int:
    """Width of one agent's flat env observation of `task` under `partner_mode`/`occlusion`.

    Single place the trainer, the checkpoint-compatibility check and the evaluators ask for
    a task's observation width (the cube via `_wrench_logic.obs_size`, the coop tasks via
    `_coop_logic.obs_size`), so a task and its width cannot drift apart.

    Args:
        task: One of `TASKS`.
        partner_mode: `"none"` or `"oracle"`.
        occlusion: `"none"` or `"partner"` (the toss-catch hidden-partner variant); the cube
            has no occlusion variant, so only `"none"` is accepted for it.

    Returns:
        The flat observation width.

    Raises:
        ValueError: for an unknown task or an occlusion the task does not support.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if task == "cube":
        if occlusion != "none":
            raise ValueError(f"the cube task has no occlusion variant, got occlusion={occlusion!r}")
        return obs_size(partner_mode)
    return coop_obs_size(task, partner_mode, occlusion=occlusion)


def task_action_size(task: str) -> int:
    """Width of one agent's action of `task` (6 for the cube, 1 or 3 for the coop tasks).

    The one-sided force tasks (`cartpole`, `pendulum`) have a scalar action; the toss-catch
    task's paddles take `(dx, dz, dtheta)`, so its width comes from the env spec.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    return 6 if task == "cube" else coop_action_size(task)


def checkpoint_task(payload: dict) -> str:
    """The task recorded in a checkpoint payload (default `"cube"` for old checkpoints).

    Checkpoints written before the task suite predate the `task` field; their runs are
    the cube runs, so the fallback keeps them loadable.
    """
    config = payload.get("config") or {}
    task = str(config.get("task", "cube"))
    if task not in TASKS:
        raise ValueError(f"checkpoint records unknown task {task!r}; known: {TASKS}")
    return task


@dataclass
class WrenchPPOConfig:
    """Hyperparameters of the continuous-wrench IPPO trainer.

    The defaults are the plan's starting point for the `none`/`oracle` ablation runs: 128
    envs x 64 control steps per update (8192 transitions), four PPO epochs, a 0.99
    discount over the 350-step episodes. The run directory is `<runs_dir>/<exp_name>`
    (an auto-generated name when `exp_name` is `None`); everything the trainer writes
    (TensorBoard events, `config.json`, `latest.pt`) lives there.
    """

    # --- Run ------------------------------------------------------------------------------
    exp_name: str | None = None
    """Run name; the run directory is `<runs_dir>/<exp_name>` (default: name from the mode,
    seed and time)."""
    runs_dir: str = "runs"
    """Parent directory of the runs (git-ignored)."""
    seed: int = 0
    """Base seed: torch/numpy RNG, the initial env reset seeds and the evaluation seeds."""
    total_updates: int = 300
    """Number of PPO updates to run (`total_steps = total_updates * batch_size`)."""
    resume: str | None = None
    """Continue a killed run: a run directory (or its `latest.pt`). The trainer then uses the
    hyperparameters saved in the checkpoint and continues from its update counter with the
    saved optimizer and torch RNG state. A finished run returns immediately."""

    # --- Environment / rollout ------------------------------------------------------------
    task: str = "cube"
    """Environment of the run, one of `TASKS`: `"cube"` (CubeWrench-v0, the thesis main line
    and the default -- the original trainer path, unchanged) or a reduced coop task
    (`"cartpole"`, `"pendulum"`, `"toss"`, PLAN.md "Two-robot task suite"). The coop tasks use the
    `callosum.experiments.coop_tasks` harness and the CPU-only vectorisation
    `callosum.experiments.coop_tasks.vec_env`; their observation width/action width come from the env
    (`coop_obs_size`/`coop_action_size`), so the Bi-JEPA `predicted` input works unchanged
    (own observation + latent)."""
    partner_mode: str = "none"
    """Partner-visibility mode of the *env*: "none" (control) or "oracle" (upper bound); it
    also fixes the env observation width (`obs_size(partner_mode)`: 26 or 35). The third
    ablation arm, `predicted`, is not an env mode: it is `partner_mode="none"` plus
    `use_bijepa=True` (see :attr:`input_mode`)."""
    occlusion: str = "none"
    """Hidden-partner observation variant of the toss-catch task (`PLAN.md` "Hidden-partner
    regime"): `"none"` (default) is the full shared block (the ball and BOTH paddles),
    `"partner"` hides the partner paddle's pose/velocity/tilt from the shared block while
    keeping the ball, the agent's own paddle and the privileged `oracle` partner block. It is
    the `OccludedToss-v0` variant and is only defined for `task="toss"`: the `none`/`oracle`
    arms then train/evaluate on the occluded observation (20/30 fields instead of 26/36) and
    `predicted` is `partner_mode="none"` + `use_bijepa=True` + `occlusion="partner"`. The
    cube, cart-pole and pendulum tasks reject it (their shared block has no per-agent
    fields)."""
    num_envs: int = 128
    """Envs stepped in one Python loop per control step (the plan's training range is
    64-256; the vectorisation is a loop, see `callosum.experiments.cube_wrench.vec_env`)."""
    rollout_steps: int = 64
    """Control steps per env per update (the plan's range is 32-128). One update therefore
    consumes `num_envs * rollout_steps` transitions."""
    backend: str = "cpu"
    """Environment backend: `"cpu"` (the single-instance MuJoCo loop, works on macOS and the
    pod) or `"warp"` (the batched mujoco_warp GPU env; requires `device="cuda"` and the
    optional `sandbox-gpu`/`mujoco-warp==4.0.1` install on the pod). The dataclass default is
    `"cpu"` so the historical cube path stays byte-identical; the CLI resolves the run's
    default by task (`--backend` omitted: cube -> cpu, coop -> warp on a CUDA host, cpu
    otherwise, see `default_backend`). The policy/checkpoint format is backend-independent, so
    a checkpoint trained on one backend evaluates on the other."""
    randomization: str = "default"
    """Episode-randomness preset of the environment (`"default"` or `"harsh"`), see
    `callosum.experiments.cube_wrench.config.RANDOMIZATION_PRESETS`. It is part of the checkpoint's
    config, so the evaluator rebuilds the env with the same preset as the run; only the
    private-randomness ranges differ between the presets."""

    # --- Bi-JEPA (`predicted` input mode) ---------------------------------------------------
    use_bijepa: bool = False
    """Train the Bi-JEPA auxiliary objective and feed `ẑ_partner` to both policies
    (`input_mode="predicted"`). Requires `partner_mode="none"`: the policy input becomes the
    agent's own 26-d observation plus `latent_dim` latent values (58 by default), and the env
    never receives the privileged partner block. `False` keeps the `none`/`oracle` baselines
    and their checkpoints bit-for-bit unchanged."""
    latent_dim: int = 32
    """Width `d` of the Bi-JEPA latent, i.e. of the extra policy input under `predicted`."""
    history_len: int = 1
    """Length `K` of the own observation/latent history window feeding the partner predictor
    (`1` = the current latent only; `K > 1` uses a rolling window, reset at episode ends)."""
    aux_coef: float = 0.1
    """Weight of the Bi-JEPA auxiliary loss added to the trainer's objective each update."""
    ema_momentum: float = 0.99
    """EMA momentum of the target encoder (`target <- m * target + (1 - m) * online`)."""
    var_weight: float = 1.0
    """VICReg-style variance-hinge coefficient on the online latents (anti-collapse)."""
    cov_weight: float = 0.04
    """VICReg-style covariance-penalty coefficient on the online latents (anti-collapse)."""
    var_target: float = 1.0
    """Target per-dim latent std of the variance hinge (a floor, not an inflation force)."""

    # --- PPO ------------------------------------------------------------------------------
    gamma: float = 0.99
    """Discount per control step (episodes are up to 350 control steps long)."""
    gae_lambda: float = 0.95
    """GAE lambda."""
    clip_coef: float = 0.2
    """Clipped-surrogate coefficient (the policy-ratio clip)."""
    clip_vloss: bool = False
    """Clip the value loss as well (ManiSkill's PPO default: off)."""
    update_epochs: int = 4
    """PPO epochs over one rollout (the plan's range is 4-10)."""
    num_minibatches: int = 8
    """Minibatches per epoch; `minibatch_size = batch_size // num_minibatches`."""
    learning_rate: float = 3e-4
    """Adam learning rate (the same for both policies; no decay schedule)."""
    ent_coef: float = 0.0
    """Entropy-bonus coefficient (the Gaussian policies' exploration term)."""
    vf_coef: float = 0.5
    """Value-loss coefficient."""
    max_grad_norm: float = 0.5
    """Gradient-norm clip of both actor-critics."""
    norm_adv: bool = True
    """Normalise the advantages per minibatch (as in PPO)."""
    target_kl: float | None = 0.1
    """Early-stop the PPO epochs when the approximate KL exceeds this (`None` = off)."""
    device: str = "cpu"
    """Torch device. The env is CPU MuJoCo, so the tiny MLPs are trained on the CPU too."""

    # --- Evaluation / logging / checkpoints ------------------------------------------------
    eval_every: int = 10
    """Greedy evaluation every this many updates (and on the last one)."""
    eval_episodes: int = 32
    """Finished episodes per greedy evaluation."""
    eval_envs: int = 8
    """Envs the greedy evaluation runs in parallel (a separate small `VecWrenchCube`)."""
    checkpoint_every: int = 25
    """Write `latest.pt` every this many updates (and on the last one / on SIGTERM)."""

    def __post_init__(self) -> None:
        """Reject a config the trainer cannot run (the checks are the plan's sanity rules)."""
        validate_partner_mode(self.partner_mode)
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        validate_occlusion(self.occlusion)
        if self.occlusion != "none" and self.task != "toss":
            raise ValueError(
                "occlusion is only defined for --task toss (the other tasks have no "
                f"per-agent shared fields), got task={self.task!r}, "
                f"occlusion={self.occlusion!r}"
            )
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.randomization not in RANDOMIZATION_PRESETS:
            raise ValueError(
                f"randomization must be one of {RANDOMIZATION_PRESETS}, got {self.randomization!r}"
            )
        # The coop suite's private randomisation is part of its task configs at this step
        # (no `default`/`harsh` presets yet).
        if self.task != "cube" and self.randomization != "default":
            raise ValueError(
                "the coop tasks do not have randomisation presets yet at this step; "
                f"randomization must be 'default', got {self.randomization!r}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.total_updates < 1:
            raise ValueError(f"total_updates must be >= 1, got {self.total_updates}")
        if self.num_envs < 2:
            raise ValueError(
                f"num_envs must be >= 2 (advantage normalisation needs several samples), got "
                f"{self.num_envs}"
            )
        if self.rollout_steps < 1:
            raise ValueError(f"rollout_steps must be >= 1, got {self.rollout_steps}")
        if not 0 < self.gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if not 0 < self.clip_coef < 1:
            raise ValueError(f"clip_coef must be in (0, 1), got {self.clip_coef}")
        if self.update_epochs < 1 or self.num_minibatches < 1:
            raise ValueError(
                f"update_epochs and num_minibatches must be >= 1, got "
                f"({self.update_epochs}, {self.num_minibatches})"
            )
        if not self.learning_rate > 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.ent_coef < 0 or self.vf_coef < 0:
            raise ValueError(
                f"ent_coef and vf_coef must be >= 0, got ({self.ent_coef}, {self.vf_coef})"
            )
        if not self.max_grad_norm > 0:
            raise ValueError(f"max_grad_norm must be > 0, got {self.max_grad_norm}")
        if self.target_kl is not None and not self.target_kl > 0:
            raise ValueError(f"target_kl must be > 0 or None, got {self.target_kl}")
        if (
            self.eval_every < 1
            or self.eval_episodes < 1
            or self.eval_envs < 1
            or self.checkpoint_every < 1
        ):
            raise ValueError(
                "eval_every, eval_episodes, eval_envs and checkpoint_every must be >= 1, got "
                f"({self.eval_every}, {self.eval_episodes}, {self.eval_envs}, "
                f"{self.checkpoint_every})"
            )
        if not self.device:
            raise ValueError("device must be a non-empty torch device string")
        if self.backend == "warp" and not self.device.startswith("cuda"):
            raise ValueError(
                "backend='warp' steps the worlds on a CUDA device of mujoco_warp, so "
                f"device must be a CUDA device string ('cuda' or 'cuda:N'), got "
                f"device={self.device!r}"
            )
        # The `predicted` input is defined on top of the unprivileged `none` observation: the
        # `oracle` env already exposes the partner block, so combining both would leak the
        # ground truth through the env input and make the ablation uninterpretable.
        if self.use_bijepa and self.partner_mode != "none":
            raise ValueError(
                "use_bijepa=True requires partner_mode='none' (the oracle env already "
                f"exposes the partner block), got partner_mode={self.partner_mode!r}"
            )
        if self.aux_coef < 0:
            raise ValueError(f"aux_coef must be >= 0, got {self.aux_coef}")
        # Delegates the latent/history/EMA/regulariser validation to the Bi-JEPA config, so an
        # invalid value is rejected here (before the trainer builds the module).
        self.bijepa_config()
        # The batch layout: `ppo_update` normalises advantages per minibatch and needs the
        # rollout to split into equal minibatches (a one-sample minibatch has an undefined
        # advantage std).
        if self.batch_size % self.num_minibatches != 0:
            valid = [n for n in range(1, self.batch_size // 2 + 1) if self.batch_size % n == 0]
            raise ValueError(
                f"batch size num_envs * rollout_steps = {self.num_envs} * "
                f"{self.rollout_steps} = {self.batch_size} must be divisible by "
                f"num_minibatches ({self.num_minibatches}); valid num_minibatches: {valid}"
            )
        if self.minibatch_size < 2:
            raise ValueError(
                f"minibatch_size = batch_size // num_minibatches = {self.minibatch_size} must "
                "be >= 2 (advantage normalisation needs more than one sample)"
            )

    @property
    def batch_size(self) -> int:
        """Transitions per update: `num_envs * rollout_steps`."""
        return self.num_envs * self.rollout_steps

    @property
    def minibatch_size(self) -> int:
        """Transitions per gradient step: `batch_size // num_minibatches`."""
        return self.batch_size // self.num_minibatches

    @property
    def total_steps(self) -> int:
        """Environment steps of the whole run: `batch_size * total_updates`."""
        return self.batch_size * self.total_updates

    @property
    def env_obs_size(self) -> int:
        """Width of one agent's flat *env* observation for this task and partner mode.

        The cube: 28 (`none`) / 45 (`oracle`) fields. The coop tasks: 11/17 (cart-pole) and
        10/16 (pendulum); see `coop_obs_size`. The policy input under `predicted` appends
        `latent_dim` to this width.
        """
        return task_obs_size(self.task, self.partner_mode, self.occlusion)

    @property
    def action_size(self) -> int:
        """Width of one agent's action: 6 (6D wrench) for the cube, 1 for the coop tasks."""
        return task_action_size(self.task)

    @property
    def agent_keys(self) -> tuple[str, str]:
        """The two agents' dict keys, in the env's action-/observation-space order.

        The cube keeps its historical `("holder", "rotator")`; the coop suite uses
        `("left", "right")` (the one-sided pushers / torque actuators). The trainer indexes
        its observation and action buffers by this tuple, so the same code drives both.
        """
        return CUBE_AGENT_KEYS if self.task == "cube" else COOP_AGENT_KEYS

    @property
    def obs_size(self) -> int:
        """Backward-compatible alias of :attr:`env_obs_size` (the env observation width).

        The *policy* input width is :attr:`policy_input_size`, which is `obs_size + latent_dim`
        under `predicted`; the env (and its checkpoints' `obs_size` key) keeps this value.
        """
        return self.env_obs_size

    @property
    def policy_input_size(self) -> int:
        """Width of one policy's input: env observation plus the predicted latent if enabled."""
        return self.env_obs_size + (self.latent_dim if self.use_bijepa else 0)

    def env_config(self) -> WrenchCubeConfig:
        """The `WrenchCubeConfig` of this run's randomisation preset (the cube task only).

        Both env backends (`VecWrenchCube`/`WrenchCubeEnv` and `WrenchCubeBatchGPU`) and the
        evaluator build their environment from this instance, so a run and its evaluation
        use the same episode randomness (the physics/timing/reward are preset-independent).
        """
        return wrench_cube_config(self.randomization)

    def coop_env_config(self) -> CartPoleConfig | PendulumConfig | TossConfig:
        """The validated per-task config of a coop run (raises for the cube task).

        `callosum.experiments.coop_tasks.vec_env` and `callosum/experiments/coop_tasks/eval.py` build their envs from this
        instance, so a run and its evaluation use the same task parameters.
        """
        if self.task == "cube":
            raise ValueError("coop_env_config() is only defined for the coop tasks")
        if self.task == "toss":
            return coop_config(self.task, occlusion=self.occlusion)
        return coop_config(self.task)

    @property
    def input_mode(self) -> str:
        """The run's policy-input mode: `"predicted"` with Bi-JEPA, else the env partner mode."""
        return "predicted" if self.use_bijepa else self.partner_mode

    def bijepa_config(self) -> BiJEPAConfig:
        """The Bi-JEPA hyperparameters this trainer instantiation builds.

        Only the fields that matter for a PPO run are exposed on `WrenchPPOConfig`; the
        encoder/predictor widths, activation, layer norm, sharing and target kind keep the
        module's own defaults (`callosum.common.bijepa_config.BiJEPAConfig`).
        """
        return BiJEPAConfig(
            latent_dim=self.latent_dim,
            history_len=self.history_len,
            ema_momentum=self.ema_momentum,
            var_weight=self.var_weight,
            cov_weight=self.cov_weight,
            var_target=self.var_target,
        )


def build_parser() -> argparse.ArgumentParser:
    """An argparse parser with one flag per `WrenchPPOConfig` field (`--field-name`)."""
    return build_dataclass_parser(
        WrenchPPOConfig,
        "Train two independent Gaussian PPO policies (IPPO) on CubeWrench-v0 or a "
        "coop task (--task).",
        choices={
            "task": TASKS,
            "partner_mode": PARTNER_MODES,
            "occlusion": OCCLUSION_MODES,
            "backend": BACKENDS,
            "randomization": RANDOMIZATION_PRESETS,
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> WrenchPPOConfig:
    """Parse CLI flags into a `WrenchPPOConfig`, e.g. `--partner-mode oracle --seed 0`.

    With `--resume <path>` the flags are still parsed and validated, but the training run
    itself continues with the config saved in the checkpoint (see
    `callosum.experiments.cube_wrench.train.run`); the runtime resources (`--backend`, `--device`)
    are the exception -- they are taken from the CLI, see
    `wrench_ppo.apply_runtime_overrides`.

    `--backend` is resolved by task when it is omitted (`default_backend`): the cube stays on
    the CPU loop, a coop task runs on the GPU when a CUDA device is visible.
    """
    parsed = vars(build_parser().parse_args(argv))
    values, _explicit = _resolve_runtime_defaults(dict(parsed), set(parsed))
    return WrenchPPOConfig(**values)


def parse_args_with_explicit(
    argv: Sequence[str] | None = None,
) -> tuple[WrenchPPOConfig, frozenset[str]]:
    """`parse_args` plus the names of the fields the user actually passed.

    The trainer needs the explicit set for `--resume`: the checkpoint wins for the learning
    hyperparameters, but `backend`/`device` are runtime resources and must come from the CLI
    (a flag that was not given keeps the checkpoint's value; the resolved task default counts
    as given, so a GPU-default coop run does not resume onto a saved CPU device).
    """
    values = vars(build_parser().parse_args(argv))
    values, explicit = _resolve_runtime_defaults(dict(values), set(values))
    return WrenchPPOConfig(**values), frozenset(explicit)
