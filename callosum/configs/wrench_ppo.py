"""Config and CLI of the continuous-wrench IPPO trainer (PLAN.md step 2, `wrench_ppo`).

The reduced model (`CubeWrench-v0`) is CPU-only MuJoCo, so its trainer is a small
CPU-side IPPO: two independent Gaussian policies (holder, rotator) on one shared team
reward, a Python-loop vectorisation of the single-instance env, clipped PPO updates,
TensorBoard logs, greedy evaluation and `latest.pt` checkpoints with `--resume`.

This module is pure stdlib on purpose (same rule as `callosum.configs.wrench_cube` and
`callosum.envs._wrench_logic`): the config must import and validate on macOS/CI, where the
sandbox extra (mujoco) is absent, so every trainer flag is testable without the simulator.
Torch is imported only by `callosum.training.wrench_ppo`.

The field groups follow the plan's step-2 description: the run identity, the environment /
rollout shape, the PPO hyperparameters, and the logging / evaluation / checkpointing
cadence. The recommended ranges of the plan are 64-256 envs and 32-128 rollout control
steps; the validation below only enforces the hard sanity conditions, so a small
`--num-envs 8` server smoke run stays possible.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from callosum.configs._cli import build_dataclass_parser
from callosum.envs._wrench_logic import PARTNER_MODES, obs_size, validate_partner_mode


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
    partner_mode: str = "none"
    """Partner-visibility mode of the env: "none" (control) or "oracle" (upper bound); it
    also fixes the policy input width (`obs_size(partner_mode)`: 26 or 35)."""
    num_envs: int = 128
    """Envs stepped in one Python loop per control step (the plan's training range is
    64-256; the vectorisation is a loop, see `callosum.training._vec_wrench`)."""
    rollout_steps: int = 64
    """Control steps per env per update (the plan's range is 32-128). One update therefore
    consumes `num_envs * rollout_steps` transitions."""

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
    def obs_size(self) -> int:
        """Width of one agent's flat observation under `partner_mode` (26 for none, 35 for
        oracle)."""
        return obs_size(self.partner_mode)


def build_parser() -> argparse.ArgumentParser:
    """An argparse parser with one flag per `WrenchPPOConfig` field (`--field-name`)."""
    return build_dataclass_parser(
        WrenchPPOConfig,
        "Train two independent Gaussian PPO policies (IPPO) on CubeWrench-v0.",
        choices={"partner_mode": PARTNER_MODES},
    )


def parse_args(argv: Sequence[str] | None = None) -> WrenchPPOConfig:
    """Parse CLI flags into a `WrenchPPOConfig`, e.g. `--partner-mode oracle --seed 0`.

    With `--resume <path>` the flags are still parsed and validated, but the training run
    itself continues with the config saved in the checkpoint (see
    `callosum.training.wrench_ppo.run`); the only value that matters is `resume` itself.
    """
    return WrenchPPOConfig(**vars(build_parser().parse_args(argv)))
