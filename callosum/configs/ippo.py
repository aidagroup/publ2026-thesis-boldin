"""Config and CLI of the IPPO trainer (`callosum.training.ippo`, step 2.1).

Plain stdlib (dataclass + argparse), no torch / mani_skill, so parsing and validation are
unit-testable on macOS and in CI. ManiSkill's PPO baseline (`examples/baselines/ppo/ppo.py` @
v3.0.1) uses `tyro` for the CLI; it is not a dependency of this project, so every dataclass
field gets an argparse flag instead (`--num-envs`, `--no-anneal-lr`, ...).
"""

import argparse
import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from callosum.configs._cli import build_dataclass_parser
from callosum.envs._partner_obs import PARTNER_OBS_MODES, validate_partner_obs
from callosum.envs._sim_compat import is_cpu_backend

SIM_BACKENDS = ("gpu", "cpu", "physx_cuda", "physx_cpu")


@dataclass
class IPPOConfig:
    """Hyperparameters of the two independent PPO learners (agent_a = `so101_pg-0`, agent_b =
    `so101_pg-1`). Defaults follow ManiSkill's PPO baseline unless a comment says otherwise.
    """

    # --- Run -----------------------------------------------------------------------------
    env_id: str = "FaceTurn-v0"
    exp_name: str | None = None
    """Run name; the output directory is `<runs_dir>/<exp_name>` (default: env, seed, time)."""
    runs_dir: str = "runs"
    seed: int = 1
    checkpoint: str | None = None
    """Warm start: load both agents' weights from this checkpoint file (any format). Optimizer
    state, counters and the LR schedule start fresh. Mutually exclusive with `resume`."""
    resume: str | None = None
    """Exact resume: a run directory (e.g. `runs/faceturn_v3_s2`) with `config.json` and a
    format-2 `latest.pt`. The run continues in that directory from the saved iteration with the
    saved optimizer state, RNG states, LR schedule position and best-eval tracking. The
    hyperparameters, seed and total_timesteps are taken from the directory's `config.json`; any
    other flag given explicitly must equal the saved value. A finished run returns immediately."""

    # --- Environment ---------------------------------------------------------------------
    sim_backend: str = "gpu"
    """"gpu" (server) or "cpu" (macOS smoke runs; forces a single env and a single eval env)."""
    partner_obs: str = "full"
    """Who sees which TCP pose in the per-agent inputs (`callosum.training._agent_obs`): "full"
    = own and partner's, "none" = own only. The env always emits both poses; the visibility rule
    is applied by the trainer's input builder, not by the env."""
    control_mode: str | None = "pd_joint_delta_pos"
    reward_mode: str = "normalized_dense"
    """Both agents learn from this one shared reward (per-agent rewards are a later concern).
    "normalized_dense" is what ManiSkill's PPO baseline uses; the reward scale is the env's
    `compute_normalized_dense_reward` (FaceTurn: dense reward / sum of the positive weights)."""
    max_episode_steps: int | None = None
    """Episode length (control steps). `None` (default) keeps the env's registered value: 400 for
    FaceTurn-v0 (the scripted expert needs ~330 steps), 100 for TwoSO101-v0. Set a number to
    override both."""

    # --- Rollout / evaluation ------------------------------------------------------------
    total_timesteps: int = 10_000_000
    num_envs: int = 256
    num_steps: int = 100
    """Rollout length per iteration and env; the batch is `num_envs * num_steps`."""
    partial_reset: bool = True
    """Training envs reset on termination (success) as well as truncation, like the baseline."""
    num_eval_envs: int = 16
    num_eval_steps: int | None = None
    """Steps per evaluation; `None` = `max_episode_steps`. Must cover a full episode."""
    eval_freq: int = 20
    """Evaluate every this many iterations (and at the first and last one)."""
    checkpoint_freq: int = 20
    """Write `latest.pt` every this many iterations (and at the end); `best.pt` follows eval."""

    # --- PPO -----------------------------------------------------------------------------
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    """Linear learning-rate decay to 0 over this run's `num_iterations`, as ManiSkill's PPO
    baseline (`--anneal_lr`): iteration `k` (1-based) uses `learning_rate * (1 - (k-1)/N)`. The
    fixed rate was too large for the late, near-deterministic policy (KL above `target_kl` on
    almost every iteration, cutting the PPO epochs short). The schedule belongs to the run: with
    a warm start (`checkpoint`) it restarts from `learning_rate` for the new run's iterations
    (the optimizer state is not restored either), while `resume` continues at the saved
    iteration. Disable with `--no-anneal-lr`."""
    gamma: float = 0.99
    """0.99, not the baseline's 0.8: 0.8 is a 5-step horizon, tuned for 50-step tasks with an
    immediate reward. FaceTurn needs hundreds of steps and its grasp/turn reward comes late."""
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.1
    reward_scale: float = 1.0

    # --- Fine-tuning from demonstrations (see callosum.training.bc) ----------------------------
    critic_warmup_iters: int = 0
    """For the first this many iterations only the critics are updated and the actors stay frozen.
    After a BC warm start (`checkpoint`) the critic does not know the noisy on-policy returns yet,
    and PPO advantages computed with it would push the BC policy around at random; 0 = off (the
    plain PPO behaviour). The window is by iteration, so `resume` continues it where it stopped."""
    demos: str | None = None
    """Demo file (`scripts/collect_demos.py`) for the auxiliary BC loss (`bc_coef`)."""
    bc_coef: float = 0.0
    """DAPG-style auxiliary loss: `bc_coef * mean((actor_mean(demo_obs) - demo_action)^2)` on a
    demo minibatch, added to every PPO minibatch step's actor loss. The coefficient decays
    linearly to 0 over `bc_decay_iters` iterations (`bc_coef_at`); 0 = off. Needs `demos`."""
    bc_decay_iters: int = 100
    """Iterations over which `bc_coef` decays linearly to 0 (iteration `k` uses
    `bc_coef * max(0, 1 - (k - 1) / bc_decay_iters)`)."""
    bc_batch_size: int = 256
    """Demo transitions per BC-loss minibatch (one demo minibatch per PPO minibatch step)."""

    # --- Evaluation only -----------------------------------------------------------------
    eval_only: bool = False
    """Do not train: load `checkpoint`, evaluate its deterministic policies on the eval envs
    `eval_repeats` times (reset seeds `seed + 1 + k`), print and save the summary to
    `<runs_dir>/<exp_name>/eval.json`. The key number of a BC checkpoint."""
    eval_repeats: int = 1
    """Number of evaluation rounds of `eval_only` (each: `num_eval_envs` episodes)."""

    def __post_init__(self) -> None:
        validate_partner_obs(self.partner_obs)
        if self.sim_backend not in SIM_BACKENDS:
            raise ValueError(f"sim_backend must be one of {SIM_BACKENDS}, got {self.sim_backend!r}")
        for name in ("num_envs", "num_steps", "num_eval_envs", "eval_freq", "checkpoint_freq"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.max_episode_steps is not None and self.max_episode_steps < 1:
            raise ValueError(f"max_episode_steps must be >= 1, got {self.max_episode_steps}")
        if (
            self.num_eval_steps is not None
            and self.max_episode_steps is not None
            and self.num_eval_steps < self.max_episode_steps
        ):
            raise ValueError(
                f"num_eval_steps ({self.num_eval_steps}) must cover a full episode "
                f"(max_episode_steps = {self.max_episode_steps}); otherwise no episode "
                "finishes during evaluation and the eval metrics stay empty"
            )
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
        if self.num_minibatches < 1 or self.update_epochs < 1:
            raise ValueError("num_minibatches and update_epochs must be >= 1")
        if self.critic_warmup_iters < 0 or self.bc_coef < 0:
            raise ValueError("critic_warmup_iters and bc_coef must be >= 0")
        if self.bc_decay_iters < 1 or self.bc_batch_size < 1 or self.eval_repeats < 1:
            raise ValueError("bc_decay_iters, bc_batch_size and eval_repeats must be >= 1")
        if self.bc_coef > 0 and not self.demos:
            raise ValueError("bc_coef > 0 needs demos (--demos <file>)")
        if self.eval_only and not self.checkpoint:
            raise ValueError("eval_only needs a checkpoint to evaluate (--checkpoint <file>)")
        if self.eval_only and self.resume:
            raise ValueError("eval_only and resume are mutually exclusive")
        # The batch layout only matters when something is trained.
        if not self.eval_only:
            if self.minibatch_size < 2:
                raise ValueError(
                    f"minibatch_size = num_envs * num_steps // num_minibatches = {self.minibatch_size}"
                    " must be >= 2 (advantage normalisation needs more than one sample)"
                )
            if self.batch_size % self.num_minibatches != 0:
                valid = [n for n in range(1, self.batch_size // 2 + 1) if self.batch_size % n == 0]
                raise ValueError(
                    f"batch size num_envs * num_steps = {self.num_envs} * {self.num_steps} = "
                    f"{self.batch_size} must be divisible by num_minibatches ({self.num_minibatches}); "
                    "otherwise the last minibatch is smaller (possibly a single sample, whose "
                    f"advantage std is undefined). Valid num_minibatches: {valid}"
                )
            if self.total_timesteps < self.batch_size:
                raise ValueError(
                    f"total_timesteps ({self.total_timesteps}) is smaller than one batch "
                    f"({self.batch_size} = num_envs * num_steps)"
                )

    @property
    def batch_size(self) -> int:
        """Transitions per iteration: `num_envs * num_steps`."""
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        """Transitions per gradient step."""
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        """Number of PPO iterations (`total_timesteps // batch_size`)."""
        return self.total_timesteps // self.batch_size


def learning_rate_at(cfg: IPPOConfig, iteration: int) -> float:
    """Learning rate of the 1-based PPO `iteration`: linear decay from `learning_rate` towards 0
    over `num_iterations` if `anneal_lr`, else constant."""
    if not cfg.anneal_lr:
        return cfg.learning_rate
    return (1.0 - (iteration - 1.0) / cfg.num_iterations) * cfg.learning_rate


def bc_coef_at(cfg: IPPOConfig, iteration: int) -> float:
    """Coefficient of the auxiliary BC loss in the 1-based PPO `iteration` (0 when it is off)."""
    if not cfg.demos or cfg.bc_coef <= 0:
        return 0.0
    return cfg.bc_coef * max(0.0, 1.0 - (iteration - 1.0) / cfg.bc_decay_iters)


def critic_warmup_active(cfg: IPPOConfig, iteration: int) -> bool:
    """True while the 1-based `iteration` is inside the critic-only warm-up window."""
    return iteration <= cfg.critic_warmup_iters


def clamp_envs_for_cpu_backend(values: dict) -> str | None:
    """Clamp the env counts in `values` (IPPOConfig field values) to one on the CPU sim.

    ManiSkill's CPU backend supports a single env per scene. Returns a note to print if
    anything was changed, else `None`.
    """
    defaults = IPPOConfig.__dataclass_fields__
    sim_backend = values.get("sim_backend", defaults["sim_backend"].default)
    if not is_cpu_backend(sim_backend):
        return None
    requested = {
        name: values.get(name, defaults[name].default) for name in ("num_envs", "num_eval_envs")
    }
    if all(n == 1 for n in requested.values()):
        return None
    values.update(num_envs=1, num_eval_envs=1)
    return (
        "note: the CPU sim backend supports a single env; using num_envs=1 and "
        f"num_eval_envs=1 (requested {requested['num_envs']} and {requested['num_eval_envs']})"
    )


def resolve_episode_length(cfg: IPPOConfig, env_max_episode_steps: int) -> int:
    """The evaluation length for an env whose episodes last `env_max_episode_steps` steps.

    Raises if `num_eval_steps` is set but too short, which also catches the case where
    `max_episode_steps` is `None` (env default) and `__post_init__` could not check it.
    """
    if cfg.num_eval_steps is None:
        return env_max_episode_steps
    if cfg.num_eval_steps < env_max_episode_steps:
        raise ValueError(
            f"num_eval_steps ({cfg.num_eval_steps}) must cover a full episode "
            f"({env_max_episode_steps} steps)"
        )
    return cfg.num_eval_steps


def build_parser() -> argparse.ArgumentParser:
    """An argparse parser with one flag per `IPPOConfig` field (`--field-name`)."""
    return build_dataclass_parser(
        IPPOConfig,
        "Train two independent PPO policies (IPPO) on a callosum two-arm env.",
        choices={"partner_obs": PARTNER_OBS_MODES, "sim_backend": SIM_BACKENDS},
    )


def env_reset_seeds(cfg: IPPOConfig, iteration: int) -> tuple[int, int]:
    """Seeds for the first reset of the training and the evaluation envs.

    A fresh run (`iteration` 0) uses `seed` and `seed + 1`. A run resumed after `iteration`
    iterations cannot restore the simulator state, so it re-seeds the resets from the iteration
    too: the initial states are not a replay of those at the start of the run.
    """
    base = cfg.seed + 1_000_003 * iteration
    return base, base + 1


def load_resume_values(resume_dir: str | Path, explicit: dict) -> dict:
    """`IPPOConfig` field values for resuming the run in `resume_dir` (pure; no torch).

    The values are the run's saved `config.json` (extra derived keys such as `obs_dims` are
    dropped) with `resume` set, so the resumed run uses exactly the original hyperparameters.
    `explicit` holds the flags given on the command line (`resume` included): each must equal
    the saved value, except that `exp_name` / `runs_dir` must match the directory itself. The run
    directory is `resume_dir`, period: `exp_name` and `runs_dir` are derived from it, so a
    renamed or moved run still resumes in place.

    Raises:
        ValueError: no `config.json`, `--checkpoint` given as well, or conflicting flags.
    """
    run_dir = Path(resume_dir)
    if not run_dir.name:
        run_dir = run_dir.resolve()
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(f"cannot resume {run_dir}: {config_path} not found")
    if explicit.get("checkpoint") is not None:
        raise ValueError("--resume and --checkpoint are mutually exclusive")
    saved = json.loads(config_path.read_text())
    field_names = {f.name for f in dataclasses.fields(IPPOConfig)}
    values = {k: v for k, v in saved.items() if k in field_names}

    explicit = dict(explicit)
    if is_cpu_backend(values.get("sim_backend", "")):
        # The original run clamped the env counts on the CPU sim; clamp the flags the same way,
        # so re-running the very same command resumes cleanly.
        for name in ("num_envs", "num_eval_envs"):
            if name in explicit:
                explicit[name] = 1

    conflicts = []
    for name, value in explicit.items():
        if name == "resume":
            continue
        if name == "exp_name":
            if value != run_dir.name:
                conflicts.append(f"--exp-name {value} (resuming {run_dir.name})")
        elif name == "runs_dir":
            if Path(value).resolve() != run_dir.parent.resolve():
                conflicts.append(f"--runs-dir {value} (resuming in {run_dir.parent})")
        elif name not in values:
            conflicts.append(f"--{name.replace('_', '-')} {value} (not in the saved config)")
        elif values[name] != value:
            flag = "--" + name.replace("_", "-")
            conflicts.append(f"{flag} {value} (saved: {values[name]})")
    if conflicts:
        raise ValueError(
            f"flags conflict with the saved config of {run_dir}: " + "; ".join(conflicts)
        )
    values.update(resume=str(run_dir), runs_dir=str(run_dir.parent), exp_name=run_dir.name)
    return values


def parse_args(argv: Sequence[str] | None = None) -> IPPOConfig:
    """Parse CLI flags into an `IPPOConfig`, e.g. `--env-id TwoSO101-v0 --num-envs 64`.

    With a CPU `--sim-backend` the env counts are clamped to one (a note is printed). With
    `--resume DIR` the config is the one saved in `DIR/config.json` (see `load_resume_values`).
    """
    parser = build_parser()
    values = vars(parser.parse_args(argv))
    if values.get("resume") is not None:
        try:
            values = load_resume_values(values["resume"], values)
        except ValueError as err:
            parser.error(str(err))
        return IPPOConfig(**values)
    note = clamp_envs_for_cpu_backend(values)
    if note is not None:
        print(note)
    return IPPOConfig(**values)
