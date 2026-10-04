"""Config and CLI of the IPPO trainer (`callosum.training.ippo`, step 2.1).

Plain stdlib (dataclass + argparse), no torch / mani_skill, so parsing and validation are
unit-testable on macOS and in CI. ManiSkill's PPO baseline (`examples/baselines/ppo/ppo.py` @
v3.0.1) uses `tyro` for the CLI; it is not a dependency of this project, so every dataclass
field gets an argparse flag instead (`--num-envs`, `--no-anneal-lr`, ...).
"""

import argparse
import dataclasses
import types
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass

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
    """Warm start: load both agents' weights from this file (optimizer state is not saved)."""

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
    (the optimizer state is not restored either). Disable with `--no-anneal-lr`."""
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


def _optional(parse: Callable[[str], object]) -> Callable[[str], object]:
    """Argparse type that also accepts `none` / `null` for an `X | None` field."""

    def convert(text: str) -> object:
        return None if text.lower() in ("none", "null") else parse(text)

    convert.__name__ = parse.__name__
    return convert


def _unwrap_optional(annotation: object) -> tuple[type, bool]:
    """`(X, True)` for `X | None`, `(X, False)` for a plain `X`."""
    if isinstance(annotation, types.UnionType) or typing.get_origin(annotation) is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    """An argparse parser with one flag per `IPPOConfig` field (`--field-name`)."""
    parser = argparse.ArgumentParser(
        description="Train two independent PPO policies (IPPO) on a callosum two-arm env.",
        argument_default=argparse.SUPPRESS,
    )
    defaults = IPPOConfig()
    hints = typing.get_type_hints(IPPOConfig)
    for field in dataclasses.fields(IPPOConfig):
        flag = "--" + field.name.replace("_", "-")
        default = getattr(defaults, field.name)
        kind, optional = _unwrap_optional(hints[field.name])
        help_text = f"(default: {default})"
        if kind is bool:
            parser.add_argument(
                flag, action=argparse.BooleanOptionalAction, dest=field.name, help=help_text
            )
        else:
            parse = _optional(kind) if optional else kind
            kwargs = {}
            if field.name == "partner_obs":
                kwargs["choices"] = PARTNER_OBS_MODES
            elif field.name == "sim_backend":
                kwargs["choices"] = SIM_BACKENDS
            parser.add_argument(flag, type=parse, dest=field.name, help=help_text, **kwargs)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> IPPOConfig:
    """Parse CLI flags into an `IPPOConfig`, e.g. `--env-id TwoSO101-v0 --num-envs 64`.

    With a CPU `--sim-backend` the env counts are clamped to one (a note is printed).
    """
    values = vars(build_parser().parse_args(argv))
    note = clamp_envs_for_cpu_backend(values)
    if note is not None:
        print(note)
    return IPPOConfig(**values)
