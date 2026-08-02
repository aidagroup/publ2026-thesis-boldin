"""IPPO trainer config (step 2.1): dataclass + CLI parsing.

Field set and most defaults mirror mani-skill's own PPO baseline
(examples/baselines/ppo/ppo.py @ v3.0.1) -- the reference
callosum/training/ippo.py is adapted from. Deliberately dropped relative to
that reference: `track` (wandb) and `capture_video`/video-recording fields.
The training server has no Vulkan/EGL display setup (state-based training
needs none -- see docs/setup.md), and the plan authorizes adding only
`tensorboard` to the lockfile for this step, not `wandb`.

Uses stdlib argparse instead of the reference's `tyro` dependency, which the
plan does not authorize adding. Kept in this dependency-light module
(stdlib only -- unlike callosum/training/ippo.py, which needs mani_skill and
torch) specifically so the CLI parsing itself is unit-testable on macOS/CI,
where mani_skill is absent.
"""

import argparse
import dataclasses
import typing
from dataclasses import dataclass


@dataclass
class IPPOConfig:
    """Hyperparameters for callosum.training.ippo's two independent PPO
    agents (one per SO-100 arm)."""

    env_id: str = "FaceTurn-v0"
    control_mode: str | None = None  # None = env default (pd_joint_delta_pos for SO-100)
    seed: int = 1
    cuda: bool = True
    torch_deterministic: bool = True

    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4
    num_envs: int = 256  # conservative default per the plan; raise on the server
    num_eval_envs: int = 8
    num_steps: int = 50
    num_eval_steps: int = 50
    reconfiguration_freq: int | None = None
    eval_reconfiguration_freq: int | None = 1
    partial_reset: bool = True
    eval_partial_reset: bool = False

    anneal_lr: bool = False
    gamma: float = 0.8
    gae_lambda: float = 0.9
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

    eval_freq: int = 25
    save_model: bool = True
    exp_name: str | None = None

    # Diagnostic "oracle" condition for the paid session: if False (the
    # decentralized baseline) fails to learn on FaceTurn-v0, flip this to
    # isolate whether that's a task/reward-shaping problem or genuinely
    # requires partner information -- see
    # callosum.training._agent_obs.build_agent_obs.
    include_partner: bool = False

    # Computed at runtime from the fields above -- see callosum.training.ippo.
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


_COMPUTED_FIELDS = {"batch_size", "minibatch_size", "num_iterations"}


def _cli_type(field: dataclasses.Field) -> type:
    """Best-effort CLI parser type for a dataclass field (unwraps `X | None` to X)."""
    field_type = field.type
    args = typing.get_args(field_type)
    if args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            field_type = non_none[0]
    return field_type


def _str_to_bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean (true/false), got {value!r}")


def parse_args(argv: list[str] | None = None) -> IPPOConfig:
    """Parse CLI args into an IPPOConfig, e.g. `--env-id FaceTurn-v0 --total-timesteps 1000`.

    Every IPPOConfig field except the three runtime-computed ones gets a
    `--<field-name-with-dashes>` flag, defaulting to that field's dataclass
    default.
    """
    defaults = IPPOConfig()
    parser = argparse.ArgumentParser(
        description="Train independent PPO (IPPO) policies on TwoSO100Base-derived envs."
    )
    for f in dataclasses.fields(IPPOConfig):
        if f.name in _COMPUTED_FIELDS:
            continue
        flag = f"--{f.name.replace('_', '-')}"
        default = getattr(defaults, f.name)
        field_type = _cli_type(f)
        if field_type is bool:
            parser.add_argument(flag, type=_str_to_bool, default=default, metavar="{true,false}")
        else:
            parser.add_argument(flag, type=field_type, default=default)
    namespace = vars(parser.parse_args(argv))
    return IPPOConfig(**namespace)
