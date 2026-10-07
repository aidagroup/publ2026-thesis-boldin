"""Config and CLI of the behaviour-cloning pretraining (`callosum.training.bc`).

Plain stdlib (no torch / mani_skill), so parsing and validation are unit-testable everywhere.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from callosum.configs._cli import build_dataclass_parser
from callosum.envs._partner_obs import PARTNER_OBS_MODES, validate_partner_obs

DEVICES = ("auto", "cpu", "cuda")


@dataclass
class BCConfig:
    """Hyperparameters of the BC pretraining of the two ActorCritics from expert demonstrations.

    The actor means are regressed to the demonstrated actions (MSE), the critics to the
    discounted Monte-Carlo returns of the demonstrations' rewards; the policy's log-std is not
    learned but set to `actor_logstd`, so that an IPPO fine-tune starts close to the expert.
    """

    demos: str = ""
    """Demo file written by `scripts/collect_demos.py` (required)."""
    exp_name: str | None = None
    """Run name; outputs go to `<runs_dir>/<exp_name>/` (default: `bc__<seed>__<time>`)."""
    runs_dir: str = "runs"
    seed: int = 1
    partner_obs: str = "full"
    """Who sees which TCP pose in the per-agent inputs, as in `IPPOConfig.partner_obs`. The
    checkpoint records it, and `ippo --checkpoint` refuses a run with another value."""
    device: str = "auto"
    """"auto" (CUDA if available), "cpu" or "cuda"."""

    epochs: int = 300
    """Passes over the training transitions."""
    batch_size: int = 256
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    """Linear learning-rate decay to 0 over the epochs."""
    grad_clip: float = 1.0
    """Max gradient norm, applied to the actor and the critic network separately."""
    val_fraction: float = 0.1
    """Share of the episodes (not transitions) held out for validation; 0 = train on everything.
    At least one episode is held out when this is positive and there are two or more."""
    log_every: int = 10
    """Print the losses every this many epochs (and at the first and last)."""

    actor_logstd: float = -1.6
    """The policy's log-std after BC (`exp(-1.6)` = 0.20): small exploration noise around the
    expert, instead of the ManiSkill default -0.5 (0.61) that would drown the cloned behaviour."""
    gamma: float = 0.99
    """Discount of the critic's Monte-Carlo targets: use the IPPO run's `gamma`."""
    reward_scale: float = 1.0
    """Multiplier of the demo rewards: use the IPPO run's `reward_scale`."""
    only_success: bool = True
    """Train only on successful demonstrations (a truncated failure has under-estimated returns
    and is a bad action target anyway)."""

    def __post_init__(self) -> None:
        if not self.demos:
            raise ValueError("demos is required (--demos <file from scripts/collect_demos.py>)")
        validate_partner_obs(self.partner_obs)
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
        for name in ("epochs", "batch_size", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("learning_rate and grad_clip must be > 0")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {self.val_fraction}")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")


def parse_args(argv: Sequence[str] | None = None) -> BCConfig:
    """Parse CLI flags into a `BCConfig`, e.g. `--demos runs/demos/a.pt --exp-name bc1`."""
    parser = build_dataclass_parser(
        BCConfig,
        "Behaviour-clone the two IPPO actor-critics from scripted-expert demonstrations.",
        choices={"partner_obs": PARTNER_OBS_MODES, "device": DEVICES},
    )
    values = vars(parser.parse_args(argv))
    try:
        return BCConfig(**values)
    except ValueError as err:
        parser.error(str(err))
