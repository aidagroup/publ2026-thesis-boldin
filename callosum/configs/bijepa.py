"""Hyperparameters of the Bi-JEPA module (step 3.1).

Stdlib-only (no torch) so the config stays importable in CI, which installs only the ``dev``
extra. Per-agent observation sizes are deliberately *not* here: they are derived from the
environment and passed to :class:`callosum.agents.bijepa.BiJEPA` as constructor arguments.
"""

from dataclasses import dataclass

ACTIVATIONS = ("tanh", "relu", "gelu", "silu")
HISTORY_AGGREGATORS = ("flatten", "gru")


@dataclass
class BiJEPAConfig:
    """Knobs of the Bi-JEPA encoder, partner predictor, target and anti-collapse terms.

    Default combination (see docs/thesis/03-method-bijepa.md, "Реализация (3.1)"): **separate
    per-agent encoders + EMA target encoders + a small VICReg-style variance/covariance
    regulariser on the online latents**. A trainable encoder with only a stop-gradient target is
    the minimal setup of the plan, but the constant latent ``z = const`` drives its loss to zero,
    and the archived SO-100 attempt measured exactly that drift (latent std shrinking 4.4x in 156
    updates while the loss looked healthy). Hence EMA and the regulariser are on by default.
    """

    # --- Encoder E: o_i -> z_i -----------------------------------------------------------
    latent_dim: int = 32
    encoder_hidden: tuple[int, ...] = (256, 256)
    activation: str = "tanh"  # one of ACTIVATIONS; tanh matches the ManiSkill PPO MLPs
    layer_norm: bool = True  # LayerNorm after every hidden Linear (encoder and predictor)

    # --- Partner predictor P: (z_i^{t-K+1..t}) -> z_hat_j ----------------------------------
    predictor_hidden: tuple[int, ...] = (256, 256)
    # Window length K of the own-latent history fed to the predictor. K=1 means "current latent
    # only", i.e. the trainer needs no rolling buffer (and no per-episode reset of it).
    history_len: int = 1
    # "flatten": (B, K, d) -> (B, K*d) -> MLP. "gru": a GRU over the K steps, last hidden state
    # -> MLP (hidden size = predictor_hidden[0]); only worthwhile for K > 1.
    history_aggregator: str = "flatten"

    # --- Encoder sharing and target ---------------------------------------------------------
    # True: one encoder for both agents (needs equal obs dims; each agent still has its own
    # predictor). False: one encoder per agent (holder / rotator observations differ); each
    # direction then encodes the partner with the PARTNER's target encoder.
    shared_encoder: bool = False
    # True: the target latent z_j comes from an EMA copy of the partner's encoder (stop-grad).
    # False: from the partner's online encoder under no_grad (plain stop-grad; collapse-prone).
    ema_target: bool = True
    ema_momentum: float = 0.99  # target <- m * target + (1 - m) * online

    # --- VICReg-style anti-collapse regulariser on the online latents -------------------------
    # loss = mse + var_weight * variance_hinge + cov_weight * covariance_penalty.
    # The hinge is zero once every latent dim has std >= var_target, so it only acts as a floor
    # against collapse and does not fight a healthy representation. 1 : 0.04 is the VICReg
    # 25 : 1 ratio rescaled to a unit variance term. Set both weights to 0 to switch it off.
    var_weight: float = 1.0
    cov_weight: float = 0.04
    var_target: float = 1.0
    # TODO(review): tune var_weight/cov_weight/var_target on the server once a real PPO run
    # provides the latent statistics (std_mean, eff_rank) to watch.

    def __post_init__(self) -> None:
        """Validate field values; raise ``ValueError`` on an invalid configuration."""
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {self.latent_dim}")
        if self.history_len < 1:
            raise ValueError(f"history_len must be >= 1, got {self.history_len}")
        if any(h < 1 for h in self.encoder_hidden) or any(h < 1 for h in self.predictor_hidden):
            raise ValueError("hidden layer sizes must be >= 1")
        if self.history_aggregator == "gru" and not self.predictor_hidden:
            raise ValueError("history_aggregator='gru' needs at least one predictor_hidden size")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {self.activation!r}")
        if self.history_aggregator not in HISTORY_AGGREGATORS:
            raise ValueError(
                f"history_aggregator must be one of {HISTORY_AGGREGATORS}, "
                f"got {self.history_aggregator!r}"
            )
        if not 0.0 <= self.ema_momentum < 1.0:
            raise ValueError(f"ema_momentum must be in [0, 1), got {self.ema_momentum}")
        if self.var_weight < 0.0 or self.cov_weight < 0.0:
            raise ValueError("var_weight and cov_weight must be >= 0")
        if self.var_target <= 0.0:
            raise ValueError(f"var_target must be > 0, got {self.var_target}")
