"""Bi-JEPA hyperparameters (step 3.1).

Deliberately torch-free (stdlib dataclass only) so it is importable in CI,
which installs just the `dev` extra -- mirrors callosum.configs.face_turn /
ippo. `obs_dim` is intentionally absent: it is env-derived (the SO-100
proprioception + shared extra fields that build_agent_obs assembles) and is
passed to `callosum.agents.bijepa.BiJEPA(obs_dim=...)` at construction, the
same way obs_dim/action_dim are passed to `_ppo_core.Agent` rather than
stored in IPPOConfig (see callosum.training.ippo.main).
"""

from dataclasses import dataclass


@dataclass
class BiJEPAConfig:
    """Knobs for the Bi-JEPA encoder / partner predictor
    (callosum.agents.bijepa).

    See docs/thesis/03-method-bijepa.md §Формулировка and §Решения for where
    each default comes from; these are the Phase-1 state-space values, tuned
    to be cheap to train on the SO-100 proprioceptive slice.
    """

    latent_dim: int = 64
    hidden_dim: int = 256
    # Phase-1 default: a length-1 context (the current own latent z_i), i.e. a
    # same-step partner prediction with NO rolling buffer -- so there is no
    # cross-episode/cross-iteration latent state to reset blind (the failure
    # mode the runbook warns about for GPU-less development). Multi-step
    # history (method doc §Формулировка z_i^{<=t}, §Открытые вопросы) is a
    # Phase-2 enhancement using LatentHistory from _bijepa_policy.
    context_len: int = 1
    # JEPA auxiliary-loss weight merged into the per-agent PPO update (step 3.2).
    aux_weight: float = 0.1

    # Phase-1 default (state space): single trainable encoder, target detached
    # inside jepa_loss (I-JEPA style). Set True to maintain an EMA copy as the
    # target encoder instead -- the Phase-2 vision path (method doc §A, §A↔C)
    # where a frozen/EMA target avoids collapse against a moving online encoder.
    ema_target: bool = False
    ema_decay: float = 0.98
