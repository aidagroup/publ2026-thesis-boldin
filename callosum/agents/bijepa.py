"""Bi-JEPA encoder and partner-predictor modules (step 3.1).

Pure PyTorch (no mani_skill): Bi-JEPA is the scientific core of this project
(docs/thesis/03-method-bijepa.md) and the method staging validates it first
in *state* space, where it can be unit-tested on macOS before touching the
paid CUDA session. This module is therefore imported only on demand
(`from callosum.agents.bijepa import ...`) and is NOT re-exported by
callosum.agents's __init__, which must stay torch-free for CI
(docs/implementation-plan.md section 0) -- the exact same pattern as
callosum.training._ppo_core / _agent_obs.

Formulation (method doc §Формулировка):
  encoder  E: o_i -> z_i        (obs -> latent). Shared across both arms in
                                the symmetric Phase-1 setup (common weights
                                per the method design table §Решения).
  partner  P: z_i^{<=t} -> z_j   (from a window of the agent's own past
                                latents, regress the partner's latent).
  target   z_j = E(o_j), detached. Stop-grad on the target latent is the
           minimal Phase-1 anti-collapse form; an EMA target encoder is
           supported but off by default (method doc §A↔C: a frozen/EMA target
           avoids collapse; trainable encoder + stop-grad is the Phase-1
           stand-in).

"Bi" = both arms share E and P and symmetrically model each other. The
asymmetric holder/rotator roles (FaceTurn) are a Phase-2 concern
(role-conditioning / separate heads), noted in the design table §C.
"""

import copy

import torch
from torch import nn

from callosum.configs.bijepa import BiJEPAConfig


def layer_init(layer: nn.Linear, std: float = 2**0.5, bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal init + zero bias.

    Mirrors callosum.training._ppo_core.layer_init so the Bi-JEPA MLPs keep
    the same activation scale as the PPO actor/critic they will be fused into
    in step 3.2.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Encoder(nn.Module):
    """Shared latent encoder E: observation -> latent embedding.

    A small MLP (obs_dim -> hidden -> latent_dim), shared by both arms in
    the symmetric Phase-1 setup. Used both to produce an agent's own latent
    (predictor context) and -- via the model's `encode_target` -- the
    partner's target latent for the JEPA loss.
    """

    def __init__(self, obs_dim: int, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, latent_dim)),
        )

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.net(o)


class PartnerPredictor(nn.Module):
    """Partner-latent predictor P: own latent history -> predicted partner latent.

    Reads the agent's own recent latent context ``z_i^{<=t}`` of shape
    (B, context_len, latent_dim) and regresses toward the partner's latent.
    Context is flattened and run through an MLP; a recurrence (GRU) is the
    documented upgrade path for longer horizons (method doc §Открытые вопросы).
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256, context_len: int = 4):
        super().__init__()
        self.context_len = context_len
        self.net = nn.Sequential(
            layer_init(nn.Linear(latent_dim * context_len, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, latent_dim)),
        )

    def forward(self, self_latent_hist: torch.Tensor) -> torch.Tensor:
        """self_latent_hist: (B, context_len, latent_dim) -> (B, latent_dim)."""
        return self.net(self_latent_hist.flatten(1))


class BiJEPA(nn.Module):
    """Owner of the shared encoder + partner predictor (and optional EMA target).

    Phase-1 default (``ema_target=False``): a single trainable encoder is used
    for both the predictor's context encoding and the target latent; the
    target is detached inside ``jepa_loss``, giving the I-JEPA-style
    asymmetric stop-grad formulation out of the box (method doc §Напоминание).

    Phase-2 vision (method doc §A, §A↔C): set ``ema_target=True`` to maintain
    an EMA copy of the encoder as the target, decoupling target drift from
    online training.
    """

    def __init__(self, cfg: BiJEPAConfig, obs_dim: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(obs_dim, cfg.latent_dim, cfg.hidden_dim)
        self.predictor = PartnerPredictor(cfg.latent_dim, cfg.hidden_dim, cfg.context_len)
        # Lazily-created EMA target encoder (None in the Phase-1 default).
        self.target_encoder: nn.Module | None = None
        if cfg.ema_target:
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)

    def encode(self, o: torch.Tensor) -> torch.Tensor:
        """Online encoding of an observation to its latent (``o_i -> z_i``)."""
        return self.encoder(o)

    def encode_target(self, o: torch.Tensor) -> torch.Tensor:
        """Encode an observation for the *target* latent ``z_j``.

        Uses the EMA target encoder if present, else the online encoder; the
        target is detached in `jepa_loss`, so gradients never return here
        through this path in the default (stop-grad) configuration.
        """
        if self.target_encoder is not None:
            return self.target_encoder(o)
        return self.encoder(o)

    def predict_partner(self, self_latent_hist: torch.Tensor) -> torch.Tensor:
        """Predict partner latent from a window of own latents."""
        return self.predictor(self_latent_hist)

    @torch.no_grad()
    def ema_update(self) -> None:
        """Moves the EMA target toward the online encoder (no-op unless ema_target)."""
        if self.target_encoder is None:
            return
        for tgt, src in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tgt.mul_(self.cfg.ema_decay).add_(src, alpha=1 - self.cfg.ema_decay)


def jepa_loss(z_target: torch.Tensor, z_pred: torch.Tensor) -> torch.Tensor:
    """Latent-space MSE with stop-grad on the target.

    ``z_target`` is detached before the squared error, so no gradient flows
    back through it -- the asymmetry that prevents representation collapse
    (method doc §Напоминание: что такое JEPA). ``z_pred`` still receives
    gradient through the predictor (and, when used as predictor context, the
    online encoder).
    """
    return (z_pred - z_target.detach()).pow(2).mean()
