"""Bi-JEPA: per-agent latent encoders and partner-latent predictors (step 3.1).

Pure PyTorch (no ``mani_skill`` import), so it is unit-tested on macOS. Importing this module
requires torch, hence ``callosum.agents.__init__`` does NOT import it (CI installs no torch).

Formulation (docs/thesis/03-method-bijepa.md, "Формулировка Bi-JEPA"). Agents are indexed
``0`` ("a", the holder) and ``1`` ("b", the rotator). For agent ``i`` with partner ``j``:

* encoder ``E_i: o_i -> z_i`` (MLP);
* predictor ``P_i``: window of own latents ``z_i^{t-K+1..t}`` of shape ``(B, K, d)`` ->
  ``z_hat_j`` of shape ``(B, d)``;
* target ``z_j = sg(E_j^target(o_j))``: the partner's observation through the PARTNER's target
  encoder (``z_j`` is "the partner's own latent"), with stop-gradient;
* loss: MSE in latent space between ``z_hat_j`` and ``z_j`` (+ optional VICReg regulariser).

"Bi" = both directions are trained (:meth:`BiJEPA.forward_both`). The policy of step 3.2 uses
:meth:`BiJEPA.encode` for its own latent and ``z_partner_pred`` as the partner input.

**Collapse.** With a trainable encoder and a stop-grad target, ``z = const`` trivially gives
zero loss. The default setup therefore combines (a) an EMA target encoder (method doc: "target-
энкодер (часто EMA онлайн-энкодера, stop-gradient на цели)") and (b) a small variance/covariance
regulariser on the online latents (:func:`variance_loss`, :func:`covariance_loss`), and (c)
:func:`latent_stats` exposes std / effective rank so the trainer can log collapse. A frozen
pre-trained encoder (phase 2 of the method) would remove the risk altogether; until then this is
the "full anti-collapse machinery" the method doc asks for with a trainable encoder.
"""

import copy
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from callosum.configs.bijepa import BiJEPAConfig

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def _build_mlp(
    in_dim: int, hidden: Sequence[int], out_dim: int, activation: str, layer_norm: bool
) -> nn.Sequential:
    """Build ``Linear [-> LayerNorm] -> act`` blocks followed by a linear output layer.

    Hidden layers use orthogonal init with gain sqrt(2), the output layer gain 1, zero biases
    (same convention as the ManiSkill PPO MLPs the policy of step 3.2 is built from).
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers.append(_init_linear(nn.Linear(prev, width), 2**0.5))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(_ACTIVATIONS[activation]())
        prev = width
    layers.append(_init_linear(nn.Linear(prev, out_dim), 1.0))
    return nn.Sequential(*layers)


def _init_linear(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Encoder(nn.Module):
    """MLP encoder ``E: o_i -> z_i`` mapping ``(..., obs_dim)`` to ``(..., latent_dim)``.

    The output layer is linear (no activation, no output normalisation), so the latent scale is
    free and the variance regulariser has a meaningful scale to act on.
    """

    def __init__(self, obs_dim: int, cfg: BiJEPAConfig) -> None:
        """Create an encoder for observations of size ``obs_dim`` using ``cfg``."""
        super().__init__()
        self.obs_dim = obs_dim
        self.net = _build_mlp(
            obs_dim, cfg.encoder_hidden, cfg.latent_dim, cfg.activation, cfg.layer_norm
        )

    def forward(self, obs: Tensor) -> Tensor:
        """Encode ``obs`` of shape ``(..., obs_dim)`` into a latent of shape ``(..., latent_dim)``."""
        return self.net(obs)


class PartnerPredictor(nn.Module):
    """Predict the partner's latent from a window of the agent's own latents.

    Input ``(B, K, d)`` with ``K == cfg.history_len``. Aggregation over the window:

    * ``"flatten"`` (default): reshape to ``(B, K*d)`` and apply an MLP. Simple and, for the
      short windows used here, enough; ``K=1`` degenerates to a plain per-step MLP.
    * ``"gru"``: a GRU runs over the K steps and its last hidden state (size
      ``predictor_hidden[0]``) feeds an MLP head over ``predictor_hidden[1:]``.

    Output: ``(B, d)``.
    """

    def __init__(self, cfg: BiJEPAConfig) -> None:
        """Create a predictor for latents of size ``cfg.latent_dim``."""
        super().__init__()
        self.history_len = cfg.history_len
        self.aggregator = cfg.history_aggregator
        d = cfg.latent_dim
        self.gru: nn.GRU | None = None
        if self.aggregator == "gru":
            hidden = cfg.predictor_hidden[0]
            self.gru = nn.GRU(d, hidden, batch_first=True)
            self.head = _build_mlp(
                hidden, cfg.predictor_hidden[1:], d, cfg.activation, cfg.layer_norm
            )
        else:
            self.head = _build_mlp(
                d * cfg.history_len, cfg.predictor_hidden, d, cfg.activation, cfg.layer_norm
            )

    def forward(self, z_hist: Tensor) -> Tensor:
        """Map own latent history ``(B, K, d)`` to the predicted partner latent ``(B, d)``."""
        if z_hist.ndim != 3:
            raise ValueError(f"z_hist must be (B, K, d), got shape {tuple(z_hist.shape)}")
        if self.gru is not None:
            _, h_last = self.gru(z_hist)
            return self.head(h_last[-1])
        if z_hist.shape[1] != self.history_len:
            raise ValueError(
                f"flatten predictor expects K={self.history_len} history steps, "
                f"got K={z_hist.shape[1]}"
            )
        return self.head(z_hist.flatten(1))


def variance_loss(z: Tensor, target_std: float = 1.0, eps: float = 1e-4) -> Tensor:
    """VICReg variance term: ``mean_d relu(target_std - std_d)`` over the batch axis.

    ``z`` is ``(N, d)`` (flatten any leading dims first). Returns 0 for ``N < 2``. The hinge
    vanishes once every dimension has std >= ``target_std``, so it is a floor against collapse
    rather than a force that inflates the latent.
    """
    if z.shape[0] < 2:
        return z.new_zeros(())
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.relu(target_std - std).mean()


def covariance_loss(z: Tensor) -> Tensor:
    """VICReg covariance term: sum of squared off-diagonal covariances divided by ``d``.

    ``z`` is ``(N, d)``. Decorrelates latent dimensions (against dimensional collapse, where
    all dims carry the same signal). Returns 0 for ``N < 2``.
    """
    n, d = z.shape
    if n < 2:
        return z.new_zeros(())
    zc = z - z.mean(dim=0)
    cov = zc.T @ zc / (n - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / d


@torch.no_grad()
def latent_stats(z: Tensor, eps: float = 1e-8) -> dict[str, float]:
    """Collapse diagnostics for a batch of latents ``(..., d)`` (leading dims are flattened).

    Returns a dict with:

    * ``std_mean`` / ``std_min``: mean / minimum over latent dims of the per-dim std across
      the batch. Both going to ~0 means the latent became constant (complete collapse);
      ``std_min`` ~ 0 with a healthy mean means some dims are dead.
    * ``eff_rank``: effective rank of the centred batch, ``exp(H(p))`` with ``p`` the singular
      values normalised to sum 1 (Roy & Vetterli 2007, the RankMe measure), in ``[1, d]``.
      ~1 means all samples lie along one direction (dimensional collapse). A (numerically)
      constant latent is reported as 1.0 by convention.

    Note ``eff_rank`` is scale-invariant: a uniformly shrinking latent keeps its rank, which is
    why ``std_mean`` must be logged alongside it. Not differentiable (diagnostic only).
    """
    flat = z.detach().reshape(-1, z.shape[-1]).float()
    if flat.shape[0] < 2:
        return {"std_mean": 0.0, "std_min": 0.0, "eff_rank": 1.0}
    std = flat.std(dim=0)
    sv = torch.linalg.svdvals(flat - flat.mean(dim=0))
    total = sv.sum()
    if total <= eps:
        eff_rank = 1.0
    else:
        p = sv / total
        eff_rank = float(torch.exp(-(p * torch.log(p.clamp_min(eps))).sum()))
    return {
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "eff_rank": max(eff_rank, 1.0),
    }


class BiJEPA(nn.Module):
    """Two-agent Bi-JEPA: encoders, partner predictors, target encoders and the JEPA loss.

    Agent indices are ``0`` (a, holder) and ``1`` (b, rotator). Each agent always has its own
    predictor; the encoder is shared across agents iff ``cfg.shared_encoder``. The target
    latent of a direction is produced by the partner's target encoder: the partner's EMA copy
    if ``cfg.ema_target`` (params ``requires_grad=False``, advanced by :meth:`update_target`),
    else the partner's online encoder evaluated under ``torch.no_grad()`` (plain stop-grad).

    Optimiser note: build the optimiser from :meth:`trainable_parameters`; call
    :meth:`update_target` once after each optimiser step.
    """

    def __init__(self, cfg: BiJEPAConfig, obs_dim_a: int, obs_dim_b: int | None = None) -> None:
        """Create the module.

        Args:
            cfg: hyperparameters.
            obs_dim_a: observation size of agent 0 (the holder).
            obs_dim_b: observation size of agent 1 (the rotator); defaults to ``obs_dim_a``.
                Must equal ``obs_dim_a`` if ``cfg.shared_encoder``.
        """
        super().__init__()
        obs_dim_b = obs_dim_a if obs_dim_b is None else obs_dim_b
        if cfg.shared_encoder and obs_dim_a != obs_dim_b:
            raise ValueError(
                f"shared_encoder needs equal obs dims, got {obs_dim_a} and {obs_dim_b}; "
                "use shared_encoder=False for agents with different observations"
            )
        self.cfg = cfg
        self.obs_dims = (obs_dim_a, obs_dim_b)
        enc_a = Encoder(obs_dim_a, cfg)
        # With sharing, the same module object sits at both indices (parameters() deduplicates).
        enc_b = enc_a if cfg.shared_encoder else Encoder(obs_dim_b, cfg)
        self.encoders = nn.ModuleList([enc_a, enc_b])
        self.predictors = nn.ModuleList([PartnerPredictor(cfg), PartnerPredictor(cfg)])
        self.target_encoders: nn.ModuleList | None = None
        if cfg.ema_target:
            tgt_a = self._frozen_copy(enc_a)
            tgt_b = tgt_a if cfg.shared_encoder else self._frozen_copy(enc_b)
            self.target_encoders = nn.ModuleList([tgt_a, tgt_b])

    @staticmethod
    def _frozen_copy(encoder: Encoder) -> Encoder:
        target = copy.deepcopy(encoder)
        for p in target.parameters():
            p.requires_grad_(False)
        return target

    @staticmethod
    def _check_agent(agent: int) -> None:
        if agent not in (0, 1):
            raise ValueError(f"agent must be 0 (holder) or 1 (rotator), got {agent}")

    def encode(self, obs: Tensor, agent: int = 0) -> Tensor:
        """Online (differentiable) latent of ``agent``'s observation ``(..., obs_dim)``.

        This is the ``z_i`` used as predictor context and by the policy of step 3.2. Any leading
        dims are kept, so ``(B, K, obs_dim)`` yields ``(B, K, d)``.
        """
        self._check_agent(agent)
        return self.encoders[agent](obs)

    @torch.no_grad()
    def encode_target(self, obs: Tensor, agent: int = 0) -> Tensor:
        """Stop-grad target latent of ``agent``'s own observation (``requires_grad=False``).

        Uses the agent's EMA target encoder, or its online encoder if ``ema_target=False``.
        To get the target for agent ``i``'s predictor, call this with the PARTNER's index.
        """
        self._check_agent(agent)
        encoders = self.target_encoders if self.target_encoders is not None else self.encoders
        return encoders[agent](obs)

    def predict_partner(self, z_hist: Tensor, agent: int = 0) -> Tensor:
        """Predicted partner latent ``z_hat_j`` ``(B, d)`` from ``agent``'s latent window."""
        self._check_agent(agent)
        return self.predictors[agent](z_hist)

    def forward(
        self, obs_hist_self: Tensor, obs_hist_partner: Tensor, agent: int = 0
    ) -> dict[str, Tensor]:
        """JEPA loss for one direction: ``agent`` predicts its partner's latent.

        Args:
            obs_hist_self: ``(B, K, obs_dim_self)`` own observations, ``K == cfg.history_len``
                for the flatten aggregator. A 2-D ``(B, obs_dim)`` input is treated as K=1.
            obs_hist_partner: partner observations; only the LAST step is used as the target
                observation, either ``(B, T, obs_dim_partner)`` or ``(B, obs_dim_partner)``.
                For a horizon ``k > 1`` (method doc: future latent ``z_j^{t+k}``) pass the
                partner observation at ``t+k`` in the last slot; this module does no
                time-shifting itself.
            agent: index of the predicting agent (the partner is ``1 - agent``).

        Returns a dict with:
            ``loss`` (scalar total = pred + var_weight*var + cov_weight*cov),
            ``loss_pred`` (scalar latent MSE, the number to compare across runs),
            ``loss_var`` / ``loss_cov`` (scalar regulariser terms, 0 when their weight is 0),
            ``z_self`` ``(B, K, d)`` online latents of the own history (with grad),
            ``z_partner_target`` ``(B, d)`` stop-grad target (``requires_grad=False``),
            ``z_partner_pred`` ``(B, d)`` prediction ``z_hat_j``.
        """
        self._check_agent(agent)
        if obs_hist_self.ndim == 2:
            obs_hist_self = obs_hist_self.unsqueeze(1)
        obs_partner = obs_hist_partner[:, -1] if obs_hist_partner.ndim == 3 else obs_hist_partner

        z_self = self.encode(obs_hist_self, agent)
        z_pred = self.predict_partner(z_self, agent)
        z_target = self.encode_target(obs_partner, 1 - agent)

        loss_pred = (z_pred - z_target).pow(2).mean()
        z_flat = z_self.reshape(-1, z_self.shape[-1])
        loss_var = loss_cov = z_flat.new_zeros(())
        if self.cfg.var_weight > 0.0:
            loss_var = variance_loss(z_flat, self.cfg.var_target)
        if self.cfg.cov_weight > 0.0:
            loss_cov = covariance_loss(z_flat)
        loss = loss_pred + self.cfg.var_weight * loss_var + self.cfg.cov_weight * loss_cov
        return {
            "loss": loss,
            "loss_pred": loss_pred,
            "loss_var": loss_var,
            "loss_cov": loss_cov,
            "z_self": z_self,
            "z_partner_target": z_target,
            "z_partner_pred": z_pred,
        }

    def forward_both(self, obs_hist_a: Tensor, obs_hist_b: Tensor) -> dict[str, Tensor | dict]:
        """Symmetric "Bi" loss: both agents predict each other's latent.

        Args:
            obs_hist_a: ``(B, K, obs_dim_a)`` observation history of agent 0.
            obs_hist_b: ``(B, K, obs_dim_b)`` observation history of agent 1.

        Returns a dict with ``loss`` (mean of the two directions), ``loss_pred`` (mean of the
        two prediction losses) and the per-direction output dicts of :meth:`forward` under
        ``"a"`` (agent 0 predicts agent 1) and ``"b"`` (agent 1 predicts agent 0).
        """
        out_a = self.forward(obs_hist_a, obs_hist_b, agent=0)
        out_b = self.forward(obs_hist_b, obs_hist_a, agent=1)
        return {
            "loss": 0.5 * (out_a["loss"] + out_b["loss"]),
            "loss_pred": 0.5 * (out_a["loss_pred"] + out_b["loss_pred"]),
            "a": out_a,
            "b": out_b,
        }

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA step ``target <- m * target + (1 - m) * online`` (no-op without ``ema_target``).

        Call once per optimiser step, after ``optimizer.step()``. Parameters follow the EMA
        formula; buffers (none in the current MLPs) are copied. A shared encoder is updated once.
        """
        if self.target_encoders is None:
            return
        m = self.cfg.ema_momentum
        seen: set[int] = set()
        for online, target in zip(self.encoders, self.target_encoders, strict=True):
            if id(target) in seen:
                continue
            seen.add(id(target))
            for p_t, p_o in zip(target.parameters(), online.parameters(), strict=True):
                p_t.mul_(m).add_(p_o.detach(), alpha=1.0 - m)
            for b_t, b_o in zip(target.buffers(), online.buffers(), strict=True):
                b_t.copy_(b_o)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters that receive gradients (encoders + predictors, no EMA targets)."""
        return [p for p in self.parameters() if p.requires_grad]
