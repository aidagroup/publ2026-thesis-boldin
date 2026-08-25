"""Bi-JEPA x policy integration helpers (step 3.2), pure PyTorch.

No mani_skill/env import (only torch + callosum.training._agent_obs), so the
latent-assembly + JEPA-loss wiring is unit-tested on macOS before touching the
paid CUDA session -- the same split that lets callosum.training._ppo_core be
verified before ippo.py runs on the server (docs/implementation-plan.md §0).
ippo.py (server-only, imports mani_skill) owns the BiJEPA module instance and
the per-agent LatentHistory buffers; it just threads them through the helpers
below.

Policy-input design (method doc §Формулировка): the policy acts on o_i plus a
partner-latent channel. So each agent's base input is its OWN decentralized
observation (the partner's raw TCP pose is dropped via include_partner=False,
see build_agent_obs) and a fixed-width partner-latent slot is appended per
partner_input (oracle/predicted/none). The encoder never appears in the base;
it only produces latents: z_i (predictor context / history, grad-bearing so
the encoder trains through the JEPA loss) and z_j (the CTDE partner target,
detached inside jepa_loss).

Training-signal budget (method doc §Решения C / §Открытые вопросы): the JEPA
aux loss is applied in ALL partner_input modes, so the encoder+predictor still
learn in the decentralized 'none' arm -- only the policy's partner channel
changes, exactly the variable the ablation isolates.
"""

import torch
from torch import nn

from callosum.training._agent_obs import (
    PARTNER_INPUT_MODES,
    build_agent_obs,
    build_policy_input,
)


def partner_idx(agent_idx: int) -> int:
    """The other arm's index (0<->1 swap)."""
    return 1 - agent_idx


class LatentHistory:
    """Rolling window of one agent's own latents (z_i^{<=t}), fed to the
    partner predictor as context.

    The newest entry (current z_i, grad-bearing) is retained; older entries
    are detached on eviction, so gradients don't traverse an unbounded rollout
    history -- only the current step's encoder is trained through the
    predictor, exactly as I-JEPA intends.
    """

    def __init__(self, num_envs: int, latent_dim: int, context_len: int, device: torch.device):
        self.context_len = context_len
        self.latent_dim = latent_dim
        self.buf = torch.zeros(num_envs, context_len, latent_dim, device=device)

    def push(self, z_i: torch.Tensor) -> None:
        """Shift the window left (dropping the oldest entry), detach the
        retained older entries, and append the current z_i (grad retained)
        at the end. Detaching the *kept* history (not the evicted one) bounds
        the backward window to the current step's encoder, exactly as I-JEPA
        intends."""
        self.buf = torch.cat([self.buf[:, 1:, :].detach(), z_i.unsqueeze(1)], dim=1).to(
            self.buf.device, dtype=self.buf.dtype
        )

    def context(self) -> torch.Tensor:
        """(num_envs, context_len, latent_dim) -- feed to PartnerPredictor."""
        return self.buf


def own_latent(
    encoder: nn.Module, raw_obs: dict, agent_idx: int, agent_uids: tuple[str, str]
) -> torch.Tensor:
    """z_i = E(o_i): this agent's decentralized obs -> own latent (grad on)."""
    own_base = build_agent_obs(raw_obs, agent_idx, agent_uids, include_partner=False)
    return encoder(own_base)


def partner_latent(
    encoder: nn.Module, raw_obs: dict, agent_idx: int, agent_uids: tuple[str, str]
) -> torch.Tensor:
    """z_j = E(o_j): partner's TRUE latent (CTDE target). The returned tensor
    carries grad through `encoder`; downstream callers are responsible for
    detaching it -- bijepa_step detaches the *slot* it appends to the policy
    input (so the policy loss never trains the encoder via z_j), and
    jepa_loss detaches it as the loss target. Pass an EMA `target_encoder`
    (via BiJEPA.encode_target) to source z_j from the frozen EMA copy instead.
    """
    pidx = partner_idx(agent_idx)
    partner_base = build_agent_obs(raw_obs, pidx, agent_uids, include_partner=False)
    return encoder(partner_base)


def bijepa_step(
    encoder: nn.Module,
    predictor: nn.Module,
    raw_obs: dict,
    agent_idx: int,
    agent_uids: tuple[str, str],
    partner_input: str,
    latent_dim: int,
    context: torch.Tensor | None = None,
    target_encoder: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One agent's full Bi-JEPA step: assemble the policy input and produce the
    latent pieces the trainer needs for the JEPA aux loss.

    Computes the own latent ``z_i = E(o_i)``, the partner true latent
    ``z_j = E_target(o_j)`` (CTDE target; ``E(o_j)`` via the online encoder
    when no ``target_encoder`` is given, or the EMA target encoder when one is),
    and the partner prediction
    ``z_hat_j = P(context)``; then assembles the policy input as
    ``[own_base, slot]`` per `partner_input` (slot detached -- see below).

    ``context`` is the predictor's input of shape (num_envs, context_len,
    latent_dim). If None (Phase-1 default), the **current** own latent
    ``z_i`` is used as a length-1 context -- i.e. a same-step prediction with
    no rolling buffer, so there is no cross-episode/cross-iteration state to
    reset (the documented failure mode of blind history-buffer wiring; see
    docs/implementation-plan.md step 3.2 caution). A multi-step history window
    can be passed here once LatentHistory is wired into the trainer (Phase 2).

    Returns (policy_input, z_i, z_j, z_hat_j). ``z_i`` is returned so the
    caller can optionally record it; ``z_j``/``z_hat_j`` feed the JEPA aux loss
    via `callosum.agents.bijepa.jepa_loss`.
    """
    own_base = build_agent_obs(raw_obs, agent_idx, agent_uids, include_partner=False)
    if partner_input not in PARTNER_INPUT_MODES:
        raise ValueError(
            f"partner_input must be one of {PARTNER_INPUT_MODES}, got {partner_input!r}"
        )
    z_i = encoder(own_base)  # grad on (predictor context; current in Phase 1)
    ctx = z_i.unsqueeze(1) if context is None else context
    z_hat_j = predictor(ctx)  # grad through predictor + current z_i -> encoder
    z_j = partner_latent(
        target_encoder if target_encoder is not None else encoder,
        raw_obs,
        agent_idx,
        agent_uids,
    )  # partner true latent (E_target if EMA, else online E; jepa_loss detaches)

    if partner_input == "none":
        slot = None
    elif partner_input == "oracle":
        slot = z_j
    else:  # "predicted"
        slot = z_hat_j
    # Detach the partner-latent slot from the policy input: the policy consumes
    # the partner signal as a *fixed* input (I-JEPA-style decoupling). The
    # encoder/predictor are trained solely by the JEPA aux loss
    # (jepa_loss(z_j, z_hat_j)) via a dedicated shared optimizer -- see the
    # trainer wiring in callosum.training.ippo. Without the detach, the policy
    # loss would also backprop into the shared encoder/predictor and collide
    # with that optimizer's step (method doc §Решения C).
    if slot is not None:
        slot = slot.detach()

    policy_input = build_policy_input(
        raw_obs,
        agent_idx,
        agent_uids,
        partner_latent=slot,
        partner_input=partner_input,
        partner_latent_dim=latent_dim,
        base=own_base,
    )
    return policy_input, z_i, z_j, z_hat_j
