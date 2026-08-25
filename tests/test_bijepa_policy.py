"""Unit tests for callosum.training._bijepa_policy (step 3.2) -- pure PyTorch,
no mani_skill, mac-runnable. Mirrors tests/test_agent_obs.py's
`pytest.importorskip("torch")` skip pattern (CI installs no torch)."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from callosum.agents.bijepa import PartnerPredictor, jepa_loss
from callosum.training._agent_obs import build_agent_obs
from callosum.training._bijepa_policy import (
    LatentHistory,
    bijepa_step,
    own_latent,
    partner_idx,
    partner_latent,
)

AGENT_UIDS = ("so100-0", "so100-1")
LATENT = 8
CONTEXT = 3  # window length exercised only by the LatentHistory utility
BATCH = 4


def _dummy_raw_obs(num_envs: int = BATCH) -> dict:
    agent = {
        "so100-0": {"qpos": torch.randn(num_envs, 6), "qvel": torch.randn(num_envs, 6)},
        "so100-1": {"qpos": torch.randn(num_envs, 6), "qvel": torch.randn(num_envs, 6)},
    }
    extra = {
        "cube_pose": torch.randn(num_envs, 7),
        "agent_a_tcp_pose": torch.randn(num_envs, 7),
        "agent_b_tcp_pose": torch.randn(num_envs, 7),
    }
    return {"agent": agent, "extra": extra}


def _dummy_models(base_dim: int, context_len: int = 1):
    """Phase-1 predictor: context_len=1 (current own latent)."""
    enc = nn.Linear(base_dim, LATENT)
    pred = PartnerPredictor(LATENT, hidden_dim=16, context_len=context_len)
    return enc, pred


def test_partner_idx_swaps() -> None:
    assert partner_idx(0) == 1
    assert partner_idx(1) == 0


def test_latent_history_shapes_and_keeps_grad() -> None:
    """LatentHistory is the Phase-2 window utility (kept tested though not used
    by the Phase-1 current-only path)."""
    hist = LatentHistory(BATCH, LATENT, CONTEXT, torch.device("cpu"))
    z = torch.randn(BATCH, LATENT, requires_grad=True)
    hist.push(z)
    ctx = hist.context()
    assert ctx.shape == (BATCH, CONTEXT, LATENT)
    # The newest entry (the one we just pushed) is still grad-bearing.
    assert ctx[:, -1, :].requires_grad


def test_bijepa_step_shapes_and_none_slot_is_zero() -> None:
    raw = _dummy_raw_obs()
    base_dim = build_agent_obs(raw, 0, AGENT_UIDS).shape[-1]
    enc, pred = _dummy_models(base_dim)
    pi, z_i, z_j, z_hat = bijepa_step(enc, pred, raw, 0, AGENT_UIDS, "none", LATENT)
    assert pi.shape == (BATCH, base_dim + LATENT)
    assert z_i.shape == (BATCH, LATENT)
    assert z_j.shape == (BATCH, LATENT)
    assert z_hat.shape == (BATCH, LATENT)
    assert torch.all(pi[:, -LATENT:] == 0)  # none -> dummy zero slot


def test_bijepa_step_shape_stable_across_modes() -> None:
    """The SAME policy network must serve all three modes -> identical dim."""
    raw = _dummy_raw_obs()
    base_dim = build_agent_obs(raw, 0, AGENT_UIDS).shape[-1]
    enc, pred = _dummy_models(base_dim)
    dims = {}
    for mode in ("none", "oracle", "predicted"):
        pi, _, _, _ = bijepa_step(enc, pred, raw, 0, AGENT_UIDS, mode, LATENT)
        dims[mode] = pi.shape[-1]
    assert len(set(dims.values())) == 1, dims


def test_bijepa_step_oracle_slot_equals_partner_latent() -> None:
    raw = _dummy_raw_obs()
    base_dim = build_agent_obs(raw, 0, AGENT_UIDS).shape[-1]
    enc, pred = _dummy_models(base_dim)
    pi, _, z_j, _ = bijepa_step(enc, pred, raw, 0, AGENT_UIDS, "oracle", LATENT)
    assert torch.allclose(pi[:, -LATENT:], z_j)


def test_bijepa_step_predicted_slot_equals_prediction() -> None:
    raw = _dummy_raw_obs()
    base_dim = build_agent_obs(raw, 0, AGENT_UIDS).shape[-1]
    enc, pred = _dummy_models(base_dim)
    pi, _, _, z_hat = bijepa_step(enc, pred, raw, 0, AGENT_UIDS, "predicted", LATENT)
    assert torch.allclose(pi[:, -LATENT:], z_hat)


def test_bijepa_step_rejects_bad_mode() -> None:
    raw = _dummy_raw_obs()
    base_dim = build_agent_obs(raw, 0, AGENT_UIDS).shape[-1]
    enc, pred = _dummy_models(base_dim)
    with pytest.raises(ValueError):
        bijepa_step(enc, pred, raw, 0, AGENT_UIDS, "maybe", LATENT)


def test_jepa_aux_loss_trains_encoder_and_predictor() -> None:
    """The encoder must receive a gradient through the JEPA aux loss (via the
    current z_i context); otherwise the 'none'/'predicted' arms can't learn the
    partner model at all -- this is the whole point of the shared encoder.
    """
    raw = _dummy_raw_obs()
    base_dim = build_agent_obs(raw, 0, AGENT_UIDS).shape[-1]
    enc, pred = _dummy_models(base_dim)
    z_i = own_latent(enc, raw, 0, AGENT_UIDS)
    # Phase-1 context = current z_i (length-1 window).
    z_hat = pred(z_i.unsqueeze(1))
    z_j = partner_latent(enc, raw, 0, AGENT_UIDS)
    loss = jepa_loss(z_j, z_hat)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pred.parameters())
