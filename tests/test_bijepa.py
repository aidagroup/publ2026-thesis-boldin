"""Unit tests for callosum.agents.bijepa (step 3.1) -- pure PyTorch, no
mani_skill, runnable locally on macOS via `make dev`. Skipped (not failed)
in CI, which installs only the `dev` extra (torch absent): mirrors
tests/test_agent_obs.py.
"""

import pytest

torch = pytest.importorskip("torch")

from callosum.agents.bijepa import (
    BiJEPA,
    Encoder,
    PartnerPredictor,
    jepa_loss,
)
from callosum.configs.bijepa import BiJEPAConfig

OBS_DIM = 20
LATENT_DIM = 8
HIDDEN_DIM = 32
CONTEXT_LEN = 3
BATCH = 4


def test_encoder_output_shape() -> None:
    enc = Encoder(OBS_DIM, LATENT_DIM, HIDDEN_DIM)
    z = enc(torch.randn(BATCH, OBS_DIM))
    assert z.shape == (BATCH, LATENT_DIM)


def test_predictor_output_shape() -> None:
    pred = PartnerPredictor(LATENT_DIM, HIDDEN_DIM, CONTEXT_LEN)
    z_hat = pred(torch.randn(BATCH, CONTEXT_LEN, LATENT_DIM))
    assert z_hat.shape == (BATCH, LATENT_DIM)


def test_bijepa_config_defaults() -> None:
    cfg = BiJEPAConfig()
    assert cfg.latent_dim == 64
    assert cfg.hidden_dim == 256
    assert cfg.context_len == 1
    assert cfg.ema_target is False
    assert cfg.aux_weight == pytest.approx(0.1)


def test_bijepa_forward_and_loss_runs() -> None:
    cfg = BiJEPAConfig(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, context_len=CONTEXT_LEN)
    model = BiJEPA(cfg, obs_dim=OBS_DIM)
    o = torch.randn(BATCH, OBS_DIM)

    z = model.encode(o)
    z_target = model.encode_target(o)
    # Build a (B, context_len, latent) window of the agent's own latents.
    hist = torch.stack([z, z, z], dim=1)
    z_hat = model.predict_partner(hist)
    loss = jepa_loss(z_target, z_hat)

    assert loss.shape == ()
    assert loss.item() >= 0.0
    loss.backward()  # one forward + backward pass over encoder + predictor


def test_jepa_loss_detaches_target() -> None:
    """stop-grad: gradient must NOT flow into the target latent."""
    z_target = torch.randn(BATCH, LATENT_DIM, requires_grad=True)
    z_pred = torch.randn(BATCH, LATENT_DIM, requires_grad=True)
    loss = jepa_loss(z_target, z_pred)
    loss.backward()
    assert z_pred.grad is not None
    assert z_target.grad is None  # detached -- no path back through the target


def test_bijepa_with_ema_target_has_frozen_target_encoder() -> None:
    cfg = BiJEPAConfig(
        latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, context_len=CONTEXT_LEN, ema_target=True
    )
    model = BiJEPA(cfg, obs_dim=OBS_DIM)
    assert model.target_encoder is not None
    for p in model.target_encoder.parameters():
        assert not p.requires_grad


def test_predictor_context_len_mismatch_raises() -> None:
    """A wrong context window width is a programmer error, not a silent reshape."""
    pred = PartnerPredictor(LATENT_DIM, HIDDEN_DIM, context_len=CONTEXT_LEN)
    bad_hist = torch.randn(BATCH, CONTEXT_LEN + 1, LATENT_DIM)
    with pytest.raises(RuntimeError):
        pred(bad_hist)
