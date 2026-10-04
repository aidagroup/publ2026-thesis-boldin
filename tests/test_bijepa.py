"""Tests for callosum.agents.bijepa (step 3.1): pure PyTorch, runs on macOS (``make dev``).

Skipped when torch is absent (CI installs only the ``dev`` extra).
"""

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from callosum.agents.bijepa import (
    BiJEPA,
    Encoder,
    PartnerPredictor,
    covariance_loss,
    latent_stats,
    variance_loss,
)
from callosum.configs.bijepa import BiJEPAConfig

B, K, D_A, D_B, LAT = 8, 3, 12, 20, 6


def _cfg(**kw) -> BiJEPAConfig:
    base = {"latent_dim": LAT, "encoder_hidden": (16,), "predictor_hidden": (16,), "history_len": K}
    return BiJEPAConfig(**{**base, **kw})


def _batch(d_a: int = D_A, d_b: int = D_B, k: int = K):
    return torch.randn(B, k, d_a), torch.randn(B, k, d_b)


def test_config_defaults_valid_and_validation() -> None:
    cfg = BiJEPAConfig()
    assert cfg.ema_target and not cfg.shared_encoder
    assert 0.0 <= cfg.ema_momentum < 1.0 and cfg.history_len >= 1
    for bad in (
        {"latent_dim": 0},
        {"history_len": 0},
        {"activation": "nope"},
        {"history_aggregator": "lstm"},
        {"ema_momentum": 1.0},
        {"var_weight": -1.0},
    ):
        with pytest.raises(ValueError):
            BiJEPAConfig(**bad)


@pytest.mark.parametrize("agg", ["flatten", "gru"])
@pytest.mark.parametrize("ema", [True, False])
def test_forward_shapes_different_obs_dims(agg: str, ema: bool) -> None:
    model = BiJEPA(_cfg(history_aggregator=agg, ema_target=ema), D_A, D_B)
    obs_a, obs_b = _batch()
    out = model(obs_a, obs_b, agent=0)
    assert out["z_self"].shape == (B, K, LAT)
    assert out["z_partner_target"].shape == (B, LAT)
    assert out["z_partner_pred"].shape == (B, LAT)
    assert out["loss"].shape == () and out["loss_pred"].shape == ()
    out_b = model(obs_b, obs_a, agent=1)
    assert out_b["z_self"].shape == (B, K, LAT)
    assert model.encode(obs_a[:, 0], 0).shape == (B, LAT)
    # 2-D partner obs (single target step) and 2-D own obs (K=1) are accepted too.
    m1 = BiJEPA(_cfg(history_len=1, history_aggregator=agg), D_A, D_B)
    assert m1(obs_a[:, 0], obs_b[:, 0])["z_partner_pred"].shape == (B, LAT)


def test_forward_both_shapes() -> None:
    model = BiJEPA(_cfg(), D_A, D_B)
    obs_a, obs_b = _batch()
    out = model.forward_both(obs_a, obs_b)
    assert out["a"]["z_partner_pred"].shape == (B, LAT)
    assert out["b"]["z_partner_target"].shape == (B, LAT)
    expected = 0.5 * (out["a"]["loss"] + out["b"]["loss"])
    assert torch.allclose(out["loss"], expected)


def test_shared_encoder_requires_equal_dims_and_shares_weights() -> None:
    with pytest.raises(ValueError):
        BiJEPA(_cfg(shared_encoder=True), D_A, D_B)
    model = BiJEPA(_cfg(shared_encoder=True), D_A, D_A)
    assert model.encoders[0] is model.encoders[1]
    assert model.predictors[0] is not model.predictors[1]
    obs_a, obs_b = _batch(D_A, D_A)
    model.forward_both(obs_a, obs_b)["loss"].backward()
    # Separate encoders are distinct modules.
    sep = BiJEPA(_cfg(shared_encoder=False), D_A, D_A)
    assert sep.encoders[0] is not sep.encoders[1]


def test_predictor_history_length_checked() -> None:
    pred = PartnerPredictor(_cfg())
    with pytest.raises(ValueError):
        pred(torch.randn(B, K + 1, LAT))
    gru = PartnerPredictor(_cfg(history_aggregator="gru"))
    assert gru(torch.randn(B, K + 2, LAT)).shape == (B, LAT)  # GRU accepts any K


def test_encoder_standalone() -> None:
    enc = Encoder(D_A, _cfg(layer_norm=False, activation="relu"))
    assert enc(torch.randn(B, K, D_A)).shape == (B, K, LAT)


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("ema", [True, False])
def test_backward_finite_grads_on_online_modules(shared: bool, ema: bool) -> None:
    model = BiJEPA(_cfg(shared_encoder=shared, ema_target=ema), D_A, D_A if shared else D_B)
    obs_a, obs_b = _batch(D_A, D_A if shared else D_B)
    model.forward_both(obs_a, obs_b)["loss"].backward()
    online = list(model.encoders.parameters()) + list(model.predictors.parameters())
    assert online
    for p in online:
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert any(p.grad.abs().sum() > 0 for p in model.encoders.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.predictors.parameters())


def test_ema_target_is_stop_grad() -> None:
    model = BiJEPA(_cfg(ema_target=True), D_A, D_B)
    assert model.target_encoders is not None
    assert all(not p.requires_grad for p in model.target_encoders.parameters())
    obs_a, obs_b = _batch()
    obs_b.requires_grad_(True)
    out = model(obs_a, obs_b, agent=0)
    assert not out["z_partner_target"].requires_grad
    out["loss"].backward()
    assert all(p.grad is None for p in model.target_encoders.parameters())
    assert obs_b.grad is None  # no path back through the partner branch
    # Partner's online encoder is untouched by this direction (separate encoders).
    assert all(p.grad is None for p in model.encoders[1].parameters())
    # trainable_parameters excludes the EMA copies.
    ids = {id(p) for p in model.trainable_parameters()}
    assert not ids & {id(p) for p in model.target_encoders.parameters()}


def test_shared_stop_grad_target_has_no_grad() -> None:
    model = BiJEPA(_cfg(shared_encoder=True, ema_target=False), D_A, D_A)
    assert model.target_encoders is None
    obs_a, obs_b = _batch(D_A, D_A)
    obs_b.requires_grad_(True)
    out = model(obs_a, obs_b, agent=0)
    assert not out["z_partner_target"].requires_grad
    assert out["z_self"].requires_grad
    out["loss"].backward()
    assert obs_b.grad is None


def test_no_ema_separate_encoders_partner_encoder_gets_no_grad_from_target() -> None:
    model = BiJEPA(_cfg(shared_encoder=False, ema_target=False), D_A, D_B)
    obs_a, obs_b = _batch()
    model(obs_a, obs_b, agent=0)["loss"].backward()
    assert all(p.grad is None for p in model.encoders[1].parameters())


def test_target_comes_from_partner_target_encoder() -> None:
    model = BiJEPA(_cfg(ema_target=True), D_A, D_B)
    obs_a, obs_b = _batch()
    out = model(obs_a, obs_b, agent=0)
    expected = model.target_encoders[1](obs_b[:, -1])
    assert torch.allclose(out["z_partner_target"], expected)
    out_b = model(obs_b, obs_a, agent=1)
    assert torch.allclose(out_b["z_partner_target"], model.target_encoders[0](obs_a[:, -1]))


def test_ema_update_formula() -> None:
    m = 0.9
    model = BiJEPA(_cfg(ema_target=True, ema_momentum=m), D_A, D_B)
    # Perturb the online encoders so that target != online.
    with torch.no_grad():
        for p in model.encoders.parameters():
            p.add_(torch.randn_like(p))
    before = [t.clone() for t in model.target_encoders.parameters()]
    online = [o.clone() for o in model.encoders.parameters()]
    model.update_target()
    for t, t0, o in zip(model.target_encoders.parameters(), before, online, strict=True):
        assert torch.allclose(t, m * t0 + (1 - m) * o, atol=1e-6)
        assert not t.requires_grad
    # Repeated updates converge to the online weights.
    for _ in range(300):
        model.update_target()
    for t, o in zip(model.target_encoders.parameters(), online, strict=True):
        assert torch.allclose(t, o, atol=1e-4)


def test_ema_update_shared_encoder_once_and_noop_without_ema() -> None:
    m = 0.5
    model = BiJEPA(_cfg(shared_encoder=True, ema_momentum=m), D_A, D_A)
    with torch.no_grad():
        for p in model.encoders[0].parameters():
            p.add_(1.0)
    t0 = [t.clone() for t in model.target_encoders[0].parameters()]
    o = [p.clone() for p in model.encoders[0].parameters()]
    model.update_target()
    for t, a, b in zip(model.target_encoders[0].parameters(), t0, o, strict=True):
        assert torch.allclose(t, m * a + (1 - m) * b, atol=1e-6)  # applied once, not twice
    BiJEPA(_cfg(ema_target=False), D_A, D_B).update_target()  # no-op, must not raise


def test_vicreg_losses_penalise_collapse() -> None:
    torch.manual_seed(0)
    spread = torch.randn(256, LAT)
    collapsed = torch.zeros(256, LAT) + 0.3
    assert variance_loss(collapsed) > 0.9
    assert variance_loss(spread) < 0.05
    correlated = spread[:, :1].repeat(1, LAT)
    assert covariance_loss(correlated) > covariance_loss(spread) * 10
    assert variance_loss(spread[:1]) == 0 and covariance_loss(spread[:1]) == 0


def test_regulariser_gives_collapsed_encoder_larger_loss() -> None:
    obs_a, obs_b = _batch(D_A, D_A)
    losses = {}
    for name, weights in (("on", (1.0, 0.04)), ("off", (0.0, 0.0))):
        torch.manual_seed(0)
        model = BiJEPA(
            _cfg(
                shared_encoder=True, ema_target=False, var_weight=weights[0], cov_weight=weights[1]
            ),
            D_A,
            D_A,
        )
        with torch.no_grad():  # collapse: the encoder output is the constant 0
            last = model.encoders[0].net[-1]
            last.weight.zero_()
        out = model(obs_a, obs_b)
        assert out["loss_pred"] == 0  # the trivial solution has zero prediction loss
        losses[name] = out["loss"].item()
    assert losses["off"] == 0.0
    assert losses["on"] > 0.9  # the regulariser makes the collapsed solution costly


def test_latent_stats_detects_constant_vs_random() -> None:
    torch.manual_seed(0)
    const = latent_stats(torch.full((64, LAT), 0.7))
    assert const["std_mean"] < 1e-6 and const["std_min"] < 1e-6
    assert const["eff_rank"] == pytest.approx(1.0, abs=1e-3)
    rand = latent_stats(torch.randn(64, LAT))
    assert rand["std_mean"] > 0.8 and rand["std_min"] > 0.5
    assert rand["eff_rank"] > LAT * 0.7
    one_dir = latent_stats(torch.randn(64, 1) * torch.ones(1, LAT))
    assert one_dir["eff_rank"] < 1.5
    # Leading dims are flattened.
    assert latent_stats(torch.randn(4, 16, LAT))["eff_rank"] > 1.0


def test_deterministic_with_fixed_seed() -> None:
    outs = []
    for _ in range(2):
        torch.manual_seed(123)
        model = BiJEPA(_cfg(), D_A, D_B)
        obs_a, obs_b = torch.randn(B, K, D_A), torch.randn(B, K, D_B)
        outs.append(model.forward_both(obs_a, obs_b)["loss"].item())
    assert outs[0] == outs[1]


def test_config_is_dataclass() -> None:
    assert dataclasses.is_dataclass(BiJEPAConfig)
