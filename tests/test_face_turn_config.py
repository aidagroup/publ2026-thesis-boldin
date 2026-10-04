"""FaceTurn reward config (pure dataclass logic; the env itself needs mani_skill)."""

import math

import pytest

from callosum.configs.face_turn import FaceTurnRewardConfig


def test_defaults() -> None:
    cfg = FaceTurnRewardConfig()
    assert cfg.gate_rotator_on_holder and cfg.hinge_drift_penalty
    assert cfg.angle_progress_shape == "linear"
    assert cfg.rotator_grasp_gate_floor == 0.5
    assert cfg.angle_gate_floor == 0.0  # hard gate: no angle reward without the holder


def test_max_positive_reward_is_the_sum_of_the_bounded_weights() -> None:
    cfg = FaceTurnRewardConfig()
    assert cfg.max_positive_reward == 7.0  # 1 + 1 + 3 + 1 + 1
    cfg = FaceTurnRewardConfig(weight_holder_grasp=2.5, weight_angle_progress=1.0)
    assert cfg.max_positive_reward == 1 + 1 + 1 + 1 + 2.5


def test_rejects_bad_options() -> None:
    with pytest.raises(ValueError, match="rotator_grasp_gate_floor"):
        FaceTurnRewardConfig(rotator_grasp_gate_floor=1.5)
    with pytest.raises(ValueError, match="angle_gate_floor"):
        FaceTurnRewardConfig(angle_gate_floor=-0.1)
    with pytest.raises(ValueError, match="angle_gate_floor"):
        FaceTurnRewardConfig(angle_gate_floor=1.5)
    with pytest.raises(ValueError, match="angle_progress_shape"):
        FaceTurnRewardConfig(angle_progress_shape="cubic")  # type: ignore[arg-type]


def test_success_bonus_default_covers_any_continuation_at_gamma_099() -> None:
    cfg = FaceTurnRewardConfig()
    # Normalised per-step reward <= 1, so continuing is worth < 1 / (1 - gamma) = 100.
    assert cfg.success_bonus == pytest.approx(1 / (1 - 0.99))
    assert cfg.dense_success_bonus == pytest.approx(cfg.success_bonus * cfg.max_positive_reward)
    with pytest.raises(ValueError, match="success_bonus"):
        FaceTurnRewardConfig(success_bonus=-1.0)


def test_success_bonus_is_applied_exactly_on_success_steps() -> None:
    torch = pytest.importorskip("torch")  # CI installs only the dev extra
    cfg = FaceTurnRewardConfig(success_bonus=100.0)
    dense = torch.tensor([0.5, 3.5, -0.1, 6.0])
    success = torch.tensor([False, True, False, True])
    out = cfg.add_success_bonus(dense, success)
    bonus = cfg.dense_success_bonus
    assert torch.allclose(out, dense + bonus * success.float())
    assert out[0] == dense[0] and out[2] == dense[2]  # untouched without success
    # The normalised reward (dense / divisor) carries exactly `success_bonus` on those steps.
    normalised = out / cfg.max_positive_reward
    assert torch.allclose(normalised - dense / cfg.max_positive_reward, 100.0 * success.float())
    assert math.isclose(float(normalised[1]), 0.5 + 100.0)
    # No bonus configured: identity.
    assert torch.equal(
        FaceTurnRewardConfig(success_bonus=0.0).add_success_bonus(dense, success), dense
    )


def test_angle_term_is_hard_gated_but_the_rotator_grasp_term_is_soft() -> None:
    torch = pytest.importorskip("torch")
    cfg = FaceTurnRewardConfig()
    holder_grasp = torch.tensor([0.0, 1.0])
    # Holder released: no angle reward at all (a partially turned face must not keep paying),
    # while the rotator's grasp term keeps its soft floor. Holder grasping: both in full.
    assert torch.equal(cfg.order_gate(cfg.angle_gate_floor, holder_grasp), torch.tensor([0.0, 1.0]))
    assert torch.equal(
        cfg.order_gate(cfg.rotator_grasp_gate_floor, holder_grasp), torch.tensor([0.5, 1.0])
    )
    # The floors are independent, and a positive angle floor re-opens the loophole.
    soft = FaceTurnRewardConfig(angle_gate_floor=0.25, rotator_grasp_gate_floor=0.0)
    assert torch.equal(
        soft.order_gate(soft.angle_gate_floor, holder_grasp), torch.tensor([0.25, 1])
    )
    assert torch.equal(
        soft.order_gate(soft.rotator_grasp_gate_floor, holder_grasp), torch.tensor([0.0, 1.0])
    )
    # Gating off: both terms ungated.
    off = FaceTurnRewardConfig(gate_rotator_on_holder=False)
    assert torch.equal(off.order_gate(off.angle_gate_floor, holder_grasp), torch.ones(2))


def test_gated_terms_keep_the_normalised_reward_bounded() -> None:
    cfg = FaceTurnRewardConfig()
    # Gates are in [0, 1], so every gated term stays <= its weight and the positive sum is
    # still the divisor.
    for floor in (cfg.rotator_grasp_gate_floor, cfg.angle_gate_floor):
        for grasp in (0.0, 1.0):
            assert 0.0 <= cfg.order_gate(floor, grasp) <= 1.0
    assert cfg.max_positive_reward == 7.0
