"""FaceTurn reward config (pure dataclass logic; the env itself needs mani_skill)."""

import pytest

from callosum.configs.face_turn import FaceTurnRewardConfig


def test_defaults() -> None:
    cfg = FaceTurnRewardConfig()
    assert cfg.gate_rotator_on_holder and cfg.hinge_drift_penalty
    assert cfg.angle_progress_shape == "linear"
    assert cfg.rotator_gate_floor == 0.5


def test_max_positive_reward_is_the_sum_of_the_bounded_weights() -> None:
    cfg = FaceTurnRewardConfig()
    assert cfg.max_positive_reward == 7.0  # 1 + 1 + 3 + 1 + 1
    cfg = FaceTurnRewardConfig(weight_holder_grasp=2.5, weight_angle_progress=1.0)
    assert cfg.max_positive_reward == 1 + 1 + 1 + 1 + 2.5


def test_rejects_bad_options() -> None:
    with pytest.raises(ValueError, match="rotator_gate_floor"):
        FaceTurnRewardConfig(rotator_gate_floor=1.5)
    with pytest.raises(ValueError, match="angle_progress_shape"):
        FaceTurnRewardConfig(angle_progress_shape="cubic")  # type: ignore[arg-type]
