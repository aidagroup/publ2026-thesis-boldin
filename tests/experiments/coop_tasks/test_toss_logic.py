"""Pure logic of `TossCatch-v0` (no mujoco): layout, geometry, rules, randomisation, config.

The environment itself needs MuJoCo (the `sandbox` extra, absent from CI), so everything that
can be decided without the simulator lives in `callosum.experiments.coop_tasks.logic` and is
checked here: the toss-catch observation layout, the paddle/ball geometry helpers the scripted
oracle uses, the bounce model, the pass/failure/success/potential rules, the extended episode
sampler and the task config validation.

The environment-side checks (spaces, the scripted oracle reaching `n_passes`, the absent-agent
baseline, seed determinism and the vec-env contract) live in
`tests/experiments/coop_tasks/test_coop_envs.py` and are skipped without mujoco.
"""

import math

import numpy as np
import pytest

from callosum.experiments.coop_tasks.config import (
    TASKS,
    TossConfig,
    TossRandomization,
    coop_action_size,
    coop_config,
    coop_obs_size,
)
from callosum.experiments.coop_tasks.logic import (
    TOSS_SPEC,
    EpisodeRandomization,
    TossEpisodeRandomization,
    ball_position_after,
    ball_velocity_after,
    bounce_velocity,
    pad_face_normal,
    pad_face_tangent,
    paddle_catch_centre,
    paddle_tilt,
    sample_toss_randomization,
    task_diagnostic,
    toss_ball_on_side,
    toss_floor_failure,
    toss_pass_count,
    toss_potential,
    toss_success,
)

# --- observation layout -------------------------------------------------------------------

TOSS_OBS_NONE = 7 + 16 + 2 + 1
"""Own block (3 actions + 3 params + start delay) + shared + goals + time."""


def test_toss_spec_layout() -> None:
    assert TOSS_SPEC.name == "toss"
    assert TOSS_SPEC.action_dim == 3
    assert TOSS_SPEC.goal_fields == ("passes", "target_passes")
    assert len(TOSS_SPEC.shared_fields) == 16
    assert TOSS_SPEC.shared_fields[:4] == ("ball_x", "ball_z", "ball_vx", "ball_vz")
    assert "left_pad_tilt" in TOSS_SPEC.shared_fields
    assert "right_pad_tilt_rate" in TOSS_SPEC.shared_fields


def test_toss_observation_widths() -> None:
    assert coop_obs_size("toss", "none") == TOSS_OBS_NONE == 26
    assert coop_obs_size("toss", "oracle") == TOSS_OBS_NONE + 2 * 3 + 4 == 36
    assert coop_action_size("toss") == 3
    assert "toss" in TASKS
    assert isinstance(coop_config("toss"), TossConfig)


# --- paddle / ball geometry ---------------------------------------------------------------


def test_pad_face_normal_and_tangent() -> None:
    level = pad_face_normal(0.0)
    assert level.tolist() == pytest.approx([0.0, 0.0, 1.0])
    tilted = pad_face_normal(math.radians(30.0))
    assert tilted[0] == pytest.approx(0.5)
    assert tilted[2] == pytest.approx(math.cos(math.radians(30.0)))
    for angle in (0.0, 0.3, -0.6):
        normal = pad_face_normal(angle)
        tangent = pad_face_tangent(angle)
        assert float(normal @ tangent) == pytest.approx(0.0)
        assert float(np.linalg.norm(normal)) == pytest.approx(1.0)
        assert float(np.linalg.norm(tangent)) == pytest.approx(1.0)


def test_paddle_tilt_mirrors_the_two_sides() -> None:
    tilt = math.radians(40.0)
    assert paddle_tilt("left", tilt) == pytest.approx(tilt)
    assert paddle_tilt("right", tilt) == pytest.approx(-tilt)
    with pytest.raises(ValueError, match="side must be one of"):
        paddle_tilt("middle", tilt)


def test_ballistic_prediction_matches_free_flight() -> None:
    position = np.array([0.2, 0.0, 1.1])
    velocity = np.array([2.0, 0.0, 1.5])
    gravity = 9.81
    for seconds in (0.1, 0.3, 0.6):
        predicted = ball_position_after(position, velocity, seconds, gravity)
        assert predicted[0] == pytest.approx(position[0] + velocity[0] * seconds)
        assert predicted[2] == pytest.approx(
            position[2] + velocity[2] * seconds - 0.5 * gravity * seconds**2
        )
        predicted_velocity = ball_velocity_after(velocity, seconds, gravity)
        assert predicted_velocity[2] == pytest.approx(velocity[2] - gravity * seconds)


def test_paddle_catch_centre_puts_the_face_centre_on_the_ball() -> None:
    position = np.array([0.0, 0.0, 1.2])
    velocity = np.array([2.0, 0.0, -0.5])
    seconds = 0.25
    tilt = math.radians(35.0)
    offset = 0.09
    centre = paddle_catch_centre(position, velocity, seconds, 9.81, tilt, offset)
    contact = ball_position_after(position, velocity, seconds, 9.81)
    # The paddle centre is `contact - offset * normal`, so the ball touches the face centre.
    assert np.allclose(centre, contact - offset * pad_face_normal(tilt))
    assert float(np.linalg.norm(centre - contact)) == pytest.approx(offset)


# --- bounce model -------------------------------------------------------------------------


def test_bounce_velocity_head_on_and_tangential() -> None:
    normal = np.array([0.0, 0.0, 1.0])
    incoming = np.array([1.0, 0.0, -3.0])
    elastic = bounce_velocity(incoming, normal, 1.0)
    assert np.allclose(elastic, [1.0, 0.0, 3.0])
    inelastic = bounce_velocity(incoming, normal, 0.0)
    assert np.allclose(inelastic, [1.0, 0.0, 0.0])  # tangential velocity preserved
    damped = bounce_velocity(incoming, normal, 0.5)
    assert damped[2] == pytest.approx(1.5)
    # Tangential component survives any restitution, normal component is scaled.
    tilt = math.radians(40.0)
    oblique = bounce_velocity(np.array([2.0, 0.0, -2.0]), pad_face_normal(-tilt), 0.8)
    tangent = pad_face_tangent(-tilt)
    incoming_tangent = float(np.array([2.0, 0.0, -2.0]) @ tangent)
    assert float(oblique @ tangent) == pytest.approx(incoming_tangent)


def test_bounce_velocity_rejects_bad_restitution() -> None:
    with pytest.raises(ValueError, match="restitution"):
        bounce_velocity([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 1.5)
    with pytest.raises(ValueError, match="restitution"):
        bounce_velocity([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], -0.1)


# --- rules --------------------------------------------------------------------------------


def test_toss_failure_bounds() -> None:
    kwargs = {"floor_z": 0.07, "court_x": 1.35, "court_y": 0.6}
    assert not toss_floor_failure(0.0, 0.0, 0.5, **kwargs)
    assert toss_floor_failure(0.0, 0.0, 0.06, **kwargs)
    assert toss_floor_failure(1.4, 0.0, 0.8, **kwargs)
    assert toss_floor_failure(0.0, 0.7, 0.8, **kwargs)
    assert not toss_floor_failure(1.3, 0.55, 0.8, **kwargs)


_SUCCESS_STATE = {
    "ball_x": 0.4,
    "ball_y": 0.0,
    "floor_z": 0.07,
    "court_x": 1.35,
    "court_y": 0.6,
    "floor_margin": 0.02,
}
"""The toss-catch success-state arguments shared by the rule tests (ball over the right half)."""


def test_toss_success_conjunction() -> None:
    assert not toss_success(0, 3, ball_z=0.8, **_SUCCESS_STATE)
    assert not toss_success(2, 3, ball_z=0.8, **_SUCCESS_STATE)
    assert toss_success(3, 3, ball_z=0.8, **_SUCCESS_STATE)
    assert toss_success(4, 3, ball_z=0.8, **_SUCCESS_STATE)
    with pytest.raises(ValueError, match="target_passes"):
        toss_success(0, 0, ball_z=0.8, **_SUCCESS_STATE)
    with pytest.raises(ValueError, match="floor_margin"):
        toss_success(3, 3, ball_z=0.8, **{**_SUCCESS_STATE, "floor_margin": -0.1})


def test_toss_success_requires_the_ball_above_the_floor() -> None:
    """A success frame with the ball below the floor (or out of court) is not a success."""
    # Enough passes but the ball has already been dropped below the floor.
    assert not toss_success(3, 3, ball_z=0.05, **_SUCCESS_STATE)
    # Above the floor but inside the success clearance: still not a success.
    assert not toss_success(3, 3, ball_z=0.07, **_SUCCESS_STATE)
    assert not toss_success(3, 3, ball_z=0.089, **_SUCCESS_STATE)
    # Strictly above floor_z + floor_margin: the first valid success frame.
    assert toss_success(3, 3, ball_z=0.091, **_SUCCESS_STATE)
    # A court failure cancels the success even when the ball is high enough.
    assert not toss_success(3, 3, ball_z=0.8, **{**_SUCCESS_STATE, "ball_x": 1.4})
    assert not toss_success(3, 3, ball_z=0.8, **{**_SUCCESS_STATE, "ball_x": -1.4})
    assert not toss_success(3, 3, ball_z=0.8, **{**_SUCCESS_STATE, "ball_y": 0.7})


def test_toss_ball_on_side() -> None:
    assert toss_ball_on_side(-0.3, "left")
    assert not toss_ball_on_side(0.3, "left")
    assert toss_ball_on_side(0.3, "right")
    assert not toss_ball_on_side(-0.3, "right")
    # Exactly on the centre line counts as neither paddle's half.
    assert not toss_ball_on_side(0.0, "left")
    assert not toss_ball_on_side(0.0, "right")
    with pytest.raises(ValueError, match="side must be one of"):
        toss_ball_on_side(0.3, "middle")


def _pass_count(
    passes: int = 0,
    last_side: str | None = None,
    side: str = "right",
    *,
    ball_x: float = 0.5,
    step: int = 10,
    last_step: int = -1,
    interval: int = 3,
) -> int:
    """One `toss_pass_count` call with the test defaults (a right-side contact on its half)."""
    return toss_pass_count(
        passes,
        last_side,
        side,
        ball_x=ball_x,
        step=step,
        last_pass_step=last_step,
        min_interval_steps=interval,
    )


def test_toss_pass_count_requires_alternation() -> None:
    assert _pass_count() == 1
    assert _pass_count(0, None, "right") == 1
    # A chattering contact on the same paddle does not count twice.
    assert _pass_count(1, "right", "right") == 1
    assert _pass_count(1, "right", "left", ball_x=-0.5) == 2
    assert _pass_count(2, "left", "right", ball_x=0.5) == 3
    with pytest.raises(ValueError, match="contact_side"):
        _pass_count(side="middle")


def test_toss_pass_count_requires_the_ball_over_the_centre() -> None:
    """A contact with the ball still on the other half is not a pass (it never crossed)."""
    # A right-paddle contact while the ball is still on the left half: no pass.
    assert _pass_count(0, None, "right", ball_x=-0.4) == 0
    assert _pass_count(1, "left", "right", ball_x=-0.4) == 1
    # A left-paddle contact while the ball is on the right half: no pass either.
    assert _pass_count(0, None, "left", ball_x=0.4) == 0
    # Right at the centre line the ball is not on either paddle's half.
    assert _pass_count(0, None, "right", ball_x=0.0) == 0


def test_toss_pass_count_enforces_the_min_interval() -> None:
    # First pass at control step 10.
    assert _pass_count(0, None, "right", step=10) == 1
    for step in (10, 11, 12):
        assert _pass_count(1, "right", "left", ball_x=-0.5, step=step, last_step=10) == 1, (
            f"a pass {step - 10} control step(s) after the previous one must be rejected"
        )
    # Exactly `min_interval_steps` control steps later the next pass counts.
    assert _pass_count(1, "right", "left", ball_x=-0.5, step=13, last_step=10) == 2
    # A fresh episode (`last_pass_step=-1`) has no interval gate.
    assert _pass_count(0, None, "left", ball_x=-0.5, step=0, last_step=-1) == 1


def test_toss_rapid_alternating_contacts_cannot_race_the_pass_counter() -> None:
    """The exploit sentinel: a ball chattering across the centre every step counts slowly."""
    passes, last_side, last_step = 0, None, -1
    interval = 3
    for step in range(12):
        side = "left" if step % 2 == 0 else "right"
        ball_x = -0.05 if side == "left" else 0.05
        new = toss_pass_count(
            passes,
            last_side,
            side,
            ball_x=ball_x,
            step=step,
            last_pass_step=last_step,
            min_interval_steps=interval,
        )
        if new != passes:
            assert last_step < 0 or step - last_step >= interval
            passes, last_side, last_step = new, side, step
    # Contact every control step can only produce a pass every `interval` steps.
    assert passes == 1 + 11 // interval


def test_toss_pass_count_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="min_interval_steps"):
        _pass_count(interval=0)
    with pytest.raises(ValueError, match="step must be >= 0"):
        _pass_count(step=-1)
    with pytest.raises(ValueError, match="last_pass_step"):
        _pass_count(last_step=-2)


def test_toss_potential_is_bounded_and_monotone() -> None:
    kwargs = {
        "target_passes": 3,
        "height_scale": 1.0,
        "pass_coef": 0.5,
        "height_coef": 0.3,
    }
    assert toss_potential(passes=0, ball_z=0.0, **kwargs) == pytest.approx(0.0)
    one_pass = toss_potential(passes=1, ball_z=0.0, **kwargs)
    two_passes = toss_potential(passes=2, ball_z=0.0, **kwargs)
    assert 0.0 < one_pass < two_passes < 0.5
    assert toss_potential(passes=3, ball_z=0.5, **kwargs) == pytest.approx(0.5 + 0.15)
    worst = toss_potential(passes=99, ball_z=99.0, **kwargs)
    assert worst == pytest.approx(0.8)  # bounded by pass_coef + height_coef
    with pytest.raises(ValueError, match="target_passes"):
        toss_potential(passes=0, ball_z=0.0, **{**kwargs, "target_passes": 0})
    with pytest.raises(ValueError, match="height_scale"):
        toss_potential(passes=0, ball_z=0.0, **{**kwargs, "height_scale": 0.0})
    with pytest.raises(ValueError, match="coefficients"):
        toss_potential(passes=0, ball_z=0.0, **{**kwargs, "pass_coef": -1.0})


def test_task_diagnostic() -> None:
    assert task_diagnostic("cartpole", {"angle_deg": 12.5}) == 12.5
    assert task_diagnostic("pendulum", {"amplitude_deg": 40.0}) == 40.0
    assert task_diagnostic("toss", {"passes": 2.0}) == 2.0
    with pytest.raises(ValueError, match="missing"):
        task_diagnostic("toss", {"angle_deg": 1.0})
    with pytest.raises(ValueError, match="task must be one of"):
        task_diagnostic("bogus", {})


# --- randomisation ------------------------------------------------------------------------


def _ranges(**overrides: object) -> TossRandomization:
    """The task's randomisation ranges with a few fields overridden."""
    defaults: dict[str, object] = {}
    defaults.update(overrides)
    return TossRandomization(**defaults)


def test_toss_sampler_is_deterministic_and_in_range() -> None:
    ranges = _ranges()
    first = sample_toss_randomization(ranges, control_dt=0.04, max_control_steps=400, seed=7)
    second = sample_toss_randomization(ranges, control_dt=0.04, max_control_steps=400, seed=7)
    assert first == second
    assert isinstance(first, TossEpisodeRandomization)
    assert isinstance(first, EpisodeRandomization)
    assert first.seed == 7
    for agent in first.agents:
        assert ranges.latency_range[0] <= agent.latency_s <= ranges.latency_range[1]
        assert ranges.gain_range[0] <= agent.gain <= ranges.gain_range[1]
        assert ranges.start_delay_range[0] <= agent.start_delay_s <= ranges.start_delay_range[1]
    assert ranges.launch_vx_range[0] <= first.launch_velocity[0] <= ranges.launch_vx_range[1]
    assert ranges.launch_vz_range[0] <= first.launch_velocity[1] <= ranges.launch_vz_range[1]
    assert (
        ranges.initial_value_range[0] <= first.launch_position[0] <= ranges.initial_value_range[1]
    )
    assert (
        ranges.launch_height_range[0] <= first.launch_position[1] <= ranges.launch_height_range[1]
    )
    assert 0 <= first.release_delay_steps <= round(ranges.release_delay_range[1] / 0.04)
    assert (
        ranges.contact_dampratio_range[0]
        <= first.contact_dampratio
        <= ranges.contact_dampratio_range[1]
    )
    assert (
        ranges.contact_friction_range[0]
        <= first.contact_friction
        <= ranges.contact_friction_range[1]
    )


def test_toss_sampler_draws_fresh_randomness_per_seed() -> None:
    ranges = _ranges()
    records = [
        sample_toss_randomization(ranges, control_dt=0.04, max_control_steps=400, seed=seed)
        for seed in range(12)
    ]
    assert len({record.launch_velocity for record in records}) > 1
    assert len({record.contact_dampratio for record in records}) > 1
    assert len({record.release_delay_steps for record in records}) > 1


def test_toss_sampler_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="control_dt"):
        sample_toss_randomization(_ranges(), control_dt=0.0, max_control_steps=400, seed=0)
    with pytest.raises(ValueError, match="max_control_steps"):
        sample_toss_randomization(_ranges(), control_dt=0.04, max_control_steps=0, seed=0)


def test_toss_randomization_validation() -> None:
    with pytest.raises(ValueError):
        TossRandomization(launch_vx_range=(2.0, 1.0))
    with pytest.raises(ValueError):
        TossRandomization(launch_vx_range=(0.0, 1.0))  # must be positive
    with pytest.raises(ValueError):
        TossRandomization(launch_height_range=(0.0, 1.0))
    with pytest.raises(ValueError):
        TossRandomization(release_delay_range=(-0.1, 0.2))
    with pytest.raises(ValueError):
        TossRandomization(contact_dampratio_range=(0.0, 1.2))
    with pytest.raises(ValueError):
        TossRandomization(contact_friction_range=(-0.1, 0.1))
    with pytest.raises(ValueError, match="below 1"):
        TossRandomization(contact_dampratio_range=(0.5, 1.0))
    with pytest.raises(ValueError, match="at most 1 s"):
        TossRandomization(release_delay_range=(0.0, 1.5))


# --- config -------------------------------------------------------------------------------


def test_toss_config_defaults_match_the_task_spec() -> None:
    cfg = TossConfig()
    assert cfg.task == "toss"
    assert math.isclose(cfg.dt * cfg.control_every, 0.04)
    assert 300 <= cfg.max_control_steps <= 400
    assert cfg.action_size == 3
    assert cfg.n_passes == 3
    assert cfg.min_pass_interval_steps == 3
    assert 0 < cfg.success_floor_margin < cfg.pad_min_z - cfg.floor_z
    assert 0 < cfg.pad_tilt < cfg.pad_tilt_limit < math.pi / 2
    assert cfg.pad_min_z < cfg.pad_start_z < cfg.pad_max_z
    assert cfg.floor_z < cfg.pad_min_z < cfg.pad_max_z
    assert cfg.pad_inner_x < cfg.pad_outer_x < cfg.court_x
    assert 0 < cfg.pad_anticipation_x < cfg.court_x
    assert cfg.paddle_x_range("left") == (-cfg.pad_outer_x, -cfg.pad_inner_x)
    assert cfg.paddle_x_range("right") == (cfg.pad_inner_x, cfg.pad_outer_x)
    assert cfg.agent_authority(0) == cfg.agent_authority(1) == cfg.pad_speed
    assert cfg.randomization.latency_range == (0.04, 0.16)
    assert cfg.randomization.contact_dampratio_range[0] < 0.5
    assert cfg.randomization.launch_vx_range[0] > 0
    assert cfg.control_dt == pytest.approx(0.04)
    with pytest.raises(ValueError, match="agent_index"):
        cfg.agent_authority(2)
    with pytest.raises(ValueError, match="side must be one of"):
        cfg.paddle_x_range("middle")
