"""Delta-mode tracking logic of the scripted expert: action, integral bias, reach monitoring.

Everything here is numpy only (`callosum.experts._tracking`, `callosum.configs.face_turn_expert`),
so it runs on macOS and in CI without the simulator.
"""

import numpy as np
import pytest

from callosum.configs.face_turn_expert import ExpertControlConfig
from callosum.experts._tracking import (
    ReachMonitor,
    describe_progress,
    effective_tol,
    noise_jitter,
    tracking_action,
    update_bias,
)

LIMIT = 0.05  # ARM_DELTA_LIMIT
STEP_FRACTION = 0.39  # share of the commanded offset the arm covers in one control step


def test_unsaturated_action_is_gain_times_error_in_limit_units() -> None:
    error = np.array([[0.01, -0.02, 0.0, 0.005, 0.0]])
    action, saturated = tracking_action(error, np.zeros_like(error), 1.5, LIMIT)
    assert np.allclose(action, 1.5 * error / LIMIT)
    assert not saturated[0]


def test_saturated_action_keeps_direction_and_peaks_at_one() -> None:
    error = np.array([[0.4, -0.1, 0.0, 0.2, 0.0], [0.01, 0.0, 0.0, 0.0, 0.0]])
    action, saturated = tracking_action(error, np.zeros_like(error), 1.0, LIMIT)
    assert saturated.tolist() == [True, False]
    assert np.isclose(np.abs(action[0]).max(), 1.0)
    assert np.allclose(action[0], error[0] / 0.4)  # same direction as the error
    assert np.allclose(action[1], error[1] / LIMIT)  # the small one is untouched


def test_bias_shifts_the_action_and_counts_in_saturation() -> None:
    error = np.zeros((1, 5))
    bias = np.array([[0.01, 0.0, -0.02, 0.0, 0.0]])
    action, saturated = tracking_action(error, bias, 1.0, LIMIT)
    assert np.allclose(action, bias / LIMIT)
    assert not saturated[0]


def test_update_bias_integrates_clips_and_freezes_when_saturated() -> None:
    bias = np.zeros((2, 5))
    error = np.full((2, 5), 0.004)
    out = update_bias(bias, error, ki=0.25, bias_max=0.006, saturated=np.array([False, True]))
    assert np.allclose(out[0], 0.001)
    assert np.allclose(out[1], 0.0)  # anti-windup: frozen while saturated
    big = update_bias(out, np.full((2, 5), 1.0), 0.25, 0.006, np.array([False, False]))
    assert np.allclose(big, 0.006)  # clipped
    neg = update_bias(out, np.full((2, 5), -1.0), 0.25, 0.006, np.array([False, False]))
    assert np.allclose(neg, -0.006)
    same = update_bias(out, error, ki=0.0, bias_max=0.006, saturated=np.array([False, False]))
    assert same is out  # ki = 0 disables the term


def _track(gain: float, ki: float, bias_max: float, sag: float, steps: int = 80) -> float:
    """Final error of one joint driven by the delta loop against a constant disturbance.

    The plant is the controller's first-order lag: the arm covers `STEP_FRACTION` of the commanded
    offset per step, minus `sag * STEP_FRACTION` (a disturbance that needs a standing offset of
    `sag` rad to be held off).
    """
    target, q = 0.5, 0.0
    bias = np.zeros((1, 1))
    for _ in range(steps):
        error = np.array([[target - q]])
        action, saturated = tracking_action(error, bias, gain, LIMIT)
        bias = update_bias(bias, error, ki, bias_max, saturated)
        q += STEP_FRACTION * (action[0, 0] * LIMIT - sag)
    return abs(target - q)


def test_pure_proportional_loop_keeps_a_steady_state_error_the_integral_removes() -> None:
    sag = 0.012  # standing offset needed against the disturbance (rad)
    p_only = _track(gain=1.5, ki=0.0, bias_max=0.0, sag=sag)
    assert p_only == pytest.approx(sag / 1.5, rel=0.02)  # 0.008 rad: above the 0.006 tolerance
    assert p_only > 0.006
    # The bias bound (0.006 here) carries half of the needed offset; the error shrinks to ~0.004.
    bounded = _track(gain=1.5, ki=0.25, bias_max=0.006, sag=sag)
    assert bounded < 0.006
    # A bound above the needed offset removes the error completely.
    assert _track(gain=1.5, ki=0.25, bias_max=0.02, sag=sag) < 1e-4


def test_higher_gain_converges_faster_without_overshoot() -> None:
    def steps_to(tol: float, gain: float) -> int:
        target, q = 0.5, 0.0
        for k in range(1, 200):
            action, _ = tracking_action(np.array([[target - q]]), np.zeros((1, 1)), gain, LIMIT)
            q += STEP_FRACTION * action[0, 0] * LIMIT
            assert q <= target + 1e-12  # monotone
            if target - q < tol:
                return k
        raise AssertionError("did not converge")

    assert steps_to(0.006, 1.5) < steps_to(0.006, 1.0)


def test_monitor_reports_reached_per_env() -> None:
    monitor = ReachMonitor(tol=0.01, stall_window=3, stall_progress=1e-3)
    reached, stalled = monitor.update(np.array([0.5, 0.005]))
    assert reached.tolist() == [False, True]
    assert not stalled.any()


def test_monitor_flags_an_env_without_progress_as_stalled() -> None:
    monitor = ReachMonitor(tol=0.01, stall_window=2, stall_progress=1e-3)
    moving, blocked = 0.5, 0.2
    for k in range(1, 10):
        moving -= 0.05  # keeps improving
        _, stalled = monitor.update(np.array([moving, blocked]))
        assert stalled.tolist() == [False, k >= 4]  # two full windows of samples are needed


def test_monitor_stall_clears_when_progress_resumes() -> None:
    monitor = ReachMonitor(tol=0.001, stall_window=1, stall_progress=1e-3)
    for err in (0.2, 0.2, 0.2):
        _, stalled = monitor.update(np.array([err]))
    assert stalled[0]
    _, stalled = monitor.update(np.array([0.1]))
    assert not stalled[0]


def test_monitor_reset_restarts_the_progress_history_of_masked_envs() -> None:
    monitor = ReachMonitor(tol=np.array([0.001, 0.001]), stall_window=1, stall_progress=1e-3)
    for _ in range(3):
        _, stalled = monitor.update(np.array([0.2, 0.2]))
    assert stalled.tolist() == [True, True]
    monitor.reset(np.array([True, False]))  # env 0 got a new target
    _, stalled = monitor.update(np.array([0.3, 0.2]))
    assert stalled.tolist() == [False, True]


def test_monitor_stall_survives_noise_jitter_which_defeats_a_best_so_far_criterion() -> None:
    jitter = noise_jitter(0.1, LIMIT)
    rng = np.random.default_rng(0)
    blocked = 0.2 + jitter * rng.standard_normal((400, 8))  # a blocked env's jittery error
    monitor = ReachMonitor(tol=0.01, stall_window=3, stall_progress=5e-4)
    verdicts = np.stack(
        [monitor.update(row, progress=5e-4 + jitter)[1] for row in blocked]  # noise-aware threshold
    )
    first = verdicts.argmax(axis=0)
    assert verdicts.any(axis=0).all() and (first < 40).all()  # every env is flagged, soon
    # The old criterion (best error so far must improve by stall_progress) keeps seeing "progress".
    best, since, old_stall = np.full(8, np.inf), np.zeros(8), []
    for row in blocked[:60]:
        improved = row < best - 5e-4
        best, since = np.where(improved, row, best), np.where(improved, 0, since + 1)
        old_stall.append(since >= 4)
    assert not np.stack(old_stall)[10:].all()  # dips keep resetting it


def test_monitor_does_not_call_a_noisy_converging_env_stalled() -> None:
    jitter = noise_jitter(0.1, LIMIT)
    rng = np.random.default_rng(1)
    monitor = ReachMonitor(tol=0.0, stall_window=3, stall_progress=5e-4)
    error = 0.5
    for _ in range(14):  # travelling at about 0.02 rad per step
        error -= 0.02
        _, stalled = monitor.update(
            np.abs(error + jitter * rng.standard_normal(16)), progress=5e-4 + jitter
        )
        assert not stalled.any()


def test_effective_tolerance_and_jitter() -> None:
    assert noise_jitter(0.0, LIMIT) == 0.0
    assert noise_jitter(0.1, LIMIT) == pytest.approx(0.1 * LIMIT * 0.39)
    assert effective_tol(0.006, 0.0, 3.0) == 0.006
    assert effective_tol(0.006, 0.002, 3.0) == pytest.approx(0.012)


def test_describe_progress_counts_stages() -> None:
    n = 4
    lines = describe_progress(
        face_angle_max=np.array([0.0, 0.5, 1.55, 1.57]),
        face_angle_end=np.array([0.0, 0.4, 1.5, 1.57]),
        holder_grasp_ever=np.array([False, True, True, True]),
        rotator_grasp_ever=np.array([False, False, True, True]),
        both_grasp_ever=np.array([False, False, True, True]),
        body_rot_max=np.array([0.0, 0.0, 0.2, 0.01]),
        body_pos_max=np.zeros(n),
        target_angle=np.pi / 2,
        angle_tol=0.05,
        body_rot_tol=0.1,
        body_pos_tol=0.01,
    )
    text = "\n".join(lines)
    assert "ever held the body (holder grasp):     3/4" in text
    assert "ever held the face (rotator grasp):    2/4" in text
    assert "within tolerance of 90 deg:  2/4" in text
    assert "body drift above tolerance:        1/4" in text  # the turned env with 0.2 rad


def test_config_defaults_and_validation() -> None:
    cfg = ExpertControlConfig(control="delta")
    assert cfg.gain >= 1 and cfg.ki > 0 and cfg.bias_max > 0
    assert cfg.final_tol < cfg.approach_tol < cfg.waypoint_tol
    assert ExpertControlConfig().control == "pos"  # the probe's default is unchanged
    for kwargs in (
        {"ki": -0.1},
        {"bias_max": -1.0},
        {"joint_limit_margin": -0.01},
        {"approach_tol": 0.0},
        {"stall_window": 0},
        {"noise_tol_scale": -1.0},
        {"stall_progress": -1e-3},
    ):
        with pytest.raises(ValueError):
            ExpertControlConfig(**kwargs)
