"""DART-style action noise of the scripted expert (numpy only)."""

import numpy as np

from callosum.experts._noise import perturb_arm_action


def test_noise_touches_only_arm_entries_and_stays_in_range() -> None:
    rng = np.random.default_rng(0)
    action = np.tile([0.5, -1.0, 1.0, 0.0, 0.9, -1.0], (1000, 1))
    out = perturb_arm_action(action, 0.2, rng)
    assert np.array_equal(out[:, 5], action[:, 5])  # gripper untouched
    assert np.all(np.abs(out) <= 1.0)
    assert out[:, :5].std(axis=0)[3] > 0.15  # the unsaturated joint carries the full noise
    assert not np.array_equal(out[:, :5], action[:, :5])
    assert np.array_equal(action[0], [0.5, -1.0, 1.0, 0.0, 0.9, -1.0])  # input not modified


def test_zero_std_is_identity_and_seeded_noise_is_reproducible() -> None:
    action = np.random.default_rng(1).uniform(-1, 1, (4, 6))
    assert np.array_equal(perturb_arm_action(action, 0.0, np.random.default_rng(0)), action)
    a = perturb_arm_action(action, 0.1, np.random.default_rng(3))
    b = perturb_arm_action(action, 0.1, np.random.default_rng(3))
    assert np.array_equal(a, b)
