"""DART-style action noise for the scripted expert (numpy only, unit-testable everywhere)."""

import numpy as np


def perturb_arm_action(
    action: np.ndarray, std: float, rng: np.random.Generator, num_arm_joints: int = 5
) -> np.ndarray:
    """Copy of the 6-D `action` `(n, 6)` with Gaussian noise on the arm entries only.

    The first `num_arm_joints` columns (normalised joint deltas) get `N(0, std)` noise and are
    clipped to [-1, 1]; the gripper column is never perturbed (it is an absolute +-1 open/close
    command: noise would only shorten the stroke, and a sign flip would be a failed grasp rather
    than a recoverable offset). `std == 0` returns an unchanged copy.
    """
    out = np.array(action, dtype=float, copy=True)
    if std > 0:
        arm = out[:, :num_arm_joints]
        out[:, :num_arm_joints] = np.clip(arm + rng.normal(0.0, std, arm.shape), -1.0, 1.0)
    return out
