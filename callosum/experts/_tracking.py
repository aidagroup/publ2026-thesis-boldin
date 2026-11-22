"""Pure-numpy tracking logic of the scripted expert's `"delta"` mode (unit-testable everywhere).

The `pd_joint_delta_pos` controller sets its target to the *current* joint position plus the
action (`use_target=False`). Under any steady disturbance (the remaining sag of a joint, contact,
a joint limit) a pure proportional action therefore leaves a steady-state error, and a closed
loop that waits for every joint of every env to get within a tolerance only ends by its timeout.
This module holds the pieces that avoid that, none of which needs the simulator:

* `tracking_action`: the normalised action towards a joint target, with a proportional gain, an
  additive per-joint bias (the integral term) and saturation that keeps the direction.
* `update_bias`: the integral term with anti-windup (frozen while the action is saturated,
  clipped to `bias_max`).
* `ReachMonitor`: per-env "reached or stalled" bookkeeping (stall judged on a windowed mean, so
  noise jitter does not defeat it), so an env that is blocked (contact, joint limit) moves on
  instead of waiting for its timeout. `noise_jitter` / `effective_tol`: how much the reach
  tolerances widen when the executed action carries DART noise.
* `describe_progress`: what the demo collector prints to make a failed run diagnosable.

The phase state machines that use these live in `callosum.experts._phases`.
"""

import numpy as np


def tracking_action(
    error: np.ndarray, bias: np.ndarray, gain: float, limit: float
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised delta action `(n, J)` towards a joint target, and which envs are saturated.

    The commanded offset (rad) is `gain * error + bias`; in units of `limit` (the controller's
    per-step bound), divided by its largest entry per env when that exceeds 1, so the direction
    is kept (a multi-joint move stays a straight line in joint space).

    Args:
        error: `(n, J)` target minus current joint position (rad).
        bias: `(n, J)` additive offset (rad) compensating a steady-state error.
        gain: multiplier of the error before saturation.
        limit: the controller's bound on one step's offset (`ARM_DELTA_LIMIT`).

    Returns:
        `(action (n, J), saturated (n,) bool)`; `saturated` is True where the unscaled action
        exceeded 1 in some joint.
    """
    action = (gain * np.asarray(error, dtype=float) + bias) / limit
    peak = np.abs(action).max(axis=1, keepdims=True)
    return action / np.maximum(peak, 1.0), peak[:, 0] > 1.0


def update_bias(
    bias: np.ndarray, error: np.ndarray, ki: float, bias_max: float, saturated: np.ndarray
) -> np.ndarray:
    """One integral step: `bias + ki * error`, clipped to `+-bias_max`, frozen where saturated.

    Freezing while the action is saturated (the arm is travelling at full speed, the error is
    large because of the distance, not because of a disturbance) is the anti-windup. `ki == 0`
    returns `bias` unchanged.
    """
    if ki == 0:
        return bias
    stepped = np.clip(bias + ki * np.asarray(error, dtype=float), -bias_max, bias_max)
    return np.where(np.asarray(saturated)[:, None], bias, stepped)


STEP_FRACTION = 0.39
"""Share of a commanded joint offset the arm covers in one control step (first-order PD lag of the
stock controller): the executed-action noise reaches the joints scaled by this factor."""


def noise_jitter(action_noise: float, limit: float) -> float:
    """Rad of joint jitter that DART action noise (`action_noise` in normalised units) causes.

    The noise is added to the normalised action, one `limit` rad per unit, and the joint follows a
    `STEP_FRACTION` of it per step; 0.1 noise with the 0.05 rad limit gives about 0.002 rad.
    """
    return float(action_noise) * limit * STEP_FRACTION


def effective_tol(tol: float, jitter: float, scale: float) -> float:
    """A reach tolerance widened by `scale * jitter`: what the noisy executed action can meet."""
    return tol + scale * jitter


class ReachMonitor:
    """Per-env "reached or stalled" bookkeeping of a target-tracking loop.

    Feed it the per-env worst joint error (to the *clean* target) after every step. An env is
    `reached` when that error is below `tol`. It is `stalled` when the mean error of the last
    `stall_window` steps is not lower than that of the `stall_window` steps before by more than
    `stall_progress` (it is blocked by contact or a joint limit: waiting longer would only run
    into the timeout). The verdict needs `2 * stall_window` samples since the env's last `reset`.

    The windowed mean is the point: a "best error so far" criterion is defeated by noise (every
    lucky dip counts as progress), whereas the mean over a window changes only by the noise's mean
    (about `jitter / sqrt(window)`), so callers with executed-action noise raise `progress` by the
    jitter. All state is per env, so a monitor serves envs that start their targets at different
    times: `reset(mask)` restarts the history of the envs whose target changed.
    """

    def __init__(self, tol: float | np.ndarray, stall_window: int, stall_progress: float) -> None:
        if stall_window < 1:
            raise ValueError(f"stall_window must be >= 1, got {stall_window}")
        self.tol = tol
        self.stall_window = stall_window
        self.stall_progress = stall_progress
        self._buf: np.ndarray | None = None  # (n, 2 * window) last errors, oldest first
        self._count: np.ndarray | None = None  # samples since the last reset (capped)

    def reset(self, mask: np.ndarray) -> None:
        """Forget the error history of the envs in `mask` (their target changed)."""
        if self._count is not None:
            self._count = np.where(mask, 0, self._count)

    def update(
        self,
        error: np.ndarray,
        tol: float | np.ndarray | None = None,
        progress: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Record this step's `(n,)` errors; returns the `(reached, stalled)` bool masks.

        `tol` / `progress` override the monitor's values for this call (a per-env tolerance, a
        noise-widened progress threshold).
        """
        error = np.asarray(error, dtype=float)
        window = self.stall_window
        if self._buf is None:
            self._buf = np.zeros((len(error), 2 * window))
            self._count = np.zeros(len(error), dtype=int)
        self._buf[:, :-1] = self._buf[:, 1:]
        self._buf[:, -1] = error
        self._count = np.minimum(self._count + 1, 2 * window)
        previous, recent = self._buf[:, :window].mean(axis=1), self._buf[:, window:].mean(axis=1)
        threshold = self.stall_progress if progress is None else progress
        stalled = (self._count >= 2 * window) & (previous - recent < threshold)
        return error < (self.tol if tol is None else tol), stalled


def _percentiles(values: np.ndarray, scale: float = 1.0, fmt: str = ".1f") -> str:
    q = np.percentile(np.asarray(values, dtype=float) * scale, [0, 50, 100])
    return f"min {q[0]:{fmt}} / median {q[1]:{fmt}} / max {q[2]:{fmt}}"


def describe_progress(
    face_angle_max: np.ndarray,
    face_angle_end: np.ndarray,
    holder_grasp_ever: np.ndarray,
    rotator_grasp_ever: np.ndarray,
    both_grasp_ever: np.ndarray,
    body_rot_max: np.ndarray,
    body_pos_max: np.ndarray,
    target_angle: float,
    angle_tol: float,
    body_rot_tol: float,
    body_pos_tol: float,
) -> list[str]:
    """Lines describing how far the episodes of a batch got (all arrays are `(n,)` per env).

    Meant for a batch without any success: which stage the envs reached (holder grasp, rotator
    grasp, both at once, face turned) and, for the turned ones, whether the body drift is what
    spoiled the success.
    """
    n = len(face_angle_max)
    turned = face_angle_max >= target_angle - angle_tol
    drifted = (body_rot_max >= body_rot_tol) | (body_pos_max >= body_pos_tol)
    return [
        f"  envs that ever held the body (holder grasp):     {int(holder_grasp_ever.sum())}/{n}",
        f"  envs that ever held the face (rotator grasp):    {int(rotator_grasp_ever.sum())}/{n}",
        f"  envs with both grasps at the same step:          {int(both_grasp_ever.sum())}/{n}",
        f"  envs whose face got within tolerance of 90 deg:  {int(turned.sum())}/{n}",
        f"  ... of those, body drift above tolerance:        {int((turned & drifted).sum())}/{n}",
        f"  max face angle (deg):   {_percentiles(face_angle_max, 180 / np.pi)}",
        f"  face angle at the end (deg): {_percentiles(face_angle_end, 180 / np.pi)}",
        (
            f"  max body rotation drift (deg): {_percentiles(body_rot_max, 180 / np.pi)} "
            f"(tolerance {body_rot_tol * 180 / np.pi:.1f})"
        ),
        (
            f"  max body position drift (cm):  {_percentiles(body_pos_max, 100, '.2f')} "
            f"(tolerance {body_pos_tol * 100:.1f})"
        ),
    ]
