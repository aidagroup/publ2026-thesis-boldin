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
* `ReachMonitor`: per-env "reached or stalled" bookkeeping, so an env that is blocked (contact,
  joint limit) does not make the whole batch wait for the timeout.
* `ReachRecord` / `summarize_reaches` / `describe_progress`: what the probe and the demo collector
  print to make a slow or failed run diagnosable from the log.
"""

from collections.abc import Sequence
from dataclasses import dataclass

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


class ReachMonitor:
    """Per-env bookkeeping of one "step until the targets are reached" loop.

    Feed it the per-env worst joint error after every step. An env is `reached` when that error is
    below `tol`, and `stalled` when its best error has not improved by more than `stall_progress`
    for `stall_steps` consecutive steps (it is blocked by contact or a joint limit: waiting longer
    would only run into the timeout). The loop can end once every env is reached or stalled.
    """

    def __init__(self, tol: float | np.ndarray, stall_steps: int, stall_progress: float) -> None:
        self.tol = tol
        self.stall_steps = stall_steps
        self.stall_progress = stall_progress
        self._best: np.ndarray | None = None
        self._since: np.ndarray | None = None

    def reset(self, mask: np.ndarray) -> None:
        """Forget the progress history of the envs in `mask` (their target changed)."""
        if self._best is not None:
            self._best = np.where(mask, np.inf, self._best)
            self._since = np.where(mask, 0, self._since)

    def update(self, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Record this step's `(n,)` errors; returns the `(reached, stalled)` bool masks."""
        error = np.asarray(error, dtype=float)
        if self._best is None:
            self._best = error.copy()
            self._since = np.zeros(error.shape, dtype=int)
        else:
            improved = error < self._best - self.stall_progress
            self._best = np.where(improved, error, self._best)
            self._since = np.where(improved, 0, self._since + 1)
        return error < self.tol, self._since >= self.stall_steps


def reach_reason(reached: np.ndarray, stalled: np.ndarray, extra: np.ndarray | None) -> str:
    """Why a reach loop ended: `"tol"`, `"stall"` (some envs stalled) or `"timeout"`.

    `extra` is an optional per-env "done anyway" mask (e.g. the face already turned). `"timeout"`
    is returned when some env is neither reached, stalled nor done.
    """
    done_clean = reached if extra is None else reached | extra
    if bool(done_clean.all()):
        return "tol"
    if bool((done_clean | stalled).all()):
        return "stall"
    return "timeout"


@dataclass
class ReachRecord:
    """One finished reach loop (see `Rig._reach`)."""

    arms: tuple[int, ...]
    steps: int
    reason: str  # "tol" | "stall" | "timeout"
    tol: float
    unreached_envs: int  # envs still above `tol` when the loop ended
    worst_error: float  # largest single joint error (rad) at the end
    worst_arm: int
    worst_env: int
    worst_joint: int


def summarize_reaches(records: Sequence[ReachRecord]) -> str:
    """One line about a phase's reach loops: how they ended and where the worst error was."""
    if not records:
        return "no closed-loop reaches"
    counts = {r: sum(rec.reason == r for rec in records) for r in ("tol", "stall", "timeout")}
    steps = sum(rec.steps for rec in records)
    worst = max(records, key=lambda rec: rec.worst_error)
    most_unreached = max(rec.unreached_envs for rec in records)
    return (
        f"{len(records)} reaches ({steps} steps): {counts['tol']} by tolerance, "
        f"{counts['stall']} by stall, {counts['timeout']} by timeout; "
        f"max unreached envs at an end {most_unreached}; worst final joint error "
        f"{worst.worst_error:.3f} rad (arm {worst.worst_arm}, env {worst.worst_env}, joint "
        f"{worst.worst_joint}, end of a {worst.steps}-step reach, tol {worst.tol})"
    )


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
