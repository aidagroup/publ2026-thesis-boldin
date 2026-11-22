"""Control-tracking parameters of the scripted FaceTurn expert (`callosum.experts.face_turn_expert`).

Plain stdlib, no torch / mani_skill, so the validation is unit-testable everywhere. The expert's
geometry (grasp offsets, tilts, roll angle) lives next to the IK in the expert module; what is
configured here is *how the arms are driven* through a given sequence of joint waypoints.
"""

from dataclasses import dataclass

CONTROL_MODES = ("pos", "delta")


@dataclass
class ExpertControlConfig:
    """How the expert turns joint-space waypoints into env actions.

    `control="pos"` drives `pd_joint_pos` (absolute joint targets that move at most
    `max_joint_step` per control step; open loop, fixed settle times): the original probe. With
    `control="delta"` it drives `pd_joint_delta_pos`, the training controller (target = current
    joint position + action * `ARM_DELTA_LIMIT`), closed loop: the action is the remaining
    joint error towards the waypoint, scaled so that the largest joint saturates at +-1 (the
    direction is kept, so the tool still moves on the straight line the waypoints describe). A
    waypoint counts as reached once every joint of the env is within a tolerance, or the env is
    stalled (blocked by contact or a joint limit), or after a timeout. In delta mode every env
    runs its own phase state machine (`callosum.experts._phases`): nobody waits for another env.
    Under this controller (target = current position + action) a steady disturbance leaves a
    steady-state error, which the integral term (`ki`, `bias_max`) compensates, so the loops end
    by tolerance and not by timeout.
    """

    control: str = "pos"
    """"pos" (`pd_joint_pos`, open loop) or "delta" (`pd_joint_delta_pos`, closed loop)."""

    overlap_approach: bool = False
    """Overlap the phases that do not need to be sequential: the rotator flies to its pre-grasp
    pose (4 cm above the face layer, clear of the holder's jaws) while the holder approaches, and
    it starts to descend as soon as the holder is at its grasp pose, while the holder's jaws are
    still closing. Saves about 90 steps under delta control (see the module doc of the expert).
    The rotator's descent then depends only on where the holder's TCP is, which the rotator can
    see (`partner_obs="full"`), not on the closing of the holder's jaws, which it cannot."""

    # --- "pos" mode ----------------------------------------------------------------------
    max_joint_step: float = 0.03
    """Rad per control step of the commanded target while moving between waypoints."""

    # --- "delta" mode --------------------------------------------------------------------
    waypoint_tol: float = 0.04
    """Largest joint error (rad) at which an intermediate waypoint of a path counts as reached.
    Loose on purpose: the waypoints lie on a straight tool path, so advancing before the arm has
    arrived (and without braking) cuts no real corner, while a tight value makes the arm stop and
    start at every one of the path's waypoints."""
    waypoint_timeout: int = 10
    """Steps after which an intermediate waypoint is given up on and the next one is started
    (in addition to the travel time, see `Rig._timeout`)."""
    approach_tol: float = 0.03
    """Largest joint error (rad) at which a fly-by phase (the pre-grasp approach, whose pose needs
    no precision because the next path starts from the commanded pose) counts as reached; such a
    phase also skips the settle steps."""
    final_tol: float = 0.006
    """Largest joint error (rad) at which the last waypoint of a movement counts as reached."""
    final_timeout: int = 25
    """Steps after which the last waypoint is given up on (a steady-state error under contact or
    at a joint limit can keep the error above `final_tol` forever; added to the travel time)."""
    stall_window: int = 3
    """Length (steps) of the window of the stall detector. Stall is judged on the mean of the
    worst joint error over two consecutive windows (so a jittery, e.g. DART-noised, error does
    not look like progress): an env whose windowed error improved by less than `stall_progress`
    (plus the noise jitter) is stalled and moves on instead of waiting for its timeout. The
    earliest verdict therefore comes `2 * stall_window` steps after a target was set."""
    stall_progress: float = 5e-4
    """Rad of improvement of the windowed worst joint error that counts as progress (see
    `stall_window`)."""
    noise_tol_scale: float = 3.0
    """With `Rig.action_noise > 0` the executed arm action jitters, so the joints jitter around
    the clean target by about `action_noise * ARM_DELTA_LIMIT * 0.39` rad (`jitter`). The reach
    tolerances then grow by `noise_tol_scale * jitter` and the stall progress threshold by
    `jitter`, so the noisy executed action can still meet them. 0 disables the widening."""
    settle_qvel_tol: float = 0.05
    """A settle phase ends early once every joint (gripper jaws included) moves slower than this
    (rad/s or m/s); it never lasts longer than the number of steps the phase asks for."""
    settle_cap: int = 8
    """Upper bound on the settle steps after a movement phase in delta mode (the closed loop
    needs less waiting than the open-loop "pos" mode). Gripper phases keep their own length."""
    gain: float = 1.5
    """Multiplier of the normalised joint error before saturation. 1.0 makes the controller's
    target the waypoint itself (when not saturated); above 1 converges faster (the error shrinks
    by `1 - 0.39 * gain` per step with the stock PD gains) and pushes through steady-state error.
    Stable and monotone up to about 2.5."""
    ki: float = 0.25
    """Integral gain: every step the per-joint bias grows by `ki * error` (rad). The bias is added
    to the commanded offset and compensates a steady-state error (the controller's target is the
    current position plus the action, so a disturbance torque needs a standing offset). It is
    frozen while the action is saturated and reset at the start of every movement. 0 disables it."""
    bias_max: float = 0.006
    """Bound (rad) of the integral bias, so that a blocked joint (table, cube, joint limit) is
    not pushed harder than `bias_max * stiffness` = 6 N*m by it."""
    joint_limit_margin: float = 0.05
    """The IK keeps the arm joints this far (rad) inside their limits. The straight-line paths
    of the stock layout end with the holder's and the rotator's wrist flex exactly at its limit,
    where the joint stops short of the commanded angle and the reach loop could not finish. The
    margin moves those targets 0.05 rad away from the limit (position error stays below 0.1 mm,
    direction error about 4 degrees of the 8 allowed). 0 gives the original IK."""

    def __post_init__(self) -> None:
        if self.control not in CONTROL_MODES:
            raise ValueError(f"control must be one of {CONTROL_MODES}, got {self.control!r}")
        if self.max_joint_step <= 0 or self.gain <= 0:
            raise ValueError("max_joint_step and gain must be > 0")
        for name in ("ki", "bias_max", "joint_limit_margin", "stall_progress", "noise_tol_scale"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in ("waypoint_tol", "approach_tol", "final_tol", "settle_qvel_tol"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("waypoint_timeout", "final_timeout", "stall_window"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.settle_cap < 0:
            raise ValueError(f"settle_cap must be >= 0, got {self.settle_cap}")
