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
    waypoint counts as reached once every joint is within a tolerance, or after a timeout.
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
    waypoint_tol: float = 0.02
    """Largest joint error (rad) at which an intermediate waypoint of a path counts as reached."""
    waypoint_timeout: int = 15
    """Steps after which an intermediate waypoint is given up on and the next one is started."""
    final_tol: float = 0.006
    """Largest joint error (rad) at which the last waypoint of a movement counts as reached."""
    final_timeout: int = 40
    """Steps after which the last waypoint is given up on (steady-state error under gravity or
    contact can keep the error above `final_tol` forever)."""
    settle_qvel_tol: float = 0.05
    """A settle phase ends early once every joint (gripper jaws included) moves slower than this
    (rad/s or m/s); it never lasts longer than the number of steps the phase asks for."""
    settle_cap: int = 8
    """Upper bound on the settle steps after a movement phase in delta mode (the closed loop
    needs less waiting than the open-loop "pos" mode). Gripper phases keep their own length."""
    gain: float = 1.0
    """Multiplier of the normalised joint error before saturation. 1.0 makes the controller's
    target the waypoint itself (when not saturated); above 1 pushes through steady-state error."""

    def __post_init__(self) -> None:
        if self.control not in CONTROL_MODES:
            raise ValueError(f"control must be one of {CONTROL_MODES}, got {self.control!r}")
        if self.max_joint_step <= 0 or self.gain <= 0:
            raise ValueError("max_joint_step and gain must be > 0")
        for name in ("waypoint_tol", "final_tol", "settle_qvel_tol"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("waypoint_timeout", "final_timeout"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.settle_cap < 0:
            raise ValueError(f"settle_cap must be >= 0, got {self.settle_cap}")
