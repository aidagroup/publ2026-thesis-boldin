"""Reward-weight and success-tolerance config for the FaceTurn task."""

from dataclasses import dataclass


@dataclass
class FaceTurnRewardConfig:
    """Dense-reward term weights and success tolerances for FaceTurn-v0.

    See `callosum.envs.face_turn.FaceTurn.compute_dense_reward` (weights) and
    `.evaluate` (tolerances) for how each field is used.

    The staircase the weights encode, in the order the task has to happen:
    the holder reaches the cube, grips it, LIFTS it to working height; only
    then can the rotator reach the nub, grip it, and turn.
    """

    # Term weights, in task order.
    weight_holder_reach: float = 1.0
    # Holder gripping the BODY. Without this the holder has no positive signal
    # for its actual job -- holding the cube up and still while the face is
    # turned. Nothing else can do it: with the cube in the air there is no
    # table friction to resist the face joint's reaction torque.
    weight_holder_grasp: float = 1.0
    # Getting the cube to `_cube_geometry.LIFT_HEIGHT`. Gated on the holder
    # actually gripping, so batting the cube upward earns nothing.
    weight_lift: float = 2.0
    # The rotator's two terms are gated on the cube being lifted. Ungated, the
    # reach term would pay the rotator to drive at a nub that is still down at
    # table height, where its own gripper cannot go: the closed gripper sweeps
    # 4.26 cm below the cube's axis (see _cube_geometry.NUB_HALF_WIDTH).
    weight_rotator_reach: float = 1.0
    weight_grasp: float = 1.0
    weight_angle_progress: float = 3.0
    # Penalty for the body TURNING. Not for translating -- the cube is meant
    # to move now, that is the lift. What must not happen is the body rotating
    # under the reaction torque instead of the face turning.
    weight_body_rot_drift: float = 5.0

    # Success/stability tolerances.
    angle_tol: float = 0.05  # rad (~3 deg) short of the quarter turn
    lift_tol: float = 0.015  # m below LIFT_HEIGHT that still counts as lifted
    body_rot_tol: float = 0.1  # rad (~6 deg) of body rotation from its spawn

# NOTE: SO-100 gripper force_limit=100 N·m is boilerplate (shared with arm joints),
# not a real servo figure. All comfortable margins (200-1500x) depend on it.
# Worth an ablation with realistic cap before trusting grasp-stability results.
