"""Reward-weight and success-tolerance config for the FaceTurn task."""

from dataclasses import dataclass


@dataclass
class FaceTurnRewardConfig:
    """Dense-reward term weights and success tolerances for FaceTurn-v0.

    See `callosum.envs.face_turn.FaceTurn.compute_dense_reward` (weights) and
    `.evaluate` (tolerances) for how each field is used.
    """

    # Term weights.
    weight_rotator_reach: float = 1.0
    # Rotator grasping the face.
    weight_grasp: float = 1.0
    # Holder grasping the BODY. Without this the holder has no positive
    # signal for its actual job -- stabilising the cube so the face can be
    # turned without the whole body spinning. The task needs a reward
    # staircase: approach, grip, turn.
    #
    # Note that both grasp terms were structurally unreachable before the
    # 2026-08-30 geometry fix: neither link had a feature the SO-100's jaw can
    # close on, so `is_grasping` could never fire and both terms were constant
    # zero while still inflating `reward_normalization_divisor`. See
    # callosum.envs._turntable_cube.HANDLE_HALF_WIDTH.
    weight_holder_grasp: float = 1.0
    weight_angle_progress: float = 3.0
    weight_holder_reach: float = 1.0
    # Separate weights: position (m) and rotation (rad) drift are not on
    # commensurate scales, and neither are their tolerances below.
    weight_body_pos_drift: float = 5.0
    weight_body_rot_drift: float = 5.0

    # Success/stability tolerances.
    angle_tol: float = 0.05  # rad (~3 deg) from the pi/2 target
    body_pos_tol: float = 0.01  # m of body-center drift from its initial pose
    body_rot_tol: float = 0.1  # rad (~6 deg) of body rotation from its initial pose
