"""Reward-weight and success-tolerance config for the RubikCube task."""

from dataclasses import dataclass


@dataclass
class RubikRewardConfig:
    """Dense-reward term weights and success tolerances for RubikCube-v0.

    First seven fields are FaceTurn's staircase (holder reach -> grasp -> lift
    -> rotator reach -> grasp -> angle), reused unchanged so a single turn is
    shaped exactly as it was there. The rest score progress toward a SOLVED
    cube, not just one turn -- see `callosum.envs.rubik.RubikCube`.
    """

    # --- FaceTurn's staircase, same defaults. ---
    weight_holder_reach: float = 1.0
    weight_holder_grasp: float = 1.0
    weight_lift: float = 2.0
    weight_rotator_reach: float = 1.0
    weight_grasp: float = 1.0
    weight_angle_progress: float = 3.0
    weight_body_rot_drift: float = 5.0

    # Progress in solved facelets since the scramble, as a fraction of 54.
    # Dense every step (not just on change), so holding a solved state pays
    # too -- matches the staircase terms' own always-on shape.
    weight_facelets: float = 5.0
    # Per completed quarter turn (RubikCube.moves_applied, cumulative for the
    # episode). At scramble_depth=1 almost any completed turn reduces
    # distance to solved, so this mostly rewards the same thing as
    # weight_facelets but survives a facelet count that briefly plateaus
    # mid-turn.
    weight_move: float = 1.0
    bonus_solved: float = 10.0

    # Quarter turns applied to reach the scrambled start state.
    scramble_depth: int = 1

    # Success/stability tolerances -- same meaning as FaceTurnRewardConfig's.
    angle_tol: float = 0.05  # rad (~3 deg) short of a quarter turn
    lift_tol: float = 0.015  # m below LIFT_HEIGHT that still counts as lifted
    body_rot_tol: float = 0.1  # rad (~6 deg) of body rotation from its spawn
