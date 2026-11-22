"""Observation-layout contract for FaceTurnSkills-v0 (single source of truth).

The env builds the shared flat observation vector, the trainer slices per-agent
policy inputs and the eval harness reads episode info -- all three consume THIS
module, never their own idea of the layout (design decision: masking happens at
the policy-input level, exactly like the joint-level `_partner_obs` rule).

Pure stdlib, like `_skills`: must import and unit test on macOS.

Layout of the shared flat state vector (obs_mode="state"), per environment:

    [0 .. own_end)      own block, IDENTICAL field order for both agents
    (the partner-related block is NOT part of the shared vector; it is appended
     by the trainer when the partner mode demands it -- see `policy_input_size`)

Own-block field order (fixed; sizes in parentheses):
    time_fraction (1)            fraction of the decision budget spent, [0, 1]
    plan_face_onehot (6)         current plan move's target face U/D/L/R/F/B
    plan_quarter_turns (1)       1 or 2, normalised by 2
    plan_moves_remaining (1)     remaining moves / initial plan length, [0, 1]
    cube_position (3)
    cube_orientation_quat (4)
    active_layer_angle (1)       radians / pi, [0, 1]
    own_qpos (7)
    own_qvel (7)
    own_gripper_opening (1)
    own_primitive_onehot (n_own) the agent's CURRENT primitive (its role registry)
    own_primitive_progress (1)   [0, 1]

Per-agent proprioception additionally travels in the obs dict under the agent
keys (same shape as FaceTurn-v0), so a policy never needs the partner's qpos.
"""

from callosum.envs._skills import (
    PARTNER_VISIBILITY_MODES,
    PartnerVisibility,
    Role,
    num_primitives,
)

# --- own block ---------------------------------------------------------------------------


def own_block_fields(n_own: int) -> tuple[tuple[str, int], ...]:
    """The own-block layout as (field_name, size) pairs, in fixed order.

    `n_own` is the size of the agent's own role registry
    (`num_primitives(role)`); the sum of sizes is `own_block_size`.
    """
    return (
        ("time_fraction", 1),
        ("plan_face_onehot", 6),
        ("plan_quarter_turns", 1),
        ("plan_moves_remaining", 1),
        ("cube_position", 3),
        ("cube_orientation_quat", 4),
        ("active_layer_angle", 1),
        ("own_qpos", 7),
        ("own_qvel", 7),
        ("own_gripper_opening", 1),
        ("own_primitive_onehot", n_own),
        ("own_primitive_progress", 1),
    )


def own_block_size(n_own: int) -> int:
    """Total width of the own block for an agent whose registry has `n_own` entries."""
    return sum(size for _, size in own_block_fields(n_own))


def own_block_slices(n_own: int) -> dict[str, slice]:
    """Index ranges of every own-block field (stable, contiguous, no padding)."""
    slices: dict[str, slice] = {}
    start = 0
    for name, size in own_block_fields(n_own):
        slices[name] = slice(start, start + size)
        start += size
    return slices


# --- policy input ------------------------------------------------------------------------


def policy_input_size(mode: str, role: Role, latent_dim: int = 16) -> int:
    """Width of the per-agent policy input: own block + the mode's partner block.

    The partner block is appended AFTER the own block (the trainer owns the
    concatenation; the shared env vector carries none of it). "predicted" and
    "predicted-direct" externals have the same env-level inputs as "none".
    """
    if mode not in PARTNER_VISIBILITY_MODES:
        raise ValueError(f"mode must be one of {PARTNER_VISIBILITY_MODES}, got {mode!r}")
    partner = Role.ROTATOR if role is Role.HOLDER else Role.HOLDER
    own = own_block_size(num_primitives(role))
    partner_block = PartnerVisibility(mode).partner_block_size(num_primitives(partner), latent_dim)
    return own + partner_block


# --- episode info keys (env -> trainer/eval) ---------------------------------------------

INFO_SUCCESS = "success"
"""bool per env: the plan is executed within tolerances (terminal)."""

INFO_TIME_TO_SUCCESS = "time_to_success_steps"
"""int per env: sim steps until success (0 / absent while not successful)."""

INFO_ROTATOR_GRASP_FACT = "rotator_grasp_fact"
"""bool per env: the env's authoritative rotator-grasp fact (contact OR, in the
`turn_grasp_mode="proximity"` simplification, an established proximity pinch --
see `FaceTurnSkillsConfig.turn_grasp_mode`). The conductor prefers this over the
jaw-opening band; the evaluation harness may score grasp timing from it."""

INFO_USELESS_HOLDER = "useless_action_holder"
INFO_USELESS_ROTATOR = "useless_action_rotator"
"""bool per env per decision: the agent's choice was physically pointless at the
moment of the decision (turn a locked layer, release what is not held, grasp
air far from the target, flip while the rotator holds the layer, ...)."""

INFO_PRIMITIVE_HOLDER = "holder_primitive"
INFO_PRIMITIVE_ROTATOR = "rotator_primitive"
"""int per env: the role's CURRENT primitive id (its registry index); the
oracle input is derived from the partner's value, prediction accuracy is scored
against it (probe / predicted-direct)."""

INFO_ACTIVE_LAYER_ANGLE_DEG = "active_layer_angle_deg"
"""float per env: the angle of the layer the current plan move targets."""

INFO_PLAN_INDEX = "plan_index"
"""int per env: how many plan moves are complete (0 = first move pending)."""
