"""Pure logic for the `partner_obs` observation-visibility flag (step 1.4).

Deliberately has zero mani_skill/torch/sapien dependency, unlike
`two_so101_base.py` (which imports mani_skill at module level and so cannot
be imported at all on macOS/CI, where mani_skill is absent -- see
docs/implementation-plan.md section 0). Keeping this decision logic in its
own dependency-free module is what makes it possible to unit-test on macOS,
per step 1.4's readiness criterion.

This is an env-level toggle: it controls whether TwoSO101Base._get_obs_extra
includes the two agents' TCP poses in the *shared* extra-obs dict at all.
Since obs_mode="state" flattens that dict into one combined tensor (see the
note in two_so101_base.py's _get_obs_extra), there is no true per-agent
(self-vs-partner) observation split at this level yet -- "none" therefore
means "neither agent's TCP pose is in the shared extra-obs dict", not
"agent i can't see agent i's own TCP pose but can see agent j's". Real
per-agent decentralized inputs are a training/policy-side concern for later
phases (see step 3.2's separate partner_input toggle).
"""

PARTNER_OBS_MODES = ("full", "none")  # later: + "predicted" (Bi-JEPA, phase 3)


def validate_partner_obs(partner_obs: str) -> None:
    """Raise ValueError if `partner_obs` isn't one of PARTNER_OBS_MODES."""
    if partner_obs not in PARTNER_OBS_MODES:
        raise ValueError(f"partner_obs must be one of {PARTNER_OBS_MODES}, got {partner_obs!r}")


def partner_tcp_pose_fields(partner_obs: str, agent_a_tcp_pose, agent_b_tcp_pose) -> dict:
    """The cross-agent TCP-pose fields to add to the shared extra-obs dict.

    Args:
        partner_obs: one of PARTNER_OBS_MODES.
        agent_a_tcp_pose: agent_a's current TCP pose (any pose-like value).
        agent_b_tcp_pose: agent_b's current TCP pose (any pose-like value).

    Returns:
        `{"agent_a_tcp_pose": ..., "agent_b_tcp_pose": ...}` if "full",
        `{}` if "none".
    """
    if partner_obs == "full":
        return {"agent_a_tcp_pose": agent_a_tcp_pose, "agent_b_tcp_pose": agent_b_tcp_pose}
    return {}
