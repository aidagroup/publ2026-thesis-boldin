"""Pure logic for the `partner_obs` observation-visibility flag (step 1.4).

Deliberately has zero mani_skill/torch/sapien dependency, unlike
`two_so101_base.py` (which imports mani_skill at module level and so cannot
be imported at all on macOS/CI, where mani_skill is absent -- see
docs/implementation-plan.md section 0). Keeping this decision logic in its
own dependency-free module is what makes it possible to unit-test on macOS,
per step 1.4's readiness criterion.

Where the rule lives: with obs_mode="state" ManiSkill flattens the env's extra-obs dict into
ONE tensor shared by both agents, so the env itself cannot hide anything from one agent. The
env therefore ALWAYS emits both agents' TCP poses (`tcp_pose_fields`) and `partner_obs` no
longer changes the env's observation. The visibility rule is `tcp_pose_visible`, applied by the
per-agent input builder `callosum.training._agent_obs`:

* an agent's OWN TCP pose is always in its input (under both modes -- otherwise "none" would
  also take away the agent's own end-effector position, which is not partner information);
* the PARTNER's TCP pose is in the input only for "full" (the oracle end of the ablation triple
  from docs/thesis/04-experiment-design.md); "none" is the no-partner end. "predicted" (Bi-JEPA,
  phase 3) will replace the partner's TCP pose by a prediction there.

The partner's joint state (`agent/<uid>/qpos`, `qvel`) is hidden under both modes.
"""

PARTNER_OBS_MODES = ("full", "none")  # later: + "predicted" (Bi-JEPA, phase 3)


def validate_partner_obs(partner_obs: str) -> None:
    """Raise ValueError if `partner_obs` isn't one of PARTNER_OBS_MODES."""
    if partner_obs not in PARTNER_OBS_MODES:
        raise ValueError(f"partner_obs must be one of {PARTNER_OBS_MODES}, got {partner_obs!r}")


def tcp_pose_fields(agent_a_tcp_pose, agent_b_tcp_pose) -> dict:
    """The TCP-pose fields the env always adds to the shared extra-obs dict.

    Args:
        agent_a_tcp_pose: agent_a's current TCP pose (any pose-like value).
        agent_b_tcp_pose: agent_b's current TCP pose (any pose-like value).

    Returns:
        `{"agent_a_tcp_pose": ..., "agent_b_tcp_pose": ...}`, independent of `partner_obs`.
    """
    return {"agent_a_tcp_pose": agent_a_tcp_pose, "agent_b_tcp_pose": agent_b_tcp_pose}


def tcp_pose_visible(partner_obs: str, agent_idx: int, owner_idx: int) -> bool:
    """Whether agent `agent_idx` may see the TCP pose of agent `owner_idx` (0 = a, 1 = b).

    True for the agent's own TCP pose under every mode, and for the partner's only under "full".
    """
    validate_partner_obs(partner_obs)
    if agent_idx not in (0, 1) or owner_idx not in (0, 1):
        raise ValueError(f"agent indices must be 0 or 1, got {agent_idx} and {owner_idx}")
    return agent_idx == owner_idx or partner_obs == "full"
