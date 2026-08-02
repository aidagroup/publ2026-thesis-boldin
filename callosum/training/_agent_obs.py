"""Pure per-agent observation assembly for independent (IPPO-style) policies
(step 2.1), given ManiSkill's `obs_mode="state_dict"` observation.

No mani_skill import, so this module is unit-testable on macOS/CI, unlike
callosum.training.ippo -- see docs/implementation-plan.md section 0. This is
the "explicitly assemble each agent's own policy input from the shared
observation" logic the plan calls out as critical for step 2.1, continuing
into step 3.2's partner_input toggle.

Why obs_mode="state_dict" instead of "state": obs_mode="state" flattens
`agent`+`extra` into ONE combined tensor
(mani_skill.utils.common.flatten_state_dict), destroying the per-agent
structure that MultiAgent's proprioception dict naturally has (established
in step 1.2/1.4). obs_mode="state_dict" does NOT flatten -- confirmed by
reading mani_skill.envs.utils.observations.parse_obs_mode_to_struct and
BaseEnv._flatten_raw_obs at v3.0.1: the "state" flattening branch only
triggers for the exact obs_mode string "state", and
ObservationModeStruct.state -- which gates the *other* flattening branch --
is False for "state_dict" specifically (it sets state_dict=True, state=
False). So env.step()/env.reset() return the raw
`{"agent": {uid: {...}}, "extra": {...}}` dict all the way through
(ManiSkillVectorEnv passes obs through unmodified -- confirmed by reading
its step()/reset()).

flatten_dict_to_tensor below is a minimal, mani_skill-independent stand-in
for mani_skill.utils.common.flatten_state_dict (same insertion-order
concatenation contract), written only so this module has no mani_skill
dependency; it is not meant to handle the general cases (numpy arrays,
scalars, ...) that function does, only plain tensors, which is all our
envs' observations ever contain.

build_agent_obs's `include_partner` flag is the env-level-impossible half of
the oracle/no-partner ablation finally becoming expressible: step 1.4's
`partner_obs` env flag could not do this, because obs_mode="state"'s shared
flat observation goes to both agents identically -- there was no per-agent
observation to selectively hide it from in the first place. Here, per-agent
assembly already exists, so "keep the partner's fields for this agent
anyway" is just skipping the drop. Also needed regardless for step 3.2's
`partner_input` ablation, so this pulls forward already-planned work.
"""

import torch

# Naming convention this relies on (see callosum.envs.two_so100_base and
# callosum.envs.face_turn's _get_obs_extra): extra-obs fields specific to one
# agent are prefixed "agent_a_"/"agent_b_"; shared/task fields (cube_pose,
# face_angle, ...) have no such prefix and belong in both agents' inputs.
_OTHER_AGENT_PREFIX = {0: "agent_b_", 1: "agent_a_"}


def select_agent_extra_fields(extra: dict, agent_idx: int, include_partner: bool = False) -> dict:
    """The obs["extra"] fields that belong in agent `agent_idx`'s own input.

    Drops fields prefixed with the *other* agent's role name; keeps
    everything else (that agent's own agent_x_* fields, plus any unprefixed
    shared field). `agent_idx` is 0 for agent_a, 1 for agent_b.

    include_partner=True skips the drop entirely (an "oracle" diagnostic
    condition -- see build_agent_obs); default False is the decentralized
    baseline.
    """
    if include_partner:
        return dict(extra)
    other_prefix = _OTHER_AGENT_PREFIX[agent_idx]
    return {k: v for k, v in extra.items() if not k.startswith(other_prefix)}


def flatten_dict_to_tensor(fields: dict) -> torch.Tensor:
    """Concatenate a dict of same-batch-size tensors (recursing into nested
    dicts) into one (num_envs, D) tensor, in dict insertion order."""
    parts = []
    for value in fields.values():
        if isinstance(value, dict):
            parts.append(flatten_dict_to_tensor(value))
        else:
            parts.append(value.reshape(value.shape[0], -1))
    return torch.cat(parts, dim=-1)


def build_agent_obs(
    raw_obs: dict,
    agent_idx: int,
    agent_uids: tuple[str, str],
    include_partner: bool = False,
) -> torch.Tensor:
    """Assemble agent `agent_idx`'s own flat policy-input tensor.

    Args:
        raw_obs: the obs_mode="state_dict" observation, i.e.
            `{"agent": {uid: {"qpos": ..., "qvel": ...}, ...}, "extra": {...}}`.
        agent_idx: 0 for agent_a, 1 for agent_b.
        agent_uids: `(agent_a_uid, agent_b_uid)`, e.g. `("so100-0", "so100-1")`.
        include_partner: if True, keep the other agent's `agent_x_*`
            extra-obs fields instead of dropping them -- an "oracle"
            diagnostic condition for isolating whether a training failure is
            due to missing partner information, rather than the task/reward
            or the gripper's ability to grasp. Default False is the
            decentralized baseline. Every call site for a given training run
            must agree on this value: the value function and the policy
            input must see the same observation, or advantage estimates
            become meaningless.

    Returns:
        (num_envs, D) tensor: this agent's own proprioception followed by
        its slice of the extra-obs fields (see select_agent_extra_fields).
    """
    fields = {"proprio": raw_obs["agent"][agent_uids[agent_idx]]}
    fields.update(select_agent_extra_fields(raw_obs["extra"], agent_idx, include_partner))
    return flatten_dict_to_tensor(fields)
