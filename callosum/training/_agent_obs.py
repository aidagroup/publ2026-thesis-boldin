"""Per-agent policy inputs for IPPO, built from the env's single flat state observation.

With `obs_mode="state"` ManiSkill returns ONE flat tensor for both agents:
`flatten_state_dict({"agent": {uid: {"qpos", "qvel"}, ...}, "extra": {...}})`, i.e. every leaf
concatenated in dict insertion order (a `(num_envs,)` leaf becomes one column). Each agent's
policy must only see its own slice, so this module cuts that flat tensor up again.

How: `obs_layout` reads the field names and widths off the *structured* observation once
(`env.unwrapped.get_obs(unflattened=True)`), `AgentObsBuilder` turns the layout into one index
vector per agent, and every later flat observation (including `final_observation` of the
vector env) is sliced with them. `check_layout` compares the layout against a real flat
observation, so a mismatch between ManiSkill's flattening order and ours fails loudly at
start-up instead of silently feeding the wrong numbers to a policy.

Field rules (a field is a path such as `agent/so101_pg-0/qpos` or `extra/cube_pose`):

* `agent/<own uid>/*` (qpos, qvel of the own arm): always in.
* `agent/<partner uid>/*`: never in, under either `partner_obs`. `partner_obs` only governs TCP
  poses, and the partner's joint state is exactly the partner information the Bi-JEPA work
  later adds back in a controlled way.
* `extra/agent_<x>_*` (the TCP poses, always emitted by the env under both modes): the agent's
  own is always in; the partner's only for `partner_obs="full"` (`tcp_pose_visible`).
* every other `extra/*` field (`cube_pose`, `face_angle`, `face_pose`) is task state shared by
  both agents, so it is in both inputs.
* Anything else (another top-level key, an unknown `agent/*` entry) raises: a new field must be
  assigned to an agent explicitly rather than leak by default.

The index logic (`obs_layout`, `select_fields`, the builder's column lists) is plain Python and
works on anything with a `.shape`, so it is unit-tested in CI, which has no torch. torch is
imported lazily by the functions that touch tensors; there is no mani_skill import.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from callosum.envs._partner_obs import tcp_pose_visible, validate_partner_obs

AGENT_UIDS = ("so101_pg-0", "so101_pg-1")
"""ManiSkill uids of agent_a (holder in FaceTurn-v0) and agent_b (rotator), in that order."""

# Prefix of the `extra` fields that belong to one agent (`agent_a_tcp_pose`, ...).
_ROLE_PREFIX = ("agent_a_", "agent_b_")

if TYPE_CHECKING:
    import torch

STATE_GROUPS = ("agent", "extra")
"""Top-level groups ManiSkill v3.0.1 flattens into `obs["state"]` for visual obs modes with a
`state` part (`BaseEnv._flatten_raw_obs`: `dict(agent=..., extra=...)`, in this order). Pass it
as `state_groups` where the structured observation also carries sensor groups."""

Path = tuple[str, ...]
Layout = list[tuple[Path, int]]


def state_view(structured: Mapping, groups: Sequence[str] = STATE_GROUPS) -> dict:
    """The structured observation restricted to `groups`, in that order (others are dropped).

    For `obs_mode="state+rgb"` and the like, `get_obs(unflattened=True)` also holds
    `sensor_param` / `sensor_data`, which are not part of the flat `obs["state"]`. Dropping them
    is only done on explicit request (`state_groups`); with no restriction `obs_layout` and
    `select_fields` still reject unknown groups. Raises `KeyError` if a group is missing.
    """
    missing = [g for g in groups if g not in structured]
    if missing:
        raise KeyError(f"structured observation has no group(s) {missing}: {list(structured)}")
    return {g: structured[g] for g in groups}


def obs_layout(structured: Mapping) -> Layout:
    """Field paths and widths of a structured observation, in ManiSkill's flattening order.

    Args:
        structured: the nested dict of `get_obs(unflattened=True)`; leaves are tensors with a
            leading batch dimension (`(n,)` counts as width 1).

    Returns:
        `[(path, width), ...]` in dict insertion order; the widths sum to the flat observation
        size. Empty sub-dicts contribute nothing (ManiSkill drops them too).
    """
    layout: Layout = []

    def walk(node: Mapping, prefix: Path) -> None:
        for key, value in node.items():
            if isinstance(value, Mapping):
                walk(value, (*prefix, key))
            else:
                width = math.prod(value.shape[1:])
                layout.append(((*prefix, key), width))

    walk(structured, ())
    return layout


def flatten_structured(structured: Mapping) -> torch.Tensor:
    """Concatenate a structured observation into a `(num_envs, D)` tensor like ManiSkill does."""
    import torch

    parts: list[torch.Tensor] = []

    def walk(node: Mapping) -> None:
        for value in node.values():
            if isinstance(value, Mapping):
                walk(value)
            else:
                parts.append(value.reshape(value.shape[0], -1))

    walk(structured)
    return torch.cat(parts, dim=-1)


def check_layout(layout: Layout, flat_obs: torch.Tensor, structured: Mapping) -> None:
    """Raise if `layout` does not describe `flat_obs`, which came from the same `structured` obs.

    Checks the total width and that re-flattening the structured observation in layout order
    reproduces the flat tensor exactly, which pins down the flattening order, not just the size.
    """
    import torch

    total = sum(width for _, width in layout)
    if flat_obs.shape[-1] != total:
        raise ValueError(f"flat observation has {flat_obs.shape[-1]} columns, layout has {total}")
    if not torch.equal(flatten_structured(structured).to(flat_obs.dtype), flat_obs):
        raise ValueError(
            "flat observation differs from the layout-ordered concatenation of the structured "
            "observation; ManiSkill's flattening order is not what _agent_obs assumes"
        )


def select_fields(
    layout: Layout,
    agent_idx: int,
    partner_obs: str,
    agent_uids: Sequence[str] = AGENT_UIDS,
) -> list[int]:
    """Indices into `layout` of the fields that make up agent `agent_idx`'s input.

    See the module docstring for the rules. `agent_idx` is 0 for agent_a and 1 for agent_b.
    """
    validate_partner_obs(partner_obs)
    if agent_idx not in (0, 1):
        raise ValueError(f"agent_idx must be 0 or 1, got {agent_idx}")
    own_uid = agent_uids[agent_idx]
    selected: list[int] = []
    for i, (path, _) in enumerate(layout):
        group = path[0]
        if group == "agent":
            if len(path) < 2 or path[1] not in agent_uids:
                raise ValueError(f"unexpected agent observation field {'/'.join(path)}")
            keep = path[1] == own_uid
        elif group == "extra":
            name = path[-1]
            owner = next((i for i, p in enumerate(_ROLE_PREFIX) if name.startswith(p)), None)
            # A role-prefixed field is visible per the partner_obs rule; the rest is shared
            # task state (cube / face).
            keep = True if owner is None else tcp_pose_visible(partner_obs, agent_idx, owner)
        else:
            raise ValueError(f"unexpected observation group {'/'.join(path)}")
        if keep:
            selected.append(i)
    return selected


class AgentObsBuilder:
    """Cuts the env's flat observation into one policy input per agent.

    Args:
        layout: `obs_layout` of the env's structured observation.
        partner_obs: the env's `partner_obs` mode (`"full"` or `"none"`).
        agent_uids: `(agent_a_uid, agent_b_uid)`.

    Attributes:
        fields: per agent, the `/`-joined names of the fields in its input, in order.
        columns: per agent, the column indices into the flat observation, in order.
        obs_dims: per agent, the input width.
    """

    def __init__(
        self,
        layout: Layout,
        partner_obs: str,
        agent_uids: Sequence[str] = AGENT_UIDS,
    ) -> None:
        self.layout = layout
        self.partner_obs = partner_obs
        self.agent_uids = tuple(agent_uids)
        offsets = [0]
        for _, width in layout:
            offsets.append(offsets[-1] + width)
        self.total_dim = offsets[-1]
        self.fields: list[list[str]] = []
        self.columns: list[list[int]] = []
        for agent_idx in (0, 1):
            chosen = select_fields(layout, agent_idx, partner_obs, self.agent_uids)
            columns = [c for i in chosen for c in range(offsets[i], offsets[i + 1])]
            self.fields.append(["/".join(layout[i][0]) for i in chosen])
            self.columns.append(columns)
        self.obs_dims = [len(c) for c in self.columns]
        self._index: dict[tuple[int, str], torch.Tensor] = {}

    def __call__(self, flat_obs: torch.Tensor) -> list[torch.Tensor]:
        """`[obs_agent_a, obs_agent_b]`, each `(batch, obs_dim)`, from a `(batch, D)` tensor."""
        if flat_obs.shape[-1] != self.total_dim:
            raise ValueError(
                f"flat observation has {flat_obs.shape[-1]} columns, expected {self.total_dim}"
            )
        return [flat_obs[:, self._index_tensor(i, flat_obs.device)] for i in (0, 1)]

    def _index_tensor(self, agent_idx: int, device: torch.device) -> torch.Tensor:
        """The agent's column indices as a long tensor on `device` (cached)."""
        import torch

        key = (agent_idx, str(device))
        if key not in self._index:
            self._index[key] = torch.tensor(
                self.columns[agent_idx], dtype=torch.long, device=device
            )
        return self._index[key]

    @classmethod
    def from_env_obs(
        cls,
        structured: Mapping,
        flat_obs: torch.Tensor,
        partner_obs: str,
        agent_uids: Sequence[str] = AGENT_UIDS,
        state_groups: Sequence[str] | None = None,
    ) -> AgentObsBuilder:
        """Builder for an env, verified against one of its real flat observations.

        `state_groups=None` (training, `obs_mode="state"`) uses every group of `structured`, so
        an unknown one raises in `select_fields`. Playback in a `state+rgb` env passes
        `STATE_GROUPS` to ignore the camera groups, which are not in the flat state vector.
        """
        if state_groups is not None:
            structured = state_view(structured, state_groups)
        layout = obs_layout(structured)
        check_layout(layout, flat_obs, structured)
        return cls(layout, partner_obs, agent_uids)
