"""Per-agent policy inputs for the skill-level (FaceTurnSkills) IPPO trainer.

The skills env returns ONE flat state observation per env, shared by both agents:
``[holder_own_block | rotator_own_block]`` in the fixed field order of
`callosum.experiments.arm.skills_obs.own_block_fields`, with each agent's proprioception additionally
under the obs dict's agent keys (the same shape as FaceTurn-v0). The partner-related block is
NOT part of that vector (see `_skills_obs`): this module appends it per the partner-visibility
mode, exactly like the joint-level `_agent_obs` applies `partner_obs` -- masking happens at
the policy-input level, never inside the env.

Two layers over the same index constants (`_skills_obs` is the single source of truth):

* numpy: `extract`, `agent_obs_dim`, `build_agent_obs` -- plain arrays, no torch; this is the
  contract the eval harness and the tests exercise;
* torch: `SkillsObsBuilder` (used by `callosum.experiments.arm.trainer`) cuts the same columns out of
  the trainer's flat torch observations and appends the same partner block on-device, so GPU
  rollouts never round-trip through numpy.

Partner block per mode (widths via `policy_input_size` / `PartnerVisibility.partner_block_size`):

* "none": nothing (the control arm of the occlusion ablation).
* "oracle": the partner's CURRENT primitive one-hot, read from `info[INFO_PRIMITIVE_*]`, plus
  the partner's primitive progress, taken from the partner's own block inside the shared
  vector -- and only those two fields.
* "predicted" / "predicted-direct": a zero STUB of the contract width. The real Bi-JEPA
  latent (spring change) / classifier distribution (baseline) arrives externally; what matters
  now is that the input width already matches `policy_input_size`.

`proprio` (`(batch, PROPRIO_DIM)` = own qpos(7) + qvel(7), the obs dict's per-agent entry) is
written over the own block's `own_qpos` / `own_qvel` fields: the per-agent keys are the
authoritative source ("a policy never needs the partner's qpos"), the shared vector's copies
are not trusted. Pass `None` to keep the shared vector's values.

Roles: agent_a = holder, agent_b = rotator (the fixed-role convention of
`callosum.experiments.arm.skills.Role`), so the shared vector's block order matches the agent order.

The index logic works on anything with a `.shape`; torch is imported lazily by the builder,
as in `_agent_obs`. No mani_skill import.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from callosum.experiments.arm.agent_obs import Layout, check_layout, obs_layout
from callosum.experiments.arm.skills import (
    Role,
    num_primitives,
    validate_partner_visibility,
)
from callosum.experiments.arm.skills_obs import (
    INFO_PRIMITIVE_HOLDER,
    INFO_PRIMITIVE_ROTATOR,
    own_block_fields,
    own_block_size,
    own_block_slices,
    policy_input_size,
)

ROLES = (Role.HOLDER, Role.ROTATOR)
"""agent_a's and agent_b's roles, in agent order (fixed-role E1/E2 configuration)."""


def _fixed_field_size(name: str) -> int:
    """Width of an own-block field that does not depend on the registry size."""
    for field, size in own_block_fields(num_primitives(Role.HOLDER)):
        if field == name:
            return size
    raise AssertionError(f"{name} is not an own-block field")


QPOS_DIM = _fixed_field_size("own_qpos")
QVEL_DIM = _fixed_field_size("own_qvel")
PROPRIO_DIM = QPOS_DIM + QVEL_DIM
"""Width of `proprio`: the agent's own qpos + qvel, as carried under the obs dict's agent key."""


def _own_offsets() -> dict[Role, int]:
    """Start column of each role's own block inside the shared vector."""
    return {Role.HOLDER: 0, Role.ROTATOR: own_block_size(num_primitives(Role.HOLDER))}


def shared_vec_size() -> int:
    """Width of the env's shared flat vector: `[holder_own_block | rotator_own_block]`."""
    return own_block_size(num_primitives(Role.HOLDER)) + own_block_size(
        num_primitives(Role.ROTATOR)
    )


def _partner_of(role: Role) -> Role:
    """The other hand's role."""
    return Role.ROTATOR if role is Role.HOLDER else Role.HOLDER


def partner_primitive_key(role: Role) -> str:
    """The info key holding the CURRENT primitive id of `role`'s partner."""
    return INFO_PRIMITIVE_HOLDER if _partner_of(role) is Role.HOLDER else INFO_PRIMITIVE_ROTATOR


def extract(shared_vec, role: Role) -> np.ndarray:
    """`role`'s own block out of the shared flat vector (a numpy view, contract field order).

    The partner's block stays reachable only through `extract(shared_vec, _partner_of(role))`,
    which is what the oracle progress (and nothing else in `none`/`predicted`) reads.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {[r.value for r in ROLES]}, got {role!r}")
    vec = np.asarray(shared_vec)
    if vec.shape[-1] != shared_vec_size():
        raise ValueError(
            f"shared vector has {vec.shape[-1]} columns, expected {shared_vec_size()} "
            "([holder_own_block | rotator_own_block], see callosum.experiments.arm.skills_obs)"
        )
    start = _own_offsets()[role]
    return vec[..., start : start + own_block_size(num_primitives(role))]


def agent_obs_dim(mode: str, role: Role) -> int:
    """Width of `role`'s policy input under `mode`; literally `_skills_obs.policy_input_size`."""
    return policy_input_size(mode, role)


def _partner_onehot(ids: np.ndarray, n_partner: int, batch_shape: tuple[int, ...]) -> np.ndarray:
    """One-hot of the partner's primitive `ids` over its role registry, shape-checking `ids`."""
    if ids.shape != batch_shape:
        raise ValueError(
            f"partner primitive ids have shape {ids.shape}, expected the batch shape {batch_shape}"
        )
    if np.any((ids < 0) | (ids >= n_partner)):
        raise ValueError(f"partner primitive ids must be in [0, {n_partner}), got {ids.tolist()}")
    onehot = np.zeros((*batch_shape, n_partner), dtype=np.float32)
    flat = onehot.reshape(-1, n_partner)  # diagonal one-hot per batch element
    flat[np.arange(flat.shape[0]), ids.reshape(-1)] = 1.0
    return onehot


def build_agent_obs(shared_vec, proprio, info, mode: str, role: Role) -> np.ndarray:
    """Assemble `role`'s policy input: own block (+ proprio) then the mode's partner block.

    Args:
        shared_vec: `(batch, shared_vec_size())` env observation (both agents' own blocks).
        proprio: `(batch, PROPRIO_DIM)` own qpos+qvel from the obs dict's agent key, written
            over the own block's `own_qpos`/`own_qvel`; `None` keeps the shared vector's values.
        info: the per-env episode info of the SAME step as `shared_vec`; read only for
            `mode="oracle"`, key `partner_primitive_key(role)` (per-env partner primitive ids).
        mode: one of `PARTNER_VISIBILITY_MODES`.
        role: the observing agent's role.

    Returns:
        `(batch, agent_obs_dim(mode, role))` float32 array: the own block with the mode's
        partner block appended (see the module docstring for each mode's contents).
    """
    validate_partner_visibility(mode)
    own = np.array(extract(shared_vec, role), dtype=np.float32)
    slices = own_block_slices(num_primitives(role))
    if proprio is not None:
        p = np.asarray(proprio, dtype=np.float32)
        if p.shape[-1] != PROPRIO_DIM:
            raise ValueError(f"proprio has width {p.shape[-1]}, expected {PROPRIO_DIM}")
        own[..., slices["own_qpos"]] = p[..., :QPOS_DIM]
        own[..., slices["own_qvel"]] = p[..., QPOS_DIM:]
    if mode == "none":
        return own
    partner = _partner_of(role)
    if mode == "oracle":
        key = partner_primitive_key(role)
        if info is None or key not in info:
            raise ValueError(f"partner mode 'oracle' needs info[{key!r}] of the observed step")
        ids = np.asarray(info[key]).astype(np.int64)
        onehot = _partner_onehot(ids, num_primitives(partner), own.shape[:-1])
        partner_progress = own_block_slices(num_primitives(partner))["own_primitive_progress"]
        progress = extract(shared_vec, partner)[..., partner_progress]
        return np.concatenate([own, onehot, progress], axis=-1)
    # "predicted" / "predicted-direct": zero stub of the contract width; the real latent /
    # classifier output arrives with the spring integration, the width must already match.
    stub = agent_obs_dim(mode, role) - own.shape[-1]
    return np.concatenate([own, np.zeros((*own.shape[:-1], stub), dtype=np.float32)], axis=-1)


class SkillsObsBuilder:
    """Cuts the skills env's flat observation into per-agent torch policy inputs.

    The torch counterpart of `build_agent_obs` for the trainer's rollout: `from_env_obs` reads
    the field layout off the env's structured observation once (verifying it against a real
    flat observation via `_agent_obs.check_layout`), `__call__` then slices every later flat
    observation (including the vector env's `final_observation`).

    Layout expectations (the `_skills_obs` contract): `agent/<uid>/{qpos, qvel}` carries each
    agent's proprioception, and ALL remaining fields form one contiguous run of width
    `shared_vec_size()` -- the shared skills vector, whatever field names the env splits it
    into. Anything else fails loudly at start-up.

    Args:
        layout: `[(path, width), ...]` of the flat observation (`_agent_obs.obs_layout`).
        mode: partner-visibility mode (`PARTNER_VISIBILITY_MODES`).
        agent_uids: `(agent_a_uid, agent_b_uid)`; agent_a is the holder.

    Attributes:
        fields: per agent, the `/`-joined names of its input's fields, in order.
        obs_dims: per agent, the input width (`agent_obs_dim`).
    """

    def __init__(self, layout: Layout, mode: str, agent_uids: Sequence[str]) -> None:
        validate_partner_visibility(mode)
        self.layout = list(layout)
        self.mode = mode
        self.agent_uids = tuple(agent_uids)
        offsets = [0]
        for _, width in self.layout:
            offsets.append(offsets[-1] + width)
        self.total_dim = offsets[-1]

        proprio: list[dict[str, slice | None]] = [
            {"qpos": None, "qvel": None} for _ in self.agent_uids
        ]
        shared_indices: list[int] = []
        for i, (path, width) in enumerate(self.layout):
            if path[:1] == ("agent",):
                if (
                    len(path) != 3
                    or path[1] not in self.agent_uids
                    or path[2] not in ("qpos", "qvel")
                ):
                    raise ValueError(f"unexpected agent observation field {'/'.join(path)}")
                expected = QPOS_DIM if path[2] == "qpos" else QVEL_DIM
                if width != expected:
                    raise ValueError(f"{'/'.join(path)} has width {width}, expected {expected}")
                proprio[self.agent_uids.index(path[1])][path[2]] = slice(offsets[i], offsets[i + 1])
            else:
                shared_indices.append(i)
        for uid, fields in zip(self.agent_uids, proprio, strict=True):
            if fields["qpos"] is None or fields["qvel"] is None:
                raise ValueError(f"missing agent/{uid}/qpos or qvel in the skills observation")

        shared_width = sum(self.layout[i][1] for i in shared_indices)
        contiguous = shared_indices == list(range(shared_indices[0], shared_indices[-1] + 1))
        if not shared_indices or not contiguous or shared_width != shared_vec_size():
            raise ValueError(
                "the non-agent observation fields must form one contiguous run of width "
                f"{shared_vec_size()} (the shared skills vector), got width {shared_width} over "
                f"{len(shared_indices)} non-contiguous fields"
            )
        self._shared_start = offsets[shared_indices[0]]

        starts = _own_offsets()
        partners = [_partner_of(role) for role in ROLES]
        self._own_starts = [starts[role] for role in ROLES]
        self._own_sizes = [own_block_size(num_primitives(role)) for role in ROLES]
        self._own_slices = [own_block_slices(num_primitives(role)) for role in ROLES]
        self._n_partner = [num_primitives(partner) for partner in partners]
        self._partner_progress = [
            slice(
                starts[partner] + own_block_slices(n)["own_primitive_progress"].start,
                starts[partner] + own_block_slices(n)["own_primitive_progress"].stop,
            )
            for partner, n in zip(partners, self._n_partner, strict=True)
        ]
        self._proprio = proprio
        self.obs_dims = [agent_obs_dim(mode, role) for role in ROLES]
        partner_names = {
            "none": [],
            "oracle": ["partner/primitive_onehot", "partner/primitive_progress"],
            "predicted": ["partner/latent_stub"],
            "predicted-direct": ["partner/distribution_stub"],
        }[mode]
        self.fields = [
            [f"own/{name}" for name, _ in own_block_fields(num_primitives(role))] + partner_names
            for role in ROLES
        ]

    @classmethod
    def from_env_obs(
        cls,
        structured: Mapping,
        flat_obs,
        mode: str,
        agent_uids: Sequence[str],
    ) -> SkillsObsBuilder:
        """Builder for a skills env, verified against one of its real flat observations."""
        layout = obs_layout(structured)
        check_layout(layout, flat_obs, structured)
        return cls(layout, mode, agent_uids)

    def __call__(self, flat_obs, infos: Mapping | None = None) -> list:
        """`[obs_agent_a, obs_agent_b]`, each `(batch, obs_dim)`, from a `(batch, D)` tensor.

        `infos` is the step/reset info dict of the SAME observation (only `mode="oracle"`
        reads it, key `partner_primitive_key` of the partner role). It may carry a per-env
        leading dim larger than the observation batch only if the caller already masked it
        (the trainer masks for the `final_observation` bootstrap).
        """
        import torch

        if flat_obs.shape[-1] != self.total_dim:
            raise ValueError(
                f"flat observation has {flat_obs.shape[-1]} columns, expected {self.total_dim}"
            )
        batch = flat_obs.shape[0]
        shared = flat_obs[:, self._shared_start : self._shared_start + shared_vec_size()]
        inputs = []
        for i, role in enumerate(ROLES):
            own = shared[:, self._own_starts[i] : self._own_starts[i] + self._own_sizes[i]].clone()
            qpos, qvel = self._proprio[i]["qpos"], self._proprio[i]["qvel"]
            assert qpos is not None and qvel is not None  # checked in __init__
            own[:, self._own_slices[i]["own_qpos"]] = flat_obs[:, qpos]
            own[:, self._own_slices[i]["own_qvel"]] = flat_obs[:, qvel]
            if self.mode == "oracle":
                key = partner_primitive_key(role)
                if infos is None or key not in infos:
                    raise ValueError(
                        f"partner mode 'oracle' needs info[{key!r}] of the observed step"
                    )
                ids = torch.as_tensor(infos[key], device=flat_obs.device).to(torch.int64).view(-1)
                if ids.shape[0] != batch:
                    raise ValueError(
                        f"info[{key!r}] has {ids.shape[0]} entries but the observation batch is "
                        f"{batch} (mask the infos of bootstrap subsets)"
                    )
                if bool((ids < 0).any() or (ids >= self._n_partner[i]).any()):
                    raise ValueError(
                        f"partner primitive ids must be in [0, {self._n_partner[i]}), got {ids.tolist()}"
                    )
                onehot = torch.nn.functional.one_hot(ids, self._n_partner[i]).to(own.dtype)
                progress = shared[:, self._partner_progress[i]]
                inputs.append(torch.cat([own, onehot, progress], dim=-1))
            elif self.mode == "none":
                inputs.append(own)
            else:  # "predicted" / "predicted-direct": zero stub of the contract width
                stub = self.obs_dims[i] - own.shape[-1]
                inputs.append(
                    torch.cat(
                        [own, torch.zeros(batch, stub, dtype=own.dtype, device=own.device)], dim=-1
                    )
                )
        return inputs
