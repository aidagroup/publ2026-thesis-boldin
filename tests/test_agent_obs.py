"""Per-agent policy inputs of the IPPO trainer (step 2.1): shapes, content, partner leakage.

`callosum.training._agent_obs` has no mani_skill dependency and its index logic no torch
dependency, so most of this runs in CI (numpy only); the tests that slice real tensors are
skipped when torch is absent. The structured observations mirror what the real envs return
for `obs_mode="state"` (checked on the Mac CPU sim): per-uid `qpos` (7) / `qvel` (7), `extra`
with the TCP poses (only for `partner_obs="full"`), `cube_pose` (7) and, for FaceTurn,
`face_angle` (a `(n,)` leaf) and `face_pose` (7).
"""

import numpy as np
import pytest

from callosum.training._agent_obs import (
    AGENT_UIDS,
    AgentObsBuilder,
    obs_layout,
    select_fields,
)

N = 3  # batch size
UID_A, UID_B = AGENT_UIDS


def make_obs(partner_obs: str, face_turn: bool = True, seed: int = 0) -> dict:
    """A structured observation shaped like the env's, with random content."""
    rng = np.random.default_rng(seed)

    def rnd(*shape: int) -> np.ndarray:
        return rng.normal(size=(N, *shape)).astype(np.float32)

    extra: dict = {}
    if partner_obs == "full":
        extra["agent_a_tcp_pose"] = rnd(7)
        extra["agent_b_tcp_pose"] = rnd(7)
    extra["cube_pose"] = rnd(7)
    if face_turn:
        extra["face_angle"] = rnd()
        extra["face_pose"] = rnd(7)
    return {
        "agent": {
            UID_A: {"qpos": rnd(7), "qvel": rnd(7)},
            UID_B: {"qpos": rnd(7), "qvel": rnd(7)},
        },
        "extra": extra,
    }


def flatten(structured: dict) -> np.ndarray:
    """ManiSkill's `flatten_state_dict`: leaves concatenated in insertion order."""
    parts = []

    def walk(node: dict) -> None:
        for value in node.values():
            if isinstance(value, dict):
                walk(value)
            else:
                parts.append(value.reshape(value.shape[0], -1))

    walk(structured)
    return np.concatenate(parts, axis=-1)


def build(partner_obs: str, face_turn: bool = True) -> AgentObsBuilder:
    return AgentObsBuilder(obs_layout(make_obs(partner_obs, face_turn)), partner_obs, AGENT_UIDS)


def column_ranges(layout: list, prefix: str) -> set[int]:
    """Flat columns of all layout fields whose joined path starts with `prefix`."""
    columns: set[int] = set()
    start = 0
    for path, width in layout:
        if "/".join(path).startswith(prefix):
            columns |= set(range(start, start + width))
        start += width
    return columns


def test_layout_matches_env_widths() -> None:
    # 49 / 35 for TwoSO101-v0 and 57 / 43 for FaceTurn-v0 are the flat sizes the real envs give.
    sizes = {
        (po, ft): sum(w for _, w in obs_layout(make_obs(po, ft)))
        for po in ("full", "none")
        for ft in (False, True)
    }
    assert sizes == {
        ("full", False): 49,
        ("none", False): 35,
        ("full", True): 57,
        ("none", True): 43,
    }
    for (po, ft), size in sizes.items():
        assert flatten(make_obs(po, ft)).shape == (N, size)


def test_input_dims() -> None:
    # own qpos+qvel (14) + cube (7) [+ face_angle, face_pose (8)] [+ both TCP poses (14)]
    expected = {("full", False): 35, ("none", False): 21, ("full", True): 43, ("none", True): 29}
    for (partner_obs, face_turn), dim in expected.items():
        assert build(partner_obs, face_turn).obs_dims == [dim, dim]


def test_none_fields_are_exactly_own_state_and_task_state() -> None:
    builder = build("none")
    shared = ["extra/cube_pose", "extra/face_angle", "extra/face_pose"]
    assert builder.fields[0] == [f"agent/{UID_A}/qpos", f"agent/{UID_A}/qvel", *shared]
    assert builder.fields[1] == [f"agent/{UID_B}/qpos", f"agent/{UID_B}/qvel", *shared]


def test_full_fields_add_both_tcp_poses_but_no_partner_joint_state() -> None:
    builder = build("full")
    for agent_idx in (0, 1):
        fields = builder.fields[agent_idx]
        assert "extra/agent_a_tcp_pose" in fields and "extra/agent_b_tcp_pose" in fields
        assert not any(AGENT_UIDS[1 - agent_idx] in name for name in fields)
        assert len(fields) == 7


@pytest.mark.parametrize("agent_idx", [0, 1])
def test_no_partner_leakage_when_none(agent_idx: int) -> None:
    """With partner_obs="none" no column of the partner's state can reach the input."""
    partner = "ab"[1 - agent_idx]
    partner_uid = AGENT_UIDS[1 - agent_idx]
    # (1) The env's own "none" layout has no TCP poses at all; the partner's joint state is out.
    layout = obs_layout(make_obs("none"))
    columns = set(build("none").columns[agent_idx])
    assert not columns & column_ranges(layout, f"agent/{partner_uid}/")

    # (2) The trainer's "none" also holds on an env that does emit TCP poses ("full"): the
    # partner's TCP pose is dropped, the agent's own one is kept.
    layout = obs_layout(make_obs("full"))
    columns = set(AgentObsBuilder(layout, "none", AGENT_UIDS).columns[agent_idx])
    assert not columns & column_ranges(layout, f"agent/{partner_uid}/")
    assert not columns & column_ranges(layout, f"extra/agent_{partner}_")
    assert columns & column_ranges(layout, f"extra/agent_{'ab'[agent_idx]}_tcp_pose")


@pytest.mark.parametrize("partner_obs", ["full", "none"])
def test_changing_partner_state_does_not_change_input(partner_obs: str) -> None:
    """Value-level check: overwrite the partner's joint state, the input stays identical."""
    base = make_obs(partner_obs, seed=1)
    builder = build(partner_obs)
    for agent_idx in (0, 1):
        reference = flatten(base)[:, builder.columns[agent_idx]]
        changed = make_obs(partner_obs, seed=1)
        changed["agent"][AGENT_UIDS[1 - agent_idx]] = {
            "qpos": np.full((N, 7), 123.0, dtype=np.float32),
            "qvel": np.full((N, 7), -456.0, dtype=np.float32),
        }
        assert np.array_equal(flatten(changed)[:, builder.columns[agent_idx]], reference)


def test_none_hides_partner_tcp_value_even_if_present() -> None:
    base = make_obs("full", seed=2)
    builder = AgentObsBuilder(obs_layout(base), "none", AGENT_UIDS)
    changed = make_obs("full", seed=2)
    changed["extra"]["agent_b_tcp_pose"] = np.full((N, 7), 9.0, dtype=np.float32)
    assert np.array_equal(
        flatten(changed)[:, builder.columns[0]], flatten(base)[:, builder.columns[0]]
    )
    changed = make_obs("full", seed=2)
    changed["extra"]["agent_a_tcp_pose"] = np.full((N, 7), 9.0, dtype=np.float32)
    assert np.array_equal(
        flatten(changed)[:, builder.columns[1]], flatten(base)[:, builder.columns[1]]
    )


def test_unknown_fields_are_not_silently_shared() -> None:
    layout = obs_layout(make_obs("none"))
    layout.append((("sensor_data", "rgb"), 4))
    with pytest.raises(ValueError, match="unexpected observation group"):
        select_fields(layout, 0, "none")
    layout = obs_layout(make_obs("none"))
    layout.append((("agent", "controller", "x"), 1))
    with pytest.raises(ValueError, match="unexpected agent observation field"):
        select_fields(layout, 0, "none")


def test_invalid_partner_obs_and_agent_index() -> None:
    layout = obs_layout(make_obs("none"))
    with pytest.raises(ValueError, match="partner_obs"):
        select_fields(layout, 0, "predicted")
    with pytest.raises(ValueError, match="agent_idx"):
        select_fields(layout, 2, "none")


# --- Tensor slicing (needs torch; skipped in CI, which installs only the dev extra) ----------


def test_builder_slices_tensors_like_the_numpy_reference() -> None:
    torch = pytest.importorskip("torch")
    for partner_obs in ("full", "none"):
        structured = make_obs(partner_obs)
        builder = build(partner_obs)
        flat = torch.from_numpy(flatten(structured))
        out = builder(flat)
        assert [tuple(o.shape) for o in out] == [(N, d) for d in builder.obs_dims]
        for agent_idx in (0, 1):
            assert np.array_equal(
                out[agent_idx].numpy(), flatten(structured)[:, builder.columns[agent_idx]]
            )
    with pytest.raises(ValueError, match="columns"):
        builder(torch.zeros(N, 5))


def test_none_input_content_is_own_proprio_then_shared_state() -> None:
    torch = pytest.importorskip("torch")
    structured = make_obs("none")
    out_a, out_b = build("none")(torch.from_numpy(flatten(structured)))
    shared = np.concatenate(
        [
            structured["extra"]["cube_pose"],
            structured["extra"]["face_angle"][:, None],
            structured["extra"]["face_pose"],
        ],
        axis=-1,
    )
    own_a = np.concatenate(
        [structured["agent"][UID_A]["qpos"], structured["agent"][UID_A]["qvel"]], axis=-1
    )
    assert np.array_equal(out_a.numpy(), np.concatenate([own_a, shared], axis=-1))
    assert np.array_equal(out_b.numpy()[:, 14:], shared)


def test_check_layout_catches_wrong_order_and_size() -> None:
    torch = pytest.importorskip("torch")
    from callosum.training._agent_obs import check_layout, flatten_structured

    structured = {k: v for k, v in make_obs("full").items()}
    tensors = {
        "agent": {
            u: {k: torch.from_numpy(v) for k, v in d.items()}
            for u, d in structured["agent"].items()
        },
        "extra": {k: torch.from_numpy(v) for k, v in structured["extra"].items()},
    }
    flat = flatten_structured(tensors)
    assert torch.equal(flat, torch.from_numpy(flatten(structured)))
    layout = obs_layout(tensors)
    check_layout(layout, flat, tensors)
    with pytest.raises(ValueError, match="columns"):
        check_layout(layout, flat[:, :-1], tensors)
    shuffled = torch.cat([flat[:, 7:], flat[:, :7]], dim=-1)  # same size, different order
    with pytest.raises(ValueError, match="flattening order"):
        check_layout(layout, shuffled, tensors)
    AgentObsBuilder.from_env_obs(tensors, flat, "full")
