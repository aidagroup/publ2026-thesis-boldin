"""Unit tests for callosum.training._agent_obs (step 2.1) -- pure PyTorch, no
mani_skill dependency, so runnable locally once the `train` extra is synced
(`make dev`). Skipped (not failed) in CI, which installs only `dev` per
.github/workflows/ci.yml -- torch is locked to the multi-GB `cu128` index on
any Linux box regardless of which extra requests it, so CI deliberately
never installs it.
"""

import pytest

torch = pytest.importorskip("torch")

from callosum.training._agent_obs import (
    PARTNER_INPUT_MODES,
    build_agent_obs,
    build_policy_input,
    flatten_dict_to_tensor,
    select_agent_extra_fields,
)

AGENT_UIDS = ("so100-0", "so100-1")


def _dummy_raw_obs(num_envs: int = 4, partner_obs: str = "full") -> dict:
    agent = {
        "so100-0": {"qpos": torch.randn(num_envs, 6), "qvel": torch.randn(num_envs, 6)},
        "so100-1": {"qpos": torch.randn(num_envs, 6), "qvel": torch.randn(num_envs, 6)},
    }
    extra = {"cube_pose": torch.randn(num_envs, 7)}
    if partner_obs == "full":
        extra["agent_a_tcp_pose"] = torch.randn(num_envs, 7)
        extra["agent_b_tcp_pose"] = torch.randn(num_envs, 7)
    return {"agent": agent, "extra": extra}


def test_select_agent_extra_fields_drops_only_other_agent() -> None:
    extra = {"agent_a_tcp_pose": "a", "agent_b_tcp_pose": "b", "cube_pose": "cube"}
    assert select_agent_extra_fields(extra, agent_idx=0) == {
        "agent_a_tcp_pose": "a",
        "cube_pose": "cube",
    }
    assert select_agent_extra_fields(extra, agent_idx=1) == {
        "agent_b_tcp_pose": "b",
        "cube_pose": "cube",
    }


def test_select_agent_extra_fields_no_prefixed_fields_untouched() -> None:
    extra = {"cube_pose": "cube", "face_angle": "angle"}
    assert select_agent_extra_fields(extra, agent_idx=0) == extra
    assert select_agent_extra_fields(extra, agent_idx=1) == extra


def test_select_agent_extra_fields_include_partner_keeps_everything() -> None:
    extra = {"agent_a_tcp_pose": "a", "agent_b_tcp_pose": "b", "cube_pose": "cube"}
    assert select_agent_extra_fields(extra, agent_idx=0, include_partner=True) == extra
    assert select_agent_extra_fields(extra, agent_idx=1, include_partner=True) == extra


def test_flatten_dict_to_tensor_concatenates_last_dim() -> None:
    fields = {"a": torch.zeros(3, 2), "b": torch.ones(3, 4)}
    flat = flatten_dict_to_tensor(fields)
    assert flat.shape == (3, 6)
    assert torch.all(flat[:, :2] == 0)
    assert torch.all(flat[:, 2:] == 1)


def test_flatten_dict_to_tensor_recurses_into_nested_dict() -> None:
    fields = {"proprio": {"qpos": torch.zeros(2, 6), "qvel": torch.ones(2, 6)}}
    flat = flatten_dict_to_tensor(fields)
    assert flat.shape == (2, 12)


def test_build_agent_obs_shape_same_for_both_agents() -> None:
    raw_obs = _dummy_raw_obs(num_envs=5, partner_obs="full")
    obs_a = build_agent_obs(raw_obs, 0, AGENT_UIDS)
    obs_b = build_agent_obs(raw_obs, 1, AGENT_UIDS)
    assert obs_a.shape[0] == 5
    assert obs_a.shape == obs_b.shape


def test_build_agent_obs_shape_shrinks_when_partner_obs_is_none() -> None:
    full_obs = _dummy_raw_obs(num_envs=3, partner_obs="full")
    none_obs = _dummy_raw_obs(num_envs=3, partner_obs="none")
    dim_full = build_agent_obs(full_obs, 0, AGENT_UIDS).shape[-1]
    dim_none = build_agent_obs(none_obs, 0, AGENT_UIDS).shape[-1]
    # For agent_a, select_agent_extra_fields only ever excludes
    # agent_b_tcp_pose (the *other* agent's field) -- agent_a_tcp_pose is
    # kept under "full". So the only field that disappears going to "none"
    # is agent_a_tcp_pose itself (7), since the env stops producing *both*
    # tcp-pose fields entirely under partner_obs="none" (step 1.4: there is
    # no per-agent split at the env level, so "none" drops even the agent's
    # own tcp pose, not just the partner's -- see callosum.envs._partner_obs).
    assert dim_full == dim_none + 7


def test_build_agent_obs_excludes_partner_tcp_pose_values() -> None:
    raw_obs = _dummy_raw_obs(num_envs=2, partner_obs="full")
    obs_a = build_agent_obs(raw_obs, 0, AGENT_UIDS)
    # agent_a's own proprio (12) + agent_a_tcp_pose (7) + cube_pose (7) = 26;
    # agent_b_tcp_pose must NOT be present.
    assert obs_a.shape[-1] == 6 + 6 + 7 + 7


def test_build_agent_obs_default_is_decentralized() -> None:
    # include_partner defaults to False -- must match passing it explicitly.
    raw_obs = _dummy_raw_obs(num_envs=2, partner_obs="full")
    default_call = build_agent_obs(raw_obs, 0, AGENT_UIDS)
    explicit_false = build_agent_obs(raw_obs, 0, AGENT_UIDS, include_partner=False)
    assert default_call.shape == explicit_false.shape


def test_build_agent_obs_include_partner_is_strictly_wider_by_partner_field_width() -> None:
    raw_obs = _dummy_raw_obs(num_envs=3, partner_obs="full")
    dim_default = build_agent_obs(raw_obs, 0, AGENT_UIDS).shape[-1]
    dim_oracle = build_agent_obs(raw_obs, 0, AGENT_UIDS, include_partner=True).shape[-1]
    partner_field_width = raw_obs["extra"]["agent_b_tcp_pose"].shape[-1]
    assert dim_oracle > dim_default
    assert dim_oracle == dim_default + partner_field_width


# --- Step 3.2: the partner_input ablation toggle (build_policy_input) ---

_PARTNER_LATENT_DIM = 8


def test_build_policy_input_shape_stable_across_modes() -> None:
    """The SAME policy network must serve all three modes -> identical dim."""
    raw_obs = _dummy_raw_obs(num_envs=3, partner_obs="full")
    partner_latent = torch.full((3, _PARTNER_LATENT_DIM), 5.0)
    dims = {
        mode: build_policy_input(
            raw_obs,
            0,
            AGENT_UIDS,
            partner_latent if mode != "none" else None,
            mode,
            partner_latent_dim=_PARTNER_LATENT_DIM,
        ).shape[-1]
        for mode in PARTNER_INPUT_MODES
    }
    assert len(set(dims.values())) == 1, dims


def test_build_policy_input_none_excludes_partner_info() -> None:
    """Scientifically critical (method doc §Фаза 1): under 'none', agent_a's
    input must contain NO partner (agent_b) information -- neither the raw
    partner TCP pose nor a nonzero latent slot."""
    raw_obs = _dummy_raw_obs(num_envs=3, partner_obs="full")
    # Make agent_b's TCP pose unmistakably unique so any leak is detectable.
    raw_obs["extra"]["agent_b_tcp_pose"] = torch.full((3, 7), 777.0)

    none_input = build_policy_input(
        raw_obs, 0, AGENT_UIDS, None, "none", partner_latent_dim=_PARTNER_LATENT_DIM
    )
    # No cell may carry the partner's distinctive value.
    assert (none_input == 777.0).sum() == 0
    # And the appended latent slot must be all zeros.
    assert torch.all(none_input[:, -_PARTNER_LATENT_DIM:] == 0)
    # Sanity: the partner field WAS present in the raw obs (we didn't test a
    # no-op): the base decentralized input still includes agent_a's OWN pose.
    assert (raw_obs["extra"]["agent_a_tcp_pose"] != 777.0).any()


def test_build_policy_input_oracle_appends_true_latent() -> None:
    raw_obs = _dummy_raw_obs(num_envs=2, partner_obs="full")
    latent = torch.full((2, _PARTNER_LATENT_DIM), 3.0)
    out = build_policy_input(raw_obs, 0, AGENT_UIDS, latent, "oracle", _PARTNER_LATENT_DIM)
    assert out.shape[-1] == build_agent_obs(raw_obs, 0, AGENT_UIDS).shape[-1] + _PARTNER_LATENT_DIM
    assert torch.all(out[:, -_PARTNER_LATENT_DIM:] == 3.0)


def test_build_policy_input_predicted_appends_prediction() -> None:
    raw_obs = _dummy_raw_obs(num_envs=2, partner_obs="full")
    latent = torch.arange(_PARTNER_LATENT_DIM, dtype=torch.float32).repeat(2, 1)
    out = build_policy_input(raw_obs, 1, AGENT_UIDS, latent, "predicted", _PARTNER_LATENT_DIM)
    assert torch.all(out[:, -_PARTNER_LATENT_DIM:] == latent)


def test_build_policy_input_rejects_bad_mode() -> None:
    raw_obs = _dummy_raw_obs(num_envs=2, partner_obs="full")
    with pytest.raises(ValueError):
        build_policy_input(raw_obs, 0, AGENT_UIDS, None, "maybe", _PARTNER_LATENT_DIM)


def test_build_policy_input_none_requires_latent_dim() -> None:
    raw_obs = _dummy_raw_obs(num_envs=2, partner_obs="full")
    with pytest.raises(ValueError):
        build_policy_input(raw_obs, 0, AGENT_UIDS, None, "none", partner_latent_dim=0)


def test_build_policy_input_oracle_without_latent_raises() -> None:
    raw_obs = _dummy_raw_obs(num_envs=2, partner_obs="full")
    with pytest.raises(ValueError):
        build_policy_input(raw_obs, 0, AGENT_UIDS, None, "oracle", _PARTNER_LATENT_DIM)
