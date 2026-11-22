"""Pure-logic tests of the OccludedToss-v0 observation mask (`TossConfig.occlusion`).

The hidden-partner variant (`PLAN.md` "Hidden-partner regime") hides the PARTNER paddle's
pose/velocity/tilt from the shared observation block while keeping the ball and the agent's
OWN paddle, its private parameters/action/goal/time and the privileged `oracle` partner
block. The mask lives in `callosum.experiments.coop_tasks.logic` (the spec's
`shared_field_owners` + `shared_fields_for`/`shared_field_indices`), not in the simulator
code, so it is testable here without mujoco/torch: width, field order, the exact
occluded-vs-full column mapping and, most importantly, that the hidden partner-paddle values
cannot change the returned vector (the leakage sentinel).

The CPU/GPU env integration tests are added after the toss-physics rework lands, because the
simulator files are owned by the physics rewrite at this step; these tests cover the pure
contract both backends have to follow.
"""

import importlib.util

import numpy as np
import pytest

from callosum.experiments.coop_tasks.config import (
    TossConfig,
    coop_config,
    coop_obs_size,
)
from callosum.experiments.coop_tasks.logic import (
    AGENT_KEYS,
    CARTPOLE_SPEC,
    OCCLUSION_MODES,
    PENDULUM_SPEC,
    TOSS_SPEC,
    CoopTaskSpec,
    build_observation,
    obs_fields,
    obs_size,
    shared_field_indices,
    shared_fields_for,
    task_obs_size,
    validate_occlusion,
)

_TOSS_BALL_FIELDS = ("ball_x", "ball_z", "ball_vx", "ball_vz")
_TOSS_LEFT_PAD_FIELDS = (
    "left_pad_x",
    "left_pad_z",
    "left_pad_tilt",
    "left_pad_vx",
    "left_pad_vz",
    "left_pad_tilt_rate",
)
_TOSS_RIGHT_PAD_FIELDS = tuple(field.replace("left_", "right_") for field in _TOSS_LEFT_PAD_FIELDS)

_TOSS_OBS_NONE = 26
"""Pre-occlusion toss observation width under `partner_mode="none"`."""
_TOSS_OBS_ORACLE = 36
"""Pre-occlusion toss observation width under `partner_mode="oracle"`."""
_PARTNER_BLOCK_WIDTH = 2 * 3 + 4
"""Width of the privileged partner block (`action_dim * 2 + 4`)."""


def _toss_values() -> dict[str, float]:
    """A full role-tagged values dict of the toss task (every key `build_observation` reads)."""
    values: dict[str, float] = {}
    for role_index, role in enumerate(AGENT_KEYS):
        for index in range(TOSS_SPEC.action_dim):
            values[f"{role}_action_{index}"] = 0.05 * role_index + 0.01 * index
            values[f"{role}_command_{index}"] = 0.10 * role_index + 0.01 * index
        values[f"{role}_latency"] = 0.05 + 0.01 * role_index
        values[f"{role}_gain"] = 1.0 + 0.1 * role_index
        values[f"{role}_limit_scale"] = 1.0 - 0.1 * role_index
        values[f"{role}_start_delay"] = 0.02 * role_index
    for offset, field_name in enumerate(TOSS_SPEC.shared_fields):
        values[field_name] = 0.5 + 0.25 * offset
    values["passes"] = 1.0
    values["target_passes"] = 3.0
    values["time_fraction"] = 0.25
    return values


def _tamper(values: dict[str, float], fields: tuple[str, ...]) -> dict[str, float]:
    """A copy of `values` with all `fields` changed by a large offset."""
    tampered = dict(values)
    for field_name in fields:
        tampered[field_name] += 1000.0
    return tampered


# --- config flag --------------------------------------------------------------------------


def test_toss_config_occlusion_defaults_to_none_and_validates() -> None:
    """The flag is `"none"` by default (the pre-regime layout) and rejects unknown modes."""
    assert TossConfig().occlusion == "none"
    assert TossConfig(occlusion="partner").occlusion == "partner"
    assert coop_config("toss", occlusion="partner").occlusion == "partner"
    for bad in ("predicted", "oracle", "partner ", ""):
        with pytest.raises(ValueError, match="occlusion"):
            TossConfig(occlusion=bad)
    assert set(OCCLUSION_MODES) == {"none", "partner"}


def test_configs_without_the_flag_cannot_be_occluded() -> None:
    """Only the toss config owns the flag; the other tasks reject it as an unknown field."""
    with pytest.raises(TypeError):
        coop_config("cartpole", occlusion="partner")
    with pytest.raises(TypeError):
        coop_config("pendulum", occlusion="partner")


def test_validate_occlusion_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="occlusion"):
        validate_occlusion("partner-paddle")


def test_validate_occlusion_returns_the_mode_for_the_env_assignment_pattern() -> None:
    """Both env backends write `self.occlusion = validate_occlusion(config.occlusion)`.

    The validator therefore has to return the mode (not None), otherwise the env's
    `self.occlusion` becomes None and the next `obs_size(..., occlusion=None)` raises.
    """
    for mode in OCCLUSION_MODES:
        assert validate_occlusion(mode) == mode


# --- widths -------------------------------------------------------------------------------


def test_occluded_toss_widths_drop_only_the_partner_paddle() -> None:
    """16 shared fields -> 10 (ball + own paddle); 26/36 -> 20/30 observations."""
    assert coop_obs_size("toss", "none") == _TOSS_OBS_NONE
    assert coop_obs_size("toss", "oracle") == _TOSS_OBS_ORACLE
    assert coop_obs_size("toss", "none", occlusion="partner") == _TOSS_OBS_NONE - 6
    assert coop_obs_size("toss", "oracle", occlusion="partner") == _TOSS_OBS_ORACLE - 6
    assert obs_size(TOSS_SPEC, "none", occlusion="partner") == 20
    assert obs_size(TOSS_SPEC, "oracle", occlusion="partner") == 30
    assert task_obs_size("toss", "none", occlusion="partner") == 20
    assert task_obs_size("toss", "oracle", occlusion="partner") == 30


def test_occluded_width_is_role_independent() -> None:
    """The mask is symmetric: each agent hides exactly the partner's six shared fields."""
    assert len(shared_fields_for(TOSS_SPEC, "left", "partner")) == 10
    assert len(shared_fields_for(TOSS_SPEC, "right", "partner")) == 10
    assert len(obs_fields(TOSS_SPEC, "none", agent="left", occlusion="partner")) == 20
    assert len(obs_fields(TOSS_SPEC, "none", agent="right", occlusion="partner")) == 20


def test_other_tasks_are_unaffected_by_occlusion() -> None:
    """The one-sided tasks have no per-agent shared fields, so the mask is a no-op."""
    for spec in (CARTPOLE_SPEC, PENDULUM_SPEC):
        for mode in ("none", "oracle"):
            assert obs_size(spec, mode, occlusion="partner") == obs_size(spec, mode)
            for agent in AGENT_KEYS:
                assert shared_fields_for(spec, agent, "partner") == spec.shared_fields
                assert shared_field_indices(spec, agent, "partner") == tuple(
                    range(len(spec.shared_fields))
                )


# --- field mask / order -------------------------------------------------------------------


def test_occluded_shared_fields_keep_the_ball_and_the_own_paddle() -> None:
    assert shared_fields_for(TOSS_SPEC, "left", "partner") == (
        _TOSS_BALL_FIELDS + _TOSS_LEFT_PAD_FIELDS
    )
    assert shared_fields_for(TOSS_SPEC, "right", "partner") == (
        _TOSS_BALL_FIELDS + _TOSS_RIGHT_PAD_FIELDS
    )
    assert shared_fields_for(TOSS_SPEC, "left", "none") == TOSS_SPEC.shared_fields


def test_obs_fields_hide_only_the_partner_paddle() -> None:
    left = obs_fields(TOSS_SPEC, "none", agent="left", occlusion="partner")
    right = obs_fields(TOSS_SPEC, "none", agent="right", occlusion="partner")
    for fields, own_prefix, partner_prefix in (
        (left, "left_pad_", "right_pad_"),
        (right, "right_pad_", "left_pad_"),
    ):
        assert not any(field.startswith(partner_prefix) for field in fields)
        assert sum(1 for field in fields if field.startswith(own_prefix)) == 6
        assert all(field in fields for field in _TOSS_BALL_FIELDS)
        # The own/params/goal/time blocks are untouched by the mask.
        assert fields[: 3 + 3 + 1] == obs_fields(TOSS_SPEC, "none")[: 3 + 3 + 1]
        assert fields[-3:] == (TOSS_SPEC.goal_fields + ("time_fraction",))
    assert len(left) == len(right) == 20


def test_shared_field_indices_match_the_masked_fields() -> None:
    """The batched-backend column slice is exactly the visible sub-sequence of the block."""
    assert shared_field_indices(TOSS_SPEC, "left", "none") == tuple(range(16))
    assert shared_field_indices(TOSS_SPEC, "left", "partner") == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert shared_field_indices(TOSS_SPEC, "right", "partner") == (
        0,
        1,
        2,
        3,
        10,
        11,
        12,
        13,
        14,
        15,
    )
    for agent in AGENT_KEYS:
        fields = shared_fields_for(TOSS_SPEC, agent, "partner")
        assert (
            tuple(
                TOSS_SPEC.shared_fields[index]
                for index in shared_field_indices(TOSS_SPEC, agent, "partner")
            )
            == fields
        )


# --- observations: exact column mapping and the leakage sentinel ---------------------------


def test_occluded_observation_is_the_full_one_with_the_partner_paddle_removed() -> None:
    """Exact mapping: own block (0..6) + ball (7..10) + own pad + goal/time of the full vector.

    Under `partner_mode="none"` the full 26-wide vector is `own(0..6)`, `ball(7..10)`,
    `left pad(11..16)`, `right pad(17..22)`, `goal(23..24)`, `time(25)`.
    """
    values = _toss_values()
    full_left = build_observation(values, "left", TOSS_SPEC, "none")
    full_right = build_observation(values, "right", TOSS_SPEC, "none")
    occluded_left = build_observation(values, "left", TOSS_SPEC, "none", occlusion="partner")
    occluded_right = build_observation(values, "right", TOSS_SPEC, "none", occlusion="partner")
    left_columns = np.r_[0:17, 23:26]
    right_columns = np.r_[0:11, 17:26]
    np.testing.assert_array_equal(occluded_left, full_left[left_columns])
    np.testing.assert_array_equal(occluded_right, full_right[right_columns])


def test_partner_paddle_values_cannot_leak_into_the_occluded_observation() -> None:
    """Changing any hidden partner-paddle field must not move one bit of the vector."""
    values = _toss_values()
    for agent, hidden_fields in (
        ("left", _TOSS_RIGHT_PAD_FIELDS),
        ("right", _TOSS_LEFT_PAD_FIELDS),
    ):
        baseline = build_observation(values, agent, TOSS_SPEC, "none", occlusion="partner")
        tampered = build_observation(
            _tamper(values, hidden_fields), agent, TOSS_SPEC, "none", occlusion="partner"
        )
        np.testing.assert_array_equal(tampered, baseline)
    # The hidden fields still change the NON-occluded observation: the mask, not the env,
    # is what removes them.
    full_baseline = build_observation(values, "left", TOSS_SPEC, "none")
    full_tampered = build_observation(
        _tamper(values, _TOSS_RIGHT_PAD_FIELDS), "left", TOSS_SPEC, "none"
    )
    assert not np.array_equal(full_tampered, full_baseline)


def test_visible_fields_still_reach_the_occluded_observation() -> None:
    """The shared object and the agent's own paddle stay observable under occlusion."""
    values = _toss_values()
    left = build_observation(values, "left", TOSS_SPEC, "none", occlusion="partner")
    for fields in (
        _TOSS_BALL_FIELDS,
        _TOSS_LEFT_PAD_FIELDS,
        ("left_action_0", "left_gain", "left_start_delay", "passes", "time_fraction"),
    ):
        changed = build_observation(
            _tamper(values, fields), "left", TOSS_SPEC, "none", occlusion="partner"
        )
        assert not np.array_equal(changed, left), fields


def test_occluded_oracle_keeps_the_privileged_partner_block() -> None:
    """`oracle` under occlusion is `none`+partner block: partner paddle hidden, commands kept.

    The privileged partner block (the partner's applied/commanded actions, private
    parameters and start delay) is NOT masked, so `oracle` stays the privileged upper bound;
    only the partner paddle's pose/velocity/tilt disappears from the shared block.
    """
    values = _toss_values()
    occluded_none = obs_fields(TOSS_SPEC, "none", agent="left", occlusion="partner")
    occluded_oracle = obs_fields(TOSS_SPEC, "oracle", agent="left", occlusion="partner")
    assert len(occluded_oracle) == len(occluded_none) + _PARTNER_BLOCK_WIDTH
    assert occluded_oracle[: len(occluded_none)] == occluded_none
    assert "partner_command_0" in occluded_oracle and "partner_latency" in occluded_oracle
    assert not any(field.startswith("right_pad_") for field in occluded_oracle)

    left = build_observation(values, "left", TOSS_SPEC, "oracle", occlusion="partner")
    changed_command = build_observation(
        _tamper(values, ("right_command_0", "right_latency")),
        "left",
        TOSS_SPEC,
        "oracle",
        occlusion="partner",
    )
    assert not np.array_equal(changed_command, left)
    hidden = build_observation(
        _tamper(values, _TOSS_RIGHT_PAD_FIELDS), "left", TOSS_SPEC, "oracle", occlusion="partner"
    )
    np.testing.assert_array_equal(hidden, left)


# --- validation / spec invariants ---------------------------------------------------------


def test_build_observation_rejects_unknown_occlusion() -> None:
    values = _toss_values()
    with pytest.raises(ValueError, match="occlusion"):
        build_observation(values, "left", TOSS_SPEC, "none", occlusion="partner_only")
    with pytest.raises(ValueError, match="agent"):
        shared_fields_for(TOSS_SPEC, "middle")
    with pytest.raises(ValueError, match="agent"):
        build_observation(values, "middle", TOSS_SPEC, "none", occlusion="partner")


def test_spec_owner_mask_validation() -> None:
    """A mis-tagged owner mask must fail at spec construction, not silently leak/hide."""
    with pytest.raises(ValueError, match="shared_field_owners"):
        CoopTaskSpec(
            name="bad",
            action_dim=1,
            shared_fields=("a", "b"),
            goal_fields=(),
            shared_field_owners=("left",),
        )
    with pytest.raises(ValueError, match="owner"):
        CoopTaskSpec(
            name="bad",
            action_dim=1,
            shared_fields=("a", "b"),
            goal_fields=(),
            shared_field_owners=("left", "middle"),
        )
    with pytest.raises(ValueError, match="no field owned"):
        CoopTaskSpec(
            name="bad",
            action_dim=1,
            shared_fields=("a", "b"),
            goal_fields=(),
            shared_field_owners=(None, None),
        )
    with pytest.raises(ValueError, match="symmetric"):
        CoopTaskSpec(
            name="bad",
            action_dim=1,
            shared_fields=("a", "b", "c"),
            goal_fields=(),
            shared_field_owners=("left", "left", "right"),
        )
    valid = CoopTaskSpec(
        name="ok",
        action_dim=1,
        shared_fields=("a", "b", "c"),
        goal_fields=(),
        shared_field_owners=("left", None, "right"),
    )
    assert shared_fields_for(valid, "left", "partner") == ("a", "b")
    assert shared_fields_for(valid, "right", "partner") == ("b", "c")


def test_toss_spec_ownership_is_the_ball_and_the_paddles() -> None:
    """The spec tags ball fields as shared and each paddle's six fields as the agent's own."""
    owners = TOSS_SPEC.shared_field_owners
    assert len(owners) == len(TOSS_SPEC.shared_fields) == 16
    for field_name, owner in zip(TOSS_SPEC.shared_fields, owners):
        if field_name in _TOSS_BALL_FIELDS:
            assert owner is None
        elif field_name in _TOSS_LEFT_PAD_FIELDS:
            assert owner == "left"
        else:
            assert field_name in _TOSS_RIGHT_PAD_FIELDS and owner == "right"


# --- CPU env integration (guarded: needs mujoco, which CI does not install) ----------------


def _make_toss_env(occlusion: str = "none", partner_mode: str = "none"):
    """A CPU toss env of the requested observation variant (skips without mujoco)."""
    pytest.importorskip("mujoco")
    from callosum.experiments.coop_tasks import make_coop_env

    return make_coop_env("toss", config=TossConfig(occlusion=occlusion), partner_mode=partner_mode)


def _numpy_obs(obs: dict) -> dict[str, np.ndarray]:
    """The step/reset observation dict as float32 numpy arrays (gymnasium boxes)."""
    return {role: np.asarray(value, dtype=np.float32) for role, value in obs.items()}


_TOSS_ENV_ERRORS = (
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
"""Errors a mid-rework toss env pair can raise while the XML and the env class disagree."""


def _toss_env_ready() -> bool:
    """Whether the CPU toss env builds and resets right now.

    The toss physics is being reworked on the shared worktree at this step (XML + env pair
    land together), so while that rewrite is in flight a toss env can legitimately fail to
    construct. The env-level tests of this file are guarded by this probe: they run once the
    env is consistent again (and skip on a host without mujoco). The pure-mask tests above
    need no simulator and always run, so CI (no mujoco, no torch) still exercises the mask.
    """
    if importlib.util.find_spec("mujoco") is None:
        return False
    try:
        env = _make_toss_env("none")
    except _TOSS_ENV_ERRORS:  # mujoco absent, or the XML/env pair is mid-rework
        return False
    try:
        env.reset(seed=0)
    except _TOSS_ENV_ERRORS:
        return False
    finally:
        env.close()
    return True


_TOSS_ENV_READY = _toss_env_ready()
_requires_toss_env = pytest.mark.skipif(
    not _TOSS_ENV_READY,
    reason="the CPU toss env does not build/reset (no mujoco, or the toss-physics rework is "
    "in flight); re-run once the XML/env pair is consistent",
)


@_requires_toss_env
def test_cpu_toss_env_hides_the_partner_paddle_under_occlusion() -> None:
    """The env's observation space/vector and info carry the reduced hidden-partner layout."""
    hidden = _make_toss_env("partner")
    full = _make_toss_env("none")
    try:
        for agent in AGENT_KEYS:
            assert hidden.observation_space[agent].shape == (20,)
            assert full.observation_space[agent].shape == (26,)
        hidden_obs, hidden_info = hidden.reset(seed=3)
        full_obs, full_info = full.reset(seed=3)
        assert hidden_info["occlusion_partner"] == 1.0
        assert full_info["occlusion_partner"] == 0.0
        for agent in AGENT_KEYS:
            assert hidden_obs[agent].shape == (20,)
            assert full_obs[agent].shape == (26,)
        # The exact column mapping: own block + ball + own pad + goal/time.
        left_columns = np.r_[0:17, 23:26]
        right_columns = np.r_[0:11, 17:26]
        np.testing.assert_array_equal(hidden_obs["left"], full_obs["left"][left_columns])
        np.testing.assert_array_equal(hidden_obs["right"], full_obs["right"][right_columns])
    finally:
        hidden.close()
        full.close()


@_requires_toss_env
def test_cpu_toss_occluded_and_full_envs_step_identically() -> None:
    """Occlusion is observation-only: same seeds/actions give the same physics and masked obs."""
    hidden = _make_toss_env("partner")
    full = _make_toss_env("none")
    left_columns = np.r_[0:17, 23:26]
    right_columns = np.r_[0:11, 17:26]
    columns = {"left": left_columns, "right": right_columns}
    actions = {
        "left": np.array([0.2, 0.1, -0.05], dtype=np.float32),
        "right": np.array([-0.1, 0.2, 0.05], dtype=np.float32),
    }
    try:
        hidden_obs, _ = hidden.reset(seed=11)
        full_obs, _ = full.reset(seed=11)
        for agent in AGENT_KEYS:
            np.testing.assert_array_equal(hidden_obs[agent], full_obs[agent][columns[agent]])
        for _step in range(3):
            hidden_obs, _, hidden_terminated, hidden_truncated, _ = hidden.step(actions)
            full_obs, _, full_terminated, full_truncated, _ = full.step(actions)
            for agent in AGENT_KEYS:
                np.testing.assert_array_equal(hidden_obs[agent], full_obs[agent][columns[agent]])
            assert hidden_terminated == full_terminated
            assert hidden_truncated == full_truncated
    finally:
        hidden.close()
        full.close()


@_requires_toss_env
def test_cpu_toss_occluded_env_is_reproducible_from_the_seed() -> None:
    """Same seed + same actions -> bit-identical occluded observations."""
    first = _make_toss_env("partner")
    second = _make_toss_env("partner")
    actions = {
        "left": np.array([0.1, 0.0, 0.0], dtype=np.float32),
        "right": np.array([0.0, 0.1, 0.0], dtype=np.float32),
    }
    try:
        obs_a, _ = first.reset(seed=5)
        obs_b, _ = second.reset(seed=5)
        for agent in AGENT_KEYS:
            np.testing.assert_array_equal(obs_a[agent], obs_b[agent])
        for _step in range(3):
            obs_a, reward_a, *_ = first.step(actions)
            obs_b, reward_b, *_ = second.step(actions)
            for agent in AGENT_KEYS:
                np.testing.assert_array_equal(obs_a[agent], obs_b[agent])
            assert reward_a == reward_b
    finally:
        first.close()
        second.close()


@_requires_toss_env
def test_cpu_toss_occluded_oracle_env_keeps_the_partner_block() -> None:
    """Under oracle the occluded env is the oracle vector with the partner paddle removed."""
    hidden = _make_toss_env("partner", partner_mode="oracle")
    full = _make_toss_env("none", partner_mode="oracle")
    left_columns = np.r_[0:17, 23:36]
    right_columns = np.r_[0:11, 17:36]
    try:
        for agent in AGENT_KEYS:
            assert hidden.observation_space[agent].shape == (30,)
            assert full.observation_space[agent].shape == (36,)
        hidden_obs, _ = hidden.reset(seed=7)
        full_obs, _ = full.reset(seed=7)
        np.testing.assert_array_equal(hidden_obs["left"], full_obs["left"][left_columns])
        np.testing.assert_array_equal(hidden_obs["right"], full_obs["right"][right_columns])
        # The privileged partner block (the oracle's upper-bound information) is present
        # unchanged: it is the last ten columns of the occluded oracle vector.
        np.testing.assert_array_equal(hidden_obs["left"][20:], full_obs["left"][26:])
        np.testing.assert_array_equal(hidden_obs["right"][20:], full_obs["right"][26:])
    finally:
        hidden.close()
        full.close()


@_requires_toss_env
def test_cpu_vec_coop_env_carries_the_occlusion_variant() -> None:
    """`VecCoopEnv` builds the masked envs from the config, so the trainer sees 20-wide rows."""
    pytest.importorskip("mujoco")
    from callosum.experiments.coop_tasks.vec_env import VecCoopEnv

    vec = VecCoopEnv("toss", 2, config=TossConfig(occlusion="partner"), seed=0)
    try:
        obs = vec.reset(seed=0)
        for agent in AGENT_KEYS:
            assert obs[agent].shape == (2, 20)
        step = vec.step(
            {
                "left": np.zeros((2, 3), dtype=np.float32),
                "right": np.zeros((2, 3), dtype=np.float32),
            }
        )
        assert step.obs["left"].shape == (2, 20)
        assert vec.obs_dim == 20
        assert all(info["occlusion_partner"] == 1.0 for info in step.infos)
    finally:
        vec.close()


# --- gymnasium id / factory ---------------------------------------------------------------


@_requires_toss_env
def test_occluded_env_id_is_registered_as_the_hidden_partner_task() -> None:
    """`TossCatchOccluded-v0` builds the toss env with `occlusion="partner"` (20/30 fields)."""
    pytest.importorskip("mujoco")
    import gymnasium

    from callosum.experiments.coop_tasks import OCCLUDED_ENV_IDS, TASK_ENV_IDS

    assert TASK_ENV_IDS["toss"] == "TossCatch-v0"
    assert OCCLUDED_ENV_IDS == {"toss": "TossCatchOccluded-v0"}
    env = gymnasium.make("TossCatchOccluded-v0")
    # gymnasium wraps the registered env (OrderEnforcing/TimeLimit), so inspect the base env.
    base = env.unwrapped
    try:
        assert base.config.occlusion == "partner"
        assert base.occlusion == "partner"
        assert base.observation_space["left"].shape == (20,)
        obs, info = env.reset(seed=0)
        assert obs["right"].shape == (20,)
        assert info["occlusion_partner"] == 1.0
    finally:
        env.close()


@_requires_toss_env
def test_occluded_env_factory_rejects_a_config_without_the_occlusion() -> None:
    """The occluded id is not a second full-layout alias: a non-occluded config is refused."""
    pytest.importorskip("mujoco")
    from callosum.experiments.coop_tasks import make_occluded_toss_env

    with pytest.raises(ValueError, match="occlusion"):
        make_occluded_toss_env(config=TossConfig(occlusion="none"))
    with pytest.raises(ValueError, match="TossConfig"):
        make_occluded_toss_env(config=object())  # type: ignore[arg-type]
    oracle = make_occluded_toss_env(partner_mode="oracle")
    try:
        assert oracle.observation_space["left"].shape == (30,)
    finally:
        oracle.close()
