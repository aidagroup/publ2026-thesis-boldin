"""Pure logic of CubeWrench-v0 (no mujoco, no torch): layout, randomisation, rules.

The environment itself needs MuJoCo (the `sandbox` extra, absent from CI and the macOS dev
venv without the extra), so the checks that can run everywhere live here: the observation
layout and partner visibility, the pure observation assembly with the `"none"` leakage
sentinel, the deterministic randomisation sampler (including the sampled layer friction),
the latency/injection helpers, the success/failure/reward rules and the config validation.

The few tests that need the simulator -- config-as-truth writes to the MuJoCo model, the
per-reset friction application and the failure-penalty reward -- are guarded with
`pytest.importorskip("mujoco")`: CI (dev extra only) skips them, the macOS sandbox venv
runs them, and `scripts/smoke_wrench_cube.py` covers the oracle / non-additivity rollouts.
"""

import math
from collections import deque
from dataclasses import replace

import numpy as np
import pytest

from callosum.configs.wrench_cube import WrenchCubeConfig
from callosum.envs._wrench_logic import (
    AGENT_KEYS,
    CUBE_STATE_FIELDS,
    GOAL_FIELDS,
    OWN_PARAM_FIELDS,
    OWN_WRENCH_FIELDS,
    PARTNER_FIELDS,
    PARTNER_MODES,
    angle_progress,
    build_observation,
    delayed_command,
    drift_excess,
    failure,
    injection_active,
    obs_fields,
    obs_size,
    param_value_fields,
    partner_fields,
    partner_of,
    potential,
    potential_shaping,
    privileged_value_fields,
    sample_episode_randomization,
    success,
    team_reward,
    validate_partner_mode,
    wrench_value_fields,
)

# --- config ------------------------------------------------------------------------------


def test_config_defaults_match_the_design_numbers() -> None:
    cfg = WrenchCubeConfig()
    assert math.isclose(cfg.dt * cfg.control_every, 0.04)  # 8-10 substeps per decision
    assert 8 <= cfg.control_every <= 10
    assert 300 <= cfg.max_control_steps <= 400
    assert math.isclose(cfg.goal_angle, math.pi / 2)
    assert math.isclose(cfg.hinge_limit, math.pi / 2)
    assert math.isclose(cfg.success_angle_tol, 0.08)
    assert math.isclose(cfg.success_pos_tol, 0.015)
    assert math.isclose(cfg.success_rot_tol, 0.15)
    assert math.isclose(cfg.failure_pos_drift, 0.05)
    assert math.isclose(cfg.failure_rot_drift, 0.5)
    assert cfg.latency_range == (0.0, 0.3)
    assert cfg.gain_range == (0.5, 1.5)
    assert cfg.limit_scale_range == (0.4, 1.6)
    assert cfg.slip_prob_range == (0.2, 0.5)
    assert cfg.slip_decisions_range == (2, 5)
    assert cfg.start_delay_range == (0.0, 0.5)
    assert cfg.layer_frictionloss_range == (0.0, 2.0e-5)
    assert cfg.initial_angle_range == (0.0, 0.05)
    assert cfg.success_bonus == 2.0
    assert cfg.step_penalty == 0.001
    assert cfg.failure_penalty == -2.0
    assert cfg.seed == 0
    # The derived helpers agree with the fields.
    assert math.isclose(cfg.control_dt, cfg.dt * cfg.control_every)
    assert math.isclose(cfg.max_episode_time, cfg.control_dt * cfg.max_control_steps)
    assert cfg.holder_authority[0] == cfg.holder_max_force
    assert cfg.holder_authority[3] == cfg.holder_max_torque
    assert cfg.rotator_authority[5] == cfg.rotator_max_torque


def test_config_initial_angle_default_is_inside_the_hinge_range() -> None:
    cfg = WrenchCubeConfig()
    assert 0.0 <= cfg.initial_angle_range[0] <= cfg.initial_angle_range[1] <= cfg.hinge_limit


def test_config_accepts_a_wider_hinge_limit() -> None:
    cfg = WrenchCubeConfig(hinge_limit=math.pi)
    assert cfg.hinge_limit == math.pi and cfg.goal_angle <= cfg.hinge_limit


def test_config_non_additivity_margin_holds_for_the_defaults() -> None:
    cfg = WrenchCubeConfig()
    resistance = cfg.hinge_damping + cfg.hinge_frictionloss
    predicted_drift = cfg.goal_angle * resistance / (resistance + cfg.core_ang_damping)
    assert predicted_drift >= 1.5 * cfg.success_rot_tol
    # ... and still holds at the lower edge of the sampled layer friction.
    sampled = cfg.hinge_damping + cfg.layer_frictionloss_range[0]
    assert sampled > 0
    assert cfg.goal_angle * sampled / (sampled + cfg.core_ang_damping) >= (
        1.5 * cfg.success_rot_tol
    )


def test_config_rejects_losing_non_additivity() -> None:
    # Too much core damping absorbs the rotator's reaction: the holder becomes unnecessary.
    with pytest.raises(ValueError, match="non-additivity"):
        WrenchCubeConfig(core_ang_damping=1.0)
    with pytest.raises(ValueError, match="hinge_damping . hinge_frictionloss"):
        WrenchCubeConfig(hinge_damping=0.0, hinge_frictionloss=0.0)
    # A frictionless sampled episode would have no reaction path either.
    with pytest.raises(ValueError, match="layer_frictionloss_range"):
        WrenchCubeConfig(hinge_damping=0.0, hinge_frictionloss=1.0e-3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"dt": -0.005},
        {"control_every": 0},
        {"max_control_steps": 0},
        {"hinge_limit": 0.0},
        {"hinge_limit": -math.pi / 2},
        {"hinge_limit": math.pi + 0.1},
        {"hinge_limit": 1.0},  # below the default goal_angle
        {"goal_angle": 0.0},
        {"goal_angle": math.pi},
        {"success_angle_tol": 0.0},
        {"success_angle_tol": 2.0},
        {"success_pos_tol": 0.0},
        {"success_pos_tol": 0.1},  # >= failure_pos_drift
        {"success_rot_tol": 0.6},  # >= failure_rot_drift
        {"success_lin_speed_tol": 0.0},
        {"holder_max_force": 0.0},
        {"rotator_max_torque": -1.0},
        {"hinge_damping": -1e-4},
        {"core_ang_damping": -1e-4},
        {"latency_range": (-0.1, 0.3)},
        {"latency_range": (0.3, 0.1)},
        {"gain_range": (0.0, 1.4)},
        {"limit_scale_range": (-1.0, 1.0)},
        {"slip_prob_range": (0.0, 1.5)},
        {"slip_decisions_range": (0, 3)},
        {"slip_decisions_range": (3, 1)},
        {"start_delay_range": (-0.1, 0.3)},
        {"layer_frictionloss_range": (-1e-6, 2e-5)},
        {"layer_frictionloss_range": (3e-5, 1e-5)},
        {"initial_angle_range": (-0.05, 0.05)},  # negative: outside the hinge range
        {"initial_angle_range": (0.05, -0.05)},
        {"initial_angle_range": (0.0, math.pi)},  # above hinge_limit
        {"impulse_prob_range": (0.0, 1.5)},
        {"impulse_magnitude_range": (-1.0, 0.05)},
        {"success_bonus": -1.0},
        {"step_penalty": 1.0},  # >= success_bonus
        {"failure_penalty": 0.0},
        {"failure_penalty": 1.0},
        {"failure_penalty": -0.0005},  # better than -step_penalty
        {"step_penalty": 0.5, "failure_penalty": -2.0},  # better than a full timeout
        {"seed": -1},
    ],
)
def test_config_rejects_bad_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        WrenchCubeConfig(**kwargs)


# --- observation layout and partner visibility -------------------------------------------


def test_partner_modes_are_none_and_oracle_for_now() -> None:
    assert PARTNER_MODES == ("none", "oracle")
    with pytest.raises(ValueError, match="partner_mode"):
        validate_partner_mode("predicted")  # Bi-JEPA block is a later step
    with pytest.raises(ValueError, match="partner_mode"):
        validate_partner_mode("full")


def test_none_observation_has_no_partner_fields() -> None:
    fields = obs_fields("none")
    assert obs_size("none") == len(fields)
    assert not any(field.startswith("partner_") for field in fields)
    assert partner_fields("none") == ()
    # Own wrench first, then own private parameters, then the shared cube state and goal.
    assert fields[: len(OWN_WRENCH_FIELDS)] == OWN_WRENCH_FIELDS
    assert fields[len(OWN_WRENCH_FIELDS) :][: len(OWN_PARAM_FIELDS)] == OWN_PARAM_FIELDS
    assert CUBE_STATE_FIELDS[0] in fields and CUBE_STATE_FIELDS[-1] in fields
    assert fields[-len(GOAL_FIELDS) :] == GOAL_FIELDS


def test_oracle_observation_adds_the_partner_block() -> None:
    none_fields = obs_fields("none")
    oracle_fields = obs_fields("oracle")
    assert oracle_fields[: len(none_fields)] == none_fields
    assert oracle_fields[len(none_fields) :] == partner_fields("oracle")
    assert obs_size("oracle") == obs_size("none") + len(partner_fields("oracle"))
    assert len(partner_fields("oracle")) == 11  # applied wrench + params + start delay/slip
    assert all(field.startswith("partner_") for field in partner_fields("oracle"))


def test_observation_fields_are_unique_in_both_modes() -> None:
    for mode in PARTNER_MODES:
        fields = obs_fields(mode)
        assert len(set(fields)) == len(fields)
        assert obs_size(mode) == len(fields)


def test_partner_fields_matches_the_shared_expectation() -> None:
    assert partner_fields("none") == ()
    assert partner_fields("oracle") == PARTNER_FIELDS


def _values(seed: int = 0) -> dict[str, float]:
    """The env's role-tagged `values` dict for one default-config episode."""
    episode = sample_episode_randomization(WrenchCubeConfig(), seed)
    values: dict[str, float] = {}
    for agent_idx, key in enumerate(AGENT_KEYS):
        agent = episode.agents[agent_idx]
        values.update(zip(wrench_value_fields(key), np.linspace(0.0, 1.0, 6) + agent_idx))
        values.update(
            zip(param_value_fields(key), (agent.latency_s, agent.gain, agent.limit_scale))
        )
        values.update(zip(privileged_value_fields(key), (agent.start_delay_s, 0.0)))
    values.update({name: (index - 7.0) / 10.0 for index, name in enumerate(CUBE_STATE_FIELDS)})
    values["goal_angle"] = math.pi / 2
    values["time_fraction"] = 0.25
    return values


def test_build_observation_has_the_layout_width_and_order() -> None:
    values = _values()
    for mode in PARTNER_MODES:
        assert obs_size(mode) == (28 if mode == "none" else 39)
        for key in AGENT_KEYS:
            vector = build_observation(values, key, mode)
            assert vector.shape == (obs_size(mode),)
            assert vector.dtype == np.float32
            # The own-wrench block is the first layout block.
            expected = [values[f"{key}_wrench_{axis}"] for axis in ("fx", "fy", "fz")]
            assert np.array_equal(vector[:3], np.asarray(expected, dtype=np.float32))


def test_build_observation_none_ignores_every_partner_field() -> None:
    """Leakage sentinel: under "none" partner fields cannot change the vector, bit-for-bit."""
    values = _values()
    for agent in AGENT_KEYS:
        baseline = build_observation(values, agent, "none")
        partner = partner_of(agent)
        for name in wrench_value_fields(partner) + param_value_fields(partner):
            saved = values[name]
            values[name] = saved + 7.0
            try:
                changed = build_observation(values, agent, "none")
                assert changed.tobytes() == baseline.tobytes(), name
            finally:
                values[name] = saved


def test_build_observation_oracle_uses_the_partner_fields() -> None:
    values = _values()
    holder_none = build_observation(values, "holder", "none")
    holder_oracle = build_observation(values, "holder", "oracle")
    assert holder_oracle.shape == (39,)
    assert np.array_equal(holder_oracle[:28], holder_none)  # same first 28 fields
    partner_values = [values[name] for name in wrench_value_fields("rotator")]
    assert np.array_equal(holder_oracle[28:34], np.asarray(partner_values, dtype=np.float32))
    # Changing a partner private parameter changes only the oracle block.
    values["rotator_gain"] += 1.0
    changed = build_observation(values, "holder", "oracle")
    assert not np.array_equal(changed, holder_oracle)
    assert np.array_equal(changed[:28], holder_none)


def test_build_observation_rejects_unknown_agents_and_missing_values() -> None:
    values = _values()
    with pytest.raises(ValueError, match="agent"):
        build_observation(values, "partner", "none")
    with pytest.raises(ValueError, match="partner_mode"):
        build_observation(values, "holder", "full")
    del values["rotator_gain"]
    with pytest.raises(ValueError, match="rotator_gain"):
        build_observation(values, "holder", "oracle")


# --- episode randomisation ---------------------------------------------------------------


def test_randomisation_is_deterministic_in_the_seed() -> None:
    cfg = WrenchCubeConfig()
    first = sample_episode_randomization(cfg, seed=7)
    second = sample_episode_randomization(cfg, seed=7)
    other = sample_episode_randomization(cfg, seed=8)
    assert first == second
    assert first != other
    assert first.seed == 7


@pytest.mark.parametrize("seed", range(12))
def test_randomisation_values_stay_inside_the_config_ranges(seed: int) -> None:
    cfg = WrenchCubeConfig()
    episode = sample_episode_randomization(cfg, seed=seed)
    max_latency_steps = round(cfg.latency_range[1] / cfg.control_dt)
    max_start_steps = round(cfg.start_delay_range[1] / cfg.control_dt)
    for agent in episode.agents:
        assert cfg.latency_range[0] <= agent.latency_s <= cfg.latency_range[1]
        assert 0 <= agent.latency_steps <= max_latency_steps
        assert math.isclose(
            agent.latency_steps * cfg.control_dt, agent.latency_s, abs_tol=cfg.control_dt
        )
        assert cfg.gain_range[0] <= agent.gain <= cfg.gain_range[1]
        assert cfg.limit_scale_range[0] <= agent.limit_scale <= cfg.limit_scale_range[1]
        assert 0 <= agent.start_delay_steps <= max_start_steps
        assert 0 <= agent.slip_step < cfg.max_control_steps
        assert cfg.slip_decisions_range[0] <= agent.slip_steps <= cfg.slip_decisions_range[1]
    assert cfg.initial_angle_range[0] <= episode.layer_initial_angle <= cfg.initial_angle_range[1]
    assert (
        cfg.layer_frictionloss_range[0]
        <= episode.layer_frictionloss
        <= cfg.layer_frictionloss_range[1]
    )
    assert 0 <= episode.impulse_step < cfg.max_control_steps
    magnitude = math.sqrt(sum(component**2 for component in episode.impulse_force))
    assert (
        cfg.impulse_magnitude_range[0] - 1e-9 <= magnitude <= cfg.impulse_magnitude_range[1] + 1e-9
    )


def test_layer_friction_is_sampled_per_episode_and_varies_by_seed() -> None:
    cfg = WrenchCubeConfig()
    samples = [sample_episode_randomization(cfg, seed).layer_frictionloss for seed in range(12)]
    assert all(0.0 <= value <= cfg.layer_frictionloss_range[1] for value in samples)
    assert len(set(samples)) > 1  # private randomness, not a constant
    assert samples[0] == sample_episode_randomization(cfg, 0).layer_frictionloss


def test_randomisation_agents_are_in_the_holder_rotator_order() -> None:
    episode = sample_episode_randomization(WrenchCubeConfig(), seed=3)
    assert len(episode.agents) == len(AGENT_KEYS) == 2
    # The two records are independently sampled (the partner's parameters are private).
    assert episode.agents[0] != episode.agents[1]


def test_slip_and_impulse_are_draw_dependent_but_fixed_by_the_seed() -> None:
    cfg = WrenchCubeConfig()
    # Across seeds the Bernoulli draws must be reproduced bit-for-bit, and with the
    # default max probabilities at least one seed of a decent sample has each event.
    episodes = [sample_episode_randomization(cfg, seed=seed) for seed in range(40)]
    assert any(agent.slip_active for episode in episodes for agent in episode.agents)
    assert any(episode.impulse_active for episode in episodes)
    again = [sample_episode_randomization(cfg, seed=seed) for seed in range(40)]
    assert episodes == again


# --- action path: latency queue and injections -------------------------------------------


def test_delayed_command_applies_the_command_after_the_latency_steps() -> None:
    command = (0.5, -0.25, 1.0, 0.0, 0.5, -1.0)
    next_command = (2.0,) * 6
    # Latency 0: the queue is empty, the command takes effect immediately.
    zero_latency: deque = deque()
    assert delayed_command(zero_latency, command) == command
    # Latency 1: the zero command enqueued at reset applies first, the command next step.
    one = deque([(0.0,) * 6])
    assert delayed_command(one, command) == (0.0,) * 6
    assert delayed_command(one, next_command) == command
    # Latency 2: the command applies two steps later.
    two = deque([(0.0,) * 6, (0.0,) * 6])
    assert delayed_command(two, command) == (0.0,) * 6
    assert delayed_command(two, next_command) == (0.0,) * 6
    assert delayed_command(two, next_command) == command


def test_injection_active_covers_the_start_delay_and_the_slip_window() -> None:
    cfg = WrenchCubeConfig()
    agent = replace(
        sample_episode_randomization(cfg, seed=0).agents[0],
        start_delay_steps=2,
        slip_active=True,
        slip_step=5,
        slip_steps=2,
    )
    assert [injection_active(agent, step) for step in range(10)] == [
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    no_slip = replace(agent, slip_active=False)
    assert not any(injection_active(no_slip, step) for step in range(2, 10))


# --- success / failure / reward -----------------------------------------------------------


def _success_kwargs(**overrides: float) -> dict[str, float]:
    """Convenient success-call keyword arguments for the default tolerances."""
    kwargs = {
        "goal_angle": math.pi / 2,
        "angle_tol": 0.05,
        "pos_tol": 0.01,
        "rot_tol": 0.1,
    }
    kwargs.update(overrides)
    return kwargs


def test_success_is_the_conjunction_of_all_tolerances() -> None:
    goal = math.pi / 2
    assert success(goal, 0.0, 0.0, **_success_kwargs())
    assert success(goal + 0.0495, 0.01, 0.1, **_success_kwargs())  # inside the bounds
    assert not success(goal + 0.0501, 0.0, 0.0, **_success_kwargs())
    assert not success(goal - 0.0501, 0.0, 0.0, **_success_kwargs())
    assert not success(goal, 0.0101, 0.0, **_success_kwargs())
    assert not success(goal, 0.0, 0.1001, **_success_kwargs())


def test_success_optionally_requires_the_cube_to_be_at_rest() -> None:
    goal = math.pi / 2
    kwargs = _success_kwargs(lin_speed_tol=0.05, ang_speed_tol=0.3)
    assert success(goal, 0.0, 0.0, lin_speed=0.05, ang_speed=0.3, **kwargs)
    assert not success(goal, 0.0, 0.0, lin_speed=0.0501, ang_speed=0.0, **kwargs)
    assert not success(goal, 0.0, 0.0, lin_speed=0.0, ang_speed=0.3001, **kwargs)
    # Without the speed arguments the speed check is disabled.
    assert success(goal, 0.0, 0.0, lin_speed=10.0, ang_speed=10.0, **_success_kwargs())


def test_failure_is_strictly_beyond_the_bounds() -> None:
    kwargs = {"pos_drift": 0.05, "rot_drift": 0.5}
    assert not failure(0.05, 0.5, **kwargs)  # exactly at the bounds is not a failure
    assert failure(0.0501, 0.0, **kwargs)
    assert failure(0.0, 0.5001, **kwargs)
    assert not failure(0.01, 0.1, **kwargs)


def test_team_reward_is_shared_and_penalises_time_and_failure() -> None:
    kwargs = {"success_bonus": 1.0, "step_penalty": 0.01, "failure_penalty": -1.0}
    assert team_reward(succeeded=False, **kwargs) == pytest.approx(-0.01)
    assert team_reward(succeeded=True, **kwargs) == pytest.approx(0.99)
    assert team_reward(succeeded=False, failed=True, **kwargs) == pytest.approx(-1.0)
    # Success wins if both flags were somehow set (they cannot be with a valid config).
    assert team_reward(succeeded=True, failed=True, **kwargs) == pytest.approx(0.99)
    assert team_reward(succeeded=True, success_bonus=0.0, step_penalty=0.0) == 0.0


# --- simulator-side checks (skipped without the sandbox extra) ---------------------------


def test_env_writes_config_truth_to_the_model() -> None:
    pytest.importorskip("mujoco")
    from callosum.envs.wrench_cube import WrenchCubeEnv

    cfg = WrenchCubeConfig(hinge_limit=math.pi)
    env = WrenchCubeEnv(config=cfg)
    try:
        hinge = env._model.joint("layer_hinge")
        hinge_dof = int(env._model.jnt_dofadr[hinge.id])
        assert env._model.opt.timestep == pytest.approx(cfg.dt)
        assert tuple(env._model.jnt_range[hinge.id]) == pytest.approx((0.0, cfg.hinge_limit))
        assert env._model.dof_damping[hinge_dof] == pytest.approx(cfg.hinge_damping)
        assert env._model.dof_armature[hinge_dof] == pytest.approx(cfg.hinge_armature)
        # The nominal friction is written at construction; every reset replaces it.
        assert env._model.dof_frictionloss[hinge_dof] == pytest.approx(cfg.hinge_frictionloss)
    finally:
        env.close()


def test_env_applies_the_sampled_layer_friction_on_every_reset() -> None:
    pytest.importorskip("mujoco")
    from callosum.envs.wrench_cube import WrenchCubeEnv

    cfg = WrenchCubeConfig()
    env = WrenchCubeEnv(config=cfg)
    try:
        applied = []
        for seed in range(6):
            env.reset(seed=seed)
            sampled = env.randomization.layer_frictionloss
            assert cfg.layer_frictionloss_range[0] <= sampled <= cfg.layer_frictionloss_range[1]
            assert env._model.dof_frictionloss[env._hinge_dof] == pytest.approx(sampled)
            applied.append(sampled)
        assert len(set(applied)) > 1  # different seeds -> different applied friction
    finally:
        env.close()


def test_env_observation_widths_and_initial_angle_inside_the_hinge_range() -> None:
    pytest.importorskip("mujoco")
    from callosum.envs.wrench_cube import WrenchCubeEnv

    for mode in PARTNER_MODES:
        env = WrenchCubeEnv(partner_mode=mode)
        try:
            obs, info = env.reset(seed=0)
            assert {key: vector.shape for key, vector in obs.items()} == {
                key: (obs_size(mode),) for key in AGENT_KEYS
            }
            assert set(info) >= {"success", "angle_deg", "drift_pos", "drift_rot", "time"}
            for seed in range(6):
                env.reset(seed=seed)
                angle = float(env._data.qpos[env._hinge_qpos])
                assert 0.0 <= angle <= env.config.hinge_limit
        finally:
            env.close()


def test_env_failure_step_pays_the_failure_penalty() -> None:
    pytest.importorskip("mujoco")
    from callosum.envs.wrench_cube import WrenchCubeEnv

    cfg = WrenchCubeConfig(
        failure_pos_drift=0.001,
        success_pos_tol=0.0005,
        latency_range=(0.0, 0.0),  # apply the push immediately
        slip_prob_range=(0.0, 0.0),
        start_delay_range=(0.0, 0.0),
    )
    env = WrenchCubeEnv(config=cfg)
    try:
        env.reset(seed=0)
        action = {key: np.zeros(6, dtype=np.float32) for key in AGENT_KEYS}
        action["holder"][0] = 1.0  # push the core sideways -> drift past the failure bound
        reward = 0.0
        terminated = False
        info: dict = {}
        for _ in range(10):
            _obs, reward, terminated, _truncated, info = env.step(action)
            if terminated:
                break
        assert terminated and not info["success"]
        assert reward == pytest.approx(cfg.failure_penalty)
    finally:
        env.close()


# --- shaping reward (sparse-success probes did not learn without it) -----------------------


def test_angle_progress_sign_and_scale() -> None:
    goal = math.pi / 2
    assert angle_progress(0.3, 0.2, goal) > 0
    assert angle_progress(0.2, 0.3, goal) < 0
    # A full quarter turn from 0 to the goal sums to ~1 over the episode.
    steps = 157
    total = sum(angle_progress((i + 1) * 0.01, i * 0.01, goal) for i in range(steps))
    assert total == pytest.approx(1.0, abs=0.01)
    assert angle_progress(0.1, 0.0, 0.0) == 0.0


def test_drift_excess_bounds_and_validation() -> None:
    kwargs = {"pos_tol": 0.001, "rot_tol": 0.1, "pos_drift": 0.05, "rot_drift": 0.5}
    assert drift_excess(0.0, 0.0, **kwargs) == 0.0
    assert drift_excess(0.05, 0.5, **kwargs) == pytest.approx(1.0)
    assert drift_excess(0.03, 0.0, **kwargs) == pytest.approx((0.03 - 0.001) / (0.05 - 0.001))
    with pytest.raises(ValueError):
        drift_excess(0.0, 0.0, pos_tol=0.1, rot_tol=0.1, pos_drift=0.05, rot_drift=0.5)


def test_potential_is_bounded_and_penalises_drift() -> None:
    kwargs = {
        "angle_initial": 0.0,
        "goal_angle": math.pi / 2,
        "pos_tol": 0.001,
        "rot_tol": 0.1,
        "pos_drift": 0.05,
        "rot_drift": 0.5,
        "angle_coef": 0.5,
        "drift_coef": 0.5,
    }
    reset = potential(angle=0.0, drift_pos=0.0, drift_rot=0.0, **kwargs)
    assert reset == 0.0
    goal = potential(angle=math.pi / 2, drift_pos=0.0, drift_rot=0.0, **kwargs)
    assert goal == pytest.approx(0.5)  # bounded by angle_coef, not path-dependent
    mid = potential(angle=math.pi / 4, drift_pos=0.0, drift_rot=0.0, **kwargs)
    assert 0.0 < mid < goal
    drifted = potential(angle=math.pi / 2, drift_pos=0.05, drift_rot=0.5, **kwargs)
    assert drifted == pytest.approx(goal - 0.5)
    with pytest.raises(ValueError):
        potential(angle=0.0, drift_pos=0.0, drift_rot=0.0, **{**kwargs, "angle_coef": -1.0})


def test_potential_shaping_telescopes() -> None:
    potentials = [i * 0.1 for i in range(6)]
    deltas = [
        potential_shaping(potential_now=potentials[i], potential_prev=potentials[i - 1])
        for i in range(1, len(potentials))
    ]
    assert sum(deltas) == pytest.approx(potentials[-1] - potentials[0])


def test_config_rejects_negative_shaping_coefficients() -> None:
    with pytest.raises(ValueError):
        WrenchCubeConfig(shaping_angle_coef=-0.1)
    with pytest.raises(ValueError):
        WrenchCubeConfig(shaping_drift_coef=-0.1)


def test_env_shaping_adds_zero_for_a_still_cube() -> None:
    pytest.importorskip("mujoco")
    from callosum.envs.wrench_cube import WrenchCubeEnv

    cfg = WrenchCubeConfig(shaping_angle_coef=1.0, shaping_drift_coef=1.0)
    env = WrenchCubeEnv(config=cfg)
    try:
        env.reset(seed=0)
        action = {key: np.zeros(6, dtype=np.float32) for key in AGENT_KEYS}
        _obs, reward, terminated, _truncated, _info = env.step(action)
        assert not terminated
        assert reward == pytest.approx(-cfg.step_penalty)
    finally:
        env.close()
