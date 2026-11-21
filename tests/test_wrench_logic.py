"""Pure logic of CubeWrench-v0 (no mujoco, no torch): layout, randomisation, rules.

The environment itself needs MuJoCo (the `sandbox` extra, absent from CI and the macOS
dev venv without the extra), so the checks that can run everywhere live here: the
observation layout and partner visibility, the deterministic randomisation sampler, the
success/failure/reward rules and the config validation. `scripts/smoke_wrench_cube.py`
covers the simulator side locally.
"""

import math

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
    failure,
    obs_fields,
    obs_size,
    partner_fields,
    sample_episode_randomization,
    success,
    team_reward,
    validate_partner_mode,
)

# --- config ------------------------------------------------------------------------------


def test_config_defaults_match_the_design_numbers() -> None:
    cfg = WrenchCubeConfig()
    assert math.isclose(cfg.dt * cfg.control_every, 0.04)  # 8-10 substeps per decision
    assert 8 <= cfg.control_every <= 10
    assert 300 <= cfg.max_control_steps <= 400
    assert math.isclose(cfg.goal_angle, math.pi / 2)
    assert math.isclose(cfg.success_angle_tol, 0.05)
    assert math.isclose(cfg.success_pos_tol, 0.01)
    assert math.isclose(cfg.success_rot_tol, 0.1)
    assert math.isclose(cfg.failure_pos_drift, 0.05)
    assert math.isclose(cfg.failure_rot_drift, 0.5)
    assert cfg.latency_range == (0.0, 0.3)
    assert cfg.gain_range == (0.6, 1.4)
    assert cfg.limit_scale_range == (0.5, 1.5)
    assert cfg.slip_prob_range == (0.0, 0.3)
    assert cfg.slip_decisions_range == (1, 3)
    assert cfg.start_delay_range == (0.0, 0.3)
    assert cfg.initial_angle_range == (-0.05, 0.05)
    assert cfg.success_bonus == 1.0
    assert cfg.step_penalty == 0.01
    assert cfg.seed == 0
    # The derived helpers agree with the fields.
    assert math.isclose(cfg.control_dt, cfg.dt * cfg.control_every)
    assert math.isclose(cfg.max_episode_time, cfg.control_dt * cfg.max_control_steps)
    assert cfg.holder_authority[0] == cfg.holder_max_force
    assert cfg.holder_authority[3] == cfg.holder_max_torque
    assert cfg.rotator_authority[5] == cfg.rotator_max_torque


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"dt": -0.005},
        {"control_every": 0},
        {"max_control_steps": 0},
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
        {"initial_angle_range": (0.05, -0.05)},
        {"impulse_prob_range": (0.0, 1.5)},
        {"impulse_magnitude_range": (-1.0, 0.05)},
        {"success_bonus": -1.0},
        {"step_penalty": 1.0},  # >= success_bonus
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
    assert len(partner_fields("oracle")) == 9  # 6D wrench + latency/gain/limit_scale
    assert all(field.startswith("partner_") for field in partner_fields("oracle"))


def test_observation_fields_are_unique_in_both_modes() -> None:
    for mode in PARTNER_MODES:
        fields = obs_fields(mode)
        assert len(set(fields)) == len(fields)
        assert obs_size(mode) == len(fields)


def test_partner_fields_matches_the_shared_expectation() -> None:
    assert partner_fields("none") == ()
    assert partner_fields("oracle") == PARTNER_FIELDS


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
    assert 0 <= episode.impulse_step < cfg.max_control_steps
    magnitude = math.sqrt(sum(component**2 for component in episode.impulse_force))
    assert (
        cfg.impulse_magnitude_range[0] - 1e-9 <= magnitude <= cfg.impulse_magnitude_range[1] + 1e-9
    )


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


def test_team_reward_is_shared_and_penalises_time() -> None:
    assert team_reward(succeeded=False, success_bonus=1.0, step_penalty=0.01) == pytest.approx(
        -0.01
    )
    assert team_reward(succeeded=True, success_bonus=1.0, step_penalty=0.01) == pytest.approx(0.99)
    assert team_reward(succeeded=True, success_bonus=0.0, step_penalty=0.0) == 0.0
