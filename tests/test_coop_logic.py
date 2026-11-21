"""Pure logic of the two-agent coop suite (no mujoco): layout, randomisation, rules, configs.

The environment itself needs MuJoCo (the `sandbox` extra, absent from CI), so the checks
that can run everywhere live here: the observation layout and partner visibility, the pure
observation assembly with the `"none"` leakage sentinel, the deterministic randomisation
sampler, the latency/start-delay helpers, the one-sided allocation, the success/failure/
reward rules, the task config validation and the (pure) task-aware trainer config.

The environment-side checks (spaces, oracle success, absent-agent baseline, seed determinism
and the vec-env contract) live in `tests/test_coop_envs.py` and are skipped without mujoco;
`scripts/smoke_coop.py` covers the scripted rollouts on a machine with the simulator.
"""

import math
from collections import deque
from dataclasses import replace

import numpy as np
import pytest

from callosum.experiments.coop_tasks.config import (
    TASKS,
    CartPoleConfig,
    CoopRandomization,
    CoopRewardConfig,
    PendulumConfig,
    coop_action_size,
    coop_config,
    coop_obs_size,
)
from callosum.experiments.coop_tasks.logic import (
    AGENT_KEYS,
    CARTPOLE_SPEC,
    PENDULUM_SPEC,
    TASK_SPECS,
    TIME_FIELD,
    amplitude_deg,
    build_observation,
    cartpole_failure,
    cartpole_potential,
    cartpole_success,
    delayed_command,
    in_start_delay,
    obs_fields,
    obs_size,
    one_sided_allocation,
    partner_fields,
    partner_of,
    pendulum_failure,
    pendulum_potential,
    pendulum_success,
    potential_shaping,
    sample_episode_randomization,
    task_obs_size,
    task_spec,
    team_reward,
    validate_partner_mode,
    validate_task,
    wind_active_on_step,
)

# --- task specs / observation layout ------------------------------------------------------


def test_task_specs_registry() -> None:
    assert set(TASK_SPECS) == {"cartpole", "pendulum"}
    assert CARTPOLE_SPEC.action_dim == PENDULUM_SPEC.action_dim == 1
    assert CARTPOLE_SPEC.shared_fields == ("cart_x", "cart_vx", "pole_angle", "pole_angvel")
    assert PENDULUM_SPEC.shared_fields == ("angle", "angvel")
    assert task_spec("cartpole") is CARTPOLE_SPEC
    with pytest.raises(ValueError, match="task must be one of"):
        validate_task("toss")
    with pytest.raises(ValueError, match="task must be one of"):
        task_spec("toss")


def test_observation_widths_match_the_documented_layout() -> None:
    # own block: 1 action + 3 private params + 1 start delay = 5
    # shared: cartpole 4 / pendulum 2; goal: cartpole 1 / pendulum 2; + time_fraction
    assert obs_size(CARTPOLE_SPEC, "none") == 5 + 4 + 1 + 1 == 11
    assert obs_size(CARTPOLE_SPEC, "oracle") == 11 + 2 * 1 + 4 == 17
    assert obs_size(PENDULUM_SPEC, "none") == 5 + 2 + 2 + 1 == 10
    assert obs_size(PENDULUM_SPEC, "oracle") == 10 + 2 * 1 + 4 == 16
    assert task_obs_size("cartpole", "none") == 11
    assert task_obs_size("pendulum", "oracle") == 16


def test_obs_fields_are_unique_and_partner_block_is_mode_gated() -> None:
    for spec in (CARTPOLE_SPEC, PENDULUM_SPEC):
        fields = obs_fields(spec, "none")
        assert len(set(fields)) == len(fields)
        assert not any(field.startswith("partner_") for field in fields)
        oracle_fields = obs_fields(spec, "oracle")
        assert len(oracle_fields) == len(fields) + len(partner_fields(spec, "oracle"))
        assert len(oracle_fields) > len(fields)
        assert TIME_FIELD in fields
        assert partner_fields(spec, "none") == ()
    with pytest.raises(ValueError, match="partner_mode must be one of"):
        partner_fields(CARTPOLE_SPEC, "predicted")
    with pytest.raises(ValueError, match="partner_mode must be one of"):
        validate_partner_mode("predicted")


def _sample_values(task: str = "cartpole") -> dict[str, float]:
    """A role-tagged values dict like the env's `_values` (both agents, shared, goal, time)."""
    spec = task_spec(task)
    values: dict[str, float] = {}
    for index, agent in enumerate(AGENT_KEYS):
        base = float(index + 1)
        for axis in range(spec.action_dim):
            values[f"{agent}_action_{axis}"] = base + 0.1 * axis
            values[f"{agent}_command_{axis}"] = base + 0.2 * axis
        values[f"{agent}_latency"] = 0.1 + 0.1 * index
        values[f"{agent}_gain"] = 1.0 + 0.1 * index
        values[f"{agent}_limit_scale"] = 0.9 + 0.1 * index
        values[f"{agent}_start_delay"] = 0.05 * index
    for offset, field in enumerate(spec.shared_fields):
        values[field] = 10.0 + offset
    for offset, field in enumerate(spec.goal_fields):
        values[field] = 100.0 + offset
    values[TIME_FIELD] = 0.25
    return values


def test_build_observation_layout_and_values() -> None:
    spec = CARTPOLE_SPEC
    values = _sample_values("cartpole")
    left = build_observation(values, "left", spec, "none")
    assert left.dtype == np.float32 and left.shape == (11,)
    expected_head = [
        values["left_action_0"],
        values["left_latency"],
        values["left_gain"],
        values["left_limit_scale"],
        values["left_start_delay"],
    ]
    assert left[:5] == pytest.approx(expected_head)
    assert left[5:9] == pytest.approx([values[field] for field in spec.shared_fields])
    assert left[9] == pytest.approx(values["target_x"])
    assert left[10] == pytest.approx(values[TIME_FIELD])

    right = build_observation(values, "right", spec, "none")
    assert right[0] == pytest.approx(values["right_action_0"])
    assert right[1] == pytest.approx(values["right_latency"])
    # The shared block is identical for both agents; only the own/partner block differs.
    assert right[5:].tolist() == left[5:].tolist()

    oracle = build_observation(values, "left", spec, "oracle")
    assert oracle.shape == (17,)
    assert oracle[:11].tolist() == left.tolist()
    partner = oracle[11:]
    assert partner[0] == pytest.approx(values["right_action_0"])
    assert partner[1] == pytest.approx(values["right_command_0"])
    assert partner[2] == pytest.approx(values["right_latency"])
    assert partner[3] == pytest.approx(values["right_gain"])
    assert partner[4] == pytest.approx(values["right_limit_scale"])
    assert partner[5] == pytest.approx(values["right_start_delay"])


@pytest.mark.parametrize("task", TASKS)
def test_none_mode_ignores_every_partner_key(task: str) -> None:
    """The leakage sentinel: under "none" changing partner facts cannot change the vector."""
    spec = task_spec(task)
    values = _sample_values(task)
    for agent in AGENT_KEYS:
        baseline = build_observation(values, agent, spec, "none")
        partner = partner_of(agent)
        tampered = dict(values)
        for key in list(tampered):
            if key.startswith(f"{partner}_"):
                tampered[key] = tampered[key] + 123.0
        # The agent's own "none" observation is bit-for-bit unchanged...
        assert np.array_equal(build_observation(tampered, agent, spec, "none"), baseline)
        # ... while under "oracle" the same tampering does reach the vector.
        oracle_now = build_observation(tampered, agent, spec, "oracle")
        assert not np.array_equal(oracle_now, build_observation(values, agent, spec, "oracle"))


def test_build_observation_rejects_bad_inputs() -> None:
    values = _sample_values("cartpole")
    del values["left_gain"]
    with pytest.raises(ValueError, match="left_gain"):
        build_observation(values, "left", CARTPOLE_SPEC, "none")
    with pytest.raises(ValueError, match="agent must be one of"):
        build_observation(_sample_values(), "middle", CARTPOLE_SPEC, "none")
    with pytest.raises(ValueError, match="partner_mode must be one of"):
        build_observation(_sample_values(), "left", CARTPOLE_SPEC, "predicted")


def test_partner_of() -> None:
    assert partner_of("left") == "right"
    assert partner_of("right") == "left"
    with pytest.raises(ValueError, match="agent must be one of"):
        partner_of("middle")


# --- episode randomisation ---------------------------------------------------------------


def _ranges(**overrides: object) -> CoopRandomization:
    """The cart-pole-like ranges (with a few fields overridden) for the sampler tests."""
    defaults: dict[str, object] = {
        "latency_range": (0.1, 0.3),
        "gain_range": (0.8, 1.2),
        "limit_scale_range": (0.8, 1.2),
        "start_delay_range": (0.0, 0.2),
        "initial_value_range": (-0.05, 0.05),
        "damping_range": (0.0, 0.02),
        "wind_prob_range": (0.0, 0.3),
        "wind_magnitude_range": (1.0, 5.0),
    }
    defaults.update(overrides)
    return CoopRandomization(**defaults)


def test_sampler_is_deterministic_and_in_range() -> None:
    ranges = _ranges(wind_prob_range=(1.0, 1.0), wind_magnitude_range=(2.0, 2.0))
    first = sample_episode_randomization(ranges, control_dt=0.04, max_control_steps=300, seed=11)
    second = sample_episode_randomization(ranges, control_dt=0.04, max_control_steps=300, seed=11)
    assert first == second
    assert first.seed == 11
    for agent in first.agents:
        assert ranges.latency_range[0] <= agent.latency_s <= ranges.latency_range[1]
        assert agent.latency_steps == round(agent.latency_s / 0.04)
        assert ranges.gain_range[0] <= agent.gain <= ranges.gain_range[1]
        assert ranges.limit_scale_range[0] <= agent.limit_scale <= ranges.limit_scale_range[1]
        assert 0 <= agent.start_delay_steps <= round(0.2 / 0.04)
    assert ranges.initial_value_range[0] <= first.initial_value <= ranges.initial_value_range[1]
    assert ranges.damping_range[0] <= first.damping <= ranges.damping_range[1]
    assert first.wind_active
    assert abs(first.wind_force) == pytest.approx(2.0)
    assert first.wind_step in range(300)


def test_sampler_uses_fresh_randomness_per_seed() -> None:
    ranges = _ranges(wind_prob_range=(1.0, 1.0), wind_magnitude_range=(2.0, 2.0))
    records = [
        sample_episode_randomization(ranges, control_dt=0.04, max_control_steps=300, seed=seed)
        for seed in range(12)
    ]
    assert len({record.initial_value for record in records}) > 1
    assert {math.copysign(1.0, record.wind_force) for record in records} == {-1.0, 1.0}


def test_sampler_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="control_dt"):
        sample_episode_randomization(_ranges(), control_dt=0.0, max_control_steps=300, seed=0)
    with pytest.raises(ValueError, match="max_control_steps"):
        sample_episode_randomization(_ranges(), control_dt=0.04, max_control_steps=0, seed=0)


# --- action path -------------------------------------------------------------------------


def test_delayed_command_semantics() -> None:
    queue: deque = deque()
    assert list(delayed_command(queue, (1.0,))) == [1.0]
    queue = deque([np.zeros(1), np.zeros(1)])
    assert delayed_command(queue, (1.0,))[0] == 0.0
    assert delayed_command(queue, (2.0,))[0] == 0.0
    assert delayed_command(queue, (3.0,))[0] == pytest.approx(1.0)
    assert delayed_command(queue, (4.0,))[0] == pytest.approx(2.0)


def test_in_start_delay_and_wind_helpers() -> None:
    ranges = _ranges(start_delay_range=(0.08, 0.08), wind_prob_range=(1.0, 1.0))
    record = sample_episode_randomization(ranges, control_dt=0.04, max_control_steps=50, seed=0)
    agent = record.agents[0]
    assert agent.start_delay_steps == 2
    assert in_start_delay(agent, 0) and in_start_delay(agent, 1)
    assert not in_start_delay(agent, 2)
    assert wind_active_on_step(record, record.wind_step)
    assert not wind_active_on_step(record, record.wind_step + 1)
    disabled = sample_episode_randomization(
        _ranges(wind_prob_range=(0.0, 0.0)),
        control_dt=0.04,
        max_control_steps=50,
        seed=0,
    )
    assert not wind_active_on_step(disabled, disabled.wind_step)


def test_one_sided_allocation() -> None:
    assert one_sided_allocation(3.0, positive_limit=10.0, negative_limit=5.0) == (3.0, 0.0)
    assert one_sided_allocation(-20.0, positive_limit=10.0, negative_limit=5.0) == (0.0, 5.0)
    assert one_sided_allocation(0.0, 1.0, 1.0) == (0.0, 0.0)
    assert one_sided_allocation(-1.0, 1.0, 0.0) == (0.0, 0.0)
    with pytest.raises(ValueError, match="limits must be >= 0"):
        one_sided_allocation(1.0, -1.0, 1.0)


# --- success / failure / reward ----------------------------------------------------------


def test_cartpole_success_conjunction_and_failure() -> None:
    kwargs = {
        "cart_x": 0.5,
        "cart_vx": 0.0,
        "pole_angle": 0.0,
        "pole_angvel": 0.0,
        "target_x": 0.5,
        "pos_tol": 0.05,
        "angle_tol": 0.1,
        "lin_speed_tol": 0.1,
        "ang_speed_tol": 0.2,
    }
    assert cartpole_success(**kwargs)
    assert not cartpole_success(**{**kwargs, "cart_x": 0.56})
    assert not cartpole_success(**{**kwargs, "pole_angle": 0.11})
    assert not cartpole_success(**{**kwargs, "cart_vx": 0.11})
    assert not cartpole_success(**{**kwargs, "pole_angvel": -0.21})
    assert not cartpole_failure(0.5, angle_fail=0.5)
    assert cartpole_failure(-0.51, angle_fail=0.5)


def test_pendulum_success_and_failure() -> None:
    assert not pendulum_success(completed_cycles=1, hold_cycles=2)
    assert pendulum_success(completed_cycles=2, hold_cycles=2)
    assert pendulum_success(completed_cycles=3, hold_cycles=2)
    with pytest.raises(ValueError, match="hold_cycles must be >= 1"):
        pendulum_success(completed_cycles=0, hold_cycles=0)
    assert not pendulum_failure(2.6, angle_fail=2.618)
    assert pendulum_failure(2.7, angle_fail=2.618)


def test_cartpole_potential_is_bounded_and_penalises_errors() -> None:
    kwargs = {
        "cart_x": 0.5,
        "pole_angle": 0.0,
        "target_x": 0.5,
        "pos_scale": 0.5,
        "angle_scale": 0.5,
        "pos_coef": 0.5,
        "angle_coef": 0.5,
    }
    assert cartpole_potential(**kwargs) == pytest.approx(0.0)
    at_start = cartpole_potential(**{**kwargs, "cart_x": 0.0})
    assert at_start == pytest.approx(-0.5)
    worst = cartpole_potential(**{**kwargs, "cart_x": -10.0, "pole_angle": 10.0})
    assert worst == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="scales must be > 0"):
        cartpole_potential(**{**kwargs, "pos_scale": 0.0})
    with pytest.raises(ValueError, match="coefficients must be >= 0"):
        cartpole_potential(**{**kwargs, "angle_coef": -1.0})


def test_pendulum_potential_is_bounded() -> None:
    assert pendulum_potential(peak_abs_angle=0.0, target_amplitude=1.0, coef=0.5) == pytest.approx(
        0.0
    )
    assert pendulum_potential(peak_abs_angle=2.0, target_amplitude=1.0, coef=0.5) == pytest.approx(
        0.5
    )
    assert pendulum_potential(peak_abs_angle=-1.0, target_amplitude=1.0, coef=0.5) == pytest.approx(
        0.0
    )
    with pytest.raises(ValueError, match="target_amplitude must be > 0"):
        pendulum_potential(peak_abs_angle=0.0, target_amplitude=0.0, coef=0.5)


def test_potential_shaping_telescopes() -> None:
    potentials = [0.1 * index for index in range(6)]
    deltas = [
        potential_shaping(potential_now=potentials[index], potential_prev=potentials[index - 1])
        for index in range(1, len(potentials))
    ]
    assert sum(deltas) == pytest.approx(potentials[-1] - potentials[0])


def test_team_reward_three_cases() -> None:
    kwargs = {"success_bonus": 2.0, "step_penalty": 0.001, "failure_penalty": -2.0}
    assert team_reward(succeeded=False, **kwargs) == pytest.approx(-0.001)
    assert team_reward(succeeded=True, **kwargs) == pytest.approx(1.999)
    assert team_reward(succeeded=False, failed=True, **kwargs) == pytest.approx(-2.0)
    # Success wins over failure if both flags are set (they cannot be with a valid config).
    assert team_reward(succeeded=True, failed=True, **kwargs) == pytest.approx(1.999)


def test_amplitude_deg() -> None:
    assert amplitude_deg(-math.pi / 2) == pytest.approx(90.0)


# --- configs -----------------------------------------------------------------------------


def test_cartpole_config_defaults_match_the_task_spec() -> None:
    cfg = CartPoleConfig()
    assert cfg.task == "cartpole"
    assert math.isclose(cfg.dt * cfg.control_every, 0.04)
    assert 250 <= cfg.max_control_steps <= 400
    assert cfg.rail_limit == 1.4 and cfg.max_force == 30.0
    assert cfg.failure_angle == 0.5
    assert cfg.success_pos_tol == 0.05 and cfg.success_angle_tol == 0.1
    assert cfg.success_lin_speed_tol == 0.1 and cfg.success_ang_speed_tol == 0.2
    assert cfg.control_dt == pytest.approx(0.04)
    assert cfg.max_episode_time == pytest.approx(12.0)
    assert cfg.action_size == 1
    assert cfg.agent_authority(0) == cfg.agent_authority(1) == cfg.max_force
    assert cfg.randomization.latency_range == (0.1, 0.3)
    assert cfg.randomization.wind_prob_range[1] > 0


def test_pendulum_config_defaults_match_the_task_spec() -> None:
    cfg = PendulumConfig()
    assert cfg.task == "pendulum"
    assert cfg.max_control_steps == 400
    assert cfg.target_amplitude == pytest.approx(math.radians(65.0))
    assert cfg.failure_angle == pytest.approx(math.radians(150.0))
    assert cfg.hold_cycles == 2
    assert cfg.max_torque == 0.65
    assert cfg.randomization.damping_range == (0.11, 0.15)
    assert cfg.randomization.wind_prob_range == (0.0, 0.0)
    assert cfg.action_size == 1
    assert cfg.agent_authority(0) == pytest.approx(cfg.max_torque)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (CartPoleConfig, {"dt": 0.0}),
        (CartPoleConfig, {"control_every": 0}),
        (CartPoleConfig, {"max_control_steps": 0}),
        (CartPoleConfig, {"gravity": -1.0}),
        (CartPoleConfig, {"rail_limit": 0.0}),
        (CartPoleConfig, {"cart_damping": -0.1}),
        (CartPoleConfig, {"pole_armature": -0.1}),
        (CartPoleConfig, {"max_force": 0.0}),
        (CartPoleConfig, {"failure_angle": math.pi}),
        (CartPoleConfig, {"success_angle_tol": 0.6}),
        (CartPoleConfig, {"success_lin_speed_tol": 0.0}),
        (CartPoleConfig, {"target_x": 2.0}),
        (CartPoleConfig, {"shaping_pos_coef": -0.1}),
        (CartPoleConfig, {"seed": -1}),
        (PendulumConfig, {"hinge_limit": 0.0}),
        (PendulumConfig, {"hinge_limit": 1.0}),  # below failure_angle
        (PendulumConfig, {"target_amplitude": 0.0}),
        (PendulumConfig, {"hold_cycles": 0}),
        (PendulumConfig, {"hold_cycles": 99}),
        (PendulumConfig, {"max_torque": 0.0}),
        (PendulumConfig, {"hinge_damping": -0.1}),
        (PendulumConfig, {"shaping_amplitude_coef": -0.1}),
    ],
)
def test_task_configs_reject_out_of_domain_fields(factory, kwargs) -> None:
    with pytest.raises((ValueError, TypeError)):
        factory(**kwargs)


def test_task_configs_reject_a_fail_fast_reward() -> None:
    # A timeout is worth -0.001 * 300 = -0.3, so -0.1 as the failure penalty would make
    # failing fast better than surviving the budget.
    reward = CoopRewardConfig(failure_penalty=-0.1)
    with pytest.raises(ValueError, match="failure_penalty must be strictly worse"):
        CartPoleConfig(reward=reward)
    with pytest.raises(ValueError, match="failure_penalty must be strictly worse"):
        PendulumConfig(reward=reward)


def test_reward_config_validation() -> None:
    assert CoopRewardConfig().success_bonus == 2.0
    with pytest.raises(ValueError, match="success_bonus"):
        CoopRewardConfig(success_bonus=-1.0)
    with pytest.raises(ValueError, match="step_penalty"):
        CoopRewardConfig(step_penalty=5.0)
    with pytest.raises(ValueError, match="failure_penalty must be < 0"):
        CoopRewardConfig(failure_penalty=0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latency_range": (0.3, 0.1)},
        {"latency_range": (-0.1, 0.2)},
        {"gain_range": (-1.0, 1.0)},
        {"limit_scale_range": (0.0, 1.0)},
        {"start_delay_range": (-0.1, 0.2)},
        {"damping_range": (-0.1, 0.1)},
        {"wind_prob_range": (-0.1, 0.5)},
        {"wind_prob_range": (0.0, 1.5)},
        {"wind_magnitude_range": (5.0, 1.0)},
    ],
)
def test_randomization_rejects_bad_ranges(kwargs) -> None:
    with pytest.raises(ValueError):
        CoopRandomization(**kwargs)


def test_task_config_helpers() -> None:
    assert coop_config("cartpole").task == "cartpole"
    assert isinstance(coop_config("pendulum"), PendulumConfig)
    assert coop_obs_size("cartpole", "none") == 11
    assert coop_obs_size("pendulum", "oracle") == 16
    assert coop_action_size("cartpole") == 1
    with pytest.raises(ValueError, match="task must be one of"):
        coop_config("toss")
    with pytest.raises(ValueError, match="task must be one of"):
        coop_obs_size("toss", "none")
    with pytest.raises(ValueError, match="partner_mode"):
        coop_obs_size("cartpole", "predicted")


def test_randomization_override_helper_keeps_the_config_valid() -> None:
    # `dataclasses.replace` rebuilds the frozen config (re-validating it); the nested
    # randomisation must be replaced through `replace(config.randomization, ...)`.
    cfg = CartPoleConfig()
    strict = replace(cfg, randomization=replace(cfg.randomization, wind_prob_range=(0.0, 0.0)))
    assert strict.randomization.wind_prob_range == (0.0, 0.0)
    assert strict.rail_limit == cfg.rail_limit


# --- task-aware trainer config (pure) ----------------------------------------------------


def test_wrench_ppo_config_is_task_aware() -> None:
    from callosum.experiments.cube_wrench.ppo_config import (
        TASKS,
        WrenchPPOConfig,
        checkpoint_task,
        task_action_size,
        task_obs_size,
    )

    assert TASKS == ("cube", "cartpole", "pendulum")
    assert WrenchPPOConfig().task == "cube"
    assert WrenchPPOConfig().agent_keys == ("holder", "rotator")
    assert WrenchPPOConfig().action_size == 6

    cfg = WrenchPPOConfig(task="cartpole")
    assert cfg.task == "cartpole"
    assert cfg.agent_keys == ("left", "right")
    assert cfg.env_obs_size == cfg.obs_size == 11
    assert cfg.action_size == 1
    assert isinstance(cfg.coop_env_config(), CartPoleConfig)
    assert WrenchPPOConfig(task="pendulum", partner_mode="oracle").env_obs_size == 16
    # `predicted` appends the latent to the task's own observation width.
    assert WrenchPPOConfig(task="pendulum", use_bijepa=True).policy_input_size == 10 + 32
    assert task_obs_size("cartpole", "oracle") == 17
    assert task_action_size("pendulum") == 1
    assert checkpoint_task({"config": {"task": "pendulum"}}) == "pendulum"
    assert checkpoint_task({}) == "cube"
    with pytest.raises(ValueError, match="task must be one of"):
        WrenchPPOConfig(task="toss")
    with pytest.raises(ValueError, match="coop_env_config"):
        WrenchPPOConfig().coop_env_config()


def test_wrench_ppo_config_restricts_the_coop_task_backend_and_preset() -> None:
    from callosum.experiments.cube_wrench.ppo_config import WrenchPPOConfig

    with pytest.raises(ValueError, match="CPU MuJoCo backend"):
        WrenchPPOConfig(task="cartpole", backend="warp")
    with pytest.raises(ValueError, match="randomisation presets"):
        WrenchPPOConfig(task="pendulum", randomization="harsh")


def test_task_flag_is_exposed_on_the_trainer_cli() -> None:
    from callosum.experiments.cube_wrench.ppo_config import (
        WrenchPPOConfig,
        parse_args_with_explicit,
    )

    cfg, explicit = parse_args_with_explicit(["--task", "pendulum", "--seed", "3"])
    assert cfg.task == "pendulum" and cfg.seed == 3
    assert "task" in explicit
    assert isinstance(cfg, WrenchPPOConfig)


def test_checkpoint_compatibility_is_task_aware() -> None:
    torch = pytest.importorskip("torch")  # the trainer module imports torch
    del torch
    from callosum.experiments.cube_wrench.train import check_checkpoint_compat

    cartpole_payload = {
        "config": {"task": "cartpole"},
        "obs_size": 11,
        "partner_mode": "none",
        "input_mode": "none",
        "policy_input_size": 11,
        "action_size": 1,
    }
    assert check_checkpoint_compat(cartpole_payload, "none") == ("none", 0)
    with pytest.raises(ValueError, match="width"):
        check_checkpoint_compat({**cartpole_payload, "obs_size": 28}, "none")
    with pytest.raises(ValueError, match="action values"):
        check_checkpoint_compat({**cartpole_payload, "action_size": 6}, "none")
    # A pre-task-suite cube checkpoint has no `task`; it falls back to the cube path.
    legacy = {
        "config": {"partner_mode": "oracle"},
        "obs_size": 45,
        "partner_mode": "oracle",
        "policy_input_size": 45,
    }
    assert check_checkpoint_compat(legacy, "oracle") == ("oracle", 0)
