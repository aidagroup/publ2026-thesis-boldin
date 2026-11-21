"""MuJoCo-guarded tests of the two-agent coop envs (`CartPoleCoop-v0`, `PendulumPump-v0`).

The environments need the `sandbox` extra (mujoco), which CI does not install, so every test
here starts with `pytest.importorskip("mujoco")` (the same pattern as the simulator-side
checks of `tests/test_wrench_logic.py`): CI skips them, the macOS sandbox venv runs them, and
`scripts/smoke_coop.py` covers the scripted oracle/baseline rollouts on a machine with the
simulator.

What is checked: the Gymnasium spaces and observation widths (against the pure
`_coop_logic` layout), the config-as-truth model writes and the sampled-damping reset, seed
determinism and reset-friendliness (no hidden state across episodes), the deterministic
scripted oracle reaching the task and the one-sided only-left baseline failing it, the
terminal reward cases, the pendulum hold-cycle success, and the `_vec_coop` auto-reset / vec
contract the trainer relies on.
"""

import numpy as np
import pytest

from callosum.configs.coop_tasks import coop_config
from callosum.envs._coop_logic import AGENT_KEYS, task_obs_size

_ORACLE_EPISODES = 4
"""Small local episode count for the oracle/baseline checks (the envs are CPU MuJoCo)."""


def _make_env(task: str, **kwargs):
    """Build a coop env (imported lazily so the module imports without mujoco)."""
    from callosum.envs.coop import make_coop_env

    return make_coop_env(task, **kwargs)


def _make_oracle(task: str, config):
    """Build the task's scripted oracle (imported lazily)."""
    from callosum.envs.coop import make_oracle

    return make_oracle(task, config)


def _zero_action(env) -> dict[str, np.ndarray]:
    """The all-zero (do-nothing) action for the given env."""
    return {key: np.zeros(env.action_space[key].shape, dtype=np.float32) for key in AGENT_KEYS}


def _run_episode(env, oracle, seed: int, *, drop_right: bool = False) -> tuple[dict, int]:
    """Run one scripted episode to termination/truncation; returns its info and step count."""
    env.reset(seed=seed)
    steps = 0
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        action = oracle.act(env)
        if drop_right:
            action["right"] = np.zeros_like(action["right"])
        _obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
    return info, steps


@pytest.mark.parametrize("task", ("cartpole", "pendulum"))
def test_spaces_and_observation_widths(task: str) -> None:
    pytest.importorskip("mujoco")
    env = _make_env(task)
    try:
        assert set(env.action_space.spaces) == set(AGENT_KEYS)
        for key in AGENT_KEYS:
            assert env.action_space[key].shape == (1,)
            assert env.observation_space[key].shape == (task_obs_size(task, "none"),)
        obs, info = env.reset(seed=0)
        assert set(obs) == set(AGENT_KEYS)
        for vector in obs.values():
            assert vector.dtype == np.float32
            assert vector.shape == (task_obs_size(task, "none"),)
            assert np.all(np.isfinite(vector))
        assert info["success"] is False
        for key in ("angle_deg", "time"):
            assert key in info
        # The oracle partner block widens every agent's observation by 2*1 + 4 = 6.
        oracle_env = _make_env(task, partner_mode="oracle")
        try:
            assert oracle_env.observation_space[AGENT_KEYS[0]].shape == (
                task_obs_size(task, "oracle"),
            )
            assert task_obs_size(task, "oracle") == task_obs_size(task, "none") + 6
        finally:
            oracle_env.close()
    finally:
        env.close()


@pytest.mark.parametrize("task", ("cartpole", "pendulum"))
def test_config_is_written_into_the_model_and_sampled_on_reset(task: str) -> None:
    pytest.importorskip("mujoco")
    cfg = coop_config(task)
    env = _make_env(task, config=cfg)
    try:
        model = env._model
        assert model.opt.timestep == pytest.approx(cfg.dt)
        assert model.opt.gravity[2] == pytest.approx(-cfg.gravity)
        if task == "cartpole":
            cart_joint = model.joint("cart_slide")
            assert tuple(model.jnt_range[cart_joint.id]) == pytest.approx(
                (-cfg.rail_limit, cfg.rail_limit)
            )
            assert model.dof_damping[env._cart_dof] == pytest.approx(cfg.cart_damping)
            assert model.dof_armature[env._pole_dof] == pytest.approx(cfg.pole_armature)
        else:
            hinge = model.joint("pivot_hinge")
            assert tuple(model.jnt_range[hinge.id]) == pytest.approx(
                (-cfg.hinge_limit, cfg.hinge_limit)
            )
            assert model.dof_armature[env._hinge_dof] == pytest.approx(cfg.hinge_armature)
        hinge_dof = env._pole_dof if task == "cartpole" else env._hinge_dof
        applied = []
        for seed in range(6):
            env.reset(seed=seed)
            sampled = env.randomization.damping
            assert (
                cfg.randomization.damping_range[0] <= sampled <= cfg.randomization.damping_range[1]
            )
            assert model.dof_damping[hinge_dof] == pytest.approx(sampled)
            applied.append(sampled)
        assert len(set(applied)) > 1  # different seeds -> different applied damping
    finally:
        env.close()


@pytest.mark.parametrize("task", ("cartpole", "pendulum"))
def test_seed_determinism_and_reset_friendliness(task: str) -> None:
    pytest.importorskip("mujoco")
    env = _make_env(task)
    try:
        action = _zero_action(env)
        first_obs, _first_info = env.reset(seed=42)
        first_state = env.state()
        for _ in range(5):
            env.step(action)
        # A second reset with the same seed must reproduce the episode exactly: nothing from
        # the previous rollout may leak into the new one.
        repeat_obs, _repeat_info = env.reset(seed=42)
        for key in AGENT_KEYS:
            assert np.array_equal(first_obs[key], repeat_obs[key])
        for name, value in first_state.items():
            assert env.state()[name] == pytest.approx(value)
        # Different seeds get different private randomness.
        seeds = set()
        for seed in range(8):
            env.reset(seed=seed)
            seeds.add(round(env.randomization.damping, 12))
        assert len(seeds) > 1
    finally:
        env.close()


@pytest.mark.parametrize("task", ("cartpole", "pendulum"))
def test_pending_commands_are_the_latency_queues(task: str) -> None:
    pytest.importorskip("mujoco")
    env = _make_env(task)
    try:
        with pytest.raises(RuntimeError, match="reset"):
            env.pending_commands()
        env.reset(seed=0)
        latency = [agent.latency_steps for agent in env.randomization.agents]
        pending = env.pending_commands()
        assert {key: len(value) for key, value in pending.items()} == {
            key: latency[index] for index, key in enumerate(AGENT_KEYS)
        }
        for value in pending.values():
            for command in value:
                assert command.shape == (1,)
                assert np.allclose(command, 0.0)  # the queues start with zeros
        env.step(_zero_action(env))
        assert {key: len(value) for key, value in env.pending_commands().items()} == {
            key: latency[index] for index, key in enumerate(AGENT_KEYS)
        }
    finally:
        env.close()


@pytest.mark.parametrize("task", ("cartpole", "pendulum"))
def test_oracle_succeeds_and_only_left_baseline_fails(task: str) -> None:
    pytest.importorskip("mujoco")
    env = _make_env(task)
    try:
        oracle = _make_oracle(task, env.config)
        oracle_successes = []
        oracle_steps = []
        for episode in range(_ORACLE_EPISODES):
            info, steps = _run_episode(env, oracle, seed=episode)
            oracle_successes.append(bool(info["success"]))
            oracle_steps.append(steps)
        baseline = [
            bool(_run_episode(env, oracle, seed=episode, drop_right=True)[0]["success"])
            for episode in range(_ORACLE_EPISODES)
        ]
        assert sum(oracle_successes) >= _ORACLE_EPISODES - 1  # the scripted upper bound
        assert not any(baseline)  # the one-sided absent-agent baseline cannot do the task
        assert sum(oracle_successes) > sum(baseline)
        assert all(steps < env.config.max_control_steps for steps in oracle_steps)
    finally:
        env.close()


def test_cartpole_terminal_reward_semantics() -> None:
    pytest.importorskip("mujoco")
    cfg = coop_config("cartpole")
    env = _make_env("cartpole", config=cfg)
    try:
        oracle = _make_oracle("cartpole", env.config)
        env.reset(seed=0)
        reward = 0.0
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            _obs, reward, terminated, truncated, info = env.step(oracle.act(env))
        assert info["success"] is True and terminated and not truncated
        assert reward == pytest.approx(cfg.reward.success_bonus - cfg.reward.step_penalty)
    finally:
        env.close()

    env = _make_env("cartpole", config=cfg)
    try:
        env.reset(seed=0)
        action = _zero_action(env)
        reward = 0.0
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            _obs, reward, terminated, truncated, info = env.step(action)
        assert info["success"] is False and terminated and not truncated
        assert reward == pytest.approx(cfg.reward.failure_penalty)
    finally:
        env.close()


def test_pendulum_success_needs_the_hold_cycles() -> None:
    pytest.importorskip("mujoco")
    cfg = coop_config("pendulum")
    env = _make_env("pendulum", config=cfg)
    try:
        oracle = _make_oracle("pendulum", env.config)
        env.reset(seed=1)
        max_amplitude = 0.0
        steps = 0
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            _obs, _reward, terminated, truncated, info = env.step(oracle.act(env))
            max_amplitude = max(max_amplitude, float(info["amplitude_deg"]))
            steps += 1
        assert info["success"] is True
        assert info["completed_cycles"] >= cfg.hold_cycles
        # The success is a HOLD: at least one half-cycle peak really reached the target
        # amplitude (the current half-cycle peak is small again right after a crossing).
        assert max_amplitude >= np.degrees(cfg.target_amplitude)
        assert steps < cfg.max_control_steps
    finally:
        env.close()


def test_vec_coop_contract_and_auto_reset() -> None:
    pytest.importorskip("mujoco")
    from callosum.training._vec_coop import VecCoopEnv

    cfg = coop_config("pendulum")
    vec = VecCoopEnv("pendulum", 2, config=cfg, seed=0)
    try:
        assert vec.obs_dim == task_obs_size("pendulum", "none")
        assert vec.action_dim == 1
        obs = vec.reset(seed=3)
        assert {key: value.shape for key, value in obs.items()} == {
            key: (2, vec.obs_dim) for key in AGENT_KEYS
        }
        actions = {key: np.zeros((2, 1), dtype=np.float32) for key in AGENT_KEYS}
        result = vec.step(actions)
        assert result.obs[AGENT_KEYS[0]].shape == (2, vec.obs_dim)
        assert result.reward.shape == (2,)
        assert not result.done.any()
        # With zero commands the pendulum never reaches the target: the episode runs out of
        # its budget, so the vec env reports the truncation and auto-resets the worlds.
        steps_taken = 1
        while not result.done.any():
            result = vec.step(actions)
            steps_taken += 1
            assert steps_taken <= cfg.max_control_steps
        assert result.truncated.all()
        assert result.finished_lengths == [cfg.max_control_steps, cfg.max_control_steps]
        assert result.finished_successes == [False, False]
        assert len(result.finished_angles) == 2
        # The returned observation is the fresh episode (length 0), not the terminal one.
        assert result.obs[AGENT_KEYS[0]].shape == (2, vec.obs_dim)
        with pytest.raises(ValueError, match="shape"):
            vec.step({key: np.zeros((3, 1), dtype=np.float32) for key in AGENT_KEYS})
        with pytest.raises(ValueError, match="action keys"):
            vec.step({"left": np.zeros((2, 1), dtype=np.float32)})
    finally:
        vec.close()


@pytest.mark.parametrize(
    ("task", "state_keys"),
    [
        ("cartpole", {"cart_x", "cart_vx", "pole_angle", "pole_angvel", "target_x"}),
        ("pendulum", {"angle", "angvel", "target_amplitude", "peak_abs_angle", "completed_cycles"}),
    ],
)
def test_state_exposes_the_oracle_fields(task: str, state_keys: set[str]) -> None:
    pytest.importorskip("mujoco")
    env = _make_env(task)
    try:
        env.reset(seed=0)
        state = env.state()
        assert state_keys <= set(state)
        assert state["step"] == 0.0
        assert env.randomization is not None
    finally:
        env.close()
