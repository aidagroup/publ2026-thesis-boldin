"""MuJoCo-guarded tests of the two-agent coop envs (all three suite tasks).

The environments need the `sandbox` extra (mujoco), which CI does not install, so every test
here starts with `pytest.importorskip("mujoco")` (the same pattern as the simulator-side
checks of `tests/test_wrench_logic.py`): CI skips them, the macOS sandbox venv runs them, and
`scripts/experiments/coop_tasks/smoke_coop.py` covers the scripted oracle/baseline rollouts on a machine with the
simulator.

What is checked: the Gymnasium spaces and observation widths (against the pure
`callosum.experiments.coop_tasks.logic` layout), the config-as-truth model writes and the
sampled-randomisation reset, seed determinism and reset-friendliness (no hidden state across
episodes), the deterministic scripted oracle reaching the task and the absent-agent
(`only-left`) baseline failing it, the terminal reward cases, the pendulum hold-cycle success,
the toss-catch pass/tilt/floor rules, and the `_vec_coop` auto-reset / vec contract the
trainer relies on.
"""

import numpy as np
import pytest

from callosum.experiments.coop_tasks.config import coop_config
from callosum.experiments.coop_tasks.logic import AGENT_KEYS, obs_size, task_obs_size

_ALL_TASKS = ("cartpole", "pendulum", "toss")

_ORACLE_EPISODES = 4
"""Small local episode count for the oracle/baseline checks (the envs are CPU MuJoCo)."""

_TOSS_ORACLE_EPISODES = 6
"""The toss-catch oracle is the suite's hardest script (contact/timing): a few extra local
episodes keep the "scripted upper bound reaches the task" check meaningful."""


def _make_env(task: str, **kwargs):
    """Build a coop env (imported lazily so the module imports without mujoco)."""
    from callosum.experiments.coop_tasks import make_coop_env

    return make_coop_env(task, **kwargs)


def _make_oracle(task: str, config):
    """Build the task's scripted oracle (imported lazily)."""
    from callosum.experiments.coop_tasks import make_oracle

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


@pytest.mark.parametrize("task", _ALL_TASKS)
def test_spaces_and_observation_widths(task: str) -> None:
    pytest.importorskip("mujoco")
    env = _make_env(task)
    action_dim = coop_config(task).action_size
    try:
        assert set(env.action_space.spaces) == set(AGENT_KEYS)
        for key in AGENT_KEYS:
            assert env.action_space[key].shape == (action_dim,)
            assert env.observation_space[key].shape == (task_obs_size(task, "none"),)
        obs, info = env.reset(seed=0)
        assert set(obs) == set(AGENT_KEYS)
        for vector in obs.values():
            assert vector.dtype == np.float32
            assert vector.shape == (task_obs_size(task, "none"),)
            assert np.all(np.isfinite(vector))
        assert info["success"] is False
        expected_info = ("ball_z", "passes", "time") if task == "toss" else ("angle_deg", "time")
        for key in expected_info:
            assert key in info
        # The oracle partner block is `2 * action_dim + 4` wide.
        partner_width = 2 * action_dim + 4
        oracle_env = _make_env(task, partner_mode="oracle")
        try:
            assert oracle_env.observation_space[AGENT_KEYS[0]].shape == (
                task_obs_size(task, "oracle"),
            )
            assert task_obs_size(task, "oracle") == task_obs_size(task, "none") + partner_width
            assert obs_size(env._task_spec, "oracle") == task_obs_size(task, "oracle")
        finally:
            oracle_env.close()
    finally:
        env.close()


@pytest.mark.parametrize("task", _ALL_TASKS)
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
        elif task == "pendulum":
            hinge = model.joint("pivot_hinge")
            assert tuple(model.jnt_range[hinge.id]) == pytest.approx(
                (-cfg.hinge_limit, cfg.hinge_limit)
            )
            assert model.dof_armature[env._hinge_dof] == pytest.approx(cfg.hinge_armature)
        applied = []
        for seed in range(6):
            env.reset(seed=seed)
            if task == "toss":
                sampled = env.randomization.contact_dampratio
                assert (
                    cfg.randomization.contact_dampratio_range[0]
                    <= sampled
                    <= cfg.randomization.contact_dampratio_range[1]
                )
                for geom_id in (env._ball_geom, *env._pad_geom.values()):
                    assert model.geom_solref[geom_id, 1] == pytest.approx(sampled)
                    assert model.geom_friction[geom_id, 0] == pytest.approx(
                        env.randomization.contact_friction
                    )
            else:
                sampled = env.randomization.damping
                assert (
                    cfg.randomization.damping_range[0]
                    <= sampled
                    <= cfg.randomization.damping_range[1]
                )
                hinge_dof = env._pole_dof if task == "cartpole" else env._hinge_dof
                assert model.dof_damping[hinge_dof] == pytest.approx(sampled)
            applied.append(sampled)
        assert len(set(applied)) > 1  # different seeds -> different applied randomness
    finally:
        env.close()


@pytest.mark.parametrize("task", _ALL_TASKS)
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
            if task == "toss":
                seeds.add(round(env.randomization.contact_dampratio, 12))
            else:
                seeds.add(round(env.randomization.damping, 12))
        assert len(seeds) > 1
    finally:
        env.close()


@pytest.mark.parametrize("task", _ALL_TASKS)
def test_pending_commands_are_the_latency_queues(task: str) -> None:
    pytest.importorskip("mujoco")
    env = _make_env(task)
    action_dim = coop_config(task).action_size
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
                assert command.shape == (action_dim,)
                assert np.allclose(command, 0.0)  # the queues start with zeros
        env.step(_zero_action(env))
        assert {key: len(value) for key, value in env.pending_commands().items()} == {
            key: latency[index] for index, key in enumerate(AGENT_KEYS)
        }
    finally:
        env.close()


@pytest.mark.parametrize("task", _ALL_TASKS)
def test_oracle_succeeds_and_only_left_baseline_fails(task: str) -> None:
    pytest.importorskip("mujoco")
    env = _make_env(task)
    episodes = _TOSS_ORACLE_EPISODES if task == "toss" else _ORACLE_EPISODES
    try:
        oracle = _make_oracle(task, env.config)
        oracle_successes = []
        oracle_steps = []
        for episode in range(episodes):
            info, steps = _run_episode(env, oracle, seed=episode)
            oracle_successes.append(bool(info["success"]))
            oracle_steps.append(steps)
        baseline = [
            bool(_run_episode(env, oracle, seed=episode, drop_right=True)[0]["success"])
            for episode in range(episodes)
        ]
        assert sum(oracle_successes) >= episodes - 1  # the scripted upper bound
        assert not any(baseline)  # the absent-agent baseline cannot do the task
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
    from callosum.experiments.coop_tasks.vec_env import VecCoopEnv

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
        (
            "toss",
            {
                "ball_x",
                "ball_z",
                "ball_vx",
                "ball_vz",
                "passes",
                "target_passes",
                "left_pad_x",
                "left_pad_z",
                "left_pad_tilt",
                "right_pad_x",
                "right_pad_z",
                "right_pad_tilt",
            },
        ),
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


# --- toss-catch specific checks -----------------------------------------------------------

_TOSS_ZERO_STEP = 1e-9


def _toss_zero_action(env) -> dict[str, np.ndarray]:
    """The all-zero (do-nothing) toss-catch action (all three dims)."""
    return {key: np.zeros(env.action_space[key].shape, dtype=np.float32) for key in AGENT_KEYS}


def test_toss_floor_failure_is_a_terminal_failure() -> None:
    pytest.importorskip("mujoco")
    cfg = coop_config("toss")
    env = _make_env("toss", config=cfg)
    try:
        env.reset(seed=0)
        action = _toss_zero_action(env)
        reward = 0.0
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            _obs, reward, terminated, truncated, info = env.step(action)
        # The ball is still released (the hidden impulse does not depend on the actions) and
        # falls to the floor because no paddle moves: a true failure, not a timeout.
        assert terminated and not truncated
        assert info["success"] is False
        assert info["ball_z"] < cfg.floor_z
        assert reward == pytest.approx(cfg.reward.failure_penalty)
    finally:
        env.close()


def test_toss_oracle_returns_the_success_reward() -> None:
    pytest.importorskip("mujoco")
    cfg = coop_config("toss")
    env = _make_env("toss", config=cfg)
    try:
        oracle = _make_oracle("toss", env.config)
        env.reset(seed=0)
        reward = 0.0
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            _obs, reward, terminated, truncated, info = env.step(oracle.act(env))
        # The scripted oracle completes `n_passes` alternating catches well inside the budget.
        assert terminated and not truncated
        assert info["success"] is True
        assert info["passes"] >= cfg.n_passes
        assert reward == pytest.approx(cfg.reward.success_bonus - cfg.reward.step_penalty)
    finally:
        env.close()


def test_toss_contact_chatter_cannot_race_the_pass_counter() -> None:
    """The pass gate: alternating contact edges alone cannot count rapid passes.

    This is the exploit sentinel for the old rule (any "other side" contact edge counted):
    a ball chattering between the two near-centre paddle faces used to reach `n_passes` in a
    few control steps. The env must now count at most one pass per
    `min_pass_interval_steps` control steps (and only with the ball on the contact side's
    half, which this scripted pattern satisfies).
    """
    pytest.importorskip("mujoco")
    cfg = coop_config("toss")
    env = _make_env("toss", config=cfg)
    try:
        env.reset(seed=0)
        call = {"substep": 0}

        def fake_contacts(side: str) -> bool:
            """Alternating contact edges: left on even substeps, right on odd ones."""
            even = call["substep"] % 2 == 0
            return even if side == "left" else not even

        env._pad_contacts_ball = fake_contacts  # type: ignore[method-assign]
        passes_per_step: list[float] = []
        for step in range(3 * cfg.min_pass_interval_steps):
            env._step_index = step
            for substep in range(cfg.control_every):
                call["substep"] = substep
                # The ball stays just across the centre, on whichever side contacts next.
                env._data.qpos[env._ball_qpos] = -0.05 if substep % 2 == 0 else 0.05
                env._on_substep(substep)
            passes_per_step.append(float(env._passes))
        interval = cfg.min_pass_interval_steps
        assert passes_per_step == [1.0 + step // interval for step in range(3 * interval)], (
            passes_per_step
        )
    finally:
        env.close()


def test_toss_floor_failure_beats_a_simultaneous_success() -> None:
    """Failure priority: enough passes with the ball below the floor is a FAILURE step."""
    pytest.importorskip("mujoco")
    cfg = coop_config("toss")
    env = _make_env("toss", config=cfg)
    try:
        env.reset(seed=0)
        # The exploit's end state: the pass target is reached, but the ball is below the floor.
        env._passes = cfg.n_passes
        env._data.qpos[env._ball_qpos + 2] = cfg.floor_z - 0.1
        assert env._success() is False
        assert env._failure() is True
        _obs, reward, terminated, truncated, info = env.step(_toss_zero_action(env))
        assert terminated and not truncated
        assert info["success"] is False
        assert reward == pytest.approx(cfg.reward.failure_penalty)
        assert info["ball_z"] < cfg.floor_z
    finally:
        env.close()


def test_toss_success_requires_the_ball_above_the_floor_margin() -> None:
    """Success is a conjunction with "the ball is still in play", not just the pass count."""
    pytest.importorskip("mujoco")
    cfg = coop_config("toss")
    env = _make_env("toss", config=cfg)
    try:
        env.reset(seed=0)
        env._passes = cfg.n_passes
        ball_z = env._data.qpos[env._ball_qpos + 2]
        env._data.qpos[env._ball_qpos + 2] = cfg.floor_z + cfg.success_floor_margin - 1e-3
        assert not env._failure()
        assert env._success() is False
        env._data.qpos[env._ball_qpos + 2] = cfg.floor_z + cfg.success_floor_margin + 1e-3
        assert not env._failure()
        assert env._success() is True
        # A court failure is not a success even high above the floor.
        env._data.qpos[env._ball_qpos + 2] = ball_z
        env._data.qpos[env._ball_qpos] = cfg.court_x + 0.01
        assert env._failure() is True
        assert env._success() is False
    finally:
        env.close()


def test_toss_actions_move_and_tilt_the_paddle() -> None:
    pytest.importorskip("mujoco")
    cfg = coop_config("toss")
    env = _make_env("toss", config=cfg)
    try:
        env.reset(seed=0)
        start = env.state()
        action = _toss_zero_action(env)
        # Up to 8 steps of the full +x / +tilt command: the right paddle must translate in
        # +x and its tilt must grow, both inside the config's workspace/tilt limits.
        action["right"] = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        for _ in range(8):
            env.step(action)
        state = env.state()
        assert state["right_pad_x"] > start["right_pad_x"] + _TOSS_ZERO_STEP
        assert state["right_pad_x"] <= cfg.pad_outer_x + _TOSS_ZERO_STEP
        assert state["right_pad_tilt"] > start["right_pad_tilt"] + _TOSS_ZERO_STEP
        assert abs(state["right_pad_tilt"]) <= cfg.pad_tilt_limit + _TOSS_ZERO_STEP
        # The vertical command is bounded too: the pad never leaves its z workspace.
        assert (
            cfg.pad_min_z - _TOSS_ZERO_STEP
            <= state["right_pad_z"]
            <= (cfg.pad_max_z + _TOSS_ZERO_STEP)
        )
    finally:
        env.close()


def test_toss_trajectory_is_reproducible_from_the_seed() -> None:
    pytest.importorskip("mujoco")
    cfg = coop_config("toss")
    env = _make_env("toss", config=cfg)
    try:
        actions = {
            "left": np.array([0.5, 0.5, 0.2], dtype=np.float32),
            "right": np.array([-0.5, 0.5, -0.2], dtype=np.float32),
        }
        trajectories = []
        for _ in range(2):
            env.reset(seed=11)
            positions = []
            for _step in range(10):
                env.step(actions)
                state = env.state()
                positions.append((state["ball_x"], state["ball_z"]))
            trajectories.append(positions)
        assert trajectories[0] == trajectories[1]
    finally:
        env.close()


def test_vec_coop_toss_contract_and_finished_diagnostic() -> None:
    pytest.importorskip("mujoco")
    from callosum.experiments.coop_tasks.vec_env import VecCoopEnv

    cfg = coop_config("toss")
    vec = VecCoopEnv("toss", 2, config=cfg, seed=0)
    try:
        assert vec.obs_dim == task_obs_size("toss", "none") == 26
        assert vec.action_dim == 3
        obs = vec.reset(seed=5)
        assert {key: value.shape for key, value in obs.items()} == {
            key: (2, vec.obs_dim) for key in AGENT_KEYS
        }
        actions = {key: np.zeros((2, 3), dtype=np.float32) for key in AGENT_KEYS}
        result = vec.step(actions)
        # Zero paddle commands cannot catch the ball: both worlds finish with a failure and
        # the vec env reports the task's primary diagnostic (the pass count) for the trainer.
        while not result.done.any():
            result = vec.step(actions)
        assert result.terminated.all()
        assert result.finished_successes == [False, False]
        assert result.finished_angles == [0.0, 0.0]
        assert len(result.finished_lengths) == 2
        assert all(length < cfg.max_control_steps for length in result.finished_lengths)
        with pytest.raises(ValueError, match="shape"):
            vec.step({key: np.zeros((3, 3), dtype=np.float32) for key in AGENT_KEYS})
    finally:
        vec.close()
