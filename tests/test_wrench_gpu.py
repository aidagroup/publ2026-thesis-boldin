"""Tests of the optional batched GPU backend of CubeWrench-v0 (`mujoco_warp`).

Two layers, so the Mac/CI run stays green without the GPU stack:

* **no GPU**: the trainer config's `backend` validation (pure stdlib) and -- under the
  locally installed torch+mujoco -- the pure-tensor pieces of the backend: the
  randomisation packer vs the CPU sampler, and the observation block mapping vs the pure
  `_wrench_logic.build_observation` layout. These import `callosum.envs.wrench_cube` (which
  needs `mujoco`), so they skip where the simulator extra is absent;
* **GPU**: the batch env itself (`WrenchCubeBatchGPU`) is only built when `mujoco_warp`
  imports AND a CUDA device exists; the shape/validation/one-step-parity tests skip
  otherwise. The full multi-step parity and the throughput numbers are the pod scripts
  (`scripts/check_wrench_gpu_parity.py`, `scripts/bench_wrench_gpu.py`), not pytest.

CI installs only the `dev` extra (no torch, no mujoco, no warp), so every test here is a
skip there; locally (`make dev` + the macOS `sandbox` extra) the GPU cases skip and the rest
runs.
"""

import dataclasses
import math

import numpy as np
import pytest

from callosum.configs.wrench_ppo import BACKENDS, WrenchPPOConfig, parse_args

# --- config / CLI (no torch, no simulator) ------------------------------------------------


def test_default_backend_is_the_cpu_loop() -> None:
    cfg = WrenchPPOConfig()
    assert cfg.backend == "cpu"
    assert cfg.device == "cpu"
    assert cfg.backend in BACKENDS
    # The backend does not change the env contract/batch layout: a checkpoint stays
    # loadable across backends because the policy input and action widths are identical.
    warp = WrenchPPOConfig(backend="warp", device="cuda")
    assert warp.policy_input_size == cfg.policy_input_size == cfg.obs_size
    assert warp.batch_size == cfg.batch_size and warp.minibatch_size == cfg.minibatch_size


def test_warp_backend_requires_a_cuda_device() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        WrenchPPOConfig(backend="warp")
    with pytest.raises(ValueError, match="CUDA"):
        WrenchPPOConfig(backend="warp", device="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        WrenchPPOConfig(backend="warp", device="mps")
    assert WrenchPPOConfig(backend="warp", device="cuda").device == "cuda"
    assert WrenchPPOConfig(backend="warp", device="cuda:1").device == "cuda:1"


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        WrenchPPOConfig(backend="gpu")
    values = dataclasses.asdict(WrenchPPOConfig())
    values["backend"] = "warp"
    values["device"] = "cuda"
    assert WrenchPPOConfig(**values).backend == "warp"


def test_cli_exposes_the_backend_and_device_flags() -> None:
    cfg = parse_args(
        ["--backend", "warp", "--device", "cuda", "--num-envs", "1024", "--rollout-steps", "64"]
    )
    assert cfg.backend == "warp"
    assert cfg.device == "cuda"
    assert cfg.num_envs == 1024
    assert parse_args([]).backend == "cpu"


# --- pure-tensor backend pieces (torch + mujoco, no warp) ----------------------------------


def _gpu_module():
    """Import `env_gpu` (and its package, i.e. mujoco) or skip without the simulator extra."""
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    from callosum.envs.wrench_cube import env_gpu

    return env_gpu


def _values_dict(seed: int = 0) -> dict[str, float]:
    """A role-tagged values dict like the CPU env's `_values` (all fields distinct)."""
    from callosum.configs.wrench_cube import WrenchCubeConfig
    from callosum.envs._wrench_logic import (
        AGENT_KEYS,
        CUBE_STATE_FIELDS,
        command_value_fields,
        param_value_fields,
        privileged_value_fields,
        sample_episode_randomization,
        wrench_value_fields,
    )

    config = WrenchCubeConfig()
    randomization = sample_episode_randomization(config, seed)
    values: dict[str, float] = {}
    for index, key in enumerate(AGENT_KEYS):
        agent = randomization.agents[index]
        values.update(zip(wrench_value_fields(key), np.linspace(-1.0, 1.0, 6) + index))
        values.update(zip(command_value_fields(key), np.linspace(-0.5, 0.5, 6) - index))
        values.update(
            zip(param_value_fields(key), (agent.latency_s, agent.gain, agent.limit_scale))
        )
        values.update(zip(privileged_value_fields(key), (agent.start_delay_s, float(index + 1))))
    values.update({name: (index - 7.0) / 10.0 for index, name in enumerate(CUBE_STATE_FIELDS)})
    values["goal_angle"] = config.goal_angle
    values["time_fraction"] = 0.25
    return values


@pytest.mark.parametrize("partner_mode", ("none", "oracle"))
def test_observation_blocks_reproduce_the_pure_layout(partner_mode: str) -> None:
    """Concatenating the GPU block tensors must give exactly `build_observation`'s vector."""
    env_gpu = _gpu_module()
    from callosum.envs._wrench_logic import (
        AGENT_KEYS,
        CUBE_STATE_FIELDS,
        GOAL_FIELDS,
        build_observation,
        command_value_fields,
        param_value_fields,
        privileged_value_fields,
        wrench_value_fields,
    )

    values = _values_dict()
    blocks = env_gpu.observation_blocks(partner_mode)
    assert set(blocks) == set(AGENT_KEYS)
    field_source = {
        "wrench": wrench_value_fields,
        "command": command_value_fields,
        "params": param_value_fields,
        "privileged": privileged_value_fields,
    }
    shared_fields = {"cube": CUBE_STATE_FIELDS, "goal": GOAL_FIELDS}
    for role in AGENT_KEYS:
        columns: list[float] = []
        for key, owner in blocks[role]:
            if owner is None:
                names = shared_fields[key]
            else:
                names = field_source[key](owner)
            columns.extend(values[name] for name in names)
        expected = build_observation(values, role, partner_mode)
        actual = np.asarray(columns, dtype=np.float32)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_randomization_packing_matches_the_cpu_sampler() -> None:
    """The GPU tensor pack holds exactly the CPU sampler's values, field by field."""
    env_gpu = _gpu_module()
    from callosum.configs.wrench_cube import WrenchCubeConfig
    from callosum.envs._wrench_logic import AGENT_KEYS, sample_episode_randomization

    config = WrenchCubeConfig()
    seeds = [3, 11, 29]
    records = [sample_episode_randomization(config, seed) for seed in seeds]
    agents, episode = env_gpu.pack_episode_randomization(records)
    assert set(agents) == set(AGENT_KEYS)
    for role_index, role in enumerate(AGENT_KEYS):
        for name, tensor in agents[role].items():
            expected = np.asarray([getattr(record.agents[role_index], name) for record in records])
            actual = tensor.cpu().numpy()
            assert actual.shape == expected.shape
            if expected.dtype == np.bool_:
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
    for name, tensor in episode.items():
        if name == "impulse_force":
            expected = np.asarray([record.impulse_force for record in records])
        else:
            expected = np.asarray([getattr(record, name) for record in records])
        actual = tensor.cpu().numpy()
        assert actual.shape == expected.shape
        if expected.dtype == np.bool_:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
    with pytest.raises(ValueError, match="at least one"):
        env_gpu.pack_episode_randomization([])


# --- the batch env on a CUDA device -------------------------------------------------------


def _gpu_env(nworld: int, partner_mode: str = "none"):
    """Build a `WrenchCubeBatchGPU` or skip when the GPU stack/CUDA is unavailable."""
    env_gpu = _gpu_module()
    pytest.importorskip("mujoco_warp")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return env_gpu, env_gpu.WrenchCubeBatchGPU(
        nworld, partner_mode=partner_mode, seed=0, device="cuda"
    )


def test_batch_env_shapes_validation_and_one_step() -> None:
    import torch

    from callosum.envs._wrench_logic import AGENT_KEYS, obs_size

    _env_gpu, env = _gpu_env(4, partner_mode="oracle")
    try:
        obs = env.reset(seed=5)
        for role in AGENT_KEYS:
            assert obs[role].shape == (4, obs_size("oracle"))
            assert obs[role].dtype == torch.float32
            assert obs[role].is_cuda
            assert torch.isfinite(obs[role]).all()
        state = env.state()
        assert state["qpos"].shape == (4, 8)
        assert state["qvel"].shape == (4, 7)
        # The CUDA-graph capture is what makes the GPU backend faster than the CPU loop; a
        # platform where capture fails must at least record why (and still step correctly).
        assert env.cuda_graph or env._cuda_graph_error is not None

        actions = {role: torch.zeros((4, 6), device="cuda") for role in AGENT_KEYS}
        step = env.step(actions)
        assert step.reward.shape == (4,)
        assert step.reward.dtype == torch.float32
        assert step.terminated.shape == (4,) and step.terminated.dtype == torch.bool
        assert step.truncated.shape == (4,) and step.truncated.dtype == torch.bool
        for role in AGENT_KEYS:
            assert step.obs[role].shape == (4, obs_size("oracle"))
            assert step.final_obs[role].shape == (4, obs_size("oracle"))
        assert step.finished_returns == [] and step.finished_lengths == []

        # numpy actions are accepted too (the eval/trainer CPU-style call path).
        numpy_actions = np.zeros((4, 6), dtype=np.float32)
        step = env.step({role: numpy_actions for role in AGENT_KEYS})
        assert step.reward.shape == (4,)

        with pytest.raises(ValueError, match="action keys"):
            env.step(
                {
                    "holder": torch.zeros((4, 6)),
                    "rotator": torch.zeros((4, 6)),
                    "holder_extra": torch.zeros((4, 6)),
                }
            )
        with pytest.raises(ValueError, match="shape"):
            env.step(
                {
                    "holder": torch.zeros((4, 5), device="cuda"),
                    "rotator": torch.zeros((4, 6), device="cuda"),
                }
            )
        with pytest.raises(TypeError, match="dict"):
            env.step(torch.zeros((4, 6)))
    finally:
        env.close()


def test_batch_env_rejects_non_cuda_and_bad_world_count() -> None:
    # These errors are raised before the Warp stack is imported, so the check also runs on a
    # machine that has torch+mujoco but no mujoco_warp.
    env_gpu = _gpu_module()
    with pytest.raises(ValueError, match="nworld"):
        env_gpu.WrenchCubeBatchGPU(0, device="cuda")
    with pytest.raises(ValueError, match="CUDA"):
        env_gpu.WrenchCubeBatchGPU(2, device="cpu")
    with pytest.raises(ValueError, match="partner_mode"):
        env_gpu.WrenchCubeBatchGPU(2, partner_mode="predicted", device="cuda")


def test_batch_env_matches_the_cpu_env_one_step() -> None:
    """One-step parity sentinel (the full multi-step check is the pod parity script)."""
    import torch

    from callosum.configs.wrench_cube import WrenchCubeConfig
    from callosum.envs._wrench_logic import AGENT_KEYS
    from callosum.envs.wrench_cube import WrenchCubeEnv

    _env_gpu, env = _gpu_env(2, partner_mode="oracle")
    config = WrenchCubeConfig()
    cpu_envs = [WrenchCubeEnv(config=config, partner_mode="oracle") for _ in range(2)]
    try:
        cpu_obs = [cpu_env.reset(seed=10 + index)[0] for index, cpu_env in enumerate(cpu_envs)]
        gpu_obs = env.reset(seed=10)
        for role in AGENT_KEYS:
            expected = np.stack([obs[role] for obs in cpu_obs])
            np.testing.assert_allclose(gpu_obs[role].cpu().numpy(), expected, rtol=0.0, atol=1e-4)
        torch.manual_seed(1234)
        raw = (torch.rand((2, 6)) * 2.0 - 1.0).numpy().astype(np.float32)
        cpu_steps = [
            cpu_env.step({role: raw[index].astype(np.float64) for role in AGENT_KEYS})
            for index, cpu_env in enumerate(cpu_envs)
        ]
        step = env.step({role: torch.as_tensor(raw, device="cuda") for role in AGENT_KEYS})
        done = np.asarray([result[2] or result[3] for result in cpu_steps])
        alive = ~done
        if alive.any():
            cpu_qpos = []
            for cpu_env in cpu_envs:
                state = cpu_env.state()
                cpu_qpos.append(
                    np.concatenate([state["core_pos"], state["core_quat"], [state["layer_angle"]]])
                )
            gpu_qpos = env.state()["qpos"].cpu().numpy()
            np.testing.assert_allclose(
                gpu_qpos[alive], np.asarray(cpu_qpos)[alive], rtol=0.0, atol=1e-5
            )
        assert math.isclose(
            float(step.reward.sum()), float(sum(r[1] for r in cpu_steps)), abs_tol=1e-4
        )
    finally:
        env.close()


# --- trainer / evaluator wiring (torch, no simulator) --------------------------------------


def test_checkpoint_keeps_the_backend_field(tmp_path) -> None:
    """A checkpoint records the backend config; the weights/policy widths stay shared."""
    torch = pytest.importorskip("torch")
    from callosum.training.wrench_ppo import (
        config_from_payload,
        load_checkpoint,
        make_policies,
        save_checkpoint,
    )

    cfg = WrenchPPOConfig(
        backend="warp",
        device="cuda",
        num_envs=4,
        rollout_steps=8,
        num_minibatches=2,
        total_updates=1,
    )
    policies = make_policies("none")
    optimizers = [
        torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate) for policy in policies
    ]
    path = tmp_path / "latest.pt"
    save_checkpoint(path, policies, optimizers, cfg, 1, cfg.batch_size)
    payload = load_checkpoint(path)
    restored = config_from_payload(payload)
    assert restored.backend == "warp" and restored.device == "cuda"
    assert payload["config"]["backend"] == "warp"
    assert payload["policy_input_size"] == cfg.policy_input_size


def test_evaluate_greedy_hands_cuda_style_tensors_to_a_torch_io_env() -> None:
    """The rollout/eval branch keys off `torch_io`, so a GPU env gets tensors, not numpy."""
    import types

    torch = pytest.importorskip("torch")
    from callosum.envs._wrench_logic import AGENT_KEYS
    from callosum.training.wrench_ppo import evaluate_greedy, make_policies

    class _FakeTorchVec:
        torch_io = True

        def __init__(self, num_envs: int, obs_dim: int, episode_steps: int) -> None:
            self.num_envs = num_envs
            self.obs_dim = obs_dim
            self.config = types.SimpleNamespace(max_control_steps=episode_steps)
            self._episode_steps = episode_steps
            self._elapsed = 0
            self.seen_tensor_actions = True

        def reset(self, seed: int | None = None) -> dict:
            del seed
            self._elapsed = 0
            return self._obs()

        def _obs(self) -> dict:
            return {name: torch.zeros(self.num_envs, self.obs_dim) for name in AGENT_KEYS}

        def step(self, actions: dict) -> types.SimpleNamespace:
            assert all(torch.is_tensor(value) for value in actions.values())
            self._elapsed += 1
            done = self._elapsed >= self._episode_steps
            finished = self.num_envs if done else 0
            return types.SimpleNamespace(
                obs=self._obs(),
                reward=torch.zeros(self.num_envs),
                terminated=torch.zeros(self.num_envs, dtype=torch.bool),
                truncated=torch.zeros(self.num_envs, dtype=torch.bool),
                final_obs=self._obs(),
                finished_returns=[1.0] * finished,
                finished_lengths=[self._episode_steps] * finished,
                finished_successes=[True] * finished,
                finished_angles=[90.0] * finished,
            )

    policies = make_policies("none")
    vec = _FakeTorchVec(num_envs=2, obs_dim=28, episode_steps=3)
    summary = evaluate_greedy(policies, vec, episodes=2)
    assert summary.episodes == 2
    assert summary.success_rate == 1.0
