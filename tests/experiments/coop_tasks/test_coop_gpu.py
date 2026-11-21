"""Tests of the optional batched GPU backend of the coop suite (`mujoco_warp`).

Two layers, so the Mac/CI run stays green without the GPU stack:

* **no GPU**: the task-dependent `--backend` default (pure stdlib), the trainer config's
  backend validation, the pure-tensor randomisation packer vs the CPU samplers, and the
  observation-block mapping vs the pure `callosum.experiments.coop_tasks.logic.obs_fields`
  layout. These need no simulator;
* **GPU**: `CoopBatchGPU` itself is only built when `mujoco_warp` imports, `mujoco` imports
  and a CUDA device exists; the shape/validation/determinism/one-step-parity tests skip
  otherwise. The full multi-step parity and the throughput numbers are the pod scripts
  (`scripts/experiments/coop_tasks/check_coop_gpu_parity.py`,
  `scripts/experiments/coop_tasks/bench_coop_gpu.py`), not pytest.

CI installs only the `dev` extra (no torch, no mujoco, no warp), so every test here is a skip
there; locally (`make dev` + the macOS `sandbox` extra) the GPU cases skip and the rest runs.
"""

import dataclasses

import numpy as np
import pytest

from callosum.experiments.coop_tasks.config import coop_config
from callosum.experiments.coop_tasks.env_gpu import observation_blocks
from callosum.experiments.coop_tasks.logic import (
    AGENT_KEYS,
    obs_fields,
    sample_episode_randomization,
    sample_toss_randomization,
    task_spec,
)
from callosum.experiments.cube_wrench.ppo_config import (
    BACKENDS,
    WrenchPPOConfig,
    default_backend,
    parse_args,
    parse_args_with_explicit,
)

_ALL_TASKS = ("cartpole", "pendulum", "toss")


# --- config / CLI (no torch, no simulator) ------------------------------------------------


def test_cube_backend_default_stays_on_the_cpu_loop() -> None:
    """The historical cube path must not move: its CLI default stays the CPU backend."""
    assert default_backend("cube") == "cpu"
    cfg = WrenchPPOConfig()
    assert cfg.backend == "cpu" and cfg.device == "cpu"
    assert cfg.backend in BACKENDS
    assert parse_args([]).backend == "cpu"
    assert parse_args(["--task", "cube"]).backend == "cpu"


def test_coop_backend_default_is_the_gpu_when_cuda_is_visible() -> None:
    """A coop run defaults to the batched GPU backend on a CUDA host, else to the CPU loop."""
    from callosum.experiments.cube_wrench.ppo_config import cuda_visible

    for task in _ALL_TASKS:
        assert default_backend(task) == ("warp" if cuda_visible() else "cpu")
        cfg, explicit = parse_args_with_explicit(["--task", task])
        if cuda_visible():
            assert cfg.backend == "warp" and cfg.device == "cuda"
            assert {"backend", "device"} <= explicit
        else:
            assert cfg.backend == "cpu"
        # An explicit flag always wins over the task-dependent default.
        explicit_cpu = parse_args(["--task", task, "--backend", "cpu"])
        assert explicit_cpu.backend == "cpu" and explicit_cpu.device == "cpu"


def test_warp_backend_for_a_coop_task_requires_a_cuda_device() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        WrenchPPOConfig(task="cartpole", backend="warp")
    with pytest.raises(ValueError, match="CUDA"):
        WrenchPPOConfig(task="toss", backend="warp", device="cpu")
    for task in _ALL_TASKS:
        cfg = WrenchPPOConfig(task=task, backend="warp", device="cuda")
        assert cfg.backend == "warp"
        # The backend is a runtime resource: it does not change the policy/checkpoint widths.
        assert cfg.policy_input_size == cfg.obs_size
        assert cfg.action_size == coop_config(task).action_size


def test_coop_randomization_preset_is_still_default_only() -> None:
    with pytest.raises(ValueError, match="randomization"):
        WrenchPPOConfig(task="cartpole", randomization="harsh")
    assert WrenchPPOConfig(task="cartpole").randomization == "default"


# --- pure observation / randomisation logic -----------------------------------------------


@pytest.mark.parametrize("task", _ALL_TASKS)
@pytest.mark.parametrize("partner_mode", ("none", "oracle"))
def test_observation_blocks_match_the_pure_layout(task: str, partner_mode: str) -> None:
    """The GPU block order must concatenate to the exact `logic.obs_fields` layout."""
    spec = task_spec(task)
    blocks = observation_blocks(spec, partner_mode)
    assert set(blocks) == set(AGENT_KEYS)
    dimensions = {
        "action": spec.action_dim,
        "command": spec.action_dim,
        "params": 3,
        "privileged": 1,
        "shared": len(spec.shared_fields),
        "goal": len(spec.goal_fields),
        "time": 1,
    }
    for role in AGENT_KEYS:
        width = sum(dimensions[key] for key, _owner in blocks[role])
        assert width == len(obs_fields(spec, partner_mode))
        # The partner block is only present under `oracle`, and only on the partner's rows.
        owners = [owner for _key, owner in blocks[role]]
        if partner_mode == "none":
            assert all(owner == role for owner in owners[:3])
            assert owners[3:] == [None, None, None]
        else:
            from callosum.experiments.coop_tasks.logic import partner_of

            assert owners[3:] == [None, None, None] + [partner_of(role)] * 4


def test_observation_blocks_reject_an_unknown_partner_mode() -> None:
    with pytest.raises(ValueError, match="partner_mode"):
        observation_blocks(task_spec("cartpole"), "predicted")


@pytest.mark.parametrize("task", _ALL_TASKS)
def test_randomization_packer_matches_the_cpu_sampler(task: str) -> None:
    """Packing must be a pure dtype upload of the CPU sampler's values (no re-sampling)."""
    torch = pytest.importorskip("torch")
    from callosum.experiments.coop_tasks.env_gpu import pack_episode_randomization

    cfg = coop_config(task)
    sampler = sample_toss_randomization if task == "toss" else sample_episode_randomization
    seeds = [3, 4, 5]
    records = [
        sampler(
            cfg.randomization,
            control_dt=cfg.control_dt,
            max_control_steps=cfg.max_control_steps,
            seed=seed,
        )
        for seed in seeds
    ]
    agents, episode = pack_episode_randomization(records, "cpu")
    for role_index, role in enumerate(AGENT_KEYS):
        for name, tensor in agents[role].items():
            values = [getattr(record.agents[role_index], name) for record in records]
            if isinstance(values[0], bool):
                assert tensor.dtype == torch.bool
                assert tensor.tolist() == values
            elif isinstance(values[0], int):
                assert tensor.dtype == torch.int64
                assert tensor.tolist() == values
            else:
                assert tensor.dtype == torch.float32
                assert torch.allclose(tensor, torch.tensor(values, dtype=torch.float32))
    expected_fields = {"initial_value", "damping", "wind_active", "wind_step", "wind_force"}
    if task == "toss":
        expected_fields |= {
            "launch_velocity",
            "launch_position",
            "release_delay_steps",
            "contact_dampratio",
            "contact_friction",
        }
    assert set(episode) == expected_fields
    for name, tensor in episode.items():
        values = np.asarray([getattr(record, name) for record in records])
        if task == "toss" and name in ("release_delay_steps", "wind_step"):
            assert tensor.dtype == torch.int64
        if name == "wind_active":
            assert tensor.dtype == torch.bool
            assert tensor.tolist() == values.tolist()


def test_packer_rejects_an_empty_batch() -> None:
    pytest.importorskip("torch")
    from callosum.experiments.coop_tasks.env_gpu import pack_episode_randomization

    with pytest.raises(ValueError, match="at least one"):
        pack_episode_randomization([], "cpu")


# --- GPU backend (skips without warp/CUDA) ------------------------------------------------


def _gpu_env(task: str, num_envs: int = 2, **kwargs):
    """Build a `CoopBatchGPU` for the test, skipping when the GPU stack is unavailable."""
    pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_warp")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    from callosum.experiments.coop_tasks.env_gpu import CoopBatchGPU

    config = kwargs.pop("config", None)
    return CoopBatchGPU(
        task, num_envs, config=config if config is not None else coop_config(task), **kwargs
    )


def _zero_actions(torch_module, num_envs: int, action_dim: int, device: str = "cuda"):
    """The all-zero normalized action batch of both roles."""
    return {
        role: torch_module.zeros((num_envs, action_dim), dtype=torch_module.float32, device=device)
        for role in AGENT_KEYS
    }


@pytest.mark.parametrize("task", _ALL_TASKS)
def test_gpu_env_shapes_and_one_step(task: str) -> None:
    torch = pytest.importorskip("torch")
    env = _gpu_env(task)
    try:
        assert env.torch_io is True
        assert env.num_envs == 2
        assert env.obs_dim == len(obs_fields(task_spec(task), "none"))
        assert env.action_dim == coop_config(task).action_size
        obs = env.reset(seed=0)
        for role in AGENT_KEYS:
            assert obs[role].shape == (2, env.obs_dim)
            assert obs[role].dtype == torch.float32
            assert torch.isfinite(obs[role]).all()
        result = env.step(_zero_actions(torch, 2, env.action_dim))
        assert result.done.shape == (2,)
        assert result.reward.shape == (2,)
        assert torch.isfinite(result.reward).all()
        for role in AGENT_KEYS:
            assert result.obs[role].shape == (2, env.obs_dim)
            assert result.final_obs[role].shape == (2, env.obs_dim)
        assert len(result.finished_returns) == len(result.finished_lengths)
        assert len(result.finished_successes) == len(result.finished_infos)
        # The CUDA graph is the production path (the eager fallback exists for capture errors).
        assert env.cuda_graph is True
        assert env.cuda_graph_error is None
    finally:
        env.close()


@pytest.mark.parametrize("task", _ALL_TASKS)
def test_gpu_env_is_bit_identical_across_instances(task: str) -> None:
    """Two independently built envs with the same seed/actions stay BIT-identical."""
    torch = pytest.importorskip("torch")
    first = _gpu_env(task, 4, seed=7)
    second = _gpu_env(task, 4, seed=7)
    try:
        obs_a = first.reset(seed=7)
        obs_b = second.reset(seed=7)
        generator = torch.Generator().manual_seed(11)
        for _step in range(6):
            actions = {
                role: (torch.rand((4, first.action_dim), generator=generator) * 2.0 - 1.0).to(
                    "cuda"
                )
                for role in AGENT_KEYS
            }
            step_a = first.step(actions)
            step_b = second.step(actions)
            assert torch.equal(first.state()["qpos"], second.state()["qpos"])
            assert torch.equal(step_a.obs["left"], step_b.obs["left"])
            assert torch.equal(step_a.reward, step_b.reward)
            assert torch.equal(step_a.terminated, step_b.terminated)
        for role in AGENT_KEYS:
            assert torch.equal(obs_a[role], obs_b[role])
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("task", _ALL_TASKS)
def test_gpu_env_auto_reset_contract(task: str) -> None:
    """A tiny episode budget forces truncations; the finished lists match the done worlds."""
    torch = pytest.importorskip("torch")
    # The toss-catch config requires n_passes * 4 <= max_control_steps, so it needs a slightly
    # larger (still tiny) budget than the two contact-free tasks.
    budget = 6 if task != "toss" else 14
    config = dataclasses.replace(coop_config(task), max_control_steps=budget)
    env = _gpu_env(task, 4, config=config)
    try:
        env.reset(seed=0)
        for step in range(budget + 1):
            result = env.step(_zero_actions(torch, 4, env.action_dim))
            done = result.terminated | result.truncated
            done_rows = done.nonzero(as_tuple=False).flatten().cpu().tolist()
            assert len(done_rows) == len(result.finished_returns)
            assert len(done_rows) == len(result.finished_lengths)
            assert len(done_rows) == len(result.finished_successes)
            assert len(done_rows) == len(result.finished_angles)
            assert len(done_rows) == len(result.finished_infos)
            for info in result.finished_infos:
                assert "success" in info
            if step == budget - 1:
                assert len(done_rows) == 4, "every world must truncate after max_control_steps"
                assert all(length == budget for length in result.finished_lengths)
            elif step < budget - 1:
                assert not done_rows, "no episode may end before the budget"
        for role in AGENT_KEYS:
            assert torch.isfinite(result.obs[role]).all()
    finally:
        env.close()


@pytest.mark.parametrize("task", ("cartpole", "pendulum"))
def test_gpu_one_step_parity_with_the_cpu_env(task: str) -> None:
    """The first control step of the two backends matches (float32 vs float64 physics)."""
    pytest.importorskip("mujoco")
    torch = pytest.importorskip("torch")
    from callosum.experiments.coop_tasks import make_coop_env

    env = _gpu_env(task, 2)
    cpu = make_coop_env(task, config=env.config)
    try:
        cpu_obs, _ = cpu.reset(seed=0)
        gpu_obs = env.reset(seed=0)
        for role in AGENT_KEYS:
            np.testing.assert_allclose(
                gpu_obs[role][0].cpu().numpy(), cpu_obs[role], rtol=0.0, atol=1e-5
            )
        actions = {
            role: np.full((2, env.action_dim), 0.25, dtype=np.float32) for role in AGENT_KEYS
        }
        cpu_next, cpu_reward, _cpu_term, _cpu_trunc, _info = cpu.step(
            {role: actions[role][0].astype(np.float64) for role in AGENT_KEYS}
        )
        result = env.step({role: torch.as_tensor(actions[role]) for role in AGENT_KEYS})
        np.testing.assert_allclose(
            env.state()["qpos"][0].cpu().numpy(),
            np.asarray(cpu._data.qpos, dtype=np.float32),
            rtol=0.0,
            atol=1e-5,
        )
        for role in AGENT_KEYS:
            np.testing.assert_allclose(
                result.obs[role][0].cpu().numpy(), cpu_next[role], rtol=0.0, atol=1e-4
            )
        np.testing.assert_allclose(float(result.reward[0].cpu()), cpu_reward, rtol=0.0, atol=1e-4)
    finally:
        env.close()
        cpu.close()


def test_gpu_eager_fallback_matches_the_graph() -> None:
    """`use_cuda_graph=False` (the capture-failure path) reproduces the graph trajectory."""
    torch = pytest.importorskip("torch")
    graph_env = _gpu_env("pendulum", 2, seed=5)
    eager_env = _gpu_env("pendulum", 2, seed=5, use_cuda_graph=False)
    try:
        assert graph_env.cuda_graph is True
        assert eager_env.cuda_graph is False
        graph_env.reset(seed=5)
        eager_env.reset(seed=5)
        generator = torch.Generator().manual_seed(3)
        for _step in range(4):
            actions = {
                role: (torch.rand((2, graph_env.action_dim), generator=generator) * 2.0 - 1.0).to(
                    "cuda"
                )
                for role in AGENT_KEYS
            }
            step_graph = graph_env.step(actions)
            step_eager = eager_env.step(actions)
            assert torch.equal(step_graph.obs["left"], step_eager.obs["left"]), (
                "the graph replay and the eager loop must be bit-identical"
            )
            assert torch.equal(step_graph.reward, step_eager.reward)
    finally:
        graph_env.close()
        eager_env.close()


def test_gpu_env_validates_actions_and_device() -> None:
    pytest.importorskip("torch")
    env = _gpu_env("cartpole", 2)
    try:
        env.reset(seed=0)
        with pytest.raises(ValueError, match="action keys"):
            env.step({"left": np.zeros((2, 1), dtype=np.float32)})
        with pytest.raises(ValueError, match="shape"):
            env.step(
                {
                    "left": np.zeros((3, 1), dtype=np.float32),
                    "right": np.zeros((2, 1), dtype=np.float32),
                }
            )
        with pytest.raises(ValueError, match="CUDA"):
            type(env)("cartpole", 2, device="cpu")
    finally:
        env.close()
