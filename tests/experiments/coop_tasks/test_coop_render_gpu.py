"""Tests of the coop demo renderer's GPU physics backend (`--backend warp`).

Three layers, so the Mac/CI run stays green without the GPU stack:

* **no GPU** (mujoco only, module-level skip): the `--backend`/`--device` CLI resolution, the
  `frame_physics` JSON labels, the pure pendulum amplitude tracker, and the observation-block
  slices the GPU adapter rebuilds `state()` from (checked against `logic.obs_fields`);
* **GPU** (`mujoco` + `mujoco_warp` + CUDA torch; skipped otherwise): the adapter's
  `state()`/`pending_commands()`/`randomization` against the CPU `CoopTaskEnv` stepped with the
  SAME seed and actions (the render parity claim), the render-only MJCF model vs the GPU
  model's `qpos`/`qvel` (the visual model is the GPU MJCF), and the end-to-end
  `run_episode` closed-loop oracle comparison (status/steps/frames/metrics).

The pod runs the actual clip renders; the pytest GPU cases are the parity guard next to them.
"""

import numpy as np
import pytest

try:
    import mujoco  # noqa: F401  (the coop_tasks package imports the simulator at module scope)
except ImportError:  # pragma: no cover - CI has no sandbox extra
    pytest.skip(
        "the coop demo renderer tests need the sandbox extra (mujoco)",
        allow_module_level=True,
    )

from callosum.experiments.coop_tasks import render as coop_render
from callosum.experiments.coop_tasks.config import coop_config
from callosum.experiments.coop_tasks.logic import (
    AGENT_KEYS,
    obs_fields,
    obs_size,
    task_spec,
)

_GPU_TASKS = ("cartpole", "pendulum")
"""The tasks whose GPU render path is finalised (the toss-catch paddle physics is being reworked)."""


# --- CPU-safe cases ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend", "expected_device"),
    (("cpu", None), ("warp", "cuda"), ("warp", "cuda:1")),
)
def test_parse_args_resolves_an_explicit_backend(backend: str, expected_device: str | None) -> None:
    """An explicit backend keeps its device (warp defaults to cuda, cpu has no device)."""
    argv = ["--task", "cartpole", "--out", "/tmp/render", "--backend", backend]
    if expected_device is not None and expected_device != "cuda":
        argv += ["--device", expected_device]
    args = coop_render.parse_args(argv)
    assert args.backend == backend
    assert args.device == expected_device


def test_parse_args_auto_backend_tracks_cuda_visibility(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--backend auto`` is the GPU-first default: warp on a CUDA host, cpu otherwise."""
    argv = ["--task", "pendulum", "--out", "/tmp/render"]
    monkeypatch.setattr("callosum.experiments.cube_wrench.ppo_config.cuda_visible", lambda: True)
    args = coop_render.parse_args(argv)
    assert args.backend == "warp" and args.device == "cuda"
    monkeypatch.setattr("callosum.experiments.cube_wrench.ppo_config.cuda_visible", lambda: False)
    args = coop_render.parse_args(argv)
    assert args.backend == "cpu" and args.device is None


def test_parse_args_rejects_an_unknown_backend() -> None:
    """The CLI only accepts the real physics backends (plus auto)."""
    with pytest.raises(SystemExit):
        coop_render.parse_args(["--task", "cartpole", "--out", "/tmp/render", "--backend", "metal"])


def test_frame_physics_labels() -> None:
    """The JSON records which physics backend produced the frames."""
    assert coop_render.frame_physics_label("warp") == "gpu (mujoco_warp)"
    assert coop_render.frame_physics_label("cpu") == "cpu (mujoco)"


def test_pendulum_tracker_matches_the_cpu_rule() -> None:
    """The renderer's pendulum tracker replays `PendulumEnv._after_step` exactly."""
    tracker = coop_render._PendulumTracker.reset(0.2)
    assert tracker.peak_abs_angle == pytest.approx(0.2)
    assert tracker.last_angle_sign == 1.0 and tracker.completed_cycles == 0
    target = 0.5
    tracker.update(0.4, target)  # same side: peak grows, no cycle
    assert tracker.peak_abs_angle == pytest.approx(0.4)
    assert tracker.completed_cycles == 0
    tracker.update(-0.3, target)  # zero crossing below the target: peak restarts
    assert tracker.completed_cycles == 0
    assert tracker.peak_abs_angle == pytest.approx(0.3)
    tracker.update(-0.6, target)  # same new side: peak grows again
    assert tracker.peak_abs_angle == pytest.approx(0.6)
    tracker.update(0.1, target)  # crossing with the previous peak >= target: +1 cycle
    assert tracker.completed_cycles == 1
    assert tracker.peak_abs_angle == pytest.approx(0.1)


@pytest.mark.parametrize("task", ("cartpole", "pendulum", "toss"))
@pytest.mark.parametrize("partner_mode", ("none", "oracle"))
def test_observation_slices_find_the_shared_blocks(task: str, partner_mode: str) -> None:
    """The adapter reads the shared/goal state from the columns `logic.obs_fields` put there."""
    spec = task_spec(task)
    slices = coop_render._observation_slices(spec, partner_mode)
    fields = obs_fields(spec, partner_mode)
    shared_start = fields.index(spec.shared_fields[0])
    assert slices["shared"] == slice(shared_start, shared_start + len(spec.shared_fields))
    goal_start = fields.index(spec.goal_fields[0])
    assert slices["goal"] == slice(goal_start, goal_start + len(spec.goal_fields))
    assert slices["time"].stop == max(slice_.stop for slice_ in slices.values())
    assert slices["time"].stop <= obs_size(spec, partner_mode)


# --- GPU cases --------------------------------------------------------------------------------

_TOLERANCE = 1e-3
"""Max abs CPU-vs-GPU difference of the demo state fields (float32 Warp vs float64 MuJoCo)."""


def _require_gpu() -> None:
    """Import the GPU stack or skip (CI/Mac); the tests below need warp + a CUDA device."""
    pytest.importorskip("mujoco_warp")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")


def _make_gpu_env(task: str, partner_mode: str = "none"):
    """Build the one-world GPU demo env (skips without the GPU stack)."""
    _require_gpu()
    return coop_render.GpuDemoEnv(
        task,
        coop_config(task),
        partner_mode,
        seed=0,
        device="cuda",
    )


def _make_cpu_env(task: str, partner_mode: str = "none"):
    """Build the CPU env the GPU path is compared against."""
    from callosum.experiments.coop_tasks import make_coop_env

    return make_coop_env(task, config=coop_config(task), partner_mode=partner_mode)


def _random_actions(task: str, steps: int, rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    """A deterministic action sequence in the env's `[-1, 1]` box."""
    action_dim = task_spec(task).action_dim
    return [
        {role: rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float64) for role in AGENT_KEYS}
        for _ in range(steps)
    ]


@pytest.mark.parametrize("task", _GPU_TASKS)
def test_gpu_demo_env_matches_the_cpu_env(task: str) -> None:
    """Same seed + same actions => the render state, queue and randomness match the CPU env."""
    config = coop_config(task)
    gpu_env = _make_gpu_env(task)
    cpu_env = _make_cpu_env(task)
    try:
        gpu_obs, _ = gpu_env.reset(seed=0)
        cpu_obs, _ = cpu_env.reset(seed=0)
        for role in AGENT_KEYS:
            assert gpu_obs[role].shape == (1, obs_size(task_spec(task), "none"))
            np.testing.assert_allclose(
                gpu_obs[role][0].cpu().numpy(),
                np.asarray(cpu_obs[role], dtype=np.float32),
                atol=_TOLERANCE,
            )
        # The sampled private randomness the demo oracle reads is the SAME record the GPU
        # physics uses (the adapter re-samples it from the episode seed for the CPU oracle).
        gpu_record, cpu_record = gpu_env.randomization, cpu_env.randomization
        assert gpu_record is not None and cpu_record is not None
        for gpu_agent, cpu_agent in zip(gpu_record.agents, cpu_record.agents):
            for name in ("latency_s", "gain", "limit_scale", "start_delay_s"):
                assert getattr(gpu_agent, name) == pytest.approx(getattr(cpu_agent, name))
        assert gpu_record.damping == pytest.approx(cpu_record.damping)
        _assert_randomization_matches_the_gpu_env(gpu_env, gpu_record)
        # The render state dict has exactly the CPU env's keys (the HUD reads them by name).
        assert set(gpu_env.state()) == set(cpu_env.state())
        _assert_states_match(gpu_env.state(), cpu_env.state())
        _assert_pending_commands_match(gpu_env, cpu_env)
        actions = _random_actions(task, 12, np.random.default_rng(17))
        for action in actions:
            cpu_step = cpu_env.step(action)
            gpu_step = gpu_env.step(action)
            assert gpu_step[2] == cpu_step[2] and gpu_step[3] == cpu_step[3]
            assert gpu_step[1] == pytest.approx(cpu_step[1], abs=1e-2)
            assert set(gpu_env.state()) == set(cpu_env.state())
            _assert_states_match(gpu_env.state(), cpu_env.state())
            _assert_pending_commands_match(gpu_env, cpu_env)
        # The same numbers drive the HUD: both produce it from the same fields and metrics.
        gpu_metrics = coop_render._task_metrics(task, gpu_env.state(), config)
        cpu_metrics = coop_render._task_metrics(task, cpu_env.state(), config)
        assert set(gpu_metrics) == set(cpu_metrics)
        for key in cpu_metrics:
            assert gpu_metrics[key] == pytest.approx(cpu_metrics[key], abs=1e-2)
        gpu_hud = coop_render._task_hud(
            task, config, gpu_env.state(), 12, config.max_control_steps, actions[-1], 0.0
        )
        cpu_hud = coop_render._task_hud(
            task, config, cpu_env.state(), 12, config.max_control_steps, actions[-1], 0.0
        )
        assert len(gpu_hud.lines) == len(cpu_hud.lines)
        assert [block[0] for block in gpu_hud.agent_blocks] == [
            block[0] for block in cpu_hud.agent_blocks
        ]
    finally:
        gpu_env.close()
        cpu_env.close()


def _assert_states_match(gpu_state: dict[str, float], cpu_state: dict[str, float]) -> None:
    """Every demo/HUD state field of the two envs agrees inside the float32 tolerance."""
    for key, cpu_value in cpu_state.items():
        assert gpu_state[key] == pytest.approx(cpu_value, abs=_TOLERANCE), key


def _assert_randomization_matches_the_gpu_env(gpu_env, record) -> None:
    """The adapter's CPU-side record is the GPU env's own (public) randomisation tensors."""
    agents = gpu_env._gpu.randomization_tensors()
    for role_index, role in enumerate(AGENT_KEYS):
        for name, tensor in agents[role].items():
            expected = getattr(record.agents[role_index], name)
            actual = tensor[0].item()
            if isinstance(expected, bool):
                assert bool(actual) == expected
            else:
                assert float(actual) == pytest.approx(float(expected))
    episode = gpu_env._gpu.episode_randomization_tensors()
    for name, tensor in episode.items():
        expected = getattr(record, name, None)
        if expected is None:  # e.g. the toss-only fields, not part of the cartpole record
            continue
        actual = tensor[0].item()
        if isinstance(expected, bool):
            assert bool(actual) == expected
        else:
            assert float(actual) == pytest.approx(float(expected))


def _assert_pending_commands_match(gpu_env, cpu_env) -> None:
    """The in-flight latency queues (the scripted oracle's Smith predictor input) agree."""
    gpu_pending = gpu_env.pending_commands()
    cpu_pending = cpu_env.pending_commands()
    assert set(gpu_pending) == set(cpu_pending)
    for role in AGENT_KEYS:
        assert len(gpu_pending[role]) == len(cpu_pending[role])
        for gpu_command, cpu_command in zip(gpu_pending[role], cpu_pending[role]):
            np.testing.assert_allclose(gpu_command, cpu_command, atol=_TOLERANCE)


@pytest.mark.parametrize("task", _GPU_TASKS)
def test_render_model_matches_the_gpu_mjcf(task: str) -> None:
    """The render-only model's `qpos`/`qvel` layout is the GPU MJCF's: `state()` maps 1:1."""
    config = coop_config(task)
    gpu_env = _make_gpu_env(task)
    try:
        gpu_env.reset(seed=0)
        actions = _random_actions(task, 5, np.random.default_rng(3))
        for action in actions:
            gpu_env.step(action)
        _model, data, _camera = coop_render._build_render_model(task, config, 64, 48)
        coop_render._write_render_state(task, data, gpu_env.state())
        raw = gpu_env._gpu.state()
        np.testing.assert_allclose(data.qpos, raw["qpos"][0].cpu().numpy(), atol=1e-5)
        np.testing.assert_allclose(data.qvel, raw["qvel"][0].cpu().numpy(), atol=1e-5)
    finally:
        gpu_env.close()


@pytest.mark.parametrize("task", _GPU_TASKS)
def test_closed_loop_oracle_episode_matches_the_cpu_demo(task: str) -> None:
    """The oracle's GPU-physics clip has the same status/steps/frames/metrics as the CPU one."""
    config = coop_config(task)
    gpu_env = _make_gpu_env(task)
    cpu_env = _make_cpu_env(task)
    try:
        kwargs = {
            "task": task,
            "config": config,
            "seed": 0,
            "step_stride": 1,
            "max_steps": int(config.max_control_steps),
            "title": "parity",
            "subtitle": "parity",
            "name": "parity",
        }
        gpu_episode = coop_render.run_episode(
            gpu_env,
            coop_render.OracleDemoController(task, config),
            None,
            **kwargs,
        )
        cpu_episode = coop_render.run_episode(
            cpu_env,
            coop_render.OracleDemoController(task, config),
            None,
            **kwargs,
        )
    finally:
        gpu_env.close()
        cpu_env.close()
    # The scripted oracle is a closed loop, so float32 noise may add/drop a step; the demo's
    # claim is the same outcome, essentially the same length and the same HUD fields.
    assert gpu_episode.status == cpu_episode.status
    assert gpu_episode.success == cpu_episode.success
    assert abs(gpu_episode.steps - cpu_episode.steps) <= 5
    # One frame at step 0 plus one per control step (the terminal frame included).
    assert len(gpu_episode.frames) == gpu_episode.steps + 1
    assert len(cpu_episode.frames) == cpu_episode.steps + 1
    assert set(gpu_episode.metrics) == set(cpu_episode.metrics)
