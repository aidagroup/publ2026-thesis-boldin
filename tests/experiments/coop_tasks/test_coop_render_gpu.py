"""Tests of the coop demo renderer's GPU physics backend (`--backend warp`).

Three layers, so the Mac/CI run stays green without the GPU stack:

* **no GPU** (mujoco only, module-level skip): the `--backend`/`--device` CLI resolution, the
  `frame_physics` JSON labels, the pure pendulum amplitude tracker, the observation-block
  slices the GPU adapter rebuilds `state()` from (checked against `logic.obs_fields`), and the
  new demo-goal layer: the per-task `GOAL:`/`SUCCESS:` text derived from the configs, the top
  goal band on the annotated frame, the intro/outro cards (never black frames) and the
  intro+episode+outro clip timeline;
* **GPU** (`mujoco` + `mujoco_warp` + CUDA torch; skipped otherwise): the adapter's
  `state()`/`pending_commands()`/`randomization` against the CPU `CoopTaskEnv` stepped with the
  SAME seed and actions (the render parity claim), the render-only MJCF model vs the GPU
  model's `qpos`/`qvel` (the visual model is the GPU MJCF), and the end-to-end
  `run_episode` closed-loop oracle comparison (status/steps/frames/metrics). For the
  contact-driven toss-catch the per-step float32 contact solver cannot match CPU MuJoCo, so
  its GPU case pins the ORACLE OUTCOME (same terminal status and episode length) instead.

The pod runs the actual clip renders; the pytest GPU cases are the parity guard next to them.
"""

import math

import numpy as np
import pytest

try:
    import mujoco  # noqa: F401  (the coop_tasks package imports the simulator at module scope)
except ImportError:  # pragma: no cover - CI has no sandbox extra
    pytest.skip(
        "the coop demo renderer tests need the sandbox extra (mujoco)",
        allow_module_level=True,
    )

from callosum.common.render_utils import CardLine, CardText, annotate_frame, card_frame
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


def test_task_goal_text_comes_from_the_configs() -> None:
    """The GOAL/SUCCESS/FAILED text shows the exact numbers of the task config."""
    cartpole = coop_config("cartpole")
    goal = coop_render.task_goal("cartpole", cartpole)
    assert goal.goal.startswith("keep the pole up")
    assert f"{cartpole.target_x:+.2f} m" in goal.goal
    assert f"{cartpole.success_pos_tol * 100.0:.0f} cm" in goal.success
    assert f"{math.degrees(cartpole.success_angle_tol):.1f}°" in goal.success
    assert f"{cartpole.success_lin_speed_tol:.2f} m/s" in goal.success
    assert f"{cartpole.success_ang_speed_tol:.2f} rad/s" in goal.success
    assert f"{math.degrees(cartpole.failure_angle):.1f}°" in goal.failure
    assert f"{cartpole.max_force:.0f} N" in " ".join(goal.controls)
    assert f"{cartpole.max_control_steps} control steps" in goal.budget

    pendulum = coop_config("pendulum")
    goal = coop_render.task_goal("pendulum", pendulum)
    target_deg = math.degrees(pendulum.target_amplitude)
    assert f"{target_deg:.1f}°" in goal.goal and f"{target_deg:.1f}°" in goal.success
    assert str(pendulum.hold_cycles) in goal.goal and str(pendulum.hold_cycles) in goal.success
    assert f"{pendulum.max_torque:.2f} Nm" in " ".join(goal.controls)

    toss = coop_config("toss")
    goal = coop_render.task_goal("toss", toss)
    assert f"{toss.n_passes} alternating passes" in goal.goal
    assert f"{toss.n_passes}/{toss.n_passes} passes" in goal.success
    floor = toss.floor_z + toss.success_floor_margin
    assert f"z > {floor:.2f} m" in goal.success
    assert f"{toss.court_x:.2f} m" in goal.success
    assert f"{toss.floor_z:.2f} m" in goal.failure


def test_task_goal_hud_lines_are_the_goal_band() -> None:
    """`_task_hud` carries the two top-band lines for every task and state step."""
    for task in ("cartpole", "pendulum", "toss"):
        config = coop_config(task)
        goal = coop_render.task_goal(task, config)
        hud = coop_render._task_hud(
            task,
            config,
            _fake_state(task),
            step=3,
            max_steps=int(config.max_control_steps),
            action={role: np.zeros(task_spec(task).action_dim) for role in AGENT_KEYS},
            best_amplitude_deg=0.0,
        )
        assert hud.goal_lines == (f"GOAL: {goal.goal}", f"SUCCESS: {goal.success}")
        assert hud.goal_lines[0].startswith("GOAL: ")
        assert hud.goal_lines[1].startswith("SUCCESS: ")


def _fake_state(task: str) -> dict[str, float]:
    """A plausible `env.state()` dict without stepping any simulator."""
    if task == "cartpole":
        return {
            "step": 3.0,
            "cart_x": 0.1,
            "cart_vx": 0.02,
            "pole_angle": 0.05,
            "pole_angvel": 0.01,
            "target_x": 0.5,
        }
    if task == "pendulum":
        return {
            "step": 3.0,
            "angle": 0.4,
            "angvel": 0.2,
            "target_amplitude": math.radians(65.0),
            "peak_abs_angle": 0.5,
            "completed_cycles": 0.0,
        }
    return {
        "step": 3.0,
        "ball_x": 0.2,
        "ball_z": 0.3,
        "ball_vx": 0.4,
        "ball_vz": 0.5,
        "passes": 1.0,
        "target_passes": 3.0,
        "left_pad_x": -0.5,
        "left_pad_z": 0.3,
        "left_pad_tilt": 0.6,
        "right_pad_x": 0.5,
        "right_pad_z": 0.3,
        "right_pad_tilt": -0.6,
        "left_contact": 0.0,
        "right_contact": 0.0,
        "left_pad_target_x": -0.5,
        "left_pad_target_z": 0.3,
        "left_pad_target_tilt": 0.6,
        "right_pad_target_x": 0.5,
        "right_pad_target_z": 0.3,
        "right_pad_target_tilt": -0.6,
    }


def _fake_frame(width: int, height: int, seed: int = 0) -> np.ndarray:
    """A non-black synthetic scene frame (a gradient plus a bright blob)."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:height, 0:width]
    frame = np.stack(
        [
            25.0 + 60.0 * cols / width,
            30.0 + 55.0 * rows / height,
            45.0 + 90.0 * (cols + rows) / (width + height),
        ],
        axis=-1,
    )
    frame += rng.normal(0.0, 6.0, frame.shape)
    image = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    image[height // 3 : height // 3 + 6, width // 4 : width // 4 + 120] = (240, 235, 220)
    return image


@pytest.mark.parametrize(("width", "height"), ((640, 480), (1280, 480)))
def test_annotate_frame_draws_the_top_goal_band(width: int, height: int) -> None:
    """The goal band is drawn on top of the frame at both demo sizes and is not a black band."""
    frame = _fake_frame(width, height)
    goal_lines = ("GOAL: park the cart at x = +0.50 m", "SUCCESS: |x - 0.50| <= 5 cm")
    without = annotate_frame(frame, title="title", subtitle="subtitle", hud_lines=("line",))
    with_goal = annotate_frame(
        frame,
        title="title",
        subtitle="subtitle",
        hud_lines=("line",),
        goal_lines=goal_lines,
    )
    assert with_goal.shape == frame.shape == (height, width, 3)
    assert float(with_goal.mean()) > 5.0
    # The band occupies the top of the frame and is NOT the plain title-only frame.
    band = with_goal[: height // 7]
    assert not np.array_equal(band, without[: height // 7])
    # The band is a dark panel with bright text on it (a full-width goal row, not a black bar).
    assert float(band.mean()) > 5.0
    assert int(band.max()) >= 180
    assert np.unique(band.reshape(-1, 3), axis=0).shape[0] > 20


@pytest.mark.parametrize(("width", "height"), ((640, 480), (1280, 480)))
def test_intro_and_outro_cards_are_not_black(width: int, height: int) -> None:
    """The intro/outro cards are drawn over a dimmed scene frame (never a black frame)."""
    background = _fake_frame(width, height, seed=3)
    intro = CardText(
        eyebrow="TASK GOAL · TossCatch-v0",
        title="two paddles keep a rally alive",
        lines=(
            CardLine("left agent: paddle x / z / tilt", coop_render.LEFT_COLOR),
            CardLine("GOAL: keep the rally alive — 3 alternating passes"),
            CardLine(
                "SUCCESS: 3/3 passes · ball above floor (z > 0.09 m)", coop_render.SUCCESS_COLOR
            ),
        ),
        footer=(
            "12.8 s sim · 12 s episode + 2.5 s intro + 2.0 s outro @ 15 fps · "
            "physics: gpu (mujoco_warp) · checkpoint update 250"
        ),
    )
    outro = CardText(
        eyebrow="RESULT",
        title="SUCCESS",
        lines=(CardLine("3/3 passes · ball z 0.35 m", coop_render.SUCCESS_COLOR),),
        footer="12.0 s episode + 2.5 s intro + 2.0 s outro",
        title_color=coop_render.SUCCESS_COLOR,
    )
    for card in (intro, outro):
        image = card_frame(background, card)
        assert image.shape == (height, width, 3)
        assert float(image.mean()) > 5.0
        assert not np.array_equal(image, background)
        # The panel covers the centre; the dimmed scene stays visible at the frame border.
        assert (
            image[height // 2, width // 2].tolist() != background[height // 2, width // 2].tolist()
        )


def test_clip_timeline_is_intro_plus_held_episode_plus_outro() -> None:
    """The clip is the intro card + the held episode + the outro card, never a bare episode."""
    episode = _fake_episode(steps=10, frames=5)
    duplicate = 3
    intro_frames = 38  # 2.5 s at 15 fps
    outro_frames = 30  # 2.0 s at 15 fps
    images = list(
        coop_render._carded_frames(
            episode,
            duplicate,
            intro_image=np.full((4, 4, 3), 9, dtype=np.uint8),
            intro_frames=intro_frames,
            outro_image=np.full((4, 4, 3), 7, dtype=np.uint8),
            outro_frames=outro_frames,
        )
    )
    assert len(images) == intro_frames + len(episode.frames) * duplicate + outro_frames
    assert all((image == 9).all() for image in images[:intro_frames])
    assert all((image == 7).all() for image in images[-outro_frames:])
    assert len(images) == 38 + 15 + 30


def test_duration_helpers_and_metadata() -> None:
    """`--target-seconds` holds the episode, cards add on top, and the JSON block is complete."""
    assert coop_render._seconds_to_frames(2.5, 15) == 38
    assert coop_render._seconds_to_frames(2.0, 15) == 30
    assert coop_render._seconds_to_frames(0.0, 15) == 0
    duplicate = coop_render._duplicate_for_target(34, 15, 12.0)
    assert duplicate == math.ceil(12.0 * 15 / 34)  # the hold factor rounds UP to reach the target
    assert coop_render._duplicate_for_target(122, 15, 12.0) == 2
    assert coop_render._duplicate_for_target(400, 15, 12.0) == 1
    episode_frames = 34 * duplicate
    metadata = coop_render._timeline_metadata(
        total_frames=38 + episode_frames + 30,
        fps=15,
        duplicate=duplicate,
        episode_sim_seconds=1.32,
        intro_seconds=2.5,
        intro_frames=38,
        outro_seconds=2.0,
        outro_frames=30,
        episode_video_frames=episode_frames,
    )
    assert metadata["intro_seconds"] == 2.5 and metadata["outro_seconds"] == 2.0
    assert metadata["intro_frames"] == 38 and metadata["outro_frames"] == 30
    assert metadata["episode_video_frames"] == episode_frames
    assert metadata["episode_sim_seconds"] == 1.32
    assert metadata["video_frames"] == 38 + episode_frames + 30
    # The intro and outro are on top of the target: the clip is longer than --target-seconds.
    assert metadata["video_duration_s"] > 12.0 + 2.5 + 2.0 - 1.0


def test_gif_duration_is_the_sum_of_the_frame_delays(tmp_path) -> None:
    """A slow-motion GIF's duration is the sum of its delays (identical frames are merged)."""
    red = np.full((24, 32, 3), (200, 30, 30), dtype=np.uint8)
    green = np.full((24, 32, 3), (30, 200, 30), dtype=np.uint8)
    images = [red, red, red, green, green, green]
    info = coop_render._write_gif_from_images(
        tmp_path / "preview.gif",
        iter(images),
        fps=15,
        gif_width=64,
        gif_stride=2,
    )
    # 6 input frames at 2 GIF frames per step merge into 2 GIF frames of 3*133 ms each.
    assert info["frames"] < len(images)
    assert info["written_frames"] == len(images)
    assert info["duration_s"] == pytest.approx(0.78, abs=0.03)


def _fake_episode(steps: int, frames: int) -> "coop_render.DemoEpisode":
    """A synthetic `DemoEpisode` (no simulator) with `frames` rendered frames."""
    demo_frames = [
        coop_render.DemoFrame(image=_fake_frame(8, 8, seed=index), step=index, time_s=index * 0.04)
        for index in range(frames)
    ]
    return coop_render.DemoEpisode(
        task="toss",
        name="fake",
        seed=0,
        title="t",
        subtitle="s",
        frames=demo_frames,
        steps=steps,
        success=True,
        terminated=True,
        truncated=False,
        status="SUCCESS",
        status_detail="3/3 passes",
        status_color=coop_render.SUCCESS_COLOR,
        metrics={
            "passes": 3.0,
            "target_passes": 3.0,
            "ball_x": 0.06,
            "ball_z": 0.35,
            "ball_vz": 0.88,
            "progress": 1.0,
        },
        control_dt=0.04,
    )


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


def test_toss_closed_loop_oracle_episode_status_matches_the_cpu_demo() -> None:
    """The contact-driven toss-catch: the GPU clip's ORACLE outcome matches the CPU demo.

    The per-step float32 contact solver diverges from CPU MuJoCo after the first ball/paddle
    impact (hence toss is not in `_GPU_TASKS`), but the demo's claim is the OUTCOME: the same
    terminal status and essentially the same episode length, which this case pins down.
    """
    config = coop_config("toss")
    gpu_env = _make_gpu_env("toss")
    cpu_env = _make_cpu_env("toss")
    try:
        kwargs = {
            "task": "toss",
            "config": config,
            "seed": 0,
            "step_stride": 1,
            "max_steps": int(config.max_control_steps),
            "title": "parity",
            "subtitle": "parity",
            "name": "parity",
        }
        gpu_episode = coop_render.run_episode(
            gpu_env, coop_render.OracleDemoController("toss", config), None, **kwargs
        )
        cpu_episode = coop_render.run_episode(
            cpu_env, coop_render.OracleDemoController("toss", config), None, **kwargs
        )
    finally:
        gpu_env.close()
        cpu_env.close()
    assert gpu_episode.status == cpu_episode.status
    assert gpu_episode.success == cpu_episode.success
    assert abs(gpu_episode.steps - cpu_episode.steps) <= 2
    assert len(gpu_episode.frames) == gpu_episode.steps + 1
    assert set(gpu_episode.metrics) == set(cpu_episode.metrics)


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
