"""Render the two-agent coop-suite demos: scripted oracle, learned policy vs oracle, attempts.

The reduced coop suite (`CartPoleCoop-v0`, `PendulumPump-v0`, `TossCatch-v0`) renders
nothing on its own and its MJCF assets have no cameras, so this module builds a
**render-only copy** of the task's MJCF through `mujoco.MjSpec`: the copy gets colours, a
ground plane, goal markers, a skybox, lights and a free camera, while the env's physics model
is never touched. The env's `state()` is written into the copy's `qpos`/`qvel` before every
frame, so the visuals are exactly the env's physics (the same pattern as
`callosum/experiments/cube_wrench/render.py`; the shared MP4/GIF/font/HUD helpers come from
`callosum/common/render_utils.py`).

Scenes (``--scene``):

1. ``oracle`` -- the task's scripted centralised oracle
   (`callosum.experiments.coop_tasks.oracle`): a success clip on the requested seed.
2. ``policy-vs-oracle`` -- the trained `wrench_ppo` policies (greedy, left) next to the
   scripted oracle (right, the privileged upper bound), both from the same episode seed. Each
   half is held in slow motion so both reach their end state at the same video time (the
   learned policy needs more control steps than the script). Requires ``--checkpoint``.
3. ``attempts`` -- N episodes on consecutive seeds (one MP4 per attempt plus a summary JSON
   with success/steps), so a demo can show successes and failures side by side. Uses the
   scripted oracle, or the checkpoint policies when ``--checkpoint`` is given.

Every frame carries a Pillow HUD: the task name, step/time, the task's score/progress (the
cart-pole's |pole angle| and cart->target distance, the pendulum's peak amplitude->target and
hold cycles, the toss-catch's passes->target with the ball height/velocity), the agents'
``left``/``right`` commanded quantities (force/torque magnitudes for the cart-pole and
pendulum, paddle pose plus commanded velocity for the toss-catch), a bottom progress bar and
a SUCCESS / FAILED / TIMEOUT status on the last frame. The task-specific metrics, goal
markers, state copy and camera poses live in one dispatch point per concern
(``_decorate_scene``, ``_write_render_state``, ``_task_hud``, ``_task_metrics`` and
``TASK_VIEWS``), which is how the three tasks share one renderer.

The learned-policy path reuses `callosum/experiments/coop_tasks/eval.py` (checkpoint loading,
input-mode resolution, greedy inference and the Bi-JEPA `predicted` input builder) instead of
duplicating it, so a clip shows exactly what the ablation evaluates.

Physics backend (``--backend``): the rollout physics runs either in the CPU MuJoCo env
(``cpu``) or on the GPU through the batched `CoopBatchGPU`
(`callosum/experiments/coop_tasks/env_gpu.py`, ``warp``). ``auto`` (the default) picks
``warp`` when a CUDA device is visible and ``cpu`` otherwise. The GPU path is the GPU-first
default on the pod: `GpuDemoEnv` drives ONE GPU world per demo episode, the scripted oracle
stays the same numpy controller (it reads the GPU world's state through the adapter) and the
learned policies run on the CUDA device with `torch_io` observation tensors -- only the
per-frame render state (~30 floats) is copied back to the host. Frame capture, the HUD, the
render-only MuJoCo visual copy (built from the same MJCF asset, CPU-side) and the MP4/GIF/JSON
writers are shared by both backends, so the only intended difference is float32 Warp physics
against float64 MuJoCo; ``frame_physics`` in the JSON records which one produced the clip.

Nothing is installed: MP4 is written by OpenCV when available and by the system ``ffmpeg``
otherwise, GIF by Pillow (`callosum.common.render_utils`). Rendering is offscreen: the script
selects ``MUJOCO_GL=egl`` on Linux and ``cgl`` on macOS unless ``MUJOCO_GL`` is already set.

Run on the pod (from the repo root; ``/root/.callosum/venv`` has mujoco + EGL + ffmpeg; the
GPU physics path needs ``uv pip install --python /root/.callosum/venv/bin/python
mujoco-warp==4.0.1``):

    MUJOCO_GL=egl /root/.callosum/venv/bin/python -m callosum.experiments.coop_tasks.render \\
        --task cartpole --scene oracle --seed 0 --attempts 3 --backend warp \\
        --width 640 --height 480 --fps 15 --out /workspace/callosum-runs/render_gpu_physics

    MUJOCO_GL=egl /root/.callosum/venv/bin/python -m callosum.experiments.coop_tasks.render \\
        --task pendulum --scene policy-vs-oracle --backend warp \\
        --checkpoint /root/callosum/runs/coop_pendulum_none_s0/latest.pt \\
        --input-mode none --out /workspace/callosum-runs/render_gpu_physics

    MUJOCO_GL=egl /root/.callosum/venv/bin/python -m callosum.experiments.coop_tasks.render \\
        --task cartpole --scene attempts --attempts 5 --seed 0 --backend warp \\
        --checkpoint .../latest.pt --input-mode none --out /workspace/callosum-runs/render_gpu_physics

Metric-only checks without OpenGL (the CPU path is also fine on the macOS dev laptop once the
``sandbox`` extra is installed; the ``warp`` path needs the GPU stack):

    python callosum/experiments/coop_tasks/render.py --task cartpole --out /tmp/render --dry-run
    python callosum/experiments/coop_tasks/render.py --task pendulum --scene attempts \\
        --attempts 3 --out /tmp/render --dry-run
    python callosum/experiments/coop_tasks/render.py --task cartpole --scene oracle \\
        --backend warp --out /tmp/render --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from callosum.common.render_utils import (
    FAILURE_COLOR,
    MUTED_COLOR,
    SUCCESS_COLOR,
    annotate_frame,
    hstack_side_by_side,
    mean_luma,
    mujoco_module,
    probe_video,
    save_png_stills,
    write_gif,
    write_mp4,
)

if TYPE_CHECKING:
    import torch

    from callosum.experiments.coop_tasks.base import CoopTaskEnv
    from callosum.experiments.coop_tasks.config import (
        CartPoleConfig,
        PendulumConfig,
        TossConfig,
    )

    TaskConfig = CartPoleConfig | PendulumConfig | TossConfig
    DemoObs = dict[str, np.ndarray] | dict[str, "torch.Tensor"]
    """One environment observation: numpy on the CPU backend, CUDA tensors on the GPU one."""

RENDER_TASKS = ("cartpole", "pendulum", "toss")
"""The coop tasks this renderer supports (the full two-robot suite)."""

PHYSICS_BACKENDS = ("cpu", "warp")
"""Physics backends of the rollout: the CPU MuJoCo env or the batched mujoco_warp GPU env."""

GPU_FRAME_PHYSICS = "gpu (mujoco_warp)"
"""The JSON `frame_physics` value of a clip whose rollout physics ran on the GPU."""

CPU_FRAME_PHYSICS = "cpu (mujoco)"
"""The JSON `frame_physics` value of a clip whose rollout physics ran in the CPU env."""

AGENT_KEYS = ("left", "right")
"""The suite's two agent keys (mirrors `coop_tasks.logic.AGENT_KEYS`).

The tuple is repeated here so this module stays importable (and ``--help`` works) without the
``sandbox`` extra, exactly like `callosum/experiments/cube_wrench/render.py` keeps its env
import out of the module scope.
"""

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
"""Directory of the task MJCF assets (the render-only copy is built from the same files)."""

LEFT_COLOR = (96, 214, 138, 255)
"""Green: the left agent, which only commands positive physical quantities (+x / +torque)."""
RIGHT_COLOR = (242, 140, 40, 255)
"""Orange: the right agent, which only commands negative physical quantities (-x / -torque)."""

CART_RGBA = (0.28, 0.44, 0.62, 1.0)
POLE_RGBA = (0.88, 0.86, 0.80, 1.0)
POLE_TIP_RGBA = (0.96, 0.55, 0.16, 1.0)
BOB_RGBA = (0.24, 0.46, 0.72, 1.0)
BOB_MARKER_RGBA = (0.96, 0.42, 0.12, 1.0)
TARGET_RGBA = (0.22, 0.85, 0.42, 0.90)
TARGET_FADE_RGBA = (0.22, 0.85, 0.42, 0.35)
RIG_RGBA = (0.42, 0.47, 0.57, 1.0)
PIVOT_RGBA = (0.52, 0.57, 0.66, 1.0)
RAIL_RGBA = (0.60, 0.66, 0.76, 1.0)
BALL_RGBA = (0.99, 0.80, 0.18, 1.0)
BALL_HALO_RGBA = (1.0, 0.85, 0.30, 0.22)
PAD_LEFT_RGBA = (0.30, 0.74, 0.47, 1.0)
PAD_RIGHT_RGBA = (0.93, 0.55, 0.17, 1.0)
PAD_FACE_RGBA = (0.93, 0.95, 1.0, 1.0)
COURT_LINE_RGBA = (0.48, 0.56, 0.68, 0.90)
COURT_CENTRE_RGBA = (0.55, 0.64, 0.78, 0.75)
GROUND_RGBA = (0.24, 0.27, 0.33, 1.0)
SKY_RGB1 = (0.030, 0.045, 0.080)
SKY_RGB2 = (0.150, 0.200, 0.330)

PNG_STILLS = 3
"""Leading frames saved as ``{stem}_frame{i}.png`` for every scene clip."""
GIF_STRIDE = 2
"""Every Nth (slow-motion-expanded) frame goes into the GIF preview, to keep it small."""


# --- per-task render data ------------------------------------------------------------------


@dataclass(frozen=True)
class CameraPose:
    """The free camera of a task's three-quarter external view."""

    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float


@dataclass(frozen=True)
class TaskView:
    """Everything task-specific that is not a per-step HUD number.

    The render-only scene, the camera and the agent vocabulary differ per task; the HUD
    numbers (`_task_metrics`) and the goal markers (`_decorate_scene`) are the only other
    task-specific pieces.
    """

    task: str
    env_title: str
    """Human title used in the frame captions (`CartPoleCoop-v0` / `PendulumPump-v0`)."""
    asset_name: str
    """MJCF file under `callosum/experiments/coop_tasks/assets/`."""
    camera: CameraPose
    ground_z: float
    """World z of the render-only ground plane (below the task's moving parts)."""
    ground_size: float
    """Half-extent of the render-only ground plane (m)."""
    left_title: str
    right_title: str
    symbol: str
    """Physical-quantity symbol in the agent blocks (`F` or the torque sign `τ`)."""
    unit: str
    """Physical unit of the commanded quantity (`N` or `Nm`)."""
    value_decimals: int
    """Decimals of the commanded magnitude in the HUD blocks (1 for N, 3 for Nm)."""


TASK_VIEWS: dict[str, TaskView] = {
    "cartpole": TaskView(
        task="cartpole",
        env_title="CartPoleCoop-v0",
        asset_name="cartpole.xml",
        camera=CameraPose(lookat=(-0.55, 0.0, 1.70), distance=5.4, azimuth=135.0, elevation=-10.0),
        ground_z=-0.052,
        ground_size=1.8,
        left_title="left: push +x",
        right_title="right: push -x",
        symbol="F",
        unit="N",
        value_decimals=1,
    ),
    "pendulum": TaskView(
        task="pendulum",
        env_title="PendulumPump-v0",
        asset_name="pendulum.xml",
        camera=CameraPose(lookat=(-0.15, 0.0, -0.13), distance=1.83, azimuth=130.0, elevation=-8.0),
        ground_z=-0.62,
        ground_size=1.1,
        left_title="left: +torque",
        right_title="right: -torque",
        symbol="τ",
        unit="Nm",
        value_decimals=3,
    ),
    "toss": TaskView(
        task="toss",
        env_title="TossCatch-v0",
        asset_name="toss_catch.xml",
        camera=CameraPose(lookat=(0.0, 0.0, 0.48), distance=3.2, azimuth=100.0, elevation=-20.0),
        ground_z=0.0,
        ground_size=1.9,
        left_title="left: paddle",
        right_title="right: paddle",
        symbol="v",
        unit="m/s",
        value_decimals=2,
    ),
}

ORACLE_SUBTITLES: dict[str, str] = {
    "cartpole": "LQR + Smith predictor: balance the pole and drive the cart to the target",
    "pendulum": "phase-based pumping: reach the target amplitude and hold it",
    "toss": "ballistic interception: catch the ball and return it for alternating passes",
}
"""One-line description of each scripted oracle (shown under the caption)."""


def _task_view(task: str) -> TaskView:
    """The render data of `task`; raises for an unsupported task."""
    if task not in TASK_VIEWS:
        raise ValueError(f"task must be one of {RENDER_TASKS}, got {task!r}")
    return TASK_VIEWS[task]


def _configure_gl() -> None:
    """Select an offscreen OpenGL backend before `mujoco` is imported."""
    if os.environ.get("MUJOCO_GL"):
        return
    os.environ["MUJOCO_GL"] = "cgl" if sys.platform == "darwin" else "egl"


# --- per-step task metrics and HUD ----------------------------------------------------------


@dataclass(frozen=True)
class TaskHud:
    """The HUD of one frame: bottom-left numbers, agent blocks and the progress bar."""

    lines: tuple[str, ...]
    agent_blocks: tuple[tuple[str, tuple[str, ...], tuple[int, int, int, int]], ...]
    progress: float | None


def _task_metrics(task: str, state: dict[str, float], config: TaskConfig) -> dict[str, float]:
    """The task's live score/progress numbers of one `env.state()`.

    The same numbers drive the HUD and the JSON summary, so the report and the video cannot
    disagree. `toss` adds its passes here (and in `_task_hud`).
    """
    if task == "cartpole":
        angle_deg = math.degrees(abs(float(state["pole_angle"])))
        distance = abs(float(state["cart_x"]) - float(config.target_x))
        span = max(abs(float(config.target_x)), 1e-6)
        return {
            "angle_deg": angle_deg,
            "cart_x": float(state["cart_x"]),
            "distance_to_goal": distance,
            "progress": float(np.clip(1.0 - distance / span, 0.0, 1.0)),
        }
    if task == "pendulum":
        amplitude_deg = math.degrees(float(state["peak_abs_angle"]))
        target_deg = math.degrees(float(config.target_amplitude))
        return {
            "angle_deg": math.degrees(abs(float(state["angle"]))),
            "amplitude_deg": amplitude_deg,
            "completed_cycles": float(state["completed_cycles"]),
            "hold_cycles": float(config.hold_cycles),
            "progress": float(np.clip(amplitude_deg / max(target_deg, 1e-6), 0.0, 1.0)),
        }
    if task == "toss":
        passes = float(state["passes"])
        target = float(state["target_passes"])
        return {
            "passes": passes,
            "target_passes": target,
            "ball_x": float(state["ball_x"]),
            "ball_z": float(state["ball_z"]),
            "ball_vz": float(state["ball_vz"]),
            "progress": float(np.clip(passes / max(target, 1e-6), 0.0, 1.0)),
        }
    raise ValueError(f"task must be one of {RENDER_TASKS}, got {task!r}")


def _peak_amplitude_deg(task: str, state: dict[str, float]) -> float:
    """The pendulum's peak |angle| of the current half-cycle (0 for the other tasks)."""
    if task != "pendulum":
        return 0.0
    return math.degrees(float(state["peak_abs_angle"]))


def _agent_blocks(
    task: str, config: TaskConfig, state: dict[str, float], action: dict[str, np.ndarray]
) -> tuple[tuple[str, tuple[str, ...], tuple[int, int, int, int]], ...]:
    """The two top-right action blocks: what each agent is commanding.

    For the one-sided cart-pole/pendulum the shown magnitude is the *commanded* physical
    quantity (`max(action, 0) * authority`, with the left/right sign of the one-sided
    actuation), like the cube demo; the env applies its private gain/limit scaling on top.
    For the toss-catch the agents command a paddle pose increment, so the block shows the
    paddle's pose and its commanded velocity/tilt instead.
    """
    if task == "toss":
        return _toss_agent_blocks(config, state, action)
    view = _task_view(task)
    authorities = [float(config.agent_authority(index)) for index in range(len(AGENT_KEYS))]
    blocks: list[tuple[str, tuple[str, ...], tuple[int, int, int, int]]] = []
    for index, key in enumerate(AGENT_KEYS):
        raw = max(float(np.asarray(action[key]).reshape(-1)[0]), 0.0)
        # The left agent commands +x/+torque, the right agent -x/-torque (AGENT_SIGNS).
        signed = (1.0 if index == 0 else -1.0) * raw * authorities[index]
        blocks.append(
            (
                view.left_title if index == 0 else view.right_title,
                (
                    f"{view.symbol} = {signed:+.{view.value_decimals}f} {view.unit}",
                    f"command {raw:.2f} / 1.00",
                ),
                LEFT_COLOR if index == 0 else RIGHT_COLOR,
            )
        )
    return tuple(blocks)


def _toss_agent_blocks(
    config: TaskConfig, state: dict[str, float], action: dict[str, np.ndarray]
) -> tuple[tuple[str, tuple[str, ...], tuple[int, int, int, int]], ...]:
    """The toss-catch action blocks: each paddle's pose and its commanded motion.

    The command is a normalized `(dx, dz, dtheta)` pose increment, so the shown
    velocity/tilt rate is `command * the paddle's maximum speed / tilt rate` (the physical
    limit of the servo); the position and tilt are the paddle's current pose.
    """
    view = _task_view("toss")
    pad_speed = float(config.pad_speed)
    blocks: list[tuple[str, tuple[str, ...], tuple[int, int, int, int]]] = []
    for index, side in enumerate(AGENT_KEYS):
        command = np.asarray(action[side], dtype=np.float64).reshape(-1)
        blocks.append(
            (
                view.left_title if index == 0 else view.right_title,
                (
                    (f"pad x {state[f'{side}_pad_x']:+.2f} · z {state[f'{side}_pad_z']:+.2f} m"),
                    (
                        f"tilt {math.degrees(float(state[f'{side}_pad_tilt'])):+.0f}° · "
                        f"cmd vx {command[0] * pad_speed:+.1f} · "
                        f"vz {command[1] * pad_speed:+.1f} m/s"
                    ),
                ),
                LEFT_COLOR if index == 0 else RIGHT_COLOR,
            )
        )
    return tuple(blocks)


def _task_hud(
    task: str,
    config: TaskConfig,
    state: dict[str, float],
    step: int,
    max_steps: int,
    action: dict[str, np.ndarray],
    best_amplitude_deg: float,
) -> TaskHud:
    """Build the HUD of one frame from the env state, the episode best and the action.

    `best_amplitude_deg` is the pendulum's running peak amplitude: the env's own tracker
    restarts every half-cycle, but the demo wants the achieved amplitude (the task's score)
    to stay visible and monotonic on the progress bar.
    """
    view = _task_view(task)
    metrics = _task_metrics(task, state, config)
    elapsed = step * config.control_dt
    header = f"{view.env_title} · step {step}/{max_steps} · t = {elapsed:.2f} s"
    if task == "cartpole":
        lines = (
            header,
            (
                f"cart x {metrics['cart_x']:+.2f} → goal {config.target_x:+.2f} m · "
                f"dist {metrics['distance_to_goal'] * 100.0:+.1f} cm"
            ),
            (
                f"|pole angle| {metrics['angle_deg']:.1f}° "
                f"(fail {math.degrees(config.failure_angle):.1f}°)"
            ),
        )
        progress = metrics["progress"]
    elif task == "pendulum":
        target_deg = math.degrees(config.target_amplitude)
        # One line for the score: the shorter bottom-left panel leaves the bob visible.
        lines = (
            header,
            (
                f"peak {best_amplitude_deg:.1f}° / {target_deg:.1f}° · "
                f"|angle| {metrics['angle_deg']:.1f}° · "
                f"hold {metrics['completed_cycles']:.0f}/{metrics['hold_cycles']:.0f}"
            ),
        )
        progress = float(np.clip(best_amplitude_deg / max(target_deg, 1e-6), 0.0, 1.0))
    elif task == "toss":
        lines = (
            header,
            (
                f"passes {metrics['passes']:.0f}/{metrics['target_passes']:.0f} · "
                f"ball z {metrics['ball_z']:.2f} m · vz {metrics['ball_vz']:+.2f} m/s"
            ),
            f"ball x {metrics['ball_x']:+.2f} m · court ±{config.court_x:.1f} m",
        )
        progress = metrics["progress"]
    else:  # pragma: no cover - guarded by _task_view / the CLI choices
        raise ValueError(f"task must be one of {RENDER_TASKS}, got {task!r}")
    return TaskHud(
        lines=lines,
        agent_blocks=_agent_blocks(task, config, state, action),
        progress=progress,
    )


def _episode_status(
    task: str,
    config: TaskConfig,
    steps: int,
    terminated: bool,
    truncated: bool,
    success: bool,
    metrics: dict[str, float],
) -> tuple[str, str, tuple[int, int, int, int]]:
    """The final-frame status word, its detail line and colour."""
    if terminated and success:
        return "SUCCESS", _success_detail(task, metrics), SUCCESS_COLOR
    if terminated:
        return "FAILED", _failure_detail(task, config, metrics), FAILURE_COLOR
    if truncated:
        return "TIMEOUT", _timeout_detail(task, steps, metrics), MUTED_COLOR
    raise RuntimeError("an episode must end terminated or truncated")  # pragma: no cover


def _success_detail(task: str, metrics: dict[str, float]) -> str:
    """The SUCCESS detail line of a task (its achieved score)."""
    if task == "cartpole":
        return f"|pole angle| {metrics['angle_deg']:.1f}° · cart {metrics['cart_x']:+.2f} m at goal"
    if task == "toss":
        return (
            f"{metrics['passes']:.0f}/{metrics['target_passes']:.0f} passes · "
            f"ball z {metrics['ball_z']:.2f} m"
        )
    return (
        f"peak {metrics['peak_amplitude_deg']:.1f}° · "
        f"{metrics['completed_cycles']:.0f}/{metrics['hold_cycles']:.0f} holds"
    )


def _failure_detail(task: str, config: TaskConfig, metrics: dict[str, float]) -> str:
    """The FAILED detail line of a task (why it crossed the failure bound)."""
    if task == "cartpole":
        return f"pole fell: {metrics['angle_deg']:.1f}° > {math.degrees(config.failure_angle):.1f}°"
    if task == "toss":
        if metrics["ball_z"] < float(config.floor_z):
            return f"ball dropped: z {metrics['ball_z']:.2f} < {float(config.floor_z):.2f} m"
        return (
            f"ball left court: x {metrics['ball_x']:+.2f} m (limit ±{float(config.court_x):.2f} m)"
        )
    return f"swung past: {metrics['angle_deg']:.1f}° > {math.degrees(config.failure_angle):.1f}°"


def _timeout_detail(task: str, steps: int, metrics: dict[str, float]) -> str:
    """The TIMEOUT detail line of a task (how far it got in the step budget)."""
    if task == "cartpole":
        return (
            f"{steps} steps · cart {metrics['cart_x']:+.2f} m · "
            f"|pole angle| {metrics['angle_deg']:.1f}°"
        )
    if task == "toss":
        return (
            f"{steps} steps · {metrics['passes']:.0f}/{metrics['target_passes']:.0f} passes · "
            f"ball z {metrics['ball_z']:.2f} m"
        )
    return f"{steps} steps · peak {metrics['peak_amplitude_deg']:.1f}°"


# --- render-only scene copy ------------------------------------------------------------------


def _add_demo_environment(spec: Any, mujoco: Any, view: TaskView) -> None:
    """Add the shared demo look to the render-only spec: ground, skybox and lights."""
    world = spec.worldbody
    # A dark plane with MuJoCo's built-in grid (the third size entry is the grid spacing in m).
    world.add_geom(
        name="coop_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[view.ground_size, view.ground_size, view.ground_size / 12.0],
        pos=[0.0, 0.0, view.ground_z],
        rgba=list(GROUND_RGBA),
        contype=0,
        conaffinity=0,
    )
    spec.add_texture(
        name="coop_sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=list(SKY_RGB1),
        rgb2=list(SKY_RGB2),
        width=256,
        height=512,
    )
    world.add_light(
        name="key",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0.6, -0.5, 1.0],
        dir=[-0.45, 0.35, -0.82],
        diffuse=[0.95, 0.93, 0.90],
        specular=[0.45, 0.45, 0.45],
        castshadow=True,
    )
    world.add_light(
        name="fill",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[-0.6, 0.6, 0.4],
        dir=[0.60, -0.50, -0.40],
        diffuse=[0.34, 0.42, 0.58],
        castshadow=False,
    )
    world.add_light(
        name="rim",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0.0, 0.8, 0.2],
        dir=[0.0, -1.0, -0.25],
        diffuse=[0.30, 0.26, 0.22],
        castshadow=False,
    )
    world.add_light(
        name="ambient",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0.0, 0.0, 2.0],
        dir=[0.0, 0.0, -1.0],
        diffuse=[0.18, 0.18, 0.18],
        castshadow=False,
    )


def _decorate_cartpole(spec: Any, mujoco: Any, config: TaskConfig) -> None:
    """Colour the cart-pole and add the goal rail markers of the render-only copy."""
    spec.geom("cart_geom").rgba = list(CART_RGBA)
    spec.geom("pole_geom").rgba = list(POLE_RGBA)
    spec.body("pole").add_geom(
        name="pole_tip",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.06, 0.0, 0.0],
        pos=[0.0, 0.0, 3.0],
        rgba=list(POLE_TIP_RGBA),
        contype=0,
        conaffinity=0,
    )
    world = spec.worldbody
    target_x = float(config.target_x)
    world.add_geom(
        name="target_column",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.008, 0.008, 0.25],
        pos=[target_x, 0.0, 0.20],
        rgba=list(TARGET_FADE_RGBA),
        contype=0,
        conaffinity=0,
    )
    world.add_geom(
        name="target_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.04, 0.11, 0.002],
        pos=[target_x, 0.0, -0.046],
        rgba=list(TARGET_RGBA),
        contype=0,
        conaffinity=0,
    )
    for side, y in (("left", 0.065), ("right", -0.065)):
        world.add_geom(
            name=f"rail_{side}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[1.45, 0.007, 0.007],
            pos=[0.0, y, -0.043],
            rgba=list(RAIL_RGBA),
            contype=0,
            conaffinity=0,
        )


def _decorate_pendulum(spec: Any, mujoco: Any, config: TaskConfig) -> None:
    """Colour the bob and add the mount plus the target-amplitude spokes of the copy."""
    spec.geom("bob_geom").rgba = list(BOB_RGBA)
    spec.body("bob").add_geom(
        name="bob_marker",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.02, 0.0, 0.0],
        pos=[0.06, 0.0, -0.5],
        rgba=list(BOB_MARKER_RGBA),
        contype=0,
        conaffinity=0,
    )
    world = spec.worldbody
    world.add_geom(
        name="mount_rod",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.012, 0.012, 0.16],
        pos=[0.0, 0.0, 0.16],
        rgba=list(RIG_RGBA),
        contype=0,
        conaffinity=0,
    )
    world.add_geom(
        name="mount_plate",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.10, 0.10, 0.012],
        pos=[0.0, 0.0, 0.33],
        rgba=list(RIG_RGBA),
        contype=0,
        conaffinity=0,
    )
    world.add_geom(
        name="pivot_hub",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.035, 0.0, 0.0],
        pos=[0.0, 0.0, 0.0],
        rgba=list(PIVOT_RGBA),
        contype=0,
        conaffinity=0,
    )
    target = float(config.target_amplitude)
    for sign in (1.0, -1.0):
        angle = sign * target
        direction = np.array([math.sin(angle), 0.0, -math.cos(angle)])
        side = "left" if sign > 0 else "right"
        world.add_geom(
            name=f"target_spoke_{side}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.004, 0.004, 0.5],
            pos=list(0.5 * direction),
            quat=[math.cos(angle / 2.0), 0.0, -math.sin(angle / 2.0), 0.0],
            rgba=list(TARGET_FADE_RGBA),
            contype=0,
            conaffinity=0,
        )
        world.add_geom(
            name=f"target_tip_{side}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.03, 0.0, 0.0],
            pos=list(0.5 * direction),
            rgba=list(TARGET_RGBA),
            contype=0,
            conaffinity=0,
        )


def _decorate_toss(spec: Any, mujoco: Any, config: TaskConfig) -> None:
    """Colour the ball/paddles and add the court markings of the render-only copy."""
    spec.geom("ball_geom").rgba = list(BALL_RGBA)
    spec.geom("pad_left_geom").rgba = list(PAD_LEFT_RGBA)
    spec.geom("pad_right_geom").rgba = list(PAD_RIGHT_RGBA)
    # A faint halo makes the 6 cm ball readable at the demo's camera distance (render only).
    spec.body("ball").add_geom(
        name="ball_halo",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.085, 0.0, 0.0],
        pos=[0.0, 0.0, 0.0],
        rgba=list(BALL_HALO_RGBA),
        contype=0,
        conaffinity=0,
    )
    for side in AGENT_KEYS:
        # A painted patch on both faces of each paddle: the plate's tilt about y (and which
        # side faces the partner) is then readable from either side.
        for face, z in (("front", 0.033), ("back", -0.033)):
            spec.body(f"pad_{side}").add_geom(
                name=f"pad_{side}_face_{face}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.08, 0.08, 0.003],
                pos=[0.0, 0.0, z],
                rgba=list(PAD_FACE_RGBA),
                contype=0,
                conaffinity=0,
            )
    world = spec.worldbody
    court_x = float(config.court_x)
    court_y = float(config.court_y)
    line_z = 0.002  # just above the render-only ground plane at z = 0
    for name, y in (("court_line_y_pos", court_y), ("court_line_y_neg", -court_y)):
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[court_x, 0.010, 0.002],
            pos=[0.0, y, line_z],
            rgba=list(COURT_LINE_RGBA),
            contype=0,
            conaffinity=0,
        )
    for name, x in (("court_line_x_pos", court_x), ("court_line_x_neg", -court_x)):
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.010, court_y, 0.002],
            pos=[x, 0.0, line_z],
            rgba=list(COURT_LINE_RGBA),
            contype=0,
            conaffinity=0,
        )
    world.add_geom(
        name="court_centre",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.010, court_y, 0.002],
        pos=[0.0, 0.0, line_z],
        rgba=list(COURT_CENTRE_RGBA),
        contype=0,
        conaffinity=0,
    )


def _decorate_scene(task: str, spec: Any, mujoco: Any, config: TaskConfig) -> None:
    """Add the task's colours and goal markers to the render-only spec."""
    if task == "cartpole":
        _decorate_cartpole(spec, mujoco, config)
    elif task == "pendulum":
        _decorate_pendulum(spec, mujoco, config)
    elif task == "toss":
        _decorate_toss(spec, mujoco, config)
    else:
        raise ValueError(f"task must be one of {RENDER_TASKS}, got {task!r}")


def _build_render_model(
    task: str, config: TaskConfig, width: int, height: int
) -> tuple[Any, Any, Any]:
    """Build the render-only model copy, its `MjData` and the free camera.

    The copy's joint layout is identical to the env's (it is compiled from the same MJCF),
    so `_write_render_state` can copy the env state into `qpos`/`qvel` verbatim.

    Args:
        task: The suite task whose MJCF and camera to use.
        config: The task config (goal position/amplitude drive the goal markers).
        width: Offscreen framebuffer width (px) the copy must support.
        height: Offscreen framebuffer height (px) the copy must support.

    Returns:
        `(model, data, camera)`: the compiled render-only scene, an `MjData` for it and the
        configured `MjvCamera`.
    """
    mujoco = mujoco_module()
    view = _task_view(task)
    spec = mujoco.MjSpec.from_file(str(_ASSETS_DIR / view.asset_name))
    _decorate_scene(task, spec, mujoco, config)
    _add_demo_environment(spec, mujoco, view)
    # MuJoCo's default offscreen framebuffer is 640x480; enlarge it for any requested size.
    spec.visual.global_.offwidth = max(int(width), int(spec.visual.global_.offwidth))
    spec.visual.global_.offheight = max(int(height), int(spec.visual.global_.offheight))
    model = spec.compile()
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = view.camera.lookat
    camera.distance = view.camera.distance
    camera.azimuth = view.camera.azimuth
    camera.elevation = view.camera.elevation
    return model, data, camera


def _write_render_state(task: str, data: Any, state: dict[str, float]) -> None:
    """Copy the env's kinematic state into the render copy's `qpos`/`qvel`.

    Args:
        task: The suite task the state belongs to.
        data: The render copy's `MjData`.
        state: The env's `state()` dict.

    Raises:
        ValueError: if the state does not fit the render copy.
    """
    if task == "cartpole":
        qpos = [float(state["cart_x"]), float(state["pole_angle"])]
        qvel = [float(state["cart_vx"]), float(state["pole_angvel"])]
    elif task == "pendulum":
        qpos = [float(state["angle"])]
        qvel = [float(state["angvel"])]
    elif task == "toss":
        # The ball is a free sphere moving in the x-z court plane (no y motion by
        # construction), so the y pose/velocity are zero; the sphere's spin is irrelevant.
        # The paddles are PHYSICAL servoed bodies: their x/z/tilt joints live in qpos/qvel
        # (the same layout as `toss_catch.xml`), and the render copy is advanced from the
        # env's measured state, exactly like the simulation.
        qpos = [float(state["ball_x"]), 0.0, float(state["ball_z"]), 1.0, 0.0, 0.0, 0.0]
        qvel = [float(state["ball_vx"]), 0.0, float(state["ball_vz"]), 0.0, 0.0, 0.0]
        for side in AGENT_KEYS:
            qpos.extend(
                (
                    float(state[f"{side}_pad_x"]),
                    float(state[f"{side}_pad_z"]),
                    float(state[f"{side}_pad_tilt"]),
                )
            )
            qvel.extend(
                (
                    float(state[f"{side}_pad_vx"]),
                    float(state[f"{side}_pad_vz"]),
                    float(state[f"{side}_pad_tilt_rate"]),
                )
            )
    else:
        raise ValueError(f"task must be one of {RENDER_TASKS}, got {task!r}")
    if data.qpos.shape != (len(qpos),) or data.qvel.shape != (len(qvel),):
        raise ValueError(
            f"the {task} render copy has qpos {data.qpos.shape} / qvel {data.qvel.shape}, "
            f"but the env state maps to {len(qpos)} / {len(qvel)} values"
        )
    data.qpos[:] = qpos
    data.qvel[:] = qvel


class CoopDemoRenderer:
    """Offscreen renderer of the render-only coop scene copy, driven by the env's state."""

    def __init__(self, task: str, config: TaskConfig, width: int, height: int) -> None:
        """Create the render copy and the offscreen `mujoco.Renderer` at `width` x `height`."""
        mujoco = mujoco_module()
        self._mujoco = mujoco
        self._task = task
        self._model, self._data, self._camera = _build_render_model(task, config, width, height)
        self._renderer = mujoco.Renderer(self._model, height, width)
        self.width = int(width)
        self.height = int(height)

    def render_state(self, state: dict[str, float]) -> np.ndarray:
        """Render one RGB frame (uint8, `height` x `width` x 3) of the given env state."""
        _write_render_state(self._task, self._data, state)
        self._mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, camera=self._camera)
        return np.asarray(self._renderer.render(), dtype=np.uint8)

    def close(self) -> None:
        """Release the offscreen OpenGL context."""
        self._renderer.close()


# --- GPU physics backend --------------------------------------------------------------------


def frame_physics_label(backend: str) -> str:
    """The JSON `frame_physics` label of a rollout backend (`cpu` or `warp`)."""
    return GPU_FRAME_PHYSICS if backend == "warp" else CPU_FRAME_PHYSICS


def require_gpu_stack() -> None:
    """Fail with an install hint when ``--backend warp`` is requested without the GPU stack.

    Raises:
        SystemExit: if torch, `mujoco_warp`/`warp` or a CUDA device is missing, so the CLI
            never silently renders a "GPU" clip on the CPU.
    """
    try:
        import torch

        from callosum.experiments.coop_tasks import env_gpu
    except ImportError as exc:  # pragma: no cover - only on a machine without the train extra
        raise SystemExit(f"--backend warp needs torch + mujoco-warp: {exc}") from exc
    if env_gpu.wp is None:
        raise SystemExit(
            "--backend warp needs the optional GPU backend; install it with "
            "`uv pip install --python /root/.callosum/venv/bin/python mujoco-warp==4.0.1`"
        )
    if not torch.cuda.is_available():
        raise SystemExit("--backend warp needs a CUDA device; none is visible to torch")


def _observation_slices(spec: Any, partner_mode: str) -> dict[str, slice]:
    """Column slices of the `shared` / `goal` / `time` blocks of a GPU env observation.

    `CoopBatchGPU` returns both agents' flat observation vectors. The renderer takes the
    shared-state and goal numbers from them (they are identical for both roles), so the
    demo state can be rebuilt from the env's `final_obs` even on the step that auto-resets
    the finished GPU world. The block order and widths come from the GPU env's own
    `observation_blocks` layout, so the two can never disagree silently.

    Args:
        spec: The task's observation/action layout (`coop_tasks.logic.task_spec`).
        partner_mode: The env's partner mode (`none` / `oracle`).

    Returns:
        `{block_key: slice}` for the `shared`, `goal` and `time` blocks.

    Raises:
        RuntimeError: if the reconstructed layout has no shared/goal/time blocks.
    """
    from callosum.experiments.coop_tasks.env_gpu import observation_blocks

    widths = {
        "action": int(spec.action_dim),
        "command": int(spec.action_dim),
        "params": 3,
        "privileged": 1,
        "shared": len(spec.shared_fields),
        "goal": len(spec.goal_fields),
        "time": 1,
    }
    blocks = observation_blocks(spec, partner_mode)
    slices: dict[str, slice] = {}
    offset = 0
    for key, owner in blocks[AGENT_KEYS[0]]:
        width = widths[key]
        if owner is None:
            slices[key] = slice(offset, offset + width)
        offset += width
    missing = {"shared", "goal", "time"} - set(slices)
    if missing:  # pragma: no cover - guarded by observation_blocks/obs_size
        raise RuntimeError(f"the GPU observation layout has no {sorted(missing)} blocks")
    return slices


@dataclass
class _PendulumTracker:
    """The `PendulumEnv._after_step` amplitude/zero-crossing tracker, replayed on the GPU state.

    `CoopBatchGPU` keeps this tracker on the GPU but exposes neither it nor a live-info method
    for a non-finished world, so the demo adapter replays the same update on the per-step
    angle it already copies to the host for rendering. The formula is transcribed from
    `callosum/experiments/coop_tasks/pendulum.py`; the GPU parity test compares
    `peak_abs_angle`/`completed_cycles` against the CPU env step by step.
    """

    peak_abs_angle: float
    last_angle_sign: float
    completed_cycles: int

    @classmethod
    def reset(cls, angle: float) -> _PendulumTracker:
        """The trackers at episode start (the CPU `_reset_task_state` initialisation)."""
        return cls(
            peak_abs_angle=abs(float(angle)),
            last_angle_sign=1.0 if float(angle) >= 0.0 else -1.0,
            completed_cycles=0,
        )

    def update(self, angle: float, target_amplitude: float) -> None:
        """Advance the tracker by one control step's angle (the CPU `_after_step`)."""
        angle = float(angle)
        self.peak_abs_angle = max(self.peak_abs_angle, abs(angle))
        sign = 1.0 if angle >= 0.0 else -1.0
        if angle != 0.0 and sign != self.last_angle_sign:
            if self.peak_abs_angle >= float(target_amplitude):
                self.completed_cycles += 1
            self.peak_abs_angle = abs(angle)
            self.last_angle_sign = sign


class GpuDemoEnv:
    """One-world GPU physics driver of the demo renderer (`CoopBatchGPU`, `num_envs=1`).

    The class is a thin adapter that gives the demo rollout the surface of the CPU
    `CoopTaskEnv` it uses: `reset(seed=...)`, `step(action)`, `state()`,
    `pending_commands()`, `randomization` and the scripted oracle's static model queries
    (`linear_dynamics`, `energy_parameters`, the toss-catch paddle geometry). Everything
    dynamic -- the control step, the per-episode private randomness, the observations, the
    success/failure rules and the reward -- is the batched `mujoco_warp` backend, so the demo
    physics runs on the GPU.

    The demo state is rebuilt from the env's `final_obs` (the observation computed after the
    control step and BEFORE the auto-reset of a finished world), so the last frame of an
    episode still shows the terminal state even though the GPU env has already restarted the
    world. Only that small render state (~30 floats per step) is copied to the host; the
    learned-policy path keeps the observations as CUDA tensors (`torch_io = True`).

    The per-episode randomisation record and the latency queues are the pure stdlib ones the
    CPU env uses (`logic.sample_episode_randomization` / `logic.delayed_command`): the GPU
    env samples the same records from the same episode seed, so the scripted oracle's
    privileged view of the episode is the CPU demo's (the parity test checks the sampled
    fields against the GPU env's tensors).
    """

    torch_io = True
    """Marker: the policy controller receives `(1, obs_dim)` CUDA tensors from this env."""

    def __init__(
        self,
        task: str,
        config: TaskConfig,
        partner_mode: str,
        *,
        seed: int = 0,
        device: str = "cuda",
    ) -> None:
        """Build the one-world GPU env and the render-state bookkeeping.

        Args:
            task: The coop task (`cartpole`, `pendulum` or `toss`).
            config: The task config the GPU env is built from (the CPU demo's own config).
            partner_mode: Observation visibility mode, `none` or `oracle`.
            seed: Default episode seed; `reset(seed=...)` overrides it per episode.
            device: CUDA device string (`cuda` / `cuda:N`).

        Raises:
            ImportError: if the GPU stack (`mujoco_warp`/`warp`/CUDA torch) is missing.
        """
        import torch

        from callosum.experiments.coop_tasks.env_gpu import CoopBatchGPU
        from callosum.experiments.coop_tasks.logic import delayed_command, task_spec

        self._torch = torch
        self._delayed_command = delayed_command
        self.task = task
        self.config = config
        self.partner_mode = partner_mode
        self._spec = task_spec(task)
        self._slices = _observation_slices(self._spec, partner_mode)
        self._gpu = CoopBatchGPU(
            task,
            1,
            partner_mode=partner_mode,
            config=config,
            seed=int(seed),
            device=device,
        )
        self._device = torch.device(self._gpu.device)
        self._default_seed = int(seed)
        self._randomization: Any | None = None
        self._static_env: Any | None = None
        self._delay: dict[str, deque[np.ndarray]] = {role: deque() for role in AGENT_KEYS}
        self._step_index = 0
        self._state: dict[str, float] = {}
        self._pendulum: _PendulumTracker | None = None

    # --- env surface --------------------------------------------------------------------
    @property
    def randomization(self) -> Any | None:
        """The episode's sampled randomisation record (the CPU env's `CoopTaskEnv.randomization`)."""
        return self._randomization

    @property
    def device(self) -> str:
        """The CUDA device string of the GPU env."""
        return str(self._gpu.device)

    @property
    def cuda_graph(self) -> bool:
        """Whether the env's control step runs from a captured CUDA graph."""
        return bool(self._gpu.cuda_graph)

    @property
    def software_versions(self) -> str:
        """The simulator/GPU software versions of the env (for the JSON report)."""
        return str(self._gpu.software_versions())

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode: sample its randomness, reset the GPU world, return the obs.

        Args:
            seed: Episode seed; `None` uses the constructor's seed. The GPU env resets its
                single world with exactly this seed, exactly like the CPU env's
                `reset(seed=...)`.
            options: Unused (gymnasium compatibility).

        Returns:
            `({role: (1, obs_dim) CUDA tensor}, {})`; the demo rollout ignores the info dict
            (the CPU env's initial diagnostics are not shown in any frame).
        """
        del options
        episode_seed = self._default_seed if seed is None else int(seed)
        observations = self._gpu.reset(seed=episode_seed)
        self._randomization = self._sample_randomization(episode_seed)
        latencies = [int(agent.latency_steps) for agent in self._randomization.agents]
        self._delay = {
            role: deque(np.zeros(self._spec.action_dim) for _ in range(latencies[index]))
            for index, role in enumerate(AGENT_KEYS)
        }
        self._step_index = 0
        self._refresh_state(observations, fresh_episode=True)
        return {role: observations[role] for role in AGENT_KEYS}, {}

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance the GPU world by one control step (auto-resetting a finished world).

        Args:
            action: `{role: (action_dim,)}` normalized commands (numpy or float32 arrays).

        Returns:
            `(obs, reward, terminated, truncated, info)` with the next-step observations as
            CUDA tensors, the shared team reward as a float and the CPU env's terminal info
            dict on the step that ends the episode (an empty dict otherwise, exactly like the
            only frame the renderer decorates with it).

        Raises:
            ValueError: if an action has the wrong shape.
        """
        commands: dict[str, np.ndarray] = {}
        for role in AGENT_KEYS:
            vector = np.asarray(action[role], dtype=np.float64).reshape(-1)
            if vector.shape != (self._spec.action_dim,):
                raise ValueError(
                    f"action[{role!r}] must have shape ({self._spec.action_dim},), "
                    f"got {vector.shape}"
                )
            commands[role] = np.clip(vector, -1.0, 1.0)
            # Keep the CPU-latency view in sync with the commands the GPU env receives, so
            # the scripted oracle's `pending_commands()` is the CPU env's own queue content.
            self._delayed_command(self._delay[role], commands[role].copy())
        tensors = {
            role: self._torch.as_tensor(
                commands[role].reshape(1, -1),
                dtype=self._torch.float32,
                device=self._device,
            )
            for role in AGENT_KEYS
        }
        previous_state = dict(self._state)
        result = self._gpu.step(tensors)
        self._step_index += 1
        terminated = bool(result.terminated[0])
        truncated = bool(result.truncated[0])
        non_finite = bool(result.nonfinite[0]) if result.nonfinite is not None else False
        if non_finite:
            # The float32 contact solver produced a non-finite state (treated as a FAILED
            # episode and reset by the GPU env); keep the last valid frame state for the
            # FAILED still instead of rendering the zeroed placeholder state.
            self._state = previous_state
        else:
            self._refresh_state(result.final_obs, fresh_episode=False)
        info: dict[str, Any] = {}
        if (terminated or truncated) and result.finished_infos:
            info = result.finished_infos[0]
        observations = {role: result.obs[role] for role in AGENT_KEYS}
        return observations, float(result.reward[0]), terminated, truncated, info

    def state(self) -> dict[str, float]:
        """The task's kinematic state plus the step index (the CPU env's `state()` dict)."""
        return dict(self._state)

    def pending_commands(self) -> dict[str, tuple[np.ndarray, ...]]:
        """The agents' in-flight commands, oldest first (the CPU env's `pending_commands()`)."""
        return {
            role: tuple(np.array(command, dtype=np.float64, copy=True) for command in queue)
            for role, queue in self._delay.items()
        }

    def close(self) -> None:
        """Release the GPU env (and the never-stepped CPU model env if it was built)."""
        self._gpu.close()
        if self._static_env is not None:
            self._static_env.close()
            self._static_env = None

    # --- scripted-oracle model queries --------------------------------------------------
    def linear_dynamics(self) -> tuple[np.ndarray, np.ndarray]:
        """The cart-pole linearisation from the task's MJCF (CPU model read, no physics)."""
        return self._model_env().linear_dynamics()

    def energy_parameters(self) -> tuple[float, float]:
        """The pendulum energy coefficients from the task's MJCF (CPU model read)."""
        return self._model_env().energy_parameters()

    def contact_offset(self) -> float:
        """The toss-catch ball/paddle contact offset (task geometry, no physics)."""
        return float(self._model_env().contact_offset())

    def paddle_receive_x(self) -> float:
        """The toss-catch receiving paddle's anticipation x (config geometry)."""
        return float(self._model_env().paddle_receive_x())

    def paddle_ready_x(self) -> float:
        """The toss-catch ready-pose paddle centre x (config geometry)."""
        return float(self._model_env().paddle_ready_x())

    def paddle_ready_tilt(self, side: str) -> float:
        """The toss-catch ready-pose tilt of one paddle (config geometry)."""
        return float(self._model_env().paddle_ready_tilt(side))

    def _model_env(self) -> Any:
        """Build (once) the never-stepped CPU env the static MJCF/geometry queries go through.

        The GPU backend builds its own model from the same MJCF, so these values are
        model-derived constants (masses/inertias/paddle geometry), not physics results; the
        CPU instance is never reset or stepped.
        """
        if self._static_env is None:
            from callosum.experiments.coop_tasks import make_coop_env

            self._static_env = make_coop_env(
                self.task, config=self.config, partner_mode=self.partner_mode
            )
        return self._static_env

    # --- render-state bookkeeping -------------------------------------------------------
    def _sample_randomization(self, episode_seed: int) -> Any:
        """Sample the episode's randomness with the CPU env's samplers (same seed/fields)."""
        from callosum.experiments.coop_tasks.logic import (
            sample_episode_randomization,
            sample_toss_randomization,
        )

        sampler = sample_toss_randomization if self.task == "toss" else sample_episode_randomization
        return sampler(
            self.config.randomization,
            control_dt=self.config.control_dt,
            max_control_steps=self.config.max_control_steps,
            seed=episode_seed,
        )

    def _refresh_state(self, observations: dict[str, Any], *, fresh_episode: bool) -> None:
        """Rebuild the demo `state()` dict from the GPU observation of the current step.

        Args:
            observations: The env's observations, `{role: (1, obs_dim) CUDA tensor}`; the
                `left` role's shared/goal blocks carry the whole task state.
            fresh_episode: True right after `reset` (initialise the pendulum tracker),
                False after a control step (advance it).
        """
        values = observations[AGENT_KEYS[0]][0].tolist()
        shared = values[self._slices["shared"]]
        goal = values[self._slices["goal"]]
        state: dict[str, float] = {"step": float(self._step_index)}
        if self.task == "cartpole":
            state.update(zip(self._spec.shared_fields, (float(value) for value in shared)))
            state["target_x"] = float(self.config.target_x)
        elif self.task == "pendulum":
            angle, angvel = float(shared[0]), float(shared[1])
            target = float(self.config.target_amplitude)
            if fresh_episode or self._pendulum is None:
                self._pendulum = _PendulumTracker.reset(angle)
            else:
                self._pendulum.update(angle, target)
            state.update(
                angle=angle,
                angvel=angvel,
                target_amplitude=target,
                peak_abs_angle=float(self._pendulum.peak_abs_angle),
                completed_cycles=float(self._pendulum.completed_cycles),
            )
        else:
            state.update(zip(self._spec.shared_fields, (float(value) for value in shared)))
            state["passes"] = float(goal[0])
            state["target_passes"] = float(self.config.n_passes)
            state.update(self._toss_contact_flags())
            # The scripted toss oracle plans its deadbeat command in SERVO-TARGET space (the
            # plates are physical bounded-force servos now, so the measured pose lags the
            # target it is chasing). The targets are not part of the shared observation block,
            # so they are read from the GPU env's task state.
            state.update(self._toss_pad_targets())
        self._state = state

    def _toss_contact_flags(self) -> dict[str, float]:
        """The toss-catch paddle contact flags of the current world (the CPU state keys).

        The batched GPU env keeps the flags in a Warp-backed tensor that the shared
        observation does not carry, while the scripted toss oracle reads them for its
        restitution estimate. The attribute is read defensively and defaults to "no contact"
        so a future backend without the flags cannot crash the renderer.
        """
        contact = getattr(self._gpu, "_contact_side", None)
        if contact is None:  # pragma: no cover - the current GPU env defines it for toss
            return {f"{side}_contact": 0.0 for side in AGENT_KEYS}
        values = contact[0].tolist()
        return {f"{side}_contact": float(values[index]) for index, side in enumerate(AGENT_KEYS)}

    def _toss_pad_targets(self) -> dict[str, float]:
        """The toss-catch paddle servo targets `(x, z, tilt)` of the current world.

        The GPU env's `_pad_target` holds the command-limited servo references the plates
        physically chase; the scripted oracle's deadbeat command is computed against them,
        so the demo adapter must expose them under the CPU state's
        `{side}_pad_target_{x,z,tilt}` keys. The attribute is read defensively and defaults
        to the measured pose so a future backend without explicit targets cannot crash the
        renderer.
        """
        targets = getattr(self._gpu, "_pad_target", None)
        if targets is None:  # pragma: no cover - the current GPU env defines it for toss
            return {
                f"{side}_pad_target_{name}": float(self._state[f"{side}_pad_{name}"])
                for side in AGENT_KEYS
                for name in ("x", "z", "tilt")
            }
        values = targets[0].tolist()
        return {
            f"{side}_pad_target_{axis}": float(values[index][position])
            for index, side in enumerate(AGENT_KEYS)
            for position, axis in enumerate(("x", "z", "tilt"))
        }


# --- action sources -------------------------------------------------------------------------


class OracleDemoController:
    """The task's scripted oracle behind the demo's controller protocol.

    The oracle is a centralised script that reads the whole `env.state()`; the demo uses it
    as the privileged upper bound, not as a policy. The adapter gives it the demo's
    `reset`/`act`/`observe` protocol (the oracle keeps no per-step state, so `reset` and
    `observe` are no-ops).
    """

    label = "oracle"

    def __init__(self, task: str, config: TaskConfig) -> None:
        """Wrap the task's scripted oracle (`callosum.experiments.coop_tasks.make_oracle`)."""
        from callosum.experiments.coop_tasks import make_oracle

        self.task = task
        self.title = "scripted oracle"
        self.subtitle = ORACLE_SUBTITLES[task]
        self._oracle = make_oracle(task, config)

    def reset(self, obs: DemoObs) -> None:
        """No-op: the oracle has no per-episode state (it reads the env in `act`)."""
        del obs

    def act(self, env: Any, obs: DemoObs) -> dict[str, np.ndarray]:
        """The oracle's normalised one-sided commands for the next control step."""
        del obs
        return self._oracle.act(env)

    def observe(self, obs: DemoObs) -> None:
        """No-op: the oracle needs no observation history."""
        del obs


class PolicyDemoController:
    """Greedy `wrench_ppo` coop policies behind the demo's controller protocol.

    Loading and inference reuse `callosum/experiments/coop_tasks/eval.py` and
    `callosum.experiments.cube_wrench.train`: `load_checkpoint` reads the payload,
    `coop_eval.resolve_input_mode` decides the policy input mode (flag or checkpoint),
    `check_checkpoint_compat` validates it, and the `predicted` input is built by
    `LatentInputBuilder` with exactly the trainer's/evaluator's `reset`/`step` seam. Actions
    are the distribution means, clipped to `[-1, 1]`.

    `predicted` keeps the env at `partner_mode="none"` (the agent cannot see the privileged
    partner block it is asked to predict); `none`/`oracle` map onto themselves.

    On the GPU physics backend the policies, the Bi-JEPA encoder/predictor and the
    `predicted` history builder are built on the CUDA device and receive the GPU env's
    observation tensors directly; the CPU backend keeps everything on the CPU unchanged.
    """

    def __init__(
        self,
        checkpoint: Path,
        input_mode: str | None,
        task: str,
        device: str = "cpu",
    ) -> None:
        """Load the checkpoint and build both greedy policies (and the Bi-JEPA builder).

        Args:
            checkpoint: A `wrench_ppo` coop `latest.pt` checkpoint.
            input_mode: `none` / `oracle` / `predicted`, or None to use the checkpoint's own
                saved mode (see `coop_tasks.eval.resolve_input_mode`).
            task: The `--task` of the render; must match the checkpoint's task.
            device: Torch device of the policies and the Bi-JEPA builder (`cpu` for the CPU
                physics backend, `cuda`/`cuda:N` for the GPU one).

        Raises:
            ValueError: if the checkpoint cannot be run for this task/input mode.
        """
        import torch

        from callosum.experiments.coop_tasks import eval as coop_eval
        from callosum.experiments.cube_wrench.train import (
            LatentInputBuilder,
            check_checkpoint_compat,
            config_from_payload,
            load_bijepa_weights,
            load_checkpoint,
            load_weights,
            make_bijepa,
            make_policies,
        )

        self._torch = torch
        self._device = torch.device(device)
        self.checkpoint = Path(checkpoint)
        payload = load_checkpoint(self.checkpoint)
        cfg = config_from_payload(payload)
        if cfg.task != task:
            raise ValueError(
                f"checkpoint {self.checkpoint} was trained on task {cfg.task!r}, "
                f"but --task is {task!r}"
            )
        resolved = coop_eval.resolve_input_mode(
            payload,
            argparse.Namespace(checkpoint=str(self.checkpoint), input_mode=input_mode),
        )
        env_mode, latent_dim = check_checkpoint_compat(payload, resolved)
        self.task = task
        self.config = cfg.coop_env_config()
        self.env_mode = env_mode
        self.input_mode = resolved
        self.latent_dim = latent_dim
        self.env_obs_size = cfg.env_obs_size
        self.action_size = cfg.action_size
        self.agent_keys = tuple(cfg.agent_keys)
        self.update = payload.get("update")
        self.format = payload.get("format")
        self.policies = make_policies(
            env_mode,
            self._device,
            latent_dim=latent_dim,
            input_size=cfg.env_obs_size,
            action_dim=cfg.action_size,
        )
        load_weights(payload, self.policies, agent_keys=cfg.agent_keys)
        self._builder: Any | None = None
        bijepa = make_bijepa(cfg, self._device)
        if bijepa is not None:
            load_bijepa_weights(payload, bijepa)
            bijepa.eval()
            self._builder = LatentInputBuilder(
                bijepa, device=self._device, agent_keys=cfg.agent_keys
            )
        self._policy_obs: dict[str, Any] | None = None
        self.title = f"learned policy ({self.input_mode})"
        self.subtitle = (
            f"greedy IPPO · {self.checkpoint.name} · update {self.update} · "
            f"{self.env_mode} observation"
        )
        self.label = f"policy_{self.input_mode}"

    def metadata(self) -> dict[str, Any]:
        """The JSON-ready checkpoint metadata stored in the demo summary."""
        return {
            "checkpoint": str(self.checkpoint),
            "task": self.task,
            "input_mode": self.input_mode,
            "env_partner_mode": self.env_mode,
            "latent_dim": self.latent_dim,
            "env_obs_size": self.env_obs_size,
            "action_size": self.action_size,
            "policy_device": str(self._device),
            "update": self.update,
            "format": self.format,
        }

    def _batch_obs(self, obs: DemoObs) -> dict[str, torch.Tensor]:
        """A single-world observation as a `(1, obs_dim)` float32 batch on the policy device.

        Accepts the CPU backend's numpy observation dict and the GPU backend's CUDA
        `(1, obs_dim)` tensors, so the two paths share the policy/Bi-JEPA seam unchanged.
        """
        batch: dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            tensor = value if self._torch.is_tensor(value) else self._torch.as_tensor(value)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            batch[key] = tensor.to(device=self._device, dtype=self._torch.float32)
        return batch

    def reset(self, obs: DemoObs) -> None:
        """Start a new episode: the policy input is the first observation (history reset)."""
        batch = self._batch_obs(obs)
        self._policy_obs = self._builder.reset(batch) if self._builder is not None else batch

    def act(self, env: Any, obs: DemoObs) -> dict[str, np.ndarray]:
        """The greedy (distribution mean) action of both policies, clipped to `[-1, 1]`."""
        del env, obs
        if self._policy_obs is None:
            raise RuntimeError("call reset() before the policy acts")
        actions: dict[str, np.ndarray] = {}
        for name, policy in zip(self.agent_keys, self.policies):
            with self._torch.no_grad():
                action = policy.get_action(self._policy_obs[name], deterministic=True)
            actions[name] = self._torch.clamp(action, -1.0, 1.0).squeeze(0).cpu().numpy()
        return actions

    def observe(self, obs: DemoObs) -> None:
        """Advance the policy input after a non-terminal step (Bi-JEPA history under `predicted`)."""
        batch = self._batch_obs(obs)
        self._policy_obs = (
            self._builder.step(batch, self._torch.zeros(1, device=self._device))
            if self._builder is not None
            else batch
        )


DemoController = OracleDemoController | PolicyDemoController
"""The demo's two action sources; both expose `reset` / `act` / `observe` and a caption."""


# --- rollout --------------------------------------------------------------------------------


@dataclass
class DemoFrame:
    """One decorated frame of a rollout."""

    image: np.ndarray
    step: int
    time_s: float
    status: str | None = None


@dataclass
class DemoEpisode:
    """One rendered rollout: its frames and the outcome (success / failure / timeout)."""

    task: str
    name: str
    seed: int
    title: str
    subtitle: str
    frames: list[DemoFrame]
    steps: int
    success: bool
    terminated: bool
    truncated: bool
    status: str
    status_detail: str
    metrics: dict[str, float]
    control_dt: float

    def summary(self) -> dict[str, Any]:
        """The JSON-serialisable metrics of the episode."""
        return {
            "seed": self.seed,
            "status": self.status,
            "status_detail": self.status_detail,
            "success": self.success,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "steps": self.steps,
            "time_s": round(self.steps * self.control_dt, 3),
            "metrics": {key: round(float(value), 4) for key, value in self.metrics.items()},
            "unique_frames": len(self.frames),
        }


def run_episode(
    env: CoopTaskEnv | GpuDemoEnv,
    controller: DemoController,
    renderer: CoopDemoRenderer | None,
    *,
    task: str,
    config: TaskConfig,
    seed: int,
    step_stride: int,
    max_steps: int,
    title: str,
    subtitle: str,
    name: str,
) -> DemoEpisode:
    """Run one rollout and return its rendered frames plus the outcome.

    Frames are captured at the start of every `step_stride`-th control step and once more
    after the terminating step (the final frame carries the SUCCESS / FAILED / TIMEOUT
    status). `renderer=None` runs the same rollout without rendering (the `--dry-run` path).

    The controller seam is `reset(obs)` once, then `act(env, obs)` before every step and
    `observe(obs)` after every non-terminal step (the Bi-JEPA `predicted` input needs it).
    The env is either a CPU `CoopTaskEnv` or the GPU `GpuDemoEnv`, which share this seam
    (physics only differs in where the control step runs).
    """
    obs, _reset_info = env.reset(seed=seed)
    controller.reset(obs)
    frames: list[DemoFrame] = []
    best_amplitude_deg = 0.0

    def capture(
        step: int,
        action: dict[str, np.ndarray],
        *,
        status: str | None = None,
        status_color: tuple[int, int, int, int] = MUTED_COLOR,
        detail: str = "",
    ) -> None:
        nonlocal best_amplitude_deg
        state = env.state()
        best_amplitude_deg = max(best_amplitude_deg, _peak_amplitude_deg(task, state))
        hud = _task_hud(task, config, state, step, max_steps, action, best_amplitude_deg)
        if renderer is None:
            image = np.zeros((8, 8, 3), dtype=np.uint8)
        else:
            raw = renderer.render_state(state)
            image = annotate_frame(
                raw,
                title=title,
                subtitle=subtitle,
                hud_lines=hud.lines,
                agent_blocks=hud.agent_blocks,
                angle_fraction=hud.progress,
                status=status,
                status_color=status_color,
                status_detail=detail or None,
            )
        frames.append(
            DemoFrame(
                image=image,
                step=step,
                time_s=step * config.control_dt,
                status=status,
            )
        )

    action = controller.act(env, obs)
    capture(0, action)
    steps = 0
    terminated = truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
        controller.observe(obs)
        if steps >= max_steps:
            truncated = True
            break
        action = controller.act(env, obs)
        if steps % step_stride == 0:
            capture(steps, action)

    final_state = env.state()
    best_amplitude_deg = max(best_amplitude_deg, _peak_amplitude_deg(task, final_state))
    metrics = _task_metrics(task, final_state, config)
    if task == "pendulum":
        # The achieved (peak) amplitude is the task's score, not the current half-cycle one.
        metrics["peak_amplitude_deg"] = best_amplitude_deg
        target_deg = math.degrees(float(config.target_amplitude))
        metrics["progress"] = float(np.clip(best_amplitude_deg / max(target_deg, 1e-6), 0.0, 1.0))
    success = bool(info.get("success", False))
    status, detail, status_color = _episode_status(
        task, config, steps, terminated, truncated, success, metrics
    )
    capture(steps, action, status=status, status_color=status_color, detail=detail)
    return DemoEpisode(
        task=task,
        name=name,
        seed=seed,
        title=title,
        subtitle=subtitle,
        frames=frames,
        steps=steps,
        success=success,
        terminated=terminated,
        truncated=truncated,
        status=status,
        status_detail=detail,
        metrics=metrics,
        control_dt=config.control_dt,
    )


# --- clip encoding ---------------------------------------------------------------------------


def _duplicate_for_target(frame_count: int, fps: int, target_seconds: float) -> int:
    """How many times each rendered frame is held to reach `target_seconds` (slow motion)."""
    if frame_count <= 0:
        return 1
    return max(1, round(float(target_seconds) * float(fps) / float(frame_count)))


def _repeat_frames(frames: Sequence[DemoFrame], duplicate: int) -> Iterator[np.ndarray]:
    """Yield each frame's image `duplicate` times (the slow-motion expansion)."""
    for frame in frames:
        for _ in range(duplicate):
            yield frame.image


def _repeat_count(frames: Sequence[DemoFrame], duplicate: int) -> int:
    """Number of video frames after the slow-motion expansion."""
    return len(frames) * duplicate


def _expanded_index(frames: Sequence[DemoFrame], index: int, duplicate: int) -> np.ndarray:
    """The image at video-frame `index` of a clip expanded `duplicate` times."""
    return frames[min(index // duplicate, len(frames) - 1)].image


def _sbs_frame_at(
    left: DemoEpisode,
    right: DemoEpisode,
    index: int,
    duplicate_left: int,
    duplicate_right: int,
) -> np.ndarray:
    """The side-by-side RGB frame at video-frame `index` (also used for the PNG stills)."""
    return hstack_side_by_side(
        _expanded_index(left.frames, index, duplicate_left),
        _expanded_index(right.frames, index, duplicate_right),
    )


def _side_by_side_frames(
    left: DemoEpisode,
    right: DemoEpisode,
    *,
    duplicate_left: int,
    duplicate_right: int,
) -> Iterator[np.ndarray]:
    """Yield the two clips side by side, each half expanded to its own slow-motion factor.

    The shorter half holds its final frame while the longer one finishes, so both end states
    are visible in the last frames (the final still is the true `max` of the two lengths).
    """
    total = max(
        _repeat_count(left.frames, duplicate_left), _repeat_count(right.frames, duplicate_right)
    )
    for index in range(total):
        yield _sbs_frame_at(left, right, index, duplicate_left, duplicate_right)


def _sbs_duplicates(
    left: DemoEpisode, right: DemoEpisode, fps: int, target_seconds: float
) -> tuple[int, int]:
    """Frame-hold factors of the two halves of a policy-vs-oracle clip.

    Each half gets the hold factor that makes it finish at the same video time, so the
    comparison stays honest while the slower policy gets the same screen time as the oracle
    (the HUD step counters stay truthful either way).
    """
    longer = left if len(left.frames) >= len(right.frames) else right
    shorter = right if longer is left else left
    duplicate_longer = _duplicate_for_target(len(longer.frames), fps, target_seconds)
    duplicate_shorter = max(
        1, round(duplicate_longer * len(longer.frames) / max(1, len(shorter.frames)))
    )
    if longer is left:
        return duplicate_longer, duplicate_shorter
    return duplicate_shorter, duplicate_longer


def _md5_image(image: np.ndarray) -> str:
    """Short md5 of an RGB frame (proof that the saved stills really differ)."""
    return hashlib.md5(np.ascontiguousarray(image, dtype=np.uint8).tobytes()).hexdigest()[:12]


def _save_image(path: Path, image: np.ndarray) -> None:
    """Save one RGB frame as PNG (creating the parent directory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(image)).save(path)


def _check_video(path: Path, written_frames: int, sample: Sequence[np.ndarray]) -> dict[str, Any]:
    """Probe a written video, check it is non-empty and not black; return its metrics."""
    metrics: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size) if path.exists() else 0,
        "video_frames": int(written_frames),
        "mean_luma": round(float(np.mean([mean_luma(frame) for frame in sample])), 2),
    }
    probe = probe_video(path)
    if probe:
        metrics["probe"] = probe
    if metrics["bytes"] <= 0:
        raise RuntimeError(f"video {path} is empty")
    if probe.get("frames", written_frames) < written_frames:
        raise RuntimeError(f"video {path} has {probe['frames']} frames, expected {written_frames}")
    if metrics["mean_luma"] < 5.0:
        raise RuntimeError(f"video {path} looks black (mean luma {metrics['mean_luma']})")
    return metrics


def _encode_clip(
    path: Path,
    frames: Iterator[np.ndarray] | Sequence[np.ndarray],
    written_frames: int,
    sample: Sequence[np.ndarray],
    fps: float,
    encoders: list[str],
) -> dict[str, Any]:
    """Encode one MP4 clip and verify the result."""
    encoders.append(write_mp4(path, frames, fps))
    return _check_video(path, written_frames, sample)


def _write_gif_from_images(
    path: Path,
    images: Iterator[np.ndarray],
    *,
    fps: int,
    gif_width: int,
    gif_stride: int,
) -> dict[str, Any]:
    """Write a looping GIF preview from a strided iterator of RGB frames."""
    written = write_gif(
        path,
        images,
        frame_duration_ms=round(1000.0 * gif_stride / fps),
        width=int(gif_width),
    )
    with Image.open(path) as gif_image:
        gif_frames = int(getattr(gif_image, "n_frames", 1))
        gif_frame_ms = int(gif_image.info.get("duration", 0) or 0)
        gif_size = gif_image.size
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "frames": gif_frames,
        "written_frames": int(written),
        "duration_s": round(gif_frames * max(gif_frame_ms, 1) / 1000.0, 2),
        "width": gif_size[0],
        "height": gif_size[1],
        "gif_stride": gif_stride,
    }


def _write_single_clip(
    out_dir: Path,
    filename: str,
    stem: str,
    episode: DemoEpisode,
    *,
    fps: int,
    target_seconds: float,
    gif_width: int | None,
    gif_stride: int,
    encoders: list[str],
    stills: bool = True,
) -> dict[str, Any]:
    """Encode a single-scene clip plus its stills (frame0-2, mid, final) and GIF preview."""
    duplicate = _duplicate_for_target(len(episode.frames), fps, target_seconds)
    total = _repeat_count(episode.frames, duplicate)
    first = [_expanded_index(episode.frames, index, duplicate) for index in range(PNG_STILLS)]
    middle = _expanded_index(episode.frames, round(total * 0.5), duplicate)
    final = episode.frames[-1].image
    clip = _encode_clip(
        out_dir / filename,
        _repeat_frames(episode.frames, duplicate),
        total,
        [first[0], middle, final],
        fps,
        encoders,
    )
    clip.update({"duplicate": duplicate, "episode": episode.summary()})
    if stills:
        save_png_stills(out_dir, stem, first, count=len(first))
        _save_image(out_dir / f"{stem}_mid.png", middle)
        hashes = {
            "frame0": _md5_image(first[0]),
            "mid": _md5_image(middle),
            "final": _md5_image(final),
        }
        _save_image(out_dir / f"{stem}_final.png", final)
        clip["still_md5"] = hashes
        print(
            f"{stem} stills: frame0={hashes['frame0']} mid={hashes['mid']} "
            f"final={hashes['final']} distinct={len(set(hashes.values())) == len(hashes)}"
        )
    else:
        still = out_dir / f"{stem}_final.png"
        _save_image(still, final)
        clip["final_still"] = {"path": str(still.resolve()), "md5": _md5_image(final)}
    if gif_width is not None:
        clip["gif"] = _write_gif_from_images(
            out_dir / f"{stem}_preview.gif",
            (
                frame
                for index, frame in enumerate(_repeat_frames(episode.frames, duplicate))
                if index % gif_stride == 0
            ),
            fps=fps,
            gif_width=gif_width,
            gif_stride=gif_stride,
        )
    return clip


def _write_sbs_clip(
    out_dir: Path,
    filename: str,
    stem: str,
    left: DemoEpisode,
    right: DemoEpisode,
    *,
    fps: int,
    target_seconds: float,
    gif_width: int | None,
    gif_stride: int,
    encoders: list[str],
) -> dict[str, Any]:
    """Encode a side-by-side clip (policy left, oracle right) plus stills and GIF preview."""
    duplicate_left, duplicate_right = _sbs_duplicates(left, right, fps, target_seconds)
    total = max(
        _repeat_count(left.frames, duplicate_left),
        _repeat_count(right.frames, duplicate_right),
    )
    first = [
        _sbs_frame_at(left, right, index, duplicate_left, duplicate_right)
        for index in range(PNG_STILLS)
    ]
    middle = _sbs_frame_at(left, right, round(total * 0.5), duplicate_left, duplicate_right)
    final = _sbs_frame_at(left, right, total - 1, duplicate_left, duplicate_right)
    if np.array_equal(final, first[0]):
        raise RuntimeError(f"{stem}: the final still equals frame 0 -- the clip has no time axis")
    clip = _encode_clip(
        out_dir / filename,
        _side_by_side_frames(
            left, right, duplicate_left=duplicate_left, duplicate_right=duplicate_right
        ),
        total,
        [first[0], middle, final],
        fps,
        encoders,
    )
    clip.update(
        {
            "duplicate_left": duplicate_left,
            "duplicate_right": duplicate_right,
            "left_episode": left.summary(),
            "right_episode": right.summary(),
        }
    )
    save_png_stills(out_dir, stem, first, count=len(first))
    _save_image(out_dir / f"{stem}_mid.png", middle)
    _save_image(out_dir / f"{stem}_final.png", final)
    hashes = {
        "frame0": _md5_image(first[0]),
        "mid": _md5_image(middle),
        "final": _md5_image(final),
    }
    clip["still_md5"] = hashes
    print(
        f"{stem} stills: frame0={hashes['frame0']} mid={hashes['mid']} "
        f"final={hashes['final']} distinct={len(set(hashes.values())) == len(hashes)}"
    )
    if gif_width is not None:
        clip["gif"] = _write_gif_from_images(
            out_dir / f"{stem}_preview.gif",
            (
                frame
                for index, frame in enumerate(
                    _side_by_side_frames(
                        left, right, duplicate_left=duplicate_left, duplicate_right=duplicate_right
                    )
                )
                if index % gif_stride == 0
            ),
            fps=fps,
            gif_width=gif_width,
            gif_stride=gif_stride,
        )
    return clip


# --- CLI ------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the coop demo-render CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Render the two-robot coop-suite demos (scripted oracle, learned policy vs "
            "oracle, or N attempts) with MuJoCo offscreen rendering."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=RENDER_TASKS,
        required=True,
        help="the coop task to render",
    )
    parser.add_argument(
        "--scene",
        choices=("oracle", "policy-vs-oracle", "attempts"),
        default="oracle",
        help=(
            "oracle: one scripted-oracle success clip; policy-vs-oracle: learned policy "
            "(left) vs scripted oracle (right), needs --checkpoint; attempts: N episodes on "
            "consecutive seeds (oracle, or the checkpoint policies when given)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="wrench_ppo coop latest.pt for the policy scenes",
    )
    parser.add_argument(
        "--input-mode",
        choices=("none", "oracle", "predicted"),
        default=None,
        help=(
            "policy input mode of the checkpoint (default: the checkpoint's own saved mode); "
            "the env uses partner_mode=oracle only for 'oracle'"
        ),
    )
    parser.add_argument(
        "--episodes",
        "--attempts",
        dest="attempts",
        type=int,
        default=3,
        help="number of episodes of the attempts scene (3-5 makes a good demo)",
    )
    parser.add_argument("--seed", type=int, default=0, help="base episode seed")
    parser.add_argument(
        "--backend",
        choices=("auto",) + PHYSICS_BACKENDS,
        default="auto",
        help=(
            "rollout physics backend: auto (default: warp when a CUDA device is visible, else "
            "cpu), warp (the rollout steps the batched mujoco_warp GPU env; frames are still "
            "rendered by EGL on the GPU) or cpu (single-instance MuJoCo, parity/CI)"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device of the warp physics backend (default: cuda)",
    )
    parser.add_argument("--width", type=int, default=640, help="frame width (px)")
    parser.add_argument("--height", type=int, default=480, help="frame height (px)")
    parser.add_argument("--fps", type=int, default=15, help="video frame rate")
    parser.add_argument("--out", type=Path, required=True, help="output directory (created)")
    parser.add_argument(
        "--policy-max-steps",
        type=int,
        default=None,
        help=(
            "rollout cap of the learned-policy episode (default: twice the oracle episode, "
            "at least 150 steps, capped by the env's max_control_steps)"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="cap the oracle/attempt rollout length (default: the env's max_control_steps)",
    )
    parser.add_argument(
        "--step-stride",
        type=int,
        default=1,
        help="render every Nth control step (1 = smoothest; 2-4 = more compact flipbook)",
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=6.0,
        help="approximate clip length: frames are held (slow motion) until it is reached",
    )
    parser.add_argument(
        "--gif-width",
        type=int,
        default=480,
        help="width of the GIF preview (0 disables the GIF)",
    )
    parser.add_argument("--dry-run", action="store_true", help="rollouts without rendering")
    args = parser.parse_args(argv)
    _resolve_backend(args)
    return args


def _resolve_backend(args: argparse.Namespace) -> None:
    """Resolve ``--backend auto`` (the GPU-first default on a CUDA host) in place.

    ``auto`` mirrors the evaluator (`callosum.experiments.coop_tasks.eval`): ``warp`` when a
    CUDA device looks visible, ``cpu`` otherwise. The GPU stack itself is validated later by
    `require_gpu_stack`, so ``--help`` and the CPU path never import it.
    """
    from callosum.experiments.cube_wrench.ppo_config import cuda_visible

    if args.backend == "auto":
        args.backend = "warp" if cuda_visible() else "cpu"
    if args.backend == "warp":
        args.device = args.device or "cuda"
    else:
        args.device = None


def _validate_args(args: argparse.Namespace) -> None:
    """Reject CLI combinations that cannot run, before any simulator work."""
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        raise SystemExit(f"--width/--height must be even positive, got {args.width}x{args.height}")
    if args.fps < 1:
        raise SystemExit(f"--fps must be >= 1, got {args.fps}")
    if args.step_stride < 1:
        raise SystemExit(f"--step-stride must be >= 1, got {args.step_stride}")
    if args.attempts < 1:
        raise SystemExit(f"--attempts must be >= 1, got {args.attempts}")
    if args.target_seconds <= 0:
        raise SystemExit(f"--target-seconds must be > 0, got {args.target_seconds}")
    if args.policy_max_steps is not None and args.policy_max_steps < 1:
        raise SystemExit(f"--policy-max-steps must be >= 1, got {args.policy_max_steps}")
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit(f"--max-steps must be >= 1, got {args.max_steps}")
    if args.scene == "policy-vs-oracle" and args.checkpoint is None:
        raise SystemExit("--scene policy-vs-oracle requires --checkpoint PATH")


def _policy_max_steps(args: argparse.Namespace, env_max_steps: int) -> int:
    """Rollout cap of the learned-policy episode.

    Default: the env's own episode budget, so the clip shows the policy's *true* outcome
    (success, failure or the env's time limit) instead of an artificial demo cut-off; the
    policy is given no less time than the env gives it during training/evaluation. Pass
    ``--policy-max-steps`` for a shorter flipbook.
    """
    if args.policy_max_steps is not None:
        return max(1, min(int(args.policy_max_steps), env_max_steps))
    return env_max_steps


def _load_policy_controller(args: argparse.Namespace) -> PolicyDemoController | None:
    """Build the learned-policy controller when the scene needs it (clean CLI error).

    The policies/Bi-JEPA run on the GPU when the rollout physics does (``--backend warp``)
    and stay on the CPU otherwise, exactly like the evaluator's two backend paths.
    """
    if args.checkpoint is None:
        return None
    if args.scene not in ("policy-vs-oracle", "attempts"):
        print(
            f"note: --checkpoint is ignored for --scene {args.scene} "
            "(use policy-vs-oracle or attempts)"
        )
        return None
    device = args.device if args.backend == "warp" else "cpu"
    try:
        return PolicyDemoController(args.checkpoint, args.input_mode, args.task, device=device)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"cannot load checkpoint {args.checkpoint}: {exc}") from exc


# --- scenes ---------------------------------------------------------------------------------


def _make_env(
    task: str,
    config: TaskConfig | None,
    partner_mode: str,
    args: argparse.Namespace,
) -> CoopTaskEnv | GpuDemoEnv:
    """Build one demo env on the requested physics backend (`cpu` or `warp`)."""
    if args.backend == "warp":
        if config is None:
            from callosum.experiments.coop_tasks.config import coop_config

            config = coop_config(task)
        return GpuDemoEnv(
            task,
            config,
            partner_mode,
            seed=args.seed,
            device=args.device or "cuda",
        )
    from callosum.experiments.coop_tasks import make_coop_env

    return make_coop_env(task, config=config, partner_mode=partner_mode)


def _backend_metadata(args: argparse.Namespace, env: CoopTaskEnv | GpuDemoEnv) -> dict[str, Any]:
    """The JSON block recording which physics backend produced a scene's frames."""
    metadata: dict[str, Any] = {
        "frame_physics": frame_physics_label(args.backend),
        "physics_backend": args.backend,
    }
    if isinstance(env, GpuDemoEnv):
        metadata["device"] = env.device
        metadata["software"] = env.software_versions
        metadata["cuda_graph"] = env.cuda_graph
    return metadata


def _render_oracle_scene(
    args: argparse.Namespace,
    config: TaskConfig,
    controller: DemoController,
    renderer: CoopDemoRenderer | None,
    encoders: list[str],
) -> dict[str, Any]:
    """Render the scripted-oracle success clip; returns its summary block."""
    view = _task_view(args.task)
    env = _make_env(args.task, config, "none", args)
    metadata = _backend_metadata(args, env)
    try:
        max_steps = int(args.max_steps or config.max_control_steps)
        episode = run_episode(
            env,
            controller,
            renderer,
            task=args.task,
            config=config,
            seed=args.seed,
            step_stride=args.step_stride,
            max_steps=max_steps,
            title=f"{view.env_title} · {controller.title}",
            subtitle=controller.subtitle,
            name="oracle",
        )
    finally:
        env.close()
    print(f"oracle scene: {episode.status} after {episode.steps} steps -> {episode.summary()}")
    stem = f"coop_{args.task}_oracle"
    clips: dict[str, Any] = {}
    if renderer is not None:
        clips["oracle"] = _write_single_clip(
            args.out,
            f"{stem}.mp4",
            stem,
            episode,
            fps=args.fps,
            target_seconds=args.target_seconds,
            gif_width=args.gif_width or None,
            gif_stride=GIF_STRIDE,
            encoders=encoders,
        )
    return {
        **metadata,
        "task": args.task,
        "scene": "oracle",
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "step_stride": args.step_stride,
        "target_seconds": args.target_seconds,
        "encoder": sorted(set(encoders)),
        "episode": episode.summary(),
        "clips": clips,
    }


def _render_policy_vs_oracle_scene(
    args: argparse.Namespace,
    config: TaskConfig,
    oracle_controller: DemoController,
    policy_controller: PolicyDemoController,
    renderer: CoopDemoRenderer | None,
    encoders: list[str],
) -> dict[str, Any]:
    """Render the learned-policy-vs-oracle clip; returns its summary block."""
    view = _task_view(args.task)
    policy_env = _make_env(args.task, policy_controller.config, policy_controller.env_mode, args)
    oracle_env = _make_env(args.task, config, "none", args)
    metadata = _backend_metadata(args, oracle_env)
    try:
        oracle_episode = run_episode(
            oracle_env,
            oracle_controller,
            renderer,
            task=args.task,
            config=config,
            seed=args.seed,
            step_stride=args.step_stride,
            max_steps=int(args.max_steps or config.max_control_steps),
            title=f"{view.env_title} · oracle (upper bound)",
            subtitle="scripted centralised controller: privileged, not a learned policy",
            name="oracle-upper",
        )
        policy_max_steps = _policy_max_steps(args, int(config.max_control_steps))
        policy_episode = run_episode(
            policy_env,
            policy_controller,
            renderer,
            task=args.task,
            config=policy_controller.config,
            seed=args.seed,
            step_stride=args.step_stride,
            max_steps=policy_max_steps,
            title=f"{view.env_title} · {policy_controller.title}",
            subtitle=policy_controller.subtitle,
            name="policy",
        )
    finally:
        policy_env.close()
        oracle_env.close()
    print(
        f"oracle: {oracle_episode.status} after {oracle_episode.steps} steps | "
        f"policy ({policy_controller.input_mode}): {policy_episode.status} after "
        f"{policy_episode.steps} steps (cap {policy_max_steps})"
    )
    stem = f"coop_{args.task}_policy_vs_oracle"
    clips: dict[str, Any] = {}
    if renderer is not None:
        clip = _write_sbs_clip(
            args.out,
            f"{stem}.mp4",
            stem,
            policy_episode,
            oracle_episode,
            fps=args.fps,
            target_seconds=args.target_seconds,
            gif_width=args.gif_width or None,
            gif_stride=GIF_STRIDE,
            encoders=encoders,
        )
        clip["policy"] = policy_controller.metadata()
        clip["policy_max_steps"] = policy_max_steps
        clips["policy-vs-oracle"] = clip
    return {
        **metadata,
        "task": args.task,
        "scene": "policy-vs-oracle",
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "step_stride": args.step_stride,
        "target_seconds": args.target_seconds,
        "encoder": sorted(set(encoders)),
        "policy": policy_controller.metadata(),
        "policy_max_steps": policy_max_steps,
        "episodes": {"policy": policy_episode.summary(), "oracle": oracle_episode.summary()},
        "clips": clips,
    }


def _render_attempts_scene(
    args: argparse.Namespace,
    config: TaskConfig,
    controller: DemoController,
    renderer: CoopDemoRenderer | None,
    encoders: list[str],
) -> dict[str, Any]:
    """Render N episodes on consecutive seeds (successes and failures); returns the summary."""
    view = _task_view(args.task)
    is_policy = isinstance(controller, PolicyDemoController)
    # A learned policy must see the env mode its checkpoint was trained for (`oracle` gets
    # the privileged partner block; `predicted` stays on `none`).
    env_config = controller.config if is_policy else config
    partner_mode = controller.env_mode if is_policy else "none"
    env = _make_env(args.task, env_config, partner_mode, args)
    metadata = _backend_metadata(args, env)
    attempts: list[dict[str, Any]] = []
    clips: dict[str, Any] = {}
    stem = f"coop_{args.task}_{controller.label}"
    try:
        for index in range(args.attempts):
            seed = args.seed + index
            if is_policy:
                max_steps = _policy_max_steps(args, int(env_config.max_control_steps))
            else:
                max_steps = int(args.max_steps or env_config.max_control_steps)
            episode = run_episode(
                env,
                controller,
                renderer,
                task=args.task,
                config=env_config,
                seed=seed,
                step_stride=args.step_stride,
                max_steps=max_steps,
                title=f"{view.env_title} · attempt {index + 1}/{args.attempts} · seed {seed}",
                subtitle=controller.subtitle,
                name=f"attempt-{index}-seed-{seed}",
            )
            attempt_stem = f"{stem}_attempt{index}_seed{seed}"
            entry: dict[str, Any] = {
                "attempt": index,
                "seed": seed,
                "stem": attempt_stem,
                **episode.summary(),
            }
            print(f"attempt {index + 1}/{args.attempts} (seed {seed}): {episode.status} -> {entry}")
            if renderer is not None:
                clip = _write_single_clip(
                    args.out,
                    f"{attempt_stem}.mp4",
                    attempt_stem,
                    episode,
                    fps=args.fps,
                    target_seconds=args.target_seconds,
                    gif_width=None,
                    gif_stride=GIF_STRIDE,
                    encoders=encoders,
                    stills=False,
                )
                clips[attempt_stem] = clip
                entry["clip"] = {
                    "path": clip["path"],
                    "bytes": clip["bytes"],
                    "probe": clip.get("probe", {}),
                    "final_still": clip.get("final_still", {}),
                }
            attempts.append(entry)
    finally:
        env.close()
    successes = sum(1 for attempt in attempts if attempt["success"])
    return {
        **metadata,
        "task": args.task,
        "scene": "attempts",
        "source": controller.label,
        "policy": controller.metadata() if isinstance(controller, PolicyDemoController) else None,
        "seed": args.seed,
        "episodes": args.attempts,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "step_stride": args.step_stride,
        "target_seconds": args.target_seconds,
        "encoder": sorted(set(encoders)),
        "successes": successes,
        "failures": len(attempts) - successes,
        "success_rate": round(successes / len(attempts), 3) if attempts else 0.0,
        "attempts": attempts,
        "clips": clips,
    }


# --- main -----------------------------------------------------------------------------------


def _print_table(rows: Sequence[Sequence[str]], header: Sequence[str]) -> None:
    """Print a small fixed-width table."""
    widths = [max(len(str(row[i])) for row in (header, *rows)) for i in range(len(header))]
    print("  ".join(str(header[i]).ljust(widths[i]) for i in range(len(header))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(header))))


def _clip_rows(report: dict[str, Any]) -> list[tuple[str, str, str, str, str, str, str]]:
    """The per-clip table rows of a scene report (name, file, frames, seconds, size, MB, luma)."""
    rows = []
    for name, clip in report.get("clips", {}).items():
        probe = clip.get("probe", {})
        gif = clip.get("gif", {})
        frames = probe.get("frames") or gif.get("frames") or clip.get("video_frames", "?")
        seconds = probe.get("duration_s") or gif.get("duration_s") or 0.0
        rows.append(
            (
                name,
                Path(clip["path"]).name,
                str(frames),
                f"{float(seconds):.2f}",
                (
                    f"{probe.get('width', report.get('width'))}x"
                    f"{probe.get('height', report.get('height'))}"
                ),
                f"{clip['bytes'] / 1e6:.2f}",
                str(clip.get("mean_luma", "-")),
            )
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Render the requested coop demo artifacts; returns the process exit code."""
    args = parse_args(argv)
    _validate_args(args)
    if args.backend == "warp":
        require_gpu_stack()
    _configure_gl()
    args.out.mkdir(parents=True, exist_ok=True)

    from callosum.experiments.coop_tasks.config import coop_config

    config = coop_config(args.task)
    oracle_controller = OracleDemoController(args.task, config)
    policy_controller = _load_policy_controller(args)
    if args.scene == "attempts" and policy_controller is None:
        print(
            "note: --scene attempts renders scripted-oracle attempts (pass --checkpoint for policies)"
        )
    print(
        f"coop demo render: scene={args.scene} task={args.task} seed={args.seed} "
        f"size={args.width}x{args.height} fps={args.fps} step_stride={args.step_stride} "
        f"target_seconds={args.target_seconds} dry_run={args.dry_run} "
        f"frame_physics={frame_physics_label(args.backend)} backend={args.backend} "
        f"device={args.device} MUJOCO_GL={os.environ.get('MUJOCO_GL')}"
    )
    if policy_controller is not None:
        print(
            f"checkpoint {policy_controller.checkpoint}: input_mode="
            f"{policy_controller.input_mode} env_partner_mode={policy_controller.env_mode} "
            f"latent_dim={policy_controller.latent_dim} update={policy_controller.update} "
            f"policy_device={policy_controller.metadata()['policy_device']} "
            "(greedy policies)"
        )

    encoders: list[str] = []
    renderer: CoopDemoRenderer | None = None
    try:
        if not args.dry_run:
            renderer = CoopDemoRenderer(args.task, config, args.width, args.height)
        if args.scene == "oracle":
            report = _render_oracle_scene(args, config, oracle_controller, renderer, encoders)
        elif args.scene == "policy-vs-oracle":
            assert policy_controller is not None  # _validate_args guarantees it
            report = _render_policy_vs_oracle_scene(
                args, config, oracle_controller, policy_controller, renderer, encoders
            )
        else:
            controller = policy_controller or oracle_controller
            report = _render_attempts_scene(args, config, controller, renderer, encoders)
    finally:
        if renderer is not None:
            renderer.close()

    if not args.dry_run:
        tag = "policy_vs_oracle" if args.scene == "policy-vs-oracle" else args.scene
        if args.scene == "attempts":
            source = policy_controller.label if policy_controller is not None else "oracle"
            tag = f"{source}_attempts"
        summary_path = args.out / f"coop_{args.task}_{tag}.json"
        report["summary_json"] = str(summary_path.resolve())
        summary_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        rows = _clip_rows(report)
        if rows:
            print()
            _print_table(rows, ("clip", "file", "frames", "seconds", "size", "MB", "luma"))
        print(f"\nartifacts in: {args.out.resolve()}")
        print(f"summary json: {summary_path.resolve()}")
        print(f"encoder(s): {sorted(set(encoders))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
