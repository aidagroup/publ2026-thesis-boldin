"""Render the two-arm SO-ARM101 FaceTurn-v0 demo clip on ManiSkill / SAPIEN.

What the clip shows
-------------------
`FaceTurn-v0` from this repository driven by the scripted two-arm expert
(`callosum.experts.face_turn_expert`): the holder SO-ARM101 clamps the cube body, then the
rotator SO-ARM101 clamps the turnable top face and rolls its wrist 90 degrees, and the
holder keeps the body still. The env reports success at about step 330 and the expert opens
the jaws at about step 350. This is the "real robotics" companion clip of the thesis demo:
the reduced MuJoCo model used for the main experiments has no robot arms at all.

Why a separate script
---------------------
The env is built for state observations and defines no cameras (`_default_human_render_
camera_configs` returns `[]`), so nothing in the pipeline can render it. This script
subclasses it (`FaceTurnDemo`, registered in-process as `FaceTurnDemo-v0`) and overrides
only the human-render camera configuration; the env, the robot model, the expert and the
physics are untouched.

Render-only markers
-------------------
The turntable cube is symmetric under a 90 degree turn (red body, yellow top face), so the
turn is invisible in a raw render. To make it visible this script adds two kinematic
visual-only boxes: a cyan bar on the face (it turns with the face) and a black bar on the
body side (it stays put, a reference). They have no collision shapes, are never touched by
physics and are not part of the env package; `--no-markers` turns them off. The HUD prints
the measured face angle and body drift in any case.

Rendering on the RunPod pod
---------------------------
The pod has no hardware Vulkan (the container's NVIDIA driver is mounted with
`NVIDIA_DRIVER_CAPABILITIES=compute,utility`, so no NVIDIA Vulkan ICD), but Mesa's lavapipe
software Vulkan works and SAPIEN 3.0.3 can render on it: `render_backend="cpu"` builds
`sapien.render.RenderSystem(sapien.Device("cpu"))`, which is the CPU render path. The sim
runs on `sim_backend="cpu"` (one env, PhysX CPU), the fastest and most reliable combination
for lavapipe. Measured on the A40 pod: about 0.5 s per 640x480 frame (the first frame pays
a one-off shader compilation of about 6 s).

If `VK_ICD_FILENAMES` is unset and the Mesa lavapipe ICD is installed, the script points
SAPIEN at lavapipe itself (SAPIEN probes the ICDs at import); export `VK_ICD_FILENAMES` to
use a hardware Vulkan device instead, and pass `--render-backend gpu` then.

Usage (pod, repo root; the sim venv is `/root/callosum-arms-venv`, built with
`uv sync --frozen --extra sim --extra train --extra dev`):
```
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  /root/callosum-arms-venv/bin/python scripts/render_face_turn_arms.py \
  --out /root/arms_demo_out --seed 0
```
Quick framing check (stills only, stop before the release):
```
... render_face_turn_arms.py --out /tmp/framing --stills-only --max-steps 250 --every 60
```
Outputs (`--out`, default `demo/`): `demo_arms_face_turn.mp4` (H.264 via ffmpeg,
640x480, 15 fps), `demo_arms_preview.gif` (320 px wide, every second frame, looping), and
four stills: `demo_arms_frame0.png`
(reset), `demo_arms_grasp.png` (both arms clamped), `demo_arms_turn.png` (mid-turn),
`demo_arms_success.png` (final state). Every frame carries the title, the phase, the face
angle, the body drift and a 0-90 degree progress bar; the final frames carry SUCCESS.

Exit status: 0 if the expert reached env success, 1 otherwise (the clip is still written as
a best effort and its final frames say INCOMPLETE).
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from callosum.common.render_utils import (
    FAILURE_COLOR,
    SUCCESS_COLOR,
    annotate_frame,
    probe_video,
    write_gif,
    write_mp4,
)

LAVAPIPE_ICD = Path("/usr/share/vulkan/icd.d/lvp_icd.json")
"""Mesa software Vulkan ICD the pod installs with `apt-get install mesa-vulkan-drivers`."""

DEMO_ENV_ID = "FaceTurnDemo-v0"
"""In-process env id of `FaceTurn-v0` + the demo camera (registered by `main`)."""

EPISODE_STEPS = 400
"""`FaceTurn-v0`'s episode limit; the expert finishes (release) at about step 350."""

TITLE = "two SO-ARM101 arms hold and turn a cube"
SUBTITLE = (
    "FaceTurn-v0 - scripted two-arm expert, real SAPIEN physics; "
    "cyan bar on the face (render-only) shows the 90 deg turn"
)
TARGET_FACE_ANGLE_DEG = 90.0
TURN_START_DEG = 3.0
"""Face angle above which the HUD calls the current phase "turn"."""
GRASP_STILL_MAX_DEG = 2.0
"""A frame counts as the "grasp" still while the face is still at (about) 0 degrees."""
TURN_STILL_MIN_DEG = 35.0
TURN_STILL_MAX_DEG = 60.0
"""Window of face angles used to pick the "turn" still."""

DEFAULT_EYE = (0.46, -0.46, 0.36)
DEFAULT_TARGET = (0.0, 0.0, 0.0)
"""Camera pose (m): a three-quarter view from the holder/rotator side of the table that
frames both arm bases and the cube at 640x480 with a 0.85 rad fov."""
DEFAULT_FOV = 0.85

CUBE_HALF_SIZE = 0.0285
"""Half edge length of the turntable cube (`callosum.envs._turntable_cube`)."""
FACE_THICKNESS = 2 * CUBE_HALF_SIZE / 3
"""Thickness of the turnable top face."""
FACE_MARKER_LOCAL = ([0.013, 0.0, FACE_THICKNESS / 2 + 0.0015], [0.0, 0.0, 0.0, 1.0])
"""Pose of the cyan face marker in the face link frame (on top of the face, near one edge)."""
BODY_MARKER_LOCAL = ([CUBE_HALF_SIZE + 0.0015, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
"""Pose of the black body marker in the body link frame (a strip on the +x side face)."""

MP4_NAME = "demo_arms_face_turn.mp4"
GIF_NAME = "demo_arms_preview.gif"
STILL_NAMES = (
    "demo_arms_frame0.png",
    "demo_arms_grasp.png",
    "demo_arms_turn.png",
    "demo_arms_success.png",
)
"""Stills written next to the clips: reset, both arms clamped, mid-turn, final state."""


def _prefer_software_vulkan() -> None:
    """Point SAPIEN at the Mesa lavapipe ICD when the caller has not picked one.

    SAPIEN probes the Vulkan ICDs when `sapien` is imported, so this must run before the sim
    imports (which `_run_demo` does lazily for that reason). On a machine with hardware
    Vulkan, export `VK_ICD_FILENAMES` to bypass this; `--render-backend gpu` then applies.
    """
    if "VK_ICD_FILENAMES" in os.environ or not LAVAPIPE_ICD.exists():
        return
    os.environ["VK_ICD_FILENAMES"] = str(LAVAPIPE_ICD)
    print(f"note: VK_ICD_FILENAMES was unset; using the software Vulkan ICD {LAVAPIPE_ICD}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the script's command line (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("demo"), help="Output directory.")
    parser.add_argument("--seed", type=int, default=0, help="Env reset seed.")
    parser.add_argument(
        "--control",
        choices=("pos", "delta"),
        default="pos",
        help='"pos": pd_joint_pos, lockstep (default, verified to success); '
        '"delta": the training controller pd_joint_delta_pos (use --overlap with it).',
    )
    parser.add_argument(
        "--overlap",
        action="store_true",
        help="ExpertControlConfig.overlap_approach: overlap the two approach phases.",
    )
    parser.add_argument(
        "--no-holder", action="store_true", help="Skip the holder: the rotator turns alone."
    )
    parser.add_argument(
        "--release-holder",
        action="store_true",
        help="The holder opens its jaws half way through the turn.",
    )
    parser.add_argument("--no-lock", action="store_true", help="Disable the face lock.")
    parser.add_argument(
        "--face-friction",
        type=float,
        default=None,
        help="Face joint friction (default: FaceTurnPhysicsConfig value).",
    )
    parser.add_argument(
        "--face-damping",
        type=float,
        default=None,
        help="Face joint damping (default: FaceTurnPhysicsConfig value).",
    )
    parser.add_argument("--width", type=int, default=640, help="Frame width (even, px).")
    parser.add_argument("--height", type=int, default=480, help="Frame height (even, px).")
    parser.add_argument("--fov", type=float, default=DEFAULT_FOV, help="Vertical camera fov (rad).")
    parser.add_argument(
        "--eye", type=float, nargs=3, default=list(DEFAULT_EYE), metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--target", type=float, nargs=3, default=list(DEFAULT_TARGET), metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--every",
        type=int,
        default=2,
        help="Render every N-th control step (default 2: ~176 frames for the ~350-step run).",
    )
    parser.add_argument("--fps", type=float, default=15.0, help="Output video fps.")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.5,
        help="Hold the final frame this long at the end of the clip (s).",
    )
    parser.add_argument("--gif-width", type=int, default=320, help="GIF preview width (px).")
    parser.add_argument(
        "--gif-every",
        type=int,
        default=2,
        help="Keep every N-th rendered frame in the GIF (smaller file, default 2).",
    )
    parser.add_argument(
        "--no-markers", action="store_true", help="Do not add the render-only orientation markers."
    )
    parser.add_argument(
        "--stills-only",
        action="store_true",
        help="Run the expert and save the stills only (no MP4/GIF); for framing checks.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Stop the expert after this many env steps."
    )
    parser.add_argument(
        "--sim-backend", default="cpu", choices=("cpu", "gpu"), help="PhysX backend."
    )
    parser.add_argument(
        "--render-backend",
        default="cpu",
        help='ManiSkill render backend: "cpu" (sapien_cpu, works on lavapipe) or "gpu".',
    )
    args = parser.parse_args(argv)
    if args.width % 2 or args.height % 2:
        parser.error(f"frame size must be even for the encoders, got {args.width}x{args.height}")
    if args.every < 1:
        parser.error("--every must be >= 1")
    if args.gif_every < 1:
        parser.error("--gif-every must be >= 1")
    if args.fps <= 0:
        parser.error("--fps must be > 0")
    if args.fov <= 0:
        parser.error("--fov must be > 0")
    return args


def _phase_label(holder_grasp: bool, rotator_grasp: bool, angle_deg: float) -> str:
    """HUD phase caption derived from the sim state (so it cannot lag behind the frames)."""
    if rotator_grasp and angle_deg > TURN_START_DEG:
        return "rotator: turn 90 deg"
    if rotator_grasp:
        return "rotator: grasp face - clamped"
    if holder_grasp:
        return "holder: grasp body - clamped"
    return "approach"


def _to_frame(image) -> np.ndarray:
    """Turn one `env.render()` result (torch/numpy, `(num_envs,) H W C`) into RGB uint8."""
    array = image.cpu().numpy() if hasattr(image, "cpu") else np.asarray(image)
    array = np.asarray(array)
    if array.ndim == 4:
        array = array[0]
    return np.ascontiguousarray(array[..., :3].astype(np.uint8))


def _write_h264_mp4(path: Path, frames: list[np.ndarray], fps: float) -> str:
    """Encode the frames as an H.264/yuv420p MP4 with ffmpeg; returns the encoder used.

    `_render_utils.write_mp4` prefers OpenCV's `mp4v` when OpenCV is installed (it is in the
    pod's sim venv), but `mp4v` plays poorly in browsers and QuickTime, and the clip is meant
    to be embedded. H.264 is what the other demo clips in `demo/` use. Nothing is installed:
    the pod and the Mac both have an `ffmpeg` binary; without one this falls back to
    `write_mp4`.
    """
    binary = shutil.which("ffmpeg")
    if binary is None or not frames:
        return write_mp4(path, frames, fps)
    height, width = frames[0].shape[:2]
    command = [
        binary,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{int(width)}x{int(height)}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:  # pragma: no cover - Popen with PIPE always provides stdin
        raise RuntimeError("ffmpeg stdin is not available")
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {stderr.strip()}")
    return "ffmpeg"


def _decorate(
    frame: np.ndarray,
    metrics: dict,
    total_steps: int,
    *,
    status: str | None = None,
    status_color=SUCCESS_COLOR,
    status_detail: str | None = None,
) -> np.ndarray:
    """Draw the demo overlay (title, phase, measured numbers, progress bar, status)."""
    angle = max(0.0, float(metrics["angle_deg"]))  # no "-0.0 deg" from the joint noise
    hud = [
        f"step {metrics['step']:3d} / {total_steps}",
        f"phase: {_phase_label(metrics['holder'], metrics['rotator'], angle)}",
        f"face angle: {angle:5.1f} deg / 90 deg",
        f"body drift: {metrics['drift_cm']:4.2f} cm",
        "holder clamps body: "
        + ("yes" if metrics["holder"] else "no")
        + " | rotator clamps face: "
        + ("yes" if metrics["rotator"] else "no"),
    ]
    return annotate_frame(
        frame,
        title=TITLE,
        subtitle=SUBTITLE,
        hud_lines=hud,
        angle_fraction=angle / TARGET_FACE_ANGLE_DEG,
        status=status,
        status_color=status_color,
        status_detail=status_detail,
    )


def _run_demo(args: argparse.Namespace) -> dict:
    """Build the demo env, run the scripted expert, render the frames and write the outputs.

    Returns a summary dict (also printed) with the success flag, the final metrics, the
    frame counts and the written file names.
    """
    # The sim imports are lazy: `_prefer_software_vulkan` must set the ICD before SAPIEN loads.
    import gymnasium as gym
    import sapien
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose as MSPose

    import callosum.envs.face_turn  # noqa: F401  (registers FaceTurn-v0)
    from callosum.configs.face_turn import FaceTurnPhysicsConfig
    from callosum.configs.face_turn_expert import ExpertControlConfig
    from callosum.envs.face_turn import FaceTurn
    from callosum.experts.face_turn_expert import ArmModel, Rig, run_expert

    camera_config = CameraConfig(
        uid="demo_camera",
        pose=sapien_utils.look_at(args.eye, args.target),
        width=args.width,
        height=args.height,
        fov=args.fov,
        near=0.01,
        far=10.0,
        shader_pack="minimal",
    )
    face_marker_local = sapien.Pose(FACE_MARKER_LOCAL[0], FACE_MARKER_LOCAL[1])
    body_marker_local = sapien.Pose(BODY_MARKER_LOCAL[0], BODY_MARKER_LOCAL[1])

    def add_marker(scene, name: str, half_size, color) -> object:
        """A kinematic, collision-free box: a visual reference that physics never sees."""
        builder = scene.create_actor_builder()
        # Overwritten by `place_markers()` before the first frame; only silences ManiSkill's
        # "No initial pose set for actor builder" warning.
        builder.initial_pose = sapien.Pose([0.0, 0.0, 0.05])
        builder.add_box_visual(
            half_size=half_size,
            material=sapien.render.RenderMaterial(base_color=list(color)),
        )
        return builder.build_kinematic(name=name)

    class FaceTurnDemo(FaceTurn):
        """FaceTurn-v0 with one offscreen camera and (optionally) the render-only markers."""

        @property
        def _default_human_render_camera_configs(self):
            return [camera_config]

        def _load_scene(self, options: dict):
            super()._load_scene(options)
            if args.no_markers:
                return
            self.face_marker = add_marker(
                self.scene, "face_marker", (0.013, 0.0022, 0.0012), (0.05, 0.85, 0.95, 1.0)
            )
            self.body_marker = add_marker(
                self.scene, "body_marker", (0.0012, 0.010, 0.0025), (0.05, 0.05, 0.05, 1.0)
            )

    register_env(DEMO_ENV_ID, max_episode_steps=EPISODE_STEPS)(FaceTurnDemo)

    physics_defaults = FaceTurnPhysicsConfig()
    physics = FaceTurnPhysicsConfig(
        lock_face_unless_held=not args.no_lock,
        face_friction=(
            args.face_friction if args.face_friction is not None else physics_defaults.face_friction
        ),
        face_damping=(
            args.face_damping if args.face_damping is not None else physics_defaults.face_damping
        ),
    )
    control_mode = "pd_joint_pos" if args.control == "pos" else "pd_joint_delta_pos"
    env = gym.make(
        DEMO_ENV_ID,
        num_envs=1,
        obs_mode="state",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        render_mode="rgb_array",
        control_mode=control_mode,
        physics_config=physics,
    )
    obs, _ = env.reset(seed=args.seed)
    base = env.unwrapped
    print(
        f"env: {DEMO_ENV_ID} sim_backend={base.backend.sim_backend} "
        f"render_backend={base.backend.render_backend} camera {args.width}x{args.height} "
        f"fov={args.fov:.2f} sim_freq={base.sim_freq} control_freq={base.control_freq}"
    )
    print("SAPIEN render devices:", " | ".join(sapien.render.get_device_summary().splitlines()))

    markers = []
    if not args.no_markers:
        markers = [
            (base.face_marker, base.cube.links_map["face"], face_marker_local),
            (base.body_marker, base.cube.links_map["body"], body_marker_local),
        ]

    def place_markers() -> None:
        """Put the render-only markers on the face/body links of the current state."""
        for actor, link, local in markers:
            actor.set_pose(link.pose * MSPose.create(local))

    rig = Rig(
        env,
        ArmModel(),
        ExpertControlConfig(control=args.control, overlap_approach=args.overlap),
        obs=obs,
        max_steps=args.max_steps,
    )

    frames: list[np.ndarray] = []
    metrics: list[dict] = []
    render_seconds = 0.0

    def capture(step: int) -> None:
        """Render the current state and store the frame plus the numbers shown in its HUD."""
        nonlocal render_seconds
        started = time.time()
        place_markers()
        frame = _to_frame(env.render())
        render_seconds += time.time() - started
        info = base.evaluate()
        metrics.append(
            {
                "step": int(step),
                "angle_deg": math.degrees(float(info["face_angle"].reshape(-1)[0])),
                "drift_cm": float(info["body_pos_drift"].reshape(-1)[0]) * 100.0,
                "holder": bool(info["holder_grasp"].reshape(-1)[0]),
                "rotator": bool(info["rotator_grasp"].reshape(-1)[0]),
                "success": bool(info["success"].reshape(-1)[0]),
            }
        )
        frames.append(frame)

    capture(0)  # reset state, before the first expert step

    def on_step(obs_before, action, reward, terminated, truncated, info) -> None:
        if rig.steps % args.every == 0:
            capture(rig.steps)

    rig.callbacks.append(on_step)
    print("running the scripted expert (holder grasp -> rotator grasp -> turn -> release)")
    run_expert(
        rig,
        no_holder=args.no_holder,
        release_holder=args.release_holder,
        on_phase=lambda name: print(f"  phase: {name} (env step {rig.steps})"),
    )
    if metrics[-1]["step"] != rig.steps:
        capture(rig.steps)  # final state (after the release) even if off the --every grid

    final = metrics[-1]
    success = any(m["success"] for m in metrics)
    hold_frames = max(1, round(args.fps * args.hold_seconds))
    decorated = []
    for frame, meta in zip(frames, metrics, strict=True):
        detail = f"face {meta['angle_deg']:.0f} deg of 90, body drift {meta['drift_cm']:.2f} cm"
        if meta["success"]:
            decorated.append(
                _decorate(
                    frame,
                    meta,
                    rig.steps,
                    status="SUCCESS",
                    status_color=SUCCESS_COLOR,
                    status_detail=detail,
                )
            )
        else:
            decorated.append(_decorate(frame, meta, rig.steps))
    if success:
        final_frame = decorated[-1]
    else:
        final_frame = _decorate(
            frames[-1],
            final,
            rig.steps,
            status="INCOMPLETE",
            status_color=FAILURE_COLOR,
            status_detail=f"face {final['angle_deg']:.0f} deg of 90, "
            f"body drift {final['drift_cm']:.2f} cm",
        )
        print("warning: the expert did not reach env success; writing a best-effort clip")
        decorated[-1] = final_frame
    decorated.extend([final_frame] * hold_frames)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stills = _write_stills(out, frames, metrics, decorated)
    summary = {
        "success": success,
        "control": args.control,
        "overlap": args.overlap,
        "seed": args.seed,
        "env_steps": rig.steps,
        "first_success": rig.first_success,
        "final": final,
        "frames": len(decorated),
        "rendered_frames": len(frames),
        "fps": args.fps,
        "every": args.every,
        "hold_frames": hold_frames,
        "duration_s": len(decorated) / args.fps,
        "render_seconds": render_seconds,
        "mp4": None,
        "gif": None,
        "stills": stills,
    }
    if not args.stills_only:
        mp4_path = out / MP4_NAME
        gif_path = out / GIF_NAME
        encoder = _write_h264_mp4(mp4_path, decorated, args.fps)
        gif_count = write_gif(
            gif_path,
            decorated[:: args.gif_every],
            frame_duration_ms=max(1, round(1000.0 * args.gif_every / args.fps)),
            width=args.gif_width,
        )
        summary["mp4"] = {"path": str(mp4_path), "encoder": encoder, **probe_video(mp4_path)}
        summary["gif"] = {"path": str(gif_path), "frames": gif_count, "width": args.gif_width}
    env.close()

    del summary["final"]["step"]
    print()
    print("=== demo summary ===")
    print(f"control mode        : {control_mode} (expert {args.control}, overlap={args.overlap})")
    print(f"seed                : {args.seed}")
    print(f"env steps           : {rig.steps} (episode limit {EPISODE_STEPS})")
    if rig.first_success is not None:
        step, reward, before = rig.first_success
        print(
            f"first success       : env step {step} (step reward {reward:.2f}, before {before:.2f})"
        )
    print(f"success             : {success}")
    print(f"final face angle    : {final['angle_deg']:.1f} deg (target 90)")
    print(f"final body drift    : {final['drift_cm']:.2f} cm")
    print(
        f"frames              : {len(decorated)} "
        f"({len(frames)} rendered every {args.every} env steps + {hold_frames} hold)"
    )
    print(f"video               : {args.fps:g} fps, {len(decorated) / args.fps:.1f} s")
    if len(frames):
        print(
            f"render time         : {render_seconds:.1f} s for {len(frames)} frames "
            f"({render_seconds / len(frames):.2f} s/frame)"
        )
    if summary["mp4"] is not None:
        print(f"mp4                 : {summary['mp4']}")
    if summary["gif"] is not None:
        print(f"gif                 : {summary['gif']}")
    print(f"stills              : {', '.join(summary['stills'])}")
    return summary


def _write_stills(
    out: Path, frames: list[np.ndarray], metrics: list[dict], decorated: list[np.ndarray]
) -> list[str]:
    """Save the four stills: reset, both grasps clamped, mid-turn, final state."""
    grasp = next(
        (
            index
            for index, meta in enumerate(metrics)
            if meta["holder"] and meta["rotator"] and meta["angle_deg"] <= GRASP_STILL_MAX_DEG
        ),
        None,
    )
    turn = next(
        (
            index
            for index, meta in enumerate(metrics)
            if TURN_STILL_MIN_DEG <= meta["angle_deg"] <= TURN_STILL_MAX_DEG
        ),
        None,
    )
    picks = {
        STILL_NAMES[0]: decorated[0],
        STILL_NAMES[1]: decorated[grasp if grasp is not None else len(decorated) - 1],
        STILL_NAMES[2]: decorated[turn if turn is not None else len(decorated) - 1],
        STILL_NAMES[3]: decorated[-1],
    }
    if grasp is None or turn is None:
        print(
            "warning: could not find a "
            + ", ".join(name for name, index in (("grasp", grasp), ("turn", turn)) if index is None)
            + " frame for the stills; the final frame is used instead"
        )
    written = []
    for name, frame in picks.items():
        path = out / name
        Image.fromarray(frame).save(path)
        written.append(name)
    return written


def main(argv: list[str] | None = None) -> int:
    """Parse the CLI, pick a Vulkan ICD, render the demo; return 0 on env success."""
    args = parse_args(argv)
    _prefer_software_vulkan()
    summary = _run_demo(args)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
