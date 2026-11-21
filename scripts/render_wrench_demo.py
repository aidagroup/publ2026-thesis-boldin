"""Render the CubeWrench-v0 demo videos: oracle success, rotator-only failure, side by side.

Three clips come out of one run (all from the env's own physics, rendered with
`mujoco.Renderer` on a render-only copy of the scene, see `scripts/_render_utils.py`):

1. ``demo_oracle.mp4`` -- the scripted oracle: the holder PD-stabilises the core while the
   rotator PD-turns the layer to 90 degrees -> SUCCESS (~50 control steps at the defaults).
2. ``demo_rotator_only.mp4`` -- the same oracle with the holder wrench forced to zero: the
   rotator's reaction torque spins the core past the 0.5 rad drift bound -> FAILED after
   ~20 control steps (the non-additivity the reduced model exists to show).
3. ``demo_side_by_side.mp4`` -- clips 1 and 2 in one frame, synchronised by control step,
   each half keeping its own caption and HUD.

Plus ``demo_preview.gif`` (looping preview of the side-by-side clip), the first three frames
of every video as PNG stills (``oracle_frame0.png`` ...; they are identical by construction,
because the slow-motion expansion holds frame 0), a middle frame (``*_mid.png``) and the
final frame (``*_final.png``) of every clip, and ``demo_summary.json`` with the episode
metrics and the video probe results.

Nothing is installed: MP4 is written by OpenCV when available and by the system ``ffmpeg``
otherwise (the pod's venv has no OpenCV), GIF by Pillow. Rendering is offscreen: the script
selects ``MUJOCO_GL=egl`` on Linux and ``cgl`` on macOS, unless ``MUJOCO_GL`` is already set.

Run on the pod (from the repo root):

    MUJOCO_GL=egl /root/.callosum/venv/bin/python scripts/render_wrench_demo.py \
        --out /workspace/callosum-runs/render --seed 0 --width 640 --height 640 \
        --fps 30 --controller oracle

Quick metric-only check (no rendering, no OpenGL):

    python scripts/render_wrench_demo.py --out /tmp/render --dry-run

Single scene / holder-off variant:

    python scripts/render_wrench_demo.py --out /tmp/render --scene oracle --holder-off
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from _render_utils import (
    FAILURE_COLOR,
    MUTED_COLOR,
    SUCCESS_COLOR,
    DemoRenderer,
    annotate_frame,
    degrees,
    hstack_side_by_side,
    mean_luma,
    probe_video,
    save_png_stills,
    write_gif,
    write_mp4,
)
from PIL import Image

if TYPE_CHECKING:
    from callosum.envs.wrench_cube import OracleController, WrenchCubeEnv

TITLE_ORACLE = "hold + turn (both agents)"
"""Caption of scene 1 (the oracle's coordinated two-agent turn)."""
SUBTITLE_ORACLE = "oracle: holder stabilises the core, rotator turns the layer to 90°"
TITLE_ROTATOR = "rotator only (no partner)"
"""Caption of scene 2 (the holder's wrench forced to zero)."""
SUBTITLE_ROTATOR = "holder wrench = 0: the reaction torque spins the core out of tolerance"

ORACLE_VIDEO = "demo_oracle.mp4"
ROTATOR_VIDEO = "demo_rotator_only.mp4"
SIDE_BY_SIDE_VIDEO = "demo_side_by_side.mp4"
PREVIEW_GIF = "demo_preview.gif"
SUMMARY_JSON = "demo_summary.json"
PNG_STILLS = 3
"""Number of leading frames saved as ``*_frame{i}.png`` per video."""

FAILURE_DRIFT_RAD = 0.5
"""Core-rotation failure bound of the env config (`failure_rot_drift`, used in labels)."""


@dataclass
class DemoFrame:
    """One decorated frame plus the numbers shown in its HUD."""

    image: np.ndarray
    step: int
    time_s: float
    angle_deg: float
    drift_mm: float
    drift_deg: float
    is_final: bool = False


@dataclass
class DemoEpisode:
    """One rendered rollout: its frames and the outcome (success / failure / timeout)."""

    name: str
    title: str
    frames: list[DemoFrame]
    steps: int
    success: bool
    terminated: bool
    truncated: bool
    final_angle_deg: float
    final_drift_mm: float
    final_drift_deg: float
    status: str
    status_detail: str

    def summary(self) -> dict[str, Any]:
        """The JSON-serialisable metrics of the episode."""
        return {
            "steps": self.steps,
            "success": self.success,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "final_angle_deg": round(self.final_angle_deg, 2),
            "final_drift_mm": round(self.final_drift_mm, 2),
            "final_drift_deg": round(self.final_drift_deg, 2),
            "status": self.status,
            "status_detail": self.status_detail,
            "unique_frames": len(self.frames),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the demo-render CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Render the CubeWrench-v0 demo videos (oracle success, rotator-only failure, "
            "side by side) with MuJoCo offscreen rendering."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory (created)")
    parser.add_argument(
        "--seed",
        "--episode-seed",
        dest="seed",
        type=int,
        default=0,
        help="episode seed of both scenes",
    )
    parser.add_argument("--width", type=int, default=640, help="frame width (px)")
    parser.add_argument("--height", type=int, default=640, help="frame height (px)")
    parser.add_argument("--fps", type=int, default=30, help="video frame rate")
    parser.add_argument(
        "--controller",
        choices=("oracle",),
        default="oracle",
        help="action source (only the scripted oracle is implemented)",
    )
    parser.add_argument(
        "--holder-off",
        action="store_true",
        help=(
            "force the holder wrench to zero in every rendered scene: with --scene oracle "
            "this renders the rotator-only failure as the main clip"
        ),
    )
    parser.add_argument(
        "--scene",
        choices=("all", "oracle", "rotator-only", "side-by-side"),
        default="all",
        help="which artifact(s) to render",
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
        default=5.0,
        help="approximate clip length: frames are held (slow motion) until it is reached",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="cap the rollout length (default: the env's max_control_steps)",
    )
    parser.add_argument(
        "--gif-width",
        type=int,
        default=640,
        help="width of the GIF preview (the side-by-side clip is downscaled to it)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the rollouts without rendering/encoding and print their metrics",
    )
    return parser.parse_args(argv)


def _configure_gl() -> None:
    """Select an offscreen OpenGL backend before `mujoco` is imported."""
    if os.environ.get("MUJOCO_GL"):
        return
    os.environ["MUJOCO_GL"] = "cgl" if sys.platform == "darwin" else "egl"


def _rotation_angle(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
    """Rotation angle (rad) between two `(w, x, y, z)` quaternions."""
    dot = float(np.clip(abs(np.dot(quat_a, quat_b)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def _state_metrics(state: dict[str, Any], initial: dict[str, Any]) -> tuple[float, float, float]:
    """Return `(layer_angle_deg, core_drift_mm, core_drift_deg)` of a state vs the initial one."""
    angle_deg = degrees(float(state["layer_angle"]))
    drift_mm = float(np.linalg.norm(np.asarray(state["core_pos"]) - initial["core_pos"])) * 1e3
    drift_deg = degrees(
        _rotation_angle(
            np.asarray(initial["core_quat"], dtype=np.float64),
            np.asarray(state["core_quat"], dtype=np.float64),
        )
    )
    return angle_deg, drift_mm, drift_deg


def run_episode(
    env: WrenchCubeEnv,
    controller: OracleController,
    renderer: DemoRenderer | None,
    *,
    name: str,
    title: str,
    subtitle: str,
    seed: int,
    step_stride: int,
    max_steps: int,
    holder_off: bool,
) -> DemoEpisode:
    """Run one rollout and return its rendered frames plus the outcome.

    Frames are captured at the start of every `step_stride`-th control step and once more
    after the terminating step (the final frame carries the SUCCESS / FAILED status).
    `renderer=None` runs the same rollout without rendering (the `--dry-run` path).
    """
    config = env.config
    env.reset(seed=seed)
    initial = env.state()
    goal = float(config.goal_angle)
    frames: list[DemoFrame] = []

    def capture(
        step: int,
        *,
        status: str | None = None,
        status_color: tuple[int, int, int, int] = MUTED_COLOR,
        detail: str = "",
    ) -> DemoFrame:
        state = env.state()
        angle_deg, drift_mm, drift_deg = _state_metrics(state, initial)
        if renderer is None:
            image = np.zeros((8, 8, 3), dtype=np.uint8)
        else:
            image = annotate_frame(
                renderer.render_state(state),
                title=title,
                subtitle=subtitle,
                hud_lines=(
                    f"t = {step * config.control_dt:.2f} s   ·   step {step}",
                    f"layer angle {angle_deg:.1f}°   →   goal {degrees(goal):.1f}°",
                    f"core drift {drift_mm:.1f} mm · {drift_deg:.1f}°",
                ),
                angle_fraction=angle_deg / degrees(goal),
                status=status,
                status_color=status_color,
                status_detail=detail or None,
            )
        frame = DemoFrame(
            image=image,
            step=step,
            time_s=step * config.control_dt,
            angle_deg=angle_deg,
            drift_mm=drift_mm,
            drift_deg=drift_deg,
            is_final=status is not None,
        )
        frames.append(frame)
        return frame

    capture(0)
    steps = 0
    terminated = truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        if steps and steps % step_stride == 0:
            capture(steps)
        action = controller.act(env)
        if holder_off:
            action = {"holder": np.zeros(6), "rotator": action["rotator"]}
        _obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
        if steps >= max_steps and not terminated:
            truncated = True
            break

    final_angle_deg, final_drift_mm, final_drift_deg = _state_metrics(env.state(), initial)
    if terminated and bool(info.get("success")):
        status = "SUCCESS"
        detail = f"{final_angle_deg:.1f} deg · core drift {final_drift_deg:.1f} deg"
        status_color = SUCCESS_COLOR
    elif terminated:
        status = "FAILED"
        detail = f"core spun {final_drift_deg:.1f} deg (bound {degrees(FAILURE_DRIFT_RAD):.1f} deg)"
        status_color = FAILURE_COLOR
    else:
        status = "TIMEOUT"
        detail = f"{steps} control steps used"
        status_color = MUTED_COLOR
    capture(steps, status=status, status_color=status_color, detail=detail)
    return DemoEpisode(
        name=name,
        title=title,
        frames=frames,
        steps=steps,
        success=bool(info.get("success")),
        terminated=terminated,
        truncated=truncated,
        final_angle_deg=final_angle_deg,
        final_drift_mm=final_drift_mm,
        final_drift_deg=final_drift_deg,
        status=status,
        status_detail=detail,
    )


def duplicate_for_target(episode: DemoEpisode, fps: int, target_seconds: float) -> int:
    """How many times each rendered frame is repeated to reach `target_seconds`."""
    if not episode.frames:
        return 1
    frames_needed = float(target_seconds) * float(fps) / len(episode.frames)
    return max(1, round(frames_needed))


def _repeat_frames(frames: Sequence[DemoFrame], duplicate: int) -> Iterator[np.ndarray]:
    """Yield each frame's image `duplicate` times (slow-motion expansion)."""
    for frame in frames:
        for _ in range(duplicate):
            yield frame.image


def _repeat_count(frames: Sequence[DemoFrame], duplicate: int) -> int:
    """Number of video frames after the slow-motion expansion."""
    return len(frames) * duplicate


def _expanded_index(frames: Sequence[DemoFrame], index: int, duplicate: int) -> np.ndarray:
    """The image at video-frame `index` of a clip expanded `duplicate` times."""
    return frames[min(index // duplicate, len(frames) - 1)].image


def side_by_side_frames(
    left: DemoEpisode,
    right: DemoEpisode,
    *,
    duplicate_left: int,
    duplicate_right: int,
) -> Iterator[np.ndarray]:
    """Yield clips `left` and `right` side by side, synchronised by control step.

    Both sides play at the same frames-per-control-step rate (`duplicate_left` must equal
    `duplicate_right` for the step sync to hold); the shorter clip holds its final frame
    while the longer one finishes.
    """
    total = max(
        _repeat_count(left.frames, duplicate_left), _repeat_count(right.frames, duplicate_right)
    )
    for index in range(total):
        yield hstack_side_by_side(
            _expanded_index(left.frames, index, duplicate_left),
            _expanded_index(right.frames, index, duplicate_right),
        )


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


def _print_table(rows: Sequence[Sequence[str]], header: Sequence[str]) -> None:
    """Print a small fixed-width table."""
    widths = [max(len(str(row[i])) for row in (header, *rows)) for i in range(len(header))]
    print("  ".join(str(header[i]).ljust(widths[i]) for i in range(len(header))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(header))))


def _scene_wanted(scene: str) -> tuple[bool, bool, bool]:
    """Return `(oracle, rotator-only, side-by-side)` flags of the `--scene` choice."""
    return (
        scene in ("all", "oracle", "side-by-side"),
        scene in ("all", "rotator-only", "side-by-side"),
        scene in ("all", "side-by-side"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Render the demo artifacts; returns the process exit code."""
    args = parse_args(argv)
    if args.width % 2 or args.height % 2:
        raise SystemExit(f"--width/--height must be even, got {args.width}x{args.height}")
    if args.step_stride < 1:
        raise SystemExit(f"--step-stride must be >= 1, got {args.step_stride}")
    if args.fps < 1:
        raise SystemExit(f"--fps must be >= 1, got {args.fps}")
    _configure_gl()
    args.out.mkdir(parents=True, exist_ok=True)

    from callosum.envs.wrench_cube import OracleController, WrenchCubeEnv

    env = WrenchCubeEnv(partner_mode="none")
    controller = OracleController(env.config)
    max_steps = int(args.max_steps or env.config.max_control_steps)
    want_oracle, want_rotator, want_side_by_side = _scene_wanted(args.scene)
    print(
        f"CubeWrench-v0 demo render: scene={args.scene} seed={args.seed} "
        f"size={args.width}x{args.height} fps={args.fps} step_stride={args.step_stride} "
        f"max_steps={max_steps} holder_off={args.holder_off} "
        f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}"
    )

    if args.dry_run:
        try:
            for name, wanted, holder_off in (
                ("oracle", want_oracle, args.holder_off),
                ("rotator-only", want_rotator, True),
            ):
                if not wanted:
                    continue
                episode = run_episode(
                    env,
                    controller,
                    None,
                    name=name,
                    title="",
                    subtitle="",
                    seed=args.seed,
                    step_stride=args.step_stride,
                    max_steps=max_steps,
                    holder_off=holder_off,
                )
                print(f"[dry-run] {name}: {json.dumps(episode.summary())}")
        finally:
            env.close()
        return 0

    renderer = DemoRenderer(args.width, args.height)
    encoders: list[str] = []
    clips: dict[str, dict[str, Any]] = {}
    oracle: DemoEpisode | None = None
    rotator: DemoEpisode | None = None
    try:
        if want_oracle:
            oracle = run_episode(
                env,
                controller,
                renderer,
                name="oracle",
                title=TITLE_ORACLE,
                subtitle=SUBTITLE_ORACLE,
                seed=args.seed,
                step_stride=args.step_stride,
                max_steps=max_steps,
                holder_off=args.holder_off,
            )
            print(
                f"scene 1 ({oracle.title}): {oracle.status} after {oracle.steps} steps, "
                f"angle {oracle.final_angle_deg:.1f} deg, drift {oracle.final_drift_deg:.1f} deg"
            )
        if want_rotator:
            rotator = run_episode(
                env,
                controller,
                renderer,
                name="rotator-only",
                title=TITLE_ROTATOR,
                subtitle=SUBTITLE_ROTATOR,
                seed=args.seed,
                step_stride=args.step_stride,
                max_steps=max_steps,
                holder_off=True,
            )
            print(
                f"scene 2 ({rotator.title}): {rotator.status} after {rotator.steps} steps, "
                f"angle {rotator.final_angle_deg:.1f} deg, drift {rotator.final_drift_deg:.1f} deg"
            )

        if oracle is not None:
            duplicate = duplicate_for_target(oracle, args.fps, args.target_seconds)
            stills = [
                _expanded_index(oracle.frames, index, duplicate) for index in range(PNG_STILLS)
            ]
            clip = _encode_clip(
                args.out / ORACLE_VIDEO,
                _repeat_frames(oracle.frames, duplicate),
                _repeat_count(oracle.frames, duplicate),
                stills,
                args.fps,
                encoders,
            )
            clip.update({"episode": oracle.summary(), "duplicate": duplicate})
            clips["oracle"] = clip
            save_png_stills(args.out, "oracle", stills, count=len(stills))
            Image.fromarray(oracle.frames[-1].image).save(args.out / "oracle_final.png")
            middle = len(oracle.frames) // 2
            Image.fromarray(oracle.frames[middle].image).save(args.out / "oracle_mid.png")
        if rotator is not None:
            duplicate_rotator = duplicate_for_target(rotator, args.fps, args.target_seconds)
            stills = [
                _expanded_index(rotator.frames, index, duplicate_rotator)
                for index in range(PNG_STILLS)
            ]
            clip = _encode_clip(
                args.out / ROTATOR_VIDEO,
                _repeat_frames(rotator.frames, duplicate_rotator),
                _repeat_count(rotator.frames, duplicate_rotator),
                stills,
                args.fps,
                encoders,
            )
            clip.update({"episode": rotator.summary(), "duplicate": duplicate_rotator})
            clips["rotator-only"] = clip
            save_png_stills(args.out, "rotator_only", stills, count=len(stills))
            Image.fromarray(rotator.frames[-1].image).save(args.out / "rotator_only_final.png")
            middle = len(rotator.frames) // 2
            Image.fromarray(rotator.frames[middle].image).save(args.out / "rotator_only_mid.png")

        if want_side_by_side and oracle is not None and rotator is not None:
            duplicate_left = duplicate_for_target(oracle, args.fps, args.target_seconds)
            duplicate_right = duplicate_left  # same frames per control step => step sync
            total = max(
                _repeat_count(oracle.frames, duplicate_left),
                _repeat_count(rotator.frames, duplicate_right),
            )
            stills = []
            for index in range(PNG_STILLS):
                stills.append(
                    hstack_side_by_side(
                        _expanded_index(oracle.frames, index, duplicate_left),
                        _expanded_index(rotator.frames, index, duplicate_right),
                    )
                )
            clip = _encode_clip(
                args.out / SIDE_BY_SIDE_VIDEO,
                side_by_side_frames(
                    oracle,
                    rotator,
                    duplicate_left=duplicate_left,
                    duplicate_right=duplicate_right,
                ),
                total,
                stills,
                args.fps,
                encoders,
            )
            clip.update({"duplicate_left": duplicate_left, "duplicate_right": duplicate_right})
            clips["side-by-side"] = clip
            save_png_stills(args.out, "side_by_side", stills, count=len(stills))
            Image.fromarray(stills[-1]).save(args.out / "side_by_side_final.png")
            middle = len(oracle.frames) // 2
            Image.fromarray(
                hstack_side_by_side(
                    oracle.frames[middle].image,
                    rotator.frames[min(middle, len(rotator.frames) - 1)].image,
                )
            ).save(args.out / "side_by_side_mid.png")

            gif_stride = 2
            gif_path = args.out / PREVIEW_GIF
            gif_frames = write_gif(
                gif_path,
                (
                    frame
                    for index, frame in enumerate(
                        side_by_side_frames(
                            oracle,
                            rotator,
                            duplicate_left=duplicate_left,
                            duplicate_right=duplicate_right,
                        )
                    )
                    if index % gif_stride == 0
                ),
                frame_duration_ms=round(1000.0 * gif_stride / args.fps),
                width=int(args.gif_width),
            )
            with Image.open(gif_path) as gif_image:
                gif_file_frames = int(getattr(gif_image, "n_frames", 1))
                gif_frame_ms = int(gif_image.info.get("duration", 0) or 0)
                gif_size = gif_image.size
            clips["preview-gif"] = {
                "path": str(gif_path.resolve()),
                "bytes": int(gif_path.stat().st_size),
                "frames": gif_file_frames,
                "written_frames": int(gif_frames),
                "duration_s": round(gif_file_frames * max(gif_frame_ms, 1) / 1000.0, 2),
                "width": gif_size[0],
                "height": gif_size[1],
                "gif_stride": gif_stride,
                "mean_luma": round(float(np.mean([mean_luma(frame) for frame in stills])), 2),
            }
    finally:
        renderer.close()
        env.close()

    if oracle is not None and rotator is not None:
        report = {
            "seed": args.seed,
            "controller": args.controller,
            "holder_off": bool(args.holder_off),
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "step_stride": args.step_stride,
            "target_seconds": args.target_seconds,
            "encoder": sorted(set(encoders)),
            "clips": clips,
        }
        (args.out / SUMMARY_JSON).write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )

    print()
    rows = []
    for name, clip in clips.items():
        probe = clip.get("probe", {})
        frames = probe.get("frames") or clip.get("frames") or clip.get("video_frames", "?")
        seconds = probe.get("duration_s") or clip.get("duration_s") or 0.0
        rows.append(
            (
                name,
                Path(clip["path"]).name,
                str(frames),
                f"{seconds:.2f}",
                "{}x{}".format(
                    probe.get("width", clip.get("width", args.width)),
                    probe.get("height", clip.get("height", args.height)),
                ),
                f"{clip['bytes'] / 1e6:.2f} MB",
                str(clip.get("mean_luma", "-")),
            )
        )
    _print_table(rows, ("clip", "file", "frames", "seconds", "size", "bytes", "luma"))
    print(f"\nartifacts in: {args.out.resolve()}")
    print(f"encoder(s): {sorted(set(encoders))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
