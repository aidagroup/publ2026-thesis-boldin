"""Render the CubeWrench-v0 demo videos: oracle success, rotator-only failure, learned policy.

Clips rendered by one run (all from the env's own physics, rendered with `mujoco.Renderer`
on a render-only copy of the scene, see `scripts/_render_utils.py`):

1. ``demo_oracle.mp4`` -- the scripted oracle: the holder PD-stabilises the core while the
   rotator PD-turns the layer to 90 degrees -> SUCCESS (~50 control steps at the defaults).
2. ``demo_rotator_only.mp4`` -- the same oracle with the holder wrench forced to zero: the
   rotator's reaction torque spins the core past the 0.5 rad drift bound -> FAILED after
   ~20 control steps (the non-additivity the reduced model exists to show).
3. ``demo_side_by_side.mp4`` -- clips 1 and 2 in one frame, synchronised by control step,
   each half keeping its own caption and HUD.
4. ``demo_policy_vs_oracle.mp4`` -- only with ``--checkpoint``: the trained `wrench_ppo`
   policies (greedy, left) next to the scripted oracle (right, the upper bound), both ending
   together (each half at its own slow-motion factor: the policy needs ~2-3x the oracle's
   control steps, so the oracle would otherwise sit frozen for most of the clip). The policy
   input mode comes from ``--input-mode`` (or the checkpoint):
   ``none`` / ``oracle`` engine observations or ``predicted`` (Bi-JEPA ``ẑ_partner``, built by
   `callosum.training.wrench_ppo.LatentInputBuilder`).

Plus ``demo_preview.gif`` (looping preview; it prefers the learned-policy-vs-oracle clip when
that scene is rendered and falls back to the oracle-vs-rotator clip), the first three
frames of every video as PNG stills (``oracle_frame0.png`` ...; they are identical by
construction, because the slow-motion expansion holds frame 0), a middle frame (``*_mid.png``)
and the final frame (``*_final.png``) of every clip, and ``demo_summary.json`` with the episode
metrics, the still md5s and the video probe results.

The learned-policy path reuses `scripts/eval_wrench_cube.py` (checkpoint loading, input-mode
resolution, greedy inference and the Bi-JEPA input builder) instead of duplicating it, so the
clip shows exactly what the ablation evaluates.

Nothing is installed: MP4 is written by OpenCV when available and by the system ``ffmpeg``
otherwise (the pod's venv has no OpenCV), GIF by Pillow. Rendering is offscreen: the script
selects ``MUJOCO_GL=egl`` on Linux and ``cgl`` on macOS, unless ``MUJOCO_GL`` is already set.

Run on the pod (from the repo root):

    MUJOCO_GL=egl /root/.callosum/venv/bin/python scripts/render_wrench_demo.py \
        --out /workspace/callosum-runs/render --seed 0 --width 640 --height 640 \
        --fps 30 --controller oracle

    MUJOCO_GL=egl /root/.callosum/venv/bin/python scripts/render_wrench_demo.py \
        --checkpoint /workspace/callosum-runs/wrench_v3_none_s0/latest.pt \
        --input-mode none --out /workspace/callosum-runs/render_policy \
        --seed 0 --width 640 --height 640 --fps 30

Quick metric-only checks (no rendering, no OpenGL):

    python scripts/render_wrench_demo.py --out /tmp/render --dry-run
    python scripts/render_wrench_demo.py --out /tmp/render --dry-run \
        --checkpoint /path/latest.pt --input-mode none --scene policy-vs-oracle

Single scene / holder-off variant:

    python scripts/render_wrench_demo.py --out /tmp/render --scene oracle --holder-off
"""

from __future__ import annotations

import argparse
import hashlib
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
    from callosum.envs.wrench_cube import WrenchCubeEnv

TITLE_ORACLE = "hold + turn (both agents)"
"""Caption of scene 1 (the oracle's coordinated two-agent turn)."""
SUBTITLE_ORACLE = "oracle: holder stabilises the core, rotator turns the layer to 90°"
TITLE_ROTATOR = "rotator only (no partner)"
"""Caption of scene 2 (the holder's wrench forced to zero)."""
SUBTITLE_ROTATOR = "holder wrench = 0: the reaction torque spins the core out of tolerance"
TITLE_ORACLE_UPPER = "oracle (upper bound)"
"""Caption of the scripted oracle half of the learned-policy comparison."""
SUBTITLE_ORACLE_UPPER = "scripted holder + rotator PD: privileged, not a learned policy"

ORACLE_VIDEO = "demo_oracle.mp4"
ROTATOR_VIDEO = "demo_rotator_only.mp4"
SIDE_BY_SIDE_VIDEO = "demo_side_by_side.mp4"
POLICY_VIDEO = "demo_policy_vs_oracle.mp4"
POLICY_STEM = "policy_vs_oracle"
PREVIEW_GIF = "demo_preview.gif"
SUMMARY_JSON = "demo_summary.json"
PNG_STILLS = 3
"""Number of leading frames saved as ``*_frame{i}.png`` per video."""
SBS_MID_FRACTION = 0.6
"""Position of the ``*_mid.png`` still inside a side-by-side clip (fraction of its length)."""

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


# --- action sources ----------------------------------------------------------------------


class OracleDemoController:
    """The scripted oracle (`callosum.envs.wrench_cube.OracleController`) behind the demo.

    It is a centralised script that reads the whole `env.state()`; the demo uses it as the
    upper bound, not as a policy. The adapter gives it the demo's `reset`/`act`/`observe`
    protocol (the oracle keeps no state between steps, so `reset`/`observe` are no-ops).
    """

    env_mode: str = "none"
    """The env's partner-visibility mode the oracle scenes run on."""

    label: str = "oracle"

    def __init__(self, config: Any) -> None:
        """Wrap a `WrenchCubeConfig` with the scripted oracle."""
        from callosum.envs.wrench_cube import OracleController

        self._oracle = OracleController(config)

    def reset(self, obs: dict[str, np.ndarray]) -> None:
        """No-op: the oracle has no per-episode state (the env state is read in `act`)."""
        del obs

    def act(self, env: WrenchCubeEnv, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The oracle's normalised wrench commands for the next control step."""
        del obs
        return self._oracle.act(env)

    def observe(self, obs: dict[str, np.ndarray]) -> None:
        """No-op: the oracle needs no observation history."""
        del obs


class PolicyDemoController:
    """Greedy `wrench_ppo` checkpoint policies behind the demo's controller protocol.

    Loading and inference reuse `scripts/eval_wrench_cube.py` / `callosum.training.wrench_ppo`:
    `load_checkpoint` reads the payload, `eval_wrench_cube.resolve_input_mode` decides the
    policy input mode (flag or checkpoint), `check_checkpoint_compat` validates it, and the
    `predicted` input is built by `LatentInputBuilder` with exactly the `reset`/`step` seam of
    the trainer and the evaluator. Actions are the distribution means, clipped to `[-1, 1]`.

    Under `predicted` the env stays at `partner_mode="none"` (the agent cannot see the
    privileged partner block it is asked to predict); `none`/`oracle` map onto themselves.
    """

    def __init__(self, checkpoint: Path, input_mode: str | None) -> None:
        """Load the checkpoint and build both greedy policies (and the Bi-JEPA builder).

        Args:
            checkpoint: A `wrench_ppo` `latest.pt` checkpoint.
            input_mode: `none` / `oracle` / `predicted`, or None to use the checkpoint's own
                saved mode (see `eval_wrench_cube.resolve_input_mode`).
        """
        import eval_wrench_cube
        import torch

        from callosum.training.wrench_ppo import (
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
        self._eval = eval_wrench_cube
        self.checkpoint = Path(checkpoint)
        payload = load_checkpoint(self.checkpoint)
        resolved = eval_wrench_cube.resolve_input_mode(
            payload,
            argparse.Namespace(
                checkpoint=str(self.checkpoint), input_mode=input_mode, partner_mode=None
            ),
        )
        env_mode, latent_dim = check_checkpoint_compat(payload, resolved)
        cfg = config_from_payload(payload)
        if resolved == "predicted" and (not cfg.use_bijepa or cfg.latent_dim != latent_dim):
            raise ValueError(
                "checkpoint is inconsistent: it stores Bi-JEPA weights (latent_dim="
                f"{latent_dim}) but its saved config has use_bijepa={cfg.use_bijepa} and "
                f"latent_dim={cfg.latent_dim}"
            )
        self.input_mode = resolved
        self.env_mode = env_mode
        self.latent_dim = latent_dim
        self.update = payload.get("update")
        self.format = payload.get("format")
        self.policies = make_policies(env_mode, "cpu", latent_dim=latent_dim)
        load_weights(payload, self.policies)
        self._builder: LatentInputBuilder | None = None
        bijepa = make_bijepa(cfg, "cpu")
        if bijepa is not None:
            load_bijepa_weights(payload, bijepa)
            bijepa.eval()
            self._builder = LatentInputBuilder(bijepa, device="cpu")
        self._policy_obs: dict[str, Any] | None = None
        from callosum.envs._wrench_logic import AGENT_KEYS

        self._agent_keys = tuple(AGENT_KEYS)

    @property
    def label(self) -> str:
        """Short name of the action source, for the summary."""
        return f"policy ({self.input_mode})"

    @property
    def title(self) -> str:
        """The clip caption, e.g. "learned policy (none)"."""
        return f"learned policy ({self.input_mode})"

    @property
    def subtitle(self) -> str:
        """One-line description under the caption (mode, checkpoint, update)."""
        detail = (
            "Bi-JEPA ẑ_partner" if self.input_mode == "predicted" else f"{self.input_mode} input"
        )
        return f"greedy policy, {detail} · {self.checkpoint.name} (update {self.update})"

    def metadata(self) -> dict[str, Any]:
        """The JSON-ready checkpoint metadata stored in the demo summary."""
        return {
            "checkpoint": str(self.checkpoint),
            "input_mode": self.input_mode,
            "env_partner_mode": self.env_mode,
            "latent_dim": self.latent_dim,
            "update": self.update,
            "format": self.format,
        }

    def reset(self, obs: dict[str, np.ndarray]) -> None:
        """Start a new episode: the policy input is the first observation (history reset)."""
        batch = self._eval._batch_obs(obs)
        self._policy_obs = self._builder.reset(batch) if self._builder is not None else batch

    def act(self, env: WrenchCubeEnv, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The greedy (distribution mean) action of both policies, clipped to `[-1, 1]`."""
        del env, obs
        if self._policy_obs is None:
            raise RuntimeError("call reset() before the policy acts")
        actions: dict[str, np.ndarray] = {}
        for name, policy in zip(self._agent_keys, self.policies):
            with self._torch.no_grad():
                action = policy.get_action(self._policy_obs[name], deterministic=True)
            actions[name] = self._torch.clamp(action, -1.0, 1.0).squeeze(0).numpy()
        return actions

    def observe(self, obs: dict[str, np.ndarray]) -> None:
        """Advance the policy input after a non-terminal step.

        `none`/`oracle`: the policy input is the current env observation itself (the same
        seam as `eval_wrench_cube.run_episode`). `predicted`: the Bi-JEPA builder rolls the
        observation history and rebuilds `[own obs, ẑ_partner]`.
        """
        batch = self._eval._batch_obs(obs)
        self._policy_obs = (
            self._builder.step(batch, self._torch.zeros(1)) if self._builder is not None else batch
        )


DemoController = OracleDemoController | PolicyDemoController
"""The demo's two action sources; both expose `reset` / `act` / `observe`."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the demo-render CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Render the CubeWrench-v0 demo videos (oracle success, rotator-only failure, "
            "side by side, and optionally learned policy vs oracle) with MuJoCo offscreen "
            "rendering."
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
        help="episode seed of all scenes",
    )
    parser.add_argument("--width", type=int, default=640, help="frame width (px)")
    parser.add_argument("--height", type=int, default=640, help="frame height (px)")
    parser.add_argument("--fps", type=int, default=30, help="video frame rate")
    parser.add_argument(
        "--controller",
        choices=("oracle",),
        default="oracle",
        help="scripted action source of the oracle / rotator-only scenes",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "wrench_ppo latest.pt: renders demo_policy_vs_oracle.mp4 (learned policies on the "
            "left, scripted oracle on the right)"
        ),
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
        "--policy-max-steps",
        type=int,
        default=None,
        help=(
            "rollout cap of the learned-policy episode (default: 2x the oracle episode, "
            "at least 100 steps, capped by --max-steps)"
        ),
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
        choices=("all", "oracle", "rotator-only", "side-by-side", "policy-vs-oracle"),
        default="all",
        help=(
            "'all' also renders the policy scene when --checkpoint is given; "
            "'policy-vs-oracle' requires --checkpoint"
        ),
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


def _policy_max_steps(
    args: argparse.Namespace, env_max_steps: int, reference_steps: int | None
) -> int:
    """Rollout cap of the learned-policy episode.

    Default: twice the oracle episode, at least 150 steps (the trained policies take ~120-180
    control steps to succeed), so a learned policy that is slower than the script still gets
    the chance to finish inside the clip; never above the env's own budget.
    """
    if args.policy_max_steps is not None:
        return max(1, min(int(args.policy_max_steps), env_max_steps))
    if reference_steps is None:
        return env_max_steps
    return min(env_max_steps, max(150, 2 * int(reference_steps)))


def _scene_wanted(scene: str) -> tuple[bool, bool, bool]:
    """Return `(oracle, rotator-only, oracle-vs-rotator)` flags of the `--scene` choice."""
    return (
        scene in ("all", "oracle", "side-by-side"),
        scene in ("all", "rotator-only", "side-by-side"),
        scene in ("all", "side-by-side"),
    )


def run_episode(
    env: WrenchCubeEnv,
    controller: DemoController,
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

    The controller seam is `reset(obs)` once, then `act(env, obs)` before every step and
    `observe(obs)` after every non-terminal step (the Bi-JEPA `predicted` input needs it).
    """
    config = env.config
    obs, _reset_info = env.reset(seed=seed)
    controller.reset(obs)
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
        action = controller.act(env, obs)
        if holder_off:
            action = {"holder": np.zeros(6), "rotator": action["rotator"]}
        obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
        controller.observe(obs)
        if steps >= max_steps:
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


def _side_by_side_at(
    left: DemoEpisode,
    right: DemoEpisode,
    index: int,
    duplicate_left: int,
    duplicate_right: int,
) -> np.ndarray:
    """The side-by-side RGB frame at video-frame `index` (used for the PNG stills)."""
    return hstack_side_by_side(
        _expanded_index(left.frames, index, duplicate_left),
        _expanded_index(right.frames, index, duplicate_right),
    )


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
        yield _side_by_side_at(left, right, index, duplicate_left, duplicate_right)


def _md5_image(image: np.ndarray) -> str:
    """Short md5 of an RGB frame (the report's proof that the stills really differ)."""
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


def _write_single_clip(
    out_dir: Path,
    filename: str,
    stem: str,
    episode: DemoEpisode,
    *,
    fps: int,
    target_seconds: float,
    encoders: list[str],
) -> dict[str, Any]:
    """Encode a single-scene clip plus its stills (frame0-2, mid, final)."""
    duplicate = duplicate_for_target(episode, fps, target_seconds)
    first = [_expanded_index(episode.frames, index, duplicate) for index in range(PNG_STILLS)]
    middle = _expanded_index(
        episode.frames, round(_repeat_count(episode.frames, duplicate) * 0.5), duplicate
    )
    final = episode.frames[-1].image
    clip = _encode_clip(
        out_dir / filename,
        _repeat_frames(episode.frames, duplicate),
        _repeat_count(episode.frames, duplicate),
        [first[0], middle, final],
        fps,
        encoders,
    )
    clip.update({"episode": episode.summary(), "duplicate": duplicate})
    save_png_stills(out_dir, stem, first, count=len(first))
    _save_image(out_dir / f"{stem}_mid.png", middle)
    _save_image(out_dir / f"{stem}_final.png", final)
    clip["still_md5"] = {
        "frame0": _md5_image(first[0]),
        "mid": _md5_image(middle),
        "final": _md5_image(final),
    }
    return clip


def _sbs_duplicates(
    left: DemoEpisode, right: DemoEpisode, fps: int, target_seconds: float, sync: str
) -> tuple[int, int]:
    """Frame-hold factors of the two halves of a side-by-side clip.

    `sync="step"` (oracle vs rotator-only) keeps both halves on the same physical time axis:
    one shared hold factor, so the shorter episode freezes at its final frame. `sync="end"`
    (learned policy vs oracle) gives each half its own factor so both reach their final state
    at the same video time -- the policy needs ~2-3x the oracle's steps, and freezing the
    oracle half for most of the clip would waste the comparison. The HUD step counters stay
    honest either way.
    """
    longer = left if len(left.frames) >= len(right.frames) else right
    duplicate_longer = duplicate_for_target(longer, fps, target_seconds)
    if sync == "step":
        return duplicate_longer, duplicate_longer
    if sync != "end":
        raise ValueError(f"sync must be 'step' or 'end', got {sync!r}")
    shorter = right if longer is left else left
    duplicate_shorter = max(
        1, round(duplicate_longer * len(longer.frames) / max(1, len(shorter.frames)))
    )
    if longer is left:
        return duplicate_longer, duplicate_shorter
    return duplicate_shorter, duplicate_longer


def _write_sbs_clip(
    out_dir: Path,
    filename: str,
    stem: str,
    left: DemoEpisode,
    right: DemoEpisode,
    *,
    fps: int,
    target_seconds: float,
    sync: str,
    encoders: list[str],
) -> dict[str, Any]:
    """Encode a side-by-side clip; returns its metrics.

    `sync` is `"step"` (the oracle-vs-rotator comparison: same hold factor on both halves,
    synchronised by control step) or `"end"` (the learned-policy-vs-oracle comparison: each
    half gets the hold factor that makes it finish at the same video time). Stills are the
    first three video frames, the frame at `SBS_MID_FRACTION` of the clip and the true last
    frame (both halves' final states); the final still must differ from frame 0 (it shows the
    SUCCESS / FAILED end state), and the report carries the still md5s as proof.
    """
    duplicate_left, duplicate_right = _sbs_duplicates(left, right, fps, target_seconds, sync)
    total = max(
        _repeat_count(left.frames, duplicate_left),
        _repeat_count(right.frames, duplicate_right),
    )
    first = [
        _side_by_side_at(left, right, index, duplicate_left, duplicate_right)
        for index in range(PNG_STILLS)
    ]
    middle = _side_by_side_at(
        left, right, round(total * SBS_MID_FRACTION), duplicate_left, duplicate_right
    )
    final = _side_by_side_at(left, right, total - 1, duplicate_left, duplicate_right)
    if np.array_equal(final, first[0]):
        raise RuntimeError(f"{stem}: the final still equals frame 0 -- the clip has no time axis")
    clip = _encode_clip(
        out_dir / filename,
        side_by_side_frames(
            left, right, duplicate_left=duplicate_left, duplicate_right=duplicate_right
        ),
        total,
        [first[0], middle, final],
        fps,
        encoders,
    )
    clip.update(
        {
            "sync": sync,
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
    return clip


def _write_gif_preview(
    out_dir: Path,
    left: DemoEpisode,
    right: DemoEpisode,
    *,
    duplicate_left: int,
    duplicate_right: int,
    fps: int,
    gif_width: int,
    gif_stride: int = 2,
) -> dict[str, Any]:
    """Write the looping GIF preview of a side-by-side clip; returns its summary block."""
    gif_path = out_dir / PREVIEW_GIF
    gif_frames = write_gif(
        gif_path,
        (
            frame
            for index, frame in enumerate(
                side_by_side_frames(
                    left,
                    right,
                    duplicate_left=duplicate_left,
                    duplicate_right=duplicate_right,
                )
            )
            if index % gif_stride == 0
        ),
        frame_duration_ms=round(1000.0 * gif_stride / fps),
        width=int(gif_width),
    )
    with Image.open(gif_path) as gif_image:
        gif_file_frames = int(getattr(gif_image, "n_frames", 1))
        gif_frame_ms = int(gif_image.info.get("duration", 0) or 0)
        gif_size = gif_image.size
    samples = [
        left.frames[min(len(left.frames) - 1, len(left.frames) // 2)].image,
        right.frames[min(len(right.frames) - 1, len(right.frames) // 2)].image,
    ]
    return {
        "path": str(gif_path.resolve()),
        "bytes": int(gif_path.stat().st_size),
        "frames": gif_file_frames,
        "written_frames": int(gif_frames),
        "duration_s": round(gif_file_frames * max(gif_frame_ms, 1) / 1000.0, 2),
        "width": gif_size[0],
        "height": gif_size[1],
        "gif_stride": gif_stride,
        "source": f"{left.name} vs {right.name}",
        "mean_luma": round(float(np.mean([mean_luma(frame) for frame in samples])), 2),
    }


def _print_table(rows: Sequence[Sequence[str]], header: Sequence[str]) -> None:
    """Print a small fixed-width table."""
    widths = [max(len(str(row[i])) for row in (header, *rows)) for i in range(len(header))]
    print("  ".join(str(header[i]).ljust(widths[i]) for i in range(len(header))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(header))))


def _load_policy_controller(
    args: argparse.Namespace,
) -> PolicyDemoController | None:
    """Build the learned-policy controller when the scene needs it (clean CLI error)."""
    if args.checkpoint is None:
        return None
    if not (args.scene in ("all", "policy-vs-oracle")):
        print(
            f"note: --checkpoint is ignored for --scene {args.scene} "
            "(use --scene all or --scene policy-vs-oracle)"
        )
        return None
    try:
        return PolicyDemoController(args.checkpoint, args.input_mode)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"cannot load checkpoint {args.checkpoint}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Render the demo artifacts; returns the process exit code."""
    args = parse_args(argv)
    if args.width % 2 or args.height % 2:
        raise SystemExit(f"--width/--height must be even, got {args.width}x{args.height}")
    if args.step_stride < 1:
        raise SystemExit(f"--step-stride must be >= 1, got {args.step_stride}")
    if args.fps < 1:
        raise SystemExit(f"--fps must be >= 1, got {args.fps}")
    if args.scene == "policy-vs-oracle" and args.checkpoint is None:
        raise SystemExit("--scene policy-vs-oracle requires --checkpoint PATH")
    _configure_gl()
    args.out.mkdir(parents=True, exist_ok=True)

    from callosum.envs.wrench_cube import WrenchCubeEnv

    want_oracle, want_rotator, want_oracle_sbs = _scene_wanted(args.scene)
    policy_controller = _load_policy_controller(args)
    want_policy = policy_controller is not None
    env = WrenchCubeEnv(partner_mode="none")
    oracle_controller = OracleDemoController(env.config)
    max_steps = int(args.max_steps or env.config.max_control_steps)
    print(
        f"CubeWrench-v0 demo render: scene={args.scene} seed={args.seed} "
        f"size={args.width}x{args.height} fps={args.fps} step_stride={args.step_stride} "
        f"max_steps={max_steps} holder_off={args.holder_off} "
        f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}"
    )
    if policy_controller is not None:
        print(
            f"checkpoint {policy_controller.checkpoint}: "
            f"input_mode={policy_controller.input_mode} "
            f"env_partner_mode={policy_controller.env_mode} "
            f"latent_dim={policy_controller.latent_dim} "
            f"update={policy_controller.update} (greedy policies)"
        )

    if args.dry_run:
        episodes: dict[str, DemoEpisode] = {}
        policy_env: WrenchCubeEnv | None = None
        try:
            for name, wanted, holder_off in (
                ("oracle", want_oracle, args.holder_off),
                ("rotator-only", want_rotator, True),
            ):
                if not wanted:
                    continue
                episodes[name] = run_episode(
                    env,
                    oracle_controller,
                    None,
                    name=name,
                    title="",
                    subtitle="",
                    seed=args.seed,
                    step_stride=args.step_stride,
                    max_steps=max_steps,
                    holder_off=holder_off,
                )
                print(f"[dry-run] {name}: {json.dumps(episodes[name].summary())}")
            if want_policy and policy_controller is not None:
                reference_steps = episodes["oracle"].steps if "oracle" in episodes else None
                if reference_steps is None:
                    upper = run_episode(
                        env,
                        oracle_controller,
                        None,
                        name="oracle-upper",
                        title="",
                        subtitle="",
                        seed=args.seed,
                        step_stride=args.step_stride,
                        max_steps=max_steps,
                        holder_off=args.holder_off,
                    )
                    print(f"[dry-run] oracle-upper: {json.dumps(upper.summary())}")
                    reference_steps = upper.steps
                policy_steps = _policy_max_steps(args, max_steps, reference_steps)
                policy_env = WrenchCubeEnv(partner_mode=policy_controller.env_mode)
                episode = run_episode(
                    policy_env,
                    policy_controller,
                    None,
                    name="policy",
                    title="",
                    subtitle="",
                    seed=args.seed,
                    step_stride=args.step_stride,
                    max_steps=policy_steps,
                    holder_off=args.holder_off,
                )
                print(f"[dry-run] policy: {json.dumps(episode.summary())}")
        finally:
            if policy_env is not None:
                policy_env.close()
            env.close()
        return 0

    renderer = DemoRenderer(args.width, args.height)
    encoders: list[str] = []
    clips: dict[str, dict[str, Any]] = {}
    oracle: DemoEpisode | None = None
    rotator: DemoEpisode | None = None
    policy_env: WrenchCubeEnv | None = None
    try:
        if want_oracle:
            oracle = run_episode(
                env,
                oracle_controller,
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
                oracle_controller,
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
            clips["oracle"] = _write_single_clip(
                args.out,
                ORACLE_VIDEO,
                "oracle",
                oracle,
                fps=args.fps,
                target_seconds=args.target_seconds,
                encoders=encoders,
            )
        if rotator is not None:
            clips["rotator-only"] = _write_single_clip(
                args.out,
                ROTATOR_VIDEO,
                "rotator_only",
                rotator,
                fps=args.fps,
                target_seconds=args.target_seconds,
                encoders=encoders,
            )

        if want_oracle_sbs and oracle is not None and rotator is not None:
            clip = _write_sbs_clip(
                args.out,
                SIDE_BY_SIDE_VIDEO,
                "side_by_side",
                oracle,
                rotator,
                fps=args.fps,
                target_seconds=args.target_seconds,
                sync="step",
                encoders=encoders,
            )
            clips["side-by-side"] = clip
            clips["preview-gif"] = _write_gif_preview(
                args.out,
                oracle,
                rotator,
                duplicate_left=int(clip["duplicate_left"]),
                duplicate_right=int(clip["duplicate_right"]),
                fps=args.fps,
                gif_width=args.gif_width,
            )

        if want_policy and policy_controller is not None:
            oracle_upper = run_episode(
                env,
                oracle_controller,
                renderer,
                name="oracle-upper",
                title=TITLE_ORACLE_UPPER,
                subtitle=SUBTITLE_ORACLE_UPPER,
                seed=args.seed,
                step_stride=args.step_stride,
                max_steps=max_steps,
                holder_off=args.holder_off,
            )
            print(
                f"oracle upper bound: {oracle_upper.status} after {oracle_upper.steps} steps, "
                f"angle {oracle_upper.final_angle_deg:.1f} deg"
            )
            policy_steps = _policy_max_steps(args, max_steps, oracle_upper.steps)
            policy_env = WrenchCubeEnv(partner_mode=policy_controller.env_mode)
            policy_episode = run_episode(
                policy_env,
                policy_controller,
                renderer,
                name="policy",
                title=policy_controller.title,
                subtitle=policy_controller.subtitle,
                seed=args.seed,
                step_stride=args.step_stride,
                max_steps=policy_steps,
                holder_off=args.holder_off,
            )
            print(
                f"learned policy ({policy_controller.input_mode}): {policy_episode.status} "
                f"after {policy_episode.steps} steps (cap {policy_steps}), "
                f"angle {policy_episode.final_angle_deg:.1f} deg, "
                f"drift {policy_episode.final_drift_deg:.1f} deg"
            )
            clip = _write_sbs_clip(
                args.out,
                POLICY_VIDEO,
                POLICY_STEM,
                policy_episode,
                oracle_upper,
                fps=args.fps,
                target_seconds=args.target_seconds,
                sync="end",
                encoders=encoders,
            )
            clip["policy"] = policy_controller.metadata()
            clip["policy_max_steps"] = policy_steps
            clips["policy-vs-oracle"] = clip
            # The preview GIF prefers the learned-policy comparison when it is rendered.
            clips["preview-gif"] = _write_gif_preview(
                args.out,
                policy_episode,
                oracle_upper,
                duplicate_left=int(clip["duplicate_left"]),
                duplicate_right=int(clip["duplicate_right"]),
                fps=args.fps,
                gif_width=args.gif_width,
            )
    finally:
        renderer.close()
        if policy_env is not None:
            policy_env.close()
        env.close()

    if clips:
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
        if policy_controller is not None:
            report["policy"] = policy_controller.metadata()
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
