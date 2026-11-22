"""[cloud] Render the pif-paf (Sexy Move `R U R' U'`) demo of FaceTurnSkills-v0.

What the clip shows
-------------------
The two SO-ARM101 hands of `FaceTurnSkills-v0` execute a FIXED plan in standard face-turn
notation: `R U R' U'` (the Sexy Move / pif-paf), the fixed-plan skill-level task. Both hands
are driven by the scripted conductor (`callosum.experiments.arm.skills_conductor`), the
skill-level line's upper bound / environment validation -- the clip is the ENV+EXPERT demo,
not a trained-policy result (the same convention as `render_face_turn_arms.py`).

The plan is parsed by `callosum.experiments.arm.skills_plans.parse_face_turn_sequence`: a
prime turn expands into two executable moves (90 + 180 deg, the layer joints turn clockwise
[0, 180] only), so `R U R' U'` is SIX plan moves: `R U R R2 U U2`. The HUD shows both the
macro turn (`R'`, pressed once by the conductor's flip/turn logic) and the executable move.

Why a separate script
---------------------
The env defines no cameras (`_default_human_render_camera_configs` is empty), so the smoke
scripts cannot render it. This script subclasses `FaceTurnSkills`, registering an in-process
demo env that overrides ONLY the human-render camera configuration; the env, the physics,
the cube and the conductor are untouched.

Rendering on the pod (CPU)
--------------------------
The pod has no hardware Vulkan, but Mesa's lavapipe software Vulkan works and SAPIEN 3.0.3
renders on it: `render_backend="cpu"` builds `sapien.render.RenderSystem(sapien.Device("cpu"))`
and the sim runs on `sim_backend="cpu"` (one env, PhysX CPU). About 0.5 s per 640x480 frame
after a one-off shader compilation.

Usage (pod, repo root):
```
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  /root/.callosum/venv/bin/python scripts/experiments/arm/render_sexy_move_skills.py \
  --out demo --sim-backend cpu --render-backend cpu \
  --plan-notation "R U R' U'" --max-episode-decisions 600
```

Outputs (`--out`, default `demo/`): `demo_sexy_move.mp4` (H.264, 10 fps),
`demo_sexy_move_preview.gif`, four stills (`..._frame0.png`, `..._flip.png`,
`..._turn.png`, `..._success.png`) and `demo_sexy_move.json` (plan, success, decisions,
flips, final angles/indices, render stats). Every frame carries the plan, the current
macro/executable move, the cube's up face and flip count, and SUCCESS on the final frames.

Exit status: 0 if the conductor reached env success, 1 otherwise (a best-effort clip is
still written and its final frames say INCOMPLETE).
"""

from __future__ import annotations

import argparse
import json
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

DEMO_ENV_ID = "FaceTurnSkillsSexyDemo-v0"
"""In-process env id: `FaceTurnSkills-v0` + the demo camera."""

DEFAULT_PLAN = "R U R' U'"
"""The pif-paf / Sexy Move in standard face-turn notation."""

TITLE = "two SO-ARM101 arms execute the Sexy Move (pif-paf)"
SUBTITLE = (
    "FaceTurnSkills-v0 - multi-face cube, fixed plan R U R' U' (6 executable moves), "
    "scripted conductor = skill-level upper bound"
)

DEFAULT_EYE = (0.15, -0.16, 0.11)
DEFAULT_TARGET = (0.0, 0.0, 0.03)
"""Camera pose (m): the three-quarter view of `render_face_turn_arms.py` (both arm bases and
the cube at 640x480, 0.85 rad fov)."""
DEFAULT_FOV = 0.70

MP4_NAME = "demo_sexy_move.mp4"
LEARNED_MP4_NAME = "demo_arms_learned_turn.mp4"
GIF_NAME = "demo_sexy_move_preview.gif"
LEARNED_GIF_NAME = "demo_arms_learned_turn_preview.gif"
JSON_NAME = "demo_sexy_move.json"
LEARNED_JSON_NAME = "demo_arms_learned_turn.json"
STILL_NAMES = (
    "demo_sexy_move_frame0.png",
    "demo_sexy_move_flip.png",
    "demo_sexy_move_turn.png",
    "demo_sexy_move_success.png",
)

TURN_STILL_MIN_DEG = 25.0
TURN_STILL_MAX_DEG = 75.0
"""Window of the active layer's angle used to pick the "turn" still."""


def _prefer_software_vulkan() -> None:
    """Point SAPIEN at the Mesa lavapipe ICD when the caller has not picked one.

    SAPIEN probes the Vulkan ICDs when `sapien` is imported, so this must run before the
    sim imports inside `_run_demo`.
    """
    if os.environ.get("VK_ICD_FILENAMES") or os.environ.get("VK_DRIVER_FILES"):
        return
    if LAVAPIPE_ICD.is_file():
        os.environ["VK_ICD_FILENAMES"] = str(LAVAPIPE_ICD)
        print(f"using the Mesa lavapipe software Vulkan ICD: {LAVAPIPE_ICD}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("demo"),
        help="Output directory.",
    )
    parser.add_argument(
        "--policy",
        choices=("conductor", "learned"),
        default="conductor",
        help="Policy driving the clip: scripted conductor (upper bound) or a learned BC checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Format-1 BC/IPPO checkpoint of the two categorical policies (--policy learned).",
    )
    parser.add_argument(
        "--partner-mode",
        default="oracle",
        help="Partner-visibility mode the env/policies use. (default: oracle)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Reset seed.")
    parser.add_argument(
        "--plan-notation",
        default=DEFAULT_PLAN,
        help="Standard face-turn notation of the fixed plan. (default: %(default)r)",
    )
    parser.add_argument(
        "--max-episode-decisions",
        type=int,
        default=600,
        help="Env decision budget of the episode. (default: %(default)s)",
    )
    parser.add_argument(
        "--sim-backend",
        default="cpu",
        help='ManiSkill sim backend ("cpu" on the lavapipe pod). (default: %(default)s)',
    )
    parser.add_argument(
        "--render-backend",
        default="cpu",
        help='ManiSkill render backend ("cpu" = sapien CPU renderer). (default: %(default)s)',
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov", type=float, default=DEFAULT_FOV)
    parser.add_argument("--eye", type=float, nargs=3, default=list(DEFAULT_EYE))
    parser.add_argument("--target", type=float, nargs=3, default=list(DEFAULT_TARGET))
    parser.add_argument(
        "--every",
        type=int,
        default=4,
        help="Render one frame every N decisions. (default: %(default)s)",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Video fps. (default: 10)")
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument(
        "--intro-seconds",
        type=float,
        default=2.0,
        help="Goal/intro hold before the clip: duplicate of the first frame. (default: 2)",
    )
    parser.add_argument("--gif-every", type=int, default=2)
    parser.add_argument("--gif-width", type=int, default=320)
    parser.add_argument("--stills-only", action="store_true", help="Only write the stills.")
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="Do not add the render-only cyan (active layer) / black (core) markers.",
    )
    parser.add_argument(
        "--show-all-slabs",
        action="store_true",
        help=(
            "Keep all six layer slabs visible. By default the five inactive slabs are "
            "hidden in the CPU renderer (they overlap at the edges and read as a broken "
            "cube), leaving only the core + the active layer."
        ),
    )
    args = parser.parse_args(argv)
    if args.every < 1:
        parser.error("--every must be >= 1")
    if args.fps <= 0:
        parser.error("--fps must be > 0")
    if args.gif_every < 1:
        parser.error("--gif-every must be >= 1")
    return args


def _draw_dial(frame: np.ndarray, fraction: float, label: str = "0 -> 90 deg") -> np.ndarray:
    """Render-only gauge: a 180 deg dial from 0 to the move target plus a needle."""
    from PIL import ImageDraw

    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    radius = max(28, height // 9)
    cx, cy = width - radius - 24, height - radius - 22
    box = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.arc(box, start=180, end=360, fill=(90, 110, 140), width=4)
    fraction = min(max(float(fraction), 0.0), 1.0)
    end = 180 + 180 * fraction
    draw.arc(box, start=180, end=end, fill=(110, 230, 140), width=5)
    angle = math.radians(end)
    draw.line(
        [cx, cy, cx + radius * math.cos(angle), cy + radius * math.sin(angle)],
        fill=(255, 255, 255),
        width=3,
    )
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 255, 255))
    draw.text((cx - radius, cy + 8), label, fill=(210, 222, 240))
    return np.asarray(image)


def _to_frame(image) -> np.ndarray:
    """Turn one `env.render()` result (torch/numpy, `(num_envs,) H W C`) into RGB uint8."""
    array = image.cpu().numpy() if hasattr(image, "cpu") else np.asarray(image)
    array = np.asarray(array)
    if array.ndim == 4:
        array = array[0]
    return np.ascontiguousarray(array[..., :3].astype(np.uint8))


def _write_h264_mp4(path: Path, frames: list[np.ndarray], fps: float) -> str:
    """Encode the frames as an H.264/yuv420p MP4 with ffmpeg; falls back to `write_mp4`."""
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


def _macro_layout(notation: str) -> list[tuple[str, int]]:
    """`(macro token, number of executable moves)` per notation token.

    A prime token (`R'`) expands to two executable moves (90 + 180 deg); every other token
    (`R`, `R2`) is one. Used by the HUD to name the macro turn while showing the executable
    move index the env's plan cursor counts.
    """
    layout: list[tuple[str, int]] = []
    for token in notation.split():
        layout.append((token, 2 if token.endswith("'") else 1))
    return layout


def _macro_at(layout: list[tuple[str, int]], plan_index: int) -> tuple[int, str]:
    """`(macro number, token)` containing executable move `plan_index` (1-based HUD index)."""
    remaining = plan_index
    for number, (token, count) in enumerate(layout, start=1):
        if remaining < count:
            return number, token
        remaining -= count
    return len(layout), layout[-1][0] if layout else "?"


def _decorate(
    frame: np.ndarray,
    meta: dict,
    *,
    plan_notation: str,
    n_executable: int,
    total_decisions: int,
    title: str = TITLE,
    subtitle: str = SUBTITLE,
    policy_label: str = "SCRIPTED CONDUCTOR (upper bound, not learned)",
    goal: str = "",
    status: str | None = None,
    status_color=SUCCESS_COLOR,
    status_detail: str | None = None,
) -> np.ndarray:
    """Draw the demo overlay (goal band, title, live HUD, plan progress, status)."""
    layout = _macro_layout(plan_notation)
    macro_number, macro_token = _macro_at(layout, min(meta["plan_index"], n_executable - 1))
    move_text = (
        "plan completed"
        if meta["plan_index"] >= n_executable
        else f"macro turn {macro_number}/{len(layout)}: {macro_token}  "
        f"(executable move {min(meta['plan_index'] + 1, n_executable)}/{n_executable}: "
        f"{meta['move_text']})"
    )
    if meta["plan_index"] >= n_executable:
        angle_line = f"plan completed: layer at target {meta['target_deg']:.1f} deg"
    else:
        angle_line = (
            f"active layer: {meta['angle_deg']:5.1f} deg -> target {meta['target_deg']:5.1f} deg"
        )
    hud = [
        f"decision {meta['decision']:3d}/{total_decisions}  (sim step {meta['sim_step']})",
        move_text,
        (
            f"cube up face: {meta['up_face']} | flips: {meta['flips']} | "
            f"up faces: {meta['faces_seen']}"
        ),
        angle_line,
        (
            f"holder: {meta['holder']} | rotator: {meta['rotator']} | "
            f"cube drift: {meta['drift_cm']:.2f} cm"
        ),
        policy_label,
    ]
    goal_lines = [
        f"PLAN {plan_notation} -> {n_executable} move"
        if n_executable == 1
        else f"PLAN {plan_notation} -> {n_executable} moves"
    ]
    if goal:
        goal_lines.append(f"GOAL: {goal}")
    return annotate_frame(
        frame,
        title=title,
        subtitle=subtitle,
        hud_lines=hud,
        angle_fraction=min(meta["plan_index"] / max(n_executable, 1), 1.0),
        status=status,
        status_color=status_color,
        status_detail=status_detail,
        goal_lines=goal_lines,
    )


def _run_demo(args: argparse.Namespace) -> dict:
    """Build the demo env, run the scripted conductor, render and write the outputs."""
    # The sim imports are lazy: `_prefer_software_vulkan` must set the ICD before SAPIEN loads.
    import gymnasium as gym
    import sapien
    import torch
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.registration import register_env

    import callosum.experiments.arm.skills_env  # noqa: F401  (registers FaceTurnSkills-v0)
    from callosum.experiments.arm.skills_agent_obs import SkillsObsBuilder
    from callosum.experiments.arm.skills_conductor import (
        HOLDER_ID,
        HOLDER_PRIMITIVES,
        ROTATOR_ID,
        ROTATOR_PRIMITIVES,
        SkillsConductor,
    )
    from callosum.experiments.arm.skills_config import FaceTurnSkillsConfig
    from callosum.experiments.arm.skills_env import FaceTurnSkills
    from callosum.experiments.arm.skills_obs import (
        INFO_ACTIVE_LAYER_ANGLE_DEG,
        INFO_PLAN_INDEX,
        INFO_SUCCESS,
        INFO_TIME_TO_SUCCESS,
    )
    from callosum.experiments.arm.skills_runtime import (
        orientation_from_quat,
        sapien_quat_to_xyzw,
    )

    holder_name = {p.index: p.name for p in HOLDER_PRIMITIVES}
    rotator_name = {p.index: p.name for p in ROTATOR_PRIMITIVES}
    flip_ids = (HOLDER_ID["flip90"], HOLDER_ID["flip180"])

    config = FaceTurnSkillsConfig(
        plan_notation=args.plan_notation,
        max_episode_decisions=int(args.max_episode_decisions),
    )
    fixed_plan = config.fixed_plan
    if fixed_plan is None:  # pragma: no cover - the CLI default is a non-empty notation
        raise SystemExit("--plan-notation must not be empty for the fixed-plan demo")
    move_labels = [str(move) for move in fixed_plan]
    n_executable = len(move_labels)

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

    created_markers: list = []

    class FaceTurnSkillsDemo(FaceTurnSkills):
        """`FaceTurnSkills-v0` + a close-up demo camera, render-only markers and the
        inactive slabs hidden (in the CPU renderer they overlap at the edges and read
        as a broken cube, which is exactly the complaint the close-up demo addresses)."""

        @property
        def _default_human_render_camera_configs(self):
            return [camera_config]

        def _load_scene(self, options: dict):
            super()._load_scene(options)
            active = fixed_plan[0].face
            if not args.show_all_slabs:
                # DIM the inactive slabs instead of hiding them: hiding leaves the core as
                # a floating box, dimming keeps the cube readable while the bright active
                # layer plus the cyan marker make the 90 deg turn unambiguous.
                dim_material = sapien.render.RenderMaterial(
                    base_color=[0.11, 0.12, 0.15, 1.0], roughness=0.9
                )
                for face, link in self.layer_links.items():
                    if face == active:
                        continue
                    for obj in link._objs:
                        component = obj.entity.find_component_by_type(
                            sapien.render.RenderBodyComponent
                        )
                        if component is None:
                            continue
                        try:
                            for shape in component.render_shapes:
                                shape.material = dim_material
                        except Exception:  # noqa: BLE001 - fall back to hiding this slab
                            component.visibility = 0
            if args.no_markers:
                return
            half_size = self.cube_config.cube_half_size
            layer_off = (
                half_size - self.cube_config.layer_thickness / 2 + self.cube_config.layer_gap / 2
            )
            marker_local_z = layer_off + self.cube_config.layer_thickness / 2 + 0.004

            def add_marker(name, half, color):
                builder = self.scene.create_actor_builder()
                builder.initial_pose = sapien.Pose([0.0, 0.0, 0.30])
                builder.add_box_visual(
                    half_size=half,
                    material=sapien.render.RenderMaterial(base_color=list(color)),
                )
                return builder.build_kinematic(name=name)

            # The cyan bar lies flat on the active slab's top face and is posed in the
            # CORE frame rotated by the layer joint angle each frame: attaching it to the
            # layer link put it inside the slab (its local frame's +z sign is not the
            # outward normal for every face), while this version shows the 90 deg turn
            # unambiguously for the U move.
            created_markers.append(
                (
                    add_marker(
                        "active_layer_marker",
                        (0.013, 0.0028, 0.0016),
                        (0.05, 0.85, 0.95, 1.0),
                    ),
                    "layer_bar",
                    marker_local_z,
                )
            )
            created_markers.append(
                (
                    add_marker(
                        "core_marker",
                        (0.0018, 0.0022, 0.011),
                        (0.02, 0.02, 0.02, 1.0),
                    ),
                    self.core_link,
                    sapien.Pose(
                        [half_size + 0.001, 0.010, 0.0],
                        sapien.Pose(q=[0.0, 0.7071068, 0.0, 0.7071068]).q,
                    ),
                )
            )

    # The base env counts sim SUBSTEPS, and one decision is `decision_period` substeps:
    # registering `max_episode_decisions` directly would truncate after a tenth of the
    # episode (an 80-decision budget ended at decision 8).
    register_env(
        DEMO_ENV_ID,
        max_episode_steps=int(args.max_episode_decisions) * config.decision_period,
    )(FaceTurnSkillsDemo)
    env = gym.make(
        DEMO_ENV_ID,
        num_envs=1,
        obs_mode="state",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        render_mode="rgb_array",
        partner_mode=args.partner_mode,
        skills_config=config,
    )
    obs, infos = env.reset(seed=args.seed)
    base = env.unwrapped
    uids = tuple(env.action_space.spaces)
    builder = SkillsObsBuilder.from_env_obs(
        base.get_obs(unflattened=True), obs, args.partner_mode, uids
    )
    learned = args.policy == "learned"
    if learned:
        if args.checkpoint is None:
            raise SystemExit("--policy learned needs --checkpoint")
        from callosum.experiments.arm import skills_eval

        holder_policy, rotator_policy = skills_eval._load_pair(
            args.checkpoint, args.checkpoint, base.device, builder.obs_dims
        )
        title = "learned skill policies turn a cube face (BC, E1)"
        subtitle = (
            "FaceTurnSkills-v0 - two SO-ARM101 agents, fixed plan U (one face turn); "
            "learned categorical BC policy, argmax actions"
        )
        policy_label = "POLICY (learned, BC)"
    else:
        holder_policy = rotator_policy = None
        title, subtitle = TITLE, SUBTITLE
        policy_label = "SCRIPTED CONDUCTOR (upper bound, not learned)"
    first_move = fixed_plan[0]
    goal = (
        f"turn face {first_move.face} by {90 if first_move.quarter_turns == 1 else 180} deg"
        if n_executable == 1
        else f"execute the plan {args.plan_notation}"
    )
    conductor = SkillsConductor()
    print(
        f"env: {DEMO_ENV_ID} sim_backend={base.backend.sim_backend} "
        f"render_backend={base.backend.render_backend} camera {args.width}x{args.height} "
        f"fov={args.fov:.2f}"
    )
    print(
        f"plan: {args.plan_notation!r} -> {n_executable} executable moves "
        f"({', '.join(move_labels)}) | budget {config.max_episode_decisions} decisions"
    )
    print("SAPIEN render devices:", " | ".join(sapien.render.get_device_summary().splitlines()))

    def _np(x) -> np.ndarray:
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    def _info_at(infos: dict, env: int) -> dict:
        return {
            key: (_np(value).reshape(-1)[env] if hasattr(value, "reshape") else value)
            for key, value in infos.items()
        }

    frames: list[np.ndarray] = []
    metas: list[dict] = []
    render_seconds = 0.0
    flips = 0
    previous_holder = -1
    holder_id = rotator_id = ROTATOR_ID["wait"]
    faces_seen: set = set()

    def capture(decision: int, sim_step: int) -> None:
        """Render the current state and store the frame plus the numbers shown in its HUD."""
        nonlocal render_seconds
        for marker, link, local in created_markers:
            if link == "layer_bar":
                angle = float(base.cube.get_qpos()[0][base.layer_col[fixed_plan[0].face]])
                half = 0.5 * angle
                spin = sapien.Pose(q=[0.0, 0.0, math.sin(half), math.cos(half)])
                marker.set_pose(base.cube.pose * sapien.Pose([0.010, 0.0, local]) * spin)
            else:
                marker.set_pose(link.pose * sapien.Pose(local.p, local.q))
        started = time.time()
        frame = _to_frame(env.render())
        render_seconds += time.time() - started
        info = base.evaluate()
        quat = _np(base.cube.pose.q)[0]
        orientation = orientation_from_quat(sapien_quat_to_xyzw(quat))
        angle_deg = math.degrees(float(_np(info[INFO_ACTIVE_LAYER_ANGLE_DEG])[0]))
        plan_index = int(_np(info[INFO_PLAN_INDEX])[0])
        pos = _np(base.cube.pose.p)[0]
        drift_cm = float(np.linalg.norm(pos - _np(base.body_init_pos)[0])) * 100.0
        target_deg = math.degrees(float(_np(base._move_target)[0]))
        faces_seen.add(orientation.up)
        window = min(plan_index, n_executable - 1)
        metas.append(
            {
                "decision": int(decision),
                "sim_step": int(sim_step),
                "plan_index": plan_index,
                "move_text": move_labels[window],
                "angle_deg": angle_deg,
                "target_deg": target_deg,
                "up_face": orientation.up,
                "faces_seen": len(faces_seen),
                "flips": flips,
                "holder": holder_name.get(int(holder_id), str(holder_id)),
                "rotator": rotator_name.get(int(rotator_id), str(rotator_id)),
                "drift_cm": drift_cm,
                "cube_pos": [float(v) for v in pos],
                "success": bool(_np(info[INFO_SUCCESS])[0]),
            }
        )
        frames.append(frame)

    print(f"running the {'learned' if learned else 'scripted conductor'} policy on the fixed plan")
    success = False
    first_success_decision = None
    sim_step = 0
    capture(0, 0)
    with torch.no_grad():
        for decision in range(int(args.max_episode_decisions)):
            inputs = builder(obs, infos)
            if learned:
                holder_id = int(holder_policy.get_action(inputs[0], deterministic=True)[0])
                rotator_id = int(rotator_policy.get_action(inputs[1], deterministic=True)[0])
            else:
                info_row = _info_at(infos, 0)
                holder_id = conductor.act("holder", inputs[0][0].cpu().numpy(), info_row)
                rotator_id = conductor.act("rotator", inputs[1][0].cpu().numpy(), info_row)
            if holder_id in flip_ids and previous_holder != holder_id:
                flips += 1
            previous_holder = holder_id
            action = {
                uids[0]: torch.as_tensor([holder_id], dtype=torch.long, device=base.device),
                uids[1]: torch.as_tensor([rotator_id], dtype=torch.long, device=base.device),
            }
            obs, _reward, _terminated, truncated, infos = env.step(action)
            sim_step += int(base.skills_config.decision_period)
            success = bool(_np(infos[INFO_SUCCESS])[0]) or success
            if success and first_success_decision is None:
                first_success_decision = decision + 1
            if decision % args.every == 0 or success:
                capture(decision + 1, sim_step)
            if success or bool(_np(truncated)[0]):
                break
    final = metas[-1]
    success = bool(final["success"]) or success
    decisions_used = int(final["decision"])

    hold_frames = max(1, round(args.fps * args.hold_seconds))
    decorated: list[np.ndarray] = []
    for frame, meta in zip(frames, metas, strict=True):
        if final["plan_index"] >= n_executable:
            detail = (
                f"decisions {decisions_used}, flips {meta['flips']}, "
                f"plan completed at target {final['target_deg']:.0f} deg, "
                f"drift {final['drift_cm']:.2f} cm"
            )
        else:
            detail = (
                f"decisions {decisions_used}, flips {meta['flips']}, "
                f"final layer {final['angle_deg']:.0f} deg, drift {final['drift_cm']:.2f} cm"
            )
        dial_fraction = (
            min(meta["angle_deg"] / meta["target_deg"], 1.0)
            if meta["target_deg"] > 1e-6
            else min(meta["plan_index"] / n_executable, 1.0)
        )
        if meta["success"]:
            decorated.append(
                _decorate(
                    frame,
                    meta,
                    plan_notation=args.plan_notation,
                    n_executable=n_executable,
                    total_decisions=int(args.max_episode_decisions),
                    title=title,
                    subtitle=subtitle,
                    policy_label=policy_label,
                    goal=goal,
                    status="SUCCESS",
                    status_color=SUCCESS_COLOR,
                    status_detail=detail,
                )
            )
            decorated[-1] = _draw_dial(decorated[-1], 1.0)
        else:
            decorated.append(
                _decorate(
                    frame,
                    meta,
                    plan_notation=args.plan_notation,
                    n_executable=n_executable,
                    total_decisions=int(args.max_episode_decisions),
                    title=title,
                    subtitle=subtitle,
                    policy_label=policy_label,
                    goal=goal,
                )
            )
            decorated[-1] = _draw_dial(decorated[-1], dial_fraction)
    if success:
        final_frame = decorated[-1]
    else:
        final_frame = _decorate(
            frames[-1],
            final,
            plan_notation=args.plan_notation,
            n_executable=n_executable,
            total_decisions=int(args.max_episode_decisions),
            title=title,
            subtitle=subtitle,
            policy_label=policy_label,
            goal=goal,
            status="INCOMPLETE",
            status_color=FAILURE_COLOR,
            status_detail=(
                f"plan {final['plan_index']}/{n_executable} moves after {decisions_used} "
                f"decisions; layer {final['angle_deg']:.0f} deg, drift {final['drift_cm']:.2f} cm"
            ),
        )
        print("warning: the policy did not reach env success; writing a best-effort clip")
    decorated[-1] = final_frame
    intro_frames = max(0, round(args.fps * args.intro_seconds))
    decorated = [decorated[0]] * intro_frames + decorated
    decorated.extend([final_frame] * hold_frames)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stills = _write_stills(out, decorated, metas, frames)
    summary = {
        "success": success,
        "policy": args.policy,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "plan_notation": args.plan_notation,
        "executable_moves": move_labels,
        "n_executable_moves": n_executable,
        "seed": args.seed,
        "decisions": decisions_used,
        "first_success_decision": first_success_decision,
        "flips": final["flips"],
        "faces_seen": final["faces_seen"],
        "final_plan_index": final["plan_index"],
        # The learned policy imitates the conductor but, once the plan turned terminal,
        # keeps selecting turn primitives, and the env's kinematic turn assist keeps
        # accumulating the layer-joint angle past the target. The DISPLAYED/JSON angle is
        # therefore the plan-relative one (the target once the plan is complete), with the
        # raw accumulated joint reading kept separately for debugging.
        "final_active_layer_angle_deg": (
            final["target_deg"] if final["plan_index"] >= n_executable else final["angle_deg"]
        ),
        "final_raw_accumulated_angle_deg": final["angle_deg"],
        "final_target_deg": final["target_deg"],
        "final_cube_position": final["cube_pos"],
        "final_drift_cm": final["drift_cm"],
        "time_to_success_steps": int(_np(infos.get(INFO_TIME_TO_SUCCESS, 0))[0]),
        "frames": len(decorated),
        "rendered_frames": len(frames),
        "fps": args.fps,
        "every": args.every,
        "duration_s": len(decorated) / args.fps,
        "render_seconds": render_seconds,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "mp4": None,
        "gif": None,
        "stills": stills,
    }
    mp4_name = LEARNED_MP4_NAME if learned else MP4_NAME
    gif_name = LEARNED_GIF_NAME if learned else GIF_NAME
    json_name = LEARNED_JSON_NAME if learned else JSON_NAME
    if not args.stills_only:
        mp4_path = out / mp4_name
        gif_path = out / gif_name
        encoder = _write_h264_mp4(mp4_path, decorated, args.fps)
        gif_count = write_gif(
            gif_path,
            decorated[:: args.gif_every],
            frame_duration_ms=max(1, round(1000.0 * args.gif_every / args.fps)),
            width=args.gif_width,
        )
        summary["mp4"] = {"path": str(mp4_path), "encoder": encoder, **probe_video(mp4_path)}
        summary["gif"] = {"path": str(gif_path), "frames": gif_count, "width": args.gif_width}
    (out / json_name).write_text(json.dumps(summary, indent=2))
    summary["json"] = str(out / json_name)
    env.close()

    print()
    print("=== sexy-move demo summary ===")
    print(f"plan                : {args.plan_notation!r} -> {n_executable} executable moves")
    print(f"success             : {success}")
    print(f"decisions           : {decisions_used}")
    print(f"flips               : {final['flips']}")
    print(f"final plan index    : {final['plan_index']}/{n_executable}")
    shown_angle = final["target_deg"] if final["plan_index"] >= n_executable else final["angle_deg"]
    print(f"final layer angle   : {shown_angle:.1f} deg (target {final['target_deg']:.1f})")
    print(f"final cube drift    : {final['drift_cm']:.2f} cm")
    print(
        f"frames              : {len(decorated)} "
        f"({len(frames)} rendered every {args.every} decisions + {hold_frames} hold)"
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
    print(f"stills              : {', '.join(stills)}")
    print(f"json                : {summary['json']}")
    return summary


def _write_stills(
    out: Path,
    decorated: list[np.ndarray],
    metas: list[dict],
    frames: list[np.ndarray],
) -> list[str]:
    """Save four stills: reset, a flip in progress, a turn in progress, the final state."""
    reset_index = 0
    flip_index = next(
        (i for i, meta in enumerate(metas) if meta["holder"] in ("flip90", "flip180")),
        None,
    )
    turn_index = next(
        (
            i
            for i, meta in enumerate(metas)
            if TURN_STILL_MIN_DEG <= meta["angle_deg"] <= TURN_STILL_MAX_DEG
        ),
        None,
    )
    picks = {
        STILL_NAMES[0]: decorated[reset_index],
        STILL_NAMES[1]: decorated[flip_index if flip_index is not None else len(decorated) - 1],
        STILL_NAMES[2]: decorated[turn_index if turn_index is not None else len(decorated) - 1],
        STILL_NAMES[3]: decorated[-1],
    }
    if flip_index is None or turn_index is None:
        print(
            "warning: could not find a "
            + ", ".join(
                name
                for name, index in (("flip", flip_index), ("turn", turn_index))
                if index is None
            )
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
    args = _parse_args(argv)
    _prefer_software_vulkan()
    summary = _run_demo(args)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
