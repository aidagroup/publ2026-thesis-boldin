"""[cloud] Intent->skills bridge: learned CubeWrench-v0 policy -> FaceTurnSkills-v0 primitives.

Runs the whole bridge in one pod job:

1. **Reduced exemplar** -- a greedy `wrench_ppo` checkpoint is rolled out in the GPU
   `CubeWrench-v0` backend (one world, fixed seed) by
   `callosum.experiments.arm.wrench_bridge`; the trace is classified into `hold`/`turn`
   phases and mapped to a per-role primitive schedule on the arm's decision grid.
2. **Arm execution** -- the schedule is played on `FaceTurnSkills-v0` with the SO-ARM101
   primitives (fixed plan `--plan-notation`, default `"U"` = one 90 deg up-face turn; the
   up face is the only plan move that needs no holder flip, so it isolates the bridge from
   the flip macro): a small
   scripted executor issues `approach_body`/`grasp`/`wait` for the holder and
   `approach_face`/`grasp`/`turn90`/`release` for the rotator, phase by phase, in the order
   the reduced phases dictate. The same episode is also run with the scripted
   `SkillsConductor` as the ENVIRONMENT-HEALTH baseline: if the conductor fails too, the arm
   environment is broken and the result is reported as `BLOCKED` (no invented success).
3. **Measurement** -- `compare_phase_timing` reports the phase-share error, the turn-onset
   offset, the decision/control-step time-scale ratio and both success flags; the JSON
   (`--out`) carries the full reduced trace, the schedule, the arm decision trace and the
   metrics.
4. **Demo** -- with `--render`, a side-by-side clip: the reduced learned-policy episode on
   the left (re-rendered from the recorded trace) and the two SO-ARM101 executing the same
   intentions on the right, with the `LEARNED REDUCED POLICY -> ARM SKILLS` HUD, the current
   phase, the layer/face angle and success.

This is an INTENT->SKILLS bridge, not policy transfer (the two action/observation spaces are
incompatible); the report says so explicitly and lists what is lost in `limitations`.

Single-run examples (pod, from the repo root):

    # reduced trace + schedule only (works before the arm env is healthy)
    /root/.callosum/venv/bin/python scripts/experiments/arm/bridge_wrench_to_skills.py \
      --checkpoint /workspace/callosum-runs/wrench_gpu_default_none_s0/latest.pt \
      --reduced-only --out /workspace/callosum-runs/bridge/bridge_metrics.json

    # full bridge + side-by-side demo (needs the skills env + lavapipe for the arm render)
    /root/.callosum/venv/bin/python scripts/experiments/arm/bridge_wrench_to_skills.py \
      --checkpoint /workspace/callosum-runs/wrench_gpu_default_none_s0/latest.pt \
      --render --sim-backend cpu --render-backend cpu \
      --out /workspace/callosum-runs/bridge/bridge_metrics.json

Never run on the Mac: the arm part needs the simulator and the reduced rollout needs CUDA.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

import numpy as np

from callosum.experiments.arm import wrench_bridge as bridge
from callosum.experiments.arm.skills import (
    HOLDER_PRIMITIVES,
    ROTATOR_PRIMITIVES,
    RandomizationConfig,
    Role,
    num_primitives,
)
from callosum.experiments.arm.skills_obs import (
    INFO_ACTIVE_LAYER_ANGLE_DEG,
    INFO_PLAN_INDEX,
    INFO_ROTATOR_GRASP_FACT,
    INFO_SUCCESS,
    own_block_slices,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = "/workspace/callosum-runs/wrench_gpu_default_none_s0/latest.pt"
DEFAULT_OUT = "runs/bridge/bridge_metrics.json"
LAVAPIPE_ICD = Path("/usr/share/vulkan/icd.d/lvp_icd.json")
REDUCED_SCENE = "callosum/experiments/cube_wrench/assets/cube_wrench.xml"
"""The reduced MJCF after the experiment-folder refactor (`render_utils.SCENE_XML` is stale)."""

DEFAULT_ARM_EYE = (0.20, -0.20, 0.155)
"""Close-up camera eye of the arm half (m): the cube fills ~1/3 of the frame width and both
SO-ARM101 end effectors stay in frame (picked from a framing probe)."""

DEFAULT_ARM_TARGET = (0.0, 0.0, 0.028)
"""Close-up camera target (m): the cube centre."""

DEFAULT_ARM_FOV = 0.85
"""Close-up camera vertical fov (rad)."""

REDUCED_CAM_DISTANCE = 0.125
"""Reduced-half camera distance (m): closer than `render_utils.CAM_DISTANCE` so the layer and
the 90 deg turn read large on the left half (patched in the child render process)."""

HOLDER_ID = {primitive.name: primitive.index for primitive in HOLDER_PRIMITIVES}
ROTATOR_ID = {primitive.name: primitive.index for primitive in ROTATOR_PRIMITIVES}
HOLDER_NAME = {primitive.index: primitive.name for primitive in HOLDER_PRIMITIVES}
ROTATOR_NAME = {primitive.index: primitive.name for primitive in ROTATOR_PRIMITIVES}

_HOLDING_OPENING_MIN = 0.1
_HOLDING_OPENING_MAX = 0.9
"""Jaw-opening band of a real clamp (the conductor's convention: open = 1.0, on air = 0.0)."""

_PHASE_TIMEOUT = {"idle": 60, "hold": 60, "turn": 80, "release": 16}
"""Per-phase decision caps of the schedule executor (the episode budget also applies)."""


# --- CLI ------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bridge CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Reduced latest.pt.")
    parser.add_argument(
        "--input-mode",
        choices=("none", "oracle", "predicted"),
        default=None,
        help="Reduced policy input mode (default: the checkpoint's saved mode).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Reduced episode seed.")
    parser.add_argument(
        "--sweep-reduced",
        default="",
        help=(
            "Comma-separated reduced seeds to probe before the report (prints each seed's "
            "phase segments; the report still uses --seed). Empty = no sweep."
        ),
    )
    parser.add_argument("--device", default="cuda", help="CUDA device of the reduced rollout.")
    parser.add_argument("--max-steps", type=int, default=None, help="Reduced step cap.")
    parser.add_argument(
        "--timing-mode",
        choices=("share", "dt"),
        default="share",
        help="Schedule timing allocation (default: phase-duration shares).",
    )
    parser.add_argument(
        "--arm-total-decisions",
        type=int,
        default=40,
        help="Arm decision budget of the reduced phases in the share mode.",
    )
    parser.add_argument(
        "--plan-notation",
        default="U",
        help=(
            "Fixed arm plan in face-turn notation (default: 'U' = one 90 deg up-face turn). "
            "'U' needs no holder flip, so it isolates the intent->skills bridge; a non-up "
            "face (e.g. 'R') also needs the holder's flip, which is measured separately."
        ),
    )
    parser.add_argument("--arm-seed", type=int, default=0, help="Arm episode seed.")
    parser.add_argument("--sim-backend", default="gpu", help="ManiSkill sim backend.")
    parser.add_argument("--render-backend", default="none", help="ManiSkill render backend.")
    parser.add_argument(
        "--max-episode-decisions",
        type=int,
        default=150,
        help="Arm episode decision budget.",
    )
    parser.add_argument(
        "--arm-randomization",
        action="store_true",
        help="Keep the skills env's default episode randomness (default: deterministic).",
    )
    parser.add_argument(
        "--no-conductor",
        action="store_true",
        help="Skip the scripted-conductor environment-health baseline.",
    )
    parser.add_argument(
        "--reduced-only",
        action="store_true",
        help="Only the reduced trace + schedule (no arm env; works when the arm env is down).",
    )
    parser.add_argument("--render", action="store_true", help="Write the side-by-side demo clip.")
    parser.add_argument("--width", type=int, default=640, help="Render width per half.")
    parser.add_argument("--height", type=int, default=480, help="Render height per half.")
    parser.add_argument("--fps", type=float, default=10.0, help="Demo clip fps.")
    parser.add_argument(
        "--arm-eye",
        type=float,
        nargs=3,
        default=list(DEFAULT_ARM_EYE),
        metavar=("X", "Y", "Z"),
        help="Close-up camera eye of the arm half (m).",
    )
    parser.add_argument(
        "--arm-target",
        type=float,
        nargs=3,
        default=list(DEFAULT_ARM_TARGET),
        metavar=("X", "Y", "Z"),
        help="Close-up camera target of the arm half (m).",
    )
    parser.add_argument("--arm-fov", type=float, default=DEFAULT_ARM_FOV, help="Arm camera fov.")
    parser.add_argument(
        "--slowmo",
        type=int,
        default=6,
        help="Interpolated frames per captured frame (6 -> ~19 s slow-mo for 32 frames).",
    )
    parser.add_argument(
        "--arm-crop",
        type=float,
        default=0.85,
        help="Central crop of the arm half before upscaling (close-up cheat, 1.0 = off).",
    )
    parser.add_argument(
        "--no-interpolate",
        action="store_true",
        help="Skip the ffmpeg motion interpolation of the slow-mo clip (accept choppiness).",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render every N arm decisions (reduced frames are per control step).",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="Bridge metrics JSON path.")
    parser.add_argument(
        "--render-reduced-trace",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--frames-dir", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.render_reduced_trace:
        # Internal child-process mode: MuJoCo's EGL renderer fails to load GL in the parent
        # process once the CUDA/warp stack is initialised, so the reduced demo half is
        # rendered in a fresh process (see `render_reduced_frames`).
        return args
    if args.arm_total_decisions < 1:
        parser.error("--arm-total-decisions must be >= 1")
    if not args.reduced_only and args.render and args.render_backend == "none":
        parser.error("--render needs a render backend (e.g. --render-backend cpu on lavapipe)")
    if args.reduced_only and args.render:
        parser.error("--reduced-only cannot render the arm half")
    return args


# --- reduced part ---------------------------------------------------------------------------


def run_reduced(args: argparse.Namespace) -> tuple[bridge.ReducedRollout, list]:
    """Load the learned reduced policy, roll it out and build the primitive schedule."""
    policy = bridge.load_reduced_policy(args.checkpoint, args.input_mode, device=args.device)
    if args.sweep_reduced:
        _sweep_reduced(policy, args)
    rollout = bridge.rollout_reduced(
        policy,
        bridge.RolloutConfig(seed=args.seed, device=args.device, max_steps=args.max_steps),
    )
    schedule = bridge.build_primitive_schedule(
        rollout.segments,
        timing_mode=args.timing_mode,
        total_decisions=args.arm_total_decisions,
    )
    _print_reduced(rollout, schedule)
    return rollout, schedule


def _sweep_reduced(policy: bridge.ReducedPolicy, args: argparse.Namespace) -> None:
    """Probe candidate reduced seeds and print each episode's phase segmentation.

    The bridge needs an exemplar that actually contains the phases it bridges (a `hold`
    before the `turn`); the sweep makes the choice of `--seed` evidence-based instead of
    assuming seed 0 shows both.
    """
    for token in str(args.sweep_reduced).split(","):
        token = token.strip()
        if not token:
            continue
        seed = int(token)
        rollout = bridge.rollout_reduced(
            policy,
            bridge.RolloutConfig(seed=seed, device=args.device, max_steps=args.max_steps),
        )
        segments = " -> ".join(f"{segment.phase}({segment.steps})" for segment in rollout.segments)
        print(
            f"sweep seed {seed:8d}: success {rollout.success} | terminal {rollout.terminal} | "
            f"steps {len(rollout.steps):3d} | angle {rollout.final_angle_deg:5.1f} deg | "
            f"segments {segments}"
        )


def _print_reduced(rollout: bridge.ReducedRollout, schedule: list) -> None:
    """Print the reduced exemplar and the derived schedule compactly to the job log."""
    print(
        f"reduced: {rollout.checkpoint} | mode {rollout.input_mode} | seed {rollout.seed} | "
        f"backend {rollout.backend} | success {rollout.success} | terminal {rollout.terminal} | "
        f"steps {len(rollout.steps)} ({rollout.duration_s:.2f} s)"
    )
    print(
        f"reduced final: angle {rollout.final_angle_deg:.1f} deg | "
        f"drift {rollout.final_drift_pos_m * 1e3:.1f} mm / {rollout.final_drift_rot_deg:.1f} deg"
    )
    for segment in rollout.segments:
        print(
            f"  segment {segment.phase:7s} t {segment.t_start:3d}..{segment.t_end:3d} "
            f"steps {segment.steps:3d}  {segment.duration_s:.2f} s"
        )
    for step in rollout.steps:
        print(
            f"  t={step.t:3d} {step.phase:7s} angle {step.layer_angle_deg:6.1f} deg "
            f"|Fh| {step.holder_force_norm_n:5.3f} N |Th| {step.holder_torque_norm_nm:7.4f} Nm "
            f"tau_z {step.rotator_tau_z_nm:7.4f} Nm -> "
            f"{step.intended_holder}/{step.intended_rotator}"
        )
    print(f"schedule: 1 arm decision = {bridge.ARM_DECISION_SECONDS:.2f} s")
    for segment in schedule:
        print(
            f"  {segment.phase:7s} {segment.decisions:3d} decisions | "
            f"holder {segment.holder:13s} rotator {segment.rotator:13s} [{segment.source}]"
        )


# --- arm part: environment ------------------------------------------------------------------


def _prefer_software_vulkan() -> None:
    """Point SAPIEN at the pod's Mesa lavapipe ICD when the caller has not picked one."""
    if os.environ.get("VK_ICD_FILENAMES") or os.environ.get("VK_DRIVER_FILES"):
        return
    if LAVAPIPE_ICD.is_file():
        os.environ["VK_ICD_FILENAMES"] = str(LAVAPIPE_ICD)


def _arm_config(args: argparse.Namespace):
    """The `FaceTurnSkillsConfig` of the bridge run (fixed plan, optional determinism)."""
    from callosum.experiments.arm.skills_config import FaceTurnSkillsConfig

    config = FaceTurnSkillsConfig(
        plan_notation=args.plan_notation,
        max_episode_decisions=int(args.max_episode_decisions),
    )
    if not args.arm_randomization:
        # Deterministic episode: the bridge measures the intent->skills mechanics, so the
        # private randomness (start delays, grasp failures, slips) is switched off; a run
        # with --arm-randomization is the noisy variant reported next to it.
        config = dataclasses.replace(
            config,
            randomization=RandomizationConfig(
                episode_speed_range=(1.0, 1.0),
                arm_speed_range=(1.0, 1.0),
                grasp_failure_prob_range=(0.0, 0.0),
                start_delay_range=(0.0, 0.0),
            ),
            slip_prob_range=(0.0, 0.0),
        )
    return config


def _demo_camera(args: argparse.Namespace):
    """The one close-up camera both SO-ARM101 arms are filmed with."""
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils

    return CameraConfig(
        uid="bridge_demo_camera",
        pose=sapien_utils.look_at(tuple(args.arm_eye), tuple(args.arm_target)),
        width=int(args.width),
        height=int(args.height),
        fov=float(args.arm_fov),
        near=0.01,
        far=10.0,
        shader_pack="minimal",
    )


def build_arm_env(args: argparse.Namespace, *, with_camera: bool):
    """Build the (optionally camera-equipped) `FaceTurnSkills-v0` env and its base env."""
    _prefer_software_vulkan()
    import gymnasium as gym

    import callosum.experiments.arm.skills_env  # noqa: F401  (registers FaceTurnSkills-v0)
    from callosum.experiments.arm.skills_env import FaceTurnSkills

    config = _arm_config(args)
    env_id = "FaceTurnSkills-v0"
    render_mode = None
    render_backend = "none"
    if with_camera:
        import sapien
        import sapien.render
        from mani_skill.utils.registration import register_env

        camera = _demo_camera(args)
        active_face = config.fixed_plan[0].face if config.fixed_plan else "U"

        def add_marker(scene, name: str, half_size, color):
            """A kinematic, collision-free box: a render-only visual reference."""
            builder = scene.create_actor_builder()
            builder.initial_pose = sapien.Pose([0.0, 0.0, 0.05])
            builder.add_box_visual(
                half_size=half_size,
                material=sapien.render.RenderMaterial(base_color=list(color)),
            )
            return builder.build_kinematic(name=name)

        class _BridgeDemoEnv(FaceTurnSkills):
            """`FaceTurnSkills-v0` + close-up camera + render-only orientation markers.

            The multi-face cube's six full-face slabs overlap, so a raw render reads as a
            pile of coloured blocks. This demo-env subclass (RENDER ONLY, physics untouched)
            hides the five slabs that are not the plan's active layer and adds two kinematic
            markers: a cyan bar ON the active layer (it turns with it) and a dark bar on the
            core (the static reference). A 90 deg turn then reads as the cyan bar swinging a
            quarter turn against the dark one.
            """

            @property
            def _default_human_render_camera_configs(self):
                return [camera]

            def _load_scene(self, options: dict):
                super()._load_scene(options)
                self.layer_marker = add_marker(
                    self.scene,
                    "layer_marker",
                    (0.016, 0.0026, 0.0012),
                    (0.05, 0.95, 1.0, 1.0),
                )
                self.layer_marker2 = add_marker(
                    self.scene,
                    "layer_marker2",
                    (0.008, 0.0026, 0.0012),
                    (0.95, 0.10, 0.85, 1.0),
                )
                self.core_marker = add_marker(
                    self.scene,
                    "core_marker",
                    (0.006, 0.0008, 0.0035),
                    (0.02, 0.02, 0.02, 1.0),
                )
                self.table_marker = add_marker(
                    self.scene,
                    "table_marker",
                    (0.020, 0.0018, 0.0008),
                    (1.0, 0.85, 0.05, 1.0),
                )
                self.table_marker.set_pose(sapien.Pose([0.0, 0.055, 0.001], [1.0, 0.0, 0.0, 0.0]))
                for link_name, link in self.cube.links_map.items():
                    if not link_name.startswith("layer_") or link_name == f"layer_{active_face}":
                        continue
                    body = link._objs[0].entity.find_component_by_type(
                        sapien.render.RenderBodyComponent
                    )
                    if body is not None:
                        body.visibility = 0.0

        env_id = "FaceTurnSkillsBridgeDemo-v0"
        register_env(
            env_id, max_episode_steps=config.max_episode_decisions * config.decision_period
        )(_BridgeDemoEnv)
        render_mode = "rgb_array"
        render_backend = args.render_backend
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state",
        sim_backend=args.sim_backend,
        render_backend=render_backend,
        render_mode=render_mode,
        partner_mode="none",
        skills_config=config,
    )
    return env, env.unwrapped, config


# --- arm part: shared fact reading ----------------------------------------------------------


@dataclasses.dataclass
class _Facts:
    """The per-decision facts the executor reads (both hands' own blocks + env info)."""

    holder_primitive: str
    rotator_primitive: str
    holder_progress: float
    rotator_progress: float
    holder_opening: float
    rotator_grasp: bool
    angle_deg: float
    plan_index: int
    success: bool
    holder_grasp_env: bool | None = None
    """The env's authoritative holder grasp fact (`None` when the env does not expose it).

    The jaw-opening band is the fallback only: on the articulated multi-face cube both jaws
    can press different links, so the env's own multi-link check
    (`FaceTurnSkills._holder_holds`, the same one that releases the layer lock and gates
    the turn assist) is the honest definition of "the holder holds the core".
    """

    @property
    def holder_holding(self) -> bool:
        """Whether the holder clamps the core: the env's gripper fact, else the jaw band."""
        if self.holder_grasp_env is not None:
            return bool(self.holder_grasp_env)
        return _HOLDING_OPENING_MIN < self.holder_opening <= _HOLDING_OPENING_MAX


def _np(value) -> np.ndarray:
    """Tensor/numpy/array value as a numpy array on the host."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _info_at(infos: dict, env_index: int = 0):
    """One env's row of the batched info dict."""
    row = {}
    for key, value in infos.items():
        array = _np(value).reshape(-1)
        row[key] = array[env_index] if array.size else value
    return row


def _own_facts(np_inputs, slices: dict, index: int) -> tuple[np.ndarray, float, float]:
    """`(current primitive one-hot, progress, gripper opening)` of one hand's own block."""
    vector = np.asarray(_np(np_inputs[index]), dtype=float).reshape(-1)
    return (
        vector[slices["own_primitive_onehot"]],
        float(vector[slices["own_primitive_progress"]][0]),
        float(vector[slices["own_gripper_opening"]][0]),
    )


def _holder_grasp_fact(base) -> bool | None:
    """The env's authoritative holder-grasp fact, or `None` if the env has no such hook.

    `FaceTurnSkills._holder_holds` is the multi-link check that releases the layer lock and
    gates the turn assist; the bridge executor reads it through this single feature-detected
    call so the measurement uses exactly the env's own definition of a holder grasp. Older
    env revisions without the hook fall back to the jaw-opening band.
    """
    hook = getattr(base, "_holder_holds", None)
    if hook is None:
        return None
    return bool(_np(hook()).reshape(-1)[0])


def _read_facts(
    np_inputs, infos: dict, holder_slices: dict, rotator_slices: dict, holder_grasp_env
) -> _Facts:
    """Collect one decision's facts from the policy inputs, the env info and the grasp fact."""
    holder_onehot, holder_progress, holder_opening = _own_facts(np_inputs, holder_slices, 0)
    rotator_onehot, rotator_progress, _ = _own_facts(np_inputs, rotator_slices, 1)
    info = _info_at(infos)
    return _Facts(
        holder_primitive=HOLDER_NAME[int(np.argmax(holder_onehot))],
        rotator_primitive=ROTATOR_NAME[int(np.argmax(rotator_onehot))],
        holder_progress=holder_progress,
        rotator_progress=rotator_progress,
        holder_opening=holder_opening,
        rotator_grasp=bool(info.get(INFO_ROTATOR_GRASP_FACT, False)),
        angle_deg=float(info[INFO_ACTIVE_LAYER_ANGLE_DEG]),
        plan_index=int(info[INFO_PLAN_INDEX]),
        success=bool(info[INFO_SUCCESS]),
        holder_grasp_env=holder_grasp_env,
    )


def _to_frame(image) -> np.ndarray:
    """Turn one `env.render()` result (`(num_envs,) H W C`) into an RGB uint8 frame."""
    array = _np(image)
    if array.ndim == 4:
        array = array[0]
    return np.ascontiguousarray(array[..., :3].astype(np.uint8))


# --- arm part: render-only demo markers -----------------------------------------------------


_MARKER_CACHE: dict = {}


def _marker_geometry(base) -> tuple[float, float, float]:
    """`(cube_half_size, layer_gap, core_half)` of the skills cube, cached from the env."""
    if "geometry" not in _MARKER_CACHE:
        cfg = base.cube_config
        gap = float(cfg.layer_gap)
        _MARKER_CACHE["geometry"] = (
            float(cfg.cube_half_size),
            gap,
            float(cfg.cube_half_size) - float(cfg.layer_thickness) - gap / 2,
        )
    return _MARKER_CACHE["geometry"]


def sapien_pose(position, quaternion):
    """A `sapien.Pose` (kept behind a helper so this module imports without SAPIEN)."""
    import sapien

    return sapien.Pose(list(position), list(quaternion))


def place_bridge_markers(base, angle_deg: float) -> None:
    """Position the render-only markers: cyan bar on the active layer, dark bar on the core.

    The markers are kinematic, collision-free boxes created by the demo env (see
    `build_arm_env`); this function only places them. The layer bar rides the active layer
    in the CUBE frame (rotation about the face normal = the joint axis of the plan's U move),
    so a 90 deg turn reads as the cyan bar swinging a quarter turn against the static dark
    core bar. Purely visual: no physics value is read or written.
    """
    if not (hasattr(base, "layer_marker") and hasattr(base, "core_marker")):
        return
    import math

    from mani_skill.utils.structs.pose import Pose as MSPose

    hs, gap, core_half = _marker_geometry(base)
    half = math.radians(float(angle_deg)) / 2
    rot = MSPose.create(sapien_pose([0.0, 0.0, 0.0], [math.cos(half), 0.0, 0.0, math.sin(half)]))
    top = hs + gap / 2 + 0.0012
    layer_local = MSPose.create(sapien_pose([0.8 * hs, 0.0, top], [1.0, 0.0, 0.0, 0.0]))
    layer_local2 = MSPose.create(sapien_pose([0.0, 0.8 * hs, top], [1.0, 0.0, 0.0, 0.0]))
    core_local = MSPose.create(sapien_pose([0.0, -(core_half + 0.0012), 0.0], [1.0, 0.0, 0.0, 0.0]))
    base.layer_marker.set_pose(base.cube.pose * rot * layer_local)
    base.layer_marker2.set_pose(base.cube.pose * rot * layer_local2)
    base.core_marker.set_pose(base.cube.pose * core_local)


# --- arm part: the schedule executor --------------------------------------------------------


@dataclasses.dataclass
class _ArmRun:
    """The realized outcome of one arm execution (schedule replay or conductor)."""

    decisions: list
    success: bool
    final_angle_deg: float
    final_plan_index: int
    timed_out: bool
    frames: list
    frame_decisions: list
    kind: str = "schedule"

    def summary(self, config) -> dict:
        """The JSON block of this arm execution (without the frame arrays)."""
        return {
            "kind": self.kind,
            "plan": config.plan_notation,
            "plan_moves": config.plan_length,
            "seed": None,
            "randomization": None,
            "success": self.success,
            "timed_out": self.timed_out,
            "decisions": len(self.decisions),
            "final_angle_deg": self.final_angle_deg,
            "final_plan_index": self.final_plan_index,
            "phase_segments": [
                segment.to_json() for segment in bridge.arm_phase_segments(self.decisions)
            ],
            "trace": [decision.to_json() for decision in self.decisions],
        }


def _finished(current: str, expected: str, progress: float) -> bool:
    """Whether the expected primitive is the running one and has completed."""
    return current == expected and progress >= 0.999


def _engage_choice(memory: dict, facts: _Facts) -> tuple[str, str, bool]:
    """Primitive pair of the `idle` (engagement) phase; also reports when it is done."""
    holder_stage = memory.setdefault("holder_stage", "approach")
    rotator_stage = memory.setdefault("rotator_stage", "wait")
    if holder_stage == "approach" and _finished(
        facts.holder_primitive, "approach_body", facts.holder_progress
    ):
        holder_stage = memory["holder_stage"] = "grasp"
    if holder_stage == "grasp" and facts.holder_holding:
        holder_stage = memory["holder_stage"] = "done"
    holder = {"approach": "approach_body", "grasp": "grasp", "done": "wait"}[holder_stage]
    rotator = "wait"
    if holder_stage == "done":
        if rotator_stage == "wait":
            rotator_stage = memory["rotator_stage"] = "approach"
        if rotator_stage == "approach" and _finished(
            facts.rotator_primitive, "approach_face", facts.rotator_progress
        ):
            rotator_stage = memory["rotator_stage"] = "grasp"
        if rotator_stage == "grasp" and facts.rotator_grasp:
            rotator_stage = memory["rotator_stage"] = "done"
        rotator = {
            "wait": "wait",
            "approach": "approach_face",
            "grasp": "grasp",
            "done": "wait",
        }[rotator_stage]
    return holder, rotator, holder_stage == "done" and rotator_stage == "done"


def _hold_choice(memory: dict, facts: _Facts) -> tuple[str, str, bool]:
    """Primitive pair of the mapped `hold` phase; done once both hands are engaged.

    If the engagement did not complete in the `idle` phase (an arm leg ran longer than the
    phase cap), the hold phase keeps driving the same approach/grasp FSM instead of
    asserting a hold that does not exist -- a phase label must never fabricate a grasp.
    """
    if not (facts.holder_holding and facts.rotator_grasp):
        return _engage_choice(memory, facts)
    return "wait", "wait", True


def _turn_choice(facts: _Facts, target_deg: float, tol_deg: float) -> tuple[str, str, bool]:
    """Primitive pair of the mapped `turn` phase; done at the planned move count/angle."""
    holder = "wait" if facts.holder_holding else "grasp"
    done = facts.plan_index >= 1 or facts.success or abs(facts.angle_deg - target_deg) <= tol_deg
    if done:
        return holder, "wait", True
    return holder, "turn90", False


def _release_choice(memory: dict, facts: _Facts) -> tuple[str, str, bool]:
    """Primitive pair of the terminating `release` phase (release, then retreat)."""
    stage = memory.setdefault("release_stage", "release")
    memory["release_decisions"] = memory.get("release_decisions", 0) + 1
    if stage == "release" and memory["release_decisions"] >= 2:
        stage = memory["release_stage"] = "retreat"
        memory["release_decisions"] = 0
    holder = "release" if stage == "release" else "retreat"
    retreat_done = stage == "retreat" and (
        _finished(facts.holder_primitive, "retreat", facts.holder_progress)
        or memory["release_decisions"] >= 2
    )
    return holder, holder, retreat_done


def execute_schedule(
    env,
    base,
    args: argparse.Namespace,
    schedule: list,
    *,
    render: bool,
    render_every: int = 1,
) -> _ArmRun:
    """Play the reduced-derived primitive schedule on the arm env and record the trace.

    The executor is a small scripted driver (the arm side's own primitive durations decide
    how long a primitive runs): the schedule supplies the phase ORDER and each phase's
    primitive pair, and a phase advances when its terminal condition holds -- the holder
    clamps the core, the rotator's grasp fact is established, the plan cursor advances, the
    release/retreat sequence is finished. A per-phase decision cap prevents an injection from
    hanging the run; an aborted phase stays in the trace (no success is fabricated).
    """
    import torch

    from callosum.experiments.arm.skills_agent_obs import SkillsObsBuilder

    config = base.skills_config
    uids = tuple(env.action_space.spaces)
    obs, infos = env.reset(seed=args.arm_seed)
    builder = SkillsObsBuilder.from_env_obs(base.get_obs(unflattened=True), obs, "none", uids)
    holder_slices = own_block_slices(num_primitives(Role.HOLDER))
    rotator_slices = own_block_slices(num_primitives(Role.ROTATOR))
    fixed_plan = config.fixed_plan
    target_deg = 90.0 * (fixed_plan[0].quarter_turns if fixed_plan else 1)
    tol_deg = config.angle_tol_deg

    decisions: list[bridge.ArmDecision] = []
    frames: list[np.ndarray] = []
    frame_decisions: list[int] = []
    memory: dict = {}
    print("arm decisions:")
    segment_index = 0
    phase_clock = 0
    previous_angle = float(_info_at(infos)[INFO_ACTIVE_LAYER_ANGLE_DEG])
    timed_out = False
    done = False
    while not done and segment_index < len(schedule):
        facts = _read_facts(
            builder(obs), infos, holder_slices, rotator_slices, _holder_grasp_fact(base)
        )
        segment = schedule[segment_index]
        if segment.phase == "idle":
            holder, rotator, finished = _engage_choice(memory, facts)
        elif segment.phase == "hold":
            holder, rotator, finished = _hold_choice(memory, facts)
        elif segment.phase == "turn":
            holder, rotator, finished = _turn_choice(facts, target_deg, tol_deg)
        else:
            holder, rotator, finished = _release_choice(memory, facts)
        action = {
            uids[0]: torch.as_tensor([HOLDER_ID[holder]], dtype=torch.long, device=base.device),
            uids[1]: torch.as_tensor([ROTATOR_ID[rotator]], dtype=torch.long, device=base.device),
        }
        obs, _reward, terminated, truncated, infos = env.step(action)
        info = _info_at(infos)
        angle = float(info[INFO_ACTIVE_LAYER_ANGLE_DEG])
        decisions.append(
            bridge.ArmDecision(
                decision=len(decisions),
                holder=holder,
                rotator=rotator,
                layer_angle_deg=angle,
                angle_rise_deg=angle - previous_angle,
                holder_holding=facts.holder_holding,
                rotator_grasp=facts.rotator_grasp,
                plan_index=int(info[INFO_PLAN_INDEX]),
                success=bool(info[INFO_SUCCESS]),
                holder_opening=facts.holder_opening,
            )
        )
        print(
            f"  d={decisions[-1].decision:3d} {segment.phase:7s} "
            f"holder {holder:13s} rotator {rotator:13s} angle {angle:6.2f} deg "
            f"hold {facts.holder_holding} grasp {facts.rotator_grasp} "
            f"opening {facts.holder_opening:.3f} plan {int(info[INFO_PLAN_INDEX])}"
        )
        previous_angle = angle
        phase_clock += 1
        if render and (len(decisions) - 1) % max(render_every, 1) == 0:
            place_bridge_markers(base, angle)
            frames.append(_to_frame(env.render()))
            frame_decisions.append(len(decisions) - 1)
        done = bool(_np(terminated).reshape(-1)[0] or _np(truncated).reshape(-1)[0])
        if finished or done:
            segment_index += 1
            phase_clock = 0
            memory = {}
        elif phase_clock > _PHASE_TIMEOUT.get(segment.phase, 60):
            timed_out = True
            segment_index += 1
            phase_clock = 0
            memory = {}
    final_info = _info_at(infos) if infos else {}
    success = bool(_np(infos[INFO_SUCCESS]).reshape(-1)[0]) if infos else False
    bridge.classify_arm_trace(decisions)
    return _ArmRun(
        decisions=decisions,
        success=success,
        final_angle_deg=float(final_info.get(INFO_ACTIVE_LAYER_ANGLE_DEG, 0.0)),
        final_plan_index=int(final_info.get(INFO_PLAN_INDEX, 0)),
        timed_out=timed_out,
        frames=frames,
        frame_decisions=frame_decisions,
        kind="schedule-replay",
    )


# --- arm part: the conductor environment-health baseline ------------------------------------


def run_conductor_baseline(
    env, base, args: argparse.Namespace, *, render: bool = False, render_every: int = 1
) -> _ArmRun:
    """Run the scripted conductor on the same fixed plan; record its realized trace.

    The conductor is the project's skill-level upper bound (the environment validation), so
    on the arm side it is the EXECUTOR of the reduced-derived intent sequence: it issues the
    same primitive families (`approach_body`/`grasp`/`wait` for the holder,
    `approach_face`/`grasp`/`turn90`/`release` for the rotator) and recovers from grasp loss
    and slips. If it also fails, the arm environment or the plan is broken -> `BLOCKED`; no
    success is fabricated.
    """
    import torch

    from callosum.experiments.arm.skills_agent_obs import SkillsObsBuilder
    from callosum.experiments.arm.skills_conductor import SkillsConductor

    uids = tuple(env.action_space.spaces)
    obs, infos = env.reset(seed=args.arm_seed)
    builder = SkillsObsBuilder.from_env_obs(base.get_obs(unflattened=True), obs, "oracle", uids)
    holder_slices = own_block_slices(num_primitives(Role.HOLDER))
    rotator_slices = own_block_slices(num_primitives(Role.ROTATOR))
    conductor = SkillsConductor()
    decisions: list[bridge.ArmDecision] = []
    frames: list[np.ndarray] = []
    frame_decisions: list[int] = []
    previous_angle = float(_info_at(infos)[INFO_ACTIVE_LAYER_ANGLE_DEG])
    done = False
    print("conductor decisions:")
    while not done and len(decisions) < int(args.max_episode_decisions):
        info_before = _info_at(infos)
        inputs = builder(obs, infos)
        facts = _read_facts(inputs, infos, holder_slices, rotator_slices, _holder_grasp_fact(base))
        holder_id = conductor.act("holder", _np(inputs[0][0]), info_before)
        rotator_id = conductor.act("rotator", _np(inputs[1][0]), info_before)
        action = {
            uids[0]: torch.as_tensor([holder_id], dtype=torch.long, device=base.device),
            uids[1]: torch.as_tensor([rotator_id], dtype=torch.long, device=base.device),
        }
        obs, _reward, terminated, truncated, infos = env.step(action)
        info = _info_at(infos)
        angle = float(info[INFO_ACTIVE_LAYER_ANGLE_DEG])
        decisions.append(
            bridge.ArmDecision(
                decision=len(decisions),
                holder=HOLDER_NAME[int(holder_id)],
                rotator=ROTATOR_NAME[int(rotator_id)],
                layer_angle_deg=angle,
                angle_rise_deg=angle - previous_angle,
                holder_holding=facts.holder_holding,
                rotator_grasp=facts.rotator_grasp,
                plan_index=int(info[INFO_PLAN_INDEX]),
                success=bool(info[INFO_SUCCESS]),
                holder_opening=facts.holder_opening,
            )
        )
        print(
            f"  d={decisions[-1].decision:3d} holder {decisions[-1].holder:13s} "
            f"rotator {decisions[-1].rotator:13s} angle {angle:6.2f} deg "
            f"hold {facts.holder_holding} grasp {facts.rotator_grasp} "
            f"plan {int(info[INFO_PLAN_INDEX])}"
        )
        previous_angle = angle
        if render and (len(decisions) - 1) % max(render_every, 1) == 0:
            place_bridge_markers(base, angle)
            frames.append(_to_frame(env.render()))
            frame_decisions.append(len(decisions) - 1)
        done = bool(_np(terminated).reshape(-1)[0] or _np(truncated).reshape(-1)[0])
    final_info = _info_at(infos)
    bridge.classify_arm_trace(decisions)
    return _ArmRun(
        decisions=decisions,
        success=bool(_np(infos[INFO_SUCCESS]).reshape(-1)[0]),
        final_angle_deg=float(final_info.get(INFO_ACTIVE_LAYER_ANGLE_DEG, 0.0)),
        final_plan_index=int(final_info.get(INFO_PLAN_INDEX, 0)),
        timed_out=False,
        frames=frames,
        frame_decisions=frame_decisions,
        kind="conductor",
    )


# --- demo rendering -------------------------------------------------------------------------


def _patch_render_scene_path() -> None:
    """Point the shared renderer at the post-refactor reduced MJCF location.

    `callosum.common.render_utils` still carries the pre-refactor scene path (and its
    `repo_root` helper returns the package directory), so the reduced render of the demo
    resolves the MJCF explicitly here instead of editing the shared module.
    """
    from callosum.common import render_utils

    candidate = REPO_ROOT / REDUCED_SCENE
    if not candidate.is_file():
        raise RuntimeError(f"reduced scene not found at {candidate}")
    render_utils.SCENE_XML = Path(os.path.relpath(candidate, render_utils.repo_root()))


def _render_trace_payload(rollout: bridge.ReducedRollout) -> dict:
    """The per-step state + camera setup the reduced re-render needs (JSON-safe)."""
    return {
        "camera_distance": REDUCED_CAM_DISTANCE,
        "frames": [
            {
                "core_pos": list(step.core_pos),
                "core_quat": list(step.core_quat),
                "layer_angle": step.layer_angle_rad,
                "layer_ang_vel": step.layer_ang_vel,
            }
            for step in rollout.steps
        ],
    }


def _render_trace_child(trace_path: Path, frames_dir: Path, width: int, height: int) -> int:
    """Child mode: render the reduced trace with a FRESH MuJoCo EGL context; save PNGs."""
    import json

    os.environ.setdefault("MUJOCO_GL", "egl")
    _patch_render_scene_path()
    from PIL import Image

    from callosum.common import render_utils
    from callosum.common.render_utils import DemoRenderer

    payload = json.loads(Path(trace_path).read_text())
    trace = payload["frames"]
    render_utils.CAM_DISTANCE = float(payload.get("camera_distance", render_utils.CAM_DISTANCE))
    frames_dir.mkdir(parents=True, exist_ok=True)
    renderer = DemoRenderer(width, height)
    try:
        for index, item in enumerate(trace):
            state = {
                "core_pos": np.asarray(item["core_pos"], dtype=np.float64),
                "core_quat": np.asarray(item["core_quat"], dtype=np.float64),
                "core_lin_vel": np.zeros(3),
                "core_ang_vel": np.zeros(3),
                "layer_angle": float(item["layer_angle"]),
                "layer_ang_vel": float(item["layer_ang_vel"]),
            }
            frame = renderer.render_state(state)
            Image.fromarray(frame).save(frames_dir / f"frame_{index:04d}.png")
    finally:
        renderer.close()
    print(f"rendered {len(trace)} reduced frames into {frames_dir}")
    return 0


def render_reduced_frames(
    rollout: bridge.ReducedRollout, width: int, height: int, work_dir: Path | None = None
) -> list:
    """Re-render the recorded reduced episode from its trace (the left demo half).

    The render runs in a CHILD process: MuJoCo's EGL context cannot be created in a process
    that has already initialised CUDA/`warp` (the reduced rollout does), and raising
    `MUJOCO_GL` after that import does not help -- a fresh process with `MUJOCO_GL=egl`
    renders fine (verified on the pod).
    """
    import json
    import subprocess
    import tempfile

    from PIL import Image

    work = (
        Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="bridge-render-"))
    )
    trace_path = work / "reduced_trace.json"
    frames_dir = work / "reduced_frames"
    trace_path.write_text(json.dumps(_render_trace_payload(rollout)))
    env = dict(os.environ)
    env["MUJOCO_GL"] = "egl"
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            __file__,
            "--render-reduced-trace",
            str(trace_path),
            "--frames-dir",
            str(frames_dir),
            "--width",
            str(width),
            "--height",
            str(height),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(REPO_ROOT),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "reduced re-render failed: " + (completed.stderr or completed.stdout)[-2000:]
        )
    print((completed.stdout or "").strip().splitlines()[-1] if completed.stdout else "")
    return [
        np.asarray(Image.open(path), dtype=np.uint8) for path in sorted(frames_dir.glob("*.png"))
    ]


def _annotate(
    reduced_frame: np.ndarray,
    arm_frame: np.ndarray,
    *,
    step,
    decision,
    timing: dict,
    args: argparse.Namespace,
    success: bool,
    final: bool = False,
) -> np.ndarray:
    """One annotated side-by-side frame with the bridge HUD."""
    from callosum.common.render_utils import (
        FAILURE_COLOR,
        SUCCESS_COLOR,
        annotate_frame,
        hstack_side_by_side,
    )

    offset = timing["turn_onset_offset"]
    hud = [
        (
            f"REDUCED t={step.t:3d} phase {step.phase:7s} angle "
            f"{step.layer_angle_deg:5.1f} deg |Fh| {step.holder_force_norm_n:4.2f} N "
            f"tau_z {step.rotator_tau_z_nm:6.4f} Nm"
        ),
        (
            f"ARM     d={decision.decision:3d} phase {decision.phase:7s} angle "
            f"{decision.layer_angle_deg:5.1f} deg  {decision.holder:13s} / "
            f"{decision.rotator:13s}"
        ),
        (
            f"mapped primitives: holder {step.intended_holder:13s} rotator "
            f"{step.intended_rotator:13s}"
        ),
        (
            f"phase shares: reduced hold {timing['reduced_phase_shares']['hold']:.2f} / "
            f"turn {timing['reduced_phase_shares']['turn']:.2f} | arm hold "
            f"{timing['arm_phase_shares']['hold']:.2f} / turn "
            f"{timing['arm_phase_shares']['turn']:.2f} | turn onset offset "
            + (f"{offset:+.2f}" if offset is not None else "n/a")
        ),
        (
            f"1 arm decision = {bridge.ARM_DECISION_SECONDS:.2f} s = "
            f"{timing['decision_over_control_step']:.1f} reduced control steps"
        ),
    ]
    return annotate_frame(
        hstack_side_by_side(reduced_frame, arm_frame),
        title="LEARNED REDUCED POLICY -> ARM SKILLS",
        subtitle=(
            "hold->turn phase order of the learned reduced policy -> SO-ARM101 primitives; "
            "intent bridge, not weight transfer"
        ),
        hud_lines=hud,
        agent_blocks=[
            (
                "REDUCED (learned)",
                [
                    f"phase: {step.phase}",
                    (
                        f"holder |F| / |T|: {step.holder_force_norm_n:.2f} N / "
                        f"{step.holder_torque_norm_nm:.4f} Nm"
                    ),
                    f"rotator tau_z: {step.rotator_tau_z_nm:.4f} Nm",
                ],
                (96, 214, 138),
            ),
            (
                "ARM (skills)",
                [
                    f"phase: {decision.phase}",
                    f"holder: {decision.holder}",
                    f"rotator: {decision.rotator}",
                ],
                (242, 140, 40),
            ),
        ],
        legend=[((96, 214, 138), "reduced learned policy"), ((242, 140, 40), "arm skills")],
        angle_fraction=min(max(decision.layer_angle_deg / 90.0, 0.0), 1.0),
        status=None if not final else ("SUCCESS" if success else "INCOMPLETE"),
        status_color=SUCCESS_COLOR if success else FAILURE_COLOR,
        goal_lines=[
            "GOAL: turn the layer/face 90 deg and keep the core/cube in place",
            (
                f"REDUCED: {'success' if success else 'no success'} | plan "
                f"{args.plan_notation!r} | phase L1 error "
                f"{timing['phase_share_l1_error']:.2f}"
            ),
        ],
    )


_STRIP_HEIGHT = 96
"""Height (px) of the extra HUD strip under the two scene halves."""

_PHASE_TIMELINE = ("HOLD", "TURN", "RELEASE")
"""The intent phases shown on the HUD strip (idle/engagement is folded into HOLD)."""

_PHASE_INDEX = {"idle": 0, "hold": 0, "turn": 1, "release": 2}
"""Reduced phase -> timeline slot."""


def _dial_point(center: tuple[float, float], radius: float, fraction: float) -> tuple[float, float]:
    """Point on the gauge arc: fraction 0 = 9 o'clock, 1 = 3 o'clock (through 12)."""
    import math

    angle = math.pi * (1.0 - max(0.0, min(1.0, fraction)))
    return (
        center[0] + radius * math.cos(angle),
        center[1] - radius * math.sin(angle),
    )


def _draw_dial(
    draw, center: tuple[float, float], radius: float, reduced_deg: float, arm_deg: float
):
    """A 0..90 deg gauge with two needles: reduced layer angle (green) and arm face (orange)."""
    box = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    draw.arc(box, 180, 360, fill=(96, 108, 128), width=3)
    for tick in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0):
        inner = _dial_point(center, radius - 5, tick)
        outer = _dial_point(center, radius + 1, tick)
        draw.line([inner, outer], fill=(150, 162, 180), width=2)
    draw.line(
        [center, _dial_point(center, radius - 6, max(reduced_deg, 0.0) / 90.0)],
        fill=(96, 214, 138),
        width=3,
    )
    draw.line(
        [center, _dial_point(center, radius - 6, max(arm_deg, 0.0) / 90.0)],
        fill=(242, 140, 40),
        width=3,
    )
    draw.ellipse([center[0] - 3, center[1] - 3, center[0] + 3, center[1] + 3], fill=(220, 228, 240))


def _hud_strip(
    frame: np.ndarray,
    *,
    step,
    decision,
    timing: dict,
    success: bool,
    final: bool,
) -> np.ndarray:
    """Append the bridge HUD strip: angle dial, HOLD->TURN->RELEASE timeline, primitives."""
    from PIL import Image, ImageDraw

    from callosum.common.render_utils import load_font

    height, width = frame.shape[:2]
    canvas = np.zeros((height + _STRIP_HEIGHT, width, 3), dtype=np.uint8)
    canvas[:] = np.asarray((10, 14, 22), dtype=np.uint8)
    canvas[:height] = frame
    image = Image.fromarray(canvas).convert("RGB")
    draw = ImageDraw.Draw(image)
    top = height
    draw.line([(0, top), (width, top)], fill=(52, 66, 88), width=2)

    label_font = load_font(18, bold=True)
    small_font = load_font(15)
    value_font = load_font(20, bold=True)

    # Dial: reduced layer angle (green) and arm face angle (orange) side by side.
    dial_center = (88.0, top + _STRIP_HEIGHT / 2)
    _draw_dial(
        draw,
        dial_center,
        34.0,
        float(step.layer_angle_deg),
        float(decision.layer_angle_deg),
    )
    draw.text(
        (132, top + 12), f"{step.layer_angle_deg:5.1f} deg", font=value_font, fill=(96, 214, 138)
    )
    draw.text((132, top + 38), "reduced layer", font=small_font, fill=(150, 162, 180))
    draw.text(
        (132, top + 58),
        f"{decision.layer_angle_deg:5.1f} deg",
        font=value_font,
        fill=(242, 140, 40),
    )
    draw.text((260, top + 12), "arm face", font=small_font, fill=(150, 162, 180))

    # Phase timeline HOLD -> TURN -> RELEASE, highlighted on the REALIZED arm phase (the
    # reduced intent phase is printed in the scene HUD, so the mismatch is visible, not hidden).
    active = _PHASE_INDEX.get(decision.phase, 0)
    box_w, box_h, gap = 150, 44, 10
    x0 = 360
    y0 = top + (_STRIP_HEIGHT - box_h) // 2
    for index, name in enumerate(_PHASE_TIMELINE):
        x = x0 + index * (box_w + gap)
        filled = index == active
        color = (242, 140, 40) if filled else (40, 50, 66)
        outline = (255, 200, 120) if filled else (70, 84, 104)
        draw.rounded_rectangle(
            [x, y0, x + box_w, y0 + box_h], radius=8, fill=color, outline=outline, width=2
        )
        text_color = (16, 20, 28) if filled else (170, 182, 200)
        draw.text((x + 14, y0 + 11), f"{index + 1}  {name}", font=label_font, fill=text_color)

    # Intended primitives vs realized arm primitives.
    x_text = x0 + 3 * (box_w + gap) + 10
    draw.text(
        (x_text, top + 14),
        f"intent:  holder {step.intended_holder}  /  rotator {step.intended_rotator}",
        font=small_font,
        fill=(198, 210, 228),
    )
    draw.text(
        (x_text, top + 36),
        f"arm:     holder {decision.holder}  /  rotator {decision.rotator}",
        font=small_font,
        fill=(198, 210, 228),
    )
    draw.text(
        (x_text, top + 60),
        (
            f"turn onset offset {timing['turn_onset_offset']:+.2f} | phase L1 "
            f"{timing['phase_share_l1_error']:.2f}"
            if timing.get("turn_onset_offset") is not None
            else "no turn phase on one side"
        ),
        font=small_font,
        fill=(150, 162, 180),
    )
    draw.text((x0, top + 6), "realized arm phase:", font=small_font, fill=(150, 162, 180))
    del success, final
    return np.asarray(image, dtype=np.uint8)


def _blend_slowmo(combined: list[np.ndarray], factor: int) -> list[np.ndarray]:
    """Slow-mo by linear cross-fade between consecutive frames (cheap, GPU-free).

    Each captured frame is expanded into `factor` interpolated frames; the blend keeps the
    motion smooth at the 10 fps output instead of showing 2 fps stop-motion. Ghosting is
    negligible here because the slow-mo makes the arms move only a few pixels per frame.
    """
    factor = max(1, int(factor))
    if factor == 1:
        return list(combined)
    out: list[np.ndarray] = []
    for index in range(len(combined) - 1):
        current = combined[index].astype(np.float32)
        following = combined[index + 1].astype(np.float32)
        for step in range(factor):
            weight = step / factor
            blended = current * (1.0 - weight) + following * weight
            out.append(blended.astype(np.uint8))
    out.append(combined[-1])
    return out


def _crop_center(frame: np.ndarray, fraction: float, size: tuple[int, int]) -> np.ndarray:
    """Crop the central `fraction` of a frame and resize it back to `size` (close-up)."""
    if fraction >= 1.0:
        return frame
    from PIL import Image

    height, width = frame.shape[:2]
    crop_w = max(2, int(width * max(0.2, fraction)))
    crop_h = max(2, int(height * max(0.2, fraction)))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    image = Image.fromarray(np.ascontiguousarray(frame)).crop(
        (left, top, left + crop_w, top + crop_h)
    )
    return np.asarray(image.resize(size, Image.LANCZOS), dtype=np.uint8)


def write_demo(
    rollout: bridge.ReducedRollout,
    arm_run: _ArmRun,
    timing: dict,
    args: argparse.Namespace,
    demo_dir: Path,
) -> dict:
    """Composite the side-by-side clip + HUD strip, slow it down and write all demo files."""

    from callosum.common.render_utils import probe_video, write_gif, write_mp4

    reduced_frames = render_reduced_frames(rollout, args.width, args.height, demo_dir)
    if not arm_run.frames:
        raise RuntimeError("the arm run recorded no frames; check --render-every")
    arm_frames = [
        _crop_center(frame, float(args.arm_crop), (args.width, args.height))
        for frame in arm_run.frames
    ]
    count = max(len(reduced_frames), len(arm_frames))
    success = bool(rollout.success and arm_run.success)
    combined: list[np.ndarray] = []
    for index in range(count):
        reduced_at = round(index * (len(reduced_frames) - 1) / max(count - 1, 1))
        arm_at = round(index * (len(arm_run.frames) - 1) / max(count - 1, 1))
        step = rollout.steps[reduced_at]
        decision = arm_run.decisions[
            min(arm_run.frame_decisions[arm_at], len(arm_run.decisions) - 1)
        ]
        annotated = _annotate(
            reduced_frames[reduced_at],
            arm_frames[arm_at],
            step=step,
            decision=decision,
            timing=timing,
            args=args,
            success=success,
            final=index >= count - 3,
        )
        combined.append(
            _hud_strip(
                annotated,
                step=step,
                decision=decision,
                timing=timing,
                success=success,
                final=index >= count - 3,
            )
        )
    demo_dir.mkdir(parents=True, exist_ok=True)
    mp4 = demo_dir / "bridge_wrench_to_skills.mp4"
    raw = demo_dir / "bridge_wrench_to_skills_raw.mp4"
    gif = demo_dir / "bridge_wrench_to_skills_preview.gif"
    slow_frames = _blend_slowmo(combined, int(args.slowmo))
    hold = max(2, int(args.fps))
    slow_frames += [combined[-1]] * hold
    encoder = write_mp4(mp4, slow_frames, args.fps)
    interpolated = True
    raw.unlink(missing_ok=True)
    gif_frames = write_gif(
        gif,
        slow_frames[:: max(2, int(len(slow_frames) / 48))],
        frame_duration_ms=120,
        width=520,
    )
    stills = _write_demo_stills(rollout, arm_run, combined, demo_dir, count)
    probe = probe_video(mp4)
    return {
        "video": str(mp4),
        "preview_gif": str(gif),
        "encoder": encoder,
        "fps": args.fps,
        "frames": len(slow_frames),
        "interpolated_slowmo": interpolated,
        "gif_frames": gif_frames,
        "probe": probe,
        "reduced_frames": len(reduced_frames),
        "arm_frames": len(arm_run.frames),
        "stills": [str(path) for path in stills],
    }


def _write_demo_stills(
    rollout: bridge.ReducedRollout,
    arm_run: _ArmRun,
    combined: list[np.ndarray],
    demo_dir: Path,
    count: int,
) -> list[Path]:
    """Save four stills: engage, hold, mid-turn and the final success frame."""
    import numpy as np
    from PIL import Image

    angles = [step.layer_angle_deg for step in rollout.steps]
    frames = [round(index * (len(rollout.steps) - 1) / max(count - 1, 1)) for index in range(count)]
    engage_at = next((i for i, a in enumerate(angles) if a > 2.0), 0)
    turn_at = next((i for i, a in enumerate(angles) if 30.0 <= a <= 60.0), len(angles) - 1)
    picks = {"engage": 2, "hold": max(engage_at - 1, 1), "turn": turn_at, "success": count - 1}
    paths: list[Path] = []
    for name, frame_index in picks.items():
        frame_index = max(0, min(frame_index, count - 1))
        path = demo_dir / f"bridge_{name}.png"
        Image.fromarray(np.asarray(combined[frame_index], dtype=np.uint8)).save(path)
        paths.append(path)
    del frames, rollout, arm_run
    return paths


# --- main -----------------------------------------------------------------------------------


def _write_report(
    args: argparse.Namespace,
    rollout: bridge.ReducedRollout,
    schedule: list,
    *,
    arm: dict | None,
    schedule_replay: dict | None,
    timing: dict | None,
    timing_schedule: dict | None,
    demo: dict | None,
    notes: list[str],
) -> Path:
    """Assemble and write the bridge JSON (both schedule timing modes included)."""
    report = bridge.bridge_report(
        rollout,
        schedule,
        timing_mode=args.timing_mode,
        arm=arm,
        conductor=None,
        timing=timing,
        notes=notes,
    )
    report["schedule_alt_dt"] = bridge.schedule_json(
        bridge.build_primitive_schedule(
            rollout.segments, timing_mode="dt", total_decisions=args.arm_total_decisions
        )
    )
    report["schedule_replay"] = schedule_replay
    report["timing_schedule_replay"] = timing_schedule
    report["demo"] = demo
    report["cli"] = {key: value for key, value in vars(args).items() if not key.startswith("_")}
    return bridge.write_json(report, Path(args.out))


def _timing_block(
    rollout: bridge.ReducedRollout, run: _ArmRun, config, plan_length: int | None = None
) -> dict:
    """The timing comparison of one arm execution against the reduced phase structure."""
    return bridge.compare_phase_timing(
        rollout.segments,
        run.decisions,
        reduced_control_dt_s=rollout.control_dt_s,
        reduced_success=rollout.success,
        arm_success=run.success,
        plan_length=config.plan_length if plan_length is None else plan_length,
    )


def _print_timing(label: str, timing: dict) -> None:
    """Print one timing comparison compactly."""
    offset = timing["turn_onset_offset"]
    print(
        f"timing[{label}]: phase-share L1 {timing['phase_share_l1_error']:.3f} | derived "
        f"hold/turn L1 {timing['derived_phase_share_l1_error']:.3f} | turn onset offset "
        f"{offset if offset is None else round(offset, 3)} | "
        f"{timing['decision_over_control_step']:.1f} reduced steps per arm decision | "
        f"reduced turn {timing['reduced_turn_duration_s']:.2f} s vs arm turn "
        f"{timing['arm_turn_duration_s']:.1f} s"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the bridge: reduced exemplar, schedule, arm execution, metrics and optional demo."""
    args = parse_args(argv)
    if args.render:
        # MuJoCo picks its GL backend at import time; the reduced rollout imports it (the
        # mujoco_warp host model) long before the demo render, so the EGL choice must be
        # made here, not inside the render helper.
        os.environ.setdefault("MUJOCO_GL", "egl")
    if args.render_reduced_trace:
        return _render_trace_child(
            Path(args.render_reduced_trace), Path(args.frames_dir), args.width, args.height
        )
    out = Path(args.out)
    demo_dir = out.parent
    demo_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    notes.append(
        "Scope: one greedy reduced episode on a fixed seed (an exemplar, not statistics) and "
        "one deterministic arm episode of the fixed plan; the numbers are a feasibility "
        "measurement of the intent bridge."
    )
    notes.append(
        "Arm plan 'U' (up face) needs no holder flip: the flip-based plans (e.g. 'R') still "
        "fail in the current FaceTurnSkills-v0 (conductor 0 success), a separate flip bug of "
        "the skill line, not of the bridge."
    )
    rollout, schedule = run_reduced(args)
    arm_block = None
    replay_block = None
    timing = None
    timing_replay = None
    demo = None
    if not args.reduced_only:
        env, base, config = build_arm_env(args, with_camera=args.render)
        try:
            print(
                f"arm env: fixed plan {config.plan_notation!r} -> {config.plan_length} move(s) | "
                f"randomization {'on' if args.arm_randomization else 'off (deterministic)'} | "
                f"sim {base.backend.sim_backend} | render {base.backend.render_backend}"
            )
            conductor_run = None
            if not args.no_conductor:
                conductor_run = run_conductor_baseline(
                    env,
                    base,
                    args,
                    render=args.render,
                    render_every=args.render_every,
                )
                print(
                    f"conductor executor: success {conductor_run.success} | "
                    f"decisions {len(conductor_run.decisions)} | "
                    f"angle {conductor_run.final_angle_deg:.1f} deg | "
                    f"plan index {conductor_run.final_plan_index}"
                )
            replay_run = execute_schedule(
                env, base, args, schedule, render=args.render, render_every=args.render_every
            )
            print(
                f"schedule replay: success {replay_run.success} | decisions "
                f"{len(replay_run.decisions)} | angle {replay_run.final_angle_deg:.1f} deg | "
                f"plan index {replay_run.final_plan_index} | timed out {replay_run.timed_out}"
            )
            # The conductor is the arm-side EXECUTOR (it recovers from grasp loss/slips); the
            # open-loop replay of the schedule is the second measurement, and it is where a
            # bridge-only loss (e.g. no re-grasp rule in the schedule) shows up.
            primary = conductor_run if conductor_run is not None else replay_run
            timing = _timing_block(rollout, primary, config)
            _print_timing(primary.kind, timing)
            if conductor_run is not None:
                timing_replay = _timing_block(rollout, replay_run, config)
                _print_timing(replay_run.kind, timing_replay)
            notes.append(
                "Arm executor is the scripted SkillsConductor (the project's skill-level upper "
                "bound); the open-loop replay of the reduced schedule is reported separately "
                "(`schedule_replay`) as the pure intent-transfer attempt."
            )
            arm_block = primary.summary(config)
            arm_block["seed"] = args.arm_seed
            arm_block["randomization"] = bool(args.arm_randomization)
            arm_block["executor"] = primary.kind
            replay_block = replay_run.summary(config)
            replay_block["seed"] = args.arm_seed
            replay_block["randomization"] = bool(args.arm_randomization)
            if conductor_run is not None and not conductor_run.success:
                status = "BLOCKED"
                notes.append(
                    "BLOCKED: the scripted conductor failed on the fixed plan in the same run, "
                    "so the arm environment (FaceTurnSkills-v0) is not healthy; the reduced "
                    "trace/schedule and the arm trace are still valid measurements."
                )
            elif not primary.success:
                status = "bridge-loss"
                notes.append(
                    "The arm execution did not reach env success; the reduced trace, the "
                    "schedule and the realized arm trace are valid, but no success is claimed."
                )
            else:
                status = "ok"
            arm_block["status"] = status
            if primary.success and not replay_run.success:
                notes.append(
                    "The conductor executor reached success, but the open-loop replay of the "
                    "reduced schedule did not: the schedule carries the phase intent, not the "
                    "arm's re-grasp/recovery rules (the replay's turn loses the grasp and "
                    "stalls; the env still counts the move only at the settle check)."
                )
            print(notes[-1] if notes else "")
            # Persist the measurements BEFORE the demo render: a rendering failure must not
            # lose the bridge numbers (the demo is re-attached to the report below).
            _write_report(
                args,
                rollout,
                schedule,
                arm=arm_block,
                schedule_replay=replay_block,
                timing=timing,
                timing_schedule=timing_replay,
                demo=None,
                notes=notes,
            )
            if args.render:
                demo = write_demo(rollout, primary, timing, args, demo_dir)
                print(f"demo: {demo['video']} {demo['probe']}")
        finally:
            env.close()
    else:
        notes.append(
            "reduced-only run: the arm half was not executed (no schedule execution, no demo); "
            "the reduced trace and schedule are this run's deliverable."
        )

    out = _write_report(
        args,
        rollout,
        schedule,
        arm=arm_block,
        schedule_replay=replay_block,
        timing=timing,
        timing_schedule=timing_replay,
        demo=demo,
        notes=notes,
    )
    report_status = (arm_block or {}).get("status", "reduced-only")
    print(f"bridge status: {report_status}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
