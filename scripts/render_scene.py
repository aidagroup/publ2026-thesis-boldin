"""Look at the scene. Pictures of the grasp, and a video of the scripted expert.

Server-only (GPU sim). Written on 2026-09-02, after three rounds in which the
grasp waypoints were re-solved from numbers alone and the probe still reported
the arms settling centimetres from their handles. `scripts/probe_grasp.py`
says *that* it fails and by how much; this says what it looks like.

Two things are rendered, and the first is the one that matters:

  By default, a still per waypoint with both arms TELEPORTED there (`set_qpos`,
  no physics, nothing to collide with or fall short of). That separates "the
  waypoint is wrong" from "the arm cannot get there" -- the question the
  distance columns alone cannot answer. Printed alongside each image are the
  same distances as the forward kinematics in
  `callosum.envs._so100_kinematics` predicts them, so the picture arrives with
  its numbers.

  --script replays the same phase table `probe_grasp.py` drives
          (`callosum.envs._scripted_expert.PHASES`), writing a frame every
          `--every` steps plus a still at each phase boundary. This is the one
          that shows contact: whatever knocks the cube 3.7 cm off its spawn is
          visible here and invisible in a table of means.

Rendering needs a Vulkan device, so this cannot use the `render_backend="none"`
the trainer and the probes run with (see docs/jupyterhub-runbook.md). On this
pod that device may be the lavapipe software rasteriser, which is slow -- hence
`--every`, and one env rather than 32.

    uv run python scripts/render_scene.py                      # the three poses
    uv run python scripts/render_scene.py --script             # + the rollout
    uv run python scripts/render_scene.py --script --only rotator
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils import common
from mani_skill.utils.structs.pose import Pose
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from PIL import Image, ImageDraw

import callosum.envs.face_turn  # noqa: F401  (registers FaceTurn-v0)
from callosum.envs._cube_geometry import (
    BODY_GRASP_LIFTED,
    BODY_GRASP_WORLD,
    CUBE_HALF_SIZE,
    FACE_GRASP_LIFTED,
    LIFT_HEIGHT,
)
from callosum.envs._scripted_expert import DELTA_LIMITS, PHASES, TURN_PHASE, arm_target
from callosum.envs._so100_kinematics import (
    HOLDER_BASE_POSE,
    READY_QPOS,
    ROTATOR_BASE_POSE,
    ROTATOR_GRASP,
    SEATING_GRIPPER_QPOS,
    tcp_position,
)
from callosum.envs.face_turn import TARGET_FACE_ANGLE

# The waypoints worth a still, in the order the task reaches them.
POSES = (
    ("ready", "ready", "ready"),
    ("holder pregrasp", "holder_pregrasp", "ready"),
    ("holder grasp", "holder_grasp", "ready"),
    ("holder lift", "holder_lift", "ready"),
    ("rotator pregrasp", "holder_lift", "rotator_pregrasp"),
    ("rotator grasp", "holder_lift", "rotator_grasp"),
)


def _frame(base) -> np.ndarray:
    """One RGB frame, both human render cameras tiled side by side."""
    img = base.render_rgb_array()
    if img is None:
        raise RuntimeError(
            "render_rgb_array() returned None -- the env has no human render camera."
        )
    return common.to_numpy(img)[0].astype(np.uint8)


def _captioned(frame: np.ndarray, caption: str) -> Image.Image:
    """The frame with a caption bar under it, so a saved still is self-describing."""
    img = Image.fromarray(frame)
    out = Image.new("RGB", (img.width, img.height + 22), (16, 16, 16))
    out.paste(img, (0, 0))
    ImageDraw.Draw(out).text((6, img.height + 6), caption, fill=(235, 235, 235))
    return out


def _place_cube(base, centre_height: float) -> None:
    """Move the cube itself, so lifted waypoints are measured against a lifted cube."""
    pose = base.cube.pose
    p = pose.p.clone()
    p[:, 2] = centre_height
    base.cube.set_pose(Pose.create_from_pq(p=p, q=pose.q))
    base.scene._gpu_apply_all()
    base.scene.px.gpu_update_articulation_kinematics()
    base.scene._gpu_fetch_all()


def _teleport(base, qpos_holder, qpos_rotator) -> None:
    """Put both arms at a joint configuration outright, skipping the controller.

    Same GPU round trip `probe_grasp._report_aperture` uses: write the qpos,
    push it to the GPU, refresh the link poses, read them back. Nothing is
    simulated, so what this renders is the waypoint itself rather than what the
    arm managed to reach.
    """
    for agent, arm in ((base.agent_a, qpos_holder), (base.agent_b, qpos_rotator)):
        q = torch.zeros((base.num_envs, 6), device=base.device)
        q[:, :] = torch.as_tensor(arm, dtype=q.dtype, device=base.device)
        agent.robot.set_qpos(q)
    base.scene._gpu_apply_all()
    base.scene.px.gpu_update_articulation_kinematics()
    base.scene._gpu_fetch_all()


def _distances(base) -> tuple[float, float]:
    """Live tcp-to-handle distances, in metres, averaged over the batch."""
    d_rot = torch.linalg.norm(base.agent_b.tcp_pos - base.face_grasp_pos, dim=1)
    d_hold = torch.linalg.norm(base.agent_a.tcp_pos - base.body_grasp_pos, dim=1)
    return float(d_rot.mean()), float(d_hold.mean())


def _fk_distances(qpos_holder, qpos_rotator, lifted: bool) -> tuple[float, float]:
    """The same two distances as the kinematics predicts them, with no simulator.

    Printing both is the point: where they agree the model matches the scene,
    and where they diverge the arm did not arrive.
    """
    body = np.asarray(BODY_GRASP_LIFTED if lifted else BODY_GRASP_WORLD)
    rot = np.linalg.norm(
        tcp_position(qpos_rotator, ROTATOR_BASE_POSE) - np.asarray(FACE_GRASP_LIFTED)
    )
    hold = np.linalg.norm(tcp_position(qpos_holder, HOLDER_BASE_POSE) - body)
    return float(rot), float(hold)


def _six(arm, grip: float) -> np.ndarray:
    return np.concatenate([np.asarray(arm, dtype=float)[:5], [grip]])


def _action(agent, target_arm: np.ndarray, target_grip: float, device) -> torch.Tensor:
    """Proportional joint-space controller in the delta action space.

    Identical to `probe_grasp._action` -- the two scripts must command the same
    thing for their outputs to be comparable.
    """
    target = torch.as_tensor(
        np.concatenate([target_arm, [target_grip]]), dtype=torch.float32, device=device
    )
    delta = torch.as_tensor(DELTA_LIMITS, dtype=torch.float32, device=device)
    return torch.clamp((target - agent.robot.get_qpos()) / delta, -1.0, 1.0)


def _render_poses(base, out: Path) -> None:
    """A still per waypoint, teleported, plus the sim/FK distance comparison."""
    print("\nwaypoints, teleported (no physics)")
    print(f"  {'pose':>17} {'rot→face':>18} {'hold→body':>18}")
    for i, (label, holder_wp, rotator_wp) in enumerate(POSES):
        grip = SEATING_GRIPPER_QPOS if holder_wp != "ready" else 0.0
        rot_grip = SEATING_GRIPPER_QPOS if rotator_wp == "rotator_grasp" else 0.0
        q_hold = _six(arm_target(holder_wp), grip)
        q_rot = _six(arm_target(rotator_wp), rot_grip)
        lifted = holder_wp == "holder_lift"
        _place_cube(base, LIFT_HEIGHT if lifted else CUBE_HALF_SIZE)
        _teleport(base, q_hold, q_rot)
        d_rot, d_hold = _distances(base)
        fk_rot, fk_hold = _fk_distances(q_hold, q_rot, lifted)
        print(
            f"  {label:>17} {d_rot * 100:7.2f} cm (fk {fk_rot * 100:5.2f})"
            f" {d_hold * 100:7.2f} cm (fk {fk_hold * 100:5.2f})"
        )
        caption = f"{label}  |  rot->face {d_rot * 100:.2f} cm  hold->body {d_hold * 100:.2f} cm"
        path = out / f"pose_{i}_{label.replace(' ', '_')}.png"
        _captioned(_frame(base), caption).save(path)
        print(f"            -> {path}")


def _render_script(env, base, uids, out: Path, args) -> None:
    """The scripted rollout, as a video plus one still per phase boundary."""
    parked = np.asarray(READY_QPOS[:5])
    device = base.device
    env.reset(seed=args.seed)
    frames: list[Image.Image] = []
    step = 0

    print("\nscripted rollout")
    with torch.no_grad():
        for name, holder_wp, rotator_wp, hold_grip, rot_grip, budget in PHASES:
            hold_arm, rot_arm = arm_target(holder_wp), arm_target(rotator_wp)
            if args.only == "rotator":
                hold_arm = parked
            elif args.only == "holder":
                rot_arm = parked
            for i in range(budget):
                a_hold = _action(base.agent_a, hold_arm, hold_grip, device)
                a_rot = _action(base.agent_b, rot_arm, rot_grip, device)
                if name == TURN_PHASE:
                    roll_target = float(ROTATOR_GRASP[4]) + TARGET_FACE_ANGLE
                    a_rot[:, 4] = torch.clamp(
                        (roll_target - base.agent_b.robot.get_qpos()[:, 4])
                        / float(DELTA_LIMITS[4]),
                        -1.0,
                        1.0,
                    )
                env.step({uids[0]: a_hold, uids[1]: a_rot})
                step += 1

                last = i == budget - 1
                if step % args.every and not last:
                    continue
                d_rot, d_hold = _distances(base)
                lift = float(base.cube.pose.p[:, 2].mean())
                caption = (
                    f"{step:>3} {name:<9} rot->face {d_rot * 100:5.2f} cm"
                    f"  hold->body {d_hold * 100:5.2f} cm  cube at {lift * 100:5.2f} cm"
                )
                shot = _captioned(_frame(base), caption)
                frames.append(shot)
                if last:
                    path = out / f"phase_{name}.png"
                    shot.save(path)
                    print(f"  {name:<9} step {step:>3} -> {path}")

    gif = out / "expert.gif"
    frames[0].save(
        gif, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0
    )
    print(f"  {len(frames)} frames -> {gif}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-id", default="FaceTurn-v0")
    ap.add_argument("--out", default="runs/render", help="directory for the images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--script", action="store_true", help="also replay the scripted expert")
    ap.add_argument("--every", type=int, default=5, help="capture a frame every N steps")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument(
        "--only",
        choices=("both", "rotator", "holder"),
        default="both",
        help="park the other arm at READY, to test whether the two collide",
    )
    ap.add_argument(
        "--render-backend",
        default="gpu",
        help='"gpu", or "cpu" to force the software rasteriser',
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = ManiSkillVectorEnv(
        gym.make(
            args.env_id,
            num_envs=1,
            obs_mode="state_dict",
            sim_backend="physx_cuda",
            # Open loop: the waypoints are solved once against the nominal
            # cube pose, so any spawn jitter is a pure miss, and the arms must
            # start exactly where the kinematics assumed.
            cube_spawn_jitter=0.0,
            robot_init_qpos_noise=0.0,
            render_backend=args.render_backend,
            render_mode="rgb_array",
        ),
        1,
        ignore_terminations=True,
        record_metrics=False,
    )
    base = env.unwrapped
    uids = tuple(base.agent.agents_dict.keys())
    if not base.scene.can_render():
        print(
            "no render device -- source .callosum-env.sh first, or pass"
            " --render-backend cpu (see docs/jupyterhub-runbook.md)"
        )
        env.close()
        return 1
    env.reset(seed=args.seed)

    path = out / "initial.png"
    _captioned(_frame(base), f"{args.env_id} after reset, seed {args.seed}").save(path)
    print(f"initial state -> {path}")

    _render_poses(base, out)
    _place_cube(base, CUBE_HALF_SIZE)
    if args.script:
        _render_script(env, base, uids, out, args)

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
