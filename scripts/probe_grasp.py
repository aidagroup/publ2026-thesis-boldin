"""Is FaceTurn-v0 solvable AT ALL? A scripted open-loop expert.

Server-only (GPU sim). Written on 2026-08-30 after ~4M steps of IPPO in which
`success_once` was exactly 0.0 and `grasped` exactly 0.00: aggregate metrics
cannot distinguish "the policy has not learned it yet" from "no policy could".
This script removes the policy from the loop and drives both arms through
hand-computed joint waypoints, so whatever it reports is a property of the
SCENE, not of training.

Two independent checks:

  1. Gripper aperture. Sweeps the gripper joint over its full [-1.1, 1.1]
     range and reports the measured jaw-tip separation. Compare against the
     handle width: `callosum.envs._cube_geometry.HANDLE_HALF_WIDTH`.
  2. Scripted grasp-and-turn. READY_QPOS -> pre-grasp -> grasp -> hold ->
     close -> quarter turn, printing the reach distances, both `is_grasping`
     flags, the face angle and the body drift at every phase boundary. If this
     does not reach `success`, no amount of RL will.

The waypoints come from `callosum.envs._so100_kinematics`. They are joint
configurations, not end-effector poses, because ManiSkill has no IK on the GPU
backend (`Articulation.create_pinocchio_model` raises when GPU sim is
enabled) -- which also means nothing at run time checks that they still point
at the handles. Two server rounds were lost to exactly that, both frame
errors:

  * 2026-08-30 run 1: the FK that solved them stopped at `wrist_flex`, which
    lands on the Wrist_Pitch_Roll frame, 6 cm short of `Fixed_Jaw` along the
    tool axis. The rotator's grasp pose put its tcp at z = 0.012, inside the
    cube, so it jammed the jaws in (a 0.69 grasp that then slipped, plus body
    drift from step 75); the holder's put its tcp at z = -0.026, below the
    tabletop, so it stalled and never closed on anything.
  * 2026-08-30 run 2: the waypoints aimed `SO100.tcp_pos` at each handle's
    axis. The tcp is the midpoint of the jaw-TIP links, which are points ~2 mm
    inside the fingertips, so that buried the fixed blade 2.1 mm inside the
    handle. Both arms settled ~2.7 cm away -- the same miss on both, which is
    the signature of a constant frame offset -- shoving the cube (dpos
    0.012 -> 0.019) as they tried to close. `GRASP_POCKET_OFFSET` is the fix.

Both are now regression-tested without a simulator, in
tests/test_grasp_waypoints.py; the `!!` line below is the run-time backstop.

    uv run python scripts/probe_grasp.py
    uv run python scripts/probe_grasp.py --envs 16 --every 20
"""

import argparse

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils import common
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import callosum.envs.face_turn  # noqa: F401  (registers FaceTurn-v0)
from callosum.envs._cube_geometry import FACE_HANDLE_HEIGHT, HANDLE_HALF_WIDTH
from callosum.envs._scripted_expert import DELTA_LIMITS, PHASES, arm_targets
from callosum.envs._so100_kinematics import (
    READY_QPOS,
    ROTATOR_GRASP,
    SEATING_GRIPPER_QPOS,
)
from callosum.envs.face_turn import TARGET_FACE_ANGLE

# At a correct grasp the tcp lands 1.9 mm from the handle's axis (the handle
# sits off the tool centreline by `GRASP_POCKET_OFFSET`, because the jaw-tip
# links are points inside the fingertips rather than the gripping surfaces).
# Anything much above that means the waypoints no longer match the scene, and
# every later column is meaningless -- so say so rather than let it read as a
# physics result. tests/test_grasp_waypoints.py checks the same thing in CI.
WAYPOINT_TOLERANCE = 0.010


def _report_aperture(base) -> None:
    """Measure the jaw opening in sim, instead of trusting the URDF read.

    Tip separation is an UPPER bound on what fits, and a loose one. The tip
    links are points ~2 mm inside the fingertips, and the jaw is a hinged
    pincer whose blades are only parallel near the middle of their travel, so
    the free gap deeper between them is smaller. The row that matters is the
    one at `SEATING_GRIPPER_QPOS`, where the gap at the handle's insertion
    depth is exactly the handle width.
    """
    print("\ngripper aperture (measured from the jaw-tip link poses)")
    print(f"  handle width to straddle: {2 * HANDLE_HALF_WIDTH * 100:.1f} cm")
    print(f"  seats at gripper qpos {SEATING_GRIPPER_QPOS:+.3f}")
    print(f"  {'gripper qpos':>13} {'tip separation':>15}")
    agent = base.agent_b
    q = torch.zeros((base.num_envs, 6), device=base.device)
    q[:, :] = torch.as_tensor(READY_QPOS, dtype=q.dtype, device=base.device)
    for g in sorted(np.linspace(-1.1, 1.1, 12).tolist() + [SEATING_GRIPPER_QPOS]):
        q[:, 5] = float(g)
        agent.robot.set_qpos(q)
        base.scene._gpu_apply_all()
        base.scene.px.gpu_update_articulation_kinematics()
        base.scene._gpu_fetch_all()
        sep = torch.linalg.norm(agent.finger1_tip.pose.p - agent.finger2_tip.pose.p, dim=1)
        flag = "  <- seating angle" if g == SEATING_GRIPPER_QPOS else ""
        print(f"  {g:>13.3f} {float(sep[0]) * 100:>14.2f} cm{flag}")


def _action(agent, target_arm: np.ndarray, target_grip: float, device) -> torch.Tensor:
    """Proportional joint-space controller in the delta action space.

    `pd_joint_delta_pos` with use_target=False sets the drive target to
    qpos + action, so saturating the action means "move this joint as fast as
    the controller allows toward the waypoint".
    """
    target = torch.as_tensor(
        np.concatenate([target_arm, [target_grip]]), dtype=torch.float32, device=device
    )
    delta = torch.as_tensor(DELTA_LIMITS, dtype=torch.float32, device=device)
    return torch.clamp((target - agent.robot.get_qpos()) / delta, -1.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--env-id", default="FaceTurn-v0")
    ap.add_argument("--every", type=int, default=25, help="print every N steps")
    ap.add_argument(
        "--only",
        choices=("both", "rotator", "holder"),
        default="both",
        help="park the other arm at READY, to test whether the two collide",
    )
    args = ap.parse_args()
    parked = np.asarray(READY_QPOS[:5])

    env = ManiSkillVectorEnv(
        gym.make(
            args.env_id,
            num_envs=args.envs,
            obs_mode="state_dict",
            sim_backend="physx_cuda",
            # Open loop: the waypoints are solved once against the nominal
            # cube pose, so any spawn jitter is a pure miss, and the arms must
            # start exactly where the kinematics assumed.
            cube_spawn_jitter=0.0,
            robot_init_qpos_noise=0.0,
            render_backend="none",
        ),
        args.envs,
        # A successful episode must keep running so the whole trajectory stays
        # visible instead of auto-resetting mid-probe.
        ignore_terminations=True,
        record_metrics=False,
    )
    base = env.unwrapped
    uids = tuple(base.agent.agents_dict.keys())
    device = base.device
    env.reset(seed=0)

    _report_aperture(base)
    env.reset(seed=0)

    # The face joint's sign convention (which way a wrist_roll of +pi/2 drives
    # it) is not knowable from the builder without simulating, and its limits
    # are [0, pi/2] so the wrong sign simply does nothing. Split the parallel
    # envs and try both at once.
    half = args.envs // 2
    roll_sign = torch.ones(args.envs, device=device)
    roll_sign[half:] = -1.0

    driving = "both arms" if args.only == "both" else f"{args.only} only (partner parked at READY)"
    print(f"\nscripted expert on {args.env_id}, {args.envs} envs, {driving}")
    print(
        f"  face handle: {2 * HANDLE_HALF_WIDTH * 100:.1f} cm wide,"
        f" {FACE_HANDLE_HEIGHT * 100:.1f} cm tall"
    )
    hdr = (
        f"{'step':>5} {'phase':>9} {'rot→face':>9} {'hold→body':>10} "
        f"{'grasped':>8} {'held':>6} {'angle+':>7} {'angle-':>7} "
        f"{'dpos':>6} {'drot':>6} {'succ':>5}"
    )
    print(f"{hdr}\n{'-' * len(hdr)}")

    step = 0
    with torch.no_grad():
        for name, waypoint, hold_grip, rot_grip, budget in PHASES:
            for i in range(budget):
                hold_arm, rot_arm = arm_targets(waypoint)
                # --only parks the other arm at READY for the whole run. If an
                # arm reaches its handle alone but not alongside its partner,
                # the blocker is the two of them, not either one's waypoints.
                if args.only == "rotator":
                    hold_arm = parked
                elif args.only == "holder":
                    rot_arm = parked
                a_hold = _action(base.agent_a, hold_arm, hold_grip, device)
                a_rot = _action(base.agent_b, rot_arm, rot_grip, device)
                if name == "turn":
                    # Drive wrist_roll away from its grasp value by a quarter
                    # turn, in both directions across the env batch. Rolling
                    # does not drag the grip off the post: with the jaws closed
                    # the tcp sits 0.1 mm from the roll axis, and it moves only
                    # 2.0-2.2 mm from the post's axis over the full pi of roll.
                    roll_target = float(ROTATOR_GRASP[4]) + roll_sign * TARGET_FACE_ANGLE
                    a_rot[:, 4] = torch.clamp(
                        (roll_target - base.agent_b.robot.get_qpos()[:, 4])
                        / float(DELTA_LIMITS[4]),
                        -1.0,
                        1.0,
                    )
                env.step({uids[0]: a_hold, uids[1]: a_rot})
                step += 1

                if step % args.every and i != budget - 1:
                    continue
                info = base.get_info()
                angle = base.face_link.joint.qpos
                d_rot = torch.linalg.norm(base.agent_b.tcp_pos - base.face_grasp_pos, dim=1)
                d_hold = torch.linalg.norm(base.agent_a.tcp_pos - base.body_grasp_pos, dim=1)
                dpos = torch.linalg.norm(base.cube.pose.p - base.body_init_pos, dim=1)
                drot = common.quat_diff_rad(base.cube.pose.q, base.body_init_q)
                print(
                    f"{step:>5} {name:>9} {d_rot.mean():>9.3f} {d_hold.mean():>10.3f} "
                    f"{base.agent_b.is_grasping(base.face_link).float().mean():>8.2f} "
                    f"{base.agent_a.is_grasping(base.body_link).float().mean():>6.2f} "
                    f"{angle[:half].mean():>7.3f} {angle[half:].mean():>7.3f} "
                    f"{dpos.mean():>6.3f} {drot.mean():>6.3f} "
                    f"{info['success'].float().mean():>5.2f}"
                )

            if name == "close":
                d_rot = torch.linalg.norm(base.agent_b.tcp_pos - base.face_grasp_pos, dim=1)
                d_hold = torch.linalg.norm(base.agent_a.tcp_pos - base.body_grasp_pos, dim=1)
                for who, d in (("rotator", d_rot), ("holder", d_hold)):
                    if float(d.mean()) > WAYPOINT_TOLERANCE:
                        print(
                            f"  !! {who} settled {float(d.mean()) * 100:.1f} cm from its handle"
                            f" (expected < {WAYPOINT_TOLERANCE * 1000:.0f} mm). The waypoint does"
                            " not match the scene -- re-solve it before reading anything below."
                        )

    info = base.get_info()
    angle = base.face_link.joint.qpos
    print("\nverdict")
    print(f"  best face angle : {float(angle.max()):.3f} rad of {TARGET_FACE_ANGLE:.3f}")
    print(f"  rotator grasped : {base.agent_b.is_grasping(base.face_link).float().mean():.2f}")
    print(f"  holder  grasped : {base.agent_a.is_grasping(base.body_link).float().mean():.2f}")
    print(f"  body stable     : {info['is_body_stable'].float().mean():.2f}")
    print(f"  success         : {info['success'].float().mean():.2f}")
    if float(info["success"].float().mean()) == 0.0:
        print("\n  >>> the scripted expert cannot solve the task either.")
        print("  >>> the blocker is the SCENE, not the policy. Read the columns above:")
        print("  >>> grasped==0 -> geometry/waypoints; angle==0 with grasped>0 -> joint")
        print("  >>> friction/damping; body drift high -> the holder is not holding.")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
