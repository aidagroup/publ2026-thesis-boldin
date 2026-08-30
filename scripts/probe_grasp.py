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
     handle width: `callosum.envs._turntable_cube.HANDLE_HALF_WIDTH`.
  2. Scripted grasp-and-turn. READY_QPOS -> pre-grasp -> grasp -> close ->
     quarter turn, printing the reach distances, both `is_grasping` flags,
     the face angle and the body drift at every phase boundary. If this does
     not reach `success`, no amount of RL will.

The waypoints below are joint configurations, not end-effector poses: there is
no IK available on the GPU backend (Articulation.create_pinocchio_model raises
for GPU sim). They were solved offline from so100.urdf by forward kinematics +
a refined grid search, for the exact arm placement in
`callosum.envs.two_so100_base` (ARM_BASE_OFFSET, base yaws pi and 0). Change
either of those and the waypoints need re-solving -- the `rot->face` and
`hold->body` columns are the check: they should fall to a few millimetres by
the end of the "descend" phase, and anything else means the waypoints no
longer match the scene.

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
from callosum.envs._turntable_cube import FACE_HANDLE_HEIGHT, HANDLE_HALF_WIDTH
from callosum.envs.face_turn import TARGET_FACE_ANGLE
from callosum.envs.two_so100_base import READY_QPOS

# Arm joint waypoints, in the SO-100's active-joint order
# (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll).
#
# wrist_roll stays at pi/2 throughout the approach: at that value the tool
# approaches straight down AND the jaws close along the world x axis, which is
# the axis both handles present a flat 2 cm face to. The rotator then sweeps
# wrist_roll by +-pi/2 to turn the face -- the only SO-100 joint whose axis is
# vertical when the tool points down.
#
# The pre-grasp waypoints sit 4.5 cm above the grasp waypoints so the descent
# onto the handle is vertical rather than a swing that would sweep the handle
# aside.
_ROLL = np.pi / 2
ROTATOR_PREGRASP = np.array([0.0, 0.4163, -0.4128, 1.5673, _ROLL])
ROTATOR_GRASP = np.array([0.0, 0.4562, -0.1305, 1.2452, _ROLL])
HOLDER_PREGRASP = np.array([0.0, -0.0614, 0.6199, 1.0123, _ROLL])
HOLDER_GRASP = np.array([0.0, 0.2052, 0.6956, 0.6701, _ROLL])

# Gripper joint targets. -1.1 is the URDF lower limit (jaws shut, tips 6.6 mm
# apart); 0.0 is mani-skill's own SO-100 ready value (tips 8.4 cm apart).
# -0.85 is where the measured aperture is ~2 cm, i.e. just touching a handle;
# commanding the full -1.1 makes the PD drive keep squeezing after contact,
# which is what generates the >= 0.5 N `is_grasping` needs.
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = -1.1

# Per-phase step budgets. Sum stays under FaceTurn-v0's 300-step limit.
PHASES = [
    ("settle", None, GRIPPER_OPEN, 10),
    ("pregrasp", "pregrasp", GRIPPER_OPEN, 50),
    ("descend", "grasp", GRIPPER_OPEN, 40),
    ("close", "grasp", GRIPPER_CLOSED, 25),
    ("turn", "grasp", GRIPPER_CLOSED, 150),
]

# pd_joint_delta_pos limits, from SO100._controller_configs. The action space
# is normalized to [-1, 1] (PDJointPosControllerConfig.normalize_action
# defaults to True), so a unit action is one full delta.
DELTA_LIMITS = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.2])


def _report_aperture(base) -> None:
    """Measure the jaw opening in sim, instead of trusting the URDF read."""
    print("\ngripper aperture (measured from the jaw-tip link poses)")
    print(f"  handle width to straddle: {2 * HANDLE_HALF_WIDTH * 100:.1f} cm")
    # Tip separation is an UPPER bound on what fits: it is the mouth opening,
    # and the usable gap deeper between the blades is smaller (the SO-100 jaw
    # is a hinged pincer, so the two blades are only parallel near the middle
    # of their travel).
    print(f"  {'gripper qpos':>13} {'tip separation':>15}")
    agent = base.agent_b
    q = torch.zeros((base.num_envs, 6), device=base.device)
    q[:, :] = torch.as_tensor(READY_QPOS, dtype=q.dtype, device=base.device)
    for g in np.linspace(-1.1, 1.1, 12):
        q[:, 5] = float(g)
        agent.robot.set_qpos(q)
        base.scene._gpu_apply_all()
        base.scene.px.gpu_update_articulation_kinematics()
        base.scene._gpu_fetch_all()
        sep = torch.linalg.norm(agent.finger1_tip.pose.p - agent.finger2_tip.pose.p, dim=1)
        flag = "  <- mouth clears the handle" if 2 * HANDLE_HALF_WIDTH < float(sep[0]) else ""
        print(f"  {g:>13.2f} {float(sep[0]) * 100:>14.2f} cm{flag}")


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
    args = ap.parse_args()

    env = ManiSkillVectorEnv(
        gym.make(
            args.env_id,
            num_envs=args.envs,
            obs_mode="state_dict",
            sim_backend="physx_cuda",
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

    print(f"\nscripted expert on {args.env_id}, {args.envs} envs")
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
        for name, waypoint, grip, budget in PHASES:
            for i in range(budget):
                rot_arm = (
                    ROTATOR_PREGRASP
                    if waypoint == "pregrasp"
                    else ROTATOR_GRASP
                    if waypoint == "grasp"
                    else READY_QPOS[:5]
                )
                hold_arm = (
                    HOLDER_PREGRASP
                    if waypoint == "pregrasp"
                    else HOLDER_GRASP
                    if waypoint == "grasp"
                    else READY_QPOS[:5]
                )
                a_hold = _action(base.agent_a, hold_arm, grip, device)
                a_rot = _action(base.agent_b, rot_arm, grip, device)
                if name == "turn":
                    # Drive wrist_roll away from its grasp value by a quarter
                    # turn, in both directions across the env batch.
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
