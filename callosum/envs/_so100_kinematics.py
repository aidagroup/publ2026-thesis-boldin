"""Pure-numpy forward kinematics for the SO-100, and the scripted grasp poses.

No simulator: every number here is transcribed from
`mani_skill/assets/robots/so100/so100.urdf` (v3.0.1) and from the jaw
collision meshes beside it. That is deliberate. ManiSkill has no IK on the GPU
backend (`Articulation.create_pinocchio_model` raises when GPU sim is
enabled), so `scripts/probe_grasp.py` has to drive the arms through
precomputed JOINT waypoints -- and a waypoint that has quietly stopped
matching the scene is invisible until a server round trip burns an hour. Two
such rounds were burned on 2026-08-30, so the same FK that solves the
waypoints now also checks them in CI (tests/test_grasp_waypoints.py).

Both failures were frame errors, and both are worth stating because they are
easy to repeat:

  * The chain below has FIVE joints. Stopping at `wrist_flex` leaves you on
    the Wrist_Pitch_Roll frame, 6 cm short of `Fixed_Jaw` along the tool
    axis, so every waypoint came out 6 cm too low -- the rotator drove its
    jaws into the cube and the holder drove its jaws into the tabletop.
  * `Fixed_Jaw_tip` and `Moving_Jaw_tip` are points INSIDE the fingertips,
    not the gripping surfaces. `SO100.tcp_pos` is their midpoint, so aiming
    the tcp at a handle's axis buries the fixed blade 2.1 mm inside it (see
    `GRASP_POCKET_OFFSET`). Position control wins that argument against a
    0.17 kg free-floating cube: the blade shoves the handle aside instead of
    straddling it.
"""

import numpy as np

# --- arm placement -------------------------------------------------------
#
# Distance of each arm's base from the table centre, along +-y, and the base
# yaws. Yaws pi and 0, NOT the panda pair's +pi/2 / -pi/2: a panda's home pose
# points along its own +x while the SO-100's points along its own -y
# (mani-skill compensates for that by giving the single-SO100 setups a base
# yaw of +pi/2 with the object at +x). Copying the panda numbers aimed both
# arms 90 deg away from the cube, which cost ~1.57 rad of shoulder_pan (~32
# control steps) on every episode and, worse, put the shoulder_pan axis --
# which sits 0.0452 m AHEAD of the base origin -- 0.3034 m from the cube axis
# while the Fixed_Jaw origin reaches only 0.2537 m at the rotator's grasp
# height with the tool pointing down. The top-down grasp the task needs was
# simply unreachable.
#
# 0.28 rather than 0.30 for the same reason, with the yaws already fixed:
#
#     base 0.30, yaws -+pi/2 (the original)   0.3034 m   short by 5.0 cm
#     base 0.30, yaws pi / 0                  0.2548 m   short by 1.1 mm
#     base 0.28, yaws pi / 0                  0.2348 m   1.9 cm of margin
#
# 0.28 also survives the cube's +-1 cm spawn jitter (wrist_flex stays between
# 1.63 and 1.77 rad against its 1.8 limit) and still leaves 3.5 cm between the
# two grippers at the start pose.
ARM_BASE_OFFSET = 0.28
HOLDER_BASE_YAW = np.pi
ROTATOR_BASE_YAW = 0.0

# Start configuration for both arms: mani-skill's own SO-100 ready pose, used
# by TableSceneBuilder's "so100" branch and by SO100GraspCube-v1.
#
# NOT SO100.keyframes["rest"] ([0, -1.5708, 1.5708, 0.66, 0, -1.1]), which the
# env used until 2026-08-30. Two measured problems with "rest":
#   * qpos[5] = -1.1 is the gripper joint's LOWER limit, and the jaw tips are
#     then 6.6 mm apart -- the arm starts with the hand CLAMPED SHUT, and
#     nothing in the dense reward pays for opening it. Worse, SO100.tcp_pos is
#     the midpoint of the two jaw tips, so the gripper joint MOVES the reach
#     reward's own measurement point: at the "rest" arm pose, sweeping the
#     gripper from -1.1 to +1.1 lifts the tcp by 6.5 cm, straight away from a
#     cube whose centre is 4.75 cm off the table. Opening the hand was
#     therefore locally reward-NEGATIVE, and `grasped` stayed 0.00.
#   * its approach axis is 38 deg off vertical. At this pose the approach is
#     exactly (0, 0, -1) and the wrist_roll axis exactly (0, 0, 1), which is
#     what turning the face about a vertical axis requires.
READY_QPOS = np.array([0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 0.0])

# --- kinematic chain, verbatim from so100.urdf ---------------------------
# (origin xyz, origin rpy, joint axis) for the five arm joints, in the order
# mani-skill's controller uses (`robot.active_joints`).
_ARM_CHAIN = (
    ((0.0, -0.0452, 0.0165), (1.5708, 0.0, 0.0), (0.0, -1.0, 0.0)),  # shoulder_pan
    ((0.0, 0.1025, 0.0306), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),  # shoulder_lift
    ((0.0, 0.11257, 0.028), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),  # elbow_flex
    ((0.0, 0.0052, 0.1349), (-1.5708, 0.0, 0.0), (1.0, 0.0, 0.0)),  # wrist_flex
    ((0.0, -0.0601, 0.0), (0.0, 1.5708, 0.0), (0.0, 1.0, 0.0)),  # wrist_roll
)
_GRIPPER_JOINT = ((-0.0202, -0.0244, 0.0), (0.0, 3.14159, -0.9), (0.0, 0.0, 1.0))
_FIXED_JAW_TIP = np.array([0.01, -0.097, 0.0])
_MOVING_JAW_TIP = np.array([-0.01, -0.073, 0.0])

# lower/upper for shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
# wrist_roll, gripper.
JOINT_LIMITS = np.array(
    [
        [-2.0, 2.0],
        [-1.5708, 1.5708],
        [-1.5708, 1.5708],
        [-1.8, 1.8],
        [-3.14159, 3.14159],
        [-1.1, 1.1],
    ]
)

# Where a grasp handle's axis must sit in the Fixed_Jaw frame -- NOT where the
# tool centre point is.
#
# The fixed blade never moves, so its gripping face is a hard wall at a fixed
# local x. Measured on Fixed_Jaw_part2.ply, at the depth below that face sits
# at x = +0.0079, while `Fixed_Jaw_tip` is at x = +0.0100: the tip link is a
# point 2.1 mm INSIDE the fingertip. Centring a 2 cm handle on the closed tcp
# (local x = -0.0001) therefore overlaps the blade by 2.1 mm, which is exactly
# what the 2026-08-30 probe run showed -- both arms settled ~2.7 cm from their
# handles, the same miss on both, with the cube shoved (dpos 0.012 -> 0.019)
# at the moment the jaws tried to close.
#
# x = fixed face - HANDLE_HALF_WIDTH - 2 mm of approach clearance, so the open
# jaw slides down past the handle with 1.9 mm to spare and only the MOVING jaw
# touches it on closing. y is the insertion depth: 9.72 cm down the blade,
# where the aperture reaches 2.0 cm at gripper qpos -0.842, mid-range of its
# [-1.1, 1.1] travel, and the lowest point of the gripper mesh is only 9.2 mm
# further down -- which is what leaves the jaws 8.3 mm of clearance over the
# cube's top face and over the body handle's bridge.
GRASP_POCKET_OFFSET = np.array([-0.00410, -0.09720, 0.0])
SEATING_GRIPPER_QPOS = -0.842

# --- scripted waypoints --------------------------------------------------
# Arm joints only (the gripper is commanded separately). Solved by the FK
# below against `_cube_geometry`'s handle positions; tests/test_grasp_waypoints
# re-derives and checks them, so they cannot silently drift out of date again.
#
# wrist_roll stays at pi/2, the value READY_QPOS already holds, so no roll
# travel is needed to reach either grasp. At pi/2 the tool approaches straight
# down and the jaws close along world x -- which also means the moving jaw,
# which swings 8 cm out along the closing axis when open, swings in x rather
# than in y, away from the cube and away from the other arm. Both handles are
# square in cross-section, so the grasp itself does not depend on the roll.
#
# The pre-grasp waypoints sit above the grasp waypoints (1 cm for the rotator,
# 3 cm for the holder) so the last move onto the handle is a vertical descent.
# The rotator cannot have more: at a 1.5 cm lift its wrist_flex is already
# pinned at the 1.8 rad limit.
PREGRASP_LIFT_ROTATOR = 0.010
PREGRASP_LIFT_HOLDER = 0.030
_ROLL = np.pi / 2
ROTATOR_PREGRASP = np.array([0.0175, 0.4393, -0.6463, 1.7779, _ROLL])
ROTATOR_GRASP = np.array([0.0175, 0.4126, -0.5296, 1.6878, _ROLL])
HOLDER_PREGRASP = np.array([0.0265, -0.3362, 0.3281, 1.5789, _ROLL])
HOLDER_GRASP = np.array([0.0265, -0.3191, 0.5354, 1.3545, _ROLL])


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw, i.e. Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll),
        np.sin(roll),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw),
        np.sin(yaw),
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _transform(rotation: np.ndarray, translation) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def _axis_rotation(axis, angle: float) -> np.ndarray:
    """Rodrigues rotation about `axis` (assumed already a unit URDF axis)."""
    a = np.asarray(axis, dtype=float)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def arm_base_pose(y: float, yaw: float) -> np.ndarray:
    """4x4 world pose of a robot base placed at (0, y, 0) with the given yaw."""
    return _transform(_rpy(0.0, 0.0, yaw), (0.0, y, 0.0))


HOLDER_BASE_POSE = arm_base_pose(-ARM_BASE_OFFSET, HOLDER_BASE_YAW)
ROTATOR_BASE_POSE = arm_base_pose(ARM_BASE_OFFSET, ROTATOR_BASE_YAW)


def fixed_jaw_pose(qpos, base_pose: np.ndarray) -> np.ndarray:
    """4x4 world pose of the `Fixed_Jaw` link. Uses the first five joints."""
    out = np.asarray(base_pose, dtype=float).copy()
    for i, (xyz, rpy, axis) in enumerate(_ARM_CHAIN):
        out = (
            out @ _transform(_rpy(*rpy), xyz) @ _transform(_axis_rotation(axis, qpos[i]), (0, 0, 0))
        )
    return out


def approach_direction(qpos, base_pose: np.ndarray) -> np.ndarray:
    """Unit vector the tool points along -- the Fixed_Jaw frame's -y axis."""
    return -fixed_jaw_pose(qpos, base_pose)[:3, 1]


def tcp_position(qpos, base_pose: np.ndarray) -> np.ndarray:
    """`SO100.tcp_pos`: the midpoint of the two jaw-tip links. Needs 6 joints."""
    fixed = fixed_jaw_pose(qpos, base_pose)
    xyz, rpy, axis = _GRIPPER_JOINT
    moving = (
        fixed @ _transform(_rpy(*rpy), xyz) @ _transform(_axis_rotation(axis, qpos[5]), (0, 0, 0))
    )
    tip1 = (fixed @ np.append(_FIXED_JAW_TIP, 1.0))[:3]
    tip2 = (moving @ np.append(_MOVING_JAW_TIP, 1.0))[:3]
    return (tip1 + tip2) / 2


def seated_handle_position(qpos, base_pose: np.ndarray) -> np.ndarray:
    """Where a handle's axis ends up if this pose is a correct grasp."""
    fixed = fixed_jaw_pose(qpos, base_pose)
    return fixed[:3, 3] + fixed[:3, :3] @ GRASP_POCKET_OFFSET
