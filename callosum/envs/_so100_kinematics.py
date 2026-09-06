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
# Both arms face the cube along y: yaws pi and 0, NOT the panda pair's
# +-pi/2. A panda's home pose points along its own +x while the SO-100's
# points along its own -y (mani-skill compensates by giving single-SO100
# setups a base yaw of +pi/2 with the object at +x). Copying the panda numbers
# aimed both arms 90 deg away from the cube.
#
# 0.34, up from the 0.28 that suited the old top-down layout, because the arms
# now hold their tools HORIZONTAL and that costs reach at the near end. Swept
# over the whole joint range (scripts/solve_waypoints.py), a tool within 2 deg
# of horizontal cannot bring the jaws' pocket closer than ~0.26 m to its own
# base at any useful height: the arm has to be extended to point flat. At 0.28
# the holder could not reach the cube on the table at all, and at 0.32 its
# wrist_flex sat exactly on its 1.8 rad limit. At 0.34 every waypoint below
# solves exactly with >= 0.19 rad of margin on every joint.
#
# The x offsets put each arm's shoulder_pan plane through the line its tool
# has to lie on, so pan stays ~0 and the tool stays exactly horizontal. The
# holder's is the larger one because it grips the whole 5.7 cm body, whose
# axis is therefore 2.16 cm off its tool axis; the rotator grips a nub sized
# to sit ON its tool axis, so it needs almost nothing.
ARM_BASE_OFFSET = 0.34
HOLDER_BASE_YAW = np.pi
ROTATOR_BASE_YAW = 0.0
HOLDER_BASE_X = 0.0216
ROTATOR_BASE_X = 0.0

# --- what the jaws can hold, from the collision meshes -------------------
#
# The fixed blade never moves, so its gripping face is a hard wall at a fixed
# local x. Measured on Fixed_Jaw_part2.ply it is x = +0.0079, while
# `Fixed_Jaw_tip` is at x = +0.0100: the tip link is a point 2.1 mm INSIDE the
# fingertip. `SO100.tcp_pos` is the midpoint of those tip links, so aiming the
# tcp at a held object's axis buries the fixed blade in it -- which is exactly
# what the 2026-08-30 probe run showed.
#
# Re-derive any of this with `scripts/measure_gripper.py`.
FIXED_BLADE_FACE_X = 0.0079
# Insertion depth: 9.72 cm down the blade, where the jaws seat.
POCKET_DEPTH = -0.0972
# Slack in x so the open jaw can be lowered around an object without the fixed
# blade scraping it. Closing then nudges the object by this much, so it is
# 1 mm, not the 2 mm the old top-down layout used.
GRASP_CLEARANCE = 0.001
SEATING_GRIPPER_QPOS = -0.842


def pocket_offset(half_width: float) -> np.ndarray:
    """Where an object of this half-width must sit, in the Fixed_Jaw frame.

    NOT where the tool centre point is. An object seats against the fixed
    blade's face, so its axis lands `FIXED_BLADE_FACE_X - half_width` from the
    wrist_roll axis once the jaws close -- which is why the nub is sized at
    exactly `FIXED_BLADE_FACE_X` (see `_cube_geometry.NUB_HALF_WIDTH`): at that
    width, and only at that width, a roll spins it about its own axis.
    """
    return np.array([FIXED_BLADE_FACE_X - half_width - GRASP_CLEARANCE, POCKET_DEPTH, 0.0])


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

# --- scripted waypoints --------------------------------------------------
# Arm joints only (the gripper is commanded separately). Solved against
# `_cube_geometry` by `scripts/solve_waypoints.py`; tests/test_grasp_waypoints
# re-derives the resulting tool poses, so they cannot silently drift again.
#
# The order the task runs in: the holder comes down onto the cube where it
# spawns, closes, and LIFTS it to `LIFT_HEIGHT`; only then does the rotator
# come down onto the nub and roll.
#
# Both pre-grasps sit 3 cm straight ABOVE their grasp, and the last move is a
# vertical descent -- not a back-off along the tool axis, which is how the
# top-down layout used to approach. Two reasons, both measured:
#   * Backing off 5 cm along the tool means moving 5 cm closer to the arm's
#     own base, and that is precisely where a horizontal tool stops being
#     reachable: the solver lands 35 mm out with the wrist_flex on its limit.
#   * The jaws' slot is bounded in x by the two blades and open in z, so a
#     vertical descent drops the object into the slot without either blade
#     sweeping across it first.
PREGRASP_LIFT = 0.030
ROTATOR_PREGRASP = np.array([0.0035, -0.3070, 1.3653, -1.0580, 1.5651])
ROTATOR_GRASP = np.array([0.0035, 0.1886, 1.3103, -1.4998, 1.5754])
HOLDER_PREGRASP = np.array([0.0000, 0.5372, 0.9194, -1.4566, 1.5689])
HOLDER_GRASP = np.array([0.0000, 0.8418, 0.7706, -1.6124, 1.5756])
# Same grip, 6.15 cm higher: the cube's centre goes from resting on the table
# to `LIFT_HEIGHT`. The grip does not slide, so the tool rises by the same
# amount the cube does.
HOLDER_LIFT = np.array([0.0000, 0.1981, 1.0111, -1.2091, 1.5672])


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


def arm_base_pose(x: float, y: float, yaw: float) -> np.ndarray:
    """4x4 world pose of a robot base placed at (x, y, 0) with the given yaw."""
    return _transform(_rpy(0.0, 0.0, yaw), (x, y, 0.0))


HOLDER_BASE_POSE = arm_base_pose(HOLDER_BASE_X, -ARM_BASE_OFFSET, HOLDER_BASE_YAW)
ROTATOR_BASE_POSE = arm_base_pose(ROTATOR_BASE_X, ARM_BASE_OFFSET, ROTATOR_BASE_YAW)


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


def seated_object_position(qpos, base_pose: np.ndarray, half_width: float) -> np.ndarray:
    """Where the axis of an object of this half-width ends up at this pose."""
    fixed = fixed_jaw_pose(qpos, base_pose)
    return fixed[:3, 3] + fixed[:3, :3] @ pocket_offset(half_width)
