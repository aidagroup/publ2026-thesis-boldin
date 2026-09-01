"""Pure geometry of the turntable cube: dimensions and grasp points.

Split out of `_turntable_cube` so it can be imported WITHOUT a simulator.
`_turntable_cube` needs sapien and mani_skill, which exist only on the Linux
+ CUDA server, so anything importing it is unavailable to CI -- and the thing
CI most needs to check is exactly this: that the scripted grasp waypoints in
`callosum.envs._so100_kinematics` still point at the handles these numbers
describe. Two server round trips were burned on waypoints that had silently
stopped matching the scene; see tests/test_grasp_waypoints.py.
"""

# ~5.7 cm real Rubik's cube edge length (matches callosum.envs.two_so100_base).
CUBE_HALF_SIZE = 0.0285
# A real 3x3 layer (one third of the cube), not an arbitrary thin plate.
FACE_THICKNESS = 2 * CUBE_HALF_SIZE / 3

# Half-width of both grasp handles, i.e. 2.0 cm across the jaws.
#
# WHY handles exist at all. The SO-100's jaw is a hinged pincer, not a
# parallel jaw, and its aperture was measured directly from
# mani_skill/assets/robots/so100/so100.urdf + the jaw collision meshes:
# projected into the Fixed_Jaw frame the gripper is planar, and the free gap
# between the two blades at a given insertion depth y (the fixed blade tip is
# at y = -0.1064, its root at y = -0.0389) is
#
#     depth y      -0.100   -0.090   -0.080   -0.070   -0.060
#     max gap       5.2 cm   5.8 cm   6.4 cm   7.0 cm   6.0 cm
#
# A 5.7 cm cube layer therefore only fits >= 3 cm back from the jaw tips. But
# the layer is only 1.9 cm tall and sits flush on a 3.8 cm body, and a
# top-down grasp (the one the design doc specifies, and the only one whose
# wrist_roll axis is vertical -- see face_turn.FaceTurn) puts the blade's
# 6.7 cm length along the vertical, so a blade deep enough to hold the layer
# unavoidably straddles the body as well and locks the joint. The bare cube
# is not graspable by this arm in the pose the task needs.
#
# 2.0 cm is inside the object scale mani-skill itself uses for the SO-100:
# PickCubeSO100-v1 uses cube_half_size 0.0125 (2.5 cm) and SO100GraspCube-v1
# randomizes 2.2-2.8 cm. At 2.0 cm the two gripping FACES are exactly one
# handle-width apart at gripper qpos -0.842 (`_so100_kinematics
# .SEATING_GRIPPER_QPOS`), mid-range of the joint's [-1.1, 1.1] travel, so
# there is room both to open before the grasp and to squeeze after it. Note
# that is the gap between the blade surfaces, which is 8 mm tighter than the
# jaw-TIP separation the tip links report -- the tip links sit ~2 mm inside
# the fingertips, and reading grasp width off them is what put the fixed blade
# through the handle on the 2026-08-30 probe run.
HANDLE_HALF_WIDTH = 0.010

# Height of the graspable section of BOTH posts. 3.5 cm: with the jaws closed
# on a 2 cm handle the tool centre point sits 9.72 cm straight down the blade
# from the Fixed_Jaw origin and the lowest point of the gripper mesh is only
# 9.2 mm below it, so pinching a post at its mid-height leaves 8.3 mm of
# clearance under the jaws -- enough for the face post to clear the cube's top
# face and for the body post to clear its own bridge.
FACE_HANDLE_HEIGHT = 0.035

# The body handle is a bridge out of the body's -y side (the holder's side)
# carrying a vertical post whose axis sits at this radius from the cube axis.
#
# A post rather than a plain horizontal bar, for two measured reasons.
# (1) A 2x2 cm post is graspable at ANY wrist_roll; the bar it replaced was
#     2 cm across x but 5.15 cm along y, so it could only be pinched if the
#     closing direction happened to be world x -- a knife edge on one joint
#     that the policy has no reason to respect.
# (2) It raises the grasp from 2.6 cm to 5.35 cm above the table, so the
#     holder's jaws have 4.4 cm of clearance instead of 1.7 cm.
#
# 8.0 cm radius, from three clearances measured on the gripper meshes -- note
# a finger blade is 2.8 cm across (Fixed_Jaw_part2.ply spans +-0.0139 m), so
# the blade, not its centreline, is what has to clear things. The
# rotating face sweeps a circle of radius CUBE_HALF_SIZE*sqrt(2) = 4.03 cm and
# the holder's gripper comes no closer than 7.35 cm to the axis inside that
# height band. The two grippers stay 3.3 cm apart at their grasp poses, and
# 1.47 cm apart at the worst point of the rotator's quarter turn, during which
# its gripper sweeps a 4.26 cm circle about the axis. And the radius is the
# holder's lever arm: resisting the 0.0835 N*m reaction torque (see
# FACE_JOINT_DAMPING) takes 1.04 N here against 2.93 N at the bare body's own
# 2.85 cm half-width.
BODY_HANDLE_RADIUS = 0.080
# Vertical gap between the top of the body handle's bridge and the underside
# of the rotating face, so the two never rub.
BODY_HANDLE_GAP = 0.002


def _bridge_top(cube_half_size: float, face_thickness: float) -> float:
    """Top of the body handle's bridge, in the body link's own frame.

    The body link's origin is the cube's rotation centre, so the underside of
    the rotating face sits at `cube_half_size - face_thickness` in this frame
    and the bridge stops `BODY_HANDLE_GAP` below it.
    """
    return cube_half_size - face_thickness - BODY_HANDLE_GAP


def _body_post_top(cube_half_size: float, face_thickness: float) -> float:
    """Top of the body handle's vertical post, in the body link's own frame."""
    return _bridge_top(cube_half_size, face_thickness) + FACE_HANDLE_HEIGHT


# Grasp points, as offsets in each link's own frame, for the reward's reach
# terms. These are points in FREE SPACE that the tool centre point can
# actually occupy; the link origins are not (they are buried inside solid
# geometry, which is what the reach reward used to aim at). Both assume the
# default cube dimensions, as build_turntable_cube's callers all use them.
#
# In world coordinates for a cube resting at the table centre this puts the
# rotator's target at (0, 0, 0.0745) and the holder's at (0, -0.080, 0.0535) --
# in both cases the mid-height of the post's 3.5 cm graspable section.
FACE_GRASP_OFFSET = (0.0, 0.0, FACE_THICKNESS / 2 + FACE_HANDLE_HEIGHT / 2)
BODY_GRASP_OFFSET = (
    0.0,
    -BODY_HANDLE_RADIUS,
    _body_post_top(CUBE_HALF_SIZE, FACE_THICKNESS) - FACE_HANDLE_HEIGHT / 2,
)

# The same two grasp points in WORLD coordinates, for a cube resting at the
# table centre with no spawn jitter (`TwoSO100Base._initialize_episode` puts
# the body link's origin at z = CUBE_HALF_SIZE). This is the pose the scripted
# waypoints in `_so100_kinematics` are solved against; at run time the env
# reads the live link poses instead (`FaceTurn.face_grasp_pos`).
_FACE_LINK_Z = CUBE_HALF_SIZE + (CUBE_HALF_SIZE - FACE_THICKNESS / 2)
FACE_GRASP_WORLD = (0.0, 0.0, _FACE_LINK_Z + FACE_GRASP_OFFSET[2])
BODY_GRASP_WORLD = (
    0.0,
    BODY_GRASP_OFFSET[1],
    CUBE_HALF_SIZE + BODY_GRASP_OFFSET[2],
)
