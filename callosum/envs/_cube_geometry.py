"""Pure geometry of the turntable cube: dimensions and grasp points.

Split out of `_turntable_cube` so it can be imported WITHOUT a simulator.
`_turntable_cube` needs sapien and mani_skill, which exist only on the Linux
+ CUDA server, so anything importing it is unavailable to CI -- and the thing
CI most needs to check is exactly this: that the scripted grasp waypoints in
`callosum.envs._so100_kinematics` still point at what these numbers describe.
Several server round trips were burned on waypoints that had silently stopped
matching the scene; see tests/test_grasp_waypoints.py.

LAYOUT (rewritten 2026-09-06). The rotating face is the one that FACES THE
ROTATOR, so the rotation axis is horizontal, along world y. The holder grips
the other two layers -- its own face plus the middle -- from its own side, and
LIFTS the cube off the table before the rotator turns anything.

The lift is not decoration. Turning the face means rolling the rotator's
wrist about the cube's axis, and the gripper's own collision geometry reaches
4.26 cm from that axis (`scripts/measure_gripper.py`, the `Fixed_Jaw_part1`
corner). With the cube resting on the table its axis is only 2.85 cm up, so a
quarter turn drives the gripper's palm through the tabletop. Held at
`LIFT_HEIGHT` the same sweep clears the table by 4.7 cm.
"""

import numpy as np

# ~5.7 cm real Rubik's cube edge length (matches callosum.envs.two_so100_base).
CUBE_HALF_SIZE = 0.0285
# A real 3x3 layer (one third of the cube), not an arbitrary thin plate.
FACE_THICKNESS = 2 * CUBE_HALF_SIZE / 3

# The cube is split across y: the rotator's side is the turning layer, the
# remaining two thirds are one rigid body for the holder to hold.
FACE_SPLIT_Y = CUBE_HALF_SIZE - FACE_THICKNESS

# Height of the cube's CENTRE while the holder holds it up, i.e. the height of
# the rotation axis during the turn. Set by the rotator's gripper, not by the
# cube: at the closed gripper the outermost collision pad sits 4.26 cm from the
# wrist_roll axis, so a quarter turn sweeps a 4.26 cm circle about the cube's
# axis. 9.0 cm leaves 4.7 cm of tabletop clearance, and the cube's own lowest
# swept point (its half-diagonal, 4.03 cm) clears by 5.0 cm.
LIFT_HEIGHT = 0.090

# --- the nub ------------------------------------------------------------
#
# The one feature the bare cube cannot do without, and the reason it is this
# size rather than any other.
#
# A face turn is a wrist ROLL, and the SO-100's roll axis is the Fixed_Jaw
# frame's own y axis -- the line x = z = 0 -- while the fixed blade's gripping
# face is the plane x = +0.0079. Anything the jaws hold seats against that
# face, so its centre lands `width/2 - 0.0079` off the roll axis. Rolling the
# wrist therefore spins a NARROW object about itself and swings a wide one
# around an arc: a bare 5.7 cm layer would come out 2.06 cm off axis and be
# dragged 2.9 cm sideways over a quarter turn, against a hinge that cannot go
# anywhere. (Clamping it is not the problem -- `scripts/measure_gripper.py`
# shows the jaws close on a bare layer perfectly well. Turning it is.)
#
# 2 * 0.0079 is therefore the widest object that sits EXACTLY on the roll axis,
# and that is what this is: a square peg on the centre of the rotator's face,
# on the axis, one facelet-ish across (a real 5.7 cm cube's facelet is 1.9 cm).
NUB_HALF_WIDTH = 0.0079
# How far it stands out from the face. The jaws seat 9.72 cm down the blade
# and the blade tip is 0.92 cm beyond that, so a grip centred 1.5 cm along the
# nub leaves the blade tip 5.8 mm clear of the cube's face -- which it must be,
# because that tip sweeps a 3.96 cm circle across the face during the turn.
NUB_LENGTH = 0.025
NUB_GRASP_DEPTH = 0.015

# --- grasp points -------------------------------------------------------
#
# Offsets from each link's origin to the point the arm's TOOL CENTRE POINT
# should occupy, which is what `SO100.tcp_pos` reports and what the reward's
# reach terms and `scripts/probe_grasp.py` measure. They are not the links'
# origins (buried in solid geometry) and not the jaws' pocket either -- the
# pocket is where the held object's axis goes, and for the holder, which grips
# the whole 5.7 cm body, that axis is 2.16 cm from its own tool axis.
#
# Both are derived in `scripts/solve_waypoints.py` from the solved waypoints
# and re-checked in CI, so they cannot drift away from the joint angles.

# Face link frame: its origin is the cube's rotation centre, so the nub's axis
# is the x = z = 0 line and the grasp sits along +y.
FACE_GRASP_OFFSET = (0.0, CUBE_HALF_SIZE + NUB_GRASP_DEPTH, 0.0)

# Body link frame: same origin. The holder's tool axis runs 1.94 cm to +x of
# the cube's axis (its blade lies flat on the body's -x face), 4.5 mm back
# from the cube's centre plane in y, and 9.6 mm above the centre -- gripping
# a little above centre keeps the blade inside the body's height and still
# lets the cube hang stably.
BODY_GRASP_OFFSET = (0.0194, -0.0045, 0.0096)

# The same two points in WORLD coordinates for the poses the scripted
# waypoints are solved against: the holder grasps the cube where it spawns, on
# the table, and the rotator only ever meets it once it has been lifted.
BODY_GRASP_WORLD = (
    BODY_GRASP_OFFSET[0],
    BODY_GRASP_OFFSET[1],
    CUBE_HALF_SIZE + BODY_GRASP_OFFSET[2],
)
BODY_GRASP_LIFTED = (
    BODY_GRASP_OFFSET[0],
    BODY_GRASP_OFFSET[1],
    LIFT_HEIGHT + BODY_GRASP_OFFSET[2],
)
FACE_GRASP_LIFTED = (
    FACE_GRASP_OFFSET[0],
    FACE_GRASP_OFFSET[1],
    LIFT_HEIGHT + FACE_GRASP_OFFSET[2],
)


def gripper_sweep_clearance(axis_height: float) -> float:
    """Tabletop clearance under the rotator's gripper during the quarter turn.

    `GRIPPER_SWEEP_RADIUS` is measured, not assumed -- run
    `scripts/measure_gripper.py` to re-derive it from the URDF collision pads.
    Negative means the palm goes through the table.
    """
    return axis_height - GRIPPER_SWEEP_RADIUS


# Largest distance from the wrist_roll axis to any of the gripper's four
# collision pads, with the jaws closed. The corner of `Fixed_Jaw_part1`.
GRIPPER_SWEEP_RADIUS = 0.0426
# Half-diagonal of the cube's own square cross-section: what the rotating
# layer itself sweeps about the axis.
CUBE_SWEEP_RADIUS = float(CUBE_HALF_SIZE * np.sqrt(2))
