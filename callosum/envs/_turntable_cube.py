"""Builder for the "turntable cube": a body + one revolute-jointed top face.

This is the "turntable-abstraction" from docs/thesis/04-experiment-design.md:
a simplified twisty-puzzle mechanism (one turnable face) instead of a full
26-cubie Rubik's cube, used by callosum.envs.face_turn.

Both links carry an explicit GRASP HANDLE. That is not decoration: measured
from mani-skill 3.0.1's own SO-100 assets, the arm cannot pinch the bare
5.7 cm cube at all (see `HANDLE_HALF_WIDTH` below), which is why every run up
to 2026-08-30 reported `success_once == 0` with `grasped == 0.00`.
"""

import numpy as np
import sapien
import sapien.render
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.articulation import Articulation

# ~5.7 cm real Rubik's cube edge length (matches callosum.envs.two_so100_base).
CUBE_HALF_SIZE = 0.0285
# A real 3x3 layer (one third of the cube), not an arbitrary thin plate.
FACE_THICKNESS = 2 * CUBE_HALF_SIZE / 3

# Half-thickness of an SO-100 finger blade, from the collision meshes
# (Fixed_Jaw_part2.ply spans +-0.0139 m across the blade, Moving_Jaw_part2/3
# +-0.0115 m). Used to place grasp points far enough from obstacles that the
# whole blade, not just its centreline, has clearance.
JAW_BLADE_HALF_THICKNESS = 0.014

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
# randomizes 2.2-2.8 cm. At 2.0 cm the jaws close on the handle at gripper
# qpos ~ -0.85, mid-range of its [-1.1, 1.1] travel, so there is room both to
# open before the grasp and to squeeze after it.
HANDLE_HALF_WIDTH = 0.010

# The face handle is a square post standing on the layer, centred on the
# rotation axis. Square, not round, so the pinch is form-closed against the
# reaction torque instead of relying on finger friction alone. 3.5 cm tall so
# that when the jaws pinch its mid-height the blade tips (~1 cm past the
# pinch) still clear the cube's top face by ~7 mm.
FACE_HANDLE_HEIGHT = 0.035

# The body handle is a horizontal bar out of the body's -y side (the holder's
# side). Its far end sits at this radius from the cube axis. The rotating face
# sweeps a circle of radius CUBE_HALF_SIZE*sqrt(2) = 4.03 cm, so a holder
# blade centred at BODY_HANDLE_TIP - JAW_BLADE_HALF_THICKNESS = 6.6 cm spans
# 5.2-8.0 cm and never enters the face's swept volume.
BODY_HANDLE_TIP = 0.080
# Vertical gap between the top of the body handle and the underside of the
# rotating face, so the two never rub.
BODY_HANDLE_GAP = 0.002


def _body_handle_centre_z(cube_half_size: float, face_thickness: float) -> float:
    """Centre height of the body's grasp bar, in the body link's own frame.

    The body link's origin is the cube's rotation centre, `face_thickness / 2`
    above the body box's centre, so the face's underside sits at
    `cube_half_size - face_thickness` in this frame. The bar hangs
    `BODY_HANDLE_GAP` below that and is `2 * HANDLE_HALF_WIDTH` tall.
    """
    return cube_half_size - face_thickness - BODY_HANDLE_GAP - HANDLE_HALF_WIDTH


# Grasp points, as offsets in each link's own frame, for the reward's reach
# terms. These are points in FREE SPACE that the tool centre point can
# actually occupy; the link origins are not (they are buried inside solid
# geometry, which is what the reach reward used to aim at). Both assume the
# default cube dimensions, as build_turntable_cube's callers all use them.
#
# In world coordinates for a cube resting at the table centre that puts the
# rotator's target at (0, 0, 0.0745) -- the mid-height of the face post -- and
# the holder's at (0, -0.066, 0.026), one jaw-blade half-thickness in from the
# body bar's far end.
FACE_GRASP_OFFSET = (0.0, 0.0, FACE_THICKNESS / 2 + FACE_HANDLE_HEIGHT / 2)
BODY_GRASP_OFFSET = (
    0.0,
    -(BODY_HANDLE_TIP - JAW_BLADE_HALF_THICKNESS),
    _body_handle_centre_z(CUBE_HALF_SIZE, FACE_THICKNESS),
)

# Rotates the joint's local X axis (SAPIEN's default joint-rotation axis) to
# point along the link's own Z axis, i.e. vertical. Copied verbatim from
# mani_skill.utils.building.articulations.robel.build_robel_valve -- the only
# hand-built revolute-joint articulation in mani-skill 3.0.1 (turn_faucet.py
# loads a pre-built PartNet-Mobility URDF instead, so it doesn't demonstrate
# this). Using the SAME rotation for both pose_in_parent and pose_in_child
# cancels out at qpos=0 (parent_to_child = R * R^-1 = identity, verified by
# hand), so the face ends up axis-aligned with the body while still rotating
# about a vertical axis as qpos changes.
_VERTICAL_AXIS_QUAT = [0.707, 0, 0.707, 0]

# Joint friction (N*m) and viscous damping (N*m*s/rad) of the face joint.
#
# The previous values (0.02 / 2.0) were copied from build_robel_valve, which
# is turned by a 9-DOF D'Claw hand pushing on 6 cm capsules. Sized for this
# task instead: an episode is 300 steps at control_freq 20 Hz = 15 s, so
# leaving ~5 s for approach and grasp the quarter turn runs at ~0.157 rad/s,
# and the drive torque is damping*0.157 + friction. At 2.0 that is 0.33 N*m,
# roughly the stall torque of the SO-100's wrist servo -- the face behaved
# like it was set in glue and gave the policy almost no angle signal per unit
# of applied torque. At 0.5 it is 0.089 N*m, comfortably inside the arm's
# capability, while still ~7x the ~0.013 N*m that table friction alone
# resists on the body (mu 0.3, ~0.19 kg cube), so the holder's grip is still
# load-bearing -- which is the entire point of the bimanual task.
# TODO(review): both numbers are analytic, not simulated; re-check with
# scripts/probe_grasp.py on the server.
FACE_JOINT_FRICTION = 0.005
FACE_JOINT_DAMPING = 0.5


def build_turntable_cube(
    scene: ManiSkillScene,
    name: str = "turntable_cube",
    cube_half_size: float = CUBE_HALF_SIZE,
    face_thickness: float = FACE_THICKNESS,
    color=(1, 0, 0, 1),
    face_color=(1, 1, 0, 1),
    handle_color=(0.1, 0.1, 0.9, 1),
    scene_idxs=None,
) -> Articulation:
    """Build a "turntable cube": a box body plus a revolute-jointed top face.

    The body is the articulation's root link and is free-floating (NOT fixed
    to the world, unlike e.g. build_robel_valve's mounted stand) so it can be
    grasped, lifted, and knocked around like a real cube. The face sits on
    top and rotates about a vertical axis through the cube's center, with
    joint limits [0, pi/2] -- a quarter turn.

    Each link also carries a 2 cm grasp handle (see `HANDLE_HALF_WIDTH`): a
    post on the face's rotation axis for the rotator, and a side bar on the
    body for the holder. Contact with a handle is contact with its link, so
    `agent.is_grasping(link)` needs no special-casing.

    Args:
        scene: the ManiSkillScene to build into.
        name: unique name for the built articulation.
        cube_half_size: half the edge length of the overall cube.
        face_thickness: thickness (full, not half) of the turnable top face.
        color: RGBA of the body.
        face_color: RGBA of the face, so it's visually distinguishable.
        handle_color: RGBA of both grasp handles.
        scene_idxs: which parallel envs to build this in (None = all envs).

    Returns:
        The built Articulation. Its root link is named "body" and its
        rotating top link is named "face" (fetch via `.links_map`).
    """
    body_half_height = cube_half_size - face_thickness / 2

    builder = scene.create_articulation_builder()
    builder.set_scene_idxs(scene_idxs)

    body = builder.create_link_builder(parent=None)
    body.set_name("body")
    # Body occupies the bottom (2*cube_half_size - face_thickness) slab, so
    # together with the face it reconstructs the full cube height.
    body.add_box_collision(
        pose=sapien.Pose([0, 0, -face_thickness / 2]),
        half_size=[cube_half_size, cube_half_size, body_half_height],
    )
    body.add_box_visual(
        pose=sapien.Pose([0, 0, -face_thickness / 2]),
        half_size=[cube_half_size, cube_half_size, body_half_height],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    # Holder's handle: a bar reaching out of the -y side to BODY_HANDLE_TIP,
    # its top BODY_HANDLE_GAP below the rotating face.
    body_handle_half_len = (BODY_HANDLE_TIP - cube_half_size) / 2
    body_handle_pose = sapien.Pose(
        [
            0,
            -(cube_half_size + body_handle_half_len),
            _body_handle_centre_z(cube_half_size, face_thickness),
        ]
    )
    body_handle_half_size = [HANDLE_HALF_WIDTH, body_handle_half_len, HANDLE_HALF_WIDTH]
    body.add_box_collision(pose=body_handle_pose, half_size=body_handle_half_size)
    body.add_box_visual(
        pose=body_handle_pose,
        half_size=body_handle_half_size,
        material=sapien.render.RenderMaterial(base_color=handle_color),
    )

    # Adjacent parent/child links connected by a joint are excluded from
    # mutual collision by SAPIEN by default (same as robot arm links), so no
    # explicit disable_self_collisions should be needed here; watch for
    # body/face jitter on the server if that assumption is wrong.
    face = builder.create_link_builder(body)
    face.set_name("face")
    # The face link's own origin is placed at its geometric center (see
    # pose_in_child below), so its shape needs no local offset.
    face.add_box_collision(
        half_size=[cube_half_size, cube_half_size, face_thickness / 2],
    )
    face.add_box_visual(
        half_size=[cube_half_size, cube_half_size, face_thickness / 2],
        material=sapien.render.RenderMaterial(base_color=face_color),
    )
    # Rotator's handle: a square post on the rotation axis, so the wrist_roll
    # torque goes straight into the joint instead of through a lever arm.
    face_handle_pose = sapien.Pose([0, 0, (face_thickness + FACE_HANDLE_HEIGHT) / 2])
    face_handle_half_size = [HANDLE_HALF_WIDTH, HANDLE_HALF_WIDTH, FACE_HANDLE_HEIGHT / 2]
    face.add_box_collision(pose=face_handle_pose, half_size=face_handle_half_size)
    face.add_box_visual(
        pose=face_handle_pose,
        half_size=face_handle_half_size,
        material=sapien.render.RenderMaterial(base_color=handle_color),
    )
    face.set_joint_name("face_joint")
    face.set_joint_properties(
        type="revolute",
        limits=[[0, np.pi / 2]],
        pose_in_parent=sapien.Pose(
            [0, 0, cube_half_size - face_thickness / 2], q=_VERTICAL_AXIS_QUAT
        ),
        pose_in_child=sapien.Pose(q=_VERTICAL_AXIS_QUAT),
        friction=FACE_JOINT_FRICTION,
        damping=FACE_JOINT_DAMPING,
    )

    # Without this the builder defaults to p=[0,0,0] -- the cube half-buried in
    # the table at build time -- and mani_skill warns that the initial pose may
    # collide with other objects. Match the nominal placement used per episode
    # (TwoSO100Base._initialize_episode puts it at z=CUBE_HALF_SIZE with a small
    # xy jitter); the per-episode set_pose still overrides this.
    builder.initial_pose = sapien.Pose(p=[0, 0, cube_half_size])
    return builder.build(name=name, fix_root_link=False)
