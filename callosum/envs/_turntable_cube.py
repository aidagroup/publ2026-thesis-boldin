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

from callosum.envs._cube_geometry import (
    BODY_HANDLE_RADIUS,
    CUBE_HALF_SIZE,
    FACE_HANDLE_HEIGHT,
    FACE_THICKNESS,
    HANDLE_HALF_WIDTH,
    _body_post_top,
    _bridge_top,
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

    Each link also carries a 2 cm square grasp post (see `HANDLE_HALF_WIDTH`):
    one on the face's rotation axis for the rotator, and one on a bridge out of
    the body's -y side for the holder. Both are square in cross-section, so
    they can be pinched at any wrist_roll. Contact with a post is contact with
    its link, so `agent.is_grasping(link)` needs no special-casing.

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
    # Holder's handle: a bridge out of the -y side carrying a vertical post.
    bridge_top = _bridge_top(cube_half_size, face_thickness)
    bridge_half_len = (BODY_HANDLE_RADIUS - cube_half_size) / 2
    bridge_pose = sapien.Pose(
        [0, -(cube_half_size + bridge_half_len), bridge_top - HANDLE_HALF_WIDTH]
    )
    bridge_half_size = [HANDLE_HALF_WIDTH, bridge_half_len, HANDLE_HALF_WIDTH]
    body.add_box_collision(pose=bridge_pose, half_size=bridge_half_size)
    body.add_box_visual(
        pose=bridge_pose,
        half_size=bridge_half_size,
        material=sapien.render.RenderMaterial(base_color=handle_color),
    )
    # The post runs all the way down to the table. That foot is load-bearing,
    # not cosmetic: with the handle cantilevered the combined body+handle
    # centre of mass sits at y = -1.97 cm, only 0.89 cm inside the body's own
    # 2.85 cm footprint edge, so a downward press of just
    # 1.69 N * 0.0089 / 0.0515 = 0.29 N at the handle would tip the whole cube
    # over -- and pressing down is exactly what a top-down holder does.
    # Standing the post on the table extends the support polygon to y = -9 cm.
    # It rests on a 2x2 cm patch, so the extra table friction it adds is small
    # next to the ~0.0835 N*m the holder still has to supply.
    post_top = _body_post_top(cube_half_size, face_thickness)
    # Table to post top. The body link's origin is cube_half_size above the
    # table whenever the cube is resting on it, which is the pose the shape is
    # designed for; the foot lifts off if the cube is ever picked up.
    post_height = post_top + cube_half_size
    post_pose = sapien.Pose([0, -BODY_HANDLE_RADIUS, post_height / 2 - cube_half_size])
    post_half_size = [HANDLE_HALF_WIDTH, HANDLE_HALF_WIDTH, post_height / 2]
    body.add_box_collision(pose=post_pose, half_size=post_half_size)
    body.add_box_visual(
        pose=post_pose,
        half_size=post_half_size,
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
