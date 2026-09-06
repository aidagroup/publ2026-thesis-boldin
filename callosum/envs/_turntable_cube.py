"""Builder for the "turntable cube": a body + one revolute-jointed side face.

This is the "turntable-abstraction" from docs/thesis/04-experiment-design.md:
a simplified twisty-puzzle mechanism (one turnable face) instead of a full
26-cubie Rubik's cube, used by callosum.envs.face_turn.

The turning layer is the one FACING THE ROTATOR, so the rotation axis is
horizontal, along world y. The holder grips the other two layers -- its own
face plus the middle -- as one rigid body, and there is nothing on them to
grip by: the arm clamps the bare 5.7 cm cube perfectly well
(`scripts/measure_gripper.py`).

The single exception is the nub on the turning layer, and it is not a grasp
aid but a rotation one. See `_cube_geometry.NUB_HALF_WIDTH`: the SO-100's
wrist_roll axis passes 0.79 cm from the fixed blade's gripping face, so
anything wider than 1.58 cm ends up off that axis and gets dragged around an
arc instead of spun in place. A bare layer would come out 2.06 cm off.
"""

import numpy as np
import sapien
import sapien.render
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.articulation import Articulation

from callosum.envs._cube_geometry import (
    CUBE_HALF_SIZE,
    FACE_THICKNESS,
    NUB_HALF_WIDTH,
    NUB_LENGTH,
)

# Rotates the joint's local X axis (SAPIEN's default joint-rotation axis) to
# point along the link's own Y axis. Same device as
# mani_skill.utils.building.articulations.robel.build_robel_valve -- the only
# hand-built revolute-joint articulation in mani-skill 3.0.1 (turn_faucet.py
# loads a pre-built PartNet-Mobility URDF instead, so it doesn't demonstrate
# this). Using the SAME rotation for both pose_in_parent and pose_in_child
# cancels out at qpos=0 (parent_to_child = R * R^-1 = identity), so the face
# stays axis-aligned with the body while still rotating about y as qpos moves.
# A +90 deg turn about z takes x to y; quaternions here are (w, x, y, z).
_Y_AXIS_QUAT = [float(np.cos(np.pi / 4)), 0.0, 0.0, float(np.sin(np.pi / 4))]

# Joint friction (N*m) and viscous damping (N*m*s/rad) of the face joint.
#
# The original values (0.02 / 2.0) were copied from build_robel_valve, which
# is turned by a 9-DOF D'Claw hand pushing on 6 cm capsules. Sized for this
# task instead: an episode is 300 steps at control_freq 20 Hz = 15 s, so
# leaving ~10 s for the pick, the lift and the approach the quarter turn runs
# at ~0.3 rad/s, and the drive torque is damping*0.3 + friction. At 2.0 that is
# 0.6 N*m, well past the stall torque of the SO-100's wrist servo -- the face
# behaved like it was set in glue and gave the policy almost no angle signal
# per unit of applied torque. At 0.5 it is 0.155 N*m, inside the arm's
# capability.
#
# With the cube held in the air rather than resting on the table, this torque
# is now carried ENTIRELY by the holder's grip -- there is no table friction to
# help, which is what makes the two arms genuinely coupled.
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
    nub_color=(0.1, 0.1, 0.9, 1),
    scene_idxs=None,
) -> Articulation:
    """Build a "turntable cube": a box body plus a revolute-jointed side face.

    The body is the articulation's root link and is free-floating (NOT fixed
    to the world, unlike e.g. build_robel_valve's mounted stand) so it can be
    grasped, lifted, and knocked around like a real cube. The face is the
    layer at +y -- the rotator's side -- and turns about the horizontal y axis
    through the cube's centre, with limits [-pi/2, pi/2].

    Symmetric limits, not [0, pi/2]: which way a wrist_roll drives the joint
    is not knowable from the builder without simulating, and with a one-sided
    limit the wrong sign simply does nothing while looking like a physics
    failure. Success is on |angle| (see face_turn.FaceTurn.evaluate), so
    either direction counts and the sign convention stops mattering.

    Args:
        scene: the ManiSkillScene to build into.
        name: unique name for the built articulation.
        cube_half_size: half the edge length of the overall cube.
        face_thickness: thickness (full, not half) of the turnable layer.
        color: RGBA of the body.
        face_color: RGBA of the face, so it's visually distinguishable.
        nub_color: RGBA of the rotator's nub.
        scene_idxs: which parallel envs to build this in (None = all envs).

    Returns:
        The built Articulation. Its root link is named "body" and its
        rotating link is named "face" (fetch via `.links_map`).
    """
    body_half_depth = cube_half_size - face_thickness / 2

    builder = scene.create_articulation_builder()
    builder.set_scene_idxs(scene_idxs)

    body = builder.create_link_builder(parent=None)
    body.set_name("body")
    # The body occupies the -y (2*cube_half_size - face_thickness) slab, so
    # together with the face it reconstructs the full cube.
    body_pose = sapien.Pose([0, -face_thickness / 2, 0])
    body_half_size = [cube_half_size, body_half_depth, cube_half_size]
    body.add_box_collision(pose=body_pose, half_size=body_half_size)
    body.add_box_visual(
        pose=body_pose,
        half_size=body_half_size,
        material=sapien.render.RenderMaterial(base_color=color),
    )

    # Adjacent parent/child links connected by a joint are excluded from
    # mutual collision by SAPIEN by default (same as robot arm links), so no
    # explicit disable_self_collisions should be needed here; watch for
    # body/face jitter on the server if that assumption is wrong.
    face = builder.create_link_builder(body)
    face.set_name("face")
    # The face link's own origin is its geometric centre (see pose_in_child
    # below), so its shape needs no local offset.
    face_half_size = [cube_half_size, face_thickness / 2, cube_half_size]
    face.add_box_collision(half_size=face_half_size)
    face.add_box_visual(
        half_size=face_half_size,
        material=sapien.render.RenderMaterial(base_color=face_color),
    )
    # The nub: a square peg on the rotation axis, standing out of the face
    # toward the rotator. On the axis, so the wrist_roll torque goes straight
    # into the joint instead of through a lever arm -- and, more to the point,
    # so a wrist_roll spins it rather than swinging it around a 2 cm arc.
    nub_pose = sapien.Pose([0, (face_thickness + NUB_LENGTH) / 2, 0])
    nub_half_size = [NUB_HALF_WIDTH, NUB_LENGTH / 2, NUB_HALF_WIDTH]
    face.add_box_collision(pose=nub_pose, half_size=nub_half_size)
    face.add_box_visual(
        pose=nub_pose,
        half_size=nub_half_size,
        material=sapien.render.RenderMaterial(base_color=nub_color),
    )
    face.set_joint_name("face_joint")
    face.set_joint_properties(
        type="revolute",
        limits=[[-np.pi / 2, np.pi / 2]],
        pose_in_parent=sapien.Pose([0, cube_half_size - face_thickness / 2, 0], q=_Y_AXIS_QUAT),
        pose_in_child=sapien.Pose(q=_Y_AXIS_QUAT),
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
