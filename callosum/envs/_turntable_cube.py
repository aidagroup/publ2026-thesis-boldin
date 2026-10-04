"""Builder for the "turntable cube": a body + one revolute-jointed top face.

This is the "turntable-abstraction" from docs/thesis/04-experiment-design.md:
a simplified twisty-puzzle mechanism (one turnable face) instead of a full
26-cubie Rubik's cube, used by callosum.envs.face_turn.
"""

import numpy as np
import sapien
import sapien.render
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.articulation import Articulation

# ~5.7 cm real Rubik's cube edge length (matches callosum.envs.two_so101_base).
CUBE_HALF_SIZE = 0.0285
# A real 3x3 layer (one third of the cube), not an arbitrary thin plate --
# also what makes the face actually graspable: a 1.9 cm layer can be pinched
# from its side faces by a parallel-jaw gripper, an 8 mm plate essentially
# can't.
FACE_THICKNESS = 2 * CUBE_HALF_SIZE / 3

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


def build_turntable_cube(
    scene: ManiSkillScene,
    name: str = "turntable_cube",
    cube_half_size: float = CUBE_HALF_SIZE,
    face_thickness: float = FACE_THICKNESS,
    color=(1, 0, 0, 1),
    face_color=(1, 1, 0, 1),
    scene_idxs=None,
    initial_pose: sapien.Pose | None = None,
    face_friction: float = 20.0,
    face_damping: float = 0.1,
) -> Articulation:
    """Build a "turntable cube": a box body plus a revolute-jointed top face.

    The body is the articulation's root link and is free-floating (NOT fixed
    to the world, unlike e.g. build_robel_valve's mounted stand) so it can be
    grasped, lifted, and knocked around like a real cube. The face sits on
    top and rotates about a vertical axis through the cube's center, with
    joint limits [0, pi/2] -- a quarter turn.

    Args:
        scene: the ManiSkillScene to build into.
        name: unique name for the built articulation.
        cube_half_size: half the edge length of the overall cube.
        face_thickness: thickness (full, not half) of the turnable top face.
        color: RGBA of the body.
        face_color: RGBA of the face, so it's visually distinguishable.
        scene_idxs: which parallel envs to build this in (None = all envs).
        initial_pose: pose of the body (root link) at build time. Default: resting on the
            table (z=0) at the table centre, i.e. body root at z=cube_half_size (the body box
            spans local z in [-cube_half_size, cube_half_size - face_thickness], so its
            bottom sits exactly cube_half_size below the root). This matches the nominal
            reset pose in TwoSO101Base._initialize_episode; ManiSkill warns if it is unset.
        face_friction: PhysX friction coefficient of the face joint (dimensionless, see
            `FaceTurnPhysicsConfig.face_friction`).
        face_damping: damping of the face joint (N*m*s/rad); see the note on the gripper below.

    Returns:
        The built Articulation. Its root link is named "body" and its
        rotating top link is named "face" (fetch via `.links_map`).
    """
    body_half_height = cube_half_size - face_thickness / 2

    builder = scene.create_articulation_builder()
    builder.set_scene_idxs(scene_idxs)
    builder.initial_pose = (
        sapien.Pose(p=[0, 0, cube_half_size]) if initial_pose is None else initial_pose
    )

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
    face.set_joint_name("face_joint")
    face.set_joint_properties(
        type="revolute",
        limits=[[0, np.pi / 2]],
        pose_in_parent=sapien.Pose(
            [0, 0, cube_half_size - face_thickness / 2], q=_VERTICAL_AXIS_QUAT
        ),
        pose_in_child=sapien.Pose(q=_VERTICAL_AXIS_QUAT),
        # Damping is far lower than the valve's 2.0: with
        # 2.0 the SO-ARM101 parallel gripper cannot turn the face (jaws slip, face_angle stalls
        # at ~0.3 rad), with 0.1-0.2 a 90 degree wrist roll turns it fully (CPU sim, body held
        # still); 0.5 already lags, 1.0 slips.
        # TODO(review): re-check on the GPU backend and with a real holder arm.
        friction=face_friction,
        damping=face_damping,
    )

    cube = builder.build(name=name, fix_root_link=False)
    # ManiSkill v3.0.1's articulation builder only applies the damping of `set_joint_properties`
    # (as a velocity drive); the friction is silently dropped (the valve builder's friction is
    # a no-op too), so set it on the built joint, before the GPU sim is initialised.
    # TODO(review): GPU backend unverified; PhysX joint friction is a per-joint property, so it
    # must be set here at build time, not later.
    cube.joints_map["face_joint"].set_friction(face_friction)
    return cube
