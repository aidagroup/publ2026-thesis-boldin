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

# ~5.7 cm real Rubik's cube edge length (matches callosum.envs.two_so100_base).
CUBE_HALF_SIZE = 0.0285
FACE_THICKNESS = 0.008

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
        # TODO(review): friction/damping copied from build_robel_valve (a
        # similarly hand-sized rotary mechanism) as a reasonable starting
        # point -- unverified for our smaller/lighter face; may need
        # retuning once actually simulated on the server.
        friction=0.02,
        damping=2.0,
    )

    return builder.build(name=name, fix_root_link=False)
