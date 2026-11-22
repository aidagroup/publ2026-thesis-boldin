"""Builder for the multi-face cube: a free core plus six revolute-jointed outer layers.

The generalization of `callosum.experiments.arm.turntable` to all six faces, driven by
`callosum.experiments.arm.skills_cube_config.MultiFaceCubeConfig`: the core is the articulation's
root link (grasped by the holder, lifted and flipped), each labeled face U/D/L/R/F/B
carries one outer layer on a revolute joint whose axis is the face normal in the body
frame, with limits [0, 180] deg, friction from the layer spec and detents at the
quarter-turn multiples. Plan moves are executed by turning the corresponding layer.

Geometry decision (the load-bearing one, see the design's risk list): each layer is a
FULL face slab spanning the whole cube cross-section, so the six slabs OVERLAP in the
edge/corner regions exactly like a real cube's shared cubies would. This is
rotation-consistent -- a square slab rotated about its own normal occupies the same
region, so the layers never change their relative overlap however far they turn -- but
it means the articulation MUST be built with self-collisions disabled: the overlapping
slabs would otherwise start in deep interpenetration and the contact solver would tear
the cube apart. External collisions (table, grippers) are unaffected.

Consequences accepted with this geometry (all TODO(review) on the first cloud probe):
- Contact/grasp detection on a slab's side can be ambiguous where two slabs' surfaces
  coincide; `is_grasping(layer_link)` may see only part of the jaw force.
- A layer turn drags the up-slab past the side slabs' surfaces with no contact between
  them (self-collisions off), i.e. the turn resists only through its own joint
  friction/damping -- the layer lock (env-side) is the actual guarantee.

The joint-frame trick is the turntable's: `pose_in_parent` and `pose_in_child` share the
same rotation (mapping SAPIEN's default joint X axis onto the face normal), which cancels
at qpos = 0, so the layer starts axis-aligned with the core while still rotating about
its face normal as qpos changes.
"""

import numpy as np
import sapien
import sapien.physx
import sapien.render
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.articulation import Articulation

from callosum.experiments.arm.skills_cube_config import FACE_NORMALS, MultiFaceCubeConfig

# Distinct face colors (the real cube's convention) so a rendered rollout shows which
# layer is which; irrelevant for state observations.
FACE_COLORS: dict[str, tuple[float, float, float, float]] = {
    "U": (1.0, 1.0, 1.0, 1.0),  # white
    "D": (1.0, 1.0, 0.0, 1.0),  # yellow
    "L": (1.0, 0.5, 0.0, 1.0),  # orange
    "R": (1.0, 0.0, 0.0, 1.0),  # red
    "F": (0.0, 0.7, 0.0, 1.0),  # green
    "B": (0.0, 0.0, 1.0, 1.0),  # blue
}
CORE_COLOR = (0.35, 0.35, 0.35, 1.0)

DEFAULT_LAYER_DAMPING = 0.1
"""Joint damping (N*m*s/rad) of every layer joint; `LayerSpec` has no damping field (its
contract is frozen), so the builder takes one value for all layers -- the same 0.1 the
one-face turntable runs with (`FaceTurnPhysicsConfig.face_damping`)."""


def _axis_quat(axis: tuple[float, float, float]) -> list[float]:
    """Quaternion (xyzw) rotating SAPIEN's default joint axis +x onto `axis` (unit).

    Any single rotation that maps +x onto `axis` works, because using it for BOTH
    `pose_in_parent` and `pose_in_child` cancels at qpos = 0 (the turntable's
    `_VERTICAL_AXIS_QUAT` is this function at axis = +z, verified there by hand). The
    180-degree rotation about the bisector of +x and `axis` is exact for the +-axis
    vectors used here; the antiparallel case (+x -> -x) falls back to 180 degrees about
    +y, which also maps +x onto -x.
    """
    x = np.array([1.0, 0.0, 0.0])
    a = np.asarray(axis, dtype=float)
    if np.allclose(a, -x):
        # Antiparallel: rotate 180 deg about y (x -> -x, z -> -z, y fixed).
        return [0.0, 1.0, 0.0, 0.0]
    bisector = x + a
    bisector /= np.linalg.norm(bisector)
    # 180-degree rotation about `bisector`: q = (bisector, 0) in xyzw.
    return [float(bisector[0]), float(bisector[1]), float(bisector[2]), 0.0]


def build_multi_face_cube(
    scene: ManiSkillScene,
    name: str = "multi_face_cube",
    config: MultiFaceCubeConfig | None = None,
    scene_idxs=None,
    initial_pose: sapien.Pose | None = None,
    layer_damping: float = DEFAULT_LAYER_DAMPING,
) -> Articulation:
    """Build the core-plus-six-layers cube articulation into `scene`.

    Args:
        scene: the ManiSkillScene to build into.
        name: unique name for the built articulation.
        config: the cube's geometry/joint configuration (axes, limits, friction,
            detents); defaults to `MultiFaceCubeConfig()`.
        scene_idxs: which parallel envs to build this in (None = all envs).
        initial_pose: pose of the core (root link) at build time. Default: resting on
            the table at the table centre (root at z=cube_half_size), matching the
            nominal reset pose of TwoSO101Base (ManiSkill warns if unset).
        layer_damping: damping of every layer joint (see `DEFAULT_LAYER_DAMPING`).

    Returns:
        The built Articulation. Root link "core"; one link per face named
        `layer_<face>` with joint `<face>_layer_joint` (fetch via `.links_map` /
        `.joints_map`). Child-link order (and therefore the qpos column order of
        `set_qpos`/`get_qpos`) follows `config.layers`, i.e. FACE_NORMALS order
        U, D, L, R, F, B by default.
    """
    config = MultiFaceCubeConfig() if config is None else config
    hs = config.cube_half_size
    t = config.layer_thickness
    gap = config.layer_gap
    # The explicit contact material: with PhysX's defaults the cube glides across
    # the table at constant speed after any lateral kick (A40 probe measurement --
    # the flip's set-down flicked it 30+ cm off the table edge).
    material = sapien.physx.PhysxMaterial(
        static_friction=config.contact_static_friction,
        dynamic_friction=config.contact_dynamic_friction,
        restitution=config.contact_restitution,
    )

    builder = scene.create_articulation_builder()
    builder.set_scene_idxs(scene_idxs)
    builder.initial_pose = sapien.Pose(p=[0, 0, hs]) if initial_pose is None else initial_pose
    # The six full-face slabs overlap at the edges/corners (see the module doc): without
    # disabling self-collisions the articulation starts in deep interpenetration.
    # TODO(review): verify the attribute name against ManiSkill 3.0.1's
    # ArticulationBuilder on the first cloud probe (the turntable never needed it); the
    # alternative spelling is builder.set_self_collisions(False).
    builder.disable_self_collisions = True

    # The core: the centred box that remains when all six layer slabs are cut free,
    # shrunk by half the gap so no collision geometry interpenetrates the core.
    core = builder.create_link_builder(parent=None)
    core.set_name("core")
    core_half = hs - t - gap / 2
    core.add_box_collision(
        half_size=[core_half, core_half, core_half],
        material=material,
    )
    core.add_box_visual(
        half_size=[core_half, core_half, core_half],
        material=sapien.render.RenderMaterial(base_color=CORE_COLOR),
    )

    lo, hi = config.joint_limits_rad
    for spec in config.layers:
        face = spec.face
        normal = np.asarray(FACE_NORMALS[face], dtype=float)
        quat = _axis_quat(spec.axis)
        # The layer slab: centred at `center` (offset outward by half the gap), spanning
        # the full cube cross-section in the two lateral axes.
        center = normal * (hs - t / 2 + gap / 2)
        lateral = [abs(c) < 0.5 for c in normal]
        half = [hs if lat else t / 2 for lat in lateral]
        layer = builder.create_link_builder(core)
        layer.set_name(f"layer_{face}")
        layer.add_box_collision(
            half_size=half,
            material=material,
        )
        layer.add_box_visual(
            half_size=half,
            material=sapien.render.RenderMaterial(base_color=FACE_COLORS[face]),
        )
        layer.set_joint_name(f"{face}_layer_joint")
        layer.set_joint_properties(
            type="revolute",
            limits=[[lo, hi]],
            pose_in_parent=sapien.Pose(p=center.tolist(), q=quat),
            pose_in_child=sapien.Pose(q=quat),
            friction=spec.friction,
            damping=layer_damping,
        )

    cube = builder.build(name=name, fix_root_link=False)
    # ManiSkill v3.0.1's articulation builder silently drops the friction of
    # `set_joint_properties` (the turntable hit the same), so set it on the built
    # joints, before the GPU sim is initialised.
    # TODO(review): GPU backend unverified; PhysX joint friction is a per-joint
    # property, so it must be set here at build time, not later.
    for spec in config.layers:
        cube.joints_map[f"{spec.face}_layer_joint"].set_friction(spec.friction)
    return cube
