"""Config of the multi-layer cube: a core body plus six outer layers (E2+).

The generalization of the one-face "turntable" of `callosum.envs._turntable_cube` to
every face: the core is the articulation's root link (grasped and flipped by the
holder), each labeled face U/D/L/R/F/B carries one outer layer on a revolute joint
whose axis is the face normal in the body frame. Plan moves are executed by turning
the corresponding layer 90/180 deg; only outer layers ever turn.

Pure dataclass logic on purpose (the same macOS rule as `_skills`/`_plans`): the
builder that turns this into a ManiSkill articulation is env-side work (task 2.1)
and consumes this config; the geometry/axis/detent contracts are validated here so
misconfiguration fails at construction, not inside the simulator.
"""

import math
from dataclasses import dataclass, field

FACE_NORMALS: dict[str, tuple[float, float, float]] = {
    "U": (0.0, 0.0, 1.0),
    "D": (0.0, 0.0, -1.0),
    "L": (-1.0, 0.0, 0.0),
    "R": (1.0, 0.0, 0.0),
    "F": (0.0, 1.0, 0.0),
    "B": (0.0, -1.0, 0.0),
}
"""Unit normal of each labeled face in the cube's BODY frame (the same face labeling
`callosum.envs._plans` uses; the labels travel with the body when it is flipped)."""

LAYER_JOINT_LIMITS_DEG = (0.0, 180.0)
"""Every layer joint turns clockwise-only [0, 180] deg: a quarter and a half turn are
single moves, a counterclockwise (prime) plan move is 90 + 180 (see `_plans`). The
limit is the same for all six layers, so it is a constant, not a per-layer field."""

_AXIS_TOL = 1e-6
"""Tolerance for the unit-length and collinearity checks (exact +-axis vectors in
practice; float dust only)."""


@dataclass(frozen=True)
class LayerSpec:
    """One outer layer: the slice of the cube that turns about its face's axis.

    The joint's positive direction is along `axis`; the sign (parallel or
    anti-parallel to the face normal) only flips which world rotation reads as a
    positive joint angle -- a [0, 180] span covers both 90-deg directions of a layer.
    """

    face: str
    """Which side of the core this layer sits on: one of U/D/L/R/F/B."""

    axis: tuple[float, float, float]
    """Revolute-joint axis in the body frame: a unit vector along +-x/+-y/+-z,
    collinear with the face's normal (`FACE_NORMALS[face]`)."""

    friction: float = 20.0
    """PhysX articulation-joint friction coefficient (dimensionless; the value the
    one-face turntable runs with, see `FaceTurnPhysicsConfig.face_friction`)."""

    detents_deg: tuple[float, ...] = (0.0, 90.0, 180.0)
    """Rest angles (deg) the layer settles at -- the quarter-turn multiples inside
    the joint limits. The env's success tolerances are measured against these."""

    lock_unless_held: bool = True
    """The layer may turn only while the holder grasps the core (the turntable's
    face-lock rule lifted to every layer; a kinematic guarantee, not a reward)."""

    def __post_init__(self) -> None:
        if self.face not in FACE_NORMALS:
            raise ValueError(f"face must be one of {tuple(FACE_NORMALS)}, got {self.face!r}")
        normal = FACE_NORMALS[self.face]
        norm = math.sqrt(sum(component * component for component in self.axis))
        if abs(norm - 1.0) > _AXIS_TOL:
            raise ValueError(
                f"axis of layer {self.face} must be a unit vector, got {self.axis} (|a| = {norm})"
            )
        dot = sum(a * n for a, n in zip(self.axis, normal, strict=True))
        if abs(abs(dot) - 1.0) > _AXIS_TOL:
            raise ValueError(
                f"axis of layer {self.face} must be collinear with the face normal "
                f"{normal}, got {self.axis}"
            )
        if self.friction < 0:
            raise ValueError(f"friction must be >= 0, got {self.friction}")
        lo, hi = LAYER_JOINT_LIMITS_DEG
        detents = self.detents_deg
        if not all(lo <= d <= hi for d in detents) or list(detents) != sorted(set(detents)):
            raise ValueError(
                f"detents_deg must be strictly increasing and within {LAYER_JOINT_LIMITS_DEG} "
                f"deg, got {detents}"
            )

    @property
    def detents_rad(self) -> tuple[float, ...]:
        """The detent angles in radians (the simulator's joint-angle unit)."""
        return tuple(math.radians(d) for d in self.detents_deg)


def default_layers() -> tuple[LayerSpec, ...]:
    """The six layers with axes on their face normals and default friction/detents."""
    return tuple(LayerSpec(face, normal) for face, normal in FACE_NORMALS.items())


@dataclass(frozen=True)
class MultiFaceCubeConfig:
    """Geometry and joint setup of the core-plus-six-layers cube.

    Sizes follow the one-face turntable (`callosum.envs._turntable_cube`): a ~5.7 cm
    cube (fits the SO-ARM101 workspace) with layers a real third of the edge thick,
    so each layer is pinchable by the parallel gripper.
    """

    cube_half_size: float = 0.0285
    """Half the cube's edge (m); the overall cube spans 2 * cube_half_size."""

    layer_thickness: float = 2 * 0.0285 / 3
    """Full thickness (m) of one outer layer: a real third of the edge (what makes a
    layer graspable), also leaving a core of the remaining third per axis."""

    layer_gap: float = 0.0005
    """Clearance (m) between the core slab and each layer, so the articulation does
    not start with interpenetrating collision geometry."""

    layers: tuple[LayerSpec, ...] = field(default_factory=default_layers)
    """The six outer layers; must cover each face exactly once."""

    contact_static_friction: float = 1.0
    """Static friction of the cube's contact material (core and slabs). Without an
    explicit material the collision boxes get PhysX defaults, and on the A40 probe
    the cube GLIDED across the table at constant speed after the flip's set-down
    tipped it off the jaw -- a lateral kick must die within millimetres, not carry
    the cube off the table (body_pos_tol is 1-2 cm)."""

    contact_dynamic_friction: float = 0.8
    """Dynamic friction of the cube's contact material."""

    contact_restitution: float = 0.12
    """Restitution of the cube's contact material: the set-down drops the cube the
    last ~2 cm onto the table/jaw, and a bouncy cube would drift and tip."""

    def __post_init__(self) -> None:
        if self.cube_half_size <= 0:
            raise ValueError(f"cube_half_size must be > 0, got {self.cube_half_size}")
        if not 0 < self.layer_thickness < 2 * self.cube_half_size:
            raise ValueError(
                f"layer_thickness must be within (0, {2 * self.cube_half_size}) m, "
                f"got {self.layer_thickness}"
            )
        if not 0 <= self.layer_gap < self.layer_thickness:
            raise ValueError(
                f"layer_gap must be within [0, {self.layer_thickness}) m, got {self.layer_gap}"
            )
        for name in ("contact_static_friction", "contact_dynamic_friction", "contact_restitution"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        faces = [layer.face for layer in self.layers]
        if len(faces) != 6 or set(faces) != set(FACE_NORMALS):
            raise ValueError(f"layers must cover each of the 6 faces exactly once, got {faces}")

    @property
    def joint_limits_rad(self) -> tuple[float, float]:
        """The layers' shared joint limits in radians."""
        lo, hi = LAYER_JOINT_LIMITS_DEG
        return (math.radians(lo), math.radians(hi))

    def layer(self, face: str) -> LayerSpec:
        """The spec of `face`'s layer (unknown face: KeyError)."""
        for layer in self.layers:
            if layer.face == face:
                return layer
        raise KeyError(f"no layer for face {face!r}")
