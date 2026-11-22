"""Pure geometry and command-projection helpers of the physical TossCatch paddles.

The `TossCatch-v0` paddles are no longer kinematic MOCAP plates: each one is a rigid body on
two prismatic joints (court x and vertical z) plus a hinge (the tilt about world +y), driven by
three bounded-force position servos (`callosum/experiments/coop_tasks/assets/toss_catch.xml`).
The env has two command-level jobs that this module keeps pure and testable without a
simulator:

* **servo-target mapping** -- a normalized `(dx, dz, dtheta)` command is a bounded pose
  increment of the servo TARGET (`pad_speed * control_dt`, `pad_tilt_rate * control_dt`,
  scaled by the agent's private authority); the target is clipped to the joint workspace.
  The plate physically tracks the target -- the env never writes a joint position directly.
* **partner-aware target projection** -- because the two plates are 0.6 m wide, a pair of
  legal per-paddle targets can still ask both plates to occupy the same space near the
  court centre. The env projects such a pair apart along x (the axis the agents command)
  before writing it to the servos. This is only an anti-windup safety: the MuJoCo
  paddle/paddle contact (the stiff collision shells in the XML) is the real constraint, so a
  plate can never pass through the partner plate -- the projection just stops a policy from
  pushing both servos into a ram that the contact solver would have to resolve.

Everything here is ordinary Python floats / numpy, so it is unit-tested on the Mac and used
by the CPU env directly; the GPU backend mirrors the same arithmetic on torch tensors
(`callosum.experiments.coop_tasks.env_gpu`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PADDLE_CONTACT_MARGIN = 0.03
"""Extra x clearance (m) the target projection keeps between the two plates' x intervals.

The projection moves the servo TARGETS apart, while the plates physically lag their targets
by up to ~1 cm each during a fast move (and the float32 GPU mirror adds a few more
millimetres), so the margin covers that tracking lag: with the margin the measured plate
separation stays above the projection of the two hulls on x, and the stiff pad/pad contact
shells never have to resolve a commanded overlap. This is not a collision distance -- the
physical contact is still the final constraint if a plate is pushed.
"""


def projected_half_extent(half_a: float, half_b: float, angle: float) -> float:
    """Half width of a box's projection on one in-plane axis for a y-rotation `angle`.

    A box with in-plane half extents `half_a` (along the un-rotated axis) and `half_b` (the
    perpendicular in-plane axis) rotated by `angle` about the out-of-plane y axis projects to
    `half_a * |cos angle| + half_b * |sin angle|`.

    Args:
        half_a: Half extent along the axis being projected (m).
        half_b: Half extent along the perpendicular in-plane axis (m).
        angle: Rotation about the out-of-plane y axis (rad); the paddle tilt.

    Returns:
        The half width of the projection (m).
    """
    return half_a * abs(math.cos(angle)) + half_b * abs(math.sin(angle))


def projected_half_x(tilt: float, half_x: float, half_z: float) -> float:
    """Half width (m) of the paddle's projection on the court x axis at tilt `tilt`."""
    return projected_half_extent(half_x, half_z, tilt)


def projected_half_z(tilt: float, half_x: float, half_z: float) -> float:
    """Half height (m) of the paddle's projection on the vertical z axis at tilt `tilt`."""
    return projected_half_extent(half_z, half_x, tilt)


def intervals_overlap(centre_a: float, half_a: float, centre_b: float, half_b: float) -> bool:
    """Whether the 1-D intervals `a +- half_a` and `b +- half_b` overlap (with no margin)."""
    return abs(centre_a - centre_b) < half_a + half_b


def paddle_pose_overlap(
    x_left: float,
    z_left: float,
    tilt_left: float,
    x_right: float,
    z_right: float,
    tilt_right: float,
    *,
    half_x: float,
    half_z: float,
) -> bool:
    """Conservative 2-D overlap test of the two paddle boxes in the court x-z plane.

    The boxes are separated when either their x projections or their z projections are
    disjoint; when both projections overlap they may still be separated along the tilted face
    normals, so this test is conservative (it can report an overlap for boxes that a full
    oriented-box SAT would find separated). That is the right direction for the target
    projection: the only cost of a false positive is a slightly wider centre gap.

    Args:
        x_left: Left paddle centre x (m).
        z_left: Left paddle centre z (m).
        tilt_left: Left paddle tilt (rad).
        x_right: Right paddle centre x (m).
        z_right: Right paddle centre z (m).
        tilt_right: Right paddle tilt (rad).
        half_x: Paddle half extent along its face x axis (m, XML-owned).
        half_z: Paddle half thickness (m, XML-owned).

    Returns:
        True when the conservative overlap test says the two boxes can intersect.
    """
    half_x_left = projected_half_x(tilt_left, half_x, half_z)
    half_x_right = projected_half_x(tilt_right, half_x, half_z)
    half_z_left = projected_half_z(tilt_left, half_x, half_z)
    half_z_right = projected_half_z(tilt_right, half_x, half_z)
    return intervals_overlap(x_left, half_x_left, x_right, half_x_right) and intervals_overlap(
        z_left, half_z_left, z_right, half_z_right
    )


@dataclass(frozen=True)
class PaddleTarget:
    """One paddle's clipped servo target `(x, z, tilt)`.

    Attributes:
        x: Target x joint position (m).
        z: Target z joint position (m).
        tilt: Target hinge angle (rad).
    """

    x: float
    z: float
    tilt: float


def servo_target(
    current: PaddleTarget,
    command: np.ndarray,
    *,
    max_step: float,
    max_dtilt: float,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    tilt_limit: float,
) -> PaddleTarget:
    """The clipped servo target of one paddle for a normalized `(dx, dz, dtheta)` command.

    The command is a bounded pose increment on the servo TARGET (the previous target, not
    the measured pose): with a zero command the target does not move, so the servo keeps
    holding the plate against gravity instead of integrating the sag away.

    Args:
        current: The previous servo target.
        command: Normalized `(dx, dz, dtheta)` command, already in `[-1, 1]`.
        max_step: Largest target translation per control step (m).
        max_dtilt: Largest target tilt change per control step (rad).
        x_range: The paddle's `(min, max)` x joint range (m).
        z_range: The paddle's `(min, max)` z joint range (m).
        tilt_limit: Absolute tilt limit (rad).

    Returns:
        The clipped target.
    """
    vector = np.asarray(command, dtype=np.float64).reshape(-1)
    if vector.shape != (3,):
        raise ValueError(f"command must have shape (3,), got {vector.shape}")
    x = float(np.clip(current.x + vector[0] * max_step, *x_range))
    z = float(np.clip(current.z + vector[1] * max_step, *z_range))
    tilt = float(np.clip(current.tilt + vector[2] * max_dtilt, -tilt_limit, tilt_limit))
    return PaddleTarget(x, z, tilt)


def project_targets_apart(
    left: PaddleTarget,
    right: PaddleTarget,
    *,
    half_x: float,
    half_z: float,
    left_x_range: tuple[float, float],
    right_x_range: tuple[float, float],
    margin: float = PADDLE_CONTACT_MARGIN,
) -> tuple[PaddleTarget, PaddleTarget]:
    """Project a pair of servo targets apart when their paddle boxes would overlap in x-z.

    Both targets are moved outward along x (each toward its own side of the court) until the
    x projections have at least the sum of the two projected half widths plus `margin`. A
    side that is already at its joint limit stops moving and the other side takes the rest
    of the required separation; if the workspace cannot provide the full separation (it
    always can for the task's ranges) the targets are left as separated as the ranges allow
    and the physical contact remains the constraint.

    The z projections decide whether the pair can collide at all: two plates at different
    heights cannot touch, so no projection is applied then.

    Args:
        left: Left paddle's clipped target.
        right: Right paddle's clipped target.
        half_x: Paddle half extent along its face x axis (m, XML-owned).
        half_z: Paddle half thickness (m, XML-owned).
        left_x_range: Left paddle `(min, max)` x joint range (m).
        right_x_range: Right paddle `(min, max)` x joint range (m).
        margin: Extra x clearance to keep (m).

    Returns:
        The possibly moved `(left, right)` targets.
    """
    if not paddle_pose_overlap(
        left.x,
        left.z,
        left.tilt,
        right.x,
        right.z,
        right.tilt,
        half_x=half_x,
        half_z=half_z,
    ):
        return left, right
    required = (
        projected_half_x(left.tilt, half_x, half_z)
        + projected_half_x(right.tilt, half_x, half_z)
        + margin
    )
    deficit = required - (right.x - left.x)
    if deficit <= 0.0:
        return left, right
    # First move both targets symmetrically outward; a side already at its limit gives its
    # share to the other side.
    left_room = left.x - left_x_range[0]
    right_room = right_x_range[1] - right.x
    symmetric = min(deficit / 2.0, left_room, right_room)
    left_x = left.x - symmetric
    right_x = right.x + symmetric
    remaining = deficit - 2.0 * symmetric
    if remaining > 0.0:
        extra_left = min(remaining, left_room - symmetric)
        left_x -= extra_left
        remaining -= extra_left
        extra_right = min(remaining, right_room - symmetric)
        right_x += extra_right
    return (
        PaddleTarget(left_x, left.z, left.tilt),
        PaddleTarget(right_x, right.z, right.tilt),
    )


def paddle_pair_geometry(
    left: PaddleTarget,
    right: PaddleTarget,
    *,
    half_x: float,
    half_z: float,
) -> tuple[float, float]:
    """Separating-axis measure of the two paddle boxes in the court x-z plane.

    A paddle is a box rotated about the out-of-plane y axis only, so its footprint in the
    x-z plane is a rectangle with half extents `half_x` (its face half width) and `half_z`
    (its half thickness). Two such rectangles intersect exactly when all four candidate
    separating axes (world x, world z, and each box's face normal) have positive interval
    overlap; this function returns the SAT measure:

    * `penetration` -- the smallest interval overlap when the boxes intersect (`0` when they
      do not): the maximum geometric interpenetration depth along a separating axis;
    * `separation` -- the largest interval gap when the boxes are disjoint (`0` when they
      intersect): a lower bound of the true Euclidean distance (equal to it when the nearest
      feature is a face, smaller for corner-to-corner gaps).

    The measure is pure geometry: it does not use the simulator's contact list, so it is the
    same number on the CPU and the GPU envs and can be reported as evidence that the two
    plates never overlap.

    Args:
        left: Left paddle pose `(x, z, tilt)`.
        right: Right paddle pose `(x, z, tilt)`.
        half_x: Paddle half extent along its face x axis (m, XML-owned).
        half_z: Paddle half thickness (m, XML-owned).

    Returns:
        `(penetration, separation)` in metres, exactly one of which is `0.0`.
    """
    delta = np.array([right.x - left.x, right.z - left.z], dtype=np.float64)
    frames = []
    for tilt in (left.tilt, right.tilt):
        normal = np.array([math.sin(tilt), math.cos(tilt)], dtype=np.float64)
        tangent = np.array([math.cos(tilt), -math.sin(tilt)], dtype=np.float64)
        frames.append((normal, tangent))
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    axes.extend(normal for normal, _tangent in frames)
    axes.extend(tangent for _normal, tangent in frames)
    overlaps = []
    for axis in axes:
        extents = [
            half_x * abs(float(tangent @ axis)) + half_z * abs(float(normal @ axis))
            for normal, tangent in frames
        ]
        overlaps.append(extents[0] + extents[1] - abs(float(delta @ axis)))
    narrowest = min(overlaps)
    if narrowest <= 0.0:
        # Some separating axis exists: the boxes are disjoint and the largest axis gap is a
        # lower bound of the true distance.
        return 0.0, -narrowest
    return narrowest, 0.0
