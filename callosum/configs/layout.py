"""Arm placement around the table centre for the two-arm environments."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ArmLayout:
    """Where the two SO-ARM101 bases stand on the table.

    Each base sits on a circle around the table centre, where the cube is, and points at the
    centre. Azimuths are world angles in radians, measured counter-clockwise from +x, so azimuth
    0 is the point `(radius, 0)` and -pi/2 is `(0, -radius)`. `agent_a` (the holder in
    FaceTurn-v0) stands at `azimuth_a` / `radius_a`, `agent_b` (the rotator) at `azimuth_b` /
    `radius_b`.

    The defaults are the layout found with `scripts/probe_face_turn.py`, where a scripted
    holder and rotator complete the face turn on the CPU sim: the arms are 90 degrees apart (the
    holder at -y, the rotator at +x). The arms can not face each other, because the holder would
    then clamp the body from the rotator's side and their wrist housings (12.8 cm wide) collide
    above the cube. They also need different radii: the rotator comes in top-down and reaches at
    most ~0.27 m, while the holder clamps the body low and almost horizontally, which needs the
    cube at least ~0.28 m from its base and puts its wrist housing well behind the cube.
    """

    radius_a: float = 0.32
    radius_b: float = 0.24
    azimuth_a: float = -math.pi / 2
    azimuth_b: float = 0.0


def base_xy_yaw(radius: float, azimuth: float) -> tuple[float, float, float]:
    """Base position `(x, y)` and yaw (rad) of an arm at `azimuth` that faces the table centre.

    The SO-ARM101's "forward" (reach direction at qpos = 0) is -y in its own base frame, so
    facing the centre (direction `(-cos(azimuth), -sin(azimuth))`) needs
    `yaw = azimuth - pi/2`.
    """
    return (
        radius * math.cos(azimuth),
        radius * math.sin(azimuth),
        azimuth - math.pi / 2,
    )
