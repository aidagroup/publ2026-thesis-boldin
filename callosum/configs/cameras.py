"""Camera configs: the wrist camera of each arm and the third-person scene camera.

Pure Python (no `mani_skill` / `sapien` import), so the geometry can be unit-tested anywhere.
Poses follow the SAPIEN camera convention: x forward, y left, z up; quaternions are `[w, x, y, z]`.
"""

import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]


def look_at_pose(
    eye: Vec3, target: Vec3, up: Vec3 = (0.0, 0.0, 1.0)
) -> tuple[Vec3, tuple[float, float, float, float]]:
    """Camera pose `(position, quaternion [w, x, y, z])` at `eye` looking at `target`.

    Same result as `mani_skill.utils.sapien_utils.look_at`: the camera x axis points from `eye`
    to `target`, its y axis is to the left (`up x forward`) and its z axis is as close to `up`
    as possible. The pose is expressed in whatever frame `eye`, `target` and `up` are given in.
    """
    forward = _unit(_sub(target, eye))
    left = _unit(_cross(up, forward))
    true_up = _cross(forward, left)
    # Rotation matrix with columns (forward, left, true_up) -> quaternion (Shepperd's method).
    m = [
        [forward[0], left[0], true_up[0]],
        [forward[1], left[1], true_up[1]],
        [forward[2], left[2], true_up[2]],
    ]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        quat = (s / 4, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2 * math.sqrt(1 + m[0][0] - m[1][1] - m[2][2])
        quat = ((m[2][1] - m[1][2]) / s, s / 4, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    elif m[1][1] > m[2][2]:
        s = 2 * math.sqrt(1 + m[1][1] - m[0][0] - m[2][2])
        quat = ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, s / 4, (m[1][2] + m[2][1]) / s)
    else:
        s = 2 * math.sqrt(1 + m[2][2] - m[0][0] - m[1][1])
        quat = ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, s / 4)
    return tuple(eye), quat


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _unit(a: Vec3) -> Vec3:
    norm = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    return (a[0] / norm, a[1] / norm, a[2] / norm)


@dataclass(frozen=True)
class WristCameraConfig:
    """Wrist camera of one SO-ARM101 + parallel gripper (agent `so101_pg_wristcam`).

    The camera is rigidly mounted on `mount_link` (the gripper housing, so it rolls with the
    wrist) at `eye`, looking at `target`; both are in the mount link's frame. At qpos = 0 the
    link frames are pure translations of the base frame: the fingers point to -y, the jaws open
    along x and z is up.

    The defaults model an Intel RealSense D405 (87 x 58 degrees field of view) sitting on the
    sloped front face of the housing, looking along the fingers, pitched 34 degrees down so that
    both jaw pads (also fully open) and the space between them are in the lower half of the
    image. `fov_y` is the *vertical* field of view in radians (SAPIEN's `fovy`); the horizontal
    one follows from the image aspect ratio. 70 degrees vertical gives the D405's 87 degrees
    horizontal at 4:3 (e.g. 320 x 240) and 70 x 70 degrees at the square 128 x 128 default.

    The 128 x 128 default is for future vision-based training; the video script asks for a
    larger image. The hardware reference is the Robonine parallel gripper's mount for a D405 /
    D435 / Orbbec Gemini 2 on the gripper housing.
    """

    uid: str = "wrist"
    mount_link: str = "link5_1"
    eye: Vec3 = (0.0, -0.058, 0.075)
    target: Vec3 = (0.0, -0.170, 0.0)
    width: int = 128
    height: int = 128
    fov_y: float = math.radians(70.0)
    near: float = 0.005
    far: float = 5.0

    def pose_in_mount(self) -> tuple[Vec3, tuple[float, float, float, float]]:
        """`(position, quaternion [w, x, y, z])` of the camera in the mount link's frame."""
        return look_at_pose(self.eye, self.target)


@dataclass(frozen=True)
class SceneCameraConfig:
    """Fixed third-person camera (ManiSkill's "human render camera") looking at the table centre.

    Placed diagonally opposite the two arms (holder at -y, rotator at +x, see `ArmLayout`) so
    that both arms and the cube are in view. World frame, metres; `fov_y` in radians.
    """

    eye: Vec3 = (-0.45, 0.45, 0.42)
    target: Vec3 = (0.10, -0.10, 0.0)
    width: int = 512
    height: int = 384
    fov_y: float = math.radians(50.0)
    near: float = 0.01
    far: float = 10.0
