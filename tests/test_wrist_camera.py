"""Pure-math checks of the wrist camera pose: the jaw pads must be in its view (no mani_skill)."""

import itertools
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import callosum
from callosum.configs.cameras import WristCameraConfig, look_at_pose

URDF = (
    Path(callosum.__file__).parent / "assets" / "so101_parallel_gripper"
    / "so101_parallel_gripper.urdf"
)  # fmt: skip


def _quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _vec(element: ET.Element, attr: str) -> np.ndarray:
    return np.array(element.attrib[attr].split(), dtype=float)


def _pad_corners(opening: float) -> dict[str, np.ndarray]:
    """Corners (8, 3) of both pad collision boxes in the link5 frame at a jaw opening (m)."""
    root = ET.parse(URDF).getroot()
    joints = {j.attrib["name"]: j for j in root.iter("joint")}
    links = {link.attrib["name"]: link for link in root.iter("link")}
    corners = {}
    for clamp, sign in (("clamp_1", 1.0), ("clamp_2", -1.0)):
        joint = joints["right_clamp" if clamp == "clamp_1" else "left_clamp"]
        axis = _vec(joint.find("axis"), "xyz")
        # clamp_2 mirrors clamp_1 (left_clamp = -right_clamp).
        clamp_origin = _vec(joint.find("origin"), "xyz") + sign * opening * axis
        box = next(c for c in links[clamp].iter("collision") if c.attrib["name"] == f"{clamp}_pad")
        centre = clamp_origin + _vec(box.find("origin"), "xyz")
        half = _vec(box.find("geometry/box"), "size") / 2
        corners[clamp] = np.array(
            [centre + half * np.array(s) for s in itertools.product((-1, 1), repeat=3)]
        )
    return corners


def _project(cfg: WristCameraConfig, points: np.ndarray, width: int, height: int):
    """Pixel coordinates (n, 2) and depths (n,) of link-frame points (SAPIEN camera axes)."""
    position, quat = cfg.pose_in_mount()
    in_cam = (points - np.array(position)) @ _quat_to_matrix(quat)  # x fwd, y left, z up
    focal = (height / 2) / math.tan(cfg.fov_y / 2)
    u = width / 2 - focal * in_cam[:, 1] / in_cam[:, 0]
    v = height / 2 - focal * in_cam[:, 2] / in_cam[:, 0]
    return np.stack([u, v], axis=1), in_cam[:, 0]


def test_look_at_pose_axes() -> None:
    eye, target = (0.1, -0.2, 0.3), (0.4, 0.2, -0.1)
    position, quat = look_at_pose(eye, target)
    rot = _quat_to_matrix(quat)
    forward = np.array(target) - np.array(eye)
    assert position == eye
    assert sum(c * c for c in quat) == pytest.approx(1.0)
    assert rot[:, 0] == pytest.approx(forward / np.linalg.norm(forward))
    assert rot @ rot.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(rot) == pytest.approx(1.0)
    assert rot[2, 2] > 0  # camera z (up) stays on the world-up side


def test_look_at_pose_matches_sapien_convention_for_default_view() -> None:
    # Looking along -y with +z up: SAPIEN's left axis (camera y) is then +x.
    _, quat = look_at_pose((0, 0, 0), (0, -1, 0))
    rot = _quat_to_matrix(quat)
    assert rot[:, 0] == pytest.approx([0, -1, 0], abs=1e-9)
    assert rot[:, 1] == pytest.approx([1, 0, 0], abs=1e-9)
    assert rot[:, 2] == pytest.approx([0, 0, 1], abs=1e-9)


@pytest.mark.parametrize("size", [(128, 128), (320, 240)])
@pytest.mark.parametrize("opening", [0.0, 0.01, 0.037])
def test_jaw_pads_are_in_view(size, opening) -> None:
    width, height = size
    cfg = WristCameraConfig()
    for corners in _pad_corners(opening).values():
        pixels, depth = _project(cfg, corners, width, height)
        assert (depth > cfg.near).all()
        assert (pixels[:, 0] >= 0).all() and (pixels[:, 0] <= width).all()
        assert (pixels[:, 1] >= 0).all() and (pixels[:, 1] <= height).all()


def test_camera_looks_along_the_fingers() -> None:
    # Fingers point to -y of link5 (qpos = 0); the camera is pitched down, never up or sideways.
    position, quat = WristCameraConfig().pose_in_mount()
    forward = _quat_to_matrix(quat)[:, 0]
    assert forward[1] < -0.7
    assert -0.7 < forward[2] < 0
    assert forward[0] == pytest.approx(0.0, abs=1e-9)
    assert position[1] > -0.0694  # behind the housing's front face, not in front of the fingers
