"""Sanity checks for the SO-ARM101 + parallel gripper URDF (pure stdlib, no mani_skill)."""

import xml.etree.ElementTree as ET
from pathlib import Path

import callosum

ASSET_DIR = Path(callosum.__file__).parent / "assets" / "so101_parallel_gripper"
URDF = ASSET_DIR / "so101_parallel_gripper.urdf"

ARM_JOINTS = [
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
]
GRIPPER_JOINTS = ["right_clamp", "left_clamp"]


def _root() -> ET.Element:
    return ET.parse(URDF).getroot()


def test_referenced_meshes_exist() -> None:
    meshes = [m.attrib["filename"] for m in _root().iter("mesh")]
    assert meshes
    for filename in meshes:
        assert (ASSET_DIR / filename).is_file(), filename


def test_actuated_joints_and_mimic() -> None:
    joints = {j.attrib["name"]: j for j in _root().iter("joint")}
    actuated = [n for n, j in joints.items() if j.attrib["type"] in ("revolute", "prismatic")]
    assert actuated == ARM_JOINTS + GRIPPER_JOINTS
    mimic = joints["left_clamp"].find("mimic")
    assert mimic is not None
    assert mimic.attrib["joint"] == "right_clamp"
    assert float(mimic.attrib["multiplier"]) == -1.0


def test_pad_links_exist() -> None:
    links = {link.attrib["name"] for link in _root().iter("link")}
    assert {"clamp_1", "clamp_2", "clamp_1_pad", "clamp_2_pad"} <= links


def test_licence_files_present() -> None:
    for name in ("README.md", "LICENSE-Apache-2.0.txt", "LICENSE-CERN-OHL-P-2.0.txt"):
        assert (ASSET_DIR / name).is_file(), name
