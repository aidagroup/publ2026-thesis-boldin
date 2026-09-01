"""The scripted grasp waypoints must still point at the cube's handles.

This is the check that two server round trips paid for. `scripts/probe_grasp.py`
drives the arms through precomputed JOINT waypoints because ManiSkill has no IK
on the GPU backend, and twice those waypoints silently stopped matching the
scene -- once because the forward kinematics that solved them stopped at
`wrist_flex` instead of `Fixed_Jaw` (6 cm too low), once because they aimed the
tool centre point at a handle's axis, which buries the fixed blade 2.1 mm
inside it. Both looked like physics failures in the probe output.

Neither needs a simulator to catch: `callosum.envs._so100_kinematics` is pure
numpy transcribed from the URDF and `callosum.envs._cube_geometry` is pure
arithmetic, so CI (which has neither mani_skill nor a GPU) can assert that the
two agree. See docs/implementation-plan.md step 2.1.
"""

import numpy as np
import pytest

from callosum.envs import _so100_kinematics as kin
from callosum.envs._cube_geometry import (
    BODY_GRASP_WORLD,
    FACE_GRASP_WORLD,
    HANDLE_HALF_WIDTH,
)

_DOWN = np.array([0.0, 0.0, -1.0])

# (name, arm waypoint, base pose, handle world position, extra height)
_WAYPOINTS = [
    (
        "rotator pregrasp",
        kin.ROTATOR_PREGRASP,
        kin.ROTATOR_BASE_POSE,
        FACE_GRASP_WORLD,
        kin.PREGRASP_LIFT_ROTATOR,
    ),
    ("rotator grasp", kin.ROTATOR_GRASP, kin.ROTATOR_BASE_POSE, FACE_GRASP_WORLD, 0.0),
    (
        "holder pregrasp",
        kin.HOLDER_PREGRASP,
        kin.HOLDER_BASE_POSE,
        BODY_GRASP_WORLD,
        kin.PREGRASP_LIFT_HOLDER,
    ),
    ("holder grasp", kin.HOLDER_GRASP, kin.HOLDER_BASE_POSE, BODY_GRASP_WORLD, 0.0),
]
_GRASPS = [w for w in _WAYPOINTS if w[4] == 0.0]


@pytest.mark.parametrize("name,qpos,base,handle,lift", _WAYPOINTS, ids=lambda v: str(v)[:24])
def test_waypoint_seats_the_handle(name, qpos, base, handle, lift) -> None:
    """The jaw pocket must land on the handle, to within a millimetre."""
    want = np.asarray(handle) + np.array([0.0, 0.0, lift])
    got = kin.seated_handle_position(qpos, base)
    assert np.linalg.norm(got - want) < 1e-3, f"{name}: pocket at {got}, handle at {want}"


@pytest.mark.parametrize("name,qpos,base,handle,lift", _WAYPOINTS, ids=lambda v: str(v)[:24])
def test_waypoint_points_straight_down(name, qpos, base, handle, lift) -> None:
    """A top-down approach is what makes wrist_roll the vertical turning axis.

    Any tilt also tips the grasped handle relative to the face joint, which is
    constrained to a vertical axis -- see callosum.envs.face_turn.
    """
    tilt = np.degrees(np.arccos(np.clip(kin.approach_direction(qpos, base) @ _DOWN, -1, 1)))
    assert tilt < 0.5, f"{name}: tool is {tilt:.2f} deg off vertical"


@pytest.mark.parametrize("name,qpos,base,handle,lift", _WAYPOINTS, ids=lambda v: str(v)[:24])
def test_waypoint_is_inside_the_joint_limits(name, qpos, base, handle, lift) -> None:
    lo, hi = kin.JOINT_LIMITS[:5, 0], kin.JOINT_LIMITS[:5, 1]
    assert np.all(qpos >= lo) and np.all(qpos <= hi), f"{name}: {qpos} outside {lo}..{hi}"


@pytest.mark.parametrize("name,qpos,base,handle,lift", _GRASPS, ids=lambda v: str(v)[:24])
def test_grasp_tcp_matches_what_the_probe_measures(name, qpos, base, handle, lift) -> None:
    """`rot->face` / `hold->body` in probe_grasp.py must read ~0 at the grasp.

    Those columns are `|SO100.tcp_pos - handle|`, not the pocket position, and
    the probe warns above 10 mm. Assert the geometry actually delivers that, so
    the warning threshold stays meaningful.
    """
    tcp = kin.tcp_position(list(qpos) + [kin.SEATING_GRIPPER_QPOS], base)
    assert np.linalg.norm(tcp - np.asarray(handle)) < 0.010, f"{name}: tcp at {tcp}"


def test_pocket_offset_leaves_the_fixed_blade_clear() -> None:
    """The handle must sit clear of the fixed blade with the jaws open.

    The fixed blade cannot retract, so its gripping face is a hard wall at
    local x = +0.0079 (measured on Fixed_Jaw_part2.ply at the insertion depth).
    Only the MOVING jaw may touch the handle before the squeeze; otherwise a
    position-controlled arm shoves the cube instead of straddling the handle.
    """
    fixed_blade_face_x = 0.0079
    clearance = fixed_blade_face_x - (kin.GRASP_POCKET_OFFSET[0] + HANDLE_HALF_WIDTH)
    assert clearance > 1e-3, f"only {clearance * 1000:.2f} mm of approach clearance"


def test_start_pose_is_open_and_vertical() -> None:
    """READY_QPOS must start the hand OPEN and pointing straight down.

    The gripper joint moves `SO100.tcp_pos` (the midpoint of the jaw tips), so
    a closed start makes opening the hand locally reward-negative; and only a
    vertical tool puts the wrist_roll axis on the face joint's axis.
    """
    assert kin.READY_QPOS[5] > kin.JOINT_LIMITS[5, 0] + 0.5, "hand starts clamped shut"
    for base in (kin.HOLDER_BASE_POSE, kin.ROTATOR_BASE_POSE):
        tilt = np.degrees(
            np.arccos(np.clip(kin.approach_direction(kin.READY_QPOS, base) @ _DOWN, -1, 1))
        )
        assert tilt < 0.5
        roll_axis = kin.fixed_jaw_pose(kin.READY_QPOS, base)[:3, 1]
        assert abs(abs(roll_axis[2]) - 1.0) < 1e-3, "wrist_roll axis is not vertical"
