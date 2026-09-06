"""The scripted waypoints must still point at what the cube geometry says.

This is the check that several server round trips paid for.
`scripts/probe_grasp.py` drives the arms through precomputed JOINT waypoints
because ManiSkill has no IK on the GPU backend, and three times those
waypoints silently stopped matching the scene: once because the forward
kinematics that solved them stopped at `wrist_flex` (6 cm too low), once
because they aimed the tool centre point at a handle's axis rather than the
jaws' pocket, and once because the cube was spawning somewhere else entirely.
All three read as physics failures in the probe output.

None of them needs a simulator to catch: `callosum.envs._so100_kinematics` is
pure numpy transcribed from the URDF and `callosum.envs._cube_geometry` is
pure arithmetic, so CI (which has neither mani_skill nor a GPU) can assert
that the two agree. `scripts/solve_waypoints.py` is what produced the joint
angles; this is what stops them drifting away from the geometry afterwards.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from callosum.envs import _cube_geometry as cube
from callosum.envs import _scripted_expert as expert
from callosum.envs import _so100_kinematics as kin

_DOWN = np.array([0.0, 0.0, -1.0])
_TOWARD_HOLDER = np.array([0.0, -1.0, 0.0])
_TOWARD_ROTATOR = np.array([0.0, 1.0, 0.0])

# (name, waypoint, base pose, object half-width, where its axis must land,
#  the direction the tool must point)
_GRASPS = [
    (
        "rotator on the nub",
        kin.ROTATOR_GRASP,
        kin.ROTATOR_BASE_POSE,
        cube.NUB_HALF_WIDTH,
        np.array([0.0, cube.CUBE_HALF_SIZE + cube.NUB_GRASP_DEPTH, cube.LIFT_HEIGHT], dtype=float),
        _TOWARD_HOLDER,
    ),
    (
        "holder on the body, on the table",
        kin.HOLDER_GRASP,
        kin.HOLDER_BASE_POSE,
        cube.CUBE_HALF_SIZE,
        np.array(
            [0.0, cube.BODY_GRASP_OFFSET[1], cube.CUBE_HALF_SIZE + cube.BODY_GRASP_OFFSET[2]],
            dtype=float,
        ),
        _TOWARD_ROTATOR,
    ),
    (
        "holder on the body, lifted",
        kin.HOLDER_LIFT,
        kin.HOLDER_BASE_POSE,
        cube.CUBE_HALF_SIZE,
        np.array(
            [0.0, cube.BODY_GRASP_OFFSET[1], cube.LIFT_HEIGHT + cube.BODY_GRASP_OFFSET[2]],
            dtype=float,
        ),
        _TOWARD_ROTATOR,
    ),
]

_PREGRASPS = [
    (
        "rotator",
        kin.ROTATOR_PREGRASP,
        kin.ROTATOR_GRASP,
        kin.ROTATOR_BASE_POSE,
        cube.NUB_HALF_WIDTH,
    ),
    ("holder", kin.HOLDER_PREGRASP, kin.HOLDER_GRASP, kin.HOLDER_BASE_POSE, cube.CUBE_HALF_SIZE),
]

_ALL = [(n, q, b) for n, q, b, _, _, _ in _GRASPS] + [
    (f"{n} pregrasp", q, b) for n, q, _, b, _ in _PREGRASPS
]


@pytest.mark.parametrize(("name", "qpos", "base", "half_width", "axis", "tool"), _GRASPS)
def test_the_grasp_seats_the_object_it_is_aimed_at(
    name, qpos, base, half_width, axis, tool
) -> None:
    """The jaws' pocket must land on the held object's axis, within a mm."""
    seated = kin.seated_object_position(qpos, base, half_width)
    assert np.linalg.norm(seated - axis) < 1e-3, f"{name}: {np.round(seated - axis, 4)}"


@pytest.mark.parametrize(("name", "qpos", "base", "half_width", "axis", "tool"), _GRASPS)
def test_the_tool_points_along_the_cube_axis(name, qpos, base, half_width, axis, tool) -> None:
    """A horizontal tool, aimed along y.

    For the rotator this is the whole task: the wrist_roll axis IS the tool
    axis, and turning the face means rolling about the cube's axis. A degree
    of tilt here is a degree the grip has to absorb over a quarter turn.
    """
    direction = kin.approach_direction(qpos, base)
    tilt = np.degrees(np.arccos(np.clip(direction @ tool, -1.0, 1.0)))
    assert tilt < 0.5, f"{name}: tool is {tilt:.2f} deg off the cube's axis"


@pytest.mark.parametrize(("name", "qpos", "base"), _ALL)
def test_the_waypoint_is_inside_the_joint_limits(name, qpos, base) -> None:
    """With margin. On the limit means the cube's spawn jitter cannot be met."""
    lower, upper = kin.JOINT_LIMITS[:5, 0], kin.JOINT_LIMITS[:5, 1]
    assert np.all(qpos > lower + 0.05), f"{name}: {np.round(qpos - lower, 3)}"
    assert np.all(qpos < upper - 0.05), f"{name}: {np.round(upper - qpos, 3)}"


@pytest.mark.parametrize(("name", "pregrasp", "grasp", "base", "half_width"), _PREGRASPS)
def test_the_approach_is_a_straight_vertical_descent(
    name, pregrasp, grasp, base, half_width
) -> None:
    """Pre-grasp directly above the grasp, so the last move is a drop.

    Not a back-off along the tool axis: 5 cm nearer its own base is where a
    horizontal tool stops being reachable. And the jaws' slot is bounded in x
    by the blades and open in z, so a descent is the one approach that does
    not sweep a blade across the object first.
    """
    above = kin.seated_object_position(pregrasp, base, half_width)
    on = kin.seated_object_position(grasp, base, half_width)
    assert abs(above[2] - on[2] - kin.PREGRASP_LIFT) < 1e-3, f"{name}: {above[2] - on[2]:.4f} m"
    assert np.linalg.norm((above - on)[:2]) < 1e-3, f"{name} drifts sideways: {above - on}"


def test_the_nub_is_exactly_as_wide_as_the_roll_axis_allows() -> None:
    """Sized so a wrist roll spins it about its OWN axis, not around an arc.

    An object seats against the fixed blade's face, so its centre lands
    `half_width - FIXED_BLADE_FACE_X` off the wrist_roll axis. Anything wider
    than twice that face offset gets dragged sideways over a quarter turn --
    2.9 cm for a bare 5.7 cm layer, against a hinge that cannot move.
    """
    seated_offset = abs(cube.NUB_HALF_WIDTH - kin.FIXED_BLADE_FACE_X)
    assert seated_offset < 1e-9, f"{seated_offset * 1000:.2f} mm off the roll axis"


def test_the_lift_is_what_keeps_the_gripper_out_of_the_table() -> None:
    """The reason the holder picks the cube up at all.

    The rotator's closed gripper reaches 4.26 cm from the wrist_roll axis, so
    a quarter turn sweeps that circle about the cube's axis. On the table that
    axis is one cube-half-size up and the palm goes through the tabletop.
    """
    assert cube.gripper_sweep_clearance(cube.CUBE_HALF_SIZE) < 0.0
    assert cube.gripper_sweep_clearance(cube.LIFT_HEIGHT) > 0.02
    assert cube.LIFT_HEIGHT - cube.CUBE_SWEEP_RADIUS > 0.02, "the cube itself would clip the table"


def test_the_holder_stops_short_of_the_turning_layer() -> None:
    """Its blade must not reach across the split, or it clamps both halves.

    The blade tip is 0.92 cm beyond the pocket; the pocket sits at
    `BODY_GRASP_OFFSET[1]` in the cube's frame and the layer starts at
    `FACE_SPLIT_Y`.
    """
    tip_y = cube.BODY_GRASP_OFFSET[1] + (kin.POCKET_DEPTH - (-0.1064))
    assert tip_y < cube.FACE_SPLIT_Y, f"blade tip at {tip_y:.4f} is past {cube.FACE_SPLIT_Y:.4f}"
    assert cube.FACE_SPLIT_Y - tip_y > 0.003, "less than 3 mm of clearance to the layer"


def test_start_pose_is_open_and_vertical() -> None:
    """READY_QPOS must start the hand OPEN.

    The gripper joint moves `SO100.tcp_pos` (the midpoint of the jaw tips), so
    a closed start makes opening the hand locally reward-negative.
    """
    assert kin.READY_QPOS[5] > kin.JOINT_LIMITS[5, 0] + 0.5, "hand starts clamped shut"
    for base in (kin.HOLDER_BASE_POSE, kin.ROTATOR_BASE_POSE):
        tilt = np.degrees(
            np.arccos(np.clip(kin.approach_direction(kin.READY_QPOS, base) @ _DOWN, -1, 1))
        )
        assert tilt < 0.5


def test_the_phase_table_fits_inside_an_episode() -> None:
    """The scripted script must finish before the env truncates it.

    `PHASES` is open loop: if the budget outgrows `max_episode_steps` the run
    is cut off mid-turn and the probe reports a failure that is really a clock.
    The limit lives in a `@register_env` decorator that CI cannot import
    (mani_skill is Linux+CUDA only), so read it out of the source.
    """
    source = (Path(__file__).parents[1] / "callosum/envs/two_so100_base.py").read_text()
    match = re.search(r"max_episode_steps=(\d+)", source)
    assert match, "no max_episode_steps in the env registration"
    assert expert.TOTAL_STEPS <= int(match.group(1))


def test_every_phase_name_resolves_to_its_waypoint() -> None:
    """A phase naming a waypoint that does not exist parks that arm, silently.

    `arm_target` falls back to READY for anything it does not recognise --
    convenient for a parked arm, dangerous for a typo, which would read in the
    probe output as an arm that simply failed to reach.
    """
    named = {p[1] for p in expert.PHASES} | {p[2] for p in expert.PHASES}
    assert named <= set(expert.WAYPOINTS), f"unknown waypoints: {named - set(expert.WAYPOINTS)}"
    assert expert.TURN_PHASE in {p[0] for p in expert.PHASES}
    holder, rotator = expert.arm_targets("close")
    np.testing.assert_allclose(holder, kin.HOLDER_LIFT)
    np.testing.assert_allclose(rotator, kin.ROTATOR_GRASP)


def test_the_rotator_waits_until_the_cube_is_off_the_table() -> None:
    """It must not reach for the nub before the holder has lifted.

    Its own gripper sweeps 4.26 cm below the cube's axis during the turn, so
    while the cube is on the table there is nowhere for the rotator to be.
    """
    lifted = False
    for label, holder, rotator, _, _, _ in expert.PHASES:
        if holder == "holder_lift":
            lifted = True
        assert lifted or rotator == "ready", f"{label}: rotator moves before the lift"


@pytest.mark.parametrize("script", ["probe_grasp.py", "render_scene.py"])
def test_the_scripted_probes_pin_the_cube(script) -> None:
    """An open-loop script must not be handed a randomly placed cube.

    The waypoints are solved once, against the nominal pose. With the env's
    default +-1 cm spawn jitter every centimetre of it is a miss the script
    cannot correct, which on 2026-09-02 read as 0.94 and 1.27 cm of waypoint
    "error" in a teleported pose -- with the physics switched off.
    """
    source = (Path(__file__).parents[1] / "scripts" / script).read_text()
    assert "cube_spawn_jitter=0.0" in source
    assert "robot_init_qpos_noise=0.0" in source


def test_the_face_grasp_offset_is_measured_from_the_face_LINK() -> None:
    """The face link's origin is the layer's centre, not the cube's.

    Measuring the nub from the cube's centre put the rotator's reach target
    1.9 cm too far out. The builder is the authority on where that origin is,
    so read its joint translation rather than restating the number.
    """
    source = (Path(__file__).parents[1] / "callosum/envs/_turntable_cube.py").read_text()
    match = re.search(r"pose_in_parent=sapien\.Pose\(\s*\[0, ([^,]+), 0\]", source)
    assert match, "the face joint's translation is no longer where this test looks"
    assert match.group(1).strip() == "cube_half_size - face_thickness / 2"
    assert abs(cube.FACE_LINK_ORIGIN_Y - (cube.CUBE_HALF_SIZE - cube.FACE_THICKNESS / 2)) < 1e-12
    # Composed through the link origin, the reach target must land on the nub.
    assert (
        abs(
            cube.FACE_LINK_ORIGIN_Y
            + cube.FACE_GRASP_OFFSET[1]
            - (cube.CUBE_HALF_SIZE + cube.NUB_GRASP_DEPTH)
        )
        < 1e-12
    )


def test_every_phase_is_long_enough_for_its_own_move() -> None:
    """A phase shorter than its synchronized move leaves the arm mid-flight.

    That reads in the probe output as an arm that missed its target, which is
    indistinguishable from a wrong waypoint until you look at the joint error.
    """
    previous = {"holder": "ready", "rotator": "ready"}
    for label, holder_wp, rotator_wp, _, _, budget in expert.PHASES:
        for role, wp in (("holder", holder_wp), ("rotator", rotator_wp)):
            start = expert.arm_target(previous[role])
            goal = expert.arm_target(wp)
            need = np.ceil((np.abs(goal - start) / expert.DELTA_LIMITS[:5]).max())
            assert budget >= need, f"{label}/{role}: {budget} steps for a {need:.0f}-step move"
            previous[role] = wp


def test_a_synchronized_step_moves_every_joint_together() -> None:
    """All joints must arrive on the same step, or the arm sweeps the table."""
    start = np.asarray(kin.READY_QPOS[:5])
    goal = np.asarray(kin.HOLDER_PREGRASP)
    qpos, arrived = start.copy(), None
    for step in range(1, 200):
        qpos = qpos + expert.synchronized_step(qpos, goal)
        if np.abs(goal - qpos).max() < 1e-9:
            arrived = step
            break
    assert arrived is not None
    # Independent per-joint driving would land the short joints ~50 steps early.
    fractions = np.abs(goal - start) / np.abs(goal - start).max()
    assert fractions.min() < 0.3, "this waypoint no longer exercises the failure"
