"""The open-loop pick-lift-and-turn script, with no simulator in it.

`scripts/probe_grasp.py` asks whether the scene is solvable; `scripts/render_scene.py`
asks what it *looks* like while failing. Both drive the arms through the same
waypoints, and they are only comparable if that is literally the same table --
so it lives here, importable without mani_skill, and the phase budget is
checked in CI (tests/test_grasp_waypoints.py).

The waypoints themselves are in `_so100_kinematics`; this module only says in
what order and for how long to command them.

The two arms are on DIFFERENT schedules, which is why each phase names a
waypoint per arm rather than one for both. The holder has to have picked the
cube up and be holding it steady before the rotator touches anything: with
the cube's axis horizontal, the rotator's own gripper sweeps 4.26 cm below
that axis during the turn, so the cube has to be off the table first.
"""

import numpy as np

from callosum.envs._so100_kinematics import (
    HOLDER_GRASP,
    HOLDER_LIFT,
    HOLDER_PREGRASP,
    READY_QPOS,
    ROTATOR_GRASP,
    ROTATOR_PREGRASP,
)

# Gripper joint targets. -1.1 is the URDF lower limit (jaws shut, tips 6.6 mm
# apart); 0.0 is mani-skill's own SO-100 ready value. The jaws first touch at
# SEATING_GRIPPER_QPOS; commanding the full -1.1 makes the PD drive keep
# squeezing past that, which is what generates the >= 0.5 N `is_grasping`
# needs -- and, for the holder, the friction that carries the cube's weight.
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = -1.1

WAYPOINTS = {
    "ready": np.asarray(READY_QPOS[:5]),
    "holder_pregrasp": np.asarray(HOLDER_PREGRASP),
    "holder_grasp": np.asarray(HOLDER_GRASP),
    "holder_lift": np.asarray(HOLDER_LIFT),
    "rotator_pregrasp": np.asarray(ROTATOR_PREGRASP),
    "rotator_grasp": np.asarray(ROTATOR_GRASP),
}

# (label, holder waypoint, rotator waypoint, holder gripper, rotator gripper,
#  steps). The rotator stays parked at READY until the cube is in the air --
# it has nothing to reach for before then, and everything to knock over.
PHASES = [
    ("settle", "ready", "ready", GRIPPER_OPEN, GRIPPER_OPEN, 10),
    ("reach", "holder_pregrasp", "ready", GRIPPER_OPEN, GRIPPER_OPEN, 40),
    ("descend", "holder_grasp", "ready", GRIPPER_OPEN, GRIPPER_OPEN, 25),
    ("hold", "holder_grasp", "ready", GRIPPER_CLOSED, GRIPPER_OPEN, 20),
    ("lift", "holder_lift", "ready", GRIPPER_CLOSED, GRIPPER_OPEN, 35),
    ("approach", "holder_lift", "rotator_pregrasp", GRIPPER_CLOSED, GRIPPER_OPEN, 40),
    ("seat", "holder_lift", "rotator_grasp", GRIPPER_CLOSED, GRIPPER_OPEN, 25),
    ("close", "holder_lift", "rotator_grasp", GRIPPER_CLOSED, GRIPPER_CLOSED, 20),
    ("turn", "holder_lift", "rotator_grasp", GRIPPER_CLOSED, GRIPPER_CLOSED, 85),
]

TOTAL_STEPS = sum(p[5] for p in PHASES)

# The phase during which wrist_roll is driven away from its grasp value. Named
# rather than positional so reordering the table cannot silently move it.
TURN_PHASE = "turn"

# pd_joint_delta_pos limits, from SO100._controller_configs. The action space
# is normalized to [-1, 1] (PDJointPosControllerConfig.normalize_action
# defaults to True), so a unit action is one full delta.
DELTA_LIMITS = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.2])


def arm_target(name: str) -> np.ndarray:
    """The five-joint target for one named waypoint.

    Unknown names fall back to READY, which is convenient for a parked arm and
    dangerous for a typo -- so the set of names actually used by `PHASES` is
    asserted in CI rather than trusted here.
    """
    return WAYPOINTS.get(name, WAYPOINTS["ready"])


def arm_targets(waypoint: str | None) -> tuple[np.ndarray, np.ndarray]:
    """The (holder, rotator) targets for a phase, by the phase's own label."""
    for label, holder, rotator, _, _, _ in PHASES:
        if label == waypoint:
            return arm_target(holder), arm_target(rotator)
    ready = WAYPOINTS["ready"]
    return ready, ready
