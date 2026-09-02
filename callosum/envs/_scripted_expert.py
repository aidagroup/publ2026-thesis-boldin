"""The open-loop grasp-and-turn script, with no simulator in it.

`scripts/probe_grasp.py` asks whether the scene is solvable; `scripts/render_scene.py`
asks what it *looks* like while failing. Both drive the arms through the same
waypoints, and they are only comparable if that is literally the same table --
so it lives here, importable without mani_skill, and the phase budget is
checked in CI (tests/test_grasp_waypoints.py).

The waypoints themselves are in `_so100_kinematics`; this module only says in
what order and for how long to command them.
"""

import numpy as np

from callosum.envs._so100_kinematics import (
    HOLDER_GRASP,
    HOLDER_PREGRASP,
    READY_QPOS,
    ROTATOR_GRASP,
    ROTATOR_PREGRASP,
)

# Gripper joint targets. -1.1 is the URDF lower limit (jaws shut, tips 6.6 mm
# apart); 0.0 is mani-skill's own SO-100 ready value. The handle first touches
# both blades at SEATING_GRIPPER_QPOS (-0.842); commanding the full -1.1 makes
# the PD drive keep squeezing past that, which is what generates the >= 0.5 N
# `is_grasping` needs.
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = -1.1

# (name, waypoint, holder gripper, rotator gripper, steps). The holder shuts
# one phase before the rotator: closing on a handle shoves it by the ~2 mm of
# approach clearance in `GRASP_POCKET_OFFSET`, and the body should be held
# before the rotator delivers that nudge to the face. Sums to 290 of the
# env's 300 steps.
PHASES = [
    ("settle", None, GRIPPER_OPEN, GRIPPER_OPEN, 10),
    ("pregrasp", "pregrasp", GRIPPER_OPEN, GRIPPER_OPEN, 50),
    ("descend", "grasp", GRIPPER_OPEN, GRIPPER_OPEN, 30),
    ("hold", "grasp", GRIPPER_CLOSED, GRIPPER_OPEN, 20),
    ("close", "grasp", GRIPPER_CLOSED, GRIPPER_CLOSED, 25),
    ("turn", "grasp", GRIPPER_CLOSED, GRIPPER_CLOSED, 155),
]

TOTAL_STEPS = sum(p[4] for p in PHASES)

# pd_joint_delta_pos limits, from SO100._controller_configs. The action space
# is normalized to [-1, 1] (PDJointPosControllerConfig.normalize_action
# defaults to True), so a unit action is one full delta.
DELTA_LIMITS = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.2])


def arm_targets(waypoint: str | None) -> tuple[np.ndarray, np.ndarray]:
    """The (holder, rotator) five-joint arm targets for one phase's waypoint."""
    if waypoint == "pregrasp":
        return np.asarray(HOLDER_PREGRASP), np.asarray(ROTATOR_PREGRASP)
    if waypoint == "grasp":
        return np.asarray(HOLDER_GRASP), np.asarray(ROTATOR_GRASP)
    ready = np.asarray(READY_QPOS[:5])
    return ready, ready
