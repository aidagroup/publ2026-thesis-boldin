"""Bookkeeping of the FaceTurn "face lock" as pure array logic (no mani_skill/torch import).

The rule (see `callosum.envs.face_turn.FaceTurn`): while the holder does not grasp the cube
body, the face joint is held at the angle it had when the lock engaged; while the holder
grasps, the face is free. The state is two per-env arrays: a boolean `locked` and the float
`lock_angle` to hold. The helper only uses elementwise operators that behave the same on numpy
arrays and torch tensors (also on CUDA), so it is unit-testable on any machine and runs on the
simulation device without a host sync.
"""


def update_face_lock[Array](  # Array: numpy array or torch tensor of shape (num_envs,)
    locked: Array, lock_angle: Array, want_locked: Array, face_angle: Array
) -> tuple[Array, Array]:
    """Advance the per-env lock state by one control step.

    Args:
        locked: bool, the lock state of the previous step.
        lock_angle: float, the angle held while locked (arbitrary where not locked).
        want_locked: bool, whether the rule requires the lock now (holder not grasping).
        face_angle: float, the current face angle.

    Returns:
        `(locked, lock_angle)` for this step. The lock angle is latched from `face_angle` only on
        the step where the lock engages (`want_locked` and not `locked`), and kept while it
        stays engaged, so a face that is held cannot creep.
    """
    engage = want_locked & ~locked
    new_angle = lock_angle + engage * (face_angle - lock_angle)
    return want_locked, new_angle
