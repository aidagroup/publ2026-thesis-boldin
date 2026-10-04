"""Face-lock bookkeeping (pure array logic) and the physics config of FaceTurn."""

import numpy as np
import pytest

from callosum.configs.face_turn import FaceTurnPhysicsConfig
from callosum.envs._face_lock import update_face_lock


def _state(locked, angle):
    return np.array(locked, dtype=bool), np.array(angle, dtype=float)


def test_lock_latches_the_angle_on_engagement_only() -> None:
    locked, angle = _state([False, False], [0.0, 0.0])
    # Env 0: holder not grasping -> lock engages at the current angle. Env 1: grasping, stays free.
    locked, angle = update_face_lock(locked, angle, np.array([True, False]), np.array([0.3, 0.7]))
    assert locked.tolist() == [True, False]
    assert angle[0] == pytest.approx(0.3)
    # The face creeps while locked: the held angle must not follow it.
    locked, angle = update_face_lock(locked, angle, np.array([True, False]), np.array([0.5, 0.9]))
    assert locked.tolist() == [True, False]
    assert angle[0] == pytest.approx(0.3)


def test_lock_releases_and_relatches_at_the_new_angle() -> None:
    locked, angle = _state([True], [0.3])
    locked, angle = update_face_lock(locked, angle, np.array([False]), np.array([0.3]))
    assert not locked[0]
    locked, angle = update_face_lock(locked, angle, np.array([True]), np.array([1.1]))
    assert locked[0] and angle[0] == pytest.approx(1.1)


def test_envs_are_independent() -> None:
    locked, angle = _state([True, True, False], [0.1, 0.2, 0.0])
    want = np.array([True, False, True])
    new_locked, new_angle = update_face_lock(locked, angle, want, np.array([0.5, 0.5, 0.5]))
    assert new_locked.tolist() == [True, False, True]
    assert new_angle.tolist() == pytest.approx([0.1, 0.2, 0.5])


def test_physics_config_defaults_and_validation() -> None:
    cfg = FaceTurnPhysicsConfig()
    assert cfg.lock_face_unless_held
    assert cfg.face_friction > 0.02  # stiffer than the old, no-op value
    with pytest.raises(ValueError, match="face_friction"):
        FaceTurnPhysicsConfig(face_friction=-1.0)
    with pytest.raises(ValueError, match="face_damping"):
        FaceTurnPhysicsConfig(face_damping=-0.1)
