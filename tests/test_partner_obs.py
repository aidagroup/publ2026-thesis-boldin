"""Unit test for the partner_obs observation-visibility flag (step 1.4).

Tests callosum.envs._partner_obs directly (pure Python, no mani_skill/torch
dependency) rather than callosum.envs.two_so101_base itself: that module
imports mani_skill at module level, which is absent on macOS/CI (mani-skill
is gated to sys_platform == 'linux' in pyproject.toml), so it cannot even be
imported here -- see docs/implementation-plan.md section 0.
"""

import pytest

from callosum.envs._partner_obs import partner_tcp_pose_fields, validate_partner_obs


def test_full_includes_both_tcp_poses() -> None:
    fields = partner_tcp_pose_fields("full", "pose_a", "pose_b")
    assert fields == {"agent_a_tcp_pose": "pose_a", "agent_b_tcp_pose": "pose_b"}


def test_none_includes_neither() -> None:
    assert partner_tcp_pose_fields("none", "pose_a", "pose_b") == {}


def test_shape_differs_between_modes() -> None:
    full = partner_tcp_pose_fields("full", "pose_a", "pose_b")
    none = partner_tcp_pose_fields("none", "pose_a", "pose_b")
    assert len(full) != len(none)


def test_validate_accepts_known_modes() -> None:
    validate_partner_obs("full")
    validate_partner_obs("none")


def test_validate_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="partner_obs"):
        validate_partner_obs("oracle")
