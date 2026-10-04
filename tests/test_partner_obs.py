"""Unit tests for the partner_obs observation-visibility rule (step 1.4, reworked in step 2.1).

Tests callosum.envs._partner_obs directly (pure Python, no mani_skill/torch
dependency) rather than callosum.envs.two_so101_base itself: that module
imports mani_skill at module level, which is absent on macOS/CI (mani-skill
is gated to sys_platform == 'linux' in pyproject.toml), so it cannot even be
imported here -- see docs/implementation-plan.md section 0.
"""

import pytest

from callosum.envs._partner_obs import tcp_pose_fields, tcp_pose_visible, validate_partner_obs


def test_env_always_emits_both_tcp_poses() -> None:
    fields = tcp_pose_fields("pose_a", "pose_b")
    assert fields == {"agent_a_tcp_pose": "pose_a", "agent_b_tcp_pose": "pose_b"}


@pytest.mark.parametrize("partner_obs", ["full", "none"])
@pytest.mark.parametrize("agent_idx", [0, 1])
def test_own_tcp_pose_is_always_visible(partner_obs: str, agent_idx: int) -> None:
    assert tcp_pose_visible(partner_obs, agent_idx, agent_idx)


@pytest.mark.parametrize("agent_idx", [0, 1])
def test_partner_tcp_pose_only_visible_under_full(agent_idx: int) -> None:
    assert tcp_pose_visible("full", agent_idx, 1 - agent_idx)
    assert not tcp_pose_visible("none", agent_idx, 1 - agent_idx)


def test_visibility_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="partner_obs"):
        tcp_pose_visible("predicted", 0, 1)
    with pytest.raises(ValueError, match="indices"):
        tcp_pose_visible("full", 2, 0)


def test_validate_accepts_known_modes() -> None:
    validate_partner_obs("full")
    validate_partner_obs("none")


def test_validate_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="partner_obs"):
        validate_partner_obs("oracle")
