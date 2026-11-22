"""Mac-side tests for the FaceTurnSkills observation contract (_skills_obs)."""

import pytest

from callosum.envs._skills import Role
from callosum.envs._skills_obs import (
    INFO_PRIMITIVE_HOLDER,
    INFO_SUCCESS,
    own_block_fields,
    own_block_size,
    own_block_slices,
    policy_input_size,
)


@pytest.mark.parametrize(
    ("role", "n_own"),
    [(Role.HOLDER, 9), (Role.ROTATOR, 7)],
)
class TestOwnBlock:
    def test_fields_are_the_documented_order(self, role, n_own):
        names = [name for name, _ in own_block_fields(n_own)]
        assert names == [
            "time_fraction",
            "plan_face_onehot",
            "plan_quarter_turns",
            "plan_moves_remaining",
            "cube_position",
            "cube_orientation_quat",
            "active_layer_angle",
            "own_qpos",
            "own_qvel",
            "own_gripper_opening",
            "own_primitive_onehot",
            "own_primitive_progress",
        ]

    def test_size_is_the_sum_of_fields(self, role, n_own):
        assert own_block_size(n_own) == sum(size for _, size in own_block_fields(n_own))

    def test_slices_cover_contiguously(self, role, n_own):
        slices = own_block_slices(n_own)
        assert slices["time_fraction"].start == 0
        end = 0
        for field, size in own_block_fields(n_own):
            assert slices[field] == slice(end, end + size)
            end += size
        assert end == own_block_size(n_own)


class TestPolicyInputSize:
    def test_none_adds_nothing(self):
        for role in (Role.HOLDER, Role.ROTATOR):
            n_own = 9 if role is Role.HOLDER else 7
            assert policy_input_size("none", role) == own_block_size(n_own)

    def test_oracle_adds_partner_onehot_plus_progress(self):
        # holder observes the rotator (7 primitives + 1 progress)
        assert policy_input_size("oracle", Role.HOLDER) == own_block_size(9) + 7 + 1
        # rotator observes the holder (9 primitives + 1 progress)
        assert policy_input_size("oracle", Role.ROTATOR) == own_block_size(7) + 9 + 1

    def test_predicted_uses_latent_dim(self):
        assert policy_input_size("predicted", Role.HOLDER, latent_dim=32) == own_block_size(9) + 32

    def test_predicted_direct_uses_partner_registry_size(self):
        assert policy_input_size("predicted-direct", Role.ROTATOR) == own_block_size(7) + 9

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            policy_input_size("full", Role.HOLDER)


class TestInfoKeys:
    def test_keys_are_distinct_strings(self):
        keys = {INFO_SUCCESS, INFO_PRIMITIVE_HOLDER}
        assert len(keys) == 2
        assert all(isinstance(k, str) for k in keys)
