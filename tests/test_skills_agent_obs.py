"""Mac-side tests for the skill-level per-agent observation assembly (_skills_agent_obs)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # only the SkillsObsBuilder half needs it

from callosum.experiments.arm.agent_obs import flatten_structured
from callosum.experiments.arm.skills import PARTNER_VISIBILITY_MODES, Role, num_primitives
from callosum.experiments.arm.skills_agent_obs import (
    PROPRIO_DIM,
    SkillsObsBuilder,
    agent_obs_dim,
    build_agent_obs,
    extract,
    partner_primitive_key,
    shared_vec_size,
)
from callosum.experiments.arm.skills_obs import (
    INFO_PRIMITIVE_HOLDER,
    INFO_PRIMITIVE_ROTATOR,
    own_block_size,
    own_block_slices,
    policy_input_size,
)

UIDS = ("so101_pg-0", "so101_pg-1")
HOLDER_N, ROTATOR_N = num_primitives(Role.HOLDER), num_primitives(Role.ROTATOR)
HOLDER_SIZE, ROTATOR_SIZE = own_block_size(HOLDER_N), own_block_size(ROTATOR_N)
PARTNER_SENTINEL = 777.0  # marks every ground-truth field of the partner's own block


def fake_shared(batch: int) -> np.ndarray:
    """A shared vector whose every column carries a distinct increasing value."""
    return np.arange(batch * shared_vec_size(), dtype=np.float32).reshape(batch, -1) / 1000.0 + 1.0


def fake_proprio(batch: int) -> np.ndarray:
    """Own proprioception with values no shared-vector column carries."""
    return -np.arange(1, batch * PROPRIO_DIM + 1, dtype=np.float32).reshape(batch, -1) / 100.0


def partner_info(role: Role, ids: np.ndarray) -> dict:
    return {partner_primitive_key(role): ids}


class TestExtract:
    def test_shared_vec_size_is_both_own_blocks(self):
        assert shared_vec_size() == HOLDER_SIZE + ROTATOR_SIZE

    def test_holder_block_comes_first(self):
        shared = fake_shared(3)
        holder, rotator = extract(shared, Role.HOLDER), extract(shared, Role.ROTATOR)
        assert holder.shape == (3, HOLDER_SIZE)
        assert rotator.shape == (3, ROTATOR_SIZE)
        assert np.array_equal(holder, shared[:, :HOLDER_SIZE])
        assert np.array_equal(rotator, shared[:, HOLDER_SIZE:])

    def test_extract_returns_contract_field_values(self):
        shared = fake_shared(2)
        holder = extract(shared, Role.HOLDER)
        slices = own_block_slices(HOLDER_N)
        assert np.array_equal(holder[..., slices["time_fraction"]], shared[:, 0:1])
        assert np.array_equal(
            holder[..., slices["own_primitive_progress"]], shared[:, HOLDER_SIZE - 1 : HOLDER_SIZE]
        )

    def test_wrong_width_rejected(self):
        with pytest.raises(ValueError, match="columns"):
            extract(np.zeros((2, shared_vec_size() - 1)), Role.HOLDER)


class TestAgentObsDim:
    @pytest.mark.parametrize("mode", PARTNER_VISIBILITY_MODES)
    @pytest.mark.parametrize("role", (Role.HOLDER, Role.ROTATOR))
    def test_matches_the_skills_obs_contract(self, mode, role):
        assert agent_obs_dim(mode, role) == policy_input_size(mode, role)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            agent_obs_dim("full", Role.HOLDER)


class TestBuildAgentObs:
    @pytest.mark.parametrize("mode", PARTNER_VISIBILITY_MODES)
    @pytest.mark.parametrize("role", (Role.HOLDER, Role.ROTATOR))
    def test_output_width_is_the_contract(self, mode, role):
        obs = build_agent_obs(
            fake_shared(4), fake_proprio(4), partner_info(role, np.zeros(4, dtype=int)), mode, role
        )
        assert obs.shape == (4, agent_obs_dim(mode, role))

    @pytest.mark.parametrize("mode", ("none", "predicted", "predicted-direct"))
    @pytest.mark.parametrize("role", (Role.HOLDER, Role.ROTATOR))
    def test_modes_without_partner_info_contain_no_partner_ground_truth(self, mode, role):
        partner = Role.ROTATOR if role is Role.HOLDER else Role.HOLDER
        shared = fake_shared(5)
        extract(shared, partner)[...] = PARTNER_SENTINEL
        obs = build_agent_obs(shared, fake_proprio(5), {}, mode, role)
        # The partner's whole own block (one-hot, progress, proprio, everything) is sentinel-
        # marked; none of it may leak into the agent's input in any form.
        assert not np.any(obs == PARTNER_SENTINEL)
        # The non-oracle partner block is exactly nothing (none) or the zero stub.
        own = own_block_size(num_primitives(role))
        assert np.all(obs[:, own:] == 0.0)

    def test_predicted_stub_widths_match_the_contract(self):
        for role in (Role.HOLDER, Role.ROTATOR):
            for mode in ("predicted", "predicted-direct"):
                obs = build_agent_obs(fake_shared(2), None, {}, mode, role)
                stub = agent_obs_dim(mode, role) - own_block_size(num_primitives(role))
                assert obs.shape == (2, own_block_size(num_primitives(role)) + stub)
                assert np.all(obs[:, -stub:] == 0.0)

    @pytest.mark.parametrize("role", (Role.HOLDER, Role.ROTATOR))
    def test_oracle_contains_exactly_onehot_and_progress(self, role):
        partner = Role.ROTATOR if role is Role.HOLDER else Role.HOLDER
        n_partner = num_primitives(partner)
        ids = np.arange(4) % n_partner
        shared = fake_shared(4)
        obs = build_agent_obs(shared, None, partner_info(role, ids), "oracle", role)
        own = own_block_size(num_primitives(role))
        assert obs.shape == (4, own + n_partner + 1)
        onehot = obs[:, own : own + n_partner]
        assert np.array_equal(onehot, np.eye(n_partner, dtype=np.float32)[ids])
        progress = obs[:, own + n_partner :]
        partner_progress = own_block_slices(n_partner)["own_primitive_progress"]
        assert np.array_equal(progress, extract(shared, partner)[..., partner_progress])

    def test_partner_progress_is_passed_through_correctly(self):
        shared = fake_shared(6)
        progress_value = 0.837
        shared[:, -1] = progress_value  # the rotator's own_primitive_progress column
        obs = build_agent_obs(
            shared, None, partner_info(Role.HOLDER, np.full(6, 3)), "oracle", Role.HOLDER
        )
        assert np.allclose(obs[:, -1], progress_value)

    def test_oracle_without_info_key_rejected(self):
        with pytest.raises(ValueError, match="'oracle' needs info"):
            build_agent_obs(fake_shared(2), None, {}, "oracle", Role.HOLDER)

    def test_oracle_with_out_of_range_ids_rejected(self):
        info = {INFO_PRIMITIVE_ROTATOR: np.array([0, ROTATOR_N])}
        with pytest.raises(ValueError, match=r"in \[0,"):
            build_agent_obs(fake_shared(2), None, info, "oracle", Role.HOLDER)

    def test_oracle_with_mismatched_batch_rejected(self):
        info = {INFO_PRIMITIVE_HOLDER: np.zeros(9, dtype=int)}
        with pytest.raises(ValueError, match="batch shape"):
            build_agent_obs(fake_shared(2), None, info, "oracle", Role.ROTATOR)

    def test_proprio_overwrites_the_own_block_fields(self):
        shared, proprio = fake_shared(3), fake_proprio(3)
        obs = build_agent_obs(shared, proprio, {}, "none", Role.HOLDER)
        slices = own_block_slices(HOLDER_N)
        assert np.array_equal(obs[..., slices["own_qpos"]], proprio[:, :7])
        assert np.array_equal(obs[..., slices["own_qvel"]], proprio[:, 7:])
        assert np.array_equal(
            obs[..., slices["own_gripper_opening"]], shared[:, slices["own_gripper_opening"]]
        )

    def test_proprio_none_keeps_shared_vector_values(self):
        shared = fake_shared(3)
        obs = build_agent_obs(shared, None, {}, "none", Role.ROTATOR)
        assert np.array_equal(obs, extract(shared, Role.ROTATOR))

    def test_wrong_proprio_width_rejected(self):
        with pytest.raises(ValueError, match="proprio has width"):
            build_agent_obs(fake_shared(2), np.zeros((2, PROPRIO_DIM + 1)), {}, "none", Role.HOLDER)


def structured_obs(batch: int) -> dict:
    """A skills env's structured state obs: agent proprio keys + the two own blocks."""
    shared = fake_shared(batch)
    return {
        "agent": {
            uid: {
                "qpos": torch.arange(batch * 7, dtype=torch.float32).reshape(batch, 7) / 7 - batch,
                "qvel": torch.arange(batch * 7, dtype=torch.float32).reshape(batch, 7) / 11 - batch,
            }
            for uid in UIDS
        },
        "extra": {
            "holder_own_block": torch.as_tensor(shared[:, :HOLDER_SIZE]),
            "rotator_own_block": torch.as_tensor(shared[:, HOLDER_SIZE:]),
        },
    }


class TestSkillsObsBuilder:
    @pytest.mark.parametrize("mode", PARTNER_VISIBILITY_MODES)
    def test_matches_build_agent_obs(self, mode):
        batch = 6
        structured = structured_obs(batch)
        flat = flatten_structured(structured)
        builder = SkillsObsBuilder.from_env_obs(structured, flat, mode, UIDS)
        # qpos(7) + qvel(7) per agent, the agent group first in this fixture's insertion order.
        proprio = [flat[:, 0:14].numpy(), flat[:, 14:28].numpy()]
        infos = {
            INFO_PRIMITIVE_HOLDER: torch.arange(batch) % HOLDER_N,
            INFO_PRIMITIVE_ROTATOR: torch.arange(batch) % ROTATOR_N,
        }
        for i, role in enumerate((Role.HOLDER, Role.ROTATOR)):
            assert builder.obs_dims[i] == agent_obs_dim(mode, role)
            torch_obs = builder(flat, infos)[i]
            numpy_obs = build_agent_obs(flat[:, 28:].numpy(), proprio[i], infos, mode, role)
            assert torch.allclose(torch_obs, torch.as_tensor(numpy_obs), atol=1e-6)

    def test_oracle_block_reads_infos_and_shared_progress(self):
        batch = 5
        structured = structured_obs(batch)
        flat = flatten_structured(structured)
        builder = SkillsObsBuilder.from_env_obs(structured, flat, "oracle", UIDS)
        ids = torch.tensor([0, 1, 2, 3, 4])
        holder, rotator = builder(
            flat, {INFO_PRIMITIVE_ROTATOR: ids, INFO_PRIMITIVE_HOLDER: torch.zeros(batch)}
        )
        onehot = holder[:, HOLDER_SIZE : HOLDER_SIZE + ROTATOR_N]
        assert torch.equal(onehot, torch.eye(ROTATOR_N)[ids])
        assert torch.allclose(
            holder[:, HOLDER_SIZE + ROTATOR_N :].flatten(),
            flat[:, 28 + HOLDER_SIZE + ROTATOR_SIZE - 1],
        )
        # The rotator observes the holder: its partner block is sized by the holder registry.
        assert rotator.shape[1] == ROTATOR_SIZE + HOLDER_N + 1

    def test_none_has_no_partner_ground_truth(self):
        structured = structured_obs(4)
        structured["extra"]["holder_own_block"][...] = 111.0
        structured["extra"]["rotator_own_block"][...] = 222.0
        flat = flatten_structured(structured)
        builder = SkillsObsBuilder.from_env_obs(structured, flat, "none", UIDS)
        holder, rotator = builder(flat, None)
        assert not bool((holder == 222.0).any())  # holder never sees the rotator's ground truth
        assert not bool((rotator == 111.0).any())  # rotator never sees the holder's

    def test_predicted_stub_is_zero(self):
        structured = structured_obs(3)
        flat = flatten_structured(structured)
        builder = SkillsObsBuilder.from_env_obs(structured, flat, "predicted", UIDS)
        for i, role in enumerate((Role.HOLDER, Role.ROTATOR)):
            obs = builder(flat, None)[i]
            stub = agent_obs_dim("predicted", role) - own_block_size(num_primitives(role))
            assert obs.shape[1] == agent_obs_dim("predicted", role)
            assert bool((obs[:, -stub:] == 0).all())

    def test_fields_and_agent_uids(self):
        structured = structured_obs(2)
        builder = SkillsObsBuilder.from_env_obs(
            structured, flatten_structured(structured), "oracle", UIDS
        )
        assert builder.agent_uids == UIDS
        assert builder.fields[0][-2:] == ["partner/primitive_onehot", "partner/primitive_progress"]
        assert builder.fields[1][0] == "own/time_fraction"

    def test_wrong_shared_width_rejected(self):
        structured = structured_obs(2)
        structured["extra"]["rotator_own_block"] = structured["extra"]["rotator_own_block"][:, :-1]
        with pytest.raises(ValueError, match="contiguous run of width"):
            SkillsObsBuilder.from_env_obs(structured, flatten_structured(structured), "none", UIDS)

    def test_missing_proprio_key_rejected(self):
        structured = structured_obs(2)
        del structured["agent"][UIDS[1]]["qvel"]
        with pytest.raises(ValueError, match="missing agent"):
            SkillsObsBuilder.from_env_obs(structured, flatten_structured(structured), "none", UIDS)
