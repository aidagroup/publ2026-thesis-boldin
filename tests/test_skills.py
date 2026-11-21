"""Unit tests for the skill-level registry, partner-visibility rule and sampler.

Tests callosum.envs._skills directly: the module is pure stdlib (no mani_skill/torch),
which is exactly what makes it testable on macOS/CI (see the module's docstring).
"""

import pytest

from callosum.envs._skills import (
    HOLDER_PRIMITIVES,
    PARTNER_VISIBILITY_MODES,
    ROTATOR_PRIMITIVES,
    EpisodeRandomization,
    PartnerVisibility,
    RandomizationConfig,
    Role,
    num_primitives,
    primitive_by_name,
    primitives,
    sample_episode_randomization,
    validate_partner_visibility,
)

# The spec's fixed sets; the order is the action-id order and part of the contract.
HOLDER_NAMES = (
    "wait",
    "approach_body",
    "grasp",
    "release",
    "lift",
    "lower",
    "retreat",
    "flip90",
    "flip180",
)
ROTATOR_NAMES = (
    "wait",
    "approach_face",
    "grasp",
    "turn90",
    "turn180",
    "release",
    "retreat",
)


# --- registry ---------------------------------------------------------------------------


def test_registries_are_complete() -> None:
    assert tuple(p.name for p in HOLDER_PRIMITIVES) == HOLDER_NAMES
    assert tuple(p.name for p in ROTATOR_PRIMITIVES) == ROTATOR_NAMES


def test_registries_have_unique_names_and_stable_indices() -> None:
    for registry, role in ((HOLDER_PRIMITIVES, Role.HOLDER), (ROTATOR_PRIMITIVES, Role.ROTATOR)):
        names = [p.name for p in registry]
        assert len(set(names)) == len(names)  # unique within the role
        assert [p.index for p in registry] == list(range(len(registry)))  # index == position
        assert all(p.role is role for p in registry)


def test_registry_accessors() -> None:
    assert primitives(Role.HOLDER) == HOLDER_PRIMITIVES
    assert primitives(Role.ROTATOR) == ROTATOR_PRIMITIVES
    assert num_primitives(Role.HOLDER) == 9
    assert num_primitives(Role.ROTATOR) == 7
    assert primitive_by_name(Role.HOLDER, "flip180").index == 8
    assert primitive_by_name(Role.ROTATOR, "turn90").index == 3
    with pytest.raises(ValueError, match="rotator"):
        primitive_by_name(Role.ROTATOR, "flip90")  # a holder-only skill
    with pytest.raises(ValueError, match="holder"):
        primitive_by_name(Role.HOLDER, "turn90")  # a rotator-only skill


def test_shared_names_are_different_primitives_per_role() -> None:
    """wait/grasp/release/retreat exist in BOTH sets but are distinct actions."""
    for name in ("wait", "grasp", "release", "retreat"):
        holder = primitive_by_name(Role.HOLDER, name)
        rotator = primitive_by_name(Role.ROTATOR, name)
        assert holder != rotator
        assert (holder.role, holder.name) != (rotator.role, rotator.name)


# --- partner visibility -----------------------------------------------------------------


@pytest.mark.parametrize("mode", PARTNER_VISIBILITY_MODES)
def test_visibility_accepts_known_modes(mode: str) -> None:
    validate_partner_visibility(mode)
    assert PartnerVisibility(mode).mode == mode


def test_visibility_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        validate_partner_visibility("full")  # the joint-level flag, not a skill-level mode
    with pytest.raises(ValueError, match="mode"):
        PartnerVisibility("sometimes")


@pytest.mark.parametrize("mode", ["none", "predicted", "predicted-direct"])
def test_non_oracle_modes_add_no_partner_fields(mode: str) -> None:
    visibility = PartnerVisibility(mode)
    assert visibility.partner_fields == ()
    assert not visibility.sees_partner


def test_oracle_adds_exactly_primitive_onehot_and_progress() -> None:
    visibility = PartnerVisibility("oracle")
    assert visibility.partner_fields == ("partner_primitive_onehot", "partner_primitive_progress")
    assert visibility.sees_partner


def test_partner_block_sizes_follow_the_mode_contract() -> None:
    n_holder, n_rotator = num_primitives(Role.HOLDER), num_primitives(Role.ROTATOR)
    # none: no partner block at all.
    assert PartnerVisibility("none").partner_block_size(n_rotator) == 0
    # oracle: the partner's one-hot plus one progress scalar.
    assert PartnerVisibility("oracle").partner_block_size(n_holder) == n_holder + 1
    assert PartnerVisibility("oracle").partner_block_size(n_rotator) == n_rotator + 1
    # predicted: exactly the externally-supplied latent width.
    assert PartnerVisibility("predicted").partner_block_size(n_rotator, latent_dim=32) == 32
    # predicted-direct: a distribution over the partner's primitives.
    assert PartnerVisibility("predicted-direct").partner_block_size(n_holder) == n_holder


def test_partner_block_size_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="partner_num_primitives"):
        PartnerVisibility("oracle").partner_block_size(0)
    with pytest.raises(ValueError, match="latent_dim"):
        PartnerVisibility("predicted").partner_block_size(7)  # latent_dim missing


# --- episode randomization --------------------------------------------------------------


def test_sampler_is_deterministic_per_seed() -> None:
    config = RandomizationConfig()
    for seed in (0, 1, 42, 123456):
        first = sample_episode_randomization(config, seed)
        second = sample_episode_randomization(config, seed)
        assert first == second
        assert first.seed == seed


def test_sampler_differs_across_seeds_and_stays_in_ranges() -> None:
    config = RandomizationConfig()
    samples = [sample_episode_randomization(config, seed) for seed in range(20)]
    assert len(set(samples)) == len(samples)
    for sample in samples:
        assert (
            config.episode_speed_range[0] <= sample.episode_speed <= (config.episode_speed_range[1])
        )
        for arm in (0, 1):
            assert config.arm_speed_range[0] <= sample.arm_speed[arm] <= (config.arm_speed_range[1])
            assert (
                config.grasp_failure_prob_range[0]
                <= sample.grasp_failure_prob[arm]
                <= (config.grasp_failure_prob_range[1])
            )
            assert (
                config.start_delay_range[0]
                <= sample.start_delay[arm]
                <= (config.start_delay_range[1])
            )


def test_effective_arm_multiplier_combines_episode_and_arm_speeds() -> None:
    sample = EpisodeRandomization(
        seed=0,
        episode_speed=0.8,
        arm_speed=(1.1, 0.9),
        grasp_failure_prob=(0.0, 0.0),
        start_delay=(0.0, 0.0),
    )
    assert sample.arm_multiplier(0) == pytest.approx(0.88)
    assert sample.arm_multiplier(1) == pytest.approx(0.72)
    with pytest.raises(ValueError, match="arm"):
        sample.arm_multiplier(2)


def test_degenerate_ranges_fix_the_value() -> None:
    config = RandomizationConfig(
        episode_speed_range=(1.0, 1.0),
        arm_speed_range=(1.0, 1.0),
        grasp_failure_prob_range=(0.0, 0.0),
        start_delay_range=(0.0, 0.0),
    )
    sample = sample_episode_randomization(config, seed=7)
    assert sample.episode_speed == 1.0
    assert sample.arm_speed == (1.0, 1.0)
    assert sample.grasp_failure_prob == (0.0, 0.0)
    assert sample.start_delay == (0.0, 0.0)
    assert sample.arm_multiplier(1) == 1.0


def test_randomization_config_rejects_bad_ranges() -> None:
    with pytest.raises(ValueError, match="episode_speed_range"):
        RandomizationConfig(episode_speed_range=(1.2, 0.8))  # lo > hi
    with pytest.raises(ValueError, match="arm_speed_range"):
        RandomizationConfig(arm_speed_range=(0.0, 1.0))  # speed must stay positive
    with pytest.raises(ValueError, match="grasp_failure_prob_range"):
        RandomizationConfig(grasp_failure_prob_range=(-0.1, 0.2))  # outside [0, 1]
    with pytest.raises(ValueError, match="grasp_failure_prob_range"):
        RandomizationConfig(grasp_failure_prob_range=(0.1, 1.2))  # outside [0, 1]
    with pytest.raises(ValueError, match="start_delay_range"):
        RandomizationConfig(start_delay_range=(-1.0, 1.0))  # negative delay
