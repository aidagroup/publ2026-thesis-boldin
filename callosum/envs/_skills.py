"""Skill-level action space: the primitive registries per role, the partner-visibility
rule and the episode-randomization sampler.

Pure stdlib on purpose (the same rule as `callosum.envs._partner_obs`, design decision 2
of openspec/changes/add-skill-level-cube-assembly): everything here must import and unit
test on macOS, where mani_skill and torch are absent. The env (task 2.x) only consumes
these tables instead of re-deriving them inside GPU codepaths.

Identity convention: a primitive's id is its `index` in its role's registry and is unique
WITHIN the role -- each hand has its own discrete action head, so a hand's action id is
the index into its own set. Names shared by both sets ("wait", "grasp", "release",
"retreat") are different primitives per role; the global key is `(role, name)`.

Nothing here imports or reuses the expert's runtime primitives
(`callosum.experts._phases`): those are executable phase machines (torch-free but
per-env numpy state). This module is only the *menu* the policy chooses from.
"""

import random
from dataclasses import dataclass
from enum import Enum


class Role(Enum):
    """Which hand plays which fixed part in the role-split configuration (E1/E2).

    agent_a (azimuth -pi/2, the low clamp) is the holder, agent_b the rotator --
    the same assignment as FaceTurn-v0 (see `callosum.configs.layout.ArmLayout`).
    """

    HOLDER = "holder"
    ROTATOR = "rotator"


@dataclass(frozen=True)
class SkillPrimitive:
    """One entry of a role's registry: a scripted skill a hand may choose.

    `index` is the primitive's stable position in its role's tuple (see the module
    doc for the identity convention); it doubles as the action id of the hand's
    discrete action space, so the order below is part of the contract.
    """

    name: str
    role: Role
    index: int


def _build_registry(role: Role, names: tuple[str, ...]) -> tuple[SkillPrimitive, ...]:
    """Freeze an ordered name tuple into primitives, rejecting duplicates."""
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate primitive names for role {role.value}: {names}")
    return tuple(SkillPrimitive(name, role, index) for index, name in enumerate(names))


# The spec's fixed sets (skill-action-space: "Наборы примитивов по ролям"). The order
# is the action-id order and MUST NOT change between runs or versions.
HOLDER_PRIMITIVES = _build_registry(
    Role.HOLDER,
    (
        "wait",
        "approach_body",
        "grasp",
        "release",
        "lift",
        "lower",
        "retreat",
        "flip90",
        "flip180",
    ),
)
ROTATOR_PRIMITIVES = _build_registry(
    Role.ROTATOR,
    (
        "wait",
        "approach_face",
        "grasp",
        "turn90",
        "turn180",
        "release",
        "retreat",
    ),
)
PRIMITIVES_BY_ROLE = {Role.HOLDER: HOLDER_PRIMITIVES, Role.ROTATOR: ROTATOR_PRIMITIVES}


def primitives(role: Role) -> tuple[SkillPrimitive, ...]:
    """The role's registry, in stable action-id order."""
    return PRIMITIVES_BY_ROLE[role]


def num_primitives(role: Role) -> int:
    """Size of the role's discrete action space."""
    return len(PRIMITIVES_BY_ROLE[role])


def primitive_by_name(role: Role, name: str) -> SkillPrimitive:
    """Look up a primitive of `role` by name (unknown name: ValueError)."""
    for primitive in PRIMITIVES_BY_ROLE[role]:
        if primitive.name == name:
            return primitive
    raise ValueError(f"unknown {role.value} primitive {name!r}")


# --- partner visibility -----------------------------------------------------------------

PARTNER_VISIBILITY_MODES = ("none", "oracle", "predicted", "predicted-direct")
"""What the policy input may know about the partner (cube-environment spec: "Наблюдения
по маскам режима партнёра"). Replaces the joint-level flag of `_partner_obs`
("full"/"none") at the skill level."""


def validate_partner_visibility(mode: str) -> None:
    """Raise ValueError if `mode` isn't one of PARTNER_VISIBILITY_MODES."""
    if mode not in PARTNER_VISIBILITY_MODES:
        raise ValueError(f"mode must be one of {PARTNER_VISIBILITY_MODES}, got {mode!r}")


@dataclass(frozen=True)
class PartnerVisibility:
    """What enters an agent's policy input about its partner, per mode.

    * "none": nothing about the partner (the control arm of the occlusion ablation).
    * "oracle": the partner's CURRENT primitive as a one-hot over the partner role's
      registry plus the partner's primitive progress [0, 1] -- and only those.
    * "predicted": the same env-level inputs as "none". The Bi-JEPA latent is computed
      from the agent's OWN observation by code outside this module (spring change), so
      this module only fixes the size contract: `partner_block_size` returns the
      configured latent width.
    * "predicted-direct": same as "none" plus a direct classifier's output over the
      partner's primitives (baseline), again supplied externally; the size contract is
      the partner role's registry size.

    The mode decides only the partner block; the agent's own block (own primitive
    one-hot + progress, gripper opening, cube state, plan cursor, time fraction) is the
    same under every mode and is built by the env.
    """

    mode: str

    def __post_init__(self) -> None:
        validate_partner_visibility(self.mode)

    @property
    def partner_fields(self) -> tuple[str, ...]:
        """Ground-truth partner fields that enter the observing agent's input."""
        if self.mode == "oracle":
            return ("partner_primitive_onehot", "partner_primitive_progress")
        return ()

    @property
    def sees_partner(self) -> bool:
        """Whether any ground-truth partner field enters the input."""
        return bool(self.partner_fields)

    def partner_block_size(self, partner_num_primitives: int, latent_dim: int = 0) -> int:
        """Width of the partner-related block of the policy input.

        "none" contributes nothing; "oracle" the partner's primitive one-hot
        (`partner_num_primitives`) plus one progress scalar; "predicted" the
        externally-supplied Bi-JEPA latent (`latent_dim`, must be >= 1);
        "predicted-direct" the externally-supplied distribution over the partner's
        primitives (`partner_num_primitives`).
        """
        if partner_num_primitives < 1:
            raise ValueError(f"partner_num_primitives must be >= 1, got {partner_num_primitives}")
        if self.mode == "none":
            return 0
        if self.mode == "oracle":
            return partner_num_primitives + 1
        if self.mode == "predicted":
            if latent_dim < 1:
                raise ValueError(f"latent_dim must be >= 1 for 'predicted', got {latent_dim}")
            return latent_dim
        return partner_num_primitives  # "predicted-direct"


# --- episode randomization ---------------------------------------------------------------


@dataclass(frozen=True)
class RandomizationConfig:
    """Ranges of the episode-private randomness (design decision 9).

    These are the knobs that make the partner's state genuinely unpredictable (an
    open-loop schedule cannot work); the defaults are placeholders, tuned on E1 so the
    oracle-none gap exists while oracle stays solvable (design open question). All
    ranges are inclusive; a degenerate (lo == hi) range fixes the value.
    """

    episode_speed_range: tuple[float, float] = (0.8, 1.2)
    """One speed multiplier sampled per episode, applied to every primitive of both
    hands (global pacing of the episode)."""
    arm_speed_range: tuple[float, float] = (0.9, 1.1)
    """An extra speed multiplier sampled per hand, on top of the episode one (a fast
    holder with a slow rotator and vice versa)."""
    grasp_failure_prob_range: tuple[float, float] = (0.05, 0.2)
    """Per-hand probability that a grasp attempt fails (the injection that demands a
    re-grasp; sampled per episode, applied to each grasp event)."""
    start_delay_range: tuple[float, float] = (0.0, 2.0)
    """Per-hand delay (seconds) before the hand's first primitive may start. To each
    hand this is the partner's private start delay -- the moment of the partner's
    readiness cannot be known in advance."""

    def __post_init__(self) -> None:
        for name in ("episode_speed_range", "arm_speed_range"):
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi:
                raise ValueError(f"{name} must be 0 < lo <= hi, got {(lo, hi)}")
        lo, hi = self.grasp_failure_prob_range
        if not 0 <= lo <= hi <= 1:
            raise ValueError(
                f"grasp_failure_prob_range must be within [0, 1], got {(lo, hi)}"
            )
        lo, hi = self.start_delay_range
        if not 0 <= lo <= hi:
            raise ValueError(f"start_delay_range must be lo >= 0 and lo <= hi, got {(lo, hi)}")


@dataclass(frozen=True)
class EpisodeRandomization:
    """The sampled private randomness of one episode (holder entry first).

    `arm_multiplier(arm)` is the hand's effective primitive-speed multiplier:
    `episode_speed * arm_speed[arm]`. Everything here is private per hand except the
    episode-wide speed; the whole record is the "hidden state" that makes the partner
    only partially observable.
    """

    seed: int
    episode_speed: float
    arm_speed: tuple[float, float]
    grasp_failure_prob: tuple[float, float]
    start_delay: tuple[float, float]

    def arm_multiplier(self, arm: int) -> float:
        """Effective speed multiplier of hand 0 (holder) or 1 (rotator)."""
        if arm not in (0, 1):
            raise ValueError(f"arm must be 0 or 1, got {arm}")
        return self.episode_speed * self.arm_speed[arm]


def sample_episode_randomization(config: RandomizationConfig, seed: int) -> EpisodeRandomization:
    """Deterministically sample one episode's private randomness from `seed`.

    Uses stdlib `random.Random` (no torch; the sampler must run on macOS and inside
    the env's reset). The draw order is fixed -- episode speed, then per-hand arm
    speeds, grasp-failure probabilities and start delays, each (hand 0, hand 1) --
    so a seed fully determines the episode's randomness.
    """
    rng = random.Random(seed)
    episode_speed = rng.uniform(*config.episode_speed_range)
    arm_speed = (
        rng.uniform(*config.arm_speed_range),
        rng.uniform(*config.arm_speed_range),
    )
    grasp_failure_prob = (
        rng.uniform(*config.grasp_failure_prob_range),
        rng.uniform(*config.grasp_failure_prob_range),
    )
    start_delay = (
        rng.uniform(*config.start_delay_range),
        rng.uniform(*config.start_delay_range),
    )
    return EpisodeRandomization(
        seed=seed,
        episode_speed=episode_speed,
        arm_speed=arm_speed,
        grasp_failure_prob=grasp_failure_prob,
        start_delay=start_delay,
    )
