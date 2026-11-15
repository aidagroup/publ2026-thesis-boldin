"""Configuration of FaceTurnSkills-v0: the skill-level cube-assembly environment.

Grouping the knobs the spec leaves open by design ("ключевые параметры неизвестны до
замеров -> всё в конфиге", design.md risks): the decision period, the episode limit in
decisions, the plan input, the episode randomization ranges (the private randomness of
the partner), the failure-injection probabilities beyond `RandomizationConfig` (layer
slips), the team reward and the generalized layer-lock tolerances.

Pure stdlib on purpose (the macOS rule of `callosum.envs._skills`): the config must
import and validate wherever the env's logic is unit-tested. The multi-face cube's own
geometry (layer axes, friction, detents) lives in
`callosum.configs.multi_face_cube.MultiFaceCubeConfig` and is referenced, not duplicated.
"""

import math
from dataclasses import dataclass, field

from callosum.envs._skills import (
    RandomizationConfig,
    validate_partner_visibility,
)

# The agent keys the env and the conductor use for the two fixed roles (agent_a = holder,
# agent_b = rotator, the FaceTurn-v0 convention).
AGENT_KEYS = ("holder", "rotator")


@dataclass(frozen=True)
class SkillsRewardConfig:
    """The team reward of one decision (design decision: no dense motor terms).

    The reward is shared by both agents (cooperative): a one-time bonus on the decision
    whose substeps finish the plan, minus a small per-decision time penalty. Discounting
    happens per decision in the trainer, so `success_bonus` dominates any continuation:
    1.0 vs `step_penalty * remaining decisions` (< 0.8 at the defaults).
    """

    success_bonus: float = 1.0
    step_penalty: float = 0.01

    def __post_init__(self) -> None:
        if self.success_bonus < 0:
            raise ValueError(f"success_bonus must be >= 0, got {self.success_bonus}")
        if self.step_penalty < 0:
            raise ValueError(f"step_penalty must be >= 0, got {self.step_penalty}")
        if self.step_penalty >= self.success_bonus:
            raise ValueError(
                f"step_penalty ({self.step_penalty}) must stay below success_bonus "
                f"({self.success_bonus}), otherwise lingering beats finishing"
            )


@dataclass(frozen=True)
class FaceTurnSkillsConfig:
    """All tunables of FaceTurnSkills-v0 except the cube's articulation geometry.

    Field groups: decision timing, the plan input, partner-visibility bookkeeping,
    randomization/failure injection, reward, and the generalized layer-lock/success
    tolerances (lifted from `FaceTurnPhysicsConfig`/`FaceTurnRewardConfig` where they
    apply to every layer now).
    """

    # --- decision timing -----------------------------------------------------------------
    decision_period: int = 20
    """Simulation substeps per decision (one env.step). ~20 steps ~ 1 s: the A40
    probe measured a primitive leg at 150-300 substeps (the wrist carries the
    cube's weight during a flip and sags to a fraction of the delta-limit speed),
    so a 0.5 s decision cadence spent most of its budget inside one leg's travel;
    1 s keeps several decisions per leg while a 150-decision episode still spans
    the full plan. The episode limit is counted in decisions, not substeps."""

    max_episode_decisions: int = 150
    """Episode budget in decisions (one decision = `decision_period` sim substeps).
    Measured on the A40 probe: with 1 s decisions a plan move WITHOUT a flip is
    ~20-25 decisions and WITH a flip ~55-70 (lift, roll, set-down, re-grasp, turn);
    E1 (1 move, flip possible) fits in ~70-80, E2 (2-3 moves) in ~120-140. NOTE:
    `gym.make(..., max_episode_steps=...)` only rewraps the outer TimeLimit
    (decision-level); to change this internal budget pass
    `skills_config=FaceTurnSkillsConfig(max_episode_decisions=...)`."""

    # --- plan input ----------------------------------------------------------------------
    plan_length: int = 1
    """Moves per episode (E1: 1, E2: 2-3, E3: 5+). The plan itself is sampled per episode
    and per env by `callosum.envs._plans.random_plan`; planning is not learned."""

    # --- partner visibility (bookkeeping only) --------------------------------------------
    partner_mode: str = "none"
    """One of `PARTNER_VISIBILITY_MODES`. The env's observation does NOT depend on it
    (masking happens in the trainer's input builder, `callosum.training.
    _skills_agent_obs`); stored and validated so the env config mirrors the run."""

    # --- randomization and failure injection ----------------------------------------------
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    """Episode-private randomness ranges (speeds, grasp-failure probability, start
    delay); see `callosum.envs._skills.RandomizationConfig`."""

    slip_prob_range: tuple[float, float] = (0.05, 0.15)
    """Per-hand probability range that a layer TURN ends in an injected slip: the held
    layer escapes mid-turn (the runtime drops the rotator's grip and aborts the turn),
    recovery is a re-grasp. Sampled per episode like the grasp-failure probability."""

    # --- reward ---------------------------------------------------------------------------
    reward: SkillsRewardConfig = field(default_factory=SkillsRewardConfig)
    """The team reward per decision (`SkillsRewardConfig`)."""

    # --- generalized layer lock and success tolerances --------------------------------------
    lock_layers_unless_held: bool = True
    """Every layer is kinematically clamped while the holder does not grasp the
    core (the FaceTurn face-lock rule lifted to all six layers; a guarantee, not a
    reward)."""

    lock_write_tol_deg: float = 1.0
    """Skip threshold (deg) of the lock's per-substep qpos correction: a locked
    layer whose angle deviates less than this from its latched value is left
    alone (the layers' own joint friction holds the sub-degree wiggle)."""

    lock_pull_step_deg: float = 0.2
    """Rate limit (deg per sim substep) of the lock's qpos correction: a deviating
    layer is pulled back towards its latched angle by at most this much per
    substep instead of being teleported. The snap-back rewrite fought the GPU
    solver -- every teleport re-introduced a micro-penetration and the solver's
    depenetration expelled the cube, observed as the cube gliding/coasting tens
    of cm after the flip's set-down (the unheld slabs jitter +-1-2.5 deg from the
    table-contact shake, so any snap threshold kept firing). A 0.2 deg step is at
    the scale of the solver's own contact micro-bounces and cannot inject a
    drive, while a real turn attempt (90 deg) is still undone within one
    decision period."""
    angle_tol_deg: float = 4.0
    """Tolerance (deg) of a layer angle to its detent target for a plan move to count
    as executed (FaceTurn-v0 used 0.05 rad ~ 2.9 deg on the single face; a touch looser
    here because flips re-seat the cube between moves)."""

    body_pos_tol: float = 0.02
    """Success tolerance (m) of the cube-center displacement from its initial pose.
    The flip macro sets the cube down with a short drop off the lower jaw (the
    loaded wrist cannot press it fully onto the table), so the cube realistically
    drifts ~1 cm per flip (roll arc + landing nudge); the friction material keeps
    the drift bounded, the tolerance admits it. Rotation vs the initial pose is
    deliberately NOT checked -- flips are required plan actions; stability instead
    means the layers are at rest and the position bound holds."""

    body_settle_qvel: float = 0.3
    """Max layer-joint speed (rad/s) for the cube to count as settled at success
    evaluation (the layers carry the whole articulation's readable velocity state;
    TODO(review): replace with the root link's twist read on the GPU backend)."""

    def __post_init__(self) -> None:
        if self.decision_period < 1:
            raise ValueError(f"decision_period must be >= 1, got {self.decision_period}")
        if self.max_episode_decisions < 1:
            raise ValueError(
                f"max_episode_decisions must be >= 1, got {self.max_episode_decisions}"
            )
        if self.plan_length < 0:
            raise ValueError(f"plan_length must be >= 0, got {self.plan_length}")
        validate_partner_visibility(self.partner_mode)
        lo, hi = self.slip_prob_range
        if not 0 <= lo <= hi <= 1:
            raise ValueError(
                f"slip_prob_range must be within [0, 1] with lo <= hi, got {self.slip_prob_range}"
            )
        if not 0 < self.angle_tol_deg < 45:
            raise ValueError(f"angle_tol_deg must be in (0, 45), got {self.angle_tol_deg}")
        if not 0 < self.lock_pull_step_deg <= self.lock_write_tol_deg:
            raise ValueError(
                f"lock_pull_step_deg must be within (0, lock_write_tol_deg] = "
                f"(0, {self.lock_write_tol_deg}], got {self.lock_pull_step_deg}"
            )
        if not self.lock_write_tol_deg < self.angle_tol_deg:
            raise ValueError(
                f"lock_write_tol_deg must be within [0, angle_tol_deg) = "
                f"[0, {self.angle_tol_deg}), got {self.lock_write_tol_deg}"
            )
        if self.body_pos_tol <= 0:
            raise ValueError(f"body_pos_tol must be > 0, got {self.body_pos_tol}")
        if self.body_settle_qvel <= 0:
            raise ValueError(f"body_settle_qvel must be > 0, got {self.body_settle_qvel}")

    @property
    def angle_tol_rad(self) -> float:
        """`angle_tol_deg` in the simulator's unit (radians)."""
        return math.radians(self.angle_tol_deg)

    @property
    def lock_write_tol_rad(self) -> float:
        """`lock_write_tol_deg` in radians (the lock correction skip threshold)."""
        return math.radians(self.lock_write_tol_deg)

    @property
    def lock_pull_step_rad(self) -> float:
        """`lock_pull_step_deg` in radians (the per-substep correction rate limit)."""
        return math.radians(self.lock_pull_step_deg)
