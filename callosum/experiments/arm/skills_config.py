"""Configuration of FaceTurnSkills-v0: the skill-level cube-assembly environment.

Grouping the knobs the spec leaves open by design ("ключевые параметры неизвестны до
замеров -> всё в конфиге", design.md risks): the decision period, the episode limit in
decisions, the plan input, the episode randomization ranges (the private randomness of
the partner), the failure-injection probabilities beyond `RandomizationConfig` (layer
slips), the team reward and the generalized layer-lock tolerances.

Pure stdlib on purpose (the macOS rule of `callosum.experiments.arm.skills`): the config must
import and validate wherever the env's logic is unit-tested. The multi-face cube's own
geometry (layer axes, friction, detents) lives in
`callosum.experiments.arm.skills_cube_config.MultiFaceCubeConfig` and is referenced, not duplicated.
"""

import math
from dataclasses import dataclass, field

from callosum.experiments.arm.skills import (
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
    decision_period: int = 10
    """Simulation substeps per decision (one env.step). ~10 steps ~ 0.5 s: the
    primitive legs (150-300 substeps) keep several decisions per travel, AND the
    conductor's partner-hold logic counts DECISIONS of consecutive `grasp`
    selections -- at 20 the whole grasp closes within 2 decisions and the rotator's
    hold-belief streak never arms (regression of 3a75d81, seen as the rotator
    waiting through the whole episode). The episode limit is in decisions."""

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
    and per env by `callosum.experiments.arm.skills_plans.random_plan`; planning is not learned."""

    # --- partner visibility (bookkeeping only) --------------------------------------------
    partner_mode: str = "none"
    """One of `PARTNER_VISIBILITY_MODES`. The env's observation does NOT depend on it
    (masking happens in the trainer's input builder, `callosum.training.
    _skills_agent_obs`); stored and validated so the env config mirrors the run."""

    # --- randomization and failure injection ----------------------------------------------
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    """Episode-private randomness ranges (speeds, grasp-failure probability, start
    delay); see `callosum.experiments.arm.skills.RandomizationConfig`."""

    slip_prob_range: tuple[float, float] = (0.05, 0.15)
    """Per-hand probability range that a layer TURN ends in an injected slip: the held
    layer escapes mid-turn (the runtime drops the rotator's grip and aborts the turn),
    recovery is a re-grasp. Sampled per episode like the grasp-failure probability."""

    # --- reward ---------------------------------------------------------------------------
    reward: SkillsRewardConfig = field(default_factory=SkillsRewardConfig)
    """The team reward per decision (`SkillsRewardConfig`)."""

    # --- generalized layer lock and success tolerances --------------------------------------
    turn_assist: bool = True
    """Кинематический ассист поворота (a scripted-motorics simplification by the
    thesis's assumption; the SUBJECT of the work is the two-hand coordination
    policy, not the low-level torque transfer): while the rotator ACTUALLY
    grasps the active layer (an is_grasping check on the layer link, the same
    contact check the useless-action facts use) and its selected primitive is
    turn90/turn180, the layer joint is driven to the authoritative move target
    by per-substep command-following at the arm delta-limit speed, in
    synchrony with the rotator's wrist roll command (the wrist and the layer
    rotate together, so the grip does not scrape the cube). The physical torque
    transfer through the rim pinch was measured as not reliable enough for the
    timeline (a week of debugging: the pinch slips, the cube drifts ~9 cm, the
    angle never moves). Without a real grasp the assist does nothing -- a turn
    attempt on an unheld cube fails exactly as before."""

    lock_layers_unless_held: bool = True
    """Every layer is kinematically clamped while the holder does not grasp the
    core (the FaceTurn face-lock rule lifted to all six layers; a guarantee, not a
    reward)."""

    lock_write_tol_deg: float = 3.0
    """Skip threshold (deg) of the lock's PER-DECISION qpos correction: a locked
    layer whose angle deviates less than this from its latched value is never
    repositioned (only its velocity is zeroed every substep, which darts the
    wiggle but cannot inject position). The unheld slabs jitter +-1-2.5 deg
    continuously -- the GPU solver's table-contact micro-bounce cycle shakes
    the whole cube and the layer joints' friction barely resists -- so ANY
    smaller threshold plus per-substep writes means a kinematic position write
    every substep, and continuous kinematic writes fight the solver (each
    re-introduces a micro-penetration the depenetration expels as a
    constant-velocity drive; observed: the cube gliding/coasting up to 48 cm
    across the table and off its edge). 3 deg sits above the jitter and below
    `angle_tol_deg`; a real turn attempt still triggers the rate-limited
    pull-back."""

    lock_pull_step_deg: float = 1.0
    """Rate limit (deg per DECISION) of the lock's qpos correction once a layer
    deviates beyond `lock_write_tol_deg`: the layer is pulled back towards its
    latched angle by at most this much per decision instead of being teleported.
    The correction runs once per decision (never per substep) and targets the
    nearest detent, a state the solver itself prefers -- so the correction
    converges instead of fighting. A 90 deg forced deviation is undone within a
    couple of decisions."""
    angle_tol_deg: float = 4.0
    """Tolerance (deg) of a layer angle to its detent target for a plan move to count
    as executed (FaceTurn-v0 used 0.05 rad ~ 2.9 deg on the single face; a touch looser
    here because flips re-seat the cube between moves)."""

    body_pos_tol: float = 0.03
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

    cube_home_x: float = 0.02
    """Bias of the cube home (spawn AND the post-flip drag-home target) toward the
    rotator (+x). The rim-grasp IK of the rotator solves only from cube x >= -0.04
    (workspace-mapped on the pod); a centred home leaves ~2 cm of margin that the
    post-flip drift eats -- +2 cm restores a workable envelope while the holder
    (approaching from -x) still reaches the body comfortably."""

    turn_grasp_mode: str = "proximity"
    """How the rotator's grasp of the active layer is established. "contact": only a
    physical `is_grasping` on the layer link counts. "proximity" (default, the
    scripted-motorics simplification of `turn_assist`): the grasp fact additionally
    holds when the rotator has navigated its TCP within the near tolerance of the
    layer AND selected grasp/turn -- the physical rim pinch was measured as
    unreliable (both arms share the space over a 5.7 cm cube; the descent fouls the
    holder's wrist, the pinch slips), while navigation, flips, the wrist roll and
    the layer rotation (the turn assist) all stay physical. The fact is exported as
    `rotator_grasp_fact` in the episode info for the conductor and the harness."""

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
