"""Pure logic of CubeWrench-v0: observation layout, partner visibility, episode
randomisation, and the success/reward/failure rules.

Deliberately has zero mujoco/torch dependency (the same macOS rule as
`callosum.envs._partner_obs` and `callosum.envs._skills`): the simulator lives in
`callosum.envs.wrench_cube.env`, which cannot even be imported on macOS/CI, while
everything that can be decided without the simulator lives here and is unit-tested
locally (`tests/test_wrench_logic.py`).

Observation of ONE agent, in field order (`obs_fields`):

* **own wrench** (6): the agent's most recent commanded normalized wrench -- what its
  latency queue will eventually apply. The agent always knows its own command; this is
  the "previous action" block a policy normally gets.
* **own private parameters** (3): its action latency (s), gain and authority scale. The
  agent knows its own hardware, never the partner's -- that asymmetry is the private
  randomness of the design.
* **shared cube state** (15): core position, orientation quaternion (wxyz), linear and
  angular velocities, layer hinge angle and angular velocity. The reduced model has no
  contacts, so this is the whole observable physics.
* **goal and time** (2): the target hinge angle (rad) and the fraction of the episode
  budget already used.
* **partner block** (9), only under `partner_mode="oracle"`: the partner's most recent
  commanded wrench and its private parameters -- the privileged upper bound of the
  ablation triple. `"predicted"` (Bi-JEPA, phase 3) will replace that block with a
  latent and is deliberately NOT part of this layout yet.

The env builds a `{field_name: value}` dict and `obs_fields` fixes the order and width,
so no observation width is hardcoded in the env.
"""

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from callosum.configs.wrench_cube import WrenchCubeConfig

AGENT_KEYS = ("holder", "rotator")
"""The two role keys, in the env's action-/observation-space order."""

PARTNER_MODES = ("none", "oracle")
"""Partner-visibility modes available at this step of the work ("predicted" arrives
with the Bi-JEPA module; its block must not be added to the layout before that)."""

_WRENCH_AXES = ("fx", "fy", "fz", "tx", "ty", "tz")

OWN_WRENCH_FIELDS = tuple(f"own_wrench_{axis}" for axis in _WRENCH_AXES)
"""Names of the agent's own commanded-wrench block (normalized, [-1, 1])."""

PARTNER_WRENCH_FIELDS = tuple(f"partner_wrench_{axis}" for axis in _WRENCH_AXES)
"""Names of the privileged partner-wrench block ("oracle" mode only)."""

OWN_PARAM_FIELDS = ("own_latency", "own_gain", "own_limit_scale")
"""Names of the agent's own private parameters (latency s, wrench gain, authority scale)."""

PARTNER_PARAM_FIELDS = ("partner_latency", "partner_gain", "partner_limit_scale")
"""Names of the partner's private parameters ("oracle" mode only)."""

PARTNER_FIELDS = PARTNER_WRENCH_FIELDS + PARTNER_PARAM_FIELDS
"""The full partner block ("oracle" mode): the partner's wrench (6) + private parameters (3)."""

CUBE_STATE_FIELDS = (
    "core_x",
    "core_y",
    "core_z",
    "core_qw",
    "core_qx",
    "core_qy",
    "core_qz",
    "core_vx",
    "core_vy",
    "core_vz",
    "core_wx",
    "core_wy",
    "core_wz",
    "layer_angle",
    "layer_ang_vel",
)
"""Names of the shared cube-state block, in the env's assembly order."""

GOAL_FIELDS = ("goal_angle", "time_fraction")
"""Names of the goal block: target hinge angle (rad) and fraction of the time budget."""


def validate_partner_mode(mode: str) -> None:
    """Raise ValueError if `mode` isn't one of `PARTNER_MODES`."""
    if mode not in PARTNER_MODES:
        raise ValueError(f"partner_mode must be one of {PARTNER_MODES}, got {mode!r}")


def partner_fields(mode: str) -> tuple[str, ...]:
    """The partner block of the observation that the mode makes visible.

    Args:
        mode: One of `PARTNER_MODES`.

    Returns:
        `()` for "none"; the partner wrench (6) followed by the partner's private
        parameters (3) for "oracle".
    """
    validate_partner_mode(mode)
    if mode == "none":
        return ()
    return PARTNER_FIELDS


def obs_fields(mode: str) -> tuple[str, ...]:
    """Full ordered field names of one agent's observation under `mode`."""
    return (
        OWN_WRENCH_FIELDS
        + OWN_PARAM_FIELDS
        + CUBE_STATE_FIELDS
        + GOAL_FIELDS
        + partner_fields(mode)
    )


def obs_size(mode: str) -> int:
    """Width of one agent's flat observation vector under `mode`."""
    return len(obs_fields(mode))


# --- episode randomisation ---------------------------------------------------------------


@dataclass(frozen=True)
class AgentRandomization:
    """The sampled private randomness of ONE agent in one episode.

    The first three fields are the agent's private hardware parameters (and the only
    ones the agent sees about itself, under the "own private parameters" obs block);
    the slip and start-delay fields are hidden state that only the environment knows.
    """

    latency_s: float
    """Action latency (s) of this agent: a command is applied `latency_steps` steps later."""
    latency_steps: int
    """`latency_s` rounded to whole control steps."""
    gain: float
    """Wrench-gain multiplier applied to every commanded wrench of this agent."""
    limit_scale: float
    """Multiplier on this agent's config authority (the private "torque limit")."""
    slip_active: bool
    """Whether the episode contains one slip event for this agent."""
    slip_step: int
    """Control step at which the slip event starts (meaningless unless `slip_active`)."""
    slip_steps: int
    """Duration of the slip event in control steps (1-3); the wrench is zero throughout."""
    start_delay_s: float
    """Delay (s) before the agent's first wrench takes effect."""
    start_delay_steps: int
    """`start_delay_s` rounded to whole control steps."""


@dataclass(frozen=True)
class EpisodeRandomization:
    """The sampled private randomness of one episode (holder record first).

    Deterministic in the episode seed (`sample_episode_randomization`); the env stores it
    for the episode and the scripted oracle may read its own agents' latency from it,
    exactly like a policy reads its own private parameters from its observation.
    """

    seed: int
    """Episode seed the record was sampled from."""
    agents: tuple[AgentRandomization, AgentRandomization]
    """Holder record, then rotator record (the `AGENT_KEYS` order)."""
    layer_initial_angle: float
    """Initial layer hinge angle (rad) around 0."""
    impulse_active: bool
    """Whether the one common impulse disturbance happens in this episode."""
    impulse_step: int
    """Control step of the impulse (meaningless unless `impulse_active`)."""
    impulse_force: tuple[float, float, float]
    """Impulse force (N) applied to the core for one control step, world frame."""


def _steps(config: "WrenchCubeConfig", seconds: float) -> int:
    """Convert seconds to whole control steps (rounded, never negative)."""
    return max(0, round(seconds / config.control_dt))


def sample_episode_randomization(config: "WrenchCubeConfig", seed: int) -> EpisodeRandomization:
    """Deterministically sample one episode's private randomness from `seed`.

    Uses stdlib `random.Random` (no torch/numpy; the sampler must run on macOS and inside
    the env's reset). The draw order is fixed so a seed fully determines the episode:
    per agent (holder, then rotator) -- latency, gain, authority scale, slip probability,
    slip Bernoulli, slip duration, slip start step, start delay; then the episode-wide
    layer initial angle, impulse probability, impulse Bernoulli, impulse step, impulse
    direction (three draws) and impulse magnitude.

    Args:
        config: The environment config whose ranges are sampled.
        seed: Episode seed.

    Returns:
        The immutable record; every drawn value is inside its config range.
    """
    rng = random.Random(seed)
    agents = []
    for _ in AGENT_KEYS:
        latency_s = rng.uniform(*config.latency_range)
        gain = rng.uniform(*config.gain_range)
        limit_scale = rng.uniform(*config.limit_scale_range)
        slip_prob = rng.uniform(*config.slip_prob_range)
        slip_active = rng.random() < slip_prob
        slip_steps = rng.randint(*config.slip_decisions_range)
        slip_step = rng.randrange(config.max_control_steps)
        start_delay_s = rng.uniform(*config.start_delay_range)
        agents.append(
            AgentRandomization(
                latency_s=latency_s,
                latency_steps=_steps(config, latency_s),
                gain=gain,
                limit_scale=limit_scale,
                slip_active=slip_active,
                slip_step=slip_step,
                slip_steps=slip_steps,
                start_delay_s=start_delay_s,
                start_delay_steps=_steps(config, start_delay_s),
            )
        )
    layer_initial_angle = rng.uniform(*config.initial_angle_range)
    impulse_prob = rng.uniform(*config.impulse_prob_range)
    impulse_active = rng.random() < impulse_prob
    impulse_step = rng.randrange(config.max_control_steps)
    direction = [rng.uniform(-1.0, 1.0) for _ in range(3)]
    norm = math.sqrt(sum(component * component for component in direction))
    if norm < 1e-9:
        direction = [1.0, 0.0, 0.0]
    else:
        direction = [component / norm for component in direction]
    magnitude = rng.uniform(*config.impulse_magnitude_range)
    impulse_force = tuple(component * magnitude for component in direction)
    return EpisodeRandomization(
        seed=seed,
        agents=(agents[0], agents[1]),
        layer_initial_angle=layer_initial_angle,
        impulse_active=impulse_active,
        impulse_step=impulse_step,
        impulse_force=impulse_force,
    )


# --- success / failure / reward -----------------------------------------------------------


def success(
    angle: float,
    drift_pos: float,
    drift_rot: float,
    *,
    goal_angle: float,
    angle_tol: float,
    pos_tol: float,
    rot_tol: float,
    lin_speed: float = 0.0,
    ang_speed: float = 0.0,
    lin_speed_tol: float = math.inf,
    ang_speed_tol: float = math.inf,
) -> bool:
    """Whether the team has succeeded on this control step.

    The design's conjunction: the layer is at the goal angle within `angle_tol`, the core
    has drifted at most `pos_tol` / `rot_tol` from its initial pose, and the cube is at
    rest (speeds within `lin_speed_tol` / `ang_speed_tol`). The speed arguments default
    to "not checked" (zero speed, infinite tolerance), so a caller that only has the
    position facts can omit them.

    Args:
        angle: Current layer hinge angle (rad).
        drift_pos: Core position drift from the initial pose (m).
        drift_rot: Core rotation drift from the initial pose (rad).
        goal_angle: Target layer hinge angle (rad).
        angle_tol: Angle tolerance (rad).
        pos_tol: Position-drift tolerance (m).
        rot_tol: Rotation-drift tolerance (rad).
        lin_speed: Core linear speed (m/s).
        ang_speed: Combined core + layer angular speed (rad/s).
        lin_speed_tol: Linear-speed tolerance (m/s).
        ang_speed_tol: Angular-speed tolerance (rad/s).

    Returns:
        True only if every condition holds.
    """
    return (
        abs(angle - goal_angle) <= angle_tol
        and drift_pos <= pos_tol
        and drift_rot <= rot_tol
        and lin_speed <= lin_speed_tol
        and ang_speed <= ang_speed_tol
    )


def failure(drift_pos: float, drift_rot: float, *, pos_drift: float, rot_drift: float) -> bool:
    """Whether the episode must stop as a failure: the core drifted past the bounds.

    Args:
        drift_pos: Core position drift from the initial pose (m).
        drift_rot: Core rotation drift from the initial pose (rad).
        pos_drift: Position failure bound (m).
        rot_drift: Rotation failure bound (rad).

    Returns:
        True if either bound is exceeded.
    """
    return drift_pos > pos_drift or drift_rot > rot_drift


def team_reward(*, succeeded: bool, success_bonus: float, step_penalty: float) -> float:
    """The shared (non-additive) team reward of one control step.

    The reward is the same for both agents: the success bonus on the step that first
    satisfies the success condition, minus the per-step time penalty. It is the
    cooperative signal the trainer discounts per control step.

    Args:
        succeeded: Whether the step satisfied `success`.
        success_bonus: Bonus on a successful step (>= 0).
        step_penalty: Per-step time penalty (>= 0, below the bonus).

    Returns:
        `success_bonus - step_penalty` on a successful step, `-step_penalty` otherwise.
    """
    return (success_bonus if succeeded else 0.0) - step_penalty
