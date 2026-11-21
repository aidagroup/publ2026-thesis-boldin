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
* **partner block** (11), only under `partner_mode="oracle"`: the partner's most recent
  APPLIED wrench, its private parameters and the remaining slip/start-delay state --
  the privileged upper bound of the ablation triple. `"predicted"` (Bi-JEPA, phase 3) will replace that block with a
  latent and is deliberately NOT part of this layout yet.

The env does not assemble the vector itself. It collects one flat `{field: value}` dict
of role-tagged facts -- `holder_wrench_fx`/`rotator_latency`/... for the two agents'
own blocks, the shared `core_*`/`layer_*` state and `goal_angle`/`time_fraction` -- and
`build_observation(values, agent, partner_mode)` selects and flattens the fields of
`obs_fields(mode)`. Partner keys are always present in `values`; the mode alone decides
whether they reach the vector, so the "none" leakage rule is concentrated in one pure
function and tested bit-for-bit (tests/test_wrench_logic.py). `obs_fields` fixes the
order and width, so no observation width is hardcoded in the env.

The action path is testable here too: `delayed_command` is the per-agent latency queue
(a command becomes the applied one exactly `latency_steps` steps later) and
`injection_active` decides whether a command is suppressed (start delay or slip window).
The episode randomisation includes the layer's sampled Coulomb friction, the private
rotator-side randomness the holder cannot observe.
"""

import math
import random
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

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
"""Names of the privileged partner-wrench block ("oracle" mode only) -- the APPLIED
(post-latency, post-injection) normalized wrench, i.e. what the partner's actuator is
doing right now."""

PARTNER_COMMAND_FIELDS = tuple(f"partner_command_{axis}" for axis in _WRENCH_AXES)
"""Names of the partner's COMMANDED (pre-latency) wrench ("oracle" mode only).

This is the anticipatory part of the upper bound: with per-episode action latency the
command reaches the physics several control steps later, and that future effect cannot
be recovered from the shared cube state -- a policy can only react to it after the fact.
The oracle sees it now, which is exactly the information a learned partner predictor
(Bi-JEPA) is supposed to supply in the `predicted` mode."""

OWN_PARAM_FIELDS = ("own_latency", "own_gain", "own_limit_scale")
"""Names of the agent's own private parameters (latency s, wrench gain, authority scale)."""

OWN_PRIVILEGED_FIELDS = ("own_start_delay", "own_slip_remaining")
"""The agent's own start delay (s) and the remaining steps of a slip event.

Both are things the hand knows about itself: it knows when it may start and whether its
actuator is slipping right now. The partner does NOT see them under `none`; under
`oracle` the partner's values are exposed (privileged upper bound)."""

PARTNER_PARAM_FIELDS = ("partner_latency", "partner_gain", "partner_limit_scale")
"""Names of the partner's private parameters ("oracle" mode only)."""

PARTNER_PRIVILEGED_FIELDS = ("partner_start_delay", "partner_slip_remaining")
"""The partner's start delay (s) and remaining slip steps ("oracle" mode only).

These are what make the oracle a true upper bound: the partner's actuator suppression is
not recoverable from the shared cube state, so without them `none` could infer the
partner's behaviour only from delayed dynamics (the observed oracle≈none collapse)."""

PARTNER_FIELDS = (
    PARTNER_WRENCH_FIELDS
    + PARTNER_COMMAND_FIELDS
    + PARTNER_PARAM_FIELDS
    + PARTNER_PRIVILEGED_FIELDS
)
"""The full partner block ("oracle" mode): applied wrench (6) + commanded wrench (6)
+ private parameters (3) + start delay and remaining slip steps (2)."""

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

_PRIVATE_PARAM_NAMES = ("latency", "gain", "limit_scale")
"""Values-dict suffixes of the agent's private parameters, in `OWN_PARAM_FIELDS` order."""


def partner_of(agent: str) -> str:
    """The other role key (`"holder"` <-> `"rotator"`).

    Args:
        agent: One of `AGENT_KEYS`.

    Returns:
        The opposite role key.
    """
    if agent not in AGENT_KEYS:
        raise ValueError(f"agent must be one of {AGENT_KEYS}, got {agent!r}")
    return AGENT_KEYS[1 - AGENT_KEYS.index(agent)]


def wrench_value_fields(agent: str) -> tuple[str, ...]:
    """Values-dict keys of `agent`'s applied-wrench block (`holder_wrench_fx`, ...)."""
    return tuple(f"{agent}_wrench_{axis}" for axis in _WRENCH_AXES)


def command_value_fields(agent: str) -> tuple[str, ...]:
    """Values-dict keys of `agent`'s commanded (pre-latency) wrench block."""
    return tuple(f"{agent}_command_{axis}" for axis in _WRENCH_AXES)


def param_value_fields(agent: str) -> tuple[str, ...]:
    """Values-dict keys of `agent`'s private parameters (`holder_latency`, ...)."""
    return tuple(f"{agent}_{name}" for name in _PRIVATE_PARAM_NAMES)


def privileged_value_fields(agent: str) -> tuple[str, ...]:
    """Values-dict keys of `agent`'s start delay and remaining slip steps."""
    return (f"{agent}_start_delay", f"{agent}_slip_remaining")


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
        + OWN_PRIVILEGED_FIELDS
        + CUBE_STATE_FIELDS
        + GOAL_FIELDS
        + partner_fields(mode)
    )


def obs_size(mode: str) -> int:
    """Width of one agent's flat observation vector under `mode`."""
    return len(obs_fields(mode))


def _observation_field_value(
    values: Mapping[str, float], agent: str, partner_mode: str, field_name: str
) -> float:
    """One observation field's value from the role-tagged `values` dict.

    The observation layout names are mapped onto the values schema by prefix:
    `own_wrench_<axis>` -> `"<agent>_wrench_<axis>"`, `own_<param>` -> `"<agent>_<param>"`,
    `partner_*` -> the same keys for the partner role, and every other field (shared cube
    state, goal) is looked up verbatim. Mapping here keeps the vector order tied to
    `obs_fields` automatically, so the layout constants cannot silently drift apart.

    Args:
        values: Role-tagged facts collected by the env (see `build_observation`).
        agent: `"holder"` or `"rotator"` -- whose observation is being built.
        partner_mode: `"none"` or `"oracle"` (only used for the error message).
        field_name: One name of `obs_fields(partner_mode)`.

    Returns:
        The field's value as a float.
    """
    if field_name.startswith("own_wrench_"):
        key = f"{agent}_wrench_{field_name.removeprefix('own_wrench_')}"
    elif field_name.startswith("partner_wrench_"):
        key = f"{partner_of(agent)}_wrench_{field_name.removeprefix('partner_wrench_')}"
    elif field_name.startswith("partner_command_"):
        key = f"{partner_of(agent)}_command_{field_name.removeprefix('partner_command_')}"
    elif field_name.startswith("own_"):
        key = f"{agent}_{field_name.removeprefix('own_')}"
    elif field_name.startswith("partner_"):
        key = f"{partner_of(agent)}_{field_name.removeprefix('partner_')}"
    else:
        key = field_name
    try:
        return float(values[key])
    except KeyError as exc:
        raise ValueError(
            f"values is missing {key!r}, required for the {agent!r} observation under "
            f"partner_mode={partner_mode!r}"
        ) from exc


def build_observation(values: Mapping[str, float], agent: str, partner_mode: str) -> np.ndarray:
    """Build one agent's flat observation vector from the env's role-tagged `values`.

    Pure logic (no mujoco): the env collects the facts once per step and calls this for
    each agent, so the partner-visibility rule is decided here, not in the simulator code.
    Under `partner_mode="none"` every partner key of `values` is ignored, so changing them
    cannot change the returned vector -- the leakage sentinel
    (`tests/test_wrench_logic.py`) pins that bit-for-bit.

    Args:
        values: Role-tagged facts: `{agent}_wrench_{axis}` and `{agent}_{param}` for both
            agents (`wrench_value_fields` / `param_value_fields`), the shared cube-state
            fields of `CUBE_STATE_FIELDS`, `goal_angle` and `time_fraction`.
        agent: `"holder"` or `"rotator"` -- whose observation this is.
        partner_mode: `"none"` or `"oracle"`; selects the partner block.

    Returns:
        `np.float32` vector of width `obs_size(partner_mode)`, in `obs_fields` order.
    """
    if agent not in AGENT_KEYS:
        raise ValueError(f"agent must be one of {AGENT_KEYS}, got {agent!r}")
    validate_partner_mode(partner_mode)
    vector = [
        _observation_field_value(values, agent, partner_mode, field_name)
        for field_name in obs_fields(partner_mode)
    ]
    return np.asarray(vector, dtype=np.float32)


# --- episode randomisation ---------------------------------------------------------------


@dataclass(frozen=True)
class AgentRandomization:
    """The sampled private randomness of ONE agent in one episode.

    The first three fields are the agent's private hardware parameters (and the only
    ones the agent sees about itself, under the "own private parameters" obs block);
    the slip and start-delay fields are private to the agent (exposed to the partner only
    under `oracle`, so the scripted oracle is a true upper bound).
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
    """Initial layer hinge angle (rad) inside `[0, hinge_limit]`."""
    layer_frictionloss: float
    """Sampled Coulomb friction loss (Nm) of the layer hinge for this episode. It is the
    layer-friction randomness the rotator cannot observe (hidden from both agents' obs);
    the env writes it into `model.dof_frictionloss` on reset."""
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
    layer initial angle, layer friction loss, impulse probability, impulse Bernoulli,
    impulse step, impulse direction (three draws) and impulse magnitude.

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
    layer_frictionloss = rng.uniform(*config.layer_frictionloss_range)
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
        layer_frictionloss=layer_frictionloss,
        impulse_active=impulse_active,
        impulse_step=impulse_step,
        impulse_force=impulse_force,
    )


# --- action path (latency queue and injections) ------------------------------------------


def delayed_command(queue: deque[Sequence[float]], command: Sequence[float]) -> Sequence[float]:
    """Push this step's `command` through an agent's latency queue, return the applied one.

    The env initialises the queue with `latency_steps` zero commands at reset. With
    `latency_steps == 0` the queue is empty before the call, so the command takes effect
    on the same step; with `latency_steps == 1` the first call returns the zero command
    enqueued at reset and the command takes effect one step later, and so on. Mutating the
    queue in place keeps the helper dependency-free (it works for tuples, lists and numpy
    arrays alike), so the latency behaviour is unit-testable without MuJoCo.

    Args:
        queue: The agent's in-flight commands, at most `latency_steps` long.
        command: The normalized wrench commanded on this step.

    Returns:
        The command that takes effect on this step (still a sequence of floats).
    """
    queue.append(command)
    return queue.popleft()


def injection_active(agent: AgentRandomization, step: int) -> bool:
    """Whether the agent's wrench is suppressed on `step` (start delay or slip window).

    Args:
        agent: The agent's episode randomisation record.
        step: Zero-based control-step index of the episode.

    Returns:
        True inside the start delay (`step < start_delay_steps`) or inside the agent's one
        slip window (`slip_step <= step < slip_step + slip_steps`); boundaries included.
    """
    in_start_delay = step < agent.start_delay_steps
    in_slip = agent.slip_active and agent.slip_step <= step < agent.slip_step + agent.slip_steps
    return in_start_delay or in_slip


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


def angle_progress(angle: float, previous_angle: float, goal_angle: float) -> float:
    """Normalised per-step progress toward the goal angle, in `[-1, 1]`.

    Positive when `|angle - goal|` shrank since the previous control step; divided by
    `goal_angle` so a full turn from 0 to the goal sums to ~1 over the episode. The env
    gates the shaping term on the drift tolerances, so a turn that moves the core gets no
    progress reward.

    Args:
        angle: Current layer hinge angle (rad).
        previous_angle: Layer hinge angle at the previous control step (rad).
        goal_angle: Target layer hinge angle (rad); non-positive disables the term.

    Returns:
        `(|previous - goal| - |angle - goal|) / goal`, or 0 for a non-positive goal.
    """
    if goal_angle <= 0:
        return 0.0
    return (abs(previous_angle - goal_angle) - abs(angle - goal_angle)) / goal_angle


def drift_excess(
    drift_pos: float,
    drift_rot: float,
    *,
    pos_tol: float,
    rot_tol: float,
    pos_drift: float,
    rot_drift: float,
) -> float:
    """Normalised core-drift excess over the success tolerances, clipped to `[0, 1]`.

    Each axis contributes `max(0, drift - tol) / (failure_bound - tol)`, so 1 means the
    axis is at its failure bound. Used as the shaping penalty that teaches the holder to
    stabilise the core before the terminal failure is reached.

    Args:
        drift_pos: Core position drift from the initial pose (m).
        drift_rot: Core rotation drift from the initial pose (rad).
        pos_tol: Success position-drift tolerance (m).
        rot_tol: Success rotation-drift tolerance (rad).
        pos_drift: Failure position-drift bound (m), must exceed `pos_tol`.
        rot_drift: Failure rotation-drift bound (rad), must exceed `rot_tol`.

    Returns:
        Sum of the two normalised excesses, clipped to [0, 1].
    """
    if pos_drift <= pos_tol or rot_drift <= rot_tol:
        raise ValueError(
            "failure bounds must exceed the success tolerances: "
            f"pos {pos_drift} > {pos_tol}, rot {rot_drift} > {rot_tol}"
        )
    excess_pos = max(0.0, (drift_pos - pos_tol) / (pos_drift - pos_tol))
    excess_rot = max(0.0, (drift_rot - rot_tol) / (rot_drift - rot_tol))
    return min(excess_pos + excess_rot, 1.0)


def potential(
    *,
    angle: float,
    angle_initial: float,
    goal_angle: float,
    drift_pos: float,
    drift_rot: float,
    angle_coef: float,
    drift_coef: float,
    pos_tol: float,
    rot_tol: float,
    pos_drift: float,
    rot_drift: float,
) -> float:
    """The bounded shaping potential `Φ(s)` of one control step.

    `angle_coef * normalised_progress - drift_coef * drift_excess`, where the progress
    `(|angle_initial - goal| - |angle - goal|) / goal` is clipped to `[0, 1]` and the
    drift excess comes from :func:`drift_excess`. The potential is used with
    :func:`potential_shaping`, so the shaping reward of an episode telescopes to
    `Φ(final) - Φ(initial)`: a successful turn pays at most `angle_coef` extra, and no
    path can farm unbounded shaping reward. Penalising the drift potential keeps the
    holder's stabilising role learnable, while the rest of the reward (success bonus /
    failure penalty) keeps the non-additive semantics.

    Args:
        angle: Current layer hinge angle (rad).
        angle_initial: Episode-initial layer angle (rad).
        goal_angle: Target layer angle (rad); non-positive disables the progress term.
        drift_pos: Core position drift (m).
        drift_rot: Core rotation drift (rad).
        angle_coef: Weight of the progress potential (>= 0).
        drift_coef: Weight of the drift penalty potential (>= 0).
        pos_tol: Success position-drift tolerance (m).
        rot_tol: Success rotation-drift tolerance (rad).
        pos_drift: Failure position-drift bound (m).
        rot_drift: Failure rotation-drift bound (rad).

    Returns:
        The potential value.
    """
    if angle_coef < 0 or drift_coef < 0:
        raise ValueError(f"shaping coefficients must be >= 0, got {(angle_coef, drift_coef)}")
    if goal_angle <= 0:
        progress = 0.0
    else:
        progress = (abs(angle_initial - goal_angle) - abs(angle - goal_angle)) / goal_angle
        progress = min(max(progress, 0.0), 1.0)
    return angle_coef * progress - drift_coef * drift_excess(
        drift_pos,
        drift_rot,
        pos_tol=pos_tol,
        rot_tol=rot_tol,
        pos_drift=pos_drift,
        rot_drift=rot_drift,
    )


def potential_shaping(*, potential_now: float, potential_prev: float) -> float:
    """Potential-based shaping reward of one step: `Φ(s_t) - Φ(s_{t-1})`.

    Telescopes over the episode to `Φ(final) - Φ(initial)`, so the shaping cannot
    dominate the terminal `success_bonus` / `failure_penalty` (the reason the naive
    per-step shaping gave the successful oracle a return of -3.7).

    Args:
        potential_now: `potential` of the current state.
        potential_prev: `potential` of the previous state (or the reset state).

    Returns:
        The shaping delta.
    """
    return potential_now - potential_prev


def team_reward(
    *,
    succeeded: bool,
    success_bonus: float,
    step_penalty: float,
    failed: bool = False,
    failure_penalty: float = 0.0,
) -> float:
    """The shared (non-additive) team reward of one control step.

    The reward is the same for both agents. Three cases, in priority order: the success
    step pays `success_bonus - step_penalty`; a step that crosses the failure bounds pays
    `failure_penalty` instead of the plain time cost; every other step pays
    `-step_penalty`. Success wins over failure if both flags are set (they cannot be at
    once with a valid config, because the success tolerances are strictly inside the
    failure bounds).

    Args:
        succeeded: Whether the step satisfied `success`.
        success_bonus: Bonus on a successful step (>= 0).
        step_penalty: Per-step time penalty (>= 0, below the bonus).
        failed: Whether the step crossed the failure bounds.
        failure_penalty: Reward on the failure step (< 0, no better than `-step_penalty`).

    Returns:
        `success_bonus - step_penalty` on success, `failure_penalty` on a failure step,
        `-step_penalty` otherwise.
    """
    if succeeded:
        return success_bonus - step_penalty
    if failed:
        return failure_penalty
    return -step_penalty
