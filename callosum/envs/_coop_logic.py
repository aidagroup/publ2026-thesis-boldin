"""Pure logic of the two-agent reduced coop suite: observation layout, partner
visibility, episode randomisation, and the success/failure/reward rules.

The suite (`PLAN.md` "Two-robot task suite") is a set of simple tasks in which two
decentralised agents cooperate on one shared object: `CartPoleCoop-v0` (two one-sided
pushers balance a shared cart-pole and drive it to a target) and `PendulumPump-v0` (two
one-sided actuators pump a pendulum to a target amplitude and hold it). Both share this
module and the harness in `callosum.envs.coop`.

Deliberately has zero mujoco/torch dependency (the same macOS/CI rule as
`callosum.envs._wrench_logic` and `callosum.envs._partner_obs`): the simulator lives in
`callosum.envs.coop`, which CI cannot import, while everything that can be decided
without the simulator lives here and is unit-tested locally (`tests/test_coop_logic.py`).

Observation of ONE agent, in field order (`obs_fields`):

* **own action** (`spec.action_dim`): the agent's most recent APPLIED normalized command
  (after its latency queue and any start-delay suppression) -- what its actuator is doing
  right now, the usual "previous action" input of a policy.
* **own private parameters** (3): its action latency (s), gain and authority scale. The
  agent knows its own hardware, never the partner's -- that asymmetry is the private
  randomness of the design.
* **own privileged state** (1): its start delay (s), known to itself only.
* **shared state** (`spec.shared_fields`): the whole observable physics of the task
  (cart position/velocity and pole angle/velocity for the cart-pole; hinge angle and
  angular velocity for the pendulum). There are no contacts in these reduced models.
* **goal and time** (`spec.goal_fields` + `time_fraction`): the task target(s) and the
  fraction of the episode budget already used.
* **partner block** (`spec.action_dim * 2 + 4`), only under `partner_mode="oracle"`: the
  partner's APPLIED and COMMANDED actions, its private parameters and its start delay --
  the privileged upper bound of the `none`/`oracle` ablation. The anticipated partner
  COMMAND matters because the per-agent action latency (0.1-0.3 s) makes the partner's
  future effect unrecoverable from the shared state -- the information `predicted`
  (Bi-JEPA) has to supply.

The env does not assemble the vector itself: it collects one flat `{field: value}` dict of
role-tagged facts and `build_observation(values, agent, spec, partner_mode)` selects and
flattens the fields of `obs_fields(spec, partner_mode)`. Partner keys are always present in
`values`; the mode alone decides whether they reach the vector, so the "none" leakage rule
is concentrated in one pure function and tested bit-for-bit.

The action path is testable here too: `delayed_command` is the per-agent latency queue (a
command becomes the applied one exactly `latency_steps` steps later) and
`in_start_delay` decides whether the command is currently suppressed. `one_sided_allocation`
is the pure actuator split shared by the scripted oracles: the `left` agent can only command
positive force/torque, the `right` agent only negative, so neither can move the shared
object in both directions alone.
"""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

AGENT_KEYS = ("left", "right")
"""The two agent keys, in the env's action-/observation-space order.

The convention is physical throughout the suite: the `left` agent can only command a
POSITIVE physical quantity (force `+x` / positive hinge torque), the `right` agent only a
NEGATIVE one. A task where the agents are not one-sided would need different semantics.
"""

PARTNER_MODES = ("none", "oracle")
"""Partner-visibility modes available for the suite ("predicted" is a policy-input mode of
the trainer, not an env mode: the env stays at "none" and the predicted latent is appended
by the trainer)."""

TIME_FIELD = "time_fraction"
"""Name of the shared time-budget field (fraction of `max_control_steps` already used)."""

_PRIVATE_PARAM_NAMES = ("latency", "gain", "limit_scale")
"""Values-dict suffixes of the agent's private parameters, in `owner param` order."""


@dataclass(frozen=True)
class CoopTaskSpec:
    """Static observation layout of one coop task.

    The env inherits all of its observation-assembly logic from the spec, so a new task
    only declares its action width, its shared-state block and its goal block; the own and
    partner blocks are shared by the whole suite.

    Attributes:
        name: Task name (`"cartpole"` / `"pendulum"`), matching the `--task` value.
        action_dim: Width of ONE agent's continuous action (1 for both suite tasks).
        shared_fields: Names of the shared-state block, in `_shared_state_values` order.
        goal_fields: Names of the task-goal block, in `_goal_values` order (the shared
            `time_fraction` field is appended automatically by `obs_fields`).
    """

    name: str
    action_dim: int
    shared_fields: tuple[str, ...]
    goal_fields: tuple[str, ...]


CARTPOLE_SPEC = CoopTaskSpec(
    name="cartpole",
    action_dim=1,
    shared_fields=("cart_x", "cart_vx", "pole_angle", "pole_angvel"),
    goal_fields=("target_x",),
)
"""CartPoleCoop-v0: cart on a rail and a pole on a hinge, two one-sided pushers."""

PENDULUM_SPEC = CoopTaskSpec(
    name="pendulum",
    action_dim=1,
    shared_fields=("angle", "angvel"),
    goal_fields=("target_amplitude", "hold_cycles"),
)
"""PendulumPump-v0: pendulum on a fixed hinge, two one-sided torque actuators."""

TASK_SPECS: dict[str, CoopTaskSpec] = {spec.name: spec for spec in (CARTPOLE_SPEC, PENDULUM_SPEC)}
"""Registry of the suite's task specs by task name."""


def validate_task(task: str) -> None:
    """Raise ValueError unless `task` is a registered coop task."""
    if task not in TASK_SPECS:
        raise ValueError(f"task must be one of {tuple(TASK_SPECS)}, got {task!r}")


def task_spec(task: str) -> CoopTaskSpec:
    """The observation layout of `task` (raises ValueError for an unknown task)."""
    validate_task(task)
    return TASK_SPECS[task]


def task_obs_size(task: str, partner_mode: str) -> int:
    """Width of one agent's flat observation of `task` under `partner_mode`."""
    return obs_size(task_spec(task), partner_mode)


def validate_partner_mode(mode: str) -> None:
    """Raise ValueError if `mode` isn't one of `PARTNER_MODES`."""
    if mode not in PARTNER_MODES:
        raise ValueError(f"partner_mode must be one of {PARTNER_MODES}, got {mode!r}")


def partner_of(agent: str) -> str:
    """The other agent key (`"left"` <-> `"right"`)."""
    if agent not in AGENT_KEYS:
        raise ValueError(f"agent must be one of {AGENT_KEYS}, got {agent!r}")
    return AGENT_KEYS[1 - AGENT_KEYS.index(agent)]


# --- observation layout ------------------------------------------------------------------


def action_value_fields(agent: str, spec: CoopTaskSpec) -> tuple[str, ...]:
    """Values-dict keys of `agent`'s APPLIED-action block (`left_action_0`, ...)."""
    return tuple(f"{agent}_action_{index}" for index in range(spec.action_dim))


def command_value_fields(agent: str, spec: CoopTaskSpec) -> tuple[str, ...]:
    """Values-dict keys of `agent`'s COMMANDED (pre-latency) action block."""
    return tuple(f"{agent}_command_{index}" for index in range(spec.action_dim))


def param_value_fields(agent: str) -> tuple[str, ...]:
    """Values-dict keys of `agent`'s private parameters (`left_latency`, ...)."""
    return tuple(f"{agent}_{name}" for name in _PRIVATE_PARAM_NAMES)


def privileged_value_fields(agent: str) -> tuple[str, ...]:
    """Values-dict keys of `agent`'s start delay (its own privileged state)."""
    return (f"{agent}_start_delay",)


def partner_fields(spec: CoopTaskSpec, mode: str) -> tuple[str, ...]:
    """The partner block that `mode` makes visible (`()` under "none")."""
    validate_partner_mode(mode)
    if mode == "none":
        return ()
    dim = spec.action_dim
    return (
        tuple(f"partner_action_{index}" for index in range(dim))
        + tuple(f"partner_command_{index}" for index in range(dim))
        + ("partner_latency", "partner_gain", "partner_limit_scale")
        + ("partner_start_delay",)
    )


def obs_fields(spec: CoopTaskSpec, mode: str) -> tuple[str, ...]:
    """Full ordered field names of one agent's observation under `mode`."""
    own = (
        tuple(f"own_action_{index}" for index in range(spec.action_dim))
        + tuple(f"own_{name}" for name in _PRIVATE_PARAM_NAMES)
        + ("own_start_delay",)
    )
    return own + spec.shared_fields + spec.goal_fields + (TIME_FIELD,) + partner_fields(spec, mode)


def obs_size(spec: CoopTaskSpec, mode: str) -> int:
    """Width of one agent's flat observation vector under `mode`."""
    return len(obs_fields(spec, mode))


def _observation_field_value(
    values: Mapping[str, float], agent: str, partner_mode: str, field_name: str
) -> float:
    """One observation field's value from the role-tagged `values` dict.

    The observation layout names are mapped onto the values schema by prefix:
    `own_action_<i>` -> `"<agent>_action_<i>"`, `own_<param>` -> `"<agent>_<param>"`,
    `partner_*` -> the same keys for the partner role, and every other field (shared state,
    goal, time) is looked up verbatim. Mapping here keeps the vector order tied to
    `obs_fields` automatically, so the layout constants cannot silently drift apart.

    Args:
        values: Role-tagged facts collected by the env (see `build_observation`).
        agent: `"left"` or `"right"` -- whose observation is being built.
        partner_mode: `"none"` or `"oracle"` (only used for the error message).
        field_name: One name of `obs_fields(spec, partner_mode)`.

    Returns:
        The field's value as a float.

    Raises:
        ValueError: if `values` is missing the key the field maps to.
    """
    if field_name.startswith("own_action_"):
        key = f"{agent}_action_{field_name.removeprefix('own_action_')}"
    elif field_name.startswith("partner_action_"):
        key = f"{partner_of(agent)}_action_{field_name.removeprefix('partner_action_')}"
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


def build_observation(
    values: Mapping[str, float], agent: str, spec: CoopTaskSpec, partner_mode: str
) -> np.ndarray:
    """Build one agent's flat observation vector from the env's role-tagged `values`.

    Pure logic (no mujoco): the env collects the facts once per step and calls this for
    each agent, so the partner-visibility rule is decided here, not in the simulator code.
    Under `partner_mode="none"` every partner key of `values` is ignored, so changing them
    cannot change the returned vector -- the leakage sentinel pins that bit-for-bit.

    Args:
        values: Role-tagged facts: `{agent}_action_{i}`, `{agent}_command_{i}` and
            `{agent}_{param}` for both agents, the task's shared-state fields, its goal
            fields and `time_fraction`.
        agent: `"left"` or `"right"` -- whose observation this is.
        spec: The task's observation layout.
        partner_mode: `"none"` or `"oracle"`; selects the partner block.

    Returns:
        `np.float32` vector of width `obs_size(spec, partner_mode)`, in `obs_fields` order.
    """
    if agent not in AGENT_KEYS:
        raise ValueError(f"agent must be one of {AGENT_KEYS}, got {agent!r}")
    validate_partner_mode(partner_mode)
    vector = [
        _observation_field_value(values, agent, partner_mode, field_name)
        for field_name in obs_fields(spec, partner_mode)
    ]
    return np.asarray(vector, dtype=np.float32)


# --- episode randomisation ---------------------------------------------------------------


class RandomizationRanges(Protocol):
    """The randomisation ranges `sample_episode_randomization` reads from a task config.

    Declared as a protocol so the sampler stays pure and independent of the config module
    (`callosum.configs.coop_tasks.CoopRandomization` is the concrete implementation).
    """

    latency_range: tuple[float, float]
    gain_range: tuple[float, float]
    limit_scale_range: tuple[float, float]
    start_delay_range: tuple[float, float]
    initial_value_range: tuple[float, float]
    damping_range: tuple[float, float]
    wind_prob_range: tuple[float, float]
    wind_magnitude_range: tuple[float, float]


@dataclass(frozen=True)
class AgentRandomization:
    """The sampled private randomness of ONE agent in one episode.

    The first fields are the agent's private hardware parameters (also in its own
    observation); `start_delay_*` is its own privileged start-up state. The partner sees
    none of this unless the env runs under `partner_mode="oracle"`.
    """

    latency_s: float
    """Action latency (s) of this agent: a command is applied `latency_steps` steps later."""
    latency_steps: int
    """`latency_s` rounded to whole control steps."""
    gain: float
    """Command-gain multiplier applied to every command of this agent."""
    limit_scale: float
    """Multiplier on this agent's config authority (its private "torque limit")."""
    start_delay_s: float
    """Delay (s) before the agent's first command takes effect."""
    start_delay_steps: int
    """`start_delay_s` rounded to whole control steps."""


@dataclass(frozen=True)
class EpisodeRandomization:
    """The sampled private randomness of one episode (left record first).

    Deterministic in the episode seed (`sample_episode_randomization`); the env stores it
    for the episode and the scripted oracle may read its own agents' latency from it,
    exactly like a policy reads its own private parameters from its observation.
    """

    seed: int
    """Episode seed the record was sampled from."""
    agents: tuple[AgentRandomization, AgentRandomization]
    """Left record, then right record (the `AGENT_KEYS` order)."""
    initial_value: float
    """Initial task state: the initial pole angle (cart-pole) or the initial hinge angle
    (pendulum), inside the config's `initial_value_range`."""
    damping: float
    """Sampled joint damping (Nms/rad) written into the model on every reset; its meaning
    is per task (pole hinge for the cart-pole, pivot hinge for the pendulum). It is hidden
    from both agents' observations (a partner-private physical parameter)."""
    wind_active: bool
    """Whether the one shared perturbation happens in this episode."""
    wind_step: int
    """Control step of the perturbation (meaningless unless `wind_active`)."""
    wind_force: float
    """Signed magnitude of the one-step perturbation (N on the cart, Nm on the pendulum);
    its sign is sampled independently, so the two agents cannot anticipate the direction."""


def _steps(control_dt: float, seconds: float) -> int:
    """Convert seconds to whole control steps (rounded, never negative)."""
    if control_dt <= 0:
        raise ValueError(f"control_dt must be > 0, got {control_dt}")
    return max(0, round(seconds / control_dt))


def sample_episode_randomization(
    ranges: RandomizationRanges,
    *,
    control_dt: float,
    max_control_steps: int,
    seed: int,
) -> EpisodeRandomization:
    """Deterministically sample one episode's private randomness from `seed`.

    Uses stdlib `random.Random` (no torch/numpy; the sampler must run on macOS and inside
    the env's reset). The draw order is fixed so a seed fully determines the episode:
    per agent (left, then right) -- latency, gain, authority scale, start delay; then the
    episode-wide initial task state, joint damping, perturbation probability, perturbation
    Bernoulli, perturbation step, perturbation sign and magnitude.

    Args:
        ranges: The task's randomisation ranges (`CoopRandomization`).
        control_dt: Seconds per control step (latency/start-delay conversion).
        max_control_steps: Episode budget in control steps (perturbation step draw).
        seed: Episode seed.

    Returns:
        The immutable record; every drawn value is inside its range.
    """
    if max_control_steps < 1:
        raise ValueError(f"max_control_steps must be >= 1, got {max_control_steps}")
    rng = random.Random(int(seed))
    agents: list[AgentRandomization] = []
    for _ in AGENT_KEYS:
        latency_s = rng.uniform(*ranges.latency_range)
        gain = rng.uniform(*ranges.gain_range)
        limit_scale = rng.uniform(*ranges.limit_scale_range)
        start_delay_s = rng.uniform(*ranges.start_delay_range)
        agents.append(
            AgentRandomization(
                latency_s=latency_s,
                latency_steps=_steps(control_dt, latency_s),
                gain=gain,
                limit_scale=limit_scale,
                start_delay_s=start_delay_s,
                start_delay_steps=_steps(control_dt, start_delay_s),
            )
        )
    initial_value = rng.uniform(*ranges.initial_value_range)
    damping = rng.uniform(*ranges.damping_range)
    wind_prob = rng.uniform(*ranges.wind_prob_range)
    wind_active = rng.random() < wind_prob
    wind_step = rng.randrange(max_control_steps)
    wind_sign = 1.0 if rng.random() < 0.5 else -1.0
    wind_magnitude = rng.uniform(*ranges.wind_magnitude_range)
    return EpisodeRandomization(
        seed=int(seed),
        agents=(agents[0], agents[1]),
        initial_value=initial_value,
        damping=damping,
        wind_active=wind_active,
        wind_step=wind_step,
        wind_force=wind_sign * wind_magnitude,
    )


# --- action path (latency queue and start delay) -----------------------------------------


def delayed_command(queue: deque[Sequence[float]], command: Sequence[float]) -> Sequence[float]:
    """Push this step's `command` through an agent's latency queue, return the applied one.

    The env initialises the queue with `latency_steps` zero commands at reset. With
    `latency_steps == 0` the queue is empty before the call, so the command takes effect on
    the same step; with `latency_steps == 1` the first call returns the zero command
    enqueued at reset and the command takes effect one step later, and so on. Mutating the
    queue in place keeps the helper dependency-free (it works for tuples, lists and numpy
    arrays alike), so the latency behaviour is unit-testable without MuJoCo.

    Args:
        queue: The agent's in-flight commands, at most `latency_steps` long.
        command: The normalized command of this step.

    Returns:
        The command that takes effect on this step (still a sequence of floats).
    """
    queue.append(command)
    return queue.popleft()


def in_start_delay(agent: AgentRandomization, step: int) -> bool:
    """Whether the agent's command is suppressed on `step` by its start delay.

    The coop suite has no slip events (its private randomness is the latency, the
    gain/authority spread, the start delay and the task's hidden physical parameter); the
    single suppression cause is therefore the start-up delay.

    Args:
        agent: The agent's episode randomisation record.
        step: Zero-based control-step index of the episode.

    Returns:
        True while `step < agent.start_delay_steps`.
    """
    return step < agent.start_delay_steps


def wind_active_on_step(randomization: EpisodeRandomization, step: int) -> bool:
    """Whether the episode's one shared perturbation is applied on `step`."""
    return randomization.wind_active and randomization.wind_step == step


# --- success / failure / reward -----------------------------------------------------------


def one_sided_allocation(
    desired: float, positive_limit: float, negative_limit: float
) -> tuple[float, float]:
    """Split a desired signed physical command between a +only and a -only actuator.

    The suite's actuators are one-sided (`left` can only command a positive force/torque,
    `right` only a negative one), so a desired signed command can always be served by at
    most ONE of them. The helper returns the two non-negative magnitudes, clipped to each
    actuator's effective limit -- the shared rule of the scripted oracles and the pure
    counterpart of `CoopTaskEnv`'s actuation mapping.

    Args:
        desired: The desired signed physical command.
        positive_limit: Maximum magnitude the `+` actuator can deliver (>= 0).
        negative_limit: Maximum magnitude the `-` actuator can deliver (>= 0).

    Returns:
        `(positive_magnitude, negative_magnitude)`; exactly one is non-zero when `desired`
        is non-zero and the corresponding limit is positive.
    """
    if positive_limit < 0 or negative_limit < 0:
        raise ValueError(f"actuator limits must be >= 0, got ({positive_limit}, {negative_limit})")
    if desired > 0:
        return min(desired, positive_limit), 0.0
    if desired < 0:
        return 0.0, min(-desired, negative_limit)
    return 0.0, 0.0


def cartpole_success(
    *,
    cart_x: float,
    cart_vx: float,
    pole_angle: float,
    pole_angvel: float,
    target_x: float,
    pos_tol: float,
    angle_tol: float,
    lin_speed_tol: float,
    ang_speed_tol: float,
) -> bool:
    """Whether the team succeeded on this control step (the cart-pole conjunction).

    The cart is at the target position within `pos_tol`, the pole is upright within
    `angle_tol`, and the cart/pole speeds are within tolerance (the "arrived and settled"
    condition the design asks for).

    Args:
        cart_x: Current cart position (m).
        cart_vx: Current cart velocity (m/s).
        pole_angle: Pole angle from upright (rad).
        pole_angvel: Pole angular velocity (rad/s).
        target_x: Goal cart position (m).
        pos_tol: Position tolerance (m).
        angle_tol: Pole-angle tolerance (rad).
        lin_speed_tol: Cart linear-speed tolerance (m/s).
        ang_speed_tol: Pole angular-speed tolerance (rad/s).

    Returns:
        True only if every condition holds.
    """
    return (
        abs(cart_x - target_x) <= pos_tol
        and abs(pole_angle) <= angle_tol
        and abs(cart_vx) <= lin_speed_tol
        and abs(pole_angvel) <= ang_speed_tol
    )


def cartpole_failure(pole_angle: float, *, angle_fail: float) -> bool:
    """Whether the pole has fallen past the failure angle (the episode must stop)."""
    return abs(pole_angle) > angle_fail


def pendulum_success(*, completed_cycles: int, hold_cycles: int) -> bool:
    """Whether the team succeeded: enough completed half-cycles reached the target amplitude.

    A half-cycle is completed at every zero crossing of the hinge angle; the env counts it
    as successful when the peak `|angle|` since the previous crossing reached the target
    amplitude. The goal is therefore "pump to the target and HOLD it for `hold_cycles`
    completed half-cycles", not a single lucky peak.

    Args:
        completed_cycles: Number of completed half-cycles whose peak reached the target.
        hold_cycles: Required number of such half-cycles.

    Returns:
        `completed_cycles >= hold_cycles`.
    """
    if hold_cycles < 1:
        raise ValueError(f"hold_cycles must be >= 1, got {hold_cycles}")
    return completed_cycles >= hold_cycles


def pendulum_failure(angle: float, *, angle_fail: float) -> bool:
    """Whether the pendulum has swung past the failure angle (a near-flip)."""
    return abs(angle) > angle_fail


def cartpole_potential(
    *,
    cart_x: float,
    pole_angle: float,
    target_x: float,
    pos_scale: float,
    angle_scale: float,
    pos_coef: float,
    angle_coef: float,
) -> float:
    """Bounded potential `Φ(s)` of the cart-pole (negative normalised penalties).

    `-pos_coef * clip(|x - target| / pos_scale, 0, 1) - angle_coef * clip(|θ| / angle_scale,
    0, 1)`. Used with :func:`potential_shaping`, so the shaping telescopes over the episode
    and cannot dominate the terminal success bonus / failure penalty; it gives the shared
    reward a gradient toward the target before the sparse success is reached, and keeps the
    balancing role learnable.

    Args:
        cart_x: Current cart position (m).
        pole_angle: Pole angle from upright (rad).
        target_x: Goal cart position (m).
        pos_scale: Position error that saturates the position term (m).
        angle_scale: Pole angle that saturates the posture term (rad).
        pos_coef: Weight of the position term (>= 0).
        angle_coef: Weight of the posture term (>= 0).

    Returns:
        The potential value, bounded by `-(pos_coef + angle_coef)` from below and 0 above.
    """
    if pos_scale <= 0 or angle_scale <= 0:
        raise ValueError(f"potential scales must be > 0, got ({pos_scale}, {angle_scale})")
    if pos_coef < 0 or angle_coef < 0:
        raise ValueError(f"potential coefficients must be >= 0, got ({pos_coef}, {angle_coef})")
    pos_excess = min(abs(cart_x - target_x) / pos_scale, 1.0)
    angle_excess = min(abs(pole_angle) / angle_scale, 1.0)
    return -pos_coef * pos_excess - angle_coef * angle_excess


def pendulum_potential(*, peak_abs_angle: float, target_amplitude: float, coef: float) -> float:
    """Bounded potential `Φ(s)` of the pendulum: `coef * clip(peak / target, 0, 1)`.

    The peak amplitude since the last zero crossing is the env's amplitude estimate, so the
    potential rewards pumping the current half-cycle further; it resets at every crossing
    (a valid bounded potential, the reset is not an issue for the telescoping property).

    Args:
        peak_abs_angle: Peak `|angle|` since the last zero crossing (rad).
        target_amplitude: Target amplitude (rad), must be > 0.
        coef: Weight of the potential (>= 0).

    Returns:
        A value in `[0, coef]`.
    """
    if target_amplitude <= 0:
        raise ValueError(f"target_amplitude must be > 0, got {target_amplitude}")
    if coef < 0:
        raise ValueError(f"coef must be >= 0, got {coef}")
    return coef * min(max(peak_abs_angle, 0.0) / target_amplitude, 1.0)


def potential_shaping(*, potential_now: float, potential_prev: float) -> float:
    """Potential-based shaping reward of one step: `Φ(s_t) - Φ(s_{t-1})`.

    Telescopes over the episode to `Φ(final) - Φ(initial)`, so the shaping cannot dominate
    the terminal `success_bonus` / `failure_penalty` (the cube trainer's lesson).

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
        succeeded: Whether the step satisfied the task's success rule.
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


def amplitude_deg(angle: float) -> float:
    """Absolute hinge angle in degrees (the pendulum's "current amplitude" diagnostic)."""
    return math.degrees(abs(angle))
