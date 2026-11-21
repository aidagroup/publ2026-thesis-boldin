"""Configs of the reduced two-agent coop suite (`CartPoleCoop-v0`, `PendulumPump-v0`).

The suite (`PLAN.md` "Two-robot task suite") is a set of simple MuJoCo tasks in which two
decentralised agents cooperate on one shared object. This module holds every tunable the
task specs leave open: decision timing, the physical parameters the MuJoCo XML only carries
as defaults (gravity, joint limits, damping, armature), the per-agent authority, the
success/failure tolerances, the episode-randomisation ranges (the private randomness that
makes the partner only partially observable), the team reward, the optional shaping
weights and the fallback seed.

Like `callosum.experiments.cube_wrench.config` this module is pure stdlib on purpose: it must import
and validate on macOS/CI, where the `sandbox` extra (mujoco) is absent, so the suite's
logic and parameters are unit-testable without the simulator. The MuJoCo models are
`callosum/experiments/coop_tasks/assets/cartpole.xml` and `pendulum.xml`.

Config as truth for timing and DOF parameters: the env writes `dt` into
`model.opt.timestep`, `gravity` into `model.opt.gravity`, the joint limits into
`model.jnt_range` and the damping/armature into `model.dof_*`; each reset then replaces the
sampled joint damping with the episode's `EpisodeRandomization.damping`. Geometry and the
masses (cart/pole, bob) are deliberately XML-owned: MuJoCo's `mj_setConst` does not
recompute a body's inertia when `body_mass` changes, so writing the masses at runtime would
leave the model with an inconsistent inertia. The XML carries them and is the source of
truth for the masses/lengths; the config docstrings state that explicitly.

Physical scale note: the defaults are tuned so that the scripted oracle in
`callosum.experiments.coop_tasks.oracle` reaches success under the full randomisation ranges, while the
one-sided "only-left" baseline (the same oracle with the right agent absent) does not reach
the success condition -- the suite's non-additivity criterion. These are the
"точные масса/инерция/параметры шарнира и лимиты моментов" open question of PLAN.md to be
re-tuned on the GPU pod.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from callosum.experiments.coop_tasks.logic import (
    AGENT_KEYS,
    PARTNER_MODES,
    TASK_SPECS,
    obs_size,
    task_spec,
)

TASKS = ("cartpole", "pendulum")
"""Coop tasks implemented at this step (`toss` is the next agent's task)."""


def _validate_common(config: _CommonTaskConfig) -> None:
    """Validate the timing, seed, nested configs and the fail-fast invariant.

    Both task configs expose the same field groups, so their shared domain rules live in
    one function the two `__post_init__` methods call (the dataclasses stay frozen).

    Args:
        config: The task config being validated.

    Raises:
        ValueError: on an out-of-domain timing/gravity/seed value or a failure penalty
            that is not strictly worse than the whole-episode timeout return.
        TypeError: when the nested reward/randomisation block is not the expected type.
    """
    if not config.dt > 0:
        raise ValueError(f"dt must be > 0, got {config.dt}")
    if config.control_every < 1:
        raise ValueError(f"control_every must be >= 1, got {config.control_every}")
    if config.max_control_steps < 1:
        raise ValueError(f"max_control_steps must be >= 1, got {config.max_control_steps}")
    if not config.gravity > 0:
        raise ValueError(f"gravity must be > 0, got {config.gravity}")
    if config.seed < 0:
        raise ValueError(f"seed must be >= 0, got {config.seed}")
    if not isinstance(config.reward, CoopRewardConfig):
        raise TypeError(f"reward must be a CoopRewardConfig, got {type(config.reward)}")
    if not isinstance(config.randomization, CoopRandomization):
        raise TypeError(
            f"randomization must be a CoopRandomization, got {type(config.randomization)}"
        )
    # Failing fast must not be optimal: a whole-episode timeout has to be strictly better
    # than the failure penalty (the cube trainer's lesson).
    timeout_return = -config.reward.step_penalty * config.max_control_steps
    if config.reward.failure_penalty >= timeout_return:
        raise ValueError(
            "failure_penalty must be strictly worse than the whole-episode timeout return "
            f"(failure_penalty < -step_penalty * max_control_steps = {timeout_return}), got "
            f"{config.reward.failure_penalty}"
        )


class _CommonTaskConfig(Protocol):
    """The field groups both task configs expose (input of `_validate_common`)."""

    dt: float
    control_every: int
    max_control_steps: int
    gravity: float
    seed: int
    reward: CoopRewardConfig
    randomization: CoopRandomization


# --- shared dataclasses ------------------------------------------------------------------


def _check_range(
    name: str,
    value: tuple[float, float],
    *,
    non_negative: bool = True,
    positive_lo: bool = False,
) -> None:
    """Raise ValueError unless `value` is an ordered finite numeric pair in the domain.

    Args:
        name: Field name for the error message.
        value: The `(lo, hi)` pair to check.
        non_negative: Require `lo >= 0`.
        positive_lo: Require `lo > 0` (implies `non_negative`).
    """
    lo, hi = value
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"{name} must be finite, got {value}")
    if lo > hi:
        raise ValueError(f"{name} must have lo <= hi, got {value}")
    if positive_lo and not lo > 0:
        raise ValueError(f"{name} must have lo > 0, got {value}")
    if non_negative and not positive_lo and lo < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass(frozen=True)
class CoopRewardConfig:
    """The shared team reward of the suite (the same three cases as CubeWrench-v0).

    The reward is identical for both agents, so the coordination is a team problem: the
    success bonus is paid once, on the success step; a failure step is strictly worse than
    a whole-episode timeout (validated by the task configs), otherwise "fail fast" would be
    the optimal policy.
    """

    success_bonus: float = 2.0
    """Team reward granted on the step that first satisfies the success condition."""

    step_penalty: float = 0.001
    """Team reward subtracted on every control step (the shared time cost)."""

    failure_penalty: float = -2.0
    """Team reward on the step that first violates the failure condition -- it replaces
    the plain `-step_penalty`, so failing is punished rather than merely costing one step."""

    def __post_init__(self) -> None:
        if not 0 <= self.success_bonus:
            raise ValueError(f"success_bonus must be >= 0, got {self.success_bonus}")
        if not 0 <= self.step_penalty < self.success_bonus:
            raise ValueError(
                f"step_penalty must be in [0, success_bonus) = [0, {self.success_bonus}), "
                f"got {self.step_penalty}"
            )
        if not self.failure_penalty < 0:
            raise ValueError(f"failure_penalty must be < 0, got {self.failure_penalty}")
        if not self.failure_penalty <= -self.step_penalty:
            raise ValueError(
                "failure_penalty must be no better than a step penalty "
                f"(<= -step_penalty = {-self.step_penalty}), got {self.failure_penalty}"
            )


@dataclass(frozen=True)
class CoopRandomization:
    """The per-episode private randomisation ranges of one coop task.

    One shared dataclass for the whole suite: the field NAMES are task-independent (both
    tasks sample a latency, a gain, an authority scale, a start delay, an initial state, a
    joint damping and one optional perturbation), while their MEANING is per task and is
    documented on each task's config:

    * `initial_value_range` -- initial pole angle (cart-pole) / initial hinge angle
      (pendulum);
    * `damping_range` -- pole-joint damping (cart-pole) / pivot-joint damping (pendulum);
    * `wind_*` -- one signed force impulse on the cart (cart-pole); disabled for the
      pendulum, whose perturbation channel is not used at this step.
    """

    latency_range: tuple[float, float] = (0.1, 0.3)
    """Per-agent action latency (s): a command is applied this many control steps later.
    Sampled per agent; the agent knows its own latency (it is in its observation), the
    partner's is hidden."""

    gain_range: tuple[float, float] = (0.8, 1.2)
    """Per-agent command-gain multiplier: the applied command is
    `sign * max(normalized_action, 0) * authority * limit_scale * gain`."""

    limit_scale_range: tuple[float, float] = (0.8, 1.2)
    """Per-agent authority multiplier of the config maximum. Sampled per agent."""

    start_delay_range: tuple[float, float] = (0.0, 0.2)
    """Per-agent delay (s) before the agent's command takes effect for the first time."""

    initial_value_range: tuple[float, float] = (-0.05, 0.05)
    """Initial task state (rad): pole angle (cart-pole) / hinge angle (pendulum)."""

    damping_range: tuple[float, float] = (0.0, 0.02)
    """Sampled joint damping (Nms/rad) of the task's hinge, written into
    `model.dof_damping` on every reset. Hidden from both observations (a private physical
    parameter)."""

    wind_prob_range: tuple[float, float] = (0.0, 0.0)
    """Episode probability range of the one shared perturbation. `(0, 0)` disables it."""

    wind_magnitude_range: tuple[float, float] = (0.0, 0.0)
    """Magnitude range of the one-step perturbation (N on the cart / Nm on the hinge)."""

    def __post_init__(self) -> None:
        _check_range("latency_range", self.latency_range)
        _check_range("gain_range", self.gain_range, positive_lo=True)
        _check_range("limit_scale_range", self.limit_scale_range, positive_lo=True)
        _check_range("start_delay_range", self.start_delay_range)
        _check_range("damping_range", self.damping_range)
        _check_range("wind_prob_range", self.wind_prob_range)
        _check_range("wind_magnitude_range", self.wind_magnitude_range)
        if self.wind_prob_range[1] > 1:
            raise ValueError(f"wind_prob_range must be within [0, 1], got {self.wind_prob_range}")


@dataclass(frozen=True)
class CartPoleConfig:
    """All tunables of `CartPoleCoop-v0` (timing, physics, authority, tolerances, reward).

    The task: two one-sided pushers on the cart's rail (`left` only pushes `+x`, `right`
    only `-x`) must drive the cart to `target_x` while keeping the pole up -- the classic
    unstable shared-object stabilisation, where neither pusher can move the cart in both
    directions alone. The pole falls out after `failure_angle`; success is the conjunction
    "cart at target, pole upright, both speeds small".

    XML-owned physics (`callosum/experiments/coop_tasks/assets/cartpole.xml`): cart mass 2.0 kg, pole
    length 3.0 m (capsule, mass 0.2 kg). The config owns gravity, the rail limit, the
    joint damping/armature and the force authority (written into the model at construction
    / on every reset).
    """

    # --- decision timing -----------------------------------------------------------------
    dt: float = 0.005
    """MuJoCo physics timestep (s). The XML default is overridden from the config."""

    control_every: int = 8
    """Physics substeps per control step (0.04 s at the defaults)."""

    max_control_steps: int = 300
    """Episode budget in control steps (12 s at the defaults)."""

    # --- physical model (timing/dof parameters written into the model; mass in XML) --------
    gravity: float = 9.81
    """Gravity magnitude (m/s^2); written into `model.opt.gravity` (x and y stay 0)."""

    rail_limit: float = 1.4
    """Half-travel of the cart's slide joint (m), written into `model.jnt_range`."""

    cart_damping: float = 0.2
    """Viscous damping (Ns/m) of the cart's slide joint (written into `model.dof_damping`;
    it keeps the cart from sliding forever when the agents are idle)."""

    pole_armature: float = 0.01
    """Pole hinge armature (kg m^2) added to the pole's own inertia, so the hinge dynamics
    stay slow enough to control under the up-to-0.3 s action latency."""

    pole_damping: float = 0.0
    """Nominal pole-hinge damping (Nms/rad) written at construction; each reset replaces it
    with the sampled `EpisodeRandomization.damping` (`CoopRandomization.damping_range`)."""

    # --- per-agent force authority (action = sign * max(a, 0) * authority * gain * scale) --
    max_force: float = 30.0
    """Maximum push force (N) of EACH one-sided pusher on the cart."""

    # --- success / failure tolerances ------------------------------------------------------
    target_x: float = 0.5
    """Goal cart position (m), inside `(-rail_limit, rail_limit)`."""

    success_pos_tol: float = 0.05
    """Success tolerance (m) on `|cart_x - target_x|`."""

    success_angle_tol: float = 0.1
    """Success tolerance (rad) on the pole angle from upright."""

    success_lin_speed_tol: float = 0.1
    """Success tolerance (m/s) on the cart speed (the cart must have almost stopped)."""

    success_ang_speed_tol: float = 0.2
    """Success tolerance (rad/s) on the pole angular speed."""

    failure_angle: float = 0.5
    """Failure bound (rad) on the pole angle from upright (the pole has fallen)."""

    # --- reward / shaping ------------------------------------------------------------------
    reward: CoopRewardConfig = field(default_factory=CoopRewardConfig)
    """The shared team reward."""

    shaping_pos_coef: float = 0.5
    """Weight of the cart-position term of the potential-based shaping (0 disables)."""

    shaping_angle_coef: float = 0.5
    """Weight of the pole-posture term of the potential-based shaping (0 disables)."""

    # --- episode randomisation -------------------------------------------------------------
    randomization: CoopRandomization = field(
        default_factory=lambda: CoopRandomization(
            latency_range=(0.1, 0.3),
            gain_range=(0.8, 1.2),
            limit_scale_range=(0.8, 1.2),
            start_delay_range=(0.0, 0.2),
            initial_value_range=(-0.05, 0.05),
            damping_range=(0.0, 0.02),
            wind_prob_range=(0.0, 0.3),
            wind_magnitude_range=(1.0, 5.0),
        )
    )
    """The per-episode private randomness (see `CoopRandomization`)."""

    # --- fallback seed ----------------------------------------------------------------------
    seed: int = 0
    """Seed used when `reset(seed=None)`; each reset without an explicit seed advances it
    by one episode, so a fresh env does not replay the same randomisation."""

    # --- derived / identity ---------------------------------------------------------------
    task: str = "cartpole"
    """Task name (`--task` value); used to select the env and the observation layout."""

    @property
    def control_dt(self) -> float:
        """Seconds per control step (`dt * control_every`)."""
        return self.dt * self.control_every

    @property
    def max_episode_time(self) -> float:
        """Simulated seconds of the episode budget (`control_dt * max_control_steps`)."""
        return self.control_dt * self.max_control_steps

    @property
    def action_size(self) -> int:
        """Width of one agent's action (1: a normalized one-sided force)."""
        return task_spec(self.task).action_dim

    def agent_authority(self, agent_index: int) -> float:
        """The physical authority (N) of one agent; both pushers share `max_force`."""
        if not 0 <= agent_index < len(AGENT_KEYS):
            raise ValueError(f"agent_index must be in [0, {len(AGENT_KEYS)}), got {agent_index}")
        return self.max_force

    def __post_init__(self) -> None:
        _validate_common(self)
        if not self.rail_limit > 0:
            raise ValueError(f"rail_limit must be > 0, got {self.rail_limit}")
        for name in ("cart_damping", "pole_armature", "pole_damping"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not 0 < self.failure_angle < math.pi:
            raise ValueError(f"failure_angle must be in (0, pi), got {self.failure_angle}")
        if not 0 < self.success_angle_tol < self.failure_angle:
            raise ValueError(
                f"success_angle_tol must be in (0, failure_angle) = "
                f"(0, {self.failure_angle}), got {self.success_angle_tol}"
            )
        if not 0 < self.success_pos_tol < 2 * self.rail_limit:
            raise ValueError(
                "success_pos_tol must be in (0, 2 * rail_limit) = "
                f"(0, {2 * self.rail_limit}), got {self.success_pos_tol}"
            )
        for name in ("success_lin_speed_tol", "success_ang_speed_tol"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if not 0 < abs(self.target_x) < self.rail_limit:
            raise ValueError(
                f"abs(target_x) must be in (0, rail_limit) = (0, {self.rail_limit}), "
                f"got {self.target_x}"
            )
        if not self.max_force > 0:
            raise ValueError(f"max_force must be > 0, got {self.max_force}")
        for name in ("shaping_pos_coef", "shaping_angle_coef"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")


@dataclass(frozen=True)
class PendulumConfig:
    """All tunables of `PendulumPump-v0` (timing, physics, authority, tolerances, reward).

    The task: two one-sided torque actuators on the fixed hinge (`left` only a positive
    torque, `right` only a negative one) must pump the pendulum to `target_amplitude` and
    HOLD it for `hold_cycles` completed half-cycles. One actuator alone can only pump every
    other half-cycle, so its reachable amplitude saturates below the target -- the task's
    non-additivity, tuned through `randomization.damping_range`.

    XML-owned physics (`callosum/experiments/coop_tasks/assets/pendulum.xml`): bob mass 0.5 kg at 0.5 m
    from the pivot. The config owns gravity, the hinge limit, the damping/armature and the
    torque authority (written into the model at construction / on every reset).
    """

    # --- decision timing -----------------------------------------------------------------
    dt: float = 0.005
    """MuJoCo physics timestep (s)."""

    control_every: int = 8
    """Physics substeps per control step (0.04 s at the defaults)."""

    max_control_steps: int = 400
    """Episode budget in control steps (16 s at the defaults, ~11 pendulum periods)."""

    # --- physical model (timing/dof parameters written into the model; mass in XML) --------
    gravity: float = 9.81
    """Gravity magnitude (m/s^2); written into `model.opt.gravity`."""

    hinge_limit: float = 3.0
    """Hinge travel limit (rad) around the downward equilibrium, written into
    `model.jnt_range` as `[-hinge_limit, hinge_limit]`; it keeps the pendulum from spinning
    around and makes the 150-degree failure bound the real terminal condition."""

    hinge_armature: float = 0.001
    """Hinge armature (kg m^2) added to the bob's own inertia."""

    hinge_damping: float = 0.11
    """Nominal hinge damping (Nms/rad) written at construction; each reset replaces it with
    the sampled `EpisodeRandomization.damping` (`CoopRandomization.damping_range`)."""

    # --- per-agent torque authority ------------------------------------------------------
    max_torque: float = 0.65
    """Maximum magnitude (Nm) of EACH one-sided hinge-torque actuator."""

    # --- success / failure tolerances ------------------------------------------------------
    target_amplitude: float = math.radians(65.0)
    """Target amplitude (rad) of the hinge angle: the peak `|angle|` of a half-cycle.

    The spec's illustrative value is 60 degrees; 65 is used so the one-sided "only-left"
    baseline (whose reachable amplitude saturates around 30-59 degrees depending on the
    sampled damping) stays clearly below the target while the two-agent oracle reaches it."""

    hold_cycles: int = 2
    """Completed half-cycles (zero crossings) whose peak reached `target_amplitude` that the
    team must accumulate to succeed (2 half-cycles = one full oscillation at amplitude)."""

    failure_angle: float = math.radians(150.0)
    """Failure bound (rad) on `|angle|`: the pendulum is about to flip over."""

    hold_cycles_max: int = 12
    """Sanity cap of `hold_cycles` (the budget must be able to contain the holds)."""

    # --- reward / shaping ------------------------------------------------------------------
    reward: CoopRewardConfig = field(default_factory=CoopRewardConfig)
    """The shared team reward."""

    shaping_amplitude_coef: float = 0.5
    """Weight of the amplitude-progress potential-based shaping (0 disables)."""

    # --- episode randomisation -------------------------------------------------------------
    randomization: CoopRandomization = field(
        default_factory=lambda: CoopRandomization(
            latency_range=(0.075, 0.2),
            gain_range=(0.8, 1.2),
            limit_scale_range=(0.8, 1.2),
            start_delay_range=(0.0, 0.2),
            initial_value_range=(-0.15, 0.15),
            damping_range=(0.11, 0.15),
            wind_prob_range=(0.0, 0.0),
            wind_magnitude_range=(0.0, 0.0),
        )
    )
    """The per-episode private randomness (see `CoopRandomization`); for the pendulum the
    perturbation channel is disabled and the hidden physical parameter is the damping."""

    # --- fallback seed ----------------------------------------------------------------------
    seed: int = 0
    """Seed used when `reset(seed=None)` (advances by one episode per reset)."""

    # --- derived / identity ---------------------------------------------------------------
    task: str = "pendulum"
    """Task name (`--task` value)."""

    @property
    def control_dt(self) -> float:
        """Seconds per control step (`dt * control_every`)."""
        return self.dt * self.control_every

    @property
    def max_episode_time(self) -> float:
        """Simulated seconds of the episode budget (`control_dt * max_control_steps`)."""
        return self.control_dt * self.max_control_steps

    @property
    def action_size(self) -> int:
        """Width of one agent's action (1: a normalized one-sided hinge torque)."""
        return task_spec(self.task).action_dim

    def agent_authority(self, agent_index: int) -> float:
        """The physical authority (Nm) of one agent; both actuators share `max_torque`."""
        if not 0 <= agent_index < len(AGENT_KEYS):
            raise ValueError(f"agent_index must be in [0, {len(AGENT_KEYS)}), got {agent_index}")
        return self.max_torque

    def __post_init__(self) -> None:
        _validate_common(self)
        if not 0 < self.hinge_limit <= math.pi:
            raise ValueError(f"hinge_limit must be in (0, pi], got {self.hinge_limit}")
        if not 0 < self.target_amplitude < self.failure_angle < self.hinge_limit:
            raise ValueError(
                "tolerances must satisfy 0 < target_amplitude < failure_angle < "
                f"hinge_limit, got {self.target_amplitude}, {self.failure_angle}, "
                f"{self.hinge_limit}"
            )
        for name in ("hinge_armature", "hinge_damping"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not self.max_torque > 0:
            raise ValueError(f"max_torque must be > 0, got {self.max_torque}")
        if not 1 <= self.hold_cycles <= self.hold_cycles_max:
            raise ValueError(
                f"hold_cycles must be in [1, hold_cycles_max={self.hold_cycles_max}], got "
                f"{self.hold_cycles}"
            )
        # A half-cycle cannot last less than ~one control step, so `hold_cycles` completed
        # crossings plus the pumping time must fit inside the episode budget.
        if self.hold_cycles * 2 > self.max_control_steps:
            raise ValueError(
                "hold_cycles does not fit the episode budget: "
                f"hold_cycles={self.hold_cycles}, max_control_steps={self.max_control_steps}"
            )
        if self.shaping_amplitude_coef < 0:
            raise ValueError(
                f"shaping_amplitude_coef must be >= 0, got {self.shaping_amplitude_coef}"
            )


TASK_CONFIGS: dict[str, type[CartPoleConfig | PendulumConfig]] = {
    "cartpole": CartPoleConfig,
    "pendulum": PendulumConfig,
}
"""The config class of each suite task (both share the same public field groups)."""


def coop_config(task: str, **overrides: object) -> CartPoleConfig | PendulumConfig:
    """Build the config of a suite task.

    Args:
        task: One of `TASKS`.
        **overrides: Fields of the task's config dataclass to override.

    Returns:
        A validated `CartPoleConfig` or `PendulumConfig`.

    Raises:
        ValueError: if `task` is unknown.
    """
    if task not in TASK_CONFIGS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    return TASK_CONFIGS[task](**overrides)


def coop_obs_size(task: str, partner_mode: str) -> int:
    """Width of one agent's flat observation of `task` under `partner_mode`."""
    if partner_mode not in PARTNER_MODES:
        raise ValueError(f"partner_mode must be one of {PARTNER_MODES}, got {partner_mode!r}")
    return obs_size(task_spec(task), partner_mode)


def coop_action_size(task: str) -> int:
    """Width of one agent's action of `task` (1 for both suite tasks)."""
    return task_spec(task).action_dim


# Import-time invariant: the task specs and the config presets must agree, so a typo in
# `task` or a new task without a config fails immediately (pure, no simulator).
for _task in TASKS:
    _config = coop_config(_task)
    if _config.task != _task or _config.action_size != TASK_SPECS[_task].action_dim:
        raise RuntimeError(f"config/spec mismatch for task {_task!r}")  # pragma: no cover
