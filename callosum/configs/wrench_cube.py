"""Configuration of CubeWrench-v0: the reduced wrench-level cube of the thesis main line.

The reduced model (PLAN.md "Среда CubeWrench-v0") is a floating cube (`core`) with one
active layer on a limited hinge (`layer`); two decentralised agents act on it through a
raw 6D wrench each -- the holder stabilises the core, the rotator turns the layer to
90 degrees. This dataclass holds every tunable the spec leaves open: the decision
timing, the physical parameters the MuJoCo XML only carries as defaults, the per-agent
action authority, the success/failure tolerances, the episode-randomisation ranges (the
private randomness that makes the partner only partially observable), the team reward
and the fallback seed.

Pure stdlib on purpose (same rule as `callosum.envs._wrench_logic`): the config must
import and validate on macOS, where mujoco is absent, so the env's logic is unit-testable
without the simulator. The MuJoCo model is `callosum/envs/wrench_cube/assets/cube_wrench.xml`;
`WrenchCubeEnv` writes the fields below into `model.dof_*`/`model.opt.timestep` at
construction, so this dataclass -- not the XML -- is the single source of truth for them.

Physical scale note: the defaults are tuned so that the scripted oracle (holder PD +
rotator PD, both lead-compensated for their own action latency) reaches success under the
full randomisation ranges, while the rotator alone drifts the core past the failure bound
(the non-additivity the design needs). Details of the balance:
`core_ang_damping` / `core_lin_damping` are the reduced model's isotropic medium damping
(the free-floating cube has no contacts and no gravity, so without it the delayed wrench
control would be an undamped oscillator); `hinge_damping` sets both the layer's turn
speed and the reaction impulse the holder must absorb. All of these are the "точные
масса/инерция/параметры шарнира и лимиты моментов" open question of PLAN.md to be
re-tuned at gates G1/G2.
"""

import math
from dataclasses import dataclass


def _check_range(
    name: str,
    value: tuple[float, float],
    *,
    non_negative: bool = True,
    positive_lo: bool = False,
) -> None:
    """Raise ValueError unless `value` is an ordered numeric pair within the allowed domain.

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
class WrenchCubeConfig:
    """All tunables of CubeWrench-v0 (timing, physics, authority, tolerances, randomness).

    The field groups follow the environment spec: decision timing, the physical model,
    the two agents' wrench authority, the success/failure tolerances, the per-episode
    randomisation ranges, the team reward and the seed fallback.
    """

    # --- decision timing -----------------------------------------------------------------
    dt: float = 0.005
    """MuJoCo physics timestep (s). The XML default is the same; the env enforces this
    value on `model.opt.timestep`."""

    control_every: int = 8
    """Physics substeps per control step. One `env.step` applies a fixed wrench for
    `control_every` substeps (0.04 s at the defaults, inside the spec's 8-10 range)."""

    max_control_steps: int = 350
    """Episode budget in control steps (14 s at the defaults, inside the spec's 300-400
    range). A policy that has not succeeded by then is truncated."""

    # --- physical model (written into model.dof_*; the XML values are defaults only) ------
    core_lin_damping: float = 0.5
    """Linear viscous damping (Ns/m) on the core's free-joint translation DOFs. The cube
    floats (no gravity, no contacts), so this is the medium damping that makes the
    position loop well-posed; it is not a contact model."""

    core_ang_damping: float = 5.0e-4
    """Angular viscous damping (Nms/rad) on the core's free-joint rotation DOFs. It also
    bounds how far the rotator's reaction impulse can drift the core without the holder
    (with the defaults the core drifts ~1 rad, past `failure_rot_drift`)."""

    hinge_damping: float = 3.0e-4
    """Viscous damping (Nms/rad) of the layer hinge. Dominates the layer's turn speed
    and therefore the reaction impulse the holder must absorb: raising it needs more
    rotator torque and more holder authority."""

    hinge_frictionloss: float = 1.0e-6
    """Coulomb friction loss (Nm) of the layer hinge. Kept small so that the rotator's
    steady-state angle error (`frictionloss / rotator PD gain`) stays far below the
    success angle tolerance."""

    hinge_armature: float = 1.0e-4
    """Hinge armature (kg m^2), added to the thin layer's own inertia so the hinge
    dynamics are slow enough to control under the 0-0.3 s action latency."""

    # --- per-agent wrench authority (action = normalized wrench * authority) -------------
    holder_max_force: float = 0.5
    """Holder wrench authority on the core: max force (N) per axis."""

    holder_max_torque: float = 5.0e-3
    """Holder wrench authority on the core: max torque (Nm) per axis (it must absorb the
    rotator's reaction impulse)."""

    rotator_max_force: float = 0.02
    """Rotator wrench authority on the layer: max force (N) per axis (the scripted
    rotator uses torque only, but the action contract is 6D for both agents)."""

    rotator_max_torque: float = 2.0e-3
    """Rotator wrench authority on the layer: max torque (Nm) per axis."""

    # --- success / failure tolerances ------------------------------------------------------
    goal_angle: float = math.pi / 2
    """Goal hinge angle (rad): the layer must turn 90 degrees from the core."""

    success_angle_tol: float = 0.05
    """Success tolerance (rad) on `|angle - goal_angle|` (the spec's 0.05 rad)."""

    success_pos_tol: float = 0.01
    """Success tolerance (m) on the core's position drift (the spec's 1 cm)."""

    success_rot_tol: float = 0.1
    """Success tolerance (rad) on the core's rotation drift (the spec's 0.1 rad)."""

    success_lin_speed_tol: float = 0.05
    """Success tolerance (m/s) on the core's linear speed: the cube must be at rest."""

    success_ang_speed_tol: float = 0.3
    """Success tolerance (rad/s) on the combined core + layer angular speed."""

    failure_pos_drift: float = 0.05
    """Failure bound (m) on the core's position drift (the spec's 5 cm)."""

    failure_rot_drift: float = 0.5
    """Failure bound (rad) on the core's rotation drift (the spec's |yaw| > 0.5 rad)."""

    # --- episode randomisation (private randomness of the partner) -------------------------
    latency_range: tuple[float, float] = (0.0, 0.3)
    """Per-agent action latency (s): a commanded wrench is applied this many control
    steps later. Sampled per agent; the agent knows its own latency (it is in its
    observation), the partner's is hidden."""

    gain_range: tuple[float, float] = (0.6, 1.4)
    """Per-agent wrench-gain multiplier: the applied wrench is `action * authority *
    limit_scale * gain`. Sampled per agent."""

    limit_scale_range: tuple[float, float] = (0.5, 1.5)
    """Per-agent authority multiplier (0.5-1.5 of the config maxima). Sampled per agent."""

    slip_prob_range: tuple[float, float] = (0.0, 0.3)
    """Per-agent probability range of one slip event per episode (a brief loss of the
    applied wrench). The event probability itself is sampled per agent, then a Bernoulli
    draw decides whether the episode has it."""

    slip_decisions_range: tuple[int, int] = (1, 3)
    """Slip duration range (control steps, inclusive) of the one slip event."""

    start_delay_range: tuple[float, float] = (0.0, 0.3)
    """Per-agent delay (s) before the agent's wrench takes effect for the first time."""

    initial_angle_range: tuple[float, float] = (-0.05, 0.05)
    """Initial layer angle offset (rad) around 0, sampled per episode."""

    impulse_prob_range: tuple[float, float] = (0.0, 0.3)
    """Episode probability range of one common impulse disturbance on the core. Like the
    slip probability, the episode probability is sampled and then a Bernoulli draw
    decides whether the impulse happens."""

    impulse_magnitude_range: tuple[float, float] = (0.005, 0.05)
    """Magnitude range (N) of the impulse force on the core (applied for one control
    step in a random direction)."""

    # --- reward ---------------------------------------------------------------------------
    success_bonus: float = 1.0
    """Team reward granted on the step that first satisfies the success condition."""

    step_penalty: float = 0.01
    """Team reward subtracted on every control step (the shared time cost)."""

    # --- fallback seed --------------------------------------------------------------------
    seed: int = 0
    """Seed used when `reset(seed=None)`; each reset without an explicit seed advances it
    by one episode, so a fresh env does not replay the same randomisation."""

    def __post_init__(self) -> None:
        if not self.dt > 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.control_every < 1:
            raise ValueError(f"control_every must be >= 1, got {self.control_every}")
        if self.max_control_steps < 1:
            raise ValueError(f"max_control_steps must be >= 1, got {self.max_control_steps}")
        for name in (
            "core_lin_damping",
            "core_ang_damping",
            "hinge_damping",
            "hinge_frictionloss",
            "hinge_armature",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in (
            "holder_max_force",
            "holder_max_torque",
            "rotator_max_force",
            "rotator_max_torque",
        ):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if not 0 < self.goal_angle <= math.pi / 2:
            raise ValueError(f"goal_angle must be in (0, pi/2], got {self.goal_angle}")
        if not 0 < self.success_angle_tol < self.goal_angle:
            raise ValueError(
                f"success_angle_tol must be in (0, goal_angle) = (0, {self.goal_angle}), "
                f"got {self.success_angle_tol}"
            )
        if not 0 < self.success_pos_tol < self.failure_pos_drift:
            raise ValueError(
                f"success_pos_tol must be in (0, failure_pos_drift) = "
                f"(0, {self.failure_pos_drift}), got {self.success_pos_tol}"
            )
        if not 0 < self.success_rot_tol < self.failure_rot_drift:
            raise ValueError(
                f"success_rot_tol must be in (0, failure_rot_drift) = "
                f"(0, {self.failure_rot_drift}), got {self.success_rot_tol}"
            )
        for name in ("success_lin_speed_tol", "success_ang_speed_tol"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        _check_range("latency_range", self.latency_range)
        _check_range("gain_range", self.gain_range, positive_lo=True)
        _check_range("limit_scale_range", self.limit_scale_range, positive_lo=True)
        _check_range("slip_prob_range", self.slip_prob_range)
        _check_range("start_delay_range", self.start_delay_range)
        _check_range("impulse_prob_range", self.impulse_prob_range)
        _check_range("impulse_magnitude_range", self.impulse_magnitude_range)
        if self.slip_prob_range[1] > 1:
            raise ValueError(f"slip_prob_range must be within [0, 1], got {self.slip_prob_range}")
        if self.impulse_prob_range[1] > 1:
            raise ValueError(
                f"impulse_prob_range must be within [0, 1], got {self.impulse_prob_range}"
            )
        slip_lo, slip_hi = self.slip_decisions_range
        if slip_lo < 1 or slip_lo > slip_hi:
            raise ValueError(
                f"slip_decisions_range must be 1 <= lo <= hi, got {self.slip_decisions_range}"
            )
        if not self.initial_angle_range[0] <= self.initial_angle_range[1]:
            raise ValueError(
                f"initial_angle_range must have lo <= hi, got {self.initial_angle_range}"
            )
        if not 0 <= self.success_bonus:
            raise ValueError(f"success_bonus must be >= 0, got {self.success_bonus}")
        if not 0 <= self.step_penalty < self.success_bonus:
            raise ValueError(
                f"step_penalty must be in [0, success_bonus) = [0, {self.success_bonus}), "
                f"got {self.step_penalty}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")

    @property
    def control_dt(self) -> float:
        """Seconds per control step (`dt * control_every`)."""
        return self.dt * self.control_every

    @property
    def max_episode_time(self) -> float:
        """Simulated seconds of the episode budget (`control_dt * max_control_steps`)."""
        return self.control_dt * self.max_control_steps

    @property
    def holder_authority(self) -> tuple[float, float, float, float, float, float]:
        """Holder action scale: max force per axis, then max torque per axis."""
        return (self.holder_max_force,) * 3 + (self.holder_max_torque,) * 3

    @property
    def rotator_authority(self) -> tuple[float, float, float, float, float, float]:
        """Rotator action scale: max force per axis, then max torque per axis."""
        return (self.rotator_max_force,) * 3 + (self.rotator_max_torque,) * 3
