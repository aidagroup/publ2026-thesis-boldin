"""PendulumPump-v0: two one-sided torque actuators pump a shared pendulum.

The task (`PLAN.md` "Two-robot task suite"): the `left` agent can only command a positive
hinge torque, the `right` agent only a negative one; the team must pump the pendulum from a
small initial angle to `target_amplitude` and hold it for `hold_cycles` completed
half-cycles (zero crossings whose peak `|angle|` reached the target), without ever swinging
past the 150-degree failure bound. Pumping requires alternating the torque with the motion
direction, so one actuator alone can only contribute every other half-cycle: its reachable
amplitude saturates below the target (the task's non-additivity, tuned through the sampled
hinge damping).

The fixed part of the harness (MJCF loading, latency queues, start-delay suppression,
one-sided scaling, observation assembly, team reward) lives in `CoopTaskEnv`; this module
only declares the pendulum's ids, its config-as-truth model parameters, its success cycle
tracker and its rules.

The XML (`assets/pendulum.xml`) owns the bob's mass/geometry and its 0.5 m body-frame
offset because MuJoCo does not recompute a body's inertia when the mass is written at
runtime; `PendulumConfig` owns everything else and documents the split.
"""

from __future__ import annotations

import math
from typing import ClassVar

from callosum.experiments.coop_tasks.base import CoopTaskEnv
from callosum.experiments.coop_tasks.config import PendulumConfig
from callosum.experiments.coop_tasks.logic import (
    PENDULUM_SPEC,
    CoopTaskSpec,
    EpisodeRandomization,
    pendulum_failure,
    pendulum_potential,
    pendulum_success,
)

_HINGE_JOINT = "pivot_hinge"
_BOB_BODY = "bob"


class PendulumEnv(CoopTaskEnv):
    """PendulumPump-v0: the one-sided torques pumped to a target amplitude and held."""

    TASK_NAME: ClassVar[str] = "pendulum"
    ASSET_NAME: ClassVar[str] = "pendulum.xml"
    SPEC: ClassVar[CoopTaskSpec] = PENDULUM_SPEC
    AGENT_SIGNS: ClassVar[tuple[float, float]] = (1.0, -1.0)
    DEFAULT_CONFIG: ClassVar[type[PendulumConfig]] = PendulumConfig

    def _bind_task_model(self) -> None:
        """Resolve the hinge joint and bob body ids the task uses."""
        model = self._model
        self._hinge_joint = int(model.joint(_HINGE_JOINT).id)
        self._hinge_dof = int(model.jnt_dofadr[self._hinge_joint])
        self._hinge_qpos = int(model.jnt_qposadr[self._hinge_joint])
        self._bob_body = int(model.body(_BOB_BODY).id)

    def energy_parameters(self) -> tuple[float, float]:
        """The pendulum's energy coefficients `(I, m g d)` derived from the loaded model.

        The potential energy is `V(θ) = m g d (1 - cos θ)` and the kinetic energy is
        `0.5 * I * θ_dot^2`, with `d` the COM distance below the fixed pivot and `I` the
        inertia about the hinge (parallel-axis from the XML's body inertia). The scripted
        oracle uses the pair to predict the angle/velocity over its action latency (a
        gravity-only linearisation, which is all the phase-based pumping needs).

        Returns:
            `(inertia, gravity_torque_coefficient)` in `kg m^2` and `N m`.
        """
        model = self._model
        bob_mass = float(model.body_mass[self._bob_body])
        com_distance = -float(model.body_ipos[self._bob_body][2])
        inertia = float(model.body_inertia[self._bob_body][1]) + bob_mass * com_distance**2
        gravity = -float(model.opt.gravity[2])
        if com_distance <= 0 or inertia <= 0:
            raise RuntimeError(
                "the pendulum XML must have the bob COM below the pivot with positive "
                f"inertia, got d={com_distance}, I={inertia}"
            )
        return inertia, bob_mass * gravity * com_distance

    def _write_task_model_params(self) -> None:
        """Write the hinge range, damping and armature from the config."""
        cfg = self._config()
        model = self._model
        model.jnt_range[self._hinge_joint] = (-cfg.hinge_limit, cfg.hinge_limit)
        model.dof_damping[self._hinge_dof] = cfg.hinge_damping
        model.dof_armature[self._hinge_dof] = cfg.hinge_armature

    def _write_sampled_randomization(self, randomization: EpisodeRandomization) -> None:
        """Write the episode's hidden hinge damping into the model."""
        self._model.dof_damping[self._hinge_dof] = randomization.damping

    def _reset_task_state(self, randomization: EpisodeRandomization) -> None:
        """Place the pendulum at the sampled initial angle and reset the cycle trackers."""
        data = self._data
        data.qpos[self._hinge_qpos] = randomization.initial_value
        data.qvel[:] = 0.0
        self._peak_abs_angle = abs(randomization.initial_value)
        self._last_angle_sign = 1.0 if randomization.initial_value >= 0.0 else -1.0
        self._completed_cycles = 0

    def _after_step(self) -> None:
        """Update the amplitude tracker / completed-cycle counter after each control step.

        A zero crossing completes the current half-cycle: when its peak reached the target
        amplitude, the completed-cycle counter grows (the "hold" part of the goal); the peak
        tracker then restarts from the crossing angle.
        """
        angle = float(self._data.qpos[self._hinge_qpos])
        self._peak_abs_angle = max(self._peak_abs_angle, abs(angle))
        sign = 1.0 if angle >= 0.0 else -1.0
        if angle != 0.0 and sign != self._last_angle_sign:
            cfg = self._config()
            if self._peak_abs_angle >= cfg.target_amplitude:
                self._completed_cycles += 1
            self._peak_abs_angle = abs(angle)
            self._last_angle_sign = sign

    def _apply_task_actions(self, physical: dict[str, float], wind_force: float) -> None:
        """Write the two signed hinge torques (plus the optional gust) on the bob body."""
        self._data.xfrc_applied[:] = 0.0
        total_torque = physical["left"] + physical["right"] + wind_force
        # World y is the hinge axis; `xfrc_applied` torque is a free vector in world frame.
        self._data.xfrc_applied[self._bob_body, 4] = total_torque

    def _shared_state_values(self) -> tuple[float, ...]:
        """Hinge angle and angular velocity, in spec order."""
        data = self._data
        return (float(data.qpos[self._hinge_qpos]), float(data.qvel[self._hinge_dof]))

    def _goal_values(self) -> tuple[float, ...]:
        """The target amplitude and the required hold length, in spec order."""
        cfg = self._config()
        return (cfg.target_amplitude, float(cfg.hold_cycles))

    def _task_state(self) -> dict[str, float]:
        """The pendulum's kinematic state and cycle tracker for the scripted oracle."""
        data = self._data
        return {
            "angle": float(data.qpos[self._hinge_qpos]),
            "angvel": float(data.qvel[self._hinge_dof]),
            "target_amplitude": self._config().target_amplitude,
            "peak_abs_angle": float(self._peak_abs_angle),
            "completed_cycles": float(self._completed_cycles),
        }

    def _diagnostics(self) -> dict[str, float]:
        """Current amplitude (degrees), cycle progress and elapsed time."""
        cfg = self._config()
        state = self._task_state()
        return {
            "angle_deg": math.degrees(abs(state["angle"])),
            "amplitude_deg": math.degrees(self._peak_abs_angle),
            "completed_cycles": float(self._completed_cycles),
            "hold_cycles": float(cfg.hold_cycles),
            "time": self._step_index * cfg.control_dt,
        }

    def _success(self) -> bool:
        """Whether enough completed half-cycles reached the target amplitude."""
        cfg = self._config()
        return pendulum_success(
            completed_cycles=self._completed_cycles, hold_cycles=cfg.hold_cycles
        )

    def _failure(self) -> bool:
        """Whether the pendulum has swung past the failure angle."""
        cfg = self._config()
        return pendulum_failure(
            float(self._data.qpos[self._hinge_qpos]), angle_fail=cfg.failure_angle
        )

    def _task_potential(self) -> float:
        """The pendulum's bounded amplitude-progress shaping potential."""
        cfg = self._config()
        return pendulum_potential(
            peak_abs_angle=self._peak_abs_angle,
            target_amplitude=cfg.target_amplitude,
            coef=cfg.shaping_amplitude_coef,
        )

    def _config(self) -> PendulumConfig:
        """The env's config as its concrete task type (the base stores it loosely)."""
        config = self.config
        if not isinstance(config, PendulumConfig):  # pragma: no cover - construction guard
            raise TypeError(f"PendulumEnv needs a PendulumConfig, got {type(config)}")
        return config
