"""Scripted oracle for CubeWrench-v0: the environment-validation / upper-bound controller.

The plan's step 1 asks for a scripted controller that validates the environment and gives
the upper bound of the coordination quality: the holder PD-stabilises the core at its
initial pose, the rotator PD-drives the hinge to `goal_angle` (90 degrees). It is a
CENTRALISED script (one program commands both agents) used only to check that the physics
and the observation/reward contract are sane, not a policy and not part of the learned
triple. The only private information it uses is each agent's OWN action latency, which a
real agent also knows (it is in its observation): the commanded wrench takes effect only
after the latency, so both loops evaluate their error at the predicted actuation time
(`x + v * latency`, first-order lead) instead of the current one.

The gains are tuning knobs of this validation controller (`OracleParams`), deliberately
separate from the env config: they describe the script, not the environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from callosum.experiments.cube_wrench.config import WrenchCubeConfig
from callosum.experiments.cube_wrench.logic import injection_active

if TYPE_CHECKING:
    from callosum.experiments.cube_wrench.env import WrenchCubeEnv

_WRENCH_SIZE = 6
_HINGE_TORQUE_AXIS = 5
"""Index of the hinge-axis torque in the 6D wrench (world z, the hinge axis)."""


@dataclass(frozen=True)
class OracleParams:
    """PD gains of the scripted oracle (tuned for the default `WrenchCubeConfig`).

    The holder loop is slow and stiff on the core's pose; the rotator loop is a gentle
    PD on the hinge angle. With the config defaults these gains reach success in about
    1.5-2.5 s for every latency up to 0.3 s, while the rotator alone drifts the core past
    the failure bound (the non-additivity check).
    """

    holder_pos_kp: float = 0.5
    """Holder position stiffness (N/m) on the core's world position."""
    holder_pos_kd: float = 0.7
    """Holder position damping (Ns/m) on the core's world linear velocity."""
    holder_rot_kp: float = 2.0e-3
    """Holder rotation stiffness (Nm/rad) on the core's rotation vector."""
    holder_rot_kd: float = 0.0
    """Holder rotation damping (Nms/rad) on the core's world angular velocity. The
    reduced model's `core_ang_damping` already dissipates the rotation, so the script
    needs no explicit derivative term."""
    rotator_kp: float = 6.0e-4
    """Rotator stiffness (Nm/rad) on the hinge angle error at the actuation time."""
    rotator_kd: float = 1.0e-4
    """Rotator damping (Nms/rad) on the hinge angular velocity."""


def _rotation_vector(quat: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, rad) of a unit `(w, x, y, z)` quaternion vs identity."""
    w = float(np.clip(quat[0], -1.0, 1.0))
    angle = 2.0 * math.acos(abs(w))
    sin_half = math.sqrt(max(1.0 - w * w, 0.0))
    if sin_half < 1e-9:
        return np.zeros(3)
    axis = np.asarray(quat[1:4], dtype=np.float64) / sin_half
    if w < 0:
        axis = -axis
    return axis * angle


class OracleController:
    """The scripted two-agent oracle (holder PD stabilisation + rotator PD turn)."""

    def __init__(self, config: WrenchCubeConfig, params: OracleParams | None = None) -> None:
        """Store the config and the PD gains.

        Args:
            config: The environment config (goal angle and action authority are read
                from it).
            params: PD gains; defaults to `OracleParams()`.
        """
        self.config = config
        self.params = params if params is not None else OracleParams()

    def act(self, env: WrenchCubeEnv) -> dict[str, np.ndarray]:
        """Return the normalized wrench actions for both agents for the next step.

        Args:
            env: The environment to control; its current `state()` and the episode's
                per-agent latency (from `env.randomization`) are read. The env must have
                been reset.

        Returns:
            `{"holder": (6,), "rotator": (6,)}` normalized wrenches in `[-1, 1]`.
        """
        state = env.state()
        randomization = env.randomization
        if randomization is None:
            raise RuntimeError("call env.reset() before the oracle acts")
        cfg = self.config
        params = self.params
        holder = randomization.agents[0]
        rotator = randomization.agents[1]
        holder_latency = holder.latency_steps * cfg.control_dt
        rotator_latency = rotator.latency_steps * cfg.control_dt

        core_pos = np.asarray(state["core_pos"], dtype=np.float64)
        core_lin_vel = np.asarray(state["core_lin_vel"], dtype=np.float64)
        core_ang_vel = np.asarray(state["core_ang_vel"], dtype=np.float64)
        rot_vec = _rotation_vector(np.asarray(state["core_quat"], dtype=np.float64))
        force = -params.holder_pos_kp * (core_pos + core_lin_vel * holder_latency)
        force -= params.holder_pos_kd * core_lin_vel
        torque = -params.holder_rot_kp * (rot_vec + core_ang_vel * holder_latency)
        torque -= params.holder_rot_kd * core_ang_vel
        holder_action = self._normalize(
            np.concatenate([force, torque]),
            cfg.holder_authority,
            holder.limit_scale * holder.gain,
        )

        layer_angle = float(state["layer_angle"])
        layer_ang_vel = float(state["layer_ang_vel"])
        predicted_angle = layer_angle + layer_ang_vel * rotator_latency
        hinge_torque = params.rotator_kp * (cfg.goal_angle - predicted_angle)
        hinge_torque -= params.rotator_kd * layer_ang_vel
        rotator_wrench = np.zeros(_WRENCH_SIZE)
        rotator_wrench[_HINGE_TORQUE_AXIS] = hinge_torque
        rotator_action = self._normalize(
            rotator_wrench,
            cfg.rotator_authority,
            rotator.limit_scale * rotator.gain,
        )
        # The scripted sanity controller knows its own actuators' suppression (the env's
        # injected start delay / slip windows): while the HOLDER cannot hold, the rotator
        # must not turn (the reaction would drift the unheld core), and a suppressed
        # rotator simply pauses. This is the same "wait for the partner" logic the
        # learned policies have to discover, used here only to validate the environment.
        step = int(state["step"])
        if injection_active(holder, step) or injection_active(rotator, step):
            rotator_action = np.zeros(_WRENCH_SIZE)
        return {"holder": holder_action, "rotator": rotator_action}

    @staticmethod
    def _normalize(wrench: np.ndarray, authority: tuple[float, ...], scale: float) -> np.ndarray:
        """Convert a desired wrench into the normalized action under the actual authority."""
        effective = np.asarray(authority, dtype=np.float64) * scale
        return np.clip(wrench / effective, -1.0, 1.0)
