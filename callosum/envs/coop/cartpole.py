"""CartPoleCoop-v0: two one-sided pushers balance and move a shared cart-pole.

The task (`PLAN.md` "Two-robot task suite"): the `left` agent can only push the cart in
`+x`, the `right` agent only in `-x`, so neither can move the cart in both directions; the
team must drive the cart to `target_x` while keeping the pole upright and the speeds small.
The pole falls out after `failure_angle`. The classic unstable-object stabilisation becomes
a coordination problem: the balancing force regularly changes sign, so the two agents must
hand the control over to each other.

The fixed part of the harness (MJCF loading, latency queues, start-delay suppression,
one-sided scaling, observation assembly, team reward) lives in `CoopTaskEnv`; this module
only declares the cart-pole's ids, its config-as-truth model parameters and its rules.

The XML (`assets/cartpole.xml`) owns the geometry and the two masses (cart 2.0 kg, pole
3.0 m / 0.2 kg) because MuJoCo does not recompute a body's inertia when the mass is written
at runtime; `CartPoleConfig` owns everything else and documents the split.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from callosum.configs.coop_tasks import CartPoleConfig
from callosum.envs._coop_logic import (
    CARTPOLE_SPEC,
    CoopTaskSpec,
    EpisodeRandomization,
    cartpole_failure,
    cartpole_potential,
    cartpole_success,
)
from callosum.envs.coop.base import CoopTaskEnv

_CART_BODY = "cart"
_CART_JOINT = "cart_slide"
_POLE_BODY = "pole"
_POLE_JOINT = "pole_hinge"


class CartPoleEnv(CoopTaskEnv):
    """CartPoleCoop-v0: the shared cart-pole driven by two one-sided pushers."""

    TASK_NAME: ClassVar[str] = "cartpole"
    ASSET_NAME: ClassVar[str] = "cartpole.xml"
    SPEC: ClassVar[CoopTaskSpec] = CARTPOLE_SPEC
    AGENT_SIGNS: ClassVar[tuple[float, float]] = (1.0, -1.0)
    DEFAULT_CONFIG: ClassVar[type[CartPoleConfig]] = CartPoleConfig

    def _bind_task_model(self) -> None:
        """Resolve the cart/pole joint and body ids the task uses."""
        model = self._model
        self._cart_joint = int(model.joint(_CART_JOINT).id)
        self._cart_dof = int(model.jnt_dofadr[self._cart_joint])
        self._cart_qpos = int(model.jnt_qposadr[self._cart_joint])
        self._pole_joint = int(model.joint(_POLE_JOINT).id)
        self._pole_dof = int(model.jnt_dofadr[self._pole_joint])
        self._pole_qpos = int(model.jnt_qposadr[self._pole_joint])
        self._cart_body = int(model.body(_CART_BODY).id)
        self._pole_body = int(model.body(_POLE_BODY).id)

    def linear_dynamics(self) -> tuple[np.ndarray, np.ndarray]:
        """Continuous-time linearisation of the upright equilibrium (`ds/dt = A s + B F`).

        The state is `s = [cart_x, cart_v_x, pole_angle, pole_angvel]` and the input is the
        TOTAL world-x force on the cart (`F = left_force + right_force`). The coefficients
        are derived from the XML-owned masses/inertia and the config gravity, so the
        scripted oracle's model-based (LQR + Smith) controller stays consistent with the
        MuJoCo model:

        * ``(M + m - (m d)^2 / I) x_dd + (m^2 g d^2 / I) theta = F``
        * ``I theta_dd = m g d theta - m d x_dd``

        with `M` the cart mass, `m` the pole mass, `d` the pole COM distance from the hinge
        and `I` the pole inertia about the hinge (parallel-axis from the XML's body inertia).
        Valid for small angles around upright; the oracle uses it only as a prediction /
        gain-computation model.

        Returns:
            `(A, B)` with shapes `(4, 4)` and `(4, 1)`.
        """
        model = self._model
        cart_mass = float(model.body_mass[self._cart_body])
        pole_mass = float(model.body_mass[self._pole_body])
        com_distance = float(model.body_ipos[self._pole_body][2])
        inertia = float(model.body_inertia[self._pole_body][1]) + pole_mass * com_distance**2
        gravity = -float(model.opt.gravity[2])
        if com_distance <= 0 or inertia <= 0:
            raise RuntimeError(
                "the cart-pole XML must have the pole COM above the hinge with positive "
                f"inertia, got d={com_distance}, I={inertia}"
            )
        mass_eff = cart_mass + pole_mass - (pole_mass * com_distance) ** 2 / inertia
        gravity_coupling = (pole_mass**2 * gravity * com_distance**2) / inertia
        angle_gain = pole_mass * gravity * com_distance / inertia
        coupling = pole_mass * com_distance / inertia
        a = np.zeros((4, 4))
        a[0, 1] = 1.0
        a[1, 2] = -gravity_coupling / mass_eff
        a[2, 3] = 1.0
        a[3, 2] = angle_gain - coupling * a[1, 2]
        b = np.zeros((4, 1))
        b[1, 0] = 1.0 / mass_eff
        b[3, 0] = -coupling / mass_eff
        return a, b

    def _write_task_model_params(self) -> None:
        """Write the cart-pole's joint limit, damping and armature from the config."""
        cfg = self._config()
        model = self._model
        model.jnt_range[self._cart_joint] = (-cfg.rail_limit, cfg.rail_limit)
        model.dof_damping[self._cart_dof] = cfg.cart_damping
        model.dof_damping[self._pole_dof] = cfg.pole_damping
        model.dof_armature[self._pole_dof] = cfg.pole_armature

    def _write_sampled_randomization(self, randomization: EpisodeRandomization) -> None:
        """Write the episode's hidden pole-joint damping into the model."""
        self._model.dof_damping[self._pole_dof] = randomization.damping

    def _reset_task_state(self, randomization: EpisodeRandomization) -> None:
        """Place the cart at the rail origin and the pole at the sampled initial angle."""
        data = self._data
        data.qpos[self._cart_qpos] = 0.0
        data.qpos[self._pole_qpos] = randomization.initial_value
        data.qvel[:] = 0.0

    def _apply_task_actions(self, physical: dict[str, float], wind_force: float) -> None:
        """Write the two signed pusher forces (plus the one-step wind) on the cart body."""
        self._data.xfrc_applied[:] = 0.0
        total_force = physical["left"] + physical["right"] + wind_force
        self._data.xfrc_applied[self._cart_body, 0] = total_force

    def _shared_state_values(self) -> tuple[float, ...]:
        """Cart position/velocity and pole angle/angular velocity, in spec order."""
        data = self._data
        return (
            float(data.qpos[self._cart_qpos]),
            float(data.qvel[self._cart_dof]),
            float(data.qpos[self._pole_qpos]),
            float(data.qvel[self._pole_dof]),
        )

    def _goal_values(self) -> tuple[float, ...]:
        """The cart-position goal, in spec order."""
        return (self._config().target_x,)

    def _task_state(self) -> dict[str, float]:
        """The cart-pole's kinematic state for the scripted oracle."""
        data = self._data
        return {
            "cart_x": float(data.qpos[self._cart_qpos]),
            "cart_vx": float(data.qvel[self._cart_dof]),
            "pole_angle": float(data.qpos[self._pole_qpos]),
            "pole_angvel": float(data.qvel[self._pole_dof]),
            "target_x": self._config().target_x,
        }

    def _diagnostics(self) -> dict[str, float]:
        """Cart position, pole angle (degrees), goal distance and elapsed time."""
        state = self._task_state()
        cfg = self._config()
        return {
            "angle_deg": math.degrees(state["pole_angle"]),
            "cart_x": state["cart_x"],
            "cart_vx": state["cart_vx"],
            "pole_angle": state["pole_angle"],
            "pole_angvel": state["pole_angvel"],
            "distance_to_goal": abs(state["cart_x"] - cfg.target_x),
            "time": self._step_index * cfg.control_dt,
        }

    def _success(self) -> bool:
        """Whether the cart is at the target with the pole upright and both speeds small."""
        cfg = self._config()
        state = self._task_state()
        return cartpole_success(
            cart_x=state["cart_x"],
            cart_vx=state["cart_vx"],
            pole_angle=state["pole_angle"],
            pole_angvel=state["pole_angvel"],
            target_x=cfg.target_x,
            pos_tol=cfg.success_pos_tol,
            angle_tol=cfg.success_angle_tol,
            lin_speed_tol=cfg.success_lin_speed_tol,
            ang_speed_tol=cfg.success_ang_speed_tol,
        )

    def _failure(self) -> bool:
        """Whether the pole has fallen past the failure angle."""
        cfg = self._config()
        return cartpole_failure(self._task_state()["pole_angle"], angle_fail=cfg.failure_angle)

    def _task_potential(self) -> float:
        """The cart-pole's bounded shaping potential (target progress + upright posture)."""
        cfg = self._config()
        state = self._task_state()
        return cartpole_potential(
            cart_x=state["cart_x"],
            pole_angle=state["pole_angle"],
            target_x=cfg.target_x,
            pos_scale=max(abs(cfg.target_x), 1e-6),
            angle_scale=cfg.failure_angle,
            pos_coef=cfg.shaping_pos_coef,
            angle_coef=cfg.shaping_angle_coef,
        )

    def _config(self) -> CartPoleConfig:
        """The env's config as its concrete task type (the base stores it loosely)."""
        config = self.config
        if not isinstance(config, CartPoleConfig):  # pragma: no cover - construction guard
            raise TypeError(f"CartPoleEnv needs a CartPoleConfig, got {type(config)}")
        return config
