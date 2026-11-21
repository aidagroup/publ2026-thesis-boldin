"""Scripted oracles of the coop suite: environment-validation / upper-bound controllers.

The plan's task-suite step asks for a scripted controller per task that validates the
environment and gives the upper bound of the coordination quality. Both oracles here are
CENTRALISED scripts (one program commands both agents) used only to check that the physics,
the one-sided authority and the observation/reward contract are sane -- not policies and not
part of the learned `none`/`oracle`/`predicted` triple. They are also the source of the
"only-left" non-additivity baseline: run the same oracle but zero the right agent, and the
task must become unachievable (see `scripts/smoke_coop.py`).

The scripted oracle is deliberately PRIVILEGED: it reads the env's sampled randomisation
(each agent's latency and start delay) and the base env's `pending_commands()` (the
in-flight command queues), which no decentralised policy can see. That is exactly what a
centralised upper bound is allowed to use; the policy observations contain only the agent's
own private parameters.

`CartPoleOracle` is a model-based full-state feedback: the discrete infinite-horizon LQR
gain of the cart-pole linearisation (`CartPoleEnv.linear_dynamics`, derived from the XML
masses) is evaluated at the state predicted over the agents' latency by a Smith predictor
(the known in-flight commands are propagated through the same linear model). This is the
delayed version of the plan's "centralised PD on the pole angle and the cart position": the
LQR state feedback contains those PD terms and, unlike a raw PD, stays stable at the
up-to-0.3 s action latency.

`PendulumOracle` is the phase-based energy-pumping controller the task calls for: pump with
the actuator whose one-sided torque agrees with the (latency-predicted) hinge velocity, and
brake once the current half-cycle peak reaches the target amplitude. Both actuators are used
across one full oscillation -- exactly what the one-sided "only-left" baseline cannot do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from callosum.configs.coop_tasks import CartPoleConfig, PendulumConfig
from callosum.envs._coop_logic import AGENT_KEYS, in_start_delay, one_sided_allocation

if TYPE_CHECKING:
    from callosum.envs.coop.cartpole import CartPoleEnv
    from callosum.envs.coop.pendulum import PendulumEnv


@dataclass(frozen=True)
class CartPoleOracleParams:
    """Weights and controller knobs of the cart-pole oracle.

    The four state weights are the diagonal `Q` of the LQR objective (cart position, cart
    velocity, pole angle, pole angular velocity) and `force_weight` is its scalar `R`; the
    defaults keep the closed-loop response fast enough for the 300-step budget while using
    roughly a third of the one-sided force authority in normal operation.
    """

    position_weight: float = 50.0
    """LQR `Q` entry of the cart-position error."""

    velocity_weight: float = 5.0
    """LQR `Q` entry of the cart-velocity error."""

    angle_weight: float = 200.0
    """LQR `Q` entry of the pole-angle error (the balancing term dominates)."""

    angular_velocity_weight: float = 20.0
    """LQR `Q` entry of the pole angular-velocity error."""

    force_weight: float = 1.0
    """LQR `R`: the control-effort penalty (larger = gentler, slower controller)."""


@dataclass(frozen=True)
class PendulumOracleParams:
    """Phase-based pumping controller knobs of the pendulum oracle."""

    pump_ratio: float = 1.0
    """Fraction of the agent's maximum torque used while pumping."""

    brake_ratio: float = 1.0
    """Fraction of the agent's maximum torque used while braking above the target."""

    min_angvel: float = 0.01
    """Dead-band (rad/s) of the velocity sign; below it the controller nudges in the
    direction of the angle so a pendulum starting exactly at rest still starts pumping."""


# --- model-based controller utilities (pure numpy) --------------------------------------


def _matrix_exponential(matrix: np.ndarray, terms: int = 40) -> np.ndarray:
    """Matrix exponential by a truncated Taylor series (small arg: < 1e-2 here)."""
    result = np.eye(matrix.shape[0])
    term = np.eye(matrix.shape[0])
    for order in range(1, terms):
        term = term @ matrix / order
        result = result + term
    return result


def discretize_linear(
    a_continuous: np.ndarray, b_continuous: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold discretisation of `ds/dt = A s + B u` at control period `dt`.

    The augmented-matrix trick computes both the state transition and the input matrix with
    one matrix exponential:
    `expm([[A, B], [0, 0]] * dt) = [[A_d, B_d], [0, I]]`.

    Args:
        a_continuous: Continuous state matrix, `(n, n)`.
        b_continuous: Continuous input matrix, `(n, m)`.
        dt: Control period (s).

    Returns:
        `(A_d, B_d)` of shapes `(n, n)` and `(n, m)`.
    """
    n, m = a_continuous.shape[0], b_continuous.shape[1]
    augmented = np.zeros((n + m, n + m))
    augmented[:n, :n] = a_continuous * dt
    augmented[:n, n:] = b_continuous * dt
    exponential = _matrix_exponential(augmented)
    return exponential[:n, :n], exponential[:n, n:]


def discrete_lqr_gain(
    a_discrete: np.ndarray,
    b_discrete: np.ndarray,
    q_weights: np.ndarray,
    r_weight: np.ndarray,
    *,
    max_iterations: int = 100_000,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Infinite-horizon discrete LQR gain by Riccati iteration.

    Iterates `P <- Q + A' P A - A' P B (R + B' P B)^-1 B' P A` to convergence and returns
    `K = (R + B' P B)^-1 B' P A`, i.e. the optimal `u = -K s`.

    Args:
        a_discrete: Discrete state matrix, `(n, n)`.
        b_discrete: Discrete input matrix, `(n, m)`.
        q_weights: State cost matrix, `(n, n)` positive semi-definite.
        r_weight: Input cost matrix, `(m, m)` positive definite.
        max_iterations: Iteration cap (the DARE solution converges in tens of iterations
            for this small slow system; the cap only guards a bad weight set).
        tolerance: Convergence threshold on the maximum Riccati-matrix change.

    Returns:
        The gain matrix `(m, n)`.

    Raises:
        RuntimeError: if the iteration does not converge within `max_iterations`.
    """
    p = np.array(q_weights, dtype=np.float64, copy=True)
    for _ in range(max_iterations):
        b_t_p = b_discrete.T @ p
        updated = (
            q_weights
            + a_discrete.T @ p @ a_discrete
            - a_discrete.T
            @ p
            @ b_discrete
            @ np.linalg.inv(r_weight + b_t_p @ b_discrete)
            @ b_t_p
            @ a_discrete
        )
        updated = 0.5 * (updated + updated.T)
        if np.max(np.abs(updated - p)) < tolerance:
            p = updated
            break
        p = updated
    else:  # pragma: no cover - the defaults converge in well under the cap
        raise RuntimeError(
            f"discrete LQR did not converge in {max_iterations} iterations; check the weights"
        )
    return np.linalg.inv(r_weight + b_discrete.T @ p @ b_discrete) @ b_discrete.T @ p @ a_discrete


# --- cart-pole oracle --------------------------------------------------------------------


class CartPoleOracle:
    """The scripted two-agent cart-pole oracle (LQR + Smith predictor + one-sided split)."""

    def __init__(self, config: CartPoleConfig, params: CartPoleOracleParams | None = None) -> None:
        """Store the config and the LQR weights.

        Args:
            config: The environment config (target position and force authority are read
                from it).
            params: LQR weights; defaults to `CartPoleOracleParams()`.
        """
        self.config = config
        self.params = params if params is not None else CartPoleOracleParams()
        self._a_discrete: np.ndarray | None = None
        self._b_discrete: np.ndarray | None = None
        self._gain: np.ndarray | None = None

    def _controller(self, env: CartPoleEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The cached `(A_d, B_d, K)` of the current env's linearisation."""
        if self._gain is None:
            a_continuous, b_continuous = env.linear_dynamics()
            self._a_discrete, self._b_discrete = discretize_linear(
                a_continuous, b_continuous, self.config.control_dt
            )
            q_weights = np.diag(
                [
                    self.params.position_weight,
                    self.params.velocity_weight,
                    self.params.angle_weight,
                    self.params.angular_velocity_weight,
                ]
            )
            self._gain = discrete_lqr_gain(
                self._a_discrete,
                self._b_discrete,
                q_weights,
                np.array([[self.params.force_weight]]),
            )
        return self._a_discrete, self._b_discrete, self._gain  # type: ignore[return-value]

    def _effective_limits(self, env: CartPoleEnv) -> list[float]:
        """The physical force authority of each agent in this episode (N)."""
        randomization = env.randomization
        if randomization is None:
            raise RuntimeError("call env.reset() before the oracle acts")
        return [
            self.config.agent_authority(index) * agent.gain * agent.limit_scale
            for index, agent in enumerate(randomization.agents)
        ]

    def _predicted_state(
        self, env: CartPoleEnv, a_discrete: np.ndarray, b_discrete: np.ndarray
    ) -> np.ndarray:
        """Smith prediction of the cart-pole state at the agents' actuation times.

        Propagates the known in-flight commands (oldest first, one per control step) through
        the linear model for the larger of the two latencies; the command computed now takes
        effect exactly then. A start delay zeroes the corresponding force, exactly like the
        env's own suppression.
        """
        state = env.state()
        randomization = env.randomization
        assert randomization is not None
        predicted = np.array(
            [state["cart_x"], state["cart_vx"], state["pole_angle"], state["pole_angvel"]],
            dtype=np.float64,
        )
        horizon = max(agent.latency_steps for agent in randomization.agents)
        pending = env.pending_commands()
        limits = self._effective_limits(env)
        signs = (1.0, -1.0)
        step_now = int(state["step"])
        for offset in range(horizon):
            total_force = 0.0
            for index, key in enumerate(AGENT_KEYS):
                queue = pending[key]
                if offset >= len(queue):
                    continue
                agent = randomization.agents[index]
                if in_start_delay(agent, step_now + offset):
                    continue
                command = max(float(queue[offset][0]), 0.0)
                total_force += signs[index] * command * limits[index]
            predicted = a_discrete @ predicted + b_discrete[:, 0] * total_force
        return predicted

    def act(self, env: CartPoleEnv) -> dict[str, np.ndarray]:
        """Return the normalized one-sided commands for both agents for the next step.

        Args:
            env: The environment to control; its `state()`, `pending_commands()` and the
                episode randomisation are read. The env must have been reset.

        Returns:
            `{"left": (1,), "right": (1,)}` normalized commands in `[0, 1]`.
        """
        randomization = env.randomization
        if randomization is None:
            raise RuntimeError("call env.reset() before the oracle acts")
        cfg = self.config
        a_discrete, b_discrete, gain = self._controller(env)
        predicted = self._predicted_state(env, a_discrete, b_discrete)
        error = predicted - np.array([cfg.target_x, 0.0, 0.0, 0.0])
        desired_force = float(-(gain @ error)[0])
        limits = self._effective_limits(env)
        positive_force, negative_force = one_sided_allocation(
            desired_force, positive_limit=limits[0], negative_limit=limits[1]
        )
        step = int(env.state()["step"])
        commands = {
            "left": _normalized(positive_force, limits[0]),
            "right": _normalized(negative_force, limits[1]),
        }
        for index, key in enumerate(AGENT_KEYS):
            if in_start_delay(randomization.agents[index], step):
                commands[key] = 0.0
        return {key: np.array([value], dtype=np.float32) for key, value in commands.items()}


# --- pendulum oracle ---------------------------------------------------------------------


class PendulumOracle:
    """The scripted two-agent pendulum oracle (phase-based pumping + braking)."""

    def __init__(self, config: PendulumConfig, params: PendulumOracleParams | None = None) -> None:
        """Store the config and the controller parameters.

        Args:
            config: The environment config (target amplitude and torque authority are read
                from it).
            params: Controller parameters; defaults to `PendulumOracleParams()`.
        """
        self.config = config
        self.params = params if params is not None else PendulumOracleParams()

    def _effective_limits(self, env: PendulumEnv) -> list[float]:
        """The physical torque authority of each agent in this episode (Nm)."""
        randomization = env.randomization
        if randomization is None:
            raise RuntimeError("call env.reset() before the oracle acts")
        return [
            self.config.agent_authority(index) * agent.gain * agent.limit_scale
            for index, agent in enumerate(randomization.agents)
        ]

    def act(self, env: PendulumEnv) -> dict[str, np.ndarray]:
        """Return the normalized one-sided torque commands for both agents.

        Args:
            env: The environment to control; its `state()`, `energy_parameters()` and the
                episode randomisation are read. The env must have been reset.

        Returns:
            `{"left": (1,), "right": (1,)}` normalized commands in `[0, 1]`.
        """
        randomization = env.randomization
        if randomization is None:
            raise RuntimeError("call env.reset() before the oracle acts")
        cfg = self.config
        params = self.params
        state = env.state()
        if state["completed_cycles"] >= cfg.hold_cycles:
            # Already successful; the env terminates on the success step, so this is a guard.
            return {key: np.zeros(1, dtype=np.float32) for key in AGENT_KEYS}

        inertia, gravity_torque = env.energy_parameters()
        latency = max(agent.latency_steps for agent in randomization.agents) * cfg.control_dt
        angle = state["angle"]
        angvel = state["angvel"]
        damping = randomization.damping
        # Predict the actuation-time state with a gravity/damping linearisation (sufficient
        # for a phase controller; the pumping only needs the sign of the predicted velocity).
        angular_accel = (-gravity_torque * math.sin(angle) - damping * angvel) / inertia
        predicted_angvel = angvel + angular_accel * latency
        predicted_angle = angle + angvel * latency
        predicted_peak = max(state["peak_abs_angle"], abs(predicted_angle))

        desired_torque = 0.0
        if predicted_peak < cfg.target_amplitude:
            if abs(predicted_angvel) >= params.min_angvel:
                desired_torque = math.copysign(cfg.max_torque * params.pump_ratio, predicted_angvel)
            else:
                # A pendulum starting exactly at rest would never get a velocity sign; nudge
                # it away from the centre so the next steps can pump.
                direction = angle if angle != 0.0 else 1.0
                desired_torque = math.copysign(cfg.max_torque * params.pump_ratio, direction)
        elif predicted_peak > cfg.target_amplitude and abs(predicted_angvel) >= params.min_angvel:
            desired_torque = math.copysign(-cfg.max_torque * params.brake_ratio, predicted_angvel)

        limits = self._effective_limits(env)
        positive_torque, negative_torque = one_sided_allocation(
            desired_torque, positive_limit=limits[0], negative_limit=limits[1]
        )
        step = int(state["step"])
        commands = {
            "left": _normalized(positive_torque, limits[0]),
            "right": _normalized(negative_torque, limits[1]),
        }
        for index, key in enumerate(AGENT_KEYS):
            if in_start_delay(randomization.agents[index], step):
                commands[key] = 0.0
        return {key: np.array([value], dtype=np.float32) for key, value in commands.items()}


def _normalized(physical: float, limit: float) -> float:
    """A physical force/torque as a normalized `[0, 1]` command of an actuator of `limit`."""
    if limit <= 0:
        return 0.0
    return float(np.clip(physical / limit, 0.0, 1.0))


def make_oracle(
    task: str,
    config: CartPoleConfig | PendulumConfig,
) -> CartPoleOracle | PendulumOracle:
    """Build the scripted oracle of `task` for `config`.

    Args:
        task: `"cartpole"` or `"pendulum"`.
        config: The env config of the same task.

    Returns:
        The task's oracle controller.

    Raises:
        ValueError: for an unknown task or a config of the wrong type.
    """
    if task == "cartpole":
        if not isinstance(config, CartPoleConfig):
            raise ValueError(f"cartpole needs a CartPoleConfig, got {type(config)}")
        return CartPoleOracle(config)
    if task == "pendulum":
        if not isinstance(config, PendulumConfig):
            raise ValueError(f"pendulum needs a PendulumConfig, got {type(config)}")
        return PendulumOracle(config)
    raise ValueError(f"task must be 'cartpole' or 'pendulum', got {task!r}")
