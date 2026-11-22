"""Scripted oracles of the coop suite: environment-validation / upper-bound controllers.

The plan's task-suite step asks for a scripted controller per task that validates the
environment and gives the upper bound of the coordination quality. All oracles here are
CENTRALISED scripts (one program commands both agents) used only to check that the physics,
the one-sided authority (the toss-catch paddles are symmetric position servos instead) and
the observation/reward contract are sane -- not policies and not part of the learned
`none`/`oracle`/`predicted` triple. They are also the source of the "only-left"
non-additivity baseline: run the same oracle but zero the right agent, and the task must
become unachievable (see `scripts/experiments/coop_tasks/smoke_coop.py`).

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

`TossOracle` is the catch-and-return script of `TossCatch-v0`: it predicts the ball's
ballistic trajectory from the observed state, picks a contact point on the receiving
paddle's face such that the paddle can still be moved there before the ball arrives (using
the known in-flight command queues -- the same Smith-prediction privilege as the cart-pole
oracle) and the frictionless rebound, with the nominal aiming tilt, sends the ball back
toward the partner's reachable paddle box. The other paddle is kept at its ready pose. It
learns the episode's hidden contact restitution from the observed bounce (it compares the
ball's velocity just before and just after a contact), so plans after the first pass use the
episode's actual elasticity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from callosum.experiments.coop_tasks.config import CartPoleConfig, PendulumConfig, TossConfig
from callosum.experiments.coop_tasks.logic import (
    AGENT_KEYS,
    ball_position_after,
    ball_velocity_after,
    bounce_velocity,
    in_start_delay,
    one_sided_allocation,
    pad_face_normal,
    paddle_tilt,
    partner_of,
)

if TYPE_CHECKING:
    from callosum.experiments.coop_tasks.cartpole import CartPoleEnv
    from callosum.experiments.coop_tasks.pendulum import PendulumEnv
    from callosum.experiments.coop_tasks.toss_catch import TossCatchEnv


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
    config: CartPoleConfig | PendulumConfig | TossConfig,
) -> CartPoleOracle | PendulumOracle | TossOracle:
    """Build the scripted oracle of `task` for `config`.

    Args:
        task: `"cartpole"`, `"pendulum"` or `"toss"`.
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
    if task == "toss":
        if not isinstance(config, TossConfig):
            raise ValueError(f"toss needs a TossConfig, got {type(config)}")
        return TossOracle(config)
    raise ValueError(f"task must be 'cartpole', 'pendulum' or 'toss', got {task!r}")


# --- toss-catch oracle -------------------------------------------------------------------


@dataclass(frozen=True)
class TossOracleParams:
    """Planning knobs of the toss-catch oracle."""

    assumed_restitution: float = 0.85
    """Initial effective-restitution estimate used before the first observed bounce; the
    oracle then replaces it with the value it estimates from every observed bounce."""

    reach_margin: float = 0.02
    """Time margin (s) reserved before the contact for the servo to reach its target.

    The paddles are physical bounded-force position servos now, not kinematic MOCAP bodies:
    the servo settles one control step (0.04 s at the default timing) after the last target
    change, and the target is already terminal at the planned contact, so the old 0.1 s
    "arrived and stopped" margin made every plan with a realistic latency infeasible (the
    ball is only airborne for ~0.5-0.8 s). Keeping one fifth of a control step of slack
    matches the measured servo tracking error (~1 cm on a 0.27 m target step)."""

    catch_clearance: float = 0.08
    """Minimum ball height above `floor_z` the oracle plans a contact for (m)."""

    plan_horizon: float = 1.6
    """Farthest future contact the planner considers (s)."""

    plan_time_step: float = 0.02
    """Contact-time search resolution (s)."""

    intercept_min_normal_speed: float = 0.15
    """Minimum approach speed along the paddle normal for a contact to be planned (m/s)."""


class TossOracle:
    """The scripted two-agent toss-catch oracle (ballistic interception + aiming tilt)."""

    def __init__(self, config: TossConfig, params: TossOracleParams | None = None) -> None:
        """Store the config and the planning knobs.

        Args:
            config: The environment config (geometry, paddle limits and tolerances are read
                from it).
            params: Planning knobs; defaults to `TossOracleParams()`.
        """
        self.config = config
        self.params = params if params is not None else TossOracleParams()
        self._restitution = float(self.params.assumed_restitution)
        self._plan_side: str | None = None
        self._plan: tuple[float, float, float, float] | None = None
        self._contact_previous = {"left": False, "right": False}
        self._velocity_before_contact = np.zeros(3)
        self._contact_side: str | None = None

    def _reset_episode(self) -> None:
        """Drop the per-episode caches (called on the first control step of an episode)."""
        self._restitution = float(self.params.assumed_restitution)
        self._plan_side = None
        self._plan = None
        self._contact_previous = {"left": False, "right": False}
        self._velocity_before_contact = np.zeros(3)
        self._contact_side = None

    # --- helpers ---------------------------------------------------------------------
    def _effective_limits(self, env: TossCatchEnv) -> dict[str, float]:
        """Each paddle's maximum pose-increment scale (its private authority)."""
        randomization = env.randomization
        if randomization is None:
            raise RuntimeError("call env.reset() before the oracle acts")
        return {
            key: self.config.pad_speed * self.config.control_dt * agent.gain * agent.limit_scale
            for key, agent in zip(AGENT_KEYS, randomization.agents)
        }

    def _pending_offset(
        self, env: TossCatchEnv, side: str, max_step: float, step: int
    ) -> tuple[np.ndarray, float]:
        """Paddle displacement/tilt still in flight (latency queues, start-delay aware).

        Returns the pose increment that will be applied once the command issued now reaches
        the physics, i.e. the sum of every queued command with its start-delay suppression.
        """
        randomization = env.randomization
        assert randomization is not None
        index = AGENT_KEYS.index(side)
        agent = randomization.agents[index]
        authority = agent.gain * agent.limit_scale
        tilt_step = self.config.pad_tilt_rate * self.config.control_dt * authority
        pending = env.pending_commands()[side]
        offset = np.zeros(2)
        tilt = 0.0
        for queue_offset, command in enumerate(pending):
            if in_start_delay(agent, step + queue_offset):
                continue
            vector = np.asarray(command, dtype=np.float64).reshape(-1)
            offset += vector[:2] * max_step
            tilt += vector[2] * tilt_step
        return offset, tilt

    def _deadbeat(
        self, env: TossCatchEnv, side: str, target: tuple[float, float, float], max_step: float
    ) -> np.ndarray:
        """Command that makes the paddle stop exactly at `target` once the latency clears."""
        cfg = self.config
        state = env.state()
        step = int(state["step"])
        # The increments accumulate on the servo TARGET (the plate physically chases it), so
        # the deadbeat command is computed in target space, not from the measured pose.
        pad = np.array([state[f"{side}_pad_target_x"], state[f"{side}_pad_target_z"]])
        pending, pending_tilt = self._pending_offset(env, side, max_step, step)
        desired = np.array([target[0], target[1]]) - pad - pending
        norm = float(np.linalg.norm(desired))
        if norm > max_step:
            desired *= max_step / norm
        randomization = env.randomization
        assert randomization is not None
        agent = randomization.agents[AGENT_KEYS.index(side)]
        authority = agent.gain * agent.limit_scale
        max_dtilt = cfg.pad_tilt_rate * cfg.control_dt * authority
        d_tilt = float(
            np.clip(
                target[2] - state[f"{side}_pad_target_tilt"] - pending_tilt,
                -max_dtilt,
                max_dtilt,
            )
        )
        return np.array([desired[0] / max_step, desired[1] / max_step, d_tilt / max_dtilt])

    def _return_catch(
        self, env: TossCatchEnv, side: str, contact: np.ndarray, velocity: np.ndarray
    ) -> tuple[float, float] | None:
        """First catchable descending point of the return trajectory on the partner paddle."""
        cfg = self.config
        offset = env.contact_offset()
        normal = pad_face_normal(paddle_tilt(side, cfg.pad_tilt))
        x_range = cfg.paddle_x_range(side)
        state = env.state()
        scale = self._effective_limits(env)[side]
        step = int(state["step"])
        pad = np.array([state[f"{side}_pad_target_x"], state[f"{side}_pad_target_z"]])
        pending, _ = self._pending_offset(env, side, scale, step)
        future = pad + pending
        randomization = env.randomization
        assert randomization is not None
        latency = randomization.agents[AGENT_KEYS.index(side)].latency_steps * cfg.control_dt
        seconds = 0.0
        while seconds < self.params.plan_horizon:
            seconds += self.params.plan_time_step
            position = ball_position_after(contact, velocity, seconds, cfg.gravity)
            if position[2] < cfg.floor_z + self.params.catch_clearance:
                return None
            approach = ball_velocity_after(velocity, seconds, cfg.gravity)
            if approach[2] >= 0.0:
                continue
            centre = position - offset * normal
            if not x_range[0] <= centre[0] <= x_range[1]:
                continue
            if not cfg.pad_min_z <= centre[2] <= cfg.pad_max_z:
                continue
            budget = cfg.pad_speed * (seconds - latency - self.params.reach_margin)
            if budget <= 0.0:
                continue
            if float(np.linalg.norm(centre[[0, 2]] - future)) <= budget + 1e-9:
                return float(position[2]), float(seconds)
            return None
        return None

    def _plan_pass(self, env: TossCatchEnv, side: str) -> tuple[float, float, float, float] | None:
        """Best `(score, seconds, centre_x, centre_z)` catch on `side` for a good return."""
        cfg = self.config
        state = env.state()
        ball = np.array([state["ball_x"], 0.0, state["ball_z"]])
        velocity = np.array([state["ball_vx"], 0.0, state["ball_vz"]])
        offset = env.contact_offset()
        scale = self._effective_limits(env)[side]
        tilt = paddle_tilt(side, cfg.pad_tilt)
        normal = pad_face_normal(tilt)
        x_range = cfg.paddle_x_range(side)
        randomization = env.randomization
        assert randomization is not None
        latency = randomization.agents[AGENT_KEYS.index(side)].latency_steps * cfg.control_dt
        step = int(state["step"])
        pad = np.array([state[f"{side}_pad_target_x"], state[f"{side}_pad_target_z"]])
        pending, _ = self._pending_offset(env, side, scale, step)
        future = pad + pending
        other = partner_of(side)
        want = -1.0 if side == "right" else 1.0
        best: tuple[float, float, float, float] | None = None
        seconds = 0.0
        while seconds < self.params.plan_horizon:
            seconds += self.params.plan_time_step
            position = ball_position_after(ball, velocity, seconds, cfg.gravity)
            if position[2] < cfg.floor_z + self.params.catch_clearance:
                break
            approach = ball_velocity_after(velocity, seconds, cfg.gravity)
            if approach[2] >= 0.0:
                continue
            if float(approach @ normal) >= -self.params.intercept_min_normal_speed:
                continue
            centre = position - offset * normal
            if not x_range[0] <= centre[0] <= x_range[1]:
                continue
            if not cfg.pad_min_z <= centre[2] <= cfg.pad_max_z:
                continue
            budget = cfg.pad_speed * (seconds - latency - self.params.reach_margin)
            if budget <= 0.0:
                continue
            if float(np.linalg.norm(centre[[0, 2]] - future)) > budget + 1e-9:
                continue
            outgoing = bounce_velocity(approach, normal, self._restitution)
            if outgoing[0] * want <= 0.2:
                continue
            if self._return_catch(env, other, position, outgoing) is None:
                continue
            score = position[2] - 0.1 * abs(outgoing[0])
            if best is None or score > best[0]:
                best = (score, seconds, float(centre[0]), float(centre[2]))
        return best

    def _fallback_target(self, env: TossCatchEnv, side: str) -> tuple[float, float, float]:
        """Best-effort intercept when no fully feasible catch was found.

        Keep the paddle at its current height (moving up into the incoming ball would bat it
        away with a large kinematic impulse) and slide it horizontally to where the ball's
        free trajectory descends to that height. Even an off-centre contact can count as a
        pass, while a stopped paddle would certainly miss the ball.
        """
        cfg = self.config
        state = env.state()
        ball_z = float(state["ball_z"])
        ball_vz = float(state["ball_vz"])
        ball_x = float(state["ball_x"])
        ball_vx = float(state["ball_vx"])
        pad_z = float(state[f"{side}_pad_target_z"])
        tilt = paddle_tilt(side, cfg.pad_tilt)
        normal = pad_face_normal(tilt)
        offset = env.contact_offset()
        x_range = cfg.paddle_x_range(side)
        contact_z = pad_z + offset * float(normal[2])
        discriminant = ball_vz * ball_vz - 2.0 * cfg.gravity * (contact_z - ball_z)
        target_x = float(np.clip(ball_x - offset * float(normal[0]), *x_range))
        if discriminant >= 0.0:
            seconds = (ball_vz + math.sqrt(discriminant)) / cfg.gravity
            if seconds > 0.0:
                target_x = float(
                    np.clip(ball_x + ball_vx * seconds - offset * float(normal[0]), *x_range)
                )
        return (target_x, pad_z, tilt)

    def _update_restitution(self, env: TossCatchEnv) -> None:
        """Estimate the episode's effective restitution from the observed bounces."""
        state = env.state()
        contact = {side: state[f"{side}_contact"] > 0.5 for side in AGENT_KEYS}
        any_contact = any(contact.values())
        any_previous = any(self._contact_previous.values())
        if any_contact and not any_previous:
            self._contact_side = next(side for side in AGENT_KEYS if contact[side])
            self._velocity_before_contact = np.array([state["ball_vx"], 0.0, state["ball_vz"]])
        elif not any_contact and any_previous and self._contact_side is not None:
            normal = pad_face_normal(paddle_tilt(self._contact_side, self.config.pad_tilt))
            incoming = float(self._velocity_before_contact @ normal)
            outgoing = float(np.array([state["ball_vx"], 0.0, state["ball_vz"]]) @ normal)
            if incoming < -0.05:
                self._restitution = float(np.clip(-outgoing / incoming, 0.2, 1.0))
        self._contact_previous = contact

    def act(self, env: TossCatchEnv) -> dict[str, np.ndarray]:
        """Return the normalized paddle commands for both agents for the next step.

        Args:
            env: The environment to control; its `state()`, `pending_commands()` and the
                episode randomisation are read. The env must have been reset.

        Returns:
            `{"left": (3,), "right": (3,)}` normalized `(dx, dz, dtheta)` commands.
        """
        cfg = self.config
        state = env.state()
        if state["step"] == 0.0:
            # A fresh episode: drop the per-episode plan/restitution caches so a reused
            # oracle instance (the smoke/eval loops build it once per env) cannot leak state.
            self._reset_episode()
        # The hidden release has not happened while the ball has no lateral velocity: the
        # pre-release ball only free-falls (vx == 0). Testing the lateral component (not the
        # full speed) keeps the anticipation loop from flip-flopping once gravity has given
        # the ball a vertical velocity: with the physical servoed paddles a flip-flopping
        # anticipation target would make the plate physically swing through the intercept.
        launched = abs(state["ball_vx"]) > 1e-6
        limits = self._effective_limits(env)
        if not launched:
            # Anticipate the hidden throw by moving the receiving (right) paddle toward the
            # middle of the court at the pre-throw height, while the left paddle waits at its
            # ready pose. The hidden impulse then has to cover a much shorter distance for the
            # first intercept.
            receive = (
                env.paddle_receive_x(),
                cfg.pad_ready_z,
                paddle_tilt("right", cfg.pad_tilt),
            )
            left_ready = (
                -env.paddle_ready_x(),
                cfg.pad_start_z,
                paddle_tilt("left", cfg.pad_tilt),
            )
            return {
                "right": self._deadbeat(env, "right", receive, limits["right"]).astype(np.float32),
                "left": self._deadbeat(env, "left", left_ready, limits["left"]).astype(np.float32),
            }
        self._update_restitution(env)
        side = "right" if state["ball_vx"] > 0.0 else "left"
        other = partner_of(side)
        if self._plan_side != side or self._plan is None:
            self._plan = self._plan_pass(env, side)
            self._plan_side = side
        actions: dict[str, np.ndarray] = {}
        if self._plan is None:
            target = self._fallback_target(env, side)
        else:
            _, _seconds, centre_x, centre_z = self._plan
            target = (centre_x, centre_z, paddle_tilt(side, cfg.pad_tilt))
        actions[side] = self._deadbeat(env, side, target, limits[side]).astype(np.float32)
        ready = (
            -env.paddle_ready_x() if other == "left" else env.paddle_ready_x(),
            cfg.pad_start_z,
            env.paddle_ready_tilt(other),
        )
        actions[other] = self._deadbeat(env, other, ready, limits[other]).astype(np.float32)
        return actions
