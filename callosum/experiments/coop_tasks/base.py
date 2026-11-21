"""Shared MuJoCo harness of the two-agent reduced coop suite (Gymnasium API).

`CoopTaskEnv` is the common base of `CartPoleCoop-v0`, `PendulumPump-v0` and
`TossCatch-v0`: it owns everything that is task-independent -- loading the MJCF, writing the
config's timing/physics into the model, the per-agent latency queues and start-delay
suppression, the one-sided actuation, the shared team reward, the observation assembly and
the Gymnasium contract -- while a subclass supplies only the task physics and rules through
a small set of hooks:

* ``_write_task_model_params`` -- joint limits/damping/armature from the config;
* ``_bind_task_model`` -- the joint/body ids the task uses;
* ``_reset_task_state`` -- place the model's qpos and reset the task's trackers;
* ``_write_sampled_randomization`` -- write the episode's hidden physical parameter;
* ``_sample_randomization`` -- the task's episode sampler (the suite sampler by default);
* ``_task_state`` -- the oracle's `state()` block;
* ``_shared_state_values`` / ``_goal_values`` -- the observation blocks;
* ``_apply_task_actions`` -- the physical commands (and the optional perturbation) into the
  model (`mjData.xfrc_applied` for the force tasks, the paddle pose for the toss-catch);
* ``_on_substep`` -- optional per-physics-substep hook (smooth actuation and contact
  sampling for the toss-catch task);
* ``_diagnostics`` / ``_success`` / ``_failure`` / ``_task_potential`` -- info/reward;
* ``_after_step`` -- optional per-control-step tracker update (the pendulum's zero crossings,
  the toss-catch paddles' velocity).

The two agents are `left` (commands only POSITIVE physical quantities) and `right` (only
NEGATIVE ones): the base maps an applied normalized action `a` to the physical command
`sign * max(a, 0) * config.authority * gain * limit_scale`, so neither agent can drive the
shared object in both directions -- the suite's non-additivity. The toss-catch task is the
exception: its paddles are symmetric position-controlled bodies, so that env overrides
`_physical_commands` and maps the normalized pose increments without the one-sided split
(the coordination there comes from timing/anticipation, not from the actuator split).

Per control step (`control_every` physics substeps): both commands go through the agent's
private latency queue and are suppressed during its start delay, the physical commands and
the episode's optional one-step perturbation are written into `xfrc_applied`, the physics
runs, and the observation/reward/termination are computed from the resulting state. Every
per-episode random parameter (latency, gain, authority scale, start delay, the hidden joint
damping, the initial state and the perturbation) is sampled deterministically from the
episode seed by `callosum.experiments.coop_tasks.logic.sample_episode_randomization`, so an episode is
reproducible from its seed and nothing crosses episode boundaries.

The observation is a per-agent flat `spaces.Box` built by the pure
`callosum.experiments.coop_tasks.logic.build_observation` from a role-tagged `values` dict; the partner
block is present only under `partner_mode="oracle"`, so the "none" mode cannot leak it.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces

from callosum.experiments.coop_tasks.logic import (
    AGENT_KEYS,
    CoopTaskSpec,
    EpisodeRandomization,
    action_value_fields,
    build_observation,
    command_value_fields,
    delayed_command,
    in_start_delay,
    obs_size,
    param_value_fields,
    potential_shaping,
    privileged_value_fields,
    sample_episode_randomization,
    team_reward,
    validate_partner_mode,
    wind_active_on_step,
)

if TYPE_CHECKING:
    from callosum.experiments.coop_tasks.config import (
        CartPoleConfig,
        PendulumConfig,
        TossConfig,
    )

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


class CoopTaskEnv(gymnasium.Env):
    """The common two-agent MuJoCo harness of the reduced coop suite.

    The `gymnasium` API is `reset(seed=None, options=None) -> (obs, info)` and
    `step(action) -> (obs, reward, terminated, truncated, info)` with `action` a dict
    `{"left": (action_dim,), "right": (action_dim,)}` of normalized one-sided commands in
    `[-1, 1]`. A negative command means "no force" for the left agent and "no force" for
    the right one as well (the right agent's positive sign is its `-1`... see
    `AGENT_SIGNS`): the commands are mapped onto the task's physical quantity in
    `_physical_commands`, so the two agents are mirror-images and a single agent can never
    command both directions.

    Subclasses set the class attributes below and implement the hooks listed in the module
    docstring; they never re-implement `reset`/`step`/the observation assembly.
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    TASK_NAME: ClassVar[str] = ""
    """Task name (`"cartpole"` / `"pendulum"`), matching the registered env id."""
    ASSET_NAME: ClassVar[str] = ""
    """MJCF file name under `callosum/experiments/coop_tasks/assets/`."""
    SPEC: ClassVar[CoopTaskSpec]
    """The task's observation/action layout (`callosum.experiments.coop_tasks.logic`)."""
    AGENT_SIGNS: ClassVar[tuple[float, float]] = (1.0, -1.0)
    """Physical sign each agent can command: `left` only `+`, `right` only `-`."""

    def __init__(
        self,
        config: CartPoleConfig | PendulumConfig | TossConfig | None = None,
        partner_mode: str = "none",
        render_mode: str | None = None,
    ) -> None:
        """Load the MJCF model, write the config's parameters and build the spaces.

        Args:
            config: The task config (the subclass's default config when `None`).
            partner_mode: Observation visibility of the partner, `"none"` or `"oracle"`.
            render_mode: Unused -- the coop suite renders nothing; a non-None value is an
                error rather than a silent no-op.
        """
        if render_mode is not None:
            raise ValueError(f"{type(self).__name__} renders nothing; render_mode must be None")
        validate_partner_mode(partner_mode)
        default_config = getattr(self, "DEFAULT_CONFIG", None)
        if config is None and default_config is None:
            raise ValueError(f"{type(self).__name__} has no DEFAULT_CONFIG; pass config=")
        self.config = config if config is not None else default_config()
        self.partner_mode = partner_mode
        self._task_spec = self.SPEC

        self._model = mujoco.MjModel.from_xml_path(str(_ASSETS_DIR / self.ASSET_NAME))
        self._data = mujoco.MjData(self._model)
        self._bind_task_model()
        self._write_model_params()

        self.action_space = spaces.Dict(
            {
                key: spaces.Box(-1.0, 1.0, (self._task_spec.action_dim,), dtype=np.float32)
                for key in AGENT_KEYS
            }
        )
        self.observation_space = spaces.Dict(
            {
                key: spaces.Box(
                    -np.inf,
                    np.inf,
                    (self._obs_size(),),
                    dtype=np.float32,
                )
                for key in AGENT_KEYS
            }
        )

        self._randomization = None
        self._episode_index = 0
        self._step_index = 0
        self._last_commands = {key: np.zeros(self._task_spec.action_dim) for key in AGENT_KEYS}
        self._last_applied = {key: np.zeros(self._task_spec.action_dim) for key in AGENT_KEYS}
        self._delay_queues: dict[str, deque[np.ndarray]] = {}
        self._prev_potential = 0.0

    # --- setup ---------------------------------------------------------------------------
    def _obs_size(self) -> int:
        """Width of one agent's flat observation under the env's partner mode."""
        return obs_size(self._task_spec, self.partner_mode)

    def _write_model_params(self) -> None:
        """Write the config's timing and physics into the MuJoCo model.

        Config as truth: `dt` -> `model.opt.timestep`, `gravity` -> `model.opt.gravity`
        (the config's positive magnitude; the model keeps x/y at zero), then the task hook
        `_write_task_model_params` writes the joint limits, damping and armature. The
        sampled per-episode damping replaces the nominal value on every `reset`.
        """
        cfg = self.config
        self._model.opt.timestep = cfg.dt
        self._model.opt.gravity[:] = (0.0, 0.0, -cfg.gravity)
        self._write_task_model_params()

    # --- gymnasium API -------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, float | bool]]:
        """Start a new episode: sample the private randomness and place the model.

        Args:
            seed: Episode seed. `None` uses `config.seed + <episode index>`, so consecutive
                resets without an explicit seed still get fresh randomness.
            options: Unused (gymnasium compatibility).

        Returns:
            `(obs, info)` with the observation dict and the initial diagnostics.
        """
        del options
        episode_seed = self.config.seed + self._episode_index if seed is None else int(seed)
        self._episode_index += 1
        super().reset(seed=episode_seed)
        randomization = self._sample_randomization(episode_seed)
        self._randomization = randomization
        self._step_index = 0
        action_dim = self._task_spec.action_dim
        self._last_commands = {key: np.zeros(action_dim) for key in AGENT_KEYS}
        self._last_applied = {key: np.zeros(action_dim) for key in AGENT_KEYS}
        self._delay_queues = {
            key: deque(
                [np.zeros(action_dim) for _ in range(randomization.agents[index].latency_steps)]
            )
            for index, key in enumerate(AGENT_KEYS)
        }

        mujoco.mj_resetData(self._model, self._data)
        self._write_sampled_randomization(randomization)
        self._reset_task_state(randomization)
        mujoco.mj_forward(self._model, self._data)
        self._prev_potential = self._task_potential()
        return self._observation(), self._info(succeeded=False)

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, float | bool]]:
        """Apply one control step: commands (latency/start delay), physics, reward, done.

        Args:
            action: `{"left": (action_dim,), "right": (action_dim,)}` normalized one-sided
                commands in `[-1, 1]`.

        Returns:
            `(obs, reward, terminated, truncated, info)`. The reward is `failure_penalty`
            on the step that first crosses the failure bounds, `success_bonus -
            step_penalty` on the success step and `-step_penalty` otherwise (plus the
            potential-based shaping delta on non-terminal steps).
        """
        if self._randomization is None:
            raise RuntimeError("call reset() before step()")
        randomization = self._randomization
        commands = self._parse_action(action)
        self._last_commands = commands

        applied: dict[str, np.ndarray] = {}
        for index, key in enumerate(AGENT_KEYS):
            agent = randomization.agents[index]
            queued = delayed_command(self._delay_queues[key], commands[key])
            vector = np.asarray(queued, dtype=np.float64).reshape(-1)
            if in_start_delay(agent, self._step_index):
                vector = np.zeros(self._task_spec.action_dim)
            # The observation reports the EFFECTIVE (applied) command, in both the own block
            # and the privileged partner block: the oracle then sees a partner whose
            # actuator is currently suppressed by its start delay.
            self._last_applied[key] = vector.copy()
            applied[key] = vector

        physical = self._physical_commands(applied)
        wind_force = (
            randomization.wind_force
            if wind_active_on_step(randomization, self._step_index)
            else 0.0
        )
        self._apply_task_actions(physical, wind_force)
        for substep in range(self.config.control_every):
            mujoco.mj_step(self._model, self._data)
            self._on_substep(substep)
        self._step_index += 1
        self._after_step()

        succeeded = self._success()
        failed = self._failure()
        cfg = self.config
        reward = team_reward(
            succeeded=succeeded,
            failed=failed,
            success_bonus=cfg.reward.success_bonus,
            step_penalty=cfg.reward.step_penalty,
            failure_penalty=cfg.reward.failure_penalty,
        )
        potential_now = self._task_potential()
        if not succeeded and not failed:
            reward += potential_shaping(
                potential_now=potential_now, potential_prev=self._prev_potential
            )
        self._prev_potential = potential_now
        terminated = succeeded or failed
        truncated = (not terminated) and self._step_index >= cfg.max_control_steps
        return self._observation(), reward, terminated, truncated, self._info(succeeded)

    # --- state ---------------------------------------------------------------------------
    @property
    def randomization(self) -> EpisodeRandomization | None:
        """The episode's sampled private randomness (None before the first `reset`).

        Read-only accessor kept for the scripted oracle: it reads its own agents' latency
        from here, exactly like a policy reads its own private parameters from its
        observation.
        """
        return self._randomization

    def state(self) -> dict[str, float]:
        """The task's kinematic state plus the step index (the scripted oracle's input)."""
        return {"step": float(self._step_index), **self._task_state()}

    def pending_commands(self) -> dict[str, tuple[np.ndarray, ...]]:
        """The agents' in-flight commands, oldest first (a centralised script's input).

        Each agent's list has exactly its sampled `latency_steps` entries: entry `k` is the
        command that will reach the physics on control step `k` from now, so a centralised
        model-based controller can predict the delayed state exactly (`CartPoleOracle`'s
        Smith predictor). A single agent knows at most its own queue; the scripted oracle
        may read both because it is a centralised validation controller, not a policy.

        Returns:
            `{agent: (command_0, command_1, ...)}` with copy of every queued vector.

        Raises:
            RuntimeError: if the env has not been reset.
        """
        if self._randomization is None:
            raise RuntimeError("call reset() before reading the pending commands")
        return {
            key: tuple(np.array(command, dtype=np.float64, copy=True) for command in queue)
            for key, queue in self._delay_queues.items()
        }

    # --- helpers -------------------------------------------------------------------------
    def _physical_commands(self, applied: dict[str, np.ndarray]) -> dict[str, float]:
        """Map the applied normalized commands to the task's signed physical commands.

        One-sided actuation: the normalized action is clipped to `[0, 1]` (a negative value
        means "no force"), scaled by the agent's config authority and its private
        gain/authority-scale, then given the agent's physical sign (`AGENT_SIGNS`). This is
        the scalar form of the suite's one-sided actuators; a future task with a wider
        action can override this hook.

        Args:
            applied: The post-latency, post-start-delay normalized commands per agent.

        Returns:
            `{agent: signed physical command}` (N for the cart-pole, Nm for the pendulum).
        """
        if self._task_spec.action_dim != 1:
            raise NotImplementedError(
                "the default one-sided scaling is scalar; override _physical_commands for "
                f"action_dim={self._task_spec.action_dim}"
            )
        if self._randomization is None:
            raise RuntimeError("call reset() before computing physical commands")
        physical: dict[str, float] = {}
        for index, key in enumerate(AGENT_KEYS):
            agent = self._randomization.agents[index]
            raw = max(float(applied[key][0]), 0.0)
            scale = self.config.agent_authority(index) * agent.gain * agent.limit_scale
            physical[key] = self.AGENT_SIGNS[index] * raw * scale
        return physical

    def _parse_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Validate the action dict and clip each command to the `[-1, 1]` box."""
        action_dim = self._task_spec.action_dim
        if not isinstance(action, dict):
            raise TypeError(f"action must be a dict with keys {AGENT_KEYS}, got {type(action)}")
        if set(action) != set(AGENT_KEYS):
            raise ValueError(f"action keys must be exactly {AGENT_KEYS}, got {sorted(action)}")
        parsed: dict[str, np.ndarray] = {}
        for key in AGENT_KEYS:
            vector = np.asarray(action[key], dtype=np.float64).reshape(-1)
            if vector.shape != (action_dim,):
                raise ValueError(
                    f"action[{key!r}] must have shape ({action_dim},), got {vector.shape}"
                )
            parsed[key] = np.clip(vector, -1.0, 1.0)
        return parsed

    def _values(self) -> dict[str, float]:
        """The role-tagged `{field: value}` facts `build_observation` flattens.

        Collected once per step for BOTH agents: the applied and commanded commands and the
        private parameters per role, the task's shared state, its goal and the time
        fraction. Partner keys are always present; `build_observation` decides by partner
        mode whether they reach a vector.
        """
        if self._randomization is None:
            raise RuntimeError("call reset() before building an observation")
        values: dict[str, float] = {}
        for index, key in enumerate(AGENT_KEYS):
            agent = self._randomization.agents[index]
            values.update(zip(action_value_fields(key, self._task_spec), self._last_applied[key]))
            values.update(zip(command_value_fields(key, self._task_spec), self._last_commands[key]))
            values.update(
                zip(param_value_fields(key), (agent.latency_s, agent.gain, agent.limit_scale))
            )
            values.update(zip(privileged_value_fields(key), (agent.start_delay_s,)))
        values.update(zip(self._task_spec.shared_fields, self._shared_state_values()))
        values.update(zip(self._task_spec.goal_fields, self._goal_values()))
        values["time_fraction"] = self._step_index / self.config.max_control_steps
        return {name: float(value) for name, value in values.items()}

    def _observation(self) -> dict[str, np.ndarray]:
        """Both agents' flat observation vectors, from the pure `build_observation`."""
        values = self._values()
        return {
            key: build_observation(values, key, self._task_spec, self.partner_mode)
            for key in AGENT_KEYS
        }

    def _info(self, succeeded: bool) -> dict[str, float | bool]:
        """The episode info dict: success flag plus the task's diagnostics."""
        return {"success": bool(succeeded), **self._diagnostics()}

    def _diagnostics(self) -> dict[str, float]:
        """Human/CLI-readable task numbers of the current state (info dict)."""
        raise NotImplementedError

    def _task_potential(self) -> float:
        """The current bounded shaping potential `Φ(s)` (see `_coop_logic`)."""
        raise NotImplementedError

    def _success(self) -> bool:
        """Whether the current state satisfies the task's success rule."""
        raise NotImplementedError

    def _failure(self) -> bool:
        """Whether the current state crosses the task's failure bounds."""
        raise NotImplementedError

    def _after_step(self) -> None:
        """Optional per-step tracker update (the pendulum's zero-crossing counter)."""
        return

    def _on_substep(self, substep: int) -> None:
        """Optional per-physics-substep hook, called after every `mj_step`.

        The toss-catch task uses it for smooth kinematic paddle motion (interpolating the
        paddle pose across the control step, which also avoids teleport-tunnelling) and for
        collision sampling that must not miss a short contact inside one control step.

        Args:
            substep: Zero-based index of the physics substep that just ran.
        """
        return

    def _sample_randomization(self, episode_seed: int) -> EpisodeRandomization:
        """Sample the episode's private randomness (the suite sampler by default).

        The toss-catch task overrides it with `sample_toss_randomization`, whose record is a
        `TossEpisodeRandomization` (a superset of `EpisodeRandomization`); every other task
        keeps the suite's deterministic sampler unchanged.

        Args:
            episode_seed: The episode seed.

        Returns:
            The immutable randomisation record of the episode.
        """
        return sample_episode_randomization(
            self.config.randomization,
            control_dt=self.config.control_dt,
            max_control_steps=self.config.max_control_steps,
            seed=episode_seed,
        )

    def _write_task_model_params(self) -> None:
        """Write the task's joint limits/damping/armature into the model (subclass hook)."""
        raise NotImplementedError

    def _bind_task_model(self) -> None:
        """Resolve the task's joint/body ids after the model is loaded (subclass hook)."""
        raise NotImplementedError

    def _write_sampled_randomization(self, randomization) -> None:
        """Write the episode's sampled hidden physical parameter (subclass hook)."""
        raise NotImplementedError

    def _reset_task_state(self, randomization) -> None:
        """Place the model's qpos and reset the task's trackers (subclass hook)."""
        raise NotImplementedError

    def _shared_state_values(self) -> tuple[float, ...]:
        """The task's shared-state numbers, in `SPEC.shared_fields` order (subclass hook)."""
        raise NotImplementedError

    def _goal_values(self) -> tuple[float, ...]:
        """The task's goal numbers, in `SPEC.goal_fields` order (subclass hook)."""
        raise NotImplementedError

    def _task_state(self) -> dict[str, float]:
        """The task's kinematic state for the oracle (subclass hook)."""
        raise NotImplementedError

    def _apply_task_actions(self, physical: dict[str, float], wind_force: float) -> None:
        """Write the physical commands (and the optional perturbation) into `xfrc_applied`."""
        raise NotImplementedError


def _angle_deg(angle: float) -> float:
    """Absolute angle in degrees (shared diagnostic helper)."""
    return math.degrees(abs(angle))
