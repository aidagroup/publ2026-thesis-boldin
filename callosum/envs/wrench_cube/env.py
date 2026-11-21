"""CubeWrench-v0: the reduced wrench-level cube environment (MuJoCo, CPU).

The environment of the thesis main line (PLAN.md "Среда CubeWrench-v0"): a floating cube
(`core`, a 5.7 cm box) with one active layer on a limited hinge (`layer`) and TWO
decentralised agents, each commanding a raw 6D wrench applied through
`mujoco.mjData.xfrc_applied` -- the holder on the core, the rotator on the layer. There
are no actuators, no equalities and no contacts: policy -> wrench -> physics is the whole
dynamic, so the coordination problem is the only thing between the agents and success.

Per control step (`control_every` physics substeps): both commands go through the agent's
private latency queue, gain / authority scale and failure injections (start delay,
slip), the impulse disturbance is added on its step, the physics runs, and the observation
/ reward / termination are computed from the resulting state:

* observation: one flat `spaces.Box` per agent with the named-field layout of
  `callosum.envs._wrench_logic` (own wrench + own private parameters + shared cube state
  + goal/time, plus the partner block under `partner_mode="oracle"`);
* reward: the shared team reward (`success_bonus` on the success step, `-step_penalty`
  every step), see `_wrench_logic.team_reward`;
* termination: success (`success`) or failure (`failure`); truncation at
  `max_control_steps`.

The config (`callosum.configs.wrench_cube.WrenchCubeConfig`) is the source of truth for
the timing, the physics parameters written into `model.dof_*`, the authority, the
tolerances and the randomisation. The XML (`assets/cube_wrench.xml`) carries only the
defaults.

The scripted oracle for environment validation lives in `oracle.py`; the local smoke
check is `scripts/smoke_wrench_cube.py`. Only one env is supported on purpose -- the
trainer (phase 2) does the vectorisation.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import ClassVar

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces

from callosum.configs.wrench_cube import WrenchCubeConfig
from callosum.envs._wrench_logic import (
    AGENT_KEYS,
    CUBE_STATE_FIELDS,
    OWN_WRENCH_FIELDS,
    PARTNER_WRENCH_FIELDS,
    EpisodeRandomization,
    failure,
    obs_fields,
    obs_size,
    partner_fields,
    sample_episode_randomization,
    success,
    team_reward,
    validate_partner_mode,
)

_ASSET_PATH = Path(__file__).resolve().parent / "assets" / "cube_wrench.xml"
_HINGE_JOINT = "layer_hinge"
_CORE_BODY = "core"
_LAYER_BODY = "layer"
_WRENCH_SIZE = 6


def _rotation_angle(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
    """Rotation angle (rad) between two unit quaternions in MuJoCo's `(w, x, y, z)` order.

    The relative rotation's scalar part is the dot product of the two quaternions, so
    the angle is `2 * acos(|dot|)` (the absolute value folds the double cover).
    """
    dot = float(np.clip(abs(np.dot(quat_a, quat_b)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


class WrenchCubeEnv(gymnasium.Env):
    """CubeWrench-v0: two agents apply wrenches to a floating two-body cube.

    The env is single-instance on purpose (the trainer will vectorise by looping); the
    `gymnasium` API is `reset(seed=None, options=None) -> (obs, info)` and
    `step(action) -> (obs, reward, terminated, truncated, info)` with `action` a dict
    `{"holder": (6,), "rotator": (6,)}` of normalized wrenches in `[-1, 1]`. A wrench is
    interpreted in the WORLD frame (the `xfrc_applied` convention), so `torque_z` is the
    hinge axis while the core stays near its initial orientation.
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        config: WrenchCubeConfig | None = None,
        partner_mode: str = "none",
        render_mode: str | None = None,
    ) -> None:
        """Build the MuJoCo model and the spaces.

        Args:
            config: Environment config; defaults to `WrenchCubeConfig()`.
            partner_mode: Observation visibility of the partner, `"none"` or `"oracle"`.
            render_mode: Unused -- the reduced sandbox renders nothing; passing a
                non-None value is an error rather than a silent no-op.
        """
        if render_mode is not None:
            raise ValueError("CubeWrench-v0 renders nothing; render_mode must be None")
        validate_partner_mode(partner_mode)
        self.config = config if config is not None else WrenchCubeConfig()
        self.partner_mode = partner_mode

        self._model = mujoco.MjModel.from_xml_path(str(_ASSET_PATH))
        self._write_model_params()
        self._data = mujoco.MjData(self._model)

        core_joint = self._model.joint("core_joint")
        hinge_joint = self._model.joint(_HINGE_JOINT)
        self._core_id = self._model.body(_CORE_BODY).id
        self._layer_id = self._model.body(_LAYER_BODY).id
        self._core_dof0 = int(self._model.jnt_dofadr[core_joint.id])
        self._hinge_qpos = int(self._model.jnt_qposadr[hinge_joint.id])
        self._hinge_dof = int(self._model.jnt_dofadr[hinge_joint.id])

        self.action_space = spaces.Dict(
            {key: spaces.Box(-1.0, 1.0, (_WRENCH_SIZE,), dtype=np.float32) for key in AGENT_KEYS}
        )
        self.observation_space = spaces.Dict(
            {
                key: spaces.Box(-np.inf, np.inf, (obs_size(partner_mode),), dtype=np.float32)
                for key in AGENT_KEYS
            }
        )

        self.randomization: EpisodeRandomization | None = None
        """The episode's sampled private randomness (None before the first `reset`)."""
        self._episode_index = 0
        self._step_index = 0
        self._last_commands = {key: np.zeros(_WRENCH_SIZE) for key in AGENT_KEYS}
        self._delay_queues: dict[str, deque[np.ndarray]] = {}
        self._initial_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._initial_pos = np.zeros(3)

    # --- setup ---------------------------------------------------------------------------
    def _write_model_params(self) -> None:
        """Write the config's timing and DOF parameters into the model (config is truth)."""
        cfg = self.config
        self._model.opt.timestep = cfg.dt
        core_joint = self._model.joint("core_joint")
        hinge_joint = self._model.joint(_HINGE_JOINT)
        core_dof0 = int(self._model.jnt_dofadr[core_joint.id])
        hinge_dof = int(self._model.jnt_dofadr[hinge_joint.id])
        self._model.dof_damping[core_dof0 : core_dof0 + 3] = cfg.core_lin_damping
        self._model.dof_damping[core_dof0 + 3 : core_dof0 + 6] = cfg.core_ang_damping
        self._model.dof_damping[hinge_dof] = cfg.hinge_damping
        self._model.dof_frictionloss[hinge_dof] = cfg.hinge_frictionloss
        self._model.dof_armature[hinge_dof] = cfg.hinge_armature

    def _authority(self, agent_idx: int) -> np.ndarray:
        """The agent's base action scale (force xyz, torque xyz) from the config."""
        authority = (
            self.config.holder_authority if agent_idx == 0 else self.config.rotator_authority
        )
        return np.asarray(authority, dtype=np.float64)

    # --- gymnasium API -------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, float | bool]]:
        """Start a new episode: sample the private randomness and place the cube.

        Args:
            seed: Episode seed. `None` uses `config.seed + <episode index>`, so
                consecutive resets without an explicit seed still get fresh randomness.
            options: Unused (gymnasium compatibility).

        Returns:
            `(obs, info)` with the observation dict and the initial diagnostics.
        """
        del options
        episode_seed = self.config.seed + self._episode_index if seed is None else int(seed)
        self._episode_index += 1
        super().reset(seed=episode_seed)
        randomization = sample_episode_randomization(self.config, episode_seed)
        self.randomization = randomization
        self._step_index = 0
        self._last_commands = {key: np.zeros(_WRENCH_SIZE) for key in AGENT_KEYS}
        self._delay_queues = {
            key: deque(
                [np.zeros(_WRENCH_SIZE) for _ in range(randomization.agents[idx].latency_steps)]
            )
            for idx, key in enumerate(AGENT_KEYS)
        }

        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[self._hinge_qpos] = randomization.layer_initial_angle
        mujoco.mj_forward(self._model, self._data)
        self._initial_pos = self._data.qpos[0:3].copy()
        self._initial_quat = self._data.qpos[3:7].copy()
        return self._observation(), self._info(succeeded=False)

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, float | bool]]:
        """Apply one control step: wrenches (latency/injections), physics, reward, done.

        Args:
            action: `{"holder": (6,), "rotator": (6,)}` normalized wrenches in `[-1, 1]`.

        Returns:
            `(obs, reward, terminated, truncated, info)`.
        """
        if self.randomization is None:
            raise RuntimeError("call reset() before step()")
        randomization = self.randomization
        commands = self._parse_action(action)
        self._last_commands = commands

        wrenches: dict[str, np.ndarray] = {}
        for agent_idx, key in enumerate(AGENT_KEYS):
            agent = randomization.agents[agent_idx]
            queue = self._delay_queues[key]
            queue.append(commands[key])
            applied = queue.popleft().copy()
            if (
                self._step_index < agent.start_delay_steps
                or agent.slip_active
                and (agent.slip_step <= self._step_index < agent.slip_step + agent.slip_steps)
            ):
                applied = np.zeros(_WRENCH_SIZE)
            wrenches[key] = applied * self._authority(agent_idx) * agent.limit_scale * agent.gain

        self._data.xfrc_applied[:] = 0.0
        self._data.xfrc_applied[self._core_id] = wrenches["holder"]
        self._data.xfrc_applied[self._layer_id] = wrenches["rotator"]
        if randomization.impulse_active and self._step_index == randomization.impulse_step:
            self._data.xfrc_applied[self._core_id, :3] += np.asarray(randomization.impulse_force)
        for _ in range(self.config.control_every):
            mujoco.mj_step(self._model, self._data)
        self._step_index += 1

        succeeded = self._success()
        reward = team_reward(
            succeeded=succeeded,
            success_bonus=self.config.success_bonus,
            step_penalty=self.config.step_penalty,
        )
        terminated = succeeded or self._failure()
        truncated = (not terminated) and self._step_index >= self.config.max_control_steps
        return self._observation(), reward, terminated, truncated, self._info(succeeded)

    # --- helpers -------------------------------------------------------------------------
    def state(self) -> dict[str, np.ndarray | float]:
        """The cube's current kinematic state (the scripted oracle's input).

        Returns:
            Copies of `core_pos` (3,), `core_quat` (4, wxyz), `core_lin_vel` (3,),
            `core_ang_vel` (3,), `layer_angle` (float), `layer_ang_vel` (float) and the
            current `step` index. No private randomisation is included beyond what an
            agent knows about itself.
        """
        data = self._data
        return {
            "core_pos": data.qpos[0:3].copy(),
            "core_quat": data.qpos[3:7].copy(),
            "core_lin_vel": data.qvel[0:3].copy(),
            "core_ang_vel": data.qvel[3:6].copy(),
            "layer_angle": float(data.qpos[self._hinge_qpos]),
            "layer_ang_vel": float(data.qvel[self._hinge_dof]),
            "step": self._step_index,
        }

    def _parse_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Validate the action dict and clip each wrench to the `[-1, 1]` box."""
        if not isinstance(action, dict):
            raise TypeError(f"action must be a dict with keys {AGENT_KEYS}, got {type(action)}")
        if set(action) != set(AGENT_KEYS):
            raise ValueError(f"action keys must be exactly {AGENT_KEYS}, got {sorted(action)}")
        parsed: dict[str, np.ndarray] = {}
        for key in AGENT_KEYS:
            vector = np.asarray(action[key], dtype=np.float64).reshape(-1)
            if vector.shape != (_WRENCH_SIZE,):
                raise ValueError(
                    f"action[{key!r}] must have shape ({_WRENCH_SIZE},), got {vector.shape}"
                )
            parsed[key] = np.clip(vector, -1.0, 1.0)
        return parsed

    def _diagnostics(self) -> dict[str, float]:
        """The current angle / drift / time numbers of the episode."""
        data = self._data
        angle = float(data.qpos[self._hinge_qpos])
        drift_pos = float(np.linalg.norm(data.qpos[0:3] - self._initial_pos))
        drift_rot = float(_rotation_angle(self._initial_quat, data.qpos[3:7]))
        return {
            "angle_deg": math.degrees(angle),
            "drift_pos": drift_pos,
            "drift_rot": drift_rot,
            "time": self._step_index * self.config.control_dt,
        }

    def _success(self) -> bool:
        """Whether the current state satisfies the success conjunction."""
        cfg = self.config
        data = self._data
        core_pos = data.qpos[0:3]
        core_ang_vel = data.qvel[3:6]
        layer_ang_vel = float(data.qvel[self._hinge_dof])
        ang_speed = float(np.linalg.norm(np.append(core_ang_vel, layer_ang_vel)))
        return success(
            angle=float(data.qpos[self._hinge_qpos]),
            drift_pos=float(np.linalg.norm(core_pos - self._initial_pos)),
            drift_rot=float(_rotation_angle(self._initial_quat, data.qpos[3:7])),
            goal_angle=cfg.goal_angle,
            angle_tol=cfg.success_angle_tol,
            pos_tol=cfg.success_pos_tol,
            rot_tol=cfg.success_rot_tol,
            lin_speed=float(np.linalg.norm(data.qvel[0:3])),
            ang_speed=ang_speed,
            lin_speed_tol=cfg.success_lin_speed_tol,
            ang_speed_tol=cfg.success_ang_speed_tol,
        )

    def _failure(self) -> bool:
        """Whether the current state crosses the failure bounds."""
        cfg = self.config
        data = self._data
        return failure(
            float(np.linalg.norm(data.qpos[0:3] - self._initial_pos)),
            float(_rotation_angle(self._initial_quat, data.qpos[3:7])),
            pos_drift=cfg.failure_pos_drift,
            rot_drift=cfg.failure_rot_drift,
        )

    def _info(self, succeeded: bool) -> dict[str, float | bool]:
        """The episode info dict: success flag plus the angle/drift/time diagnostics."""
        return {"success": bool(succeeded), **self._diagnostics()}

    def _cube_state_values(self) -> tuple[float, ...]:
        """The shared cube-state numbers, in `CUBE_STATE_FIELDS` order."""
        data = self._data
        values = (
            *data.qpos[0:3],
            *data.qpos[3:7],
            *data.qvel[0:3],
            *data.qvel[3:6],
            float(data.qpos[self._hinge_qpos]),
            float(data.qvel[self._hinge_dof]),
        )
        return tuple(float(value) for value in values)

    def _obs_values(self, agent_idx: int) -> dict[str, float]:
        """The `{field_name: value}` dict of `agent_idx`'s observation under the mode."""
        if self.randomization is None:
            raise RuntimeError("call reset() before building an observation")
        key = AGENT_KEYS[agent_idx]
        partner_idx = 1 - agent_idx
        partner_key = AGENT_KEYS[partner_idx]
        agent = self.randomization.agents[agent_idx]
        partner = self.randomization.agents[partner_idx]

        values: dict[str, float] = {}
        values.update(zip(OWN_WRENCH_FIELDS, self._last_commands[key]))
        values["own_latency"] = agent.latency_s
        values["own_gain"] = agent.gain
        values["own_limit_scale"] = agent.limit_scale
        values.update(zip(CUBE_STATE_FIELDS, self._cube_state_values()))
        values["goal_angle"] = self.config.goal_angle
        values["time_fraction"] = self._step_index / self.config.max_control_steps
        if partner_fields(self.partner_mode):
            values.update(zip(PARTNER_WRENCH_FIELDS, self._last_commands[partner_key]))
            values["partner_latency"] = partner.latency_s
            values["partner_gain"] = partner.gain
            values["partner_limit_scale"] = partner.limit_scale
        return {name: float(value) for name, value in values.items()}

    def _observation(self) -> dict[str, np.ndarray]:
        """Both agents' flat observation vectors, in the `obs_fields` order."""
        fields = obs_fields(self.partner_mode)
        return {
            key: np.asarray(
                [self._obs_values(agent_idx)[name] for name in fields], dtype=np.float32
            )
            for agent_idx, key in enumerate(AGENT_KEYS)
        }
