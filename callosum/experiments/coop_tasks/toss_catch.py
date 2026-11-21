"""TossCatch-v0: two tilted position-controlled paddles catch and return a free ball.

The task (`PLAN.md` "Two-robot task suite"): the `left` agent releases a hidden-impulse
throw toward the `right` paddle, the receiver must place its paddle under the ball and tilt
it to toss the ball back, and the team must complete `n_passes` alternating catches without
the ball dropping below the floor or leaving the court. It is the suite's contact/timing
task: unlike the cart-pole and the pendulum there is no shared joint, so coordination is
pure anticipation -- each agent must move its own paddle to the predicted intercept while its
action is delayed, and the partner's in-flight command is what the `oracle` partner block
makes visible.

The paddles are 0.60 m square plates on MOCAP bodies, i.e. kinematic position-controlled
bodies (the ideal limit of a stiff position servo; the task design allows it). An action is
3 normalized values `(dx, dz, dtheta)`: a bounded pose increment for the paddle centre and a
bounded tilt increment. The env moves the paddles smoothly over the control step (interpolated
on every physics substep, so a fast paddle does not teleport), and the ball-racket bounce is
the physical MuJoCo contact with a per-episode sampled restitution and friction. The paddle
tilt maps the ball's vertical momentum into horizontal momentum, which is what lets a static
paddle return the ball; the `TossOracle` holds the nominal aiming tilt and positions the
paddle, and a learned policy can also vary the tilt.

The fixed part of the harness (MJCF loading, latency queues, start-delay suppression,
observation assembly, team reward) lives in `CoopTaskEnv`; this module declares the ids, the
config-as-truth model parameters, the paddle actuation and its trackers. Episode randomness
is the extended `TossEpisodeRandomization` (hidden throw impulse and release delay, sampled
contact restitution/friction, plus the suite's per-agent latency/gain/start delay).
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from callosum.experiments.coop_tasks.base import CoopTaskEnv
from callosum.experiments.coop_tasks.config import TossConfig
from callosum.experiments.coop_tasks.logic import (
    AGENT_KEYS,
    TOSS_SPEC,
    CoopTaskSpec,
    TossEpisodeRandomization,
    paddle_tilt,
    sample_toss_randomization,
    toss_floor_failure,
    toss_pass_count,
    toss_potential,
    toss_success,
)

_BALL_BODY = "ball"
_BALL_JOINT = "ball_free"
_PAD_BODIES = {"left": "pad_left", "right": "pad_right"}
_PAD_GEOMS = {"left": "pad_left_geom", "right": "pad_right_geom"}
_BALL_GEOM = "ball_geom"


def _yaw_quat(tilt: float) -> np.ndarray:
    """Quaternion `(w, x, y, z)` of a rotation about world +y by `tilt` (paddle tilt)."""
    return np.array([math.cos(tilt / 2.0), 0.0, math.sin(tilt / 2.0), 0.0], dtype=np.float64)


class TossCatchEnv(CoopTaskEnv):
    """TossCatch-v0: the two tilted paddles and the free ball."""

    TASK_NAME: ClassVar[str] = "toss"
    ASSET_NAME: ClassVar[str] = "toss_catch.xml"
    SPEC: ClassVar[CoopTaskSpec] = TOSS_SPEC
    # The paddles are symmetric position-controlled bodies, so the physical signs are equal;
    # `_physical_commands` overrides the base's one-sided split for this task.
    AGENT_SIGNS: ClassVar[tuple[float, float]] = (1.0, 1.0)
    DEFAULT_CONFIG: ClassVar[type[TossConfig]] = TossConfig

    def _bind_task_model(self) -> None:
        """Resolve the ball/paddle bodies and geoms the task uses."""
        model = self._model
        self._ball_body = int(model.body(_BALL_BODY).id)
        self._ball_geom = int(model.geom(_BALL_GEOM).id)
        self._ball_joint = int(model.joint(_BALL_JOINT).id)
        self._ball_qpos = int(model.jnt_qposadr[self._ball_joint])
        self._ball_dof = int(model.jnt_dofadr[self._ball_joint])
        self._pad_geom = {side: int(model.geom(name).id) for side, name in _PAD_GEOMS.items()}
        self._pad_body = {side: int(model.body(name).id) for side, name in _PAD_BODIES.items()}
        self._pad_mocap = {
            side: int(model.body(name).mocapid[0]) for side, name in _PAD_BODIES.items()
        }
        # The contact offset along the paddle normal at which the ball touches the paddle
        # face: ball radius + half the paddle thickness. Read from the XML-owned geometry.
        ball_radius = float(model.geom_size[self._ball_geom][0])
        pad_thickness = float(model.geom_size[self._pad_geom["left"]][2])
        self._contact_offset = ball_radius + pad_thickness

    def _write_task_model_params(self) -> None:
        """Write the contact time constant the config implies (damping ratio is sampled)."""
        # The nominal solref time constant is XML-owned; only the sampled damping ratio and
        # friction are written per episode in `_write_sampled_randomization`.
        return

    def paddle_ready_x(self) -> float:
        """Paddle centre x of the ready pose (near the inner edge of each half).

        Near the court centre the ready pose is out of a typical incoming ball's path, so an
        agent that never moves misses the ball (the "only-left" baseline), while the
        controlled paddle can still reach the first intercept in time.
        """
        cfg = self._config()
        return cfg.pad_inner_x + 0.10

    def paddle_receive_x(self) -> float:
        """Court x of the receiving paddle's pre-throw anticipation pose (mirrored)."""
        return self._config().pad_anticipation_x

    def paddle_ready_tilt(self, side: str) -> float:
        """Ready-pose tilt: the nominal aiming tilt (the paddle is ready to return)."""
        return paddle_tilt(side, self._config().pad_tilt)

    def paddle_half_extents(self) -> tuple[float, float]:
        """The paddle's half extents `(x, z)` in its own frame (XML-owned geometry)."""
        size = self._model.geom_size[self._pad_geom["left"]]
        return float(size[0]), float(size[2])

    def contact_offset(self) -> float:
        """Ball-centre to paddle-centre distance along the face normal at contact (m)."""
        return self._contact_offset

    def _sample_randomization(self, episode_seed: int) -> TossEpisodeRandomization:
        """Sample the toss-catch episode randomness (hidden impulse/contact + suite fields)."""
        return sample_toss_randomization(
            self.config.randomization,
            control_dt=self.config.control_dt,
            max_control_steps=self.config.max_control_steps,
            seed=episode_seed,
        )

    def _write_sampled_randomization(self, randomization: TossEpisodeRandomization) -> None:
        """Write the sampled contact restitution/friction into the model's geoms."""
        model = self._model
        geoms = (self._ball_geom, *self._pad_geom.values())
        for geom in geoms:
            model.geom_solref[geom, 1] = randomization.contact_dampratio
            model.geom_friction[geom, 0] = randomization.contact_friction

    def _reset_task_state(self, randomization: TossEpisodeRandomization) -> None:
        """Release the ball at the sampled hidden throw pose and place the paddles."""
        cfg = self._config()
        data = self._data
        launch_x, launch_z = randomization.launch_position
        data.qpos[self._ball_qpos : self._ball_qpos + 3] = (launch_x, 0.0, launch_z)
        data.qpos[self._ball_qpos + 3 : self._ball_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[self._ball_dof : self._ball_dof + 3] = 0.0
        data.qvel[self._ball_dof + 3 : self._ball_dof + 6] = 0.0
        self._launch_velocity = np.array(
            [randomization.launch_velocity[0], 0.0, randomization.launch_velocity[1]],
            dtype=np.float64,
        )
        self._release_step = int(randomization.release_delay_steps)
        self._passes = 0
        self._last_pass_side: str | None = None
        self._contact_side = {"left": False, "right": False}
        self._pad_pose: dict[str, tuple[float, float, float]] = {}
        self._pad_start_pose: dict[str, tuple[float, float, float]] = {}
        self._pad_velocity: dict[str, tuple[float, float, float]] = {}
        self._pad_target_pose: dict[str, tuple[float, float, float]] = {}
        mid_x = self.paddle_ready_x()
        for index, side in enumerate(AGENT_KEYS):
            pose = (
                -mid_x if side == "left" else mid_x,
                cfg.pad_start_z,
                self.paddle_ready_tilt(side),
            )
            self._set_pad_pose(side, pose)
            self._pad_pose[side] = pose
            self._pad_start_pose[side] = pose
            self._pad_target_pose[side] = pose
            self._pad_velocity[side] = (0.0, 0.0, 0.0)

    def _apply_task_actions(self, physical: dict[str, np.ndarray], wind_force: float) -> None:
        """Apply the paddles' pose increments (and the hidden throw impulse on its step)."""
        del wind_force  # the task has no external perturbation channel
        cfg = self._config()
        randomization = self._randomization
        if randomization is None:  # pragma: no cover - the base guarantees a reset
            raise RuntimeError("call reset() before step()")
        if self._step_index == self._release_step:
            self._data.qvel[self._ball_dof : self._ball_dof + 3] = self._launch_velocity
        for index, side in enumerate(AGENT_KEYS):
            agent = randomization.agents[index]
            authority = agent.gain * agent.limit_scale
            max_step = cfg.pad_speed * cfg.control_dt * authority
            max_dtilt = cfg.pad_tilt_rate * cfg.control_dt * authority
            command = np.asarray(physical[side], dtype=np.float64).reshape(-1)
            start = self._pad_pose[side]
            x_range = cfg.paddle_x_range(side)
            target = (
                float(np.clip(start[0] + command[0] * max_step, *x_range)),
                float(np.clip(start[1] + command[1] * max_step, cfg.pad_min_z, cfg.pad_max_z)),
                float(
                    np.clip(
                        start[2] + command[2] * max_dtilt,
                        -cfg.pad_tilt_limit,
                        cfg.pad_tilt_limit,
                    )
                ),
            )
            self._pad_start_pose[side] = start
            self._pad_target_pose[side] = target
        self._apply_interpolated_paddles(1.0 / cfg.control_every)

    def _on_substep(self, substep: int) -> None:
        """Move the paddles toward their target pose and sample the ball/paddle contacts."""
        cfg = self._config()
        alpha = min((substep + 2) / cfg.control_every, 1.0)
        self._apply_interpolated_paddles(alpha)
        for side in AGENT_KEYS:
            contact = self._pad_contacts_ball(side)
            if contact and not self._contact_side[side]:
                self._passes = toss_pass_count(self._passes, self._last_pass_side, side)
                self._last_pass_side = side
            self._contact_side[side] = contact

    def _apply_interpolated_paddles(self, alpha: float) -> None:
        """Set every paddle's mocap pose to `start + alpha * (target - start)`."""
        for side in AGENT_KEYS:
            start = self._pad_start_pose[side]
            target = self._pad_target_pose[side]
            pose = tuple(
                start[index] + alpha * (target[index] - start[index]) for index in range(3)
            )
            self._set_pad_pose(side, pose)
            self._pad_pose[side] = pose

    def _set_pad_pose(self, side: str, pose: tuple[float, float, float]) -> None:
        """Write one paddle's pose into the model's mocap position/quaternion."""
        x, z, tilt = pose
        mocap = self._pad_mocap[side]
        self._data.mocap_pos[mocap] = (x, 0.0, z)
        self._data.mocap_quat[mocap] = _yaw_quat(tilt)

    def _pad_contacts_ball(self, side: str) -> bool:
        """Whether the ball geom currently contacts that paddle's geom."""
        pair = {self._pad_geom[side], self._ball_geom}
        return any(
            {int(contact.geom1), int(contact.geom2)} == pair
            for contact in self._data.contact[: self._data.ncon]
        )

    def _after_step(self) -> None:
        """Update the paddles' reported velocities for the observation and the oracle."""
        cfg = self._config()
        for side in AGENT_KEYS:
            start = self._pad_start_pose[side]
            end = self._pad_pose[side]
            self._pad_velocity[side] = (
                (end[0] - start[0]) / cfg.control_dt,
                (end[1] - start[1]) / cfg.control_dt,
                (end[2] - start[2]) / cfg.control_dt,
            )

    def _physical_commands(self, applied: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Symmetric pose-increment commands (the toss-catch task has no one-sided split)."""
        return {
            key: np.asarray(applied[key], dtype=np.float64).reshape(self._task_spec.action_dim)
            for key in AGENT_KEYS
        }

    def _shared_state_values(self) -> tuple[float, ...]:
        """Ball position/velocity and both paddles' pose/velocity, in spec order."""
        data = self._data
        values = [
            float(data.qpos[self._ball_qpos]),
            float(data.qpos[self._ball_qpos + 2]),
            float(data.qvel[self._ball_dof]),
            float(data.qvel[self._ball_dof + 2]),
        ]
        for side in AGENT_KEYS:
            pose = self._pad_pose[side]
            velocity = self._pad_velocity[side]
            values.extend((pose[0], pose[1], pose[2], velocity[0], velocity[1], velocity[2]))
        return tuple(values)

    def _goal_values(self) -> tuple[float, ...]:
        """The pass counter and its target, in spec order."""
        return (float(self._passes), float(self._config().n_passes))

    def _task_state(self) -> dict[str, float]:
        """The toss-catch kinematic state and pass trackers for the scripted oracle."""
        data = self._data
        state = {
            "ball_x": float(data.qpos[self._ball_qpos]),
            "ball_z": float(data.qpos[self._ball_qpos + 2]),
            "ball_vx": float(data.qvel[self._ball_dof]),
            "ball_vz": float(data.qvel[self._ball_dof + 2]),
            "passes": float(self._passes),
            "target_passes": float(self._config().n_passes),
        }
        for side in AGENT_KEYS:
            pose = self._pad_pose[side]
            velocity = self._pad_velocity[side]
            state[f"{side}_pad_x"] = pose[0]
            state[f"{side}_pad_z"] = pose[1]
            state[f"{side}_pad_tilt"] = pose[2]
            state[f"{side}_pad_vx"] = velocity[0]
            state[f"{side}_pad_vz"] = velocity[1]
            state[f"{side}_pad_tilt_rate"] = velocity[2]
            state[f"{side}_contact"] = float(self._contact_side[side])
        return state

    def _diagnostics(self) -> dict[str, float]:
        """Pass progress, the ball's state and the elapsed time."""
        cfg = self._config()
        state = self._task_state()
        return {
            "passes": float(self._passes),
            "target_passes": float(cfg.n_passes),
            "ball_x": state["ball_x"],
            "ball_z": state["ball_z"],
            "ball_speed": math.hypot(state["ball_vx"], state["ball_vz"]),
            "time": self._step_index * cfg.control_dt,
        }

    def _success(self) -> bool:
        """Whether the team completed the required number of alternating passes."""
        return toss_success(self._passes, self._config().n_passes)

    def _failure(self) -> bool:
        """Whether the ball dropped below the floor or left the court."""
        cfg = self._config()
        data = self._data
        return toss_floor_failure(
            float(data.qpos[self._ball_qpos]),
            float(data.qpos[self._ball_qpos + 1]),
            float(data.qpos[self._ball_qpos + 2]),
            floor_z=cfg.floor_z,
            court_x=cfg.court_x,
            court_y=cfg.court_y,
        )

    def _task_potential(self) -> float:
        """The toss-catch pass-progress plus ball-height shaping potential."""
        cfg = self._config()
        return toss_potential(
            passes=self._passes,
            target_passes=cfg.n_passes,
            ball_z=float(self._data.qpos[self._ball_qpos + 2]),
            height_scale=cfg.shaping_height_scale,
            pass_coef=cfg.shaping_pass_coef,
            height_coef=cfg.shaping_height_coef,
        )

    def _config(self) -> TossConfig:
        """The env's config as its concrete task type (the base stores it loosely)."""
        config = self.config
        if not isinstance(config, TossConfig):  # pragma: no cover - construction guard
            raise TypeError(f"TossCatchEnv needs a TossConfig, got {type(config)}")
        return config
