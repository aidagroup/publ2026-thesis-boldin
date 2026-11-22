"""TossCatch-v0: two actuated paddles catch and return a free ball.

The task (`PLAN.md` "Two-robot task suite"): the `left` agent releases a hidden-impulse
throw toward the `right` paddle, the receiver must place its paddle under the ball and tilt
it to toss the ball back, and the team must complete `n_passes` alternating catches without
the ball dropping below the floor or leaving the court. It is the suite's contact/timing
task: unlike the cart-pole and the pendulum there is no shared joint, so coordination is
pure anticipation -- each agent must move its own paddle to the predicted intercept while its
action is delayed, and the partner's in-flight command is what the `oracle` partner block
makes visible.

The paddles are PHYSICAL bodies, not kinematic MOCAP plates. Each one is a 0.60 x 0.60 x
0.06 m, 0.8 kg plate on two prismatic joints (court x and vertical z) and a hinge (the tilt
about world +y), mounted on the world, and driven by three bounded-force position servos
(`assets/toss_catch.xml` owns the geometry, mass/inertia, servo gains and force limits). An
action is 3 normalized values `(dx, dz, dtheta)`: a bounded increment of the servo TARGET
(the same `pad_speed`/`pad_tilt_rate` limits as before). The env writes `data.ctrl`; the
plate physically tracks it, so it accelerates/follows with finite force and is held against
gravity by the servo -- it never teleports, and with a zero command it stays at its target
(the servo's static sag is ~1 mm). The ball/plate bounce is the physical MuJoCo contact with
a per-episode sampled restitution and friction; the plate/plate contact is a stiff contact
shell pair, so the two plates can never pass through or interpenetrate each other -- the
contact is the real constraint, and the env only projects a commanded target pair apart as
an anti-ram safety (see `paddle_model.project_targets_apart`).

The paddle tilt maps the ball's vertical momentum into horizontal momentum, which is what
lets a static paddle return the ball; the `TossOracle` holds the nominal aiming tilt and
positions the paddle, and a learned policy can also vary the tilt.

The fixed part of the harness (MJCF loading, latency queues, start-delay suppression,
observation assembly, team reward) lives in `CoopTaskEnv`; this module declares the ids, the
config-as-truth model parameters, the servo-target actuation and its trackers. Episode
randomness is the extended `TossEpisodeRandomization` (hidden throw impulse and release
delay, sampled contact restitution/friction, plus the suite's per-agent latency/gain/start
delay).
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
from callosum.experiments.coop_tasks.paddle_model import (
    PaddleTarget,
    paddle_pair_geometry,
    project_targets_apart,
    servo_target,
)

_BALL_BODY = "ball"
_BALL_JOINT = "ball_free"
_BALL_GEOM = "ball_geom"
_PAD_BODIES = {"left": "pad_left", "right": "pad_right"}
_PAD_GEOMS = {"left": "pad_left_geom", "right": "pad_right_geom"}
_PAD_SHELLS = {"left": "pad_left_shell", "right": "pad_right_shell"}
_PAD_JOINTS = {
    "left": ("pad_left_x", "pad_left_z", "pad_left_tilt"),
    "right": ("pad_right_x", "pad_right_z", "pad_right_tilt"),
}
_PAD_ACTUATORS = {
    "left": ("pad_left_x_servo", "pad_left_z_servo", "pad_left_tilt_servo"),
    "right": ("pad_right_x_servo", "pad_right_z_servo", "pad_right_tilt_servo"),
}

_READY_X_OFFSET = 0.22
"""Ready-pose offset (m) of each paddle centre from the court centre, added to
`pad_inner_x` (i.e. `paddle_ready_x() = pad_inner_x + _READY_X_OFFSET`).

The plates are 0.6 m wide; at the nominal tilt their projection on x is ~0.237 m half width,
so with the ready pose at `0.05 + 0.22 = 0.27 m` the two idle plates keep a ~0.067 m gap and
the physical contact shells are not in contact at reset. The old MOCAP model used
`pad_inner_x + 0.10 = 0.15 m`, where the two plates interpenetrated by ~0.17 m -- harmless
for kinematic bodies, impossible for rigid ones. The lower ready height
(`TossConfig.pad_start_z = 0.30 -> 0.22`) keeps the idle plate out of the descending ball's
path, so the absent-agent ("only-left") baseline has to miss it (~1% accidental contacts).
"""


class TossCatchEnv(CoopTaskEnv):
    """TossCatch-v0: the two physical servoed paddles and the free ball."""

    TASK_NAME: ClassVar[str] = "toss"
    ASSET_NAME: ClassVar[str] = "toss_catch.xml"
    SPEC: ClassVar[CoopTaskSpec] = TOSS_SPEC
    # The paddles are symmetric position-servo bodies, so the physical signs are equal;
    # `_physical_commands` overrides the base's one-sided split for this task.
    AGENT_SIGNS: ClassVar[tuple[float, float]] = (1.0, 1.0)
    DEFAULT_CONFIG: ClassVar[type[TossConfig]] = TossConfig

    def _bind_task_model(self) -> None:
        """Resolve the ball/paddle bodies, joints, geoms and servos the task uses."""
        model = self._model
        self._ball_body = int(model.body(_BALL_BODY).id)
        self._ball_geom = int(model.geom(_BALL_GEOM).id)
        self._ball_joint = int(model.joint(_BALL_JOINT).id)
        self._ball_qpos = int(model.jnt_qposadr[self._ball_joint])
        self._ball_dof = int(model.jnt_dofadr[self._ball_joint])
        self._pad_geom = {side: int(model.geom(name).id) for side, name in _PAD_GEOMS.items()}
        self._pad_shell = {side: int(model.geom(name).id) for side, name in _PAD_SHELLS.items()}
        self._pad_body = {side: int(model.body(name).id) for side, name in _PAD_BODIES.items()}
        self._pad_joint = {}
        self._pad_qpos = {}
        self._pad_dof = {}
        self._pad_actuator = {}
        for side, joint_names in _PAD_JOINTS.items():
            joints = [int(model.joint(name).id) for name in joint_names]
            self._pad_joint[side] = tuple(joints)
            self._pad_qpos[side] = tuple(int(model.jnt_qposadr[joint]) for joint in joints)
            self._pad_dof[side] = tuple(int(model.jnt_dofadr[joint]) for joint in joints)
            self._pad_actuator[side] = tuple(
                int(model.actuator(name).id) for name in _PAD_ACTUATORS[side]
            )
        if int(model.nmocap) != 0:
            raise RuntimeError(
                "the physical toss_catch.xml must have no MOCAP bodies (the paddles are "
                f"actuated dynamic bodies now), got nmocap={model.nmocap}"
            )
        # The contact offset along the paddle normal at which the ball touches the paddle
        # face: ball radius + half the paddle thickness. Read from the XML-owned geometry.
        ball_radius = float(model.geom_size[self._ball_geom][0])
        pad_thickness = float(model.geom_size[self._pad_geom["left"]][2])
        self._contact_offset = ball_radius + pad_thickness
        self._pad_half_x = float(model.geom_size[self._pad_geom["left"]][0])
        self._pad_half_z = float(model.geom_size[self._pad_geom["left"]][2])

    def _write_task_model_params(self) -> None:
        """Write the config-as-truth joint workspace into the joints and the servo ranges.

        The XML's ranges are placeholders (defaults matching `TossConfig`); the config is the
        source of truth, so a config override moves the joint limit and the position servo's
        legal target range together -- a target can never ask the servo to violate a limit.
        """
        cfg = self._config()
        model = self._model
        z_range = (cfg.pad_min_z, cfg.pad_max_z)
        tilt_range = (-cfg.pad_tilt_limit, cfg.pad_tilt_limit)
        for side in AGENT_KEYS:
            x_range = cfg.paddle_x_range(side)
            ranges = (x_range, z_range, tilt_range)
            for joint, actuator, joint_range in zip(
                self._pad_joint[side], self._pad_actuator[side], ranges
            ):
                model.jnt_range[joint] = joint_range
                model.actuator_ctrlrange[actuator] = joint_range
                model.actuator_ctrllimited[actuator] = 1

    def paddle_ready_x(self) -> float:
        """Paddle centre |x| of the ready pose (out of the partner's way, plates separated).

        The ready pose is far enough from the court centre that the two idle plates do not
        touch (`_READY_X_OFFSET`) and an agent that never moves misses the ball (the
        "only-left" baseline), while the controlled paddle can still reach the first
        intercept in time.
        """
        return self._config().pad_inner_x + _READY_X_OFFSET

    def paddle_receive_x(self) -> float:
        """Court x of the receiving paddle's pre-throw anticipation pose (mirrored)."""
        return self._config().pad_anticipation_x

    def paddle_ready_tilt(self, side: str) -> float:
        """Ready-pose tilt: the nominal aiming tilt (the paddle is ready to return)."""
        return paddle_tilt(side, self._config().pad_tilt)

    def paddle_half_extents(self) -> tuple[float, float]:
        """The paddle's half extents `(x, z)` in its own frame (XML-owned geometry)."""
        return self._pad_half_x, self._pad_half_z

    def paddle_half_thickness(self) -> float:
        """Half thickness (m) of the plate along its face normal (XML-owned geometry)."""
        return self._pad_half_z

    def contact_offset(self) -> float:
        """Ball-centre to paddle-centre distance along the face normal at contact (m)."""
        return self._contact_offset

    def paddle_target(self, side: str) -> tuple[float, float, float]:
        """The paddle's current servo target `(x, z, tilt)` (its commanded pose).

        This is the reference the position servo tracks, not the measured pose: the pose
        increment commands accumulate on the target, so the scripted oracle plans in target
        space while the measured state (`state()`) reports what the plate actually did.
        """
        target = self._pad_target[side]
        return target.x, target.z, target.tilt

    def _sample_randomization(self, episode_seed: int) -> TossEpisodeRandomization:
        """Sample the toss-catch episode randomness (hidden impulse/contact + suite fields)."""
        return sample_toss_randomization(
            self.config.randomization,
            control_dt=self.config.control_dt,
            max_control_steps=self.config.max_control_steps,
            seed=episode_seed,
        )

    def _write_sampled_randomization(self, randomization: TossEpisodeRandomization) -> None:
        """Write the sampled contact restitution/friction into the ball/paddle geoms.

        Only the ball/plate contact geoms are randomised (the contact shells that implement
        the stiff plate/plate collision keep their XML contact parameters: the plates'
        mutual contact is a hard constraint, not a tunable bounce).
        """
        model = self._model
        geoms = (self._ball_geom, *self._pad_geom.values())
        for geom in geoms:
            model.geom_solref[geom, 1] = randomization.contact_dampratio
            model.geom_friction[geom, 0] = randomization.contact_friction

    def _reset_task_state(self, randomization: TossEpisodeRandomization) -> None:
        """Release the ball at the sampled hidden throw pose and place the paddles at ready."""
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
        self._last_pass_step = -1
        """Control step of the last COUNTED pass (`-1` before the first one)."""
        self._contact_side = {"left": False, "right": False}
        self._pad_pose: dict[str, tuple[float, float, float]] = {}
        self._pad_target: dict[str, PaddleTarget] = {}
        self._pad_velocity: dict[str, tuple[float, float, float]] = {}
        self._pad_min_gap = math.inf
        """Smallest measured gap (m) between the two plate collision shells this episode."""
        self._pad_penetration = 0.0
        """Largest measured plate/plate contact penetration (m) this episode (0 = none)."""
        self._pad_contact_steps = 0
        """Control steps on which the two plates were in physical contact."""
        self._pad_contact_this_step = False
        """Whether the current control step already saw a plate/plate contact."""
        mid_x = self.paddle_ready_x()
        for side in AGENT_KEYS:
            pose = (
                -mid_x if side == "left" else mid_x,
                cfg.pad_start_z,
                self.paddle_ready_tilt(side),
            )
            target = PaddleTarget(*pose)
            self._set_pad_target(side, target)
            self._set_pad_joints(side, target)
            self._pad_target[side] = target
            self._pad_pose[side] = (target.x, target.z, target.tilt)
            self._pad_velocity[side] = (0.0, 0.0, 0.0)

    def _apply_task_actions(self, physical: dict[str, np.ndarray], wind_force: float) -> None:
        """Turn the normalized commands into servo targets and write them into `data.ctrl`."""
        del wind_force  # the task has no external perturbation channel
        cfg = self._config()
        randomization = self._randomization
        if randomization is None:  # pragma: no cover - the base guarantees a reset
            raise RuntimeError("call reset() before step()")
        if self._step_index == self._release_step:
            self._data.qvel[self._ball_dof : self._ball_dof + 3] = self._launch_velocity
        targets: dict[str, PaddleTarget] = {}
        for index, side in enumerate(AGENT_KEYS):
            agent = randomization.agents[index]
            authority = agent.gain * agent.limit_scale
            target = servo_target(
                self._pad_target[side],
                physical[side],
                max_step=cfg.pad_speed * cfg.control_dt * authority,
                max_dtilt=cfg.pad_tilt_rate * cfg.control_dt * authority,
                x_range=cfg.paddle_x_range(side),
                z_range=(cfg.pad_min_z, cfg.pad_max_z),
                tilt_limit=cfg.pad_tilt_limit,
            )
            targets[side] = target
        left, right = project_targets_apart(
            targets["left"],
            targets["right"],
            half_x=self._pad_half_x,
            half_z=self._pad_half_z,
            left_x_range=cfg.paddle_x_range("left"),
            right_x_range=cfg.paddle_x_range("right"),
        )
        self._pad_target["left"] = left
        self._pad_target["right"] = right
        for side in AGENT_KEYS:
            self._set_pad_target(side, self._pad_target[side])

    def _set_pad_target(self, side: str, target: PaddleTarget) -> None:
        """Write one paddle's servo target `(x, z, tilt)` into the three actuators."""
        for actuator, value in zip(self._pad_actuator[side], (target.x, target.z, target.tilt)):
            self._data.ctrl[actuator] = value

    def _set_pad_joints(self, side: str, target: PaddleTarget) -> None:
        """Place one paddle's joints at `target` (reset only; the servos move them afterwards)."""
        for address, value in zip(self._pad_qpos[side], (target.x, target.z, target.tilt)):
            self._data.qpos[address] = value
        for address in self._pad_dof[side]:
            self._data.qvel[address] = 0.0

    def _on_substep(self, substep: int) -> None:
        """Sample the ball/paddle contacts and the plate/plate penetration of a substep.

        A new ball/paddle contact is only counted as a pass when the pure `toss_pass_count`
        rule accepts it (the ball is on the contacting paddle's own half, the sides alternate
        and `min_pass_interval_steps` control steps passed since the last counted pass); the
        raw contact flags are still tracked for the oracle and the observation. The plate
        collision shells are not touched by the servos' target projection, so their contact
        (and its penetration depth) is the physical evidence that the plates cannot pass
        through each other.
        """
        del substep
        cfg = self._config()
        ball_x = float(self._data.qpos[self._ball_qpos])
        for side in AGENT_KEYS:
            contact = self._pad_contacts_ball(side)
            if contact and not self._contact_side[side]:
                passes = toss_pass_count(
                    self._passes,
                    self._last_pass_side,
                    side,
                    ball_x=ball_x,
                    step=self._step_index,
                    last_pass_step=self._last_pass_step,
                    min_interval_steps=cfg.min_pass_interval_steps,
                )
                if passes != self._passes:
                    self._passes = passes
                    self._last_pass_side = side
                    self._last_pass_step = self._step_index
            self._contact_side[side] = contact
        if self._pad_contacts_partner():
            self._pad_contact_this_step = True

    def _pad_contacts_ball(self, side: str) -> bool:
        """Whether the ball geom currently contacts that paddle's contact geom."""
        pair = {self._pad_geom[side], self._ball_geom}
        return any(
            {int(contact.geom1), int(contact.geom2)} == pair
            for contact in self._data.contact[: self._data.ncon]
        )

    def _pad_contacts_partner(self) -> bool:
        """Whether the two plates' contact shells are in contact in the current substep."""
        pair = {self._pad_shell["left"], self._pad_shell["right"]}
        return any(
            {int(contact.geom1), int(contact.geom2)} == pair
            for contact in self._data.contact[: self._data.ncon]
        )

    def _pad_pair_gap(self) -> tuple[float, float]:
        """The plate/plate `(penetration, separation)` SAT measure (pure geometry, m)."""
        return paddle_pair_geometry(
            PaddleTarget(*self._pad_pose["left"]),
            PaddleTarget(*self._pad_pose["right"]),
            half_x=self._pad_half_x,
            half_z=self._pad_half_z,
        )

    def _after_step(self) -> None:
        """Refresh the measured paddle pose/velocity and the plate/plate geometry diagnostic."""
        data = self._data
        for side in AGENT_KEYS:
            qpos = self._pad_qpos[side]
            dof = self._pad_dof[side]
            self._pad_pose[side] = (
                float(data.qpos[qpos[0]]),
                float(data.qpos[qpos[1]]),
                float(data.qpos[qpos[2]]),
            )
            self._pad_velocity[side] = (
                float(data.qvel[dof[0]]),
                float(data.qvel[dof[1]]),
                float(data.qvel[dof[2]]),
            )
        penetration, separation = self._pad_pair_gap()
        self._pad_penetration = max(self._pad_penetration, penetration)
        self._pad_min_gap = min(self._pad_min_gap, separation)
        if self._pad_contact_this_step:
            self._pad_contact_steps += 1
            self._pad_contact_this_step = False

    def _physical_commands(self, applied: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Symmetric pose-increment commands (the toss-catch task has no one-sided split)."""
        return {
            key: np.asarray(applied[key], dtype=np.float64).reshape(self._task_spec.action_dim)
            for key in AGENT_KEYS
        }

    def _shared_state_values(self) -> tuple[float, ...]:
        """Ball position/velocity and both paddles' measured pose/velocity, in spec order."""
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
        """The toss-catch kinematic state, servo targets and pass trackers for the oracle."""
        data = self._data
        state = {
            "ball_x": float(data.qpos[self._ball_qpos]),
            "ball_z": float(data.qpos[self._ball_qpos + 2]),
            "ball_vx": float(data.qvel[self._ball_dof]),
            "ball_vz": float(data.qvel[self._ball_dof + 2]),
            "passes": float(self._passes),
            "target_passes": float(self._config().n_passes),
            "pad_min_gap": float(self._pad_min_gap),
            "pad_penetration": float(self._pad_penetration),
        }
        for side in AGENT_KEYS:
            pose = self._pad_pose[side]
            velocity = self._pad_velocity[side]
            target = self._pad_target[side]
            state[f"{side}_pad_x"] = pose[0]
            state[f"{side}_pad_z"] = pose[1]
            state[f"{side}_pad_tilt"] = pose[2]
            state[f"{side}_pad_vx"] = velocity[0]
            state[f"{side}_pad_vz"] = velocity[1]
            state[f"{side}_pad_tilt_rate"] = velocity[2]
            state[f"{side}_pad_target_x"] = target.x
            state[f"{side}_pad_target_z"] = target.z
            state[f"{side}_pad_target_tilt"] = target.tilt
            state[f"{side}_contact"] = float(self._contact_side[side])
        return state

    def _diagnostics(self) -> dict[str, float]:
        """Pass progress, the pass gate, the ball's state, the plate gap and the time."""
        cfg = self._config()
        state = self._task_state()
        return {
            "passes": float(self._passes),
            "target_passes": float(cfg.n_passes),
            "last_pass_step": float(self._last_pass_step),
            "min_pass_interval_steps": float(cfg.min_pass_interval_steps),
            "ball_x": state["ball_x"],
            "ball_z": state["ball_z"],
            "ball_speed": math.hypot(state["ball_vx"], state["ball_vz"]),
            "pad_min_gap": float(self._pad_min_gap),
            "pad_penetration": float(self._pad_penetration),
            "pad_contact_steps": float(self._pad_contact_steps),
            "time": self._step_index * cfg.control_dt,
        }

    def _success(self) -> bool:
        """Whether the team completed the passes while the ball was still above the floor."""
        cfg = self._config()
        data = self._data
        return toss_success(
            self._passes,
            cfg.n_passes,
            ball_x=float(data.qpos[self._ball_qpos]),
            ball_y=float(data.qpos[self._ball_qpos + 1]),
            ball_z=float(data.qpos[self._ball_qpos + 2]),
            floor_z=cfg.floor_z,
            court_x=cfg.court_x,
            court_y=cfg.court_y,
            floor_margin=cfg.success_floor_margin,
        )

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
