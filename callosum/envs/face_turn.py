"""FaceTurn-v0: a holder SO-ARM101 arm and a rotator SO-ARM101 arm turn a
turntable cube's face 90 degrees.

Overrides TwoSO101Base's simple loose cube with the turntable-cube
articulation from `_turntable_cube.py`. Roles are fixed: agent_a is the
"holder" (keeps the body in place), agent_b is the "rotator" (grasps and
turns the face) -- see docs/thesis/04-experiment-design.md.
"""

import math
from typing import Any

import torch
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from callosum.configs.face_turn import FaceTurnPhysicsConfig, FaceTurnRewardConfig
from callosum.envs._face_lock import update_face_lock
from callosum.envs._turntable_cube import build_turntable_cube
from callosum.envs.two_so101_base import TwoSO101Base

TARGET_FACE_ANGLE = math.pi / 2  # a quarter turn
_DEFAULT_REWARD_CONFIG = FaceTurnRewardConfig()
_DEFAULT_PHYSICS_CONFIG = FaceTurnPhysicsConfig()


# 400 control steps: the scripted holder-then-rotator expert (scripts/probe_face_turn.py, CPU sim)
# first reaches success after 329 steps (350 incl. the release), so 300 would cut it off. It is
# deliberately slow (waypoint paths, settling steps), but a learned policy needs room for
# exploration and regrasps too. TwoSO101-v0 (reach only) stays at 100.
@register_env("FaceTurn-v0", max_episode_steps=400)
class FaceTurn(TwoSO101Base):
    """Bimanual face-turn task on top of TwoSO101Base's two-arm plumbing.

    `agent_a` (holder) is rewarded for staying near the cube body; `agent_b`
    (rotator) is rewarded for reaching the face, grasping it, and turning it
    toward `TARGET_FACE_ANGLE`. Success additionally requires the body to
    have stayed within its initial pose's position/rotation tolerance --
    turning the face by knocking the whole cube around does not count.

    `partner_obs` does not change this env's observation (both TCP poses are always in the
    shared extra-obs dict; who sees which one is decided in `callosum.training._agent_obs`), and
    `compute_dense_reward` / `evaluate` read TCP poses straight off `self.agent_a`/`agent_b`
    (privileged, CTDE-style access).

    Face lock (`FaceTurnPhysicsConfig.lock_face_unless_held`, on by default): the face can only
    be turned while the holder grasps the body. Every control step the env checks
    `agent_a.is_grasping(body_link)` (the check the reward uses); while it is False the face
    joint is held at the angle it had when the lock engaged, by writing that angle back (and
    zeroing the joint velocity) after every physics substep.

    Why this mechanism (ManiSkill v3.0.1, SAPIEN 3.0.3), per env and on both backends:
    - A PD drive on the face joint (stiffness on to lock, off to unlock) does not work: on the
      GPU backend drive properties (and joint limits, friction) are per joint *type*, not per
      env, and cannot be changed after the sim is built (`ArticulationJoint` docstring,
      `@before_gpu_init` on `limits`). A constant stiff drive with per-env targets cannot
      express "free" either: tracking the current angle as target acts as damping K * dt.
    - Explicit joint torques (`Articulation.qf`) are per env, but an explicit stiff spring at
      the 100 Hz sim step is unstable for the face's tiny inertia (about 3e-5 kg m^2).
    - So the lock is a kinematic clamp: after every substep, for the locked envs, write the
      held angle into the face qpos and zero its qvel. On the CPU backend (one env) that is
      `Articulation.set_qpos/set_qvel`. On the GPU backend the qpos/qvel buffers are only
      fetched once per control step (`_step_action`), so the hook fetches them itself, writes
      the locked envs' values and applies them back.
    Caveats: a clamp is not a force constraint, so the reaction torque of a locked face is not
    passed on to the body, and with a jaw squeezing a locked face the contact solver sees a
    penetration kick each substep; both only happen while the holder is not grasping, i.e. in
    states the rule exists to make unproductive. The decision uses the contact forces of the
    previous control step, so a grasp is noticed one step late. `is_grasping` can flicker
    while the face is being turned (see `FaceTurnRewardConfig.rotator_grasp_gate_floor`); every
    flicker re-latches the lock at the current angle, which is a brief stall, not a reset.
    """

    def __init__(
        self,
        *args,
        reward_config: FaceTurnRewardConfig = _DEFAULT_REWARD_CONFIG,
        physics_config: FaceTurnPhysicsConfig = _DEFAULT_PHYSICS_CONFIG,
        **kwargs,
    ):
        self.reward_config = reward_config
        self.physics_config = physics_config
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        # Re-implements TwoSO101Base._load_scene's table setup instead of
        # calling super() -- the parent also builds the plain loose cube,
        # which this task replaces with the turntable-cube articulation, so
        # there is nothing to reuse from the parent's cube-building line.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cube = build_turntable_cube(
            self.scene,
            name="turntable_cube",
            face_friction=self.physics_config.face_friction,
            face_damping=self.physics_config.face_damping,
        )
        self.body_link = self.cube.links_map["body"]
        self.face_link = self.cube.links_map["face"]

        # Filled in per env_idx in _initialize_episode; used by evaluate() and
        # compute_dense_reward() to detect body drift from its initial pose.
        self.body_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.body_init_q = torch.zeros((self.num_envs, 4), device=self.device)

        # Face-lock state, per env (see the class docstring); reset in _initialize_episode for
        # the reset envs only. `face_lock_engaged_steps` counts the control steps each env spent
        # locked (a diagnostic: the GPU probe prints it to confirm the lock acts per env).
        self.face_locked = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.face_lock_angle = torch.zeros(self.num_envs, device=self.device)
        self.face_lock_engaged_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._any_face_locked = False

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Places both arms (rest keyframe + noise) and the cube body (jittered
        # center position) -- see TwoSO101Base._initialize_episode.
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            # Articulation.set_qpos (not `joint.qpos = ...`, whose setter only accepts a
            # batch on the GPU backend) so this also works on the CPU backend.
            self.cube.set_qpos(torch.zeros((b, 1)))
            # The holder is not grasping at reset, so the lock starts engaged at angle 0. The
            # state is per env: a partial reset must not touch the other envs' lock.
            self.face_locked[env_idx] = True
            self.face_lock_angle[env_idx] = 0.0
            self.face_lock_engaged_steps[env_idx] = 0
            self.body_init_pos[env_idx] = self.cube.pose.p[env_idx]
            self.body_init_q[env_idx] = self.cube.pose.q[env_idx]

    def _before_control_step(self):
        # Decide the lock for this control step from the holder's grasp (contact forces of the
        # previous step; privileged, like the reward) and the face angle at the step start.
        if not self.physics_config.lock_face_unless_held:
            return
        held = self.agent_a.is_grasping(self.body_link)
        self.face_locked, self.face_lock_angle = update_face_lock(
            self.face_locked, self.face_lock_angle, ~held, self.face_link.joint.qpos
        )
        self.face_lock_engaged_steps += self.face_locked.long()
        # One host sync per control step (not per substep) to skip the clamp when nothing is locked.
        self._any_face_locked = bool(self.face_locked.any())

    def _after_simulation_step(self):
        # Clamp the locked envs' face joint after every physics substep (see the class
        # docstring). The held angle was latched in _before_control_step.
        if not (self.physics_config.lock_face_unless_held and self._any_face_locked):
            return
        if self.gpu_sim_enabled:
            # TODO(review): GPU path unverified (no GPU on the dev machine). The qpos/qvel
            # buffers are normally fetched once per control step, so fetch them here, overwrite
            # the locked envs, and apply. The apply writes all articulations' qpos/qvel, which
            # is the identity for everything except the face because the buffers were just
            # fetched. Kinematics are refreshed so the link poses match the written qpos.
            px = self.scene.px
            px.gpu_fetch_articulation_qpos()
            px.gpu_fetch_articulation_qvel()
        locked = self.face_locked.unsqueeze(1)
        qpos = torch.where(locked, self.face_lock_angle.unsqueeze(1), self.cube.get_qpos())
        qvel = torch.where(locked, torch.zeros_like(qpos), self.cube.get_qvel())
        self.cube.set_qpos(qpos)
        self.cube.set_qvel(qvel)
        if self.gpu_sim_enabled:
            px.gpu_apply_articulation_qpos()
            px.gpu_apply_articulation_qvel()
            px.gpu_update_articulation_kinematics()

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if "state" in self.obs_mode:
            obs["face_angle"] = info["face_angle"]
            obs["face_pose"] = self.face_link.pose.raw_pose
        return obs

    def evaluate(self):
        cfg = self.reward_config
        face_angle = self.face_link.joint.qpos
        angle_ok = torch.abs(TARGET_FACE_ANGLE - face_angle) < cfg.angle_tol

        pos_drift = torch.linalg.norm(self.cube.pose.p - self.body_init_pos, dim=1)
        rot_drift = common.quat_diff_rad(self.cube.pose.q, self.body_init_q)
        is_body_stable = (pos_drift < cfg.body_pos_tol) & (rot_drift < cfg.body_rot_tol)

        return {
            "success": angle_ok & is_body_stable,
            "face_angle": face_angle,
            "is_body_stable": is_body_stable,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        cfg = self.reward_config

        # (a) rotator reaches for the face, (d) holder reaches for the body.
        # TODO(review): reach target is the face layer's geometric center.
        # Now that FACE_THICKNESS is a real 3x3 layer (1.9 cm, not a thin
        # plate), the center is a defensible grasp target -- a parallel-jaw
        # gripper should be able to pinch the layer from its side faces.
        # Unconfirmed without GPU sim whether the gripper actually closes on
        # it there.
        rotator_to_face = torch.linalg.norm(self.agent_b.tcp_pos - self.face_link.pose.p, dim=1)
        rotator_reach = 1 - torch.tanh(5 * rotator_to_face)
        holder_to_body = torch.linalg.norm(self.agent_a.tcp_pos - self.cube.pose.p, dim=1)
        holder_reach = 1 - torch.tanh(5 * holder_to_body)

        # (b) the rotator grasping the face, (f) the holder grasping the body.
        rotator_grasp = self.agent_b.is_grasping(self.face_link).float()
        holder_grasp = self.agent_a.is_grasping(self.body_link).float()

        # (c) progress of the face angle toward the target, in [0, 1].
        angle_remaining = (TARGET_FACE_ANGLE - info["face_angle"]).clamp(min=0)
        if cfg.angle_progress_shape == "linear":
            # Constant gradient over the whole 0..90 deg range.
            angle_progress = (1 - angle_remaining / TARGET_FACE_ANGLE).clamp(0, 1)
        else:
            # turn_faucet.py's own (commented-out, unshipped) draft shape; ~0.004 at 0 deg.
            angle_progress = 1 - torch.tanh(2 * angle_remaining)

        # Order gates (see FaceTurnRewardConfig.rotator_grasp_gate_floor / angle_gate_floor): the
        # rotator's grasp counts in full only while the holder holds the body (soft floor), and the
        # face-angle progress is paid only then (hard gate by default: a partially turned face must
        # not keep earning after the holder lets go). The rotator's reach term is not gated, so it
        # still gets pulled toward the face meanwhile.
        grasp_gate = cfg.order_gate(cfg.rotator_grasp_gate_floor, holder_grasp)
        angle_gate = cfg.order_gate(cfg.angle_gate_floor, holder_grasp)

        # (e) penalty for the body drifting from its initial pose. Separate
        # weights since position (m) and rotation (rad) drift aren't on
        # commensurate scales. Hinged at the success tolerances if configured.
        pos_drift = torch.linalg.norm(self.cube.pose.p - self.body_init_pos, dim=1)
        rot_drift = common.quat_diff_rad(self.cube.pose.q, self.body_init_q)
        if cfg.hinge_drift_penalty:
            pos_drift = (pos_drift - cfg.body_pos_tol).clamp(min=0)
            rot_drift = (rot_drift - cfg.body_rot_tol).clamp(min=0)

        dense = (
            cfg.weight_rotator_reach * rotator_reach
            + cfg.weight_grasp * grasp_gate * rotator_grasp
            + cfg.weight_angle_progress * angle_gate * angle_progress
            + cfg.weight_holder_reach * holder_reach
            + cfg.weight_holder_grasp * holder_grasp
            - cfg.weight_body_pos_drift * pos_drift
            - cfg.weight_body_rot_drift * rot_drift
        )
        # One-step bonus on the step where the episode terminates with success (see
        # FaceTurnRewardConfig.success_bonus): finishing must beat lingering near the goal.
        # The dense reward carries it in raw units, so the normalised reward (dense / divisor)
        # carries exactly `success_bonus`.
        return cfg.add_success_bonus(dense, info["success"])

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # The divisor comes from the config weights (single source of truth).
        return (
            self.compute_dense_reward(obs=obs, action=action, info=info)
            / self.reward_config.max_positive_reward
        )
