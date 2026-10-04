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

from callosum.configs.face_turn import FaceTurnRewardConfig
from callosum.envs._turntable_cube import build_turntable_cube
from callosum.envs.two_so101_base import TwoSO101Base

TARGET_FACE_ANGLE = math.pi / 2  # a quarter turn
_DEFAULT_REWARD_CONFIG = FaceTurnRewardConfig()


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
    """

    def __init__(
        self,
        *args,
        reward_config: FaceTurnRewardConfig = _DEFAULT_REWARD_CONFIG,
        **kwargs,
    ):
        self.reward_config = reward_config
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

        self.cube = build_turntable_cube(self.scene, name="turntable_cube")
        self.body_link = self.cube.links_map["body"]
        self.face_link = self.cube.links_map["face"]

        # Filled in per env_idx in _initialize_episode; used by evaluate() and
        # compute_dense_reward() to detect body drift from its initial pose.
        self.body_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.body_init_q = torch.zeros((self.num_envs, 4), device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Places both arms (rest keyframe + noise) and the cube body (jittered
        # center position) -- see TwoSO101Base._initialize_episode.
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            # Articulation.set_qpos (not `joint.qpos = ...`, whose setter only accepts a
            # batch on the GPU backend) so this also works on the CPU backend.
            self.cube.set_qpos(torch.zeros((b, 1)))
            self.body_init_pos[env_idx] = self.cube.pose.p[env_idx]
            self.body_init_q[env_idx] = self.cube.pose.q[env_idx]

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

        # Order gate: the rotator's grasp and turn count in full only while the holder holds
        # the body (see FaceTurnRewardConfig.rotator_gate_floor for hard vs soft). The rotator's
        # reach term is not gated, so it still gets pulled toward the face meanwhile.
        if cfg.gate_rotator_on_holder:
            gate = cfg.rotator_gate_floor + (1 - cfg.rotator_gate_floor) * holder_grasp
        else:
            gate = torch.ones_like(holder_grasp)

        # (e) penalty for the body drifting from its initial pose. Separate
        # weights since position (m) and rotation (rad) drift aren't on
        # commensurate scales. Hinged at the success tolerances if configured.
        pos_drift = torch.linalg.norm(self.cube.pose.p - self.body_init_pos, dim=1)
        rot_drift = common.quat_diff_rad(self.cube.pose.q, self.body_init_q)
        if cfg.hinge_drift_penalty:
            pos_drift = (pos_drift - cfg.body_pos_tol).clamp(min=0)
            rot_drift = (rot_drift - cfg.body_rot_tol).clamp(min=0)

        return (
            cfg.weight_rotator_reach * rotator_reach
            + cfg.weight_grasp * gate * rotator_grasp
            + cfg.weight_angle_progress * gate * angle_progress
            + cfg.weight_holder_reach * holder_reach
            + cfg.weight_holder_grasp * holder_grasp
            - cfg.weight_body_pos_drift * pos_drift
            - cfg.weight_body_rot_drift * rot_drift
        )

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # The divisor comes from the config weights (single source of truth).
        return (
            self.compute_dense_reward(obs=obs, action=action, info=info)
            / self.reward_config.max_positive_reward
        )
