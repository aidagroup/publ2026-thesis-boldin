"""FaceTurn-v0: a holder SO-100 arm and a rotator SO-100 arm turn a
turntable cube's face 90 degrees.

Overrides TwoSO100Base's simple loose cube with the turntable-cube
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
from callosum.envs.two_so100_base import TwoSO100Base

TARGET_FACE_ANGLE = math.pi / 2  # a quarter turn
_DEFAULT_REWARD_CONFIG = FaceTurnRewardConfig()


@register_env("FaceTurn-v0", max_episode_steps=100)
class FaceTurn(TwoSO100Base):
    """Bimanual face-turn task on top of TwoSO100Base's two-arm plumbing.

    `agent_a` (holder) is rewarded for staying near the cube body; `agent_b`
    (rotator) is rewarded for reaching the face, grasping it, and turning it
    toward `TARGET_FACE_ANGLE`. Success additionally requires the body to
    have stayed within its initial pose's position/rotation tolerance --
    turning the face by knocking the whole cube around does not count.
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
        # Re-implements TwoSO100Base._load_scene's table setup instead of
        # calling super() -- the parent also builds the plain loose cube,
        # which this task replaces with the turntable-cube articulation, so
        # there is nothing to reuse from the parent's cube-building line.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cube = build_turntable_cube(self.scene, name="turntable_cube")
        self.face_link = self.cube.links_map["face"]

        # Filled in per env_idx in _initialize_episode; used by evaluate() and
        # compute_dense_reward() to detect body drift from its initial pose.
        self.body_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.body_init_q = torch.zeros((self.num_envs, 4), device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Places both arms (rest keyframe + noise) and the cube body (jittered
        # center position) -- see TwoSO100Base._initialize_episode.
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.face_link.joint.qpos = torch.zeros(b)
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

        # (b) grasping the face.
        is_grasped = self.agent_b.is_grasping(self.face_link)

        # (c) progress of the face angle toward the target. Shape (and the
        # 2.0 scale) matches turn_faucet.py's own (commented-out, unshipped)
        # draft reward for this same family of task -- the closest available
        # precedent, since that file's shipped compute_dense_reward is a
        # "TODO (stao, tmu): finalize a dense reward" stub.
        angle_remaining = (TARGET_FACE_ANGLE - info["face_angle"]).clamp(min=0)
        angle_progress = 1 - torch.tanh(2 * angle_remaining)

        # (e) penalty for the body drifting from its initial pose. Separate
        # weights since position (m) and rotation (rad) drift aren't on
        # commensurate scales.
        pos_drift = torch.linalg.norm(self.cube.pose.p - self.body_init_pos, dim=1)
        rot_drift = common.quat_diff_rad(self.cube.pose.q, self.body_init_q)

        return (
            cfg.weight_rotator_reach * rotator_reach
            + cfg.weight_grasp * is_grasped
            + cfg.weight_angle_progress * angle_progress
            + cfg.weight_holder_reach * holder_reach
            - cfg.weight_body_pos_drift * pos_drift
            - cfg.weight_body_rot_drift * rot_drift
        )

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        cfg = self.reward_config
        # Sum of the positive, bounded ([0, 1]-ish) term weights; the drift
        # penalty is excluded since it is unbounded and ~0 in the successful
        # (no-drift) case this normalization targets.
        max_reward = (
            cfg.weight_rotator_reach
            + cfg.weight_grasp
            + cfg.weight_angle_progress
            + cfg.weight_holder_reach
        )
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
