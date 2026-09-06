"""FaceTurn-v0: a holder SO-100 arm and a rotator SO-100 arm turn a
turntable cube's face 90 degrees.

Overrides TwoSO100Base's simple loose cube with the turntable-cube
articulation from `_turntable_cube.py`. Roles are fixed: agent_a is the
"holder" (keeps the body in place), agent_b is the "rotator" (grasps and
turns the face) -- see docs/thesis/04-experiment-design.md.
"""

import math
from typing import Any

import sapien
import torch
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from callosum.configs.face_turn import FaceTurnRewardConfig
from callosum.envs._cube_geometry import (
    BODY_GRASP_OFFSET,
    CUBE_HALF_SIZE,
    FACE_GRASP_OFFSET,
    LIFT_HEIGHT,
)
from callosum.envs._turntable_cube import build_turntable_cube
from callosum.envs.two_so100_base import TwoSO100Base

TARGET_FACE_ANGLE = math.pi / 2  # a quarter turn
_DEFAULT_REWARD_CONFIG = FaceTurnRewardConfig()


# See the step-budget note on TwoSO100-v0.
@register_env("FaceTurn-v0", max_episode_steps=300)
class FaceTurn(TwoSO100Base):
    """Bimanual face-turn task on top of TwoSO100Base's two-arm plumbing.

    `agent_a` (holder) is rewarded for staying near the cube body; `agent_b`
    (rotator) is rewarded for reaching the face, grasping it, and turning it
    toward `TARGET_FACE_ANGLE`. Success additionally requires the body to
    have stayed within its initial pose's position/rotation tolerance --
    turning the face by knocking the whole cube around does not count.

    Both arms reach for the grasp HANDLES on the cube, not for the links'
    origins (see `callosum.envs._turntable_cube`). The rotator's handle sits
    on the rotation axis on purpose: the SO-100 has 5 DOF -- a base yaw, three
    parallel pitch joints and a wrist roll about the tool's own approach axis
    -- so the ONLY way it can spin a grasped object about a world-vertical
    axis without dragging it sideways is to approach straight down and use
    `wrist_roll`. That matches the design doc's "the rotator comes in from
    above" (docs/thesis/04-experiment-design.md, v1 simplifications), and it
    is why `TwoSO100Base._load_agent` now points the arms at the cube (see
    the base-yaw note there): with the previous base yaws the rotator's wrist
    could not be placed above the cube at all.

    Inherits TwoSO100Base's `partner_obs` flag unchanged: `compute_dense_reward`
    and `evaluate` always read TCP poses straight off `self.agent_a`/`agent_b`
    (privileged, CTDE-style access, not the observation dict), so they are
    unaffected by it either way -- only the shared extra-obs dict changes.
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
        # The holder grips the body link, not the articulation as a whole:
        # is_grasping takes an Actor or a Link, and contact must be measured
        # against the part that is actually being held.
        self.body_link = self.cube.links_map["body"]

        # Grasp handles, as poses in their links' frames. Composed with the
        # link pose (mani_skill Pose.__mul__ broadcasts a single sapien.Pose
        # against a batched one) so the targets follow the face as it turns
        # and the body if it gets nudged.
        self._face_grasp_local = sapien.Pose(p=list(FACE_GRASP_OFFSET))
        self._body_grasp_local = sapien.Pose(p=list(BODY_GRASP_OFFSET))

        # Filled in per env_idx in _initialize_episode; used by evaluate() and
        # compute_dense_reward() to detect body drift from its initial pose.
        self.body_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.body_init_q = torch.zeros((self.num_envs, 4), device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Places both arms (READY_QPOS + noise) and the cube body (jittered
        # center position) -- see TwoSO100Base._initialize_episode.
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.face_link.joint.qpos = torch.zeros(b)
            self.body_init_pos[env_idx] = self.cube.pose.p[env_idx]
            self.body_init_q[env_idx] = self.cube.pose.q[env_idx]

    @property
    def face_grasp_pos(self) -> torch.Tensor:
        """World position of the rotator's grasp handle, shape (num_envs, 3)."""
        return (self.face_link.pose * self._face_grasp_local).p

    @property
    def body_grasp_pos(self) -> torch.Tensor:
        """World position of the holder's grasp handle, shape (num_envs, 3)."""
        return (self.body_link.pose * self._body_grasp_local).p

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if "state" in self.obs_mode:
            obs["face_angle"] = info["face_angle"]
            obs["face_pose"] = self.face_link.pose.raw_pose
            # Where to actually put the gripper. Without these the policy has
            # to infer the handle offsets from the link poses; they are fixed
            # offsets, but only in the LINKS' frames, so recovering them costs
            # the network a rotation it does not need to learn.
            obs["face_grasp_pos"] = self.face_grasp_pos
            obs["body_grasp_pos"] = self.body_grasp_pos
        return obs

    def evaluate(self):
        cfg = self.reward_config
        # |angle|, not the signed value: which way a wrist_roll drives the
        # joint is a sign convention nobody can read off the builder, and the
        # joint's limits are symmetric so either direction is a quarter turn.
        face_angle = self.face_link.joint.qpos
        angle_ok = torch.abs(face_angle) > TARGET_FACE_ANGLE - cfg.angle_tol

        # The cube is SUPPOSED to move now -- the holder picks it up -- so
        # what "stable" means is that the body has not TURNED. Rotation is the
        # failure the reaction torque produces; translation is the task.
        rot_drift = common.quat_diff_rad(self.cube.pose.q, self.body_init_q)
        is_lifted = self.cube.pose.p[:, 2] > LIFT_HEIGHT - cfg.lift_tol
        is_body_stable = (rot_drift < cfg.body_rot_tol) & is_lifted

        # The staircase is gated: nothing pays for lifting unless the holder
        # is gripping, and none of the rotator's terms pay unless the cube is
        # lifted. So a run where is_lifted stays 0 has to say WHICH rung the
        # policy is stuck on, or the log cannot distinguish "never touches the
        # cube" from "grips it but will not carry it".
        return {
            "success": angle_ok & is_body_stable,
            "face_angle": face_angle,
            "is_body_stable": is_body_stable,
            "is_lifted": is_lifted,
            "holder_grasped": self.agent_a.is_grasping(self.body_link),
            "rotator_grasped": self.agent_b.is_grasping(self.face_link),
            "holder_dist": torch.linalg.norm(self.agent_a.tcp_pos - self.body_grasp_pos, dim=1),
            "rotator_dist": torch.linalg.norm(self.agent_b.tcp_pos - self.face_grasp_pos, dim=1),
            "cube_height": self.cube.pose.p[:, 2],
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        cfg = self.reward_config

        # (a) the holder reaches its grip on the body, and (b) closes on it.
        #
        # The reach target is NOT a link origin. Both link origins are inside
        # solid geometry, and a distance-to-target reward is monotone, so its
        # greedy optimum was to drive the jaw tips into the cube's surface --
        # which is not a grasp pose. `BODY_GRASP_OFFSET` is the point the tool
        # centre point actually occupies at the solved grasp, 1.94 cm to the
        # side of the cube's axis because the jaws hold the body against the
        # flat of the fixed blade.
        holder_to_body = torch.linalg.norm(self.agent_a.tcp_pos - self.body_grasp_pos, dim=1)
        holder_reach = 1 - torch.tanh(5 * holder_to_body)
        holder_grasped = self.agent_a.is_grasping(self.body_link)

        # (c) lift it to working height. Gated on the grip: without the gate
        # the term pays for knocking the cube upward, which is the opposite of
        # the behaviour wanted.
        height = self.cube.pose.p[:, 2]
        lift_progress = ((height - CUBE_HALF_SIZE) / (LIFT_HEIGHT - CUBE_HALF_SIZE)).clamp(
            0.0, 1.0
        ) * holder_grasped

        # (d) the rotator reaches the nub and (e) closes on it -- both gated on
        # the cube being off the table, because until then the rotator has
        # nowhere to be: its own closed gripper sweeps 4.26 cm below the cube's
        # axis, which on the table is inside the tabletop.
        lifted = info["is_lifted"].float()
        rotator_to_face = torch.linalg.norm(self.agent_b.tcp_pos - self.face_grasp_pos, dim=1)
        rotator_reach = (1 - torch.tanh(5 * rotator_to_face)) * lifted
        is_grasped = self.agent_b.is_grasping(self.face_link) * lifted

        # (f) progress of the face angle toward the target. Shape (and the
        # 2.0 scale) matches turn_faucet.py's own (commented-out, unshipped)
        # draft reward for this same family of task -- the closest available
        # precedent, since that file's shipped compute_dense_reward is a
        # "TODO (stao, tmu): finalize a dense reward" stub.
        angle_remaining = (TARGET_FACE_ANGLE - torch.abs(info["face_angle"])).clamp(min=0)
        angle_progress = 1 - torch.tanh(2 * angle_remaining)

        # (g) penalty for the body turning BEYOND the tolerance evaluate()
        # accepts. Clamped at the tolerance, not linear from zero: measured on
        # the 2M-step run of 2026-08-30, an unclamped penalty made merely
        # touching the cube cost more per step than approaching it gained, so
        # standing still was a local optimum -- and the policy found it.
        rot_drift = (
            common.quat_diff_rad(self.cube.pose.q, self.body_init_q) - cfg.body_rot_tol
        ).clamp(min=0)

        return (
            cfg.weight_holder_reach * holder_reach
            + cfg.weight_holder_grasp * holder_grasped
            + cfg.weight_lift * lift_progress
            + cfg.weight_rotator_reach * rotator_reach
            + cfg.weight_grasp * is_grasped
            + cfg.weight_angle_progress * angle_progress
            - cfg.weight_body_rot_drift * rot_drift
        )

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return (
            self.compute_dense_reward(obs=obs, action=action, info=info)
            / self.reward_normalization_divisor
        )

    @property
    def reward_normalization_divisor(self) -> float:
        """Sum of the POSITIVE term weights -- what normalized_dense divides by.

        Single source of truth: the trainer logs this with the run, because any
        change to it silently rescales train/return and makes runs incomparable.
        The drift weights are deliberately excluded (they are unbounded penalties,
        ~0 in the successful case this normalization targets), which is exactly
        the subtlety that makes recomputing it elsewhere error-prone.
        """
        cfg = self.reward_config
        return (
            cfg.weight_holder_reach
            + cfg.weight_holder_grasp
            + cfg.weight_lift
            + cfg.weight_rotator_reach
            + cfg.weight_grasp
            + cfg.weight_angle_progress
        )
