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
from callosum.envs._turntable_cube import (
    BODY_GRASP_OFFSET,
    FACE_GRASP_OFFSET,
    build_turntable_cube,
)
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

        # (a) rotator reaches for the face handle, (d) holder for the body
        # handle.
        #
        # These used to aim at `face_link.pose.p` and `cube.pose.p`, both of
        # which are INSIDE solid geometry -- the face link's origin is the
        # layer's geometric centre, the articulation root pose is a point
        # 9.5 mm above the body box's centre. A distance-to-target reward is
        # monotone, so its greedy optimum was to drive the jaw tips into the
        # cube's surface, which is not a grasp pose. It was also actively at
        # odds with grasping: SO100.tcp_pos is the midpoint of the two jaw
        # TIPS, so for a 5.7 cm object held between open jaws the tcp sits
        # ~2.1 cm from the object's centre (measured from the URDF), i.e.
        # opening the gripper to grasp LOWERED the reach reward. Aiming at
        # a 2 cm handle removes both problems: with the jaws closed on it the
        # tcp and the handle centre coincide to within ~3 mm.
        rotator_to_face = torch.linalg.norm(self.agent_b.tcp_pos - self.face_grasp_pos, dim=1)
        rotator_reach = 1 - torch.tanh(5 * rotator_to_face)
        holder_to_body = torch.linalg.norm(self.agent_a.tcp_pos - self.body_grasp_pos, dim=1)
        holder_reach = 1 - torch.tanh(5 * holder_to_body)

        # (b) grasping: rotator on the face, holder on the body. Both are needed
        # and both are rewarded -- the holder's grip is what makes the face
        # turnable at all, since the reaction torque would otherwise just spin
        # the free-floating cube.
        is_grasped = self.agent_b.is_grasping(self.face_link)
        holder_grasped = self.agent_a.is_grasping(self.body_link)

        # (c) progress of the face angle toward the target. Shape (and the
        # 2.0 scale) matches turn_faucet.py's own (commented-out, unshipped)
        # draft reward for this same family of task -- the closest available
        # precedent, since that file's shipped compute_dense_reward is a
        # "TODO (stao, tmu): finalize a dense reward" stub.
        angle_remaining = (TARGET_FACE_ANGLE - info["face_angle"]).clamp(min=0)
        angle_progress = 1 - torch.tanh(2 * angle_remaining)

        # (e) penalty for the body drifting BEYOND the tolerance evaluate()
        # accepts. Separate weights since position (m) and rotation (rad) drift
        # aren't on commensurate scales.
        #
        # The hinge is not cosmetic. Measured on the 2M-step run of 2026-08-30,
        # with the penalty linear from zero: merely touching the cube within the
        # tolerance that still counts as success cost 5*0.01 + 5*0.10 = 0.55 per
        # step, while approaching from the rest pose (0.37 m) to 0.20 m gains
        # only 0.19. Approaching was net-negative in expectation, so standing
        # still was a local optimum -- and the policy found it: after 156 updates
        # the tool centre points had not left the rest pose (0.370 -> ~0.36 m)
        # and success_once was 0 in all 78 logged points. Clamping at the
        # tolerance makes the dense reward agree with the success predicate:
        # incidental contact is free, only real destabilisation is punished.
        pos_drift = (
            torch.linalg.norm(self.cube.pose.p - self.body_init_pos, dim=1) - cfg.body_pos_tol
        ).clamp(min=0)
        rot_drift = (
            common.quat_diff_rad(self.cube.pose.q, self.body_init_q) - cfg.body_rot_tol
        ).clamp(min=0)

        return (
            cfg.weight_rotator_reach * rotator_reach
            + cfg.weight_grasp * is_grasped
            + cfg.weight_holder_grasp * holder_grasped
            + cfg.weight_angle_progress * angle_progress
            + cfg.weight_holder_reach * holder_reach
            - cfg.weight_body_pos_drift * pos_drift
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
            cfg.weight_rotator_reach
            + cfg.weight_grasp
            + cfg.weight_holder_grasp
            + cfg.weight_angle_progress
            + cfg.weight_holder_reach
        )
