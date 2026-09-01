"""Two-arm SO-100 base environment: plumbing for observations/actions/reward.

Sets up a bimanual scene (two SO-100 arms + a loose cube) with per-agent
observation fields and a reach-only dense reward, ahead of the articulated
face-turn task in step 1.3. There is no task/goal yet -- `evaluate()` is a
stub that always reports failure.
"""

from typing import Any, ClassVar

import sapien
import torch
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.so100 import SO100
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from callosum.envs._cube_geometry import CUBE_HALF_SIZE
from callosum.envs._partner_obs import partner_tcp_pose_fields, validate_partner_obs
from callosum.envs._so100_kinematics import (
    ARM_BASE_OFFSET,
    HOLDER_BASE_YAW,
    READY_QPOS,
    ROTATOR_BASE_YAW,
)

# Arm placement and start pose live in `_so100_kinematics`, next to the
# forward kinematics that justifies them and where CI can check them without a
# simulator (tests/test_grasp_waypoints.py). The cube's dimensions likewise
# live in `_cube_geometry`.


# 300, not 100. The budget an optimal FaceTurn episode needs, from the
# kinematic solution for the grasp poses: >=10 control steps for the rotator
# and >=19 for the holder to travel from READY_QPOS to their grasp
# configurations at the 0.05 rad/step delta limit, ~6 steps to close the
# gripper at 0.2 rad/step, and >=32 steps of wrist_roll for the quarter turn.
# ~70 steps of pure motion, so 300 leaves room to correct and settle. (With
# the pre-2026-08-30 base yaws the approach alone cost ~90 steps, because
# ~32 of them went into rotating shoulder_pan to face the cube.)
@register_env("TwoSO100-v0", max_episode_steps=300)
class TwoSO100Base(BaseEnv):
    """Two SO-100 arms around a table with a single loose cube.

    Both arms observe their own proprioception plus the cube pose (see
    `_get_obs_extra`); the dense reward simply pulls both TCPs toward the
    cube. This class only wires up the plumbing -- agents, observations,
    actions, and reward shapes -- so it can be smoke-tested before the real
    articulated face-turn task (step 1.3) lands.

    `partner_obs` (`"full"` or `"none"`) toggles whether both agents' TCP
    poses are included in the shared extra-obs dict -- the oracle/no-partner
    ends of the ablation triple from docs/thesis/04-experiment-design.md
    (`"predicted"`, from Bi-JEPA, lands in a later phase). See
    `callosum.envs._partner_obs` for exactly what this can and can't express
    at this stage.
    """

    SUPPORTED_ROBOTS: ClassVar[list[tuple[str, str]]] = [("so100", "so100")]
    agent: MultiAgent[tuple[SO100, SO100]]

    def __init__(
        self,
        *args,
        robot_uids=("so100", "so100"),
        robot_init_qpos_noise=0.02,
        partner_obs="full",
        **kwargs,
    ):
        validate_partner_obs(partner_obs)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.partner_obs = partner_obs
        # No explicit control_mode: SO100's first configured controller is
        # already "pd_joint_delta_pos" (SO100._controller_configs), which
        # BaseAgent picks by default whenever control_mode is None.
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def agent_a(self) -> SO100:
        """The first arm (uid `so100-0`)."""
        return self.agent.agents[0]

    @property
    def agent_b(self) -> SO100:
        """The second arm (uid `so100-1`)."""
        return self.agent.agents[1]

    def _load_agent(self, options: dict):
        # Base yaws pi and 0, NOT the panda pair's +pi/2 / -pi/2 -- see
        # `_so100_kinematics.ARM_BASE_OFFSET` for why, and for the reach
        # numbers that set the spacing.
        super()._load_agent(
            options,
            [
                sapien.Pose(p=[0, -ARM_BASE_OFFSET, 0], q=euler2quat(0, 0, HOLDER_BASE_YAW)),
                sapien.Pose(p=[0, ARM_BASE_OFFSET, 0], q=euler2quat(0, 0, ROTATOR_BASE_YAW)),
            ],
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cube = actors.build_cube(
            self.scene,
            half_size=CUBE_HALF_SIZE,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, CUBE_HALF_SIZE]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # Resets the table/ground pose. TableSceneBuilder.initialize() has
            # no branch for robot_uids == ("so100", "so100") as of mani-skill
            # 3.0.1 (only single-so100 and panda-pair combinations are known
            # to it), so it silently skips robot placement -- both arms' qpos
            # are reset explicitly below instead. Base poses are fixed (set
            # once in _load_agent) and do not need resetting per episode.
            self.table_scene.initialize(env_idx)

            start_qpos = common.to_tensor(READY_QPOS, device=self.device)
            for agent in (self.agent_a, self.agent_b):
                noise = torch.randn((b, start_qpos.shape[-1])) * self.robot_init_qpos_noise
                agent.reset(start_qpos + noise)

            # Cube: fixed at the table center with a small xy jitter. +-1 cm,
            # not +-2 cm: the top-down grasp pose has 1.9 cm of reach margin
            # (see ARM_BASE_OFFSET) and the jitter has to fit inside it in
            # EVERY episode, not on average.
            cube_xyz = torch.zeros((b, 3))
            cube_xyz[:, :2] = torch.rand((b, 2)) * 0.02 - 0.01
            cube_xyz[:, 2] = CUBE_HALF_SIZE
            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz))

    def _get_obs_extra(self, info: dict):
        # Own qpos/qvel per agent already come from the default
        # _get_obs_agent (MultiAgent.get_proprioception, keyed per sub-agent
        # uid) -- this only adds what BaseEnv doesn't already provide: TCP
        # poses (gated by partner_obs) and cube pose (always present; it's
        # task-object state, not "partner" info).
        #
        # obs_mode="state" flattens this whole dict into one combined tensor
        # (mani_skill.utils.common.flatten_state_dict), so there is no
        # per-agent split at this level either way -- see
        # callosum.envs._partner_obs for exactly what partner_obs does and
        # doesn't control here.
        obs = partner_tcp_pose_fields(
            self.partner_obs, self.agent_a.tcp_pose.raw_pose, self.agent_b.tcp_pose.raw_pose
        )
        if "state" in self.obs_mode:
            obs["cube_pose"] = self.cube.pose.raw_pose
        return obs

    def evaluate(self):
        # No task/goal defined at this step -- see step 1.3 for the face-turn
        # task and its real success condition.
        return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Simple reach reward: negative summed distance from each TCP to the
        # cube, just to confirm both arms learn to reach for it. Real task
        # reward (grasp/twist/hold terms) arrives in step 1.3.
        dist_a = torch.linalg.norm(self.agent_a.tcp_pos - self.cube.pose.p, axis=1)
        dist_b = torch.linalg.norm(self.agent_b.tcp_pos - self.cube.pose.p, axis=1)
        return -(dist_a + dist_b)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # 0.5 m/arm: SO-100's own max kinematic reach, from summing the URDF
        # joint-origin offsets from base to jaw tip (~0.498 m); used here only
        # to keep the normalized reward roughly within [-1, 0].
        max_dist_per_arm = 0.5
        return self.compute_dense_reward(obs=obs, action=action, info=info) / (2 * max_dist_per_arm)
