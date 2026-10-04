"""Two-arm SO-ARM101 base environment: plumbing for observations/actions/reward.

Sets up a bimanual scene (two SO-ARM101 arms with Robonine parallel grippers + a loose cube)
with per-agent observation fields and a reach-only dense reward, ahead of the articulated
face-turn task in step 1.3. There is no task/goal yet -- `evaluate()` is a
stub that always reports failure.
"""

from typing import Any, ClassVar

import sapien
import torch
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from callosum.configs.cameras import SceneCameraConfig, look_at_pose
from callosum.configs.layout import ArmLayout, base_xy_yaw
from callosum.envs._partner_obs import partner_tcp_pose_fields, validate_partner_obs
from callosum.robots.so101_parallel_gripper import SO101ParallelGripper

# ~5.7 cm real Rubik's cube edge length.
CUBE_HALF_SIZE = 0.0285

_DEFAULT_ARM_LAYOUT = ArmLayout()
_DEFAULT_SCENE_CAMERA = SceneCameraConfig()
# Number of arm (non-gripper) joints; the gripper joints come after them in qpos.
NUM_ARM_JOINTS = len(SO101ParallelGripper.arm_joint_names)
# Max distance from the shoulder (joint 2) to the TCP over the joint limits: ~0.494 m (numpy FK
# sampling + optimisation over the URDF), rounded up. Used only to normalise the reward.
MAX_REACH_PER_ARM = 0.5


@register_env("TwoSO101-v0", max_episode_steps=100)
class TwoSO101Base(BaseEnv):
    """Two SO-ARM101 arms (parallel grippers) around a table with a single loose cube.

    The bases stand 90 degrees apart on circles around the table centre (the cube), both
    pointing at it; the placement is the `arm_layout` argument (`ArmLayout`).

    `robot_uids` is `("so101_pg", "so101_pg")` (no sensors; use this for state-only training with
    `render_backend="none"`) or `("so101_pg_wristcam", "so101_pg_wristcam")` (one wrist camera per
    arm). The fixed third-person "human render camera" (`render_camera`, used by
    `render_mode="rgb_array"`) is the `scene_camera` argument (`SceneCameraConfig`).

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

    # `so101_pg_wristcam` is `so101_pg` plus a wrist camera per arm (for rendering / vision).
    SUPPORTED_ROBOTS: ClassVar[list[tuple[str, str]]] = [
        ("so101_pg", "so101_pg"),
        ("so101_pg_wristcam", "so101_pg_wristcam"),
    ]
    agent: MultiAgent[tuple[SO101ParallelGripper, SO101ParallelGripper]]

    def __init__(
        self,
        *args,
        robot_uids=("so101_pg", "so101_pg"),
        robot_init_qpos_noise=0.02,
        partner_obs="full",
        arm_layout: ArmLayout = _DEFAULT_ARM_LAYOUT,
        scene_camera: SceneCameraConfig = _DEFAULT_SCENE_CAMERA,
        **kwargs,
    ):
        validate_partner_obs(partner_obs)
        self.arm_layout = arm_layout
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.partner_obs = partner_obs
        self.scene_camera = scene_camera
        # No explicit control_mode: SO101ParallelGripper's first configured controller is
        # already "pd_joint_delta_pos" (5 arm deltas + 1 gripper target), which BaseAgent
        # picks by default whenever control_mode is None.
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        cfg = self.scene_camera
        position, quat = look_at_pose(cfg.eye, cfg.target)
        return CameraConfig(
            "render_camera",
            sapien.Pose(p=position, q=quat),
            cfg.width,
            cfg.height,
            cfg.fov_y,
            cfg.near,
            cfg.far,
        )

    @property
    def agent_a(self) -> SO101ParallelGripper:
        """The first arm (uid `so101_pg-0`, or `so101_pg_wristcam-0`)."""
        return self.agent.agents[0]

    @property
    def agent_b(self) -> SO101ParallelGripper:
        """The second arm (uid `so101_pg-1`, or `so101_pg_wristcam-1`)."""
        return self.agent.agents[1]

    def _load_agent(self, options: dict):
        # Both bases stand on a circle around the table centre (the cube) and point at it; see
        # callosum.configs.layout.ArmLayout for the radius and azimuths.
        layout = self.arm_layout
        poses = []
        for radius, azimuth in (
            (layout.radius_a, layout.azimuth_a),
            (layout.radius_b, layout.azimuth_b),
        ):
            x, y, yaw = base_xy_yaw(radius, azimuth)
            poses.append(sapien.Pose(p=[x, y, 0], q=euler2quat(0, 0, yaw)))
        super()._load_agent(options, poses)

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
            # no branch for robot_uids == ("so101_pg", "so101_pg") (nor the wristcam
            # variant) as of mani-skill 3.0.1, so it silently skips robot placement -- both
            # arms' qpos are reset explicitly below instead. Base poses are
            # fixed (set once in _load_agent) and do not need resetting per
            # episode.
            self.table_scene.initialize(env_idx)

            rest_qpos = common.to_tensor(self.agent_a.keyframes["rest"].qpos, device=self.device)
            for agent in (self.agent_a, self.agent_b):
                # Noise on the arm joints only: the two gripper joints are prismatic (metres),
                # and the mimic joint must stay exactly -right_clamp.
                noise = torch.zeros((b, rest_qpos.shape[-1]))
                noise[:, :NUM_ARM_JOINTS] = (
                    torch.randn((b, NUM_ARM_JOINTS)) * self.robot_init_qpos_noise
                )
                agent.reset(rest_qpos + noise)

            # Cube: fixed at the table center with a small xy jitter.
            cube_xyz = torch.zeros((b, 3))
            cube_xyz[:, :2] = torch.rand((b, 2)) * 0.04 - 0.02
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
        # MAX_REACH_PER_ARM: the arm's max kinematic reach (shoulder to TCP); used here only
        # to keep the normalized reward roughly within [-1, 0].
        return self.compute_dense_reward(obs=obs, action=action, info=info) / (
            2 * MAX_REACH_PER_ARM
        )
