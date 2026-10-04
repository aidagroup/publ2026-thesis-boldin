"""SO-ARM101 arm with the Robonine parallel gripper, as a ManiSkill agent (uid `so101_pg`).

Simulation-only robot. Geometry comes from
https://github.com/roboninecom/SO-ARM100-101-Parallel-Gripper (see
`callosum/assets/so101_parallel_gripper/README.md` for source commit, licences and the list of
modifications).

Frame conventions (robot base frame, `base_link`): at qpos = 0 the arm is folded and reaches
towards **-y**; the two jaws point along -y and open along x (`clamp_1` moves +x as
`right_clamp` grows, `clamp_2` moves -x as `left_clamp` shrinks). The two gripper joints are
driven by one scalar action through a mimic controller (`left_clamp = -right_clamp`).
"""

import copy
from pathlib import Path
from typing import ClassVar

import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
)
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from callosum.configs.cameras import WristCameraConfig

_ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "so101_parallel_gripper"

# Upper limit of `right_clamp` (metres). `left_clamp` mirrors it with the opposite sign.
GRIPPER_MAX_OPENING = 0.037


@register_agent()
class SO101ParallelGripper(BaseAgent):
    """SO-ARM101 (5 revolute joints) + Robonine parallel gripper (2 mirrored prismatic jaws).

    Active joint order is `arm_joint_names + gripper_joint_names`. The default controller is
    `pd_joint_delta_pos`: 5 arm deltas (+-0.05 rad per step) and one *absolute* gripper target,
    i.e. a 6-D action in [-1, 1] (normalised, as for Panda). The gripper entry maps linearly to
    the opening of `right_clamp` in `[0, GRIPPER_MAX_OPENING]` (-1 closed, +1 fully open), so a
    zero action half-closes the gripper (0.0185 m per jaw). `pd_joint_pos` (absolute arm angles
    in radians) and `pd_joint_target_delta_pos` are also available.
    """

    uid = "so101_pg"
    urdf_path = str(_ASSET_DIR / "so101_parallel_gripper.urdf")
    urdf_config: ClassVar[dict] = {
        "_materials": {
            "gripper": {"static_friction": 2.0, "dynamic_friction": 2.0, "restitution": 0.0}
        },
        "link": {
            "clamp_1": {"material": "gripper", "patch_radius": 0.1, "min_patch_radius": 0.1},
            "clamp_2": {"material": "gripper", "patch_radius": 0.1, "min_patch_radius": 0.1},
        },
    }

    # qpos order: 5 arm joints, then right_clamp, left_clamp (checked against
    # `robot.active_joints`). `rest`: fingers pointing down (tilted ~20 degrees towards
    # the front), TCP ~0.18 m in front of the base axis and ~0.10 m above the table, jaws half
    # open. Derived with a numpy forward-kinematics solve of the URDF, then checked in sim.
    keyframes: ClassVar[dict[str, Keyframe]] = {
        "rest": Keyframe(
            qpos=np.array([-0.055, -0.90, 1.20, -1.52, 0.045, 0.035, -0.035]),
            pose=sapien.Pose(),
        ),
        "zero": Keyframe(qpos=np.zeros(7), pose=sapien.Pose()),
    }

    arm_joint_names: ClassVar[list[str]] = [
        "base_link_to_link1",
        "link1_to_link2",
        "link2_to_link3",
        "link3_to_link4",
        "link4_to_link5",
    ]
    gripper_joint_names: ClassVar[list[str]] = ["right_clamp", "left_clamp"]

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        # Max delta of 0.05 rad/step, as for the SO-100 (cheap servos, avoid shaking).
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.05,
            upper=0.05,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = copy.deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # One scalar action: the opening of `right_clamp` in metres. In
        # PDJointPosMimicControllerConfig the `mimic` dict maps the mimic joint (key) to its
        # controlling joint (`"joint"`); the action space has one dim per controlling joint, with
        # limits taken from `lower`/`upper` (the controlling joint's range).
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=0.0,
            upper=GRIPPER_MAX_OPENING,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"left_clamp": {"joint": "right_clamp", "multiplier": -1.0}},
        )

        controller_configs = {
            "pd_joint_delta_pos": {"arm": arm_pd_joint_delta_pos, "gripper": gripper_pd_joint_pos},
            "pd_joint_pos": {"arm": arm_pd_joint_pos, "gripper": gripper_pd_joint_pos},
            "pd_joint_target_delta_pos": {
                "arm": arm_pd_joint_target_delta_pos,
                "gripper": gripper_pd_joint_pos,
            },
        }
        return copy.deepcopy(controller_configs)

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["clamp_1"]
        self.finger2_link = self.robot.links_map["clamp_2"]
        self.finger1_pad = self.robot.links_map["clamp_1_pad"]
        self.finger2_pad = self.robot.links_map["clamp_2_pad"]

    @property
    def tcp_pos(self) -> torch.Tensor:
        """Tool centre point: midpoint between the two jaw pads, shape (N, 3)."""
        return (self.finger1_pad.pose.p + self.finger2_pad.pose.p) / 2

    @property
    def tcp_pose(self) -> Pose:
        """TCP position with the orientation of `clamp_1` (as for the SO-100)."""
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    def is_grasping(self, object: Actor, min_force: float = 0.5, max_angle: float = 85.0):
        """Check whether both jaws press on `object`.

        Follows Panda's convention: each jaw must feel at least `min_force` newtons from the
        object, and the contact force must point within `max_angle` degrees of the jaw's
        *opening* direction (+x of `clamp_1`, -x of `clamp_2`), i.e. the object is squeezed
        between the jaws rather than e.g. only touched from the side.

        Args:
            object: the actor (or link) to test.
            min_force: minimum contact force per jaw, in newtons.
            max_angle: maximum angle between jaw opening direction and contact force, degrees.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_contact_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 0]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 0]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(lforce >= min_force, torch.rad2deg(langle) <= max_angle)
        rflag = torch.logical_and(rforce >= min_force, torch.rad2deg(rangle) <= max_angle)
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        """True where all arm joints (gripper excluded) move slower than `threshold` rad/s."""
        qvel = self.robot.get_qvel()[:, :-2]  # exclude the 2 gripper joints
        return torch.max(torch.abs(qvel), 1)[0] <= threshold


@register_agent()
class SO101ParallelGripperWristCam(SO101ParallelGripper):
    """`SO101ParallelGripper` plus a wrist camera on the gripper housing (uid `so101_pg_wristcam`).

    Opt-in on purpose: state-only training (`render_backend="none"`) keeps using `so101_pg`,
    which has no sensors, so nothing is rendered there. With this agent every arm adds one
    camera, which a `MultiAgent` env exposes as `<agent uid>-<index>-<camera uid>`, e.g.
    `so101_pg_wristcam-0-wrist`, in `obs["sensor_data"]` (for a visual `obs_mode`).

    The mount pose, image size and field of view come from `wrist_camera`
    (`callosum.configs.cameras.WristCameraConfig`; override on a subclass to change the camera).
    Per episode they can also be overridden through the env's `sensor_configs` argument, e.g.
    `sensor_configs=dict(width=320, height=240)` for all cameras.
    """

    uid = "so101_pg_wristcam"

    wrist_camera: ClassVar[WristCameraConfig] = WristCameraConfig()

    # TODO(review): the mount pose was checked geometrically only (jaw pads project into the
    # image, the camera is outside the housing/jaw meshes, see tests/test_wrist_camera.py); the
    # rendered images could not be checked on the Mac. The camera body itself is not modelled.
    @property
    def _sensor_configs(self):
        cfg = self.wrist_camera
        position, quat = cfg.pose_in_mount()
        return [
            CameraConfig(
                uid=cfg.uid,
                pose=sapien.Pose(p=position, q=quat),
                width=cfg.width,
                height=cfg.height,
                fov=cfg.fov_y,
                near=cfg.near,
                far=cfg.far,
                mount=self.robot.links_map[cfg.mount_link],
            )
        ]
