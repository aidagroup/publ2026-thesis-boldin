"""Scripted two-arm expert for FaceTurn-v0, shared by `probe_face_turn.py` and `render_episode.py`.

There is no teleporting or pinning of anything: both arms are driven only through their
`pd_joint_pos` controllers and the cube is only ever touched by the grippers. Create the env with
`control_mode="pd_joint_pos"`, wrap it in a `Rig` and call `run_face_turn`.

Strategy (arms stand 90 degrees apart around the cube, see `callosum.configs.layout`):

  1. `holder pregrasp`, `holder grasp`: the holder (agent_a) comes in low and almost horizontal,
     fingers pointing at the cube 10 degrees below horizontal and jaws open horizontally across
     the approach, and clamps the body (the lower layer, below the face) from its sides. It grips
     2 cm in front of the body centre so that its wrist housing stays clear of the rotator.
  2. `rotator pregrasp`, `rotator grasp`: only then does the rotator (agent_b) come in top-down
     (fingers 6 degrees off vertical), lowers its open jaws around the face layer and clamps it
     from its sides. The two arms must not have their open jaws near the cube at the same time.
  3. `turn`: the rotator rolls its wrist by -90 degrees (a bit more, so the face joint's limit
     stops the face at exactly 90 degrees); the face turns by +90 degrees about the vertical axis
     while the holder keeps the body still.
  4. `release`: the rotator opens its jaws; success is evaluated again.

Joint targets come from a small numpy IK of the URDF (`ArmModel`); the cube pose is read from
the sim (privileged).

Not part of the `callosum` package because it touches `mani_skill` (Linux-only); the scripts
import it as a sibling module.
"""

import math
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from mani_skill.utils import common
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from callosum.envs.two_so101_base import NUM_ARM_JOINTS
from callosum.robots.so101_parallel_gripper import SO101ParallelGripper

# Strategy parameters (metres / radians). The body is 3.8 cm tall (mid height 1.9 cm), the face
# layer on top of it spans z = 3.8 .. 5.7 cm. The jaw pads are 1.9 cm tall and 2.5 cm long, and
# the finger body behind each pad is 4.5 cm tall, so the holder's TCP height has a window of only
# a few mm: lower and the fingers rub on the table, higher and the pads touch the face layer.
HOLDER_GRASP_OFFSET = 0.02  # holder TCP sits this far from the body centre, towards the holder
HOLDER_GRASP_Z = 0.024  # TCP height when clamping the body (about its mid height)
HOLDER_DEPRESSION = np.deg2rad(10)  # holder fingers point this far below horizontal
ROTATOR_GRASP_Z = 0.056  # TCP height when clamping the face layer (mid height is 0.0475)
ROTATOR_TILT = np.deg2rad(6)  # rotator fingers are this far off vertical, towards the cube
HOLDER_BACKOFF = 0.03  # holder pre-grasp: this far back along the approach direction ...
HOLDER_LIFT = 0.01  # ... and this far above the grasp pose (jaws open)
ROTATOR_LIFT = 0.04  # rotator pre-grasp: this far straight above the grasp pose
MAX_JOINT_STEP = 0.03  # rad per control step while moving between waypoints
IK_POSITION_TOL = 3e-3  # m
IK_ANGLE_TOL = np.deg2rad(8)
SETTLE_STEPS = 10
PATH_STEPS = 8  # IK waypoints on each straight-line approach
GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, -1.0  # normalised absolute gripper action
# Wrist roll that turns the face by +pi/2 (checked in sim). It overshoots by 0.1 rad on purpose:
# the face joint's upper limit (pi/2) stops the face exactly at the target, so a slightly
# slipping grip does not leave the angle short of the tolerance.
ROLL_ANGLE = -math.pi / 2 - 0.1

_UP = np.array([0.0, 0.0, 1.0])


class ArmModel:
    """Numpy forward kinematics and IK of the SO-ARM101 + parallel gripper (from the URDF).

    Everything is expressed in the arm's base frame. The "tool" is `link5`: its TCP (midpoint of
    the two jaw pads), the direction the fingers point (-y of link5) and the jaw opening axis
    (+x of link5).
    """

    def __init__(self) -> None:
        root = ET.parse(SO101ParallelGripper.urdf_path).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}

        def vec(joint: ET.Element, tag: str, attr: str) -> np.ndarray:
            return np.array(joint.find(tag).get(attr).split(), dtype=float)

        arm = [joints[n] for n in SO101ParallelGripper.arm_joint_names]
        self.origins = [vec(j, "origin", "xyz") for j in arm]
        axes = [vec(j, "axis", "xyz") for j in arm]
        self.axes = [a / np.linalg.norm(a) for a in axes]
        self.lower = np.array([float(j.find("limit").get("lower")) for j in arm])
        self.upper = np.array([float(j.find("limit").get("upper")) for j in arm])

        # TCP in the link5 frame: both clamp joints sit at their origin at q=0 and the jaws move
        # symmetrically, so the pad midpoint does not depend on the gripper opening.
        pads = [
            vec(joints[clamp_joint], "origin", "xyz") + vec(joints[pad_joint], "origin", "xyz")
            for clamp_joint, pad_joint in (
                ("right_clamp", "clamp_1_to_pad"),
                ("left_clamp", "clamp_2_to_pad"),
            )
        ]
        self.tcp_local = (pads[0] + pads[1]) / 2
        # The fingers point against the roll axis; the jaws open along the clamp joints' axis.
        self.finger_dir_local = -self.axes[-1]
        clamp_axis = vec(joints["right_clamp"], "axis", "xyz")
        self.open_axis_local = clamp_axis / np.linalg.norm(clamp_axis)

    def tool(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """TCP position, finger direction and jaw-opening axis (base frame) at arm angles `q`."""
        rot, pos = np.eye(3), np.zeros(3)
        for origin, axis, angle in zip(self.origins, self.axes, q, strict=True):
            pos = pos + rot @ origin
            rot = rot @ Rotation.from_rotvec(axis * angle).as_matrix()
        return (
            pos + rot @ self.tcp_local,
            rot @ self.finger_dir_local,
            rot @ self.open_axis_local,
        )

    def solve(
        self,
        tcp: np.ndarray,
        finger_dir: np.ndarray,
        open_axis: np.ndarray,
        q_init: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """IK: arm angles putting the TCP at `tcp` with the given finger/opening directions.

        The arm is one yaw joint plus three pitch joints and a roll, so it cannot realise an
        arbitrary finger direction on top of an arbitrary TCP position: the azimuth of the
        fingers follows from the TCP position. The position is therefore weighted much higher
        than the directions. Returns `(q, position_error_m, direction_error_rad)` with the
        direction error being the worse of the finger direction and the jaw opening axis.
        """

        def residual(q: np.ndarray) -> np.ndarray:
            p, d, o = self.tool(q)
            return np.concatenate([(p - tcp) * 100, d - finger_dir, 0.5 * (o - open_axis)])

        sol = least_squares(residual, q_init, bounds=(self.lower, self.upper))
        p, d, o = self.tool(sol.x)
        angle = max(
            np.arccos(np.clip(d @ finger_dir, -1, 1)), np.arccos(np.clip(o @ open_axis, -1, 1))
        )
        return sol.x, float(np.linalg.norm(p - tcp)), float(angle)


@dataclass
class Rig:
    """Wraps the env: base poses, commanded targets, and stepping with `pd_joint_pos`."""

    env: object
    model: ArmModel
    # Called with the observation after every `env.step` (e.g. to grab a video frame); the
    # expert itself does not use observations.
    on_step: Callable[[dict], None] | None = None

    def __post_init__(self) -> None:
        self.base = self.env.unwrapped
        self.n = self.base.num_envs
        self.agents = (self.base.agent_a, self.base.agent_b)
        # Action-dict keys, e.g. `so101_pg-0` / `so101_pg-1` (`so101_pg_wristcam-*` with cameras).
        self.uids = tuple(self.base.agent.agents_dict)
        # Commanded arm targets (n, 5) and gripper targets (n,) per arm.
        qpos = [a.robot.get_qpos().cpu().numpy() for a in self.agents]
        self.q = [x[:, :NUM_ARM_JOINTS].copy() for x in qpos]
        self.grip = [np.full(self.n, GRIPPER_OPEN) for _ in self.agents]
        self.last_info: dict = {}

    def base_frame(self, arm: int) -> tuple[np.ndarray, np.ndarray]:
        """World poses of an arm's base: rotations (n, 3, 3) and positions (n, 3)."""
        mat = self.agents[arm].robot.pose.to_transformation_matrix().cpu().numpy()
        return mat[:, :3, :3], mat[:, :3, 3]

    def ik(self, arm: int, tcp_w, finger_dir_w, open_axis_w, q_init=None) -> np.ndarray:
        """Per-env IK for world-frame targets (each `(n, 3)`); raises if a target is unreachable.

        Starts from `q_init` (default: the current targets); only if that fails does it fall
        back to random restarts. Among valid solutions it prefers the one closest to `q_init`.
        """
        rot, pos = self.base_frame(arm)
        q_init = self.q[arm] if q_init is None else q_init
        out = np.zeros((self.n, NUM_ARM_JOINTS))
        for i in range(self.n):
            to_base = rot[i].T
            tcp = to_base @ (tcp_w[i] - pos[i])
            finger_dir = to_base @ finger_dir_w[i]
            # The jaws are symmetric: flipping the opening axis is the same grasp, so try both.
            axes = [to_base @ open_axis_w[i] * sign for sign in (1.0, -1.0)]
            best = self._solve_best(tcp, finger_dir, axes, [q_init[i]], q_init[i])
            if best is None:
                rng = np.random.default_rng(0)
                seeds = [rng.uniform(self.model.lower, self.model.upper) for _ in range(10)]
                best = self._solve_best(tcp, finger_dir, axes, seeds, q_init[i])
            if best is None:
                raise RuntimeError(f"IK failed for arm {arm}, env {i}")
            out[i] = best
        return out

    def _solve_best(self, tcp, finger_dir, axes, seeds, q_ref) -> np.ndarray | None:
        """The valid IK solution closest to `q_ref` over all seeds and opening axes, if any."""
        best, best_dist = None, np.inf
        for axis in axes:
            for seed in seeds:
                q, pos_err, angle_err = self.model.solve(tcp, finger_dir, axis, seed)
                dist = np.abs(q - q_ref).max()
                if pos_err < IK_POSITION_TOL and angle_err < IK_ANGLE_TOL and dist < best_dist:
                    best, best_dist = q, dist
        return best

    def tool_world(self, arm: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World TCP, finger direction and jaw axis (each `(n, 3)`) at the commanded targets."""
        rot, pos = self.base_frame(arm)
        out = [self.model.tool(self.q[arm][i]) for i in range(self.n)]
        tcp, finger, axis = (np.stack(x) for x in zip(*out, strict=True))
        return (
            np.einsum("nij,nj->ni", rot, tcp) + pos,
            np.einsum("nij,nj->ni", rot, finger),
            np.einsum("nij,nj->ni", rot, axis),
        )

    def path(self, arm: int, tcp, finger_dir, open_axis, steps: int = PATH_STEPS) -> list:
        """Joint waypoints that move the tool in a straight line to the goal pose.

        Interpolates the TCP linearly (and the two direction vectors, renormalised) from the
        current commanded pose and solves IK at each waypoint, seeded with the previous one, so
        the fingers do not swing through the cube the way a joint-space interpolation can.
        """
        start = self.tool_world(arm)
        goal = (tcp, finger_dir, open_axis)
        qs, q_prev = [], self.q[arm]
        for k in range(1, steps + 1):
            t = k / steps
            tcp_k, d_k, o_k = (a + (b - a) * t for a, b in zip(start, goal, strict=True))
            d_k = d_k / np.linalg.norm(d_k, axis=1, keepdims=True)
            o_k = o_k / np.linalg.norm(o_k, axis=1, keepdims=True)
            q_prev = self.ik(arm, tcp_k, d_k, o_k, q_prev)
            qs.append(q_prev)
        return qs

    def follow(self, arm: int, waypoints: list, settle: int = 0) -> None:
        """Drive one arm through joint waypoints (the other arm keeps its targets)."""
        for q in waypoints:
            self.move(**{("q_a", "q_b")[arm]: q}, settle=0)
        for _ in range(settle):
            self.step()

    def step(self) -> None:
        action = {}
        for arm, uid in enumerate(self.uids):
            act = np.concatenate([self.q[arm], self.grip[arm][:, None]], axis=1)
            action[uid] = torch.as_tensor(act, dtype=torch.float32, device=self.base.device)
        obs, _, _, _, self.last_info = self.env.step(action)
        if self.on_step is not None:
            self.on_step(obs)

    def move(self, q_a=None, q_b=None, grip_a=None, grip_b=None, settle=SETTLE_STEPS) -> None:
        """Linearly interpolate the arms to new joint targets (and set grippers), then settle."""
        goals = [q_a, q_b]
        for arm, g in enumerate((grip_a, grip_b)):
            if g is not None:
                self.grip[arm] = np.full(self.n, g)
        start = [q.copy() for q in self.q]
        delta = max(
            (np.abs(g - s).max() for g, s in zip(goals, start, strict=True) if g is not None),
            default=0.0,
        )
        steps = max(1, math.ceil(delta / MAX_JOINT_STEP))
        for k in range(1, steps + 1):
            for arm, goal in enumerate(goals):
                if goal is not None:
                    self.q[arm] = start[arm] + (goal - start[arm]) * k / steps
            self.step()
        for _ in range(settle):
            self.step()

    def report(self, phase: str) -> dict:
        """Print the per-phase status line and return it as a dict of numpy values."""
        b = self.base
        info = b.evaluate()
        body = b.cube.links_map["body"]
        face = b.face_link
        pos_drift = torch.linalg.norm(b.cube.pose.p - b.body_init_pos, dim=1)
        rot_drift = common.quat_diff_rad(b.cube.pose.q, b.body_init_q)
        row = {
            "holder_grasps_body": self.agents[0].is_grasping(body).cpu().numpy(),
            "rotator_grasps_face": self.agents[1].is_grasping(face).cpu().numpy(),
            "face_angle_deg": np.rad2deg(info["face_angle"].cpu().numpy()),
            "body_drift_cm": pos_drift.cpu().numpy() * 100,
            "body_rot_deg": np.rad2deg(rot_drift.cpu().numpy()),
            "success": info["success"].cpu().numpy(),
        }
        print(f"[{phase}]")
        for key, val in row.items():
            if val.dtype.kind == "b":
                shown = str(bool(val[0])) if len(val) == 1 else f"{int(val.sum())}/{len(val)}"
            elif len(val) == 1:
                shown = f"{val[0]:.2f}"
            else:
                shown = f"{val.min():.2f} / {val.mean():.2f} / {val.max():.2f}"
            print(f"  {key:>20}: {shown}")
        return row


def _horizontal_unit(rig: Rig, arm: int) -> np.ndarray:
    """World direction (n, 3) from an arm's base towards the cube, flattened to the table."""
    _, base_pos = rig.base_frame(arm)
    cube = rig.base.cube.pose.p.cpu().numpy()
    direction = cube[:, :2] - base_pos[:, :2]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return np.concatenate([direction, np.zeros((rig.n, 1))], axis=1)


def run_face_turn(rig: Rig, on_phase: Callable[[str], None] | None = None) -> None:
    """Run the whole scripted face turn from the freshly reset env state.

    `on_phase(name)` is called after each phase (`holder pregrasp`, `holder grasp`,
    `rotator pregrasp`, `rotator grasp`, `turn`, `release`), e.g. to print `rig.report(name)`.
    """

    def phase(name: str) -> None:
        if on_phase is not None:
            on_phase(name)

    cube_xy = rig.base.cube.pose.p.cpu().numpy().copy()
    cube_xy[:, 2] = 0.0

    # Holder: fingers point at the cube, tilted down; jaws open horizontally across the approach.
    fwd_a = _horizontal_unit(rig, 0)
    dir_a = np.cos(HOLDER_DEPRESSION) * fwd_a - np.sin(HOLDER_DEPRESSION) * _UP
    open_a = np.cross(_UP, fwd_a)
    grasp_a = cube_xy - HOLDER_GRASP_OFFSET * fwd_a + HOLDER_GRASP_Z * _UP
    pregrasp_a = grasp_a - HOLDER_BACKOFF * fwd_a + HOLDER_LIFT * _UP

    # Rotator: fingers point (almost) straight down, jaws across the approach direction.
    fwd_b = _horizontal_unit(rig, 1)
    dir_b = np.sin(ROTATOR_TILT) * fwd_b - np.cos(ROTATOR_TILT) * _UP
    open_b = np.cross(_UP, fwd_b)
    grasp_b = cube_xy + ROTATOR_GRASP_Z * _UP
    pregrasp_b = grasp_b + ROTATOR_LIFT * _UP

    # The holder goes first: while its jaws are open next to the cube they would hit the
    # rotator's open jaws, so the rotator only comes in once the holder has clamped the body.
    rig.follow(0, rig.path(0, pregrasp_a, dir_a, open_a), settle=SETTLE_STEPS)
    phase("holder pregrasp")

    rig.follow(0, rig.path(0, grasp_a, dir_a, open_a), settle=SETTLE_STEPS)
    rig.move(grip_a=GRIPPER_CLOSED, settle=20)
    phase("holder grasp")

    rig.follow(1, rig.path(1, pregrasp_b, dir_b, open_b), settle=SETTLE_STEPS)
    phase("rotator pregrasp")

    q_b_grasp = rig.path(1, grasp_b, dir_b, open_b)
    rig.follow(1, q_b_grasp, settle=SETTLE_STEPS)
    rig.move(grip_b=GRIPPER_CLOSED, settle=20)
    phase("rotator grasp")

    q_b_turned = q_b_grasp[-1].copy()
    q_b_turned[:, -1] += ROLL_ANGLE
    rig.move(q_b=q_b_turned, settle=30)
    phase("turn")

    rig.move(grip_b=GRIPPER_OPEN, settle=20)
    phase("release")
