"""Scripted two-arm expert for FaceTurn-v0, importable by the probe and the demo collector.

There is no teleporting or pinning of anything: both arms are driven only through their joint
controllers and the cube is only ever touched by the grippers. Strategy (arms stand 90 degrees
apart around the cube, see `callosum.configs.layout`):

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

Two ways of driving the arms (`ExpertControlConfig.control`):

* `"pos"`: the env uses `pd_joint_pos`; the commanded joint targets move at most
  `max_joint_step` per step, open loop with fixed settle times (the original probe).
* `"delta"`: the env uses `pd_joint_delta_pos`, the controller the policies are trained with.
  Its target is the *current* joint position plus the action (+-`ARM_DELTA_LIMIT` rad), so the
  arm can move at most about 0.02 rad per step (first-order PD lag). The expert tracks the same
  waypoints closed loop: the action is the remaining joint error, scaled to saturate at +-1
  with the direction kept (the tool keeps moving on the straight line between waypoints), a
  waypoint is left when the joints are within a tolerance (or after a timeout), and settling
  ends as soon as everything is still. The gripper action is absolute (+-1) in both modes.

  Under this controller a disturbance (or a joint at its limit) leaves a steady-state error, so
  a loop that waits for every joint of every env would end only by its timeout. The delta mode
  therefore (see `callosum.experts._tracking` and `ExpertControlConfig`): adds an integral bias
  per joint to the commanded offset, stops waiting for envs that are stalled, advances through
  the intermediate waypoints of a path with a loose tolerance, flies through the pre-grasp
  without settling, keeps the IK away from the joint limits, and ends the turn as soon as the
  face is at its target (the wrist target overshoots, so it is never reached). The reach loops
  are logged in `Rig.reach_log` (the probe prints a summary per phase).

`Rig.step` reports every env step to registered callbacks (observation before the step, the
action dict that was passed to `env.step`, reward, terminated, truncated, info), which is what
`scripts/collect_demos.py` records.

Needs mani_skill and scipy (not importable on the CI runner); the pure-Python tracking config is
`callosum.configs.face_turn_expert`.
"""

import math
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from callosum.configs.face_turn_expert import ExpertControlConfig
from callosum.envs.face_turn import TARGET_FACE_ANGLE
from callosum.envs.two_so101_base import NUM_ARM_JOINTS
from callosum.experts._noise import perturb_arm_action
from callosum.experts._tracking import (
    ReachMonitor,
    ReachRecord,
    reach_reason,
    tracking_action,
    update_bias,
)
from callosum.robots.so101_parallel_gripper import ARM_DELTA_LIMIT, SO101ParallelGripper

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
IK_POSITION_TOL = 3e-3  # m
IK_ANGLE_TOL = np.deg2rad(8)
# Delta mode: a joint-limit-margin IK solution farther than this (rad, largest joint) from the
# previous waypoint is discarded in favour of the unrestricted search (see `Rig.ik`).
MARGIN_MAX_JUMP = 2.0
SETTLE_STEPS = 10  # settle after a movement phase (the delta mode waits less, see the config)
PATH_STEPS = 8  # IK waypoints on each straight-line approach
GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, -1.0  # normalised absolute gripper action
# Wrist roll that turns the face by +pi/2 (checked in sim). It overshoots by 0.1 rad on purpose:
# the face joint's upper limit (pi/2) stops the face exactly at the target, so a slightly
# slipping grip does not leave the angle short of the tolerance.
ROLL_ANGLE = -math.pi / 2 - 0.1
# Delta mode: the turn is done once the face is this close (rad) to its target (the env's success
# tolerance is 0.05); the wrist keeps pushing until the arm is told to hold.
TURN_DONE_MARGIN = 0.03

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
        margin: float = 0.0,
    ) -> tuple[np.ndarray, float, float]:
        """IK: arm angles putting the TCP at `tcp` with the given finger/opening directions.

        The arm is one yaw joint plus three pitch joints and a roll, so it cannot realise an
        arbitrary finger direction on top of an arbitrary TCP position: the azimuth of the
        fingers follows from the TCP position. The position is therefore weighted much higher
        than the directions. Returns `(q, position_error_m, direction_error_rad)` with the
        direction error being the worse of the finger direction and the jaw opening axis.
        With `margin > 0` the solution keeps that far (rad) inside the joint limits.
        """

        def residual(q: np.ndarray) -> np.ndarray:
            p, d, o = self.tool(q)
            return np.concatenate([(p - tcp) * 100, d - finger_dir, 0.5 * (o - open_axis)])

        if margin > 0:
            lower, upper = self.lower + margin, self.upper - margin
            q_init = np.clip(q_init, lower, upper)
        else:
            lower, upper = self.lower, self.upper
        sol = least_squares(residual, q_init, bounds=(lower, upper))
        p, d, o = self.tool(sol.x)
        angle = max(
            np.arccos(np.clip(d @ finger_dir, -1, 1)), np.arccos(np.clip(o @ open_axis, -1, 1))
        )
        return sol.x, float(np.linalg.norm(p - tcp)), float(angle)


class ExpertFinished(Exception):
    """Raised by `Rig.step` when the run should stop (all envs succeeded, or the step budget)."""


# (obs before the step, action dict, reward, terminated, truncated, info) of one env step.
StepCallback = Callable[[torch.Tensor, dict, torch.Tensor, torch.Tensor, torch.Tensor, dict], None]


@dataclass
class Rig:
    """Wraps the env: commanded targets and stepping in either control mode (see the module doc).

    Args:
        env: the FaceTurn-v0 env, already reset; created with `control_mode="pd_joint_pos"` for
            `control.control == "pos"` and `"pd_joint_delta_pos"` for `"delta"`.
        model: the numpy arm model used for IK.
        control: tracking parameters and the control mode.
        obs: the observation returned by the env's reset (kept for the step callbacks).
        max_steps: raise `ExpertFinished` after this many env steps (`None`: no limit).
        stop_on_success: raise `ExpertFinished` once every env reports success.
        action_noise: standard deviation of Gaussian noise added to the 5 arm entries of the
            executed `"delta"` action (in normalised units, so 0.1 is 0.005 rad of target
            offset), clipped to [-1, 1]; the gripper entry is never perturbed. DART-style
            demonstrations: the expert's closed loop corrects the perturbation, the clean action
            is what `last_clean_action` and the recorded label hold (`0` = off).
        noise_seed: seed of the noise generator.
        tolerate_ik_failures: when the IK finds no valid solution for an env (an unreachable
            cube placement), keep that env's arm where it is, mark it in `ik_failed` and carry on
            with the others (raise `ExpertFinished` once every env has failed) instead of
            raising `RuntimeError`. The demo collector uses this: such envs just never succeed.
    """

    env: object
    model: ArmModel
    control: ExpertControlConfig = field(default_factory=ExpertControlConfig)
    obs: torch.Tensor | None = None
    max_steps: int | None = None
    stop_on_success: bool = False
    action_noise: float = 0.0
    noise_seed: int = 0
    tolerate_ik_failures: bool = False

    def __post_init__(self) -> None:
        self.base = self.env.unwrapped
        self.n = self.base.num_envs
        self.agents = (self.base.agent_a, self.base.agent_b)
        self.uids = ("so101_pg-0", "so101_pg-1")
        # Commanded arm targets (n, 5) and gripper targets (n,) per arm.
        qpos = [self._qpos(a) for a in range(2)]
        self.q = [x[:, :NUM_ARM_JOINTS].copy() for x in qpos]
        self.grip = [np.full(self.n, GRIPPER_OPEN) for _ in self.agents]
        self.last_info: dict = {}
        self.callbacks: list[StepCallback] = []
        if self.action_noise < 0:
            raise ValueError(f"action_noise must be >= 0, got {self.action_noise}")
        if self.action_noise > 0 and self.control.control != "delta":
            raise ValueError('action_noise is only supported in "delta" control mode')
        self.ik_failed = np.zeros(self.n, dtype=bool)  # envs whose IK failed (tolerated)
        self._noise_rng = np.random.default_rng(self.noise_seed)
        # The clean (noise-free) action dict of the last step, `{uid: (n, 6) tensor}`.
        self.last_clean_action: dict = {}
        self.steps = 0  # env steps taken through this rig
        # Integral bias (rad) added to each arm's commanded offset in delta mode, and the log of
        # the closed-loop reaches (one `ReachRecord` each; the probe prints a summary per phase).
        self.bias = [np.zeros((self.n, NUM_ARM_JOINTS)) for _ in self.agents]
        self.reach_log: list[ReachRecord] = []
        # (env step, reward returned by env.step on that step, previous step's reward) of the
        # first step on which env 0 reports success; shows the success bonus as a one-step jump.
        self.first_success: tuple[int, float, float] | None = None
        self._last_reward = float("nan")

    def _qpos(self, arm: int) -> np.ndarray:
        """Joint positions of an arm, `(n, 7)`: 5 arm joints, then the two jaws."""
        return self.agents[arm].robot.get_qpos().cpu().numpy()

    def _qvel(self, arm: int) -> np.ndarray:
        return self.agents[arm].robot.get_qvel().cpu().numpy()

    def base_frame(self, arm: int) -> tuple[np.ndarray, np.ndarray]:
        """World poses of an arm's base: rotations (n, 3, 3) and positions (n, 3)."""
        mat = self.agents[arm].robot.pose.to_transformation_matrix().cpu().numpy()
        return mat[:, :3, :3], mat[:, :3, 3]

    def ik(self, arm: int, tcp_w, finger_dir_w, open_axis_w, q_init=None) -> np.ndarray:
        """Per-env IK for world-frame targets (each `(n, 3)`); raises if a target is unreachable.

        Starts from `q_init` (default: the current targets); only if that fails does it fall
        back to random restarts. Among valid solutions it prefers the one closest to `q_init`.
        In delta mode it first tries a solution that keeps `joint_limit_margin` away from the
        joint limits (from `q_init` only) and falls back to the unrestricted search for the
        poses that need a joint at its limit (the rotator's pre-grasp), so the margin never
        turns a solvable pose into an IK failure.
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
            best = None
            if self._ik_margin() > 0:
                best = self._solve_best(
                    tcp, finger_dir, axes, [q_init[i]], q_init[i], self._ik_margin()
                )
                # The tighter bounds can make the nearby solution invalid and leave only a
                # mirrored one (wrist roll turned by about pi): not worth 150 steps of extra wrist travel.
                if best is not None and np.abs(best - q_init[i]).max() > MARGIN_MAX_JUMP:
                    best = None
            if best is None:
                best = self._solve_best(tcp, finger_dir, axes, [q_init[i]], q_init[i])
            if best is None:
                rng = np.random.default_rng(0)
                seeds = [rng.uniform(self.model.lower, self.model.upper) for _ in range(10)]
                best = self._solve_best(tcp, finger_dir, axes, seeds, q_init[i])
            if best is None:
                if not self.tolerate_ik_failures:
                    raise RuntimeError(f"IK failed for arm {arm}, env {i}")
                best = q_init[i]  # hold the pose; the env is marked and will not succeed
                self.ik_failed[i] = True
                if self.ik_failed.all():
                    raise ExpertFinished("IK failed in every env")
            out[i] = best
        return out

    def _ik_margin(self) -> float:
        """Distance (rad) the IK keeps from the joint limits: delta mode only (pos mode: 0)."""
        return self.control.joint_limit_margin if self.control.control == "delta" else 0.0

    def _solve_best(
        self, tcp, finger_dir, axes, seeds, q_ref, margin: float = 0.0
    ) -> np.ndarray | None:
        """The valid IK solution closest to `q_ref` over all seeds and opening axes, if any."""
        best, best_dist = None, np.inf
        for axis in axes:
            for seed in seeds:
                q, pos_err, angle_err = self.model.solve(tcp, finger_dir, axis, seed, margin)
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

    def follow(self, arm: int, waypoints: list, settle: int = 0, fly_by: bool = False) -> None:
        """Drive one arm through joint waypoints (the other arm keeps its targets)."""
        self.follow_many({arm: waypoints}, settle, fly_by)

    def follow_many(self, paths: dict[int, list], settle: int = 0, fly_by: bool = False) -> None:
        """Drive several arms through their joint waypoints at once, in lockstep.

        Waypoint `k` of every arm is started together (paths of different length keep their
        last waypoint). Arms that are not in `paths` keep their targets. `fly_by` (delta mode
        only): the end of the path needs no precision (the next path starts from the commanded
        pose), so it is left at `approach_tol` and without settling.
        """
        cfg = self.control
        length = max(len(p) for p in paths.values())

        def goals(k: int) -> dict:
            return {("q_a", "q_b")[arm]: path[min(k, len(path) - 1)] for arm, path in paths.items()}

        if cfg.control == "pos":
            for k in range(length):
                self.move(**goals(k), settle=0)
            for _ in range(settle):
                self.step()
            return
        self._track_paths(paths, cfg.approach_tol if fly_by else cfg.final_tol)
        self._settle(0 if fly_by else min(settle, cfg.settle_cap))

    def _track_paths(self, paths: dict[int, list], final_tol: float) -> None:
        """Delta mode: track the waypoint paths closed loop, each env advancing on its own.

        Every (arm, env) has its own waypoint pointer: it moves on to the next waypoint as soon
        as that env's joints are within `waypoint_tol` of the current one (or are stalled), not
        when the slowest env gets there. With many envs the waypoint at which the joint travel is
        largest differs from env to env, and a lockstep advance would pay the sum of the
        per-waypoint maxima over envs, a lot more than the longest single path (about 1.4x for
        the holder's approach in 48 envs). The loop ends once every env has reached the last
        waypoint within `final_tol` (or stalled there), or after a budget derived from the path
        lengths.
        """
        cfg = self.control
        arms = tuple(paths)
        stacks = {arm: np.stack(path) for arm, path in paths.items()}  # (waypoints, n, 5)
        rows = np.arange(self.n)
        pointer = {arm: np.zeros(self.n, dtype=int) for arm in arms}
        monitors = {
            arm: ReachMonitor(final_tol, cfg.stall_steps, cfg.stall_progress) for arm in arms
        }
        # Budget: the longest per-env path at top speed with 50% margin, plus the slacks.
        longest = 0.0
        for arm in arms:
            points = np.concatenate([self._qpos(arm)[None, :, :NUM_ARM_JOINTS], stacks[arm]])
            longest = max(
                longest, float(np.abs(np.diff(points, axis=0)).max(axis=2).sum(axis=0).max())
            )
        budget = (
            math.ceil(1.5 * longest / (0.4 * ARM_DELTA_LIMIT))
            + cfg.waypoint_timeout * max(len(p) for p in paths.values())
            + cfg.final_timeout
        )
        self._reset_bias(arms)
        for arm in arms:
            self.q[arm] = stacks[arm][pointer[arm], rows]
        for k in range(1, budget + 1):
            self.step()
            finished = np.ones(self.n, dtype=bool)
            stalled_any = np.zeros(self.n, dtype=bool)
            for arm in arms:
                last = len(stacks[arm]) - 1
                on_last = pointer[arm] >= last
                monitor = monitors[arm]
                monitor.tol = np.where(on_last, final_tol, cfg.waypoint_tol)
                error = np.abs(self.q[arm] - self._qpos(arm)[:, :NUM_ARM_JOINTS]).max(axis=1)
                reached, stalled = monitor.update(error)
                advance = (reached | stalled) & ~on_last
                if advance.any():
                    pointer[arm] = pointer[arm] + advance
                    self.q[arm] = stacks[arm][pointer[arm], rows]
                    monitor.reset(advance)
                finished &= on_last & (reached | stalled)
                stalled_any |= on_last & stalled & ~reached
            if finished.all() or k == budget:
                reason = (
                    "timeout" if not finished.all() else "stall" if stalled_any.any() else "tol"
                )
                self._log_reach(arms, k, reason, final_tol, self._joint_errors(arms))
                return

    def step(self) -> None:
        """One env step with the current targets (see the module doc for the two modes)."""
        action, clean = {}, {}
        for arm, uid in enumerate(self.uids):
            joints = self.q[arm] if self.control.control == "pos" else self._delta_action(arm)
            act = np.concatenate([joints, self.grip[arm][:, None]], axis=1)
            clean[uid] = torch.as_tensor(act, dtype=torch.float32, device=self.base.device)
            if self.action_noise > 0:
                act = perturb_arm_action(act, self.action_noise, self._noise_rng, NUM_ARM_JOINTS)
            action[uid] = torch.as_tensor(act, dtype=torch.float32, device=self.base.device)
        self.last_clean_action = clean
        obs_before = self.obs
        self.obs, reward, terminated, truncated, self.last_info = self.env.step(action)
        self.steps += 1
        reward0 = float(reward.reshape(-1)[0])
        success = self.last_info["success"].reshape(-1)
        if self.first_success is None and bool(success[0]):
            self.first_success = (int(self.base.elapsed_steps[0]), reward0, self._last_reward)
        self._last_reward = reward0
        for callback in self.callbacks:
            callback(obs_before, action, reward, terminated, truncated, self.last_info)
        if self.stop_on_success and bool(success.all()):
            raise ExpertFinished("all envs succeeded")
        if self.max_steps is not None and self.steps >= self.max_steps:
            raise ExpertFinished(f"step budget of {self.max_steps} reached")

    def _delta_action(self, arm: int) -> np.ndarray:
        """Arm part of a `pd_joint_delta_pos` action `(n, 5)` towards the commanded target.

        The remaining joint error times `gain` plus the integral bias, in units of
        `ARM_DELTA_LIMIT`, divided by its largest entry when that exceeds 1: the direction is
        kept, so a multi-joint move stays a straight line in joint space (and the waypoints'
        straight tool path is followed). The bias integrates the error (anti-windup, see
        `callosum.experts._tracking`).
        """
        cfg = self.control
        error = self.q[arm] - self._qpos(arm)[:, :NUM_ARM_JOINTS]
        action, saturated = tracking_action(error, self.bias[arm], cfg.gain, ARM_DELTA_LIMIT)
        self.bias[arm] = update_bias(self.bias[arm], error, cfg.ki, cfg.bias_max, saturated)
        return action

    def _reset_bias(self, arms) -> None:
        """Forget the integral bias of the given arms (a new movement starts)."""
        for arm in arms:
            self.bias[arm] = np.zeros_like(self.bias[arm])

    def hold(self, arm: int) -> None:
        """Delta mode: command the arm's current joint positions, i.e. stop pushing (no-op in
        pos mode). Used after the turn: the wrist target overshoots the face's limit on purpose,
        and the open jaws must not keep rolling when the grip is released."""
        if self.control.control == "delta":
            self.q[arm] = self._qpos(arm)[:, :NUM_ARM_JOINTS].copy()
            self._reset_bias((arm,))

    def _joint_errors(self, arms) -> np.ndarray:
        """`(len(arms), n, 5)` absolute joint errors of the given arms."""
        return np.stack(
            [np.abs(self.q[a] - self._qpos(a)[:, :NUM_ARM_JOINTS]) for a in arms], axis=0
        )

    def _max_error(self, arms) -> float:
        """Largest joint error over the given arms and all envs."""
        return float(self._joint_errors(arms).max())

    def _timeout(self, arms, slack: int) -> int:
        """Step budget for reaching the current targets at the controller's top speed (about
        0.02 rad per step, with 50% margin) plus `slack` steps for the last millimetres."""
        remaining = self._max_error(arms)
        return slack + math.ceil(1.5 * remaining / (0.4 * ARM_DELTA_LIMIT))

    def _reach(
        self, arms, tol: float, timeout: int, also_done: Callable[[], np.ndarray] | None = None
    ) -> int:
        """Step until the given arms are within `tol` of their targets (at least one step).

        Ends once every env is within `tol`, stalled (see `ReachMonitor`) or `also_done()` (an
        optional per-env mask), or after `timeout` steps. Returns the number of steps taken and
        appends a `ReachRecord` to `reach_log`.
        """
        cfg = self.control
        monitor = ReachMonitor(tol, cfg.stall_steps, cfg.stall_progress)
        for k in range(1, timeout + 1):
            self.step()
            errors = self._joint_errors(arms)  # (arms, n, 5)
            reached, stalled = monitor.update(errors.max(axis=(0, 2)))
            extra = None if also_done is None else np.asarray(also_done(), dtype=bool)
            reason = reach_reason(reached, stalled, extra)
            if reason != "timeout" or k == timeout:
                self._log_reach(arms, k, reason, tol, errors)
                return k
        raise AssertionError("unreachable: timeout >= 1")

    def _log_reach(self, arms, steps: int, reason: str, tol: float, errors: np.ndarray) -> None:
        arm_i, env_i, joint_i = np.unravel_index(int(errors.argmax()), errors.shape)
        self.reach_log.append(
            ReachRecord(
                arms=tuple(arms),
                steps=steps,
                reason=reason,
                tol=tol,
                unreached_envs=int((errors.max(axis=(0, 2)) >= tol).sum()),
                worst_error=float(errors.max()),
                worst_arm=int(arms[arm_i]),
                worst_env=int(env_i),
                worst_joint=int(joint_i),
            )
        )

    def _is_still(self, jaw_arms: tuple[int, ...] | None = None) -> bool:
        """True when no joint moves faster than the tolerance: the arm joints and jaws of both
        arms, or (`jaw_arms`) only the jaws of those arms."""
        tol = self.control.settle_qvel_tol
        if jaw_arms is not None:
            return all(
                float(np.abs(self._qvel(a)[:, NUM_ARM_JOINTS:]).max()) < tol for a in jaw_arms
            )
        return all(float(np.abs(self._qvel(a)).max()) < tol for a in range(2))

    def _settle(
        self, steps: int, min_steps: int = 1, jaw_arms: tuple[int, ...] | None = None
    ) -> None:
        """Delta mode: wait up to `steps` steps, ending early once everything is still.

        `jaw_arms`: wait only for those arms' jaws (a gripper phase). With action noise the arm
        joints never come to rest (the jitter is the point), so a settle after an arm movement is
        cut to two steps.
        """
        if jaw_arms is None and self.action_noise > 0:
            steps = min(steps, 2)
        for k in range(1, steps + 1):
            self.step()
            if k >= min_steps and self._is_still(jaw_arms):
                return

    def move(
        self,
        q_a=None,
        q_b=None,
        grip_a=None,
        grip_b=None,
        settle=SETTLE_STEPS,
        until: Callable[[], np.ndarray] | None = None,
    ) -> None:
        """Move the arms to new joint targets (and set grippers), then settle.

        "pos" mode interpolates the commanded targets linearly at `max_joint_step` per step;
        "delta" mode tracks the goal closed loop (all given arms at once). `until` (delta mode
        only) is an optional per-env "this env is done" mask: the reach ends once every env is
        within tolerance, stalled or done.
        """
        goals = [q_a, q_b]
        for arm, g in enumerate((grip_a, grip_b)):
            if g is not None:
                self.grip[arm] = np.full(self.n, g)
        cfg = self.control
        if cfg.control == "delta":
            moving = tuple(arm for arm, goal in enumerate(goals) if goal is not None)
            for arm in moving:
                self.q[arm] = goals[arm]
            self._reset_bias(moving)
            if moving:
                timeout = self._timeout(moving, cfg.final_timeout)
                self._reach(moving, cfg.final_tol, timeout, until)
            if grip_a is not None or grip_b is not None:
                # A gripper phase waits for the jaws to close or open: its own length, at least 4.
                jaws = tuple(arm for arm, g in enumerate((grip_a, grip_b)) if g is not None)
                self._settle(settle, 4, jaws)
            else:
                self._settle(min(settle, cfg.settle_cap))
            return
        start = [q.copy() for q in self.q]
        delta = max(
            (np.abs(g - s).max() for g, s in zip(goals, start, strict=True) if g is not None),
            default=0.0,
        )
        steps = max(1, math.ceil(delta / cfg.max_joint_step))
        for k in range(1, steps + 1):
            for arm, goal in enumerate(goals):
                if goal is not None:
                    self.q[arm] = start[arm] + (goal - start[arm]) * k / steps
            self.step()
        for _ in range(settle):
            self.step()


def _horizontal_unit(rig: Rig, arm: int) -> np.ndarray:
    """World direction (n, 3) from an arm's base towards the cube, flattened to the table."""
    _, base_pos = rig.base_frame(arm)
    cube = rig.base.cube.pose.p.cpu().numpy()
    direction = cube[:, :2] - base_pos[:, :2]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return np.concatenate([direction, np.zeros((rig.n, 1))], axis=1)


def _face_turned(rig: Rig) -> np.ndarray:
    """`(n,)` bool: the face angle of each env is within `TURN_DONE_MARGIN` of the target."""
    angle = rig.base.face_link.joint.qpos.reshape(rig.n, -1)[:, 0].cpu().numpy()
    return angle >= TARGET_FACE_ANGLE - TURN_DONE_MARGIN


def run_expert(
    rig: Rig,
    *,
    no_holder: bool = False,
    release_holder: bool = False,
    on_phase: Callable[[str], None] | None = None,
) -> None:
    """Run the whole strategy on `rig`: holder grasp, rotator grasp, turn, release.

    Args:
        no_holder: skip the holder phases (the rotator turns alone; for the face-lock checks).
        release_holder: the holder opens its jaws half way through the turn.
        on_phase: called with the phase name after each phase (the probe prints its report).

    Stops silently when `Rig.step` raises `ExpertFinished` (`stop_on_success`, `max_steps`).
    """
    try:
        _run_phases(rig, no_holder, release_holder, on_phase or (lambda name: None))
    except ExpertFinished:
        pass


def _run_phases(rig: Rig, no_holder: bool, release_holder: bool, phase: Callable[[str], None]):
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
    # rotator's open jaws, so the rotator only comes down to the face layer once the holder is
    # at its grasp pose (and clamping the body).
    overlap = rig.control.overlap_approach and not no_holder
    if overlap:
        # The rotator's pre-grasp pose is 4 cm above the face layer: clear of the holder.
        rig.follow_many(
            {
                0: rig.path(0, pregrasp_a, dir_a, open_a),
                1: rig.path(1, pregrasp_b, dir_b, open_b),
            },
            settle=SETTLE_STEPS,
            fly_by=True,
        )
        phase("holder and rotator pregrasp")
        rig.follow(0, rig.path(0, grasp_a, dir_a, open_a), settle=0)
        rig.grip[0] = np.full(rig.n, GRIPPER_CLOSED)  # the holder's jaws close during the descent
        q_b_grasp = rig.path(1, grasp_b, dir_b, open_b)
        rig.follow(1, q_b_grasp, settle=SETTLE_STEPS)
        phase("holder grasp, rotator descent")
        rig.move(grip_b=GRIPPER_CLOSED, settle=20)
        phase("rotator grasp")
    else:
        if not no_holder:
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
    if release_holder:
        # Half the roll, then the holder lets go and the rotator keeps rolling: with the lock
        # the face must stop where it was when the holder released.
        q_b_half = q_b_grasp[-1].copy()
        q_b_half[:, -1] += ROLL_ANGLE / 2
        rig.move(q_b=q_b_half, settle=30)
        phase("half turn (holder still holds)")
        rig.move(grip_a=GRIPPER_OPEN, settle=10)
        phase("holder released")
    # Delta mode ends the turn as soon as every env's face is at its target: the wrist target
    # overshoots the face's limit, so the wrist's own error would never get within tolerance.
    delta = rig.control.control == "delta"
    rig.move(q_b=q_b_turned, settle=0 if delta else 30, until=lambda: _face_turned(rig))
    rig.hold(1)
    phase("turn")

    rig.move(grip_b=GRIPPER_OPEN, settle=20)
    phase("release")
