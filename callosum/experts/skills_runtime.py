"""Skill-level primitive runtime: turns discrete primitive choices into joint commands.

The adapter between the skill action space (`callosum.envs._skills` registries) and the
executable phase machinery of `callosum.experts._phases`: the ENV decodes each decision
into per-arm primitive ids, and this module keeps, per arm and per env, WHICH primitive
is currently selected, starts a fresh one when the choice changes (planning from the
CURRENT physical state -- never from a pose fixed at reset), interrupts the old one,
steps the running primitives after every simulation substep, and turns the commanded
targets into `pd_joint_delta_pos` actions with the episode's speed multipliers.

Importability follows the `_phases` rule (numpy + stdlib at module level, testable on
macOS): the scipy IK (`callosum.experts.face_turn_expert.ArmModel`) is imported lazily
inside the planners, which only run inside the simulator process. The pure pieces --
the decision/interruption machine, the useless-action classifier, the layer-target and
orientation math, the own-block layout helper -- are exercised by
`tests/test_face_turn_skills_logic.py` with mock primitives.

Failure injection (task 2.3) is delegated to an `InjectionSource` the env supplies (its
per-episode RNG decides): a grasp attempt may be doomed to close on nothing (the grip
command stays open, a re-selection retries), and a layer turn may be scheduled to slip
mid-motion (the grip opens and the primitive aborts to `wait`; recovery is a re-grasp).
"""

import math
from dataclasses import dataclass, field

import numpy as np

from callosum.configs.face_turn_expert import ExpertControlConfig
from callosum.envs._plans import _FACE_VECTORS, CubeOrientation
from callosum.envs._skills import (
    HOLDER_PRIMITIVES,
    ROTATOR_PRIMITIVES,
    Role,
    num_primitives,
)
from callosum.envs._skills_obs import own_block_fields
from callosum.experts._phases import (
    Commands,
    Context,
    Grip,
    Observation,
    Primitive,
    TrackPath,
)
from callosum.experts._tracking import tracking_action, update_bias

NUM_ARM_JOINTS = 5
GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, -1.0
TURN_OVERSHOOT = 0.1  # rad of wrist overshoot past the layer target (face_turn convention)
CLOSE_STEPS = 20  # a jaw close/open phase ends at the latest after this many substeps
SETTLE_STEPS = 10  # settle after a movement phase (capped by ExpertControlConfig)
PATH_STEPS = 8  # IK waypoints on each straight-line leg
STEP_SECONDS = 0.05
"""One control substep's duration (s): decision_period=20 steps ~ 1 s, sized on the
A40 probe so a 150-decision episode spans a multi-move plan (see
`callosum.configs.face_turn_skills`). Used only to convert the randomization's
start-delay seconds into substeps.
TODO(review): read the real control-step dt on the first cloud probe instead."""

# Strategy geometry, copied from `callosum.experts.face_turn_expert` (whose module pulls
# torch/mani_skill at import time and so cannot be imported here). Metres / radians.
HOLDER_GRASP_OFFSET = 0.02  # holder TCP this far from the cube centre, towards the holder
HOLDER_GRASP_Z = -0.005  # TCP height offset from the cube centre (the core's equator)
HOLDER_DEPRESSION = np.deg2rad(10)  # holder fingers point this far below horizontal
"""The depressed finger is what the arm can reach (horizontal fingers make the
grasp pose unreachable for this base layout), but it tilts the lower jaw like a
10 deg incline: during the flip's set-down ride the cube slides 2.5+ cm down it,
past the success position tolerance. The macro therefore LEVELS the tine (a
small pitch-up at constant TCP) right after the release, before the unroll."""
ROTATOR_TILT = np.deg2rad(6)  # rotator fingers this far off vertical, towards the cube
ROTATOR_GRASP_Z = 0.018  # TCP height above the cube centre: mid of the up layer (hs - t/2)
HOLDER_BACKOFF = 0.03  # pre-grasp: this far back along the approach direction ...
HOLDER_LIFT = 0.01  # ... and this far above the grasp pose (jaws open)
ROTATOR_LIFT = 0.04  # rotator pre-grasp: this far straight above the grasp pose
FLIP_LIFT = 0.06  # how high the holder lifts the cube before flipping it
LIFT_HEIGHT = 0.04  # the `lift` primitive's rise (a preparation hold, not a flip)
_UNROLL_SPEED = 0.5
"""Speed scale of the flip macro's unroll leg. The released cube rests on the
lower jaw like a fork tine (the roll turned the jaw plane vertical); unrolling
at the delta-limit speed flings it off with the tine's tangential velocity and
it lands edge-first, bounces and tumbles (observed: a 5 cm toss and a quarter
turn past the target). A slowed unroll lets the cube leave gently near the
friction angle and land flat; 0.5 balances that against the cube sliding down
the tine's 10 deg depression incline while it rides (the walk scales with the
ride time)."""
IK_POSITION_TOL = 3e-3  # m
IK_ANGLE_TOL = np.deg2rad(8)
_DELTA_LIMIT = 0.05
"""`ARM_DELTA_LIMIT` (rad per substep of the delta controller), duplicated from
`callosum.robots.so101_parallel_gripper` (not importable on macOS) only to size the
offline timeout/turn-duration estimates; at runtime `SkillsRuntime` receives the real
constant from the env (`delta_limit=`)."""

LAYER_DETENTS_RAD = (0.0, math.pi / 2, math.pi)
_UP = np.array([0.0, 0.0, 1.0])

# Registry lookups by action id (the id IS the registry index).
_HOLDER_NAME = {p.index: p.name for p in HOLDER_PRIMITIVES}
_ROTATOR_NAME = {p.index: p.name for p in ROTATOR_PRIMITIVES}
_HOLDER_ID = {name: idx for idx, name in _HOLDER_NAME.items()}
_ROTATOR_ID = {name: idx for idx, name in _ROTATOR_NAME.items()}


# --- pure decision logic -------------------------------------------------------------------


def apply_decision(current: np.ndarray, new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The interruption machine of one decision, per env (pure).

    Returns `(selection, changed)`: envs whose choice CHANGED interrupt their running
    primitive immediately and start the newly chosen one (planning from the current
    state); envs that re-select the running primitive continue it without a restart
    (skill-action-space spec: "Продолжение примитива между решениями"). The caller
    derives both the interrupted and the started set from `changed`.
    """
    current = np.asarray(current, dtype=int)
    new = np.asarray(new, dtype=int)
    changed = new != current
    return new.copy(), changed


def turn_target_rad(current_rad: float, quarter_turns: int) -> float:
    """The layer-angle target of a plan move executed from `current_rad` (pure).

    The current angle is snapped to its nearest detent (0/90/180 deg -- layer angles
    only ever rest there), advanced by the move's clockwise quarter turns, and mapped
    back into the joint's [0, 180] deg range by subtracting 180 deg when the sum
    overflows. The mapping is exact for the unmarked-square-slab abstraction the cube
    uses (a slab turned 270 deg is indistinguishable from one turned 90 deg the other
    way, and 90+180 deg lands on the same resting configuration as no turn at all);
    sticker state is deliberately not tracked anywhere in this project.
    TODO(review): if marked faces are ever rendered for the demo, revisit this.
    """
    if quarter_turns not in (1, 2):
        raise ValueError(f"quarter_turns must be 1 or 2, got {quarter_turns}")
    detent = min(LAYER_DETENTS_RAD, key=lambda d: abs(d - current_rad))
    target = detent + quarter_turns * (math.pi / 2)
    while target > math.pi + 1e-9:
        target -= math.pi
    return target


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix/matrices `(..., 3, 3)` from xyzw quaternion(s) (pure numpy)."""
    q = np.asarray(quat, dtype=float)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    matrix = np.empty(q.shape[:-1] + (3, 3))
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def up_face_from_quat(quat: np.ndarray) -> str:
    """Which labeled face currently points up (world +z), from the body quaternion."""
    rot = _quat_to_matrix(quat)
    body_dir = rot.T @ _UP
    return max(_FACE_VECTORS, key=lambda face: float(body_dir @ _FACE_VECTORS[face]))


def orientation_from_quat(quat: np.ndarray) -> CubeOrientation:
    """The full `CubeOrientation` (which face points along each world slot).

    Greedy one-to-one assignment (all (slot, face) dot products sorted descending,
    first come first served) instead of per-slot argmax: mid-flip quaternions (cube
    held at 45 degrees) make two world slots maximise the SAME cube face, and
    per-slot argmax then returns a non-permutation, which `CubeOrientation` rejects.
    For axis-aligned orientations this picks exactly the argmax permutation.
    """
    rot = _quat_to_matrix(quat)
    scores: list[tuple[float, str, str]] = []
    for slot in ("U", "D", "F", "B", "R", "L"):
        body_dir = rot.T @ np.asarray(_FACE_VECTORS[slot], dtype=float)
        for face in _FACE_VECTORS:
            scores.append((float(body_dir @ _FACE_VECTORS[face]), slot, face))
    scores.sort(reverse=True)
    by_slot: dict[str, str] = {}
    used: set[str] = set()
    for _score, slot, face in scores:
        if slot in by_slot or face in used:
            continue
        by_slot[slot] = face
        used.add(face)
    return CubeOrientation(tuple(by_slot[slot] for slot in ("U", "D", "F", "B", "R", "L")))


def own_block_key_layout(role: Role) -> tuple[tuple[str, int], ...]:
    """The env's obs keys of `role`'s own block: `own_block_fields` prefixed and ordered.

    `_get_obs_extra` emits exactly these keys in this order (holder block first, then
    the rotator's), so the flattened state observation carries the shared skills vector
    `[holder_own_block | rotator_own_block]` that `callosum.training._skills_agent_obs`
    expects (all non-agent fields contiguous, width `shared_vec_size()`).
    """
    prefix = "holder_" if role is Role.HOLDER else "rotator_"
    return tuple((prefix + name, size) for name, size in own_block_fields(num_primitives(role)))


@dataclass(frozen=True)
class DecisionFacts:
    """Per-env facts the useless-action classifier needs, at the moment of a decision.

    All arrays are `(n,)` bools the env computes from the live state (privileged, like
    the reward): both grasps (the same `is_grasping` checks FaceTurn uses), whether each
    TCP is near enough to its grasp target for a grasp attempt to be able to engage,
    and whether the plan's current move face is already on top.
    """

    holder_grasp: np.ndarray
    rotator_grasp: np.ndarray
    holder_near_core: np.ndarray
    rotator_near_layer: np.ndarray
    up_face_is_move_face: np.ndarray


def useless_actions(role: Role, ids: np.ndarray, facts: DecisionFacts) -> np.ndarray:
    """Which of this decision's choices were physically pointless, per env (pure).

    The rule set (cube-environment spec, "Ход без удержания не исполняется" and the
    `_skills_obs` key docs): a turn of a locked or un-held layer, releasing what is not
    held, grasping air far from the target (or re-grasping what is already held),
    lifting/lowering without holding the cube, and a flip while the rotator holds the
    layer (it would tear it out of the jaws). `wait`, `retreat` and the approaches are
    preparation and never classified -- the design's open question about counting
    lift/lower preparation explicitly is resolved as: only the not-holding case counts.
    """
    ids = np.asarray(ids, dtype=int)
    useless = np.zeros(len(ids), dtype=bool)
    if role is Role.HOLDER:
        names = _HOLDER_ID
        useless |= (ids == names["release"]) & ~facts.holder_grasp
        useless |= np.isin(ids, (names["lift"], names["lower"])) & ~facts.holder_grasp
        useless |= np.isin(ids, (names["flip90"], names["flip180"])) & facts.rotator_grasp
        useless |= (ids == names["grasp"]) & (~facts.holder_near_core | facts.holder_grasp)
        return useless
    names = _ROTATOR_ID
    useless |= np.isin(ids, (names["turn90"], names["turn180"])) & (
        ~facts.rotator_grasp | ~facts.holder_grasp
    )
    useless |= (ids == names["release"]) & ~facts.rotator_grasp
    useless |= (ids == names["grasp"]) & (~facts.rotator_near_layer | facts.rotator_grasp)
    return useless


class InjectionSource:
    """The env-side failure oracle (per-episode RNG); the runtime calls these when a
    grasp or a turn starts. The default never injects (tests, clean runs)."""

    def fail_grasp(self, arm: int, env: int) -> bool:
        return False

    def slip_at(self, arm: int, env: int, est_steps: int, now: int) -> int | None:
        return None


# --- runtime ------------------------------------------------------------------------------


@dataclass
class RuntimeView:
    """Numpy snapshot of what the primitives need after a substep (per-arm entries)."""

    qpos: list[np.ndarray]  # (n, 5) arm joint positions per arm
    arm_speed: list[np.ndarray]  # (n,) largest arm joint speed per arm
    jaw_speed: list[np.ndarray]  # (n,) largest jaw speed per arm
    base_pos: list[np.ndarray]  # (n, 3) arm base positions (world)
    base_rot: list[np.ndarray]  # (n, 3, 3) arm base rotations (world)
    cube_pos: np.ndarray  # (n, 3)
    cube_quat: np.ndarray  # (n, 4) xyzw
    layer_angle: np.ndarray  # (n,) rad, angle of the current plan move's layer
    turn_target: np.ndarray  # (n,) rad, the env's authoritative target of the current move
    layer_at_target: np.ndarray  # (n,) bool, that layer within tolerance of its target
    needed_flip: list = field(default_factory=list)  # per env: Flip | None
    t: int = 0  # simulation substeps since the env was created


class Hold(Primitive):
    """`wait`: command the arm's current joint positions forever, keep the last grip.

    The grip command deliberately passes through unchanged: waiting with closed jaws
    keeps holding, waiting with open jaws stays open -- `wait` never opens a grasp."""

    def start(self, ctx: Context, arm: int, envs: np.ndarray) -> np.ndarray:
        self.steps[envs] = 0
        self.reason[envs] = -1
        return np.ones(len(envs), dtype=bool)

    def step(self, ctx: Context, arm: int, envs: np.ndarray) -> np.ndarray:
        self.steps[envs] += 1
        ctx.cmd.hold(arm, envs, ctx.obs.qpos[arm])
        return np.zeros(len(envs), dtype=bool)


class Sequence(Primitive):
    """A macro primitive: child primitives one after another, per env (e.g. the flip).

    The children are FACTORIES: fresh instances are built (and planned from the live
    state) when this Sequence starts, and each child starts per env only when that env
    finished the previous one -- so an interrupted and re-selected Sequence replans
    from wherever the arm actually is. A child whose start fails parks that env past
    the last stage (done, holding)."""

    def __init__(self, n: int, name: str, child_factories: list) -> None:
        super().__init__(n, name)
        # Children are built ONCE: rebuilding them on a later start() would wipe the
        # per-env state of envs that are mid-flight in a later stage (their paths
        # would become None and the next step would crash). Planning from the live
        # state happens inside each child's own `start(envs)` for the entering envs.
        self.children: list[Primitive] = [factory() for factory in child_factories]
        self.stage = np.zeros(n, dtype=int)

    def start(self, ctx: Context, arm: int, envs: np.ndarray) -> np.ndarray:
        self.stage[envs] = 0
        ok = np.asarray(self.children[0].start(ctx, arm, envs), dtype=bool)
        # Park envs whose first leg could not plan past the last stage (done,
        # holding) -- the runtime falls them back to `wait`.
        self.stage[np.asarray(envs)[~ok]] = len(self.children)
        return ok

    def step(self, ctx: Context, arm: int, envs: np.ndarray) -> np.ndarray:
        active = np.zeros(self.n, dtype=bool)
        active[envs] = True
        # One ascending pass: an env that finishes stage k enters stage k+1 within the
        # same pass (children need >= 1 step each, so this cannot cascade infinitely).
        for stage, child in enumerate(self.children):
            running = np.flatnonzero(active & (self.stage == stage))
            if len(running) == 0:
                continue
            finished = np.asarray(child.step(ctx, arm, running), dtype=bool)
            done_envs = running[finished]
            if stage + 1 < len(self.children):
                self.stage[done_envs] = stage + 1
                if len(done_envs):
                    ok = np.asarray(
                        self.children[stage + 1].start(ctx, arm, done_envs), dtype=bool
                    )
                    self.stage[done_envs[~ok]] = len(self.children)
            else:
                self.stage[done_envs] = len(self.children)
        return self.stage[envs] >= len(self.children)


class SkillsRuntime:
    """Drives both arms' selected primitives; see the module doc.

    The env calls `reset` per episode, `set_decision` per decision (both arms), then
    alternates `actions(view)` (before the substep) and `substep(view)` (after it).
    Selection state is per env, so a batch of envs runs independent choices.
    """

    def __init__(
        self,
        n: int,
        control: ExpertControlConfig | None = None,
        inject: InjectionSource | None = None,
        delta_limit: float = _DELTA_LIMIT,
    ) -> None:
        self.n = n
        self.control = control if control is not None else ExpertControlConfig(control="delta")
        self.inject = inject if inject is not None else InjectionSource()
        self._delta_limit = float(delta_limit)
        self.cmd = Commands.from_qpos([np.zeros((n, NUM_ARM_JOINTS)) for _ in range(2)])
        self.ctx = Context(self.control, self.cmd, self._empty_observation())
        self.selection = np.zeros((2, n), dtype=int)
        self.speed = np.ones((2, n))
        self.delay_until = np.zeros((2, n), dtype=int)
        self._primitives: dict[tuple[int, int], Primitive] = {}
        self._slip_at = np.full((2, n), -1, dtype=int)
        self._model = None
        self._view: RuntimeView | None = None

    # --- lifecycle -----------------------------------------------------------------------

    def _empty_observation(self) -> Observation:
        zeros = np.zeros(self.n)
        return Observation(
            qpos=[np.zeros((self.n, NUM_ARM_JOINTS)) for _ in range(2)],
            arm_speed=[zeros.copy() for _ in range(2)],
            jaw_speed=[zeros.copy() for _ in range(2)],
            face_turned=zeros.astype(bool),
        )

    def reset(
        self,
        view: RuntimeView,
        speed: np.ndarray,
        delay_steps: np.ndarray,
        envs: np.ndarray | None = None,
    ) -> None:
        """Per-episode reset: everything holds, selections fall back to `wait`.

        `speed`/`delay_steps` are `(2, b)` for the `b` envs in `envs` (None = all envs,
        the full-batch first reset); a partial reset leaves the other envs' episode
        randomization and selections untouched.
        """
        self._view = view
        if envs is None:
            envs = np.arange(self.n)
            self.cmd = Commands.from_qpos([q.copy() for q in view.qpos], GRIPPER_OPEN)
            self.ctx = Context(self.control, self.cmd, self._empty_observation())
            self.selection[:] = 0
            self.speed[:] = 1.0
            self.delay_until[:] = 0
            self._primitives = {(arm, 0): Hold(self.n, "wait") for arm in range(2)}
            self._slip_at[:] = -1
        self.speed[:, envs] = np.asarray(speed, dtype=float)
        self.delay_until[:, envs] = np.asarray(delay_steps, dtype=int) + view.t
        self.selection[:, envs] = 0
        self.cmd.q[0][envs] = view.qpos[0][envs]
        self.cmd.q[1][envs] = view.qpos[1][envs]
        self.cmd.bias[0][envs] = 0.0
        self.cmd.bias[1][envs] = 0.0
        self.cmd.grip[0][envs] = GRIPPER_OPEN
        self.cmd.grip[1][envs] = GRIPPER_OPEN
        self._slip_at[:, envs] = -1
        for primitive in self._primitives.values():
            primitive.steps[envs] = 0
            primitive.reason[envs] = -1

    # --- decisions -----------------------------------------------------------------------

    def set_decision(self, arm: int, ids: np.ndarray, view: RuntimeView | None = None) -> None:
        """Apply one arm's decision: interrupt changed envs, start their new primitives."""
        if view is not None:
            self._view = view
        view = self._require_view()
        new, started = apply_decision(self.selection[arm], ids)
        self.selection[arm] = new
        self.ctx.obs = self._observation(view)
        for primitive_id in np.unique(new[started]):
            envs = np.flatnonzero(started & (new == primitive_id))
            key = (arm, int(primitive_id))
            # Reuse the running instance when other envs already execute this
            # primitive (its per-env state keeps their progress; `start` initializes
            # only the entering envs); otherwise build a fresh one.
            already = np.flatnonzero(new == primitive_id)
            if key in self._primitives and not np.isin(already, envs).all():
                primitive = self._primitives[key]
            else:
                primitive = self._build_primitive(arm, int(primitive_id))
                self._primitives[key] = primitive
            ok = np.asarray(primitive.start(self.ctx, arm, envs), dtype=bool)
            if not ok.all():
                # The primitive could not start (e.g. IK failure): those envs hold and
                # fall back to `wait`; the policy sees no progress and may re-choose.
                bad = envs[~ok]
                self.selection[arm][bad] = 0
                self.cmd.hold(arm, bad, view.qpos[arm])
            self._inject_at_start(arm, int(primitive_id), envs, ok)

    def _inject_at_start(
        self, arm: int, primitive_id: int, envs: np.ndarray, ok: np.ndarray
    ) -> None:
        """Grasp-failure and slip scheduling for freshly started primitives."""
        view = self._require_view()
        names = _HOLDER_ID if arm == 0 else _ROTATOR_ID
        if primitive_id == names["grasp"]:
            for env in envs[ok]:
                if self.inject.fail_grasp(arm, int(env)):
                    # Doomed attempt: the jaws never close (the grip command stays
                    # open), so the grasp cannot engage; a re-selected grasp retries.
                    self.cmd.grip[arm][env] = GRIPPER_OPEN
        if primitive_id in (names.get("turn90", -1), names.get("turn180", -1)):
            for env in envs[ok]:
                wrist = abs(self.cmd.q[arm][env][-1] - view.qpos[arm][env][-1])
                est = max(int(1.5 * wrist / (0.4 * self._delta_limit)) + 10, 10)
                at = self.inject.slip_at(arm, int(env), est, view.t)
                self._slip_at[arm][env] = -1 if at is None else int(at)

    # --- stepping ------------------------------------------------------------------------

    def _observation(self, view: RuntimeView) -> Observation:
        return Observation(
            qpos=[q.copy() for q in view.qpos],
            arm_speed=[s.copy() for s in view.arm_speed],
            jaw_speed=[s.copy() for s in view.jaw_speed],
            face_turned=view.layer_at_target.copy(),
        )

    def substep(self, view: RuntimeView) -> None:
        """Advance every running primitive by one substep (call after the env stepped)."""
        self._view = view
        self.ctx.obs = self._observation(view)
        self.ctx.t = view.t
        for arm in range(2):
            due = view.t >= self.delay_until[arm]
            for primitive_id in np.unique(self.selection[arm]):
                if primitive_id == 0:
                    continue  # Hold steps in its own branch below
                envs = np.flatnonzero((self.selection[arm] == primitive_id) & due)
                if len(envs) == 0:
                    continue
                primitive = self._primitives.get((arm, int(primitive_id)))
                if primitive is None:
                    continue
                envs = envs[~self._check_slip(arm, int(primitive_id), envs, view)]
                if len(envs) == 0:
                    continue
                done = np.asarray(primitive.step(self.ctx, arm, envs), dtype=bool)
                if done.any():
                    # Completed but still selected: hold the end pose (the choice
                    # persists until the policy changes it).
                    self.cmd.hold(arm, envs[done], view.qpos[arm])
            waiting = np.flatnonzero((self.selection[arm] == 0) & due)
            if len(waiting):
                hold = self._primitives.get((arm, 0))
                if hold is not None:
                    hold.step(self.ctx, arm, waiting)
            if not due.all():
                # The randomization's start delay: the hand has not woken up yet.
                self.cmd.hold(arm, np.flatnonzero(~due), view.qpos[arm])

    def _check_slip(
        self, arm: int, primitive_id: int, envs: np.ndarray, view: RuntimeView
    ) -> np.ndarray:
        """Scheduled slip events of a running turn: drop the grip and abort to `wait`."""
        slipped = (self.selection[arm] == primitive_id) & (self._slip_at[arm] == view.t)
        slipped &= np.isin(np.arange(self.n), envs)
        for env in np.flatnonzero(slipped):
            # Kinematics of the failure (chosen deliberately simple, task 2.3): the
            # held layer escapes -- open the jaws mid-motion and abort the primitive.
            # The layer stops wherever it is; recovery is a fresh approach/grasp/turn.
            self.cmd.grip[arm][env] = GRIPPER_OPEN
            self.cmd.hold(arm, np.array([env], dtype=int), view.qpos[arm])
            self.selection[arm][env] = 0
            self._slip_at[arm][env] = -1
        return slipped[envs]

    def actions(self, view: RuntimeView) -> list[np.ndarray]:
        """`(n, 6)` `pd_joint_delta_pos` actions per arm from the commanded targets.

        The same closed-loop tracking as the scripted expert (`Rig._delta_action`):
        normalised error towards the commanded joints plus the integral bias, here
        additionally scaled by the episode's per-hand speed multiplier (randomization)
        and the running primitive's own `speed_scale` (a macro leg may move
        quasi-statically), then clipped -- a slower hand converges over more
        substeps, never a jumpier one. The gripper entry is absolute and never
        scaled. The controller's per-substep bound (`ARM_DELTA_LIMIT`) is injected
        by the env (the robots module pulls sapien and is not importable on macOS).
        """
        limit = self._delta_limit
        outs = []
        for arm in range(2):
            error = self.cmd.q[arm] - view.qpos[arm]
            action, saturated = tracking_action(
                error, self.cmd.bias[arm], self.control.gain, limit
            )
            self.cmd.bias[arm] = update_bias(
                self.cmd.bias[arm], error, self.control.ki, self.control.bias_max, saturated
            )
            speed = self.speed[arm] * self._running_speed_scale(arm)
            action = np.clip(action * speed[:, None], -1.0, 1.0)
            outs.append(np.concatenate([action, self.cmd.grip[arm][:, None]], axis=1))
        return outs

    def _running_speed_scale(self, arm: int) -> np.ndarray:
        """`(n,)` speed scale of each env's running primitive (Sequence: its
        current child; `wait` and unknown primitives: 1.0)."""
        scale = np.ones(self.n)
        for primitive_id in np.unique(self.selection[arm]):
            if primitive_id == 0:
                continue
            primitive = self._primitives.get((arm, int(primitive_id)))
            if primitive is None:
                continue
            for env in np.flatnonzero(self.selection[arm] == primitive_id):
                running = primitive
                if isinstance(primitive, Sequence):
                    stage = min(int(primitive.stage[env]), len(primitive.children) - 1)
                    running = primitive.children[stage]
                scale[env] = getattr(running, "speed_scale", 1.0)
        return scale

    def selections(self) -> np.ndarray:
        """`(2, n)` currently-selected primitive ids (a slip abort reads as `wait`)."""
        return self.selection.copy()

    def progresses(self) -> np.ndarray:
        """`(2, n)` [0, 1] progress of the running primitives (1 while `wait`)."""
        out = np.ones((2, self.n))
        for arm in range(2):
            for primitive_id in np.unique(self.selection[arm]):
                if primitive_id == 0:
                    continue
                primitive = self._primitives.get((arm, int(primitive_id)))
                if primitive is None:
                    continue
                envs = np.flatnonzero(self.selection[arm] == primitive_id)
                out[arm][envs] = _progress_of(primitive)[envs]
        return out

    def _require_view(self) -> RuntimeView:
        if self._view is None:
            raise RuntimeError("call SkillsRuntime.reset(view, ...) before using the runtime")
        return self._view

    # --- primitive construction ------------------------------------------------------------

    def _build_primitive(self, arm: int, primitive_id: int) -> Primitive:
        name = (_HOLDER_NAME if arm == 0 else _ROTATOR_NAME).get(int(primitive_id))
        if name is None:
            raise ValueError(f"no primitive with id {primitive_id} for arm {arm}")
        return self._holder_primitive(name) if arm == 0 else self._rotator_primitive(name)

    def _holder_primitive(self, name: str) -> Primitive:
        n = self.n
        if name == "wait":
            return Hold(n, name)
        if name == "approach_body":
            return TrackPath(n, name, lambda envs: self._plan_approach(0, envs), settle=SETTLE_STEPS)
        if name == "grasp":
            return Grip(n, name, GRIPPER_CLOSED, CLOSE_STEPS)
        if name == "release":
            return Grip(n, name, GRIPPER_OPEN, CLOSE_STEPS)
        if name == "lift":
            return TrackPath(
                n, name, lambda envs: self._plan_vertical(0, envs, +LIFT_HEIGHT), settle=SETTLE_STEPS
            )
        if name == "lower":
            return TrackPath(
                n, name, lambda envs: self._plan_vertical(0, envs, -LIFT_HEIGHT), settle=SETTLE_STEPS
            )
        if name == "retreat":
            return TrackPath(n, name, lambda envs: self._plan_retreat(0, envs), settle=SETTLE_STEPS)
        if name in ("flip90", "flip180"):
            quarter = 1 if name == "flip90" else 2
            # The macro ends by SETTING THE CUBE DOWN: the loaded wrist-pitch cannot
            # press a gripped cube the last centimetres onto the table (the joint
            # sags into its limit under the hanging weight), so the lower leg brings
            # the cube as far down as the arm manages and the release drops it the
            # short remaining distance onto the lower jaw (the roll turned the jaw
            # plane vertical); the unroll tips the cube off onto the table (the
            # ride down the 10 deg incline first is admitted by body_pos_tol) and
            # the retreat clears the gripper. The conductor re-grasps afterwards
            # (a table-supported grasp has no sag). NOTE: re-orientations beyond
            # the unroll are IK-infeasible from the stalled posture -- the level
            # leg experiment (pitching the tine flat) parked the whole macro.
            return Sequence(
                n,
                name,
                [
                    lambda: TrackPath(
                        n,
                        "flip lift",
                        lambda envs: self._plan_vertical(0, envs, +FLIP_LIFT),
                        settle=SETTLE_STEPS,
                    ),
                    lambda: TrackPath(
                        n,
                        "flip roll",
                        lambda envs: self._plan_wrist_roll(0, envs, quarter),
                        settle=SETTLE_STEPS,
                    ),
                    lambda: TrackPath(
                        n,
                        "flip lower",
                        lambda envs: self._plan_vertical(0, envs, -FLIP_LIFT),
                        settle=SETTLE_STEPS,
                    ),
                    lambda: Grip(n, "flip release", GRIPPER_OPEN, CLOSE_STEPS),
                    lambda: TrackPath(
                        n,
                        "flip unroll",
                        lambda envs: self._plan_reorient(0, envs),
                        settle=SETTLE_STEPS,
                        speed_scale=_UNROLL_SPEED,
                    ),
                    lambda: TrackPath(
                        n, "flip retreat", lambda envs: self._plan_retreat(0, envs), settle=SETTLE_STEPS
                    ),
                ],
            )
        raise ValueError(f"unknown holder primitive {name!r}")

    def _rotator_primitive(self, name: str) -> Primitive:
        n = self.n
        if name == "wait":
            return Hold(n, name)
        if name == "approach_face":
            return TrackPath(n, name, lambda envs: self._plan_approach(1, envs), settle=SETTLE_STEPS)
        if name == "grasp":
            return Grip(n, name, GRIPPER_CLOSED, CLOSE_STEPS)
        if name in ("turn90", "turn180"):
            quarter = 1 if name == "turn90" else 2
            return TrackPath(
                n,
                name,
                lambda envs: self._plan_turn(envs, quarter),
                until=lambda ctx: ctx.obs.face_turned,
                hold_on_done=True,
            )
        if name == "release":
            return Grip(n, name, GRIPPER_OPEN, CLOSE_STEPS)
        if name == "retreat":
            return TrackPath(n, name, lambda envs: self._plan_retreat(1, envs), settle=SETTLE_STEPS)
        raise ValueError(f"unknown rotator primitive {name!r}")

    # --- planning (IK; runs inside the simulator process only) ------------------------------

    def _arm_model(self):
        """The numpy FK/IK model of the SO-ARM101 (lazy: scipy + URDF, sim machines only)."""
        if self._model is None:
            from callosum.experts.face_turn_expert import ArmModel

            self._model = ArmModel()
        return self._model

    def _tool_world(self, arm: int, q: np.ndarray, envs: np.ndarray):
        """World TCP / finger direction / jaw axis (each `(len(envs), 3)`) at `q`."""
        if len(envs) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
        view = self._require_view()
        model = self._arm_model()
        rot, pos = view.base_rot[arm][envs], view.base_pos[arm][envs]
        tools = [model.tool(qq) for qq in q]
        tcp, finger, axis = (np.stack(x) for x in zip(*tools, strict=True))
        return (
            np.einsum("nij,nj->ni", rot, tcp) + pos,
            np.einsum("nij,nj->ni", rot, finger),
            np.einsum("nij,nj->ni", rot, axis),
        )

    def _solve_ik(self, arm: int, envs: np.ndarray, tcp, finger, open_axis, seeds):
        """Batched IK to world-frame goals; returns `(qs, ok)` for the entering envs.

        Tries both jaw-opening signs (the jaws are symmetric) from the single seed and
        keeps the joint-limit margin of the control config, like `Rig.ik` but per
        entering env only (planning happens at decision boundaries, not per substep).
        TODO(review): unify with `Rig._solve_best` on the cloud probe (task 2.4 measures
        this path's cost; a batched/GPU IK replaces it if reset+decision planning is
        too slow for 256 envs).
        """
        view = self._require_view()
        model = self._arm_model()
        rot, pos = view.base_rot[arm][envs], view.base_pos[arm][envs]
        out = np.zeros((len(envs), NUM_ARM_JOINTS))
        ok = np.zeros(len(envs), dtype=bool)
        for i in range(len(envs)):
            to_base = rot[i].T
            goal_tcp = to_base @ (tcp[i] - pos[i])
            goal_finger = to_base @ finger[i]
            goal_open = to_base @ open_axis[i]
            found = False
            for sign in (1.0, -1.0):
                q, pos_err, angle_err = model.solve(
                    goal_tcp,
                    goal_finger,
                    sign * goal_open,
                    seeds[i],
                    self.control.joint_limit_margin,
                )
                if pos_err < IK_POSITION_TOL and angle_err < IK_ANGLE_TOL:
                    out[i], found = q, True
                    break
            if not found:
                out[i] = seeds[i]
            ok[i] = found
        return out, ok

    def _plan_path_world(
        self, arm: int, envs: np.ndarray, tcp, finger, open_axis, steps: int = PATH_STEPS
    ):
        """Joint waypoints moving the tool along a straight line to the goal pose."""
        q_start = self.cmd.q[arm][envs]
        start_tcp, start_f, start_o = self._tool_world(arm, q_start, envs)
        paths = np.zeros((len(envs), steps, NUM_ARM_JOINTS))
        ok = np.ones(len(envs), dtype=bool)
        q_prev = q_start.copy()
        for k in range(1, steps + 1):
            t = k / steps
            tcp_k = start_tcp + (np.asarray(tcp) - start_tcp) * t
            f_k = start_f + (np.asarray(finger) - start_f) * t
            f_k /= np.maximum(np.linalg.norm(f_k, axis=1, keepdims=True), 1e-9)
            o_k = start_o + (np.asarray(open_axis) - start_o) * t
            o_k /= np.maximum(np.linalg.norm(o_k, axis=1, keepdims=True), 1e-9)
            q_new, ok_k = self._solve_ik(arm, envs, tcp_k, f_k, o_k, q_prev)
            paths[:, k - 1] = q_new
            ok &= ok_k
            q_prev = np.where(ok_k[:, None], q_new, q_prev)
        return paths, ok

    # Geometry of the two roles' key poses, from the live cube pose (face_turn_expert's
    # strategy generalized to the cube's CURRENT pose and orientation).

    def _horizontal_unit(self, arm: int, envs: np.ndarray) -> np.ndarray:
        """World direction `(len(envs), 3)` from an arm's base towards the cube."""
        view = self._require_view()
        fwd = view.cube_pos[envs][:, :2] - view.base_pos[arm][envs][:, :2]
        fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
        return np.concatenate([fwd, np.zeros((len(envs), 1))], axis=1)

    def _flip_axis_of(self, env: int) -> np.ndarray | None:
        """The needed flip's world axis of `env`, or None (default axis handled by callers)."""
        view = self._require_view()
        flip = view.needed_flip[env] if env < len(view.needed_flip) else None
        if flip is None:
            return None
        return np.asarray(_FACE_VECTORS[flip.axis], dtype=float)

    def _holder_directions(self, envs: np.ndarray):
        """Per-env `(fwd, grasp_finger, finger_dir, open_axis)` of the holder's approach.

        `fwd` is the natural horizontal base->cube direction; `grasp_finger` is tilted
        down by the depression angle. When the current plan move needs a flip,
        `finger_dir` rotates onto the flip's axis instead of `fwd` -- the wrist roll
        then tips the cube about exactly that axis (an approach planned for the
        upcoming flip; if the axis is unreachable the IK fails loudly and the policy's
        choice simply holds).
        """
        fwd = self._horizontal_unit(0, envs)
        finger_dir = fwd.copy()
        for i, env in enumerate(envs):
            axis = self._flip_axis_of(int(env))
            if axis is not None:
                finger_dir[i] = axis if float(axis @ fwd[i]) >= 0 else -axis
        grasp_finger = np.cos(HOLDER_DEPRESSION) * finger_dir - np.sin(HOLDER_DEPRESSION) * _UP
        open_axis = np.cross(_UP, finger_dir)
        return fwd, grasp_finger, finger_dir, open_axis

    def _holder_grasp_pose(self, envs: np.ndarray):
        """`(tcp, finger_dir, open_axis)` of the holder clamping the core's equator."""
        view = self._require_view()
        _, grasp_finger, finger_dir, open_axis = self._holder_directions(envs)
        tcp = view.cube_pos[envs] + HOLDER_GRASP_Z * _UP - HOLDER_GRASP_OFFSET * finger_dir
        return tcp, grasp_finger, open_axis

    def _rotator_grasp_pose(self, envs: np.ndarray):
        """`(tcp, finger_dir, open_axis)` of the rotator pinching the up layer top-down."""
        view = self._require_view()
        fwd = self._horizontal_unit(1, envs)
        finger = np.sin(ROTATOR_TILT) * fwd - np.cos(ROTATOR_TILT) * _UP
        open_axis = np.cross(_UP, fwd)
        tcp = view.cube_pos[envs] + ROTATOR_GRASP_Z * _UP
        return tcp, finger, open_axis

    def _plan_approach(self, arm: int, envs: np.ndarray):
        """`approach_body` / `approach_face`: backoff leg then the descent to the grasp."""
        if arm == 0:
            tcp, finger, open_axis = self._holder_grasp_pose(envs)
            backoff = tcp - HOLDER_BACKOFF * finger + HOLDER_LIFT * _UP
        else:
            tcp, finger, open_axis = self._rotator_grasp_pose(envs)
            backoff = tcp + ROTATOR_LIFT * _UP
        pre, ok_pre = self._plan_path_world(arm, envs, backoff, finger, open_axis)
        # The descent starts where the backoff leg ends (chained, like the expert's paths).
        saved = self.cmd.q[arm][envs].copy()
        self.cmd.q[arm][envs] = pre[:, -1]
        grasp, ok_grasp = self._plan_path_world(arm, envs, tcp, finger, open_axis)
        self.cmd.q[arm][envs] = saved
        return np.concatenate([pre, grasp], axis=1), ok_pre & ok_grasp

    def _plan_vertical(self, arm: int, envs: np.ndarray, height: float):
        """`lift` / `lower` (and the flip macro's lift/lower legs): straight vertical."""
        q = self.cmd.q[arm][envs]
        tcp, finger, open_axis = self._tool_world(arm, q, envs)
        return self._plan_path_world(arm, envs, tcp + height * _UP, finger, open_axis)

    def _plan_reorient(self, arm: int, envs: np.ndarray):
        """The flip macro's `flip unroll` leg: wrist back to the natural jaw
        orientation at the SAME tool pose.

        The flip roll turns the jaw plane vertical, so the released cube balances
        on the lower jaw like a fork tine; rotating the jaw plane back horizontal
        tips the cube off deterministically (it slides within the first degrees)
        and leaves the arm unrolled for the retreat."""
        q = self.cmd.q[arm][envs]
        tcp, finger, _ = self._tool_world(arm, q, envs)
        if arm == 0:
            _, _, _, open_axis = self._holder_directions(envs)
        else:
            fwd = self._horizontal_unit(arm, envs)
            open_axis = np.cross(_UP, fwd)
        return self._plan_path_world(arm, envs, tcp, finger, open_axis)

    def _plan_retreat(self, arm: int, envs: np.ndarray):
        """`retreat`: back along the approach and up (holder), or straight up (rotator)."""
        q = self.cmd.q[arm][envs]
        tcp, finger, open_axis = self._tool_world(arm, q, envs)
        if arm == 0:
            _, grasp_finger, finger_dir, open_axis = self._holder_directions(envs)
            goal = tcp - HOLDER_BACKOFF * finger_dir + HOLDER_LIFT * _UP
            return self._plan_path_world(arm, envs, goal, grasp_finger, open_axis)
        goal = tcp + ROTATOR_LIFT * _UP
        return self._plan_path_world(arm, envs, goal, finger, open_axis)

    def _plan_wrist_roll(self, arm: int, envs: np.ndarray, quarter_turns: int):
        """The flip macro's roll leg: wrist rotation about the CURRENT finger axis.

        The cube is rigidly held, so rotating the wrist by `delta` about the finger
        axis `f` rotates the cube by `delta * sign(w . f)` about the flip axis `w` --
        the sign below makes the cube turn the way `Flip`'s right-hand convention
        demands. No IK: the target is the current command with the wrist offset.
        """
        q = self.cmd.q[arm][envs].copy()
        _, finger, _ = self._tool_world(arm, q, envs)
        theta = quarter_turns * (math.pi / 2)
        for i, env in enumerate(envs):
            axis = self._flip_axis_of(int(env))
            if axis is None:
                # No flip needed for the current move: roll about the natural approach
                # axis (F) anyway -- the choice is the policy's (the conductor never
                # flips then); the rotation is real and changes the orientation.
                axis = np.asarray(_FACE_VECTORS["F"], dtype=float)
            sign = 1.0 if float(axis @ finger[i]) >= 0 else -1.0
            q[i, -1] += sign * theta
        return q[:, None, :], np.ones(len(envs), dtype=bool)

    def _plan_turn(self, envs: np.ndarray, quarter_turns: int):
        """`turn90` / `turn180`: wrist roll dragging the up layer toward the move target.

        The direction comes from the env's authoritative move target minus the current
        angle (signed, wrapped); the TRAVEL is the primitive's own quarter turns in that
        direction -- choosing `turn90` for a 180 deg move makes partial progress and the
        move completes only after another decision, choosing `turn180` for a 90 deg move
        overshoots through the target window (the `until` flag still ends it there).
        The wrist rolls the opposite way (face_turn's convention: a +90 deg face turn
        needs a -pi/2 wrist roll) with a small overshoot past the target, so a slightly
        slipping grip does not leave the angle short.
        """
        view = self._require_view()
        q = self.cmd.q[1][envs].copy()
        for i, env in enumerate(envs):
            current = float(view.layer_angle[int(env)])
            remaining = (float(view.turn_target[int(env)]) - current) % (2 * math.pi)
            if remaining > math.pi:
                remaining -= 2 * math.pi
            if abs(remaining) < 1e-9:
                continue  # already at the target: no wrist travel, `until` fires at once
            sign = 1.0 if remaining > 0 else -1.0
            q[i, -1] -= sign * quarter_turns * (math.pi / 2) + sign * TURN_OVERSHOOT
        return q[:, None, :], np.ones(len(envs), dtype=bool)


def _progress_of(primitive: Primitive) -> np.ndarray:
    """`(n,)` [0, 1] progress of a batched primitive (best effort, for observations)."""
    if isinstance(primitive, Grip):
        return np.clip(primitive.steps / max(primitive.max_steps, 1), 0, 1)
    if isinstance(primitive, TrackPath):
        if primitive.paths is None or primitive.paths.shape[1] == 0:
            return np.zeros(primitive.n)
        last = primitive.paths.shape[1] - 1
        frac = np.clip(primitive.pointer / max(last, 1), 0, 1)
        return np.where(primitive.arrived, 1.0, frac)
    if isinstance(primitive, Sequence):
        return np.clip(primitive.stage / max(len(primitive.children), 1), 0, 1)
    return np.zeros(primitive.n)
