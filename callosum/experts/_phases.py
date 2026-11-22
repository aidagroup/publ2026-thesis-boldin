"""Per-env asynchronous phase state machines for the scripted expert (pure numpy, no simulator).

Why: with hundreds of envs in one batch, a phase that starts only when *every* env has finished
the previous one is paced by the slowest env (an env with an IK failure that never "arrives", a
noisy env that never meets its tolerance), however well each phase tracks its waypoints. Here
every env has its own position in every arm's program and moves on the moment its *own*
completion criterion is met; nobody waits for another env.

Vocabulary:

* A **primitive** (`Primitive`, e.g. `TrackPath`, `Grip`) is one skill an arm can execute. Given
  the env's state (`Context`: current joint positions and speeds, face flag, step counter) it
  writes the per-env commanded target (joint target, gripper command) and reports a per-env
  *done* flag every step. All its counters (steps in the phase, waypoint pointer, error history)
  are per-env arrays held by the primitive itself. It never looks at another env.
* A **program** is the ordered list of `PhaseSpec`s of one arm: a primitive plus an optional
  **gate**, a predicate on the machine that has to hold before the phase may start. Gates are
  how ordering constraints between the arms are written (`after("holder close")`,
  `started("holder close")`); they are evaluated per env.
* The `PhaseMachine` steps the primitives of all envs and arms once per env step, records for
  each phase and env the step at which it started / completed (and why: tolerance, stall,
  timeout, ...), and starts the next phase of an env as soon as its gate allows. An env whose
  primitive cannot start (e.g. IK failure: `start` returns False) is marked **failed**: its arms
  hold their pose and it never blocks anything (`finished` counts it as done).

`Commands` are the per-arm targets the env actions are derived from (`Rig` turns them into
`pd_joint_delta_pos` actions); the primitives mutate them in place. The scripted FaceTurn program
lives in `callosum.experts._program`; a skill-level env can reuse the same primitives with a
planner that plans from the live state instead of a precomputed plan.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from callosum.configs.face_turn_expert import ExpertControlConfig
from callosum.experts._tracking import ReachMonitor, effective_tol

NUM_ARM_JOINTS = 5
REASONS = ("tol", "stall", "timeout", "extra", "still")
"""Why a phase ended: reached its tolerance, stalled, ran into its timeout, an extra per-env
condition held (face turned), or the jaws came to rest."""
TOL, STALL, TIMEOUT, EXTRA, STILL = range(len(REASONS))


@dataclass
class Observation:
    """What the primitives see of the env after a step (numpy, per arm, `(n, ...)`)."""

    qpos: Sequence[np.ndarray]  # (n, 5) arm joint positions
    arm_speed: Sequence[np.ndarray]  # (n,) largest arm joint speed
    jaw_speed: Sequence[np.ndarray]  # (n,) largest gripper jaw speed
    face_turned: np.ndarray  # (n,) bool: the face is at its target angle


@dataclass
class Commands:
    """Per-arm commanded targets, mutated in place by the primitives and read by the `Rig`.

    `q`: `(n, 5)` joint targets, `grip`: `(n,)` absolute gripper action (+-1), `bias`: `(n, 5)`
    integral term of the delta controller (forgotten when a new movement starts).
    """

    q: list[np.ndarray]
    grip: list[np.ndarray]
    bias: list[np.ndarray]

    @classmethod
    def from_qpos(cls, qpos: Sequence[np.ndarray], grip_open: float = 1.0) -> "Commands":
        """Commands that hold the given joint positions, jaws open."""
        return cls(
            q=[np.array(q, dtype=float) for q in qpos],
            grip=[np.full(len(q), grip_open) for q in qpos],
            bias=[np.zeros((len(q), NUM_ARM_JOINTS)) for q in qpos],
        )

    def hold(self, arm: int, envs: np.ndarray, qpos: np.ndarray) -> None:
        """Stop pushing: command the arm's current joint positions in `envs`."""
        self.q[arm][envs] = qpos[envs]
        self.bias[arm][envs] = 0.0


@dataclass
class Context:
    """Shared state the primitives read: config, commands, the latest observation, the clock.

    `jitter` (rad) is the joint jitter DART action noise causes (`noise_jitter`); the reach
    tolerances are widened by it. `delta_limit` is the controller's per-step bound
    (`ARM_DELTA_LIMIT`), used to size the timeouts. `t` counts env steps taken so far.
    """

    cfg: ExpertControlConfig
    cmd: Commands
    obs: Observation
    jitter: float = 0.0
    delta_limit: float = 0.05
    t: int = 0

    def tol(self, base: float) -> float:
        """`base` widened for the executed-action noise (see `effective_tol`)."""
        return effective_tol(base, self.jitter, self.cfg.noise_tol_scale)

    @property
    def settle_tol(self) -> float:
        """Speed (rad/s or m/s) below which a joint counts as at rest."""
        return self.cfg.settle_qvel_tol


class Primitive:
    """One skill of an arm, for all envs at once (see the module doc).

    Subclasses implement `start` (called once per env when it enters the phase; sets the targets,
    initialises the env's counters and returns whether it could start) and `step` (called after
    every env step while the env is in the phase; returns the per-env done flag). They store the
    reason of a completed env in `reason[env]` (an index into `REASONS`).
    """

    def __init__(self, n: int, name: str) -> None:
        self.n = n
        self.name = name
        self.steps = np.zeros(n, dtype=int)  # env steps spent in the phase
        self.reason = np.full(n, -1, dtype=int)

    def start(self, ctx: Context, arm: int, envs: np.ndarray) -> np.ndarray:
        """Enter the phase in `envs` (int indices); returns `(len(envs),)` bool "started"."""
        raise NotImplementedError

    def step(self, ctx: Context, arm: int, envs: np.ndarray) -> np.ndarray:
        """One env step later: `(len(envs),)` bool, which of `envs` completed the phase."""
        raise NotImplementedError


# `planner(envs) -> (waypoints (k, W, 5), ok (k,))`: joint waypoints for the envs entering a
# `TrackPath` (batched, so several envs entering at the same step are planned together). `ok` is
# False where there is no valid path (IK failure): the env is marked failed.
Planner = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


class TrackPath(Primitive):
    """Drive an arm through joint waypoints, closed loop, ending when the env is done with them.

    Every env has its own waypoint pointer: it moves to the next waypoint once the joint error to
    the current (clean) target is within `waypoint_tol` or stalled, and the phase ends at the last
    one within `final_tol` (`approach_tol` for a `fly_by`: the pose needs no precision because
    the next path starts from the commanded pose, and it is not settled), or stalled, or when
    the optional per-env `until(ctx)` flag holds (e.g. the face is turned), or when the env's
    own timeout (travel time at the controller's top speed plus slack) runs out. All tolerances
    are widened by the executed-action noise (`Context.jitter`) and the stall detector works on
    a windowed mean (see `ReachMonitor`), so noise neither blocks reaching nor hides a stall.
    Afterwards the arm settles for up to `settle` steps (capped by `ExpertControlConfig.
    settle_cap`, at most 2 under noise: the jitter never lets the joints come to rest) or until
    its joints are still. `hold_on_done` commands the current pose when the phase ends (the
    target overshoots on purpose, e.g. the wrist roll of the turn).
    """

    def __init__(
        self,
        n: int,
        name: str,
        planner: Planner,
        *,
        fly_by: bool = False,
        settle: int = 0,
        until: Callable[[Context], np.ndarray] | None = None,
        hold_on_done: bool = False,
    ) -> None:
        super().__init__(n, name)
        self.planner = planner
        self.fly_by = fly_by
        self.settle = settle
        self.until = until
        self.hold_on_done = hold_on_done
        self.paths: np.ndarray | None = None  # (n, W, 5)
        self.pointer = np.zeros(n, dtype=int)
        self.budget = np.zeros(n, dtype=int)
        self.arrived = np.zeros(n, dtype=bool)  # at the end of the path, now settling
        self.settled_steps = np.zeros(n, dtype=int)
        self.monitor: ReachMonitor | None = None

    def start(self, ctx: Context, arm: int, envs: np.ndarray) -> np.ndarray:
        cfg = ctx.cfg
        paths, ok = self.planner(envs)
        ok = np.asarray(ok, dtype=bool)
        good = envs[ok]
        if self.paths is None:
            self.paths = np.zeros((self.n, paths.shape[1], NUM_ARM_JOINTS))
            self.monitor = ReachMonitor(0.0, cfg.stall_window, cfg.stall_progress)
        if len(good) == 0:
            return ok
        self.paths[good] = paths[ok]
        self.pointer[good] = 0
        self.steps[good] = 0
        self.arrived[good] = False
        self.settled_steps[good] = 0
        self.reason[good] = -1
        ctx.cmd.q[arm][good] = self.paths[good, 0]
        ctx.cmd.bias[arm][good] = 0.0
        mask = np.zeros(self.n, dtype=bool)
        mask[good] = True
        self.monitor.reset(mask)
        # The env's own timeout: its longest joint travel at top speed with 50% margin, plus
        # slack per waypoint and for the last millimetres.
        points = np.concatenate([ctx.obs.qpos[arm][good][:, None], self.paths[good]], axis=1)
        travel = np.abs(np.diff(points, axis=1)).max(axis=2).sum(axis=1)
        self.budget[good] = (
            np.ceil(1.5 * travel / (0.4 * ctx.delta_limit)).astype(int)
            + cfg.waypoint_timeout * self.paths.shape[1]
            + cfg.final_timeout
        )
        return ok

    def step(self, ctx: Context, arm: int, envs: np.ndarray) -> np.ndarray:
        cfg, cmd = ctx.cfg, ctx.cmd
        active = np.zeros(self.n, dtype=bool)
        active[envs] = True
        self.steps[envs] += 1
        qpos = ctx.obs.qpos[arm]
        error = np.abs(cmd.q[arm] - qpos).max(axis=1)
        last = self.paths.shape[1] - 1
        on_last = self.pointer >= last
        final_tol = cfg.approach_tol if self.fly_by else cfg.final_tol
        reached, stalled = self.monitor.update(
            error,
            np.where(on_last, ctx.tol(final_tol), ctx.tol(cfg.waypoint_tol)),
            cfg.stall_progress + ctx.jitter,
        )

        # Settling envs: the path is done, wait for the arm to be still (or the settle budget).
        settling = active & self.arrived
        done = np.zeros(self.n, dtype=bool)
        if settling.any():
            self.settled_steps[settling] += 1
            still = ctx.obs.arm_speed[arm] < ctx.settle_tol
            done |= settling & (still | (self.settled_steps >= self._settle_steps(ctx)))

        moving = active & ~self.arrived
        advance = moving & (reached | stalled) & ~on_last
        if advance.any():
            self.pointer[advance] += 1
            moved = np.flatnonzero(advance)
            cmd.q[arm][moved] = self.paths[moved, self.pointer[moved]]
            self.monitor.reset(advance)
        extra = self._until(ctx)
        arrive = moving & on_last & (reached | stalled | extra | (self.steps >= self.budget))
        if arrive.any():
            self.arrived[arrive] = True
            self.reason[arrive] = np.where(
                reached[arrive],
                TOL,
                np.where(extra[arrive], EXTRA, np.where(stalled[arrive], STALL, TIMEOUT)),
            )
            if self.hold_on_done:
                cmd.hold(arm, np.flatnonzero(arrive), qpos)
            if self._settle_steps(ctx) == 0:
                done |= arrive
        return done[envs]

    def _until(self, ctx: Context) -> np.ndarray:
        if self.until is None:
            return np.zeros(self.n, dtype=bool)
        return np.asarray(self.until(ctx), dtype=bool)

    def _settle_steps(self, ctx: Context) -> int:
        if self.fly_by:
            return 0
        steps = min(self.settle, ctx.cfg.settle_cap)
        return min(steps, 2) if ctx.jitter > 0 else steps


class Grip(Primitive):
    """Open or close an arm's jaws: done once the jaws are at rest (after `min_steps`).

    The gripper action is absolute (+-1) and never noised. `max_steps` is the per-env timeout.
    """

    def __init__(self, n: int, name: str, value: float, max_steps: int, min_steps: int = 4) -> None:
        super().__init__(n, name)
        self.value = value
        self.max_steps = max_steps
        self.min_steps = min(min_steps, max_steps)

    def start(self, ctx: Context, arm: int, envs: np.ndarray) -> np.ndarray:
        ctx.cmd.grip[arm][envs] = self.value
        self.steps[envs] = 0
        self.reason[envs] = -1
        return np.ones(len(envs), dtype=bool)

    def step(self, ctx: Context, arm: int, envs: np.ndarray) -> np.ndarray:
        self.steps[envs] += 1
        steps = self.steps[envs]
        still = (steps >= self.min_steps) & (ctx.obs.jaw_speed[arm][envs] < ctx.settle_tol)
        done = still | (steps >= self.max_steps)
        self.reason[envs[done]] = np.where(still[done], STILL, TIMEOUT)
        return done


Gate = Callable[["PhaseMachine"], np.ndarray]


def after(name: str) -> Gate:
    """Gate: the phase `name` (of either arm) has completed in this env."""
    return lambda machine: machine.completed(name)


def started(name: str) -> Gate:
    """Gate: the phase `name` (of either arm) has started in this env."""
    return lambda machine: machine.has_started(name)


def all_of(*gates: Gate) -> Gate:
    """Gate: every one of `gates` holds."""
    return lambda machine: np.logical_and.reduce([g(machine) for g in gates])


@dataclass
class PhaseSpec:
    """One phase of an arm's program: a primitive and the gate that must hold to start it."""

    primitive: Primitive
    gate: Gate | None = None

    @property
    def name(self) -> str:
        return self.primitive.name


@dataclass
class PhaseMachine:
    """Per-env state machines of all arms; see the module doc.

    Args:
        programs: one program (list of `PhaseSpec`) per arm; phase names must be unique.
        ctx: the shared `Context`.
        n: number of envs.
        failed: envs that failed before the run (e.g. IK failure while planning): they hold.

    Call `begin(obs)` once, then `update(obs)` after every env step.
    """

    programs: Sequence[Sequence[PhaseSpec]]
    ctx: Context
    n: int
    failed: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.failed = np.zeros(self.n, dtype=bool) if self.failed is None else self.failed.copy()
        self.names = [spec.name for prog in self.programs for spec in prog]
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"phase names must be unique, got {self.names}")
        self.index = [np.zeros(self.n, dtype=int) for _ in self.programs]
        self.entered = [np.zeros(self.n, dtype=bool) for _ in self.programs]
        self.start_step = {name: np.full(self.n, -1, dtype=int) for name in self.names}
        self.done_step = {name: np.full(self.n, -1, dtype=int) for name in self.names}
        self.done_reason = {name: np.full(self.n, -1, dtype=int) for name in self.names}
        self._spec = {spec.name: spec for prog in self.programs for spec in prog}

    # --- queries (also what the gates use) ------------------------------------------------

    def completed(self, name: str) -> np.ndarray:
        """`(n,)` bool: the phase `name` has completed in each env."""
        return self.done_step[name] >= 0

    def has_started(self, name: str) -> np.ndarray:
        """`(n,)` bool: the phase `name` has been started in each env."""
        return self.start_step[name] >= 0

    @property
    def finished(self) -> np.ndarray:
        """`(n,)` bool: the env completed both arms' programs, or failed."""
        done = self.failed.copy()
        done |= np.logical_and.reduce(
            [idx >= len(prog) for idx, prog in zip(self.index, self.programs, strict=True)]
        )
        return done

    def all_finished(self) -> bool:
        return bool(self.finished.all())

    def completed_fraction(self, name: str) -> float:
        """Share of the non-failed envs that completed the phase (1.0 when all failed)."""
        live = ~self.failed
        return float((self.completed(name) & live).sum() / max(int(live.sum()), 1))

    # --- driving -------------------------------------------------------------------------

    def begin(self, obs: Observation) -> None:
        """Start: failed envs hold, every env enters the first phases whose gates hold."""
        self.ctx.obs = obs
        self.mark_failed(np.flatnonzero(self.failed))
        self._start_phases()

    def update(self, obs: Observation) -> None:
        """After an env step: step the running primitives, then start whatever became possible."""
        self.ctx.obs = obs
        self.ctx.t += 1
        t = self.ctx.t
        for arm, program in enumerate(self.programs):
            for p, spec in enumerate(program):
                running = np.flatnonzero((self.index[arm] == p) & self.entered[arm] & ~self.failed)
                if len(running) == 0:
                    continue
                done = spec.primitive.step(self.ctx, arm, running)
                finished = running[done]
                self.done_step[spec.name][finished] = t
                self.done_reason[spec.name][finished] = spec.primitive.reason[finished]
                self.index[arm][finished] += 1
                self.entered[arm][finished] = False
        self._start_phases()

    def mark_failed(self, envs: np.ndarray) -> None:
        """Mark envs failed: both arms hold their current pose and the env never blocks anyone."""
        envs = np.asarray(envs, dtype=int)
        self.failed[envs] = True
        if len(envs):
            for arm in range(len(self.programs)):
                self.ctx.cmd.hold(arm, envs, self.ctx.obs.qpos[arm])

    def _start_phases(self) -> None:
        """Start the next phase of every env whose gate holds (repeat: starts can enable gates)."""
        total = sum(len(prog) for prog in self.programs)
        for _ in range(total + 1):
            changed = False
            for arm, program in enumerate(self.programs):
                for p, spec in enumerate(program):
                    waiting = (self.index[arm] == p) & ~self.entered[arm] & ~self.failed
                    if spec.gate is not None:
                        waiting &= np.asarray(spec.gate(self), dtype=bool)
                    envs = np.flatnonzero(waiting)
                    if len(envs) == 0:
                        continue
                    ok = spec.primitive.start(self.ctx, arm, envs)
                    self.entered[arm][envs[ok]] = True
                    self.start_step[spec.name][envs[ok]] = self.ctx.t
                    if not ok.all():
                        self.mark_failed(envs[~ok])
                    changed = True
            if not changed:
                return

    # --- reporting -----------------------------------------------------------------------

    def describe(self) -> list[str]:
        """One line per phase: how many envs completed it, the distribution of the env step at
        which they did, and why they ended it; plus the failed / unfinished counts."""
        lines = []
        live = ~self.failed
        for name in self.names:
            done = self.done_step[name]
            mask = (done >= 0) & live
            row = f"  {name:<24} done {int(mask.sum()):>4}/{int(live.sum()):<4}"
            if mask.any():
                q = np.percentile(done[mask], [0, 50, 100])
                row += f"  env step min {q[0]:.0f} / median {q[1]:.0f} / max {q[2]:.0f}"
                counts = [
                    f"{r} {int((self.done_reason[name][mask] == i).sum())}"
                    for i, r in enumerate(REASONS)
                    if (self.done_reason[name][mask] == i).any()
                ]
                row += f"  [{', '.join(counts)}]"
            lines.append(row)
        unfinished = int((~self.finished).sum())
        lines.append(
            f"  envs failed (held, e.g. IK failure): {int(self.failed.sum())}/{self.n}; "
            f"envs that did not finish their program: {unfinished}/{self.n}"
        )
        return lines
