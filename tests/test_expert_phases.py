"""Per-env asynchronous phase machines of the scripted expert (`_phases`, `_program`): pure numpy.

A toy plant stands in for the simulator: every arm follows the delta controller's first-order lag
(about 0.02 rad per step at most, per-env speed factor, optionally blocked or noised), jaws follow
the gripper command, and the "face" counts as turned once the rotator's wrist rolled far enough
while its jaws are closed. The expert's own tracking functions drive it, exactly as `Rig` does.
"""

import numpy as np
import pytest

from callosum.configs.face_turn_expert import ExpertControlConfig
from callosum.experts._phases import (
    STALL,
    TIMEOUT,
    TOL,
    Commands,
    Context,
    Grip,
    Observation,
    PhaseMachine,
    TrackPath,
    started,
)
from callosum.experts._program import (
    HALF_TURN,
    HOLDER_CLOSE,
    HOLDER_GRASP,
    HOLDER_PREGRASP,
    HOLDER_RELEASE,
    RELEASE,
    ROTATOR_CLOSE,
    ROTATOR_DESCENT,
    ROTATOR_PREGRASP,
    TURN,
    ExpertPlan,
    build_programs,
)
from callosum.experts._tracking import noise_jitter, tracking_action, update_bias

LIMIT = 0.05
STEP_FRACTION = 0.39
TURNED_WRIST = -1.67
FACE_WRIST = -1.55  # the toy face counts as turned when the rotator's wrist is below this


def _legs(start: np.ndarray, goal: np.ndarray, waypoints: int = 4) -> np.ndarray:
    return np.stack([start + (goal - start) * (k / waypoints) for k in range(1, waypoints + 1)], 1)


def make_plan(scale: np.ndarray) -> ExpertPlan:
    """A plan whose joint distances scale with `scale` (per env): longer plans take longer."""
    n = len(scale)
    s = scale[:, None]
    zero = np.zeros((n, 5))
    pre_a = _legs(zero, s * np.array([0.8, 0.6, -0.5, 0.4, 0.0]))
    grasp_a = _legs(pre_a[:, -1], pre_a[:, -1] + s * np.array([0.0, 0.1, 0.0, 0.1, 0.0]))
    pre_b = _legs(zero, s * np.array([0.5, -0.5, 0.4, 0.4, 0.0]))
    grasp_b = _legs(pre_b[:, -1], pre_b[:, -1] * np.array([1, 1, 1, 1, 0]) + s * 0.1)
    grasp_b[:, :, 4] = 0.0  # the wrist roll starts at 0
    last = grasp_b[:, -1:].copy()
    turned, half = last.copy(), last.copy()
    turned[:, :, 4] = TURNED_WRIST
    half[:, :, 4] = TURNED_WRIST / 2
    return ExpertPlan(pre_a, grasp_a, pre_b, grasp_b, half, turned, np.ones(n, dtype=bool))


class Plant:
    """Toy arms + jaws + face for `n` envs; `speed[arm]` scales each env's joint speed."""

    def __init__(self, n, cfg, speed=None, blocked=None, noise=0.0, seed=0):
        self.n, self.cfg, self.noise = n, cfg, noise
        self.rng = np.random.default_rng(seed)
        self.qpos = [np.zeros((n, 5)) for _ in range(2)]
        self.jaw = [np.ones(n) for _ in range(2)]
        self.speed = np.ones((2, n)) if speed is None else np.asarray(speed, dtype=float)
        self.blocked = np.zeros((2, n), dtype=bool) if blocked is None else np.asarray(blocked)
        self.cmd = Commands.from_qpos(self.qpos)
        self._arm_speed = [np.zeros(n) for _ in range(2)]
        self._jaw_speed = [np.zeros(n) for _ in range(2)]

    def observe(self) -> Observation:
        face = (self.qpos[1][:, 4] <= FACE_WRIST) & (self.jaw[1] < 0)
        return Observation(
            [q.copy() for q in self.qpos], list(self._arm_speed), list(self._jaw_speed), face
        )

    def step(self) -> None:
        for arm in range(2):
            error = self.cmd.q[arm] - self.qpos[arm]
            action, saturated = tracking_action(error, self.cmd.bias[arm], self.cfg.gain, LIMIT)
            self.cmd.bias[arm] = update_bias(
                self.cmd.bias[arm], error, self.cfg.ki, self.cfg.bias_max, saturated
            )
            if self.noise > 0:
                action = np.clip(action + self.rng.normal(0, self.noise, action.shape), -1, 1)
            move = STEP_FRACTION * action * LIMIT * self.speed[arm][:, None]
            move[self.blocked[arm]] = 0.0
            self.qpos[arm] = self.qpos[arm] + move
            self._arm_speed[arm] = np.abs(move).max(axis=1) * 20
            jaw = self.jaw[arm] + 0.4 * (self.cmd.grip[arm] - self.jaw[arm])
            self._jaw_speed[arm] = np.abs(jaw - self.jaw[arm]) * 20
            self.jaw[arm] = jaw


def run(
    plan,
    cfg=None,
    *,
    speed=None,
    blocked=None,
    noise=0.0,
    no_holder=False,
    release_holder=False,
    max_steps=1500,
    tweak=None,
):
    """Run the scripted program on the toy plant; returns `(machine, plant, steps_taken)`."""
    n = len(plan.ok)
    cfg = cfg or ExpertControlConfig(control="delta", overlap_approach=True)
    plant = Plant(n, cfg, speed, blocked, noise)
    programs = build_programs(plan, cfg, no_holder=no_holder, release_holder=release_holder)
    if tweak:
        tweak(programs)
    ctx = Context(cfg, plant.cmd, plant.observe(), jitter=noise_jitter(noise, LIMIT))
    machine = PhaseMachine(programs, ctx, n, failed=~plan.ok)
    machine.begin(plant.observe())
    steps = 0
    while not machine.all_finished() and steps < max_steps:
        plant.step()
        machine.update(plant.observe())
        steps += 1
    return machine, plant, steps


SCALE = np.array([0.6, 1.0, 1.6, 0.8])
SPEED = np.array([[1.0, 0.7, 1.0, 0.5], [1.0, 1.0, 0.6, 0.8]])  # arm 0, arm 1 per env


def test_the_whole_program_completes_in_every_env() -> None:
    machine, plant, _ = run(make_plan(SCALE), speed=SPEED)
    assert machine.finished.all() and not machine.failed.any()
    for name in machine.names:
        assert machine.completed(name).all(), name
    assert (plant.qpos[1][:, 4] <= FACE_WRIST).all()  # the face got turned


def test_envs_advance_independently_each_one_matches_its_solo_run() -> None:
    cfg = ExpertControlConfig(control="delta", overlap_approach=True)
    plan = make_plan(SCALE)
    batch, _, _ = run(plan, cfg, speed=SPEED)
    # Different envs take different times (they are not paced by the slowest one) ...
    assert len(set(batch.done_step[TURN])) == len(SCALE)
    # ... and every env's own completion steps are exactly those of running it alone.
    for i in range(len(SCALE)):
        solo_plan = make_plan(SCALE[i : i + 1])
        solo, _, _ = run(solo_plan, cfg, speed=SPEED[:, i : i + 1])
        for name in batch.names:
            assert batch.done_step[name][i] == solo.done_step[name][0], (i, name)
            assert batch.start_step[name][i] == solo.start_step[name][0], (i, name)


def test_a_failed_env_holds_and_blocks_nobody() -> None:
    plan = make_plan(SCALE)
    healthy, _, healthy_steps = run(plan, speed=SPEED)
    plan.ok[2] = False
    machine, plant, steps = run(plan, speed=SPEED)
    assert machine.failed.tolist() == [False, False, True, False]
    assert machine.finished.all()  # counted as done
    # The failed env holds still ...
    assert not machine.completed(TURN)[2] and not machine.has_started(HOLDER_PREGRASP)[2]
    assert np.allclose(plant.qpos[0][2], 0.0) and np.allclose(plant.qpos[1][2], 0.0)
    # ... and the others are not delayed by it at all.
    for name in machine.names:
        for i in (0, 1, 3):
            assert machine.done_step[name][i] == healthy.done_step[name][i], (i, name)
    assert steps <= healthy_steps


def test_an_env_that_fails_midway_is_marked_and_held() -> None:
    def fail_env_1_at_descent(programs) -> None:
        descent = programs[1][1].primitive
        real = descent.planner

        def planner(envs):
            paths, ok = real(envs)
            return paths, ok & (envs != 1)

        descent.planner = planner

    machine, plant, _ = run(make_plan(SCALE), speed=SPEED, tweak=fail_env_1_at_descent)
    assert machine.failed.tolist() == [False, True, False, False]
    assert machine.finished.all()
    assert not machine.completed(ROTATOR_DESCENT)[1]
    assert machine.completed(TURN)[[0, 2, 3]].all()
    # It froze where it was when it failed: the target is its current pose, no pushing.
    assert np.allclose(plant.cmd.q[1][1], plant.qpos[1][1], atol=1e-6)


def test_rotator_descends_only_after_this_envs_holder_started_closing() -> None:
    # Env 0 has a very slow holder, env 1 a fast one; both rotators are equally fast.
    speed = np.array([[0.3, 1.0], [1.0, 1.0]])
    machine, _, _ = run(make_plan(np.array([1.0, 1.0])), speed=speed)
    close_start, descent_start = (
        machine.start_step[HOLDER_CLOSE],
        machine.start_step[ROTATOR_DESCENT],
    )
    assert (descent_start >= close_start).all()
    # The fast env's rotator does not wait for the slow env's holder.
    assert descent_start[1] < close_start[0]
    # Env-wise, the rotator's pregrasp finished before it could descend.
    assert (descent_start >= machine.done_step[ROTATOR_PREGRASP]).all()


def test_turn_needs_both_grasps_and_follows_the_rotator_close_in_this_env() -> None:
    machine, _, _ = run(make_plan(SCALE), speed=SPEED)
    turn_start = machine.start_step[TURN]
    assert (turn_start >= machine.done_step[ROTATOR_CLOSE]).all()
    assert (turn_start >= machine.done_step[HOLDER_CLOSE]).all()
    assert (machine.start_step[RELEASE] >= machine.done_step[TURN]).all()

    # A holder whose jaws take long to close (60 steps) is what the turn waits for.
    def slow_jaws(programs) -> None:
        close = programs[0][2].primitive
        close.min_steps = close.max_steps = 60

    slow, _, _ = run(make_plan(np.array([1.0, 1.0])), tweak=slow_jaws)
    assert (slow.done_step[HOLDER_CLOSE] > slow.done_step[ROTATOR_CLOSE]).all()
    assert (slow.start_step[TURN] >= slow.done_step[HOLDER_CLOSE]).all()


def test_without_overlap_the_rotator_waits_for_the_holder_grasp() -> None:
    cfg = ExpertControlConfig(control="delta", overlap_approach=False)
    machine, _, _ = run(make_plan(SCALE), cfg, speed=SPEED)
    assert (machine.start_step[ROTATOR_PREGRASP] >= machine.done_step[HOLDER_CLOSE]).all()
    assert machine.finished.all()


def test_no_holder_program_only_has_the_rotator() -> None:
    machine, _, _ = run(make_plan(SCALE), no_holder=True, speed=SPEED)
    assert machine.finished.all()
    assert HOLDER_GRASP not in machine.names and machine.completed(TURN).all()


def test_release_holder_opens_after_the_half_turn_and_the_turn_waits_for_it() -> None:
    machine, _, _ = run(make_plan(SCALE), release_holder=True, speed=SPEED)
    assert machine.finished.all()
    assert (machine.start_step[HOLDER_RELEASE] >= machine.done_step[HALF_TURN]).all()
    assert (machine.start_step[TURN] >= machine.done_step[HOLDER_RELEASE]).all()


def test_the_turn_ends_on_the_face_flag_not_the_unreachable_wrist_target() -> None:
    machine, plant, _ = run(make_plan(SCALE), speed=SPEED)
    assert (machine.done_reason[TURN] == 3).all()  # EXTRA: the face flag
    # The arm was told to hold the pose it had when the face got there.
    assert np.allclose(plant.cmd.q[1], plant.qpos[1], atol=1e-6)
    assert (plant.qpos[1][:, 4] > TURNED_WRIST + 0.05).all()


def test_noise_neither_blocks_reaching_nor_causes_timeouts() -> None:
    plan = make_plan(np.tile(SCALE, 8))
    machine, _, steps = run(plan, speed=np.tile(SPEED, 8), noise=0.1)
    assert machine.finished.all() and steps < 1500
    for name in (HOLDER_PREGRASP, HOLDER_GRASP, ROTATOR_PREGRASP, ROTATOR_DESCENT):
        reasons = machine.done_reason[name]
        assert (reasons != TIMEOUT).all(), name
        assert (reasons == TOL).mean() > 0.9, name  # by reaching, not by a lucky stall


def test_a_blocked_arm_is_ended_by_stall_not_by_the_timeout_even_under_noise() -> None:
    n = 6
    blocked = np.zeros((2, n), dtype=bool)
    blocked[1, :3] = True  # the rotator of envs 0-2 cannot move at all
    plan = make_plan(np.ones(n))
    cfg = ExpertControlConfig(control="delta", overlap_approach=True)
    for noise in (0.0, 0.1):
        machine, _, _ = run(plan, cfg, blocked=blocked, noise=noise)
        stalled = machine.done_reason[ROTATOR_PREGRASP]
        assert (stalled[:3] == STALL).all() or (stalled[:3] == TOL).all()
        assert (stalled[3:] == TOL).all()
        # It ended long before its own timeout (budget = travel time + slacks, > 100 steps).
        budget = machine._spec[ROTATOR_PREGRASP].primitive.budget
        assert (machine.done_step[ROTATOR_PREGRASP][:3] < budget[:3] / 2).all()
        # The healthy envs are unaffected by the blocked ones.
        assert machine.completed(TURN)[3:].all()


def test_describe_lists_every_phase_and_the_failed_count() -> None:
    plan = make_plan(SCALE)
    plan.ok[0] = False
    machine, _, _ = run(plan, speed=SPEED)
    text = "\n".join(machine.describe())
    for name in machine.names:
        assert name in text
    assert "done    3/3" in text
    assert "min " in text and "median " in text and "max " in text
    assert "envs failed (held, e.g. IK failure): 1/4" in text


def test_a_primitive_can_be_reused_with_a_live_planner_and_a_custom_gate() -> None:
    # The building blocks of the skill-level env: a TrackPath whose planner plans from the live
    # state, gated by a condition on the other arm, and a Grip.
    n = 2
    cfg = ExpertControlConfig(control="delta")
    plant = Plant(n, cfg)
    target = np.array([[0.3, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0, 0.0]])

    def live_planner(envs):
        return target[envs][:, None, :], np.ones(len(envs), dtype=bool)

    go = TrackPath(n, "go", live_planner, settle=0)
    close = Grip(n, "close", -1.0, 20)
    from callosum.experts._phases import PhaseSpec

    programs = [[PhaseSpec(go)], [PhaseSpec(close, gate=started("go"))]]
    ctx = Context(cfg, plant.cmd, plant.observe())
    machine = PhaseMachine(programs, ctx, n)
    machine.begin(plant.observe())
    while not machine.all_finished():
        plant.step()
        machine.update(plant.observe())
    assert np.allclose(plant.qpos[0], target, atol=cfg.final_tol)
    assert (machine.done_step["go"] > 0).all() and machine.completed("close").all()
    with pytest.raises(ValueError, match="unique"):
        PhaseMachine([[PhaseSpec(go)], [PhaseSpec(go)]], ctx, n)
