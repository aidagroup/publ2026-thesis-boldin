"""Unit tests of the FaceTurnSkills pure logic (config, runtime, conductor).

Everything here imports ONLY the macOS-safe modules (`callosum.configs.face_turn_skills`,
`callosum.experts.skills_runtime`, `callosum.experts.skills_conductor`, the `_skills*`
contracts): no mani_skill / sapien / torch anywhere, per the project's standing rule
(the env module itself is exercised by the [cloud] smoke script).

Covers: config validation, the decision/interruption machine (mock primitives), the
useless-action classifier, the layer-target mapping and quaternion/orientation math,
the own-block layout against the trainer's slicing contract, and the conductor's
decision sequences on synthetic observations.
"""

import math

import numpy as np
import pytest

from callosum.configs.face_turn_skills import FaceTurnSkillsConfig, SkillsRewardConfig
from callosum.envs._plans import FACES, CubeOrientation, Flip, PlanMove
from callosum.envs._skills import Role, num_primitives
from callosum.envs._skills_obs import own_block_size, own_block_slices
from callosum.experts.skills_conductor import HOLDER_ID, ROTATOR_ID, SkillsConductor
from callosum.experts.skills_runtime import (
    DecisionFacts,
    RuntimeView,
    SkillsRuntime,
    apply_decision,
    orientation_from_quat,
    own_block_key_layout,
    turn_target_rad,
    up_face_from_quat,
    useless_actions,
)

N_HOLDER, N_ROTATOR = num_primitives(Role.HOLDER), num_primitives(Role.ROTATOR)


# --- config ------------------------------------------------------------------------------


def test_config_defaults_match_the_specs() -> None:
    cfg = FaceTurnSkillsConfig()
    assert cfg.decision_period == 20
    assert 100 <= cfg.max_episode_decisions <= 200
    assert cfg.partner_mode == "none"
    assert cfg.plan_length == 1  # E1
    assert cfg.reward.success_bonus == 1.0
    assert 0 < cfg.reward.step_penalty < 0.1
    assert cfg.lock_layers_unless_held
    assert 0 < cfg.lock_pull_step_deg <= cfg.lock_write_tol_deg < cfg.angle_tol_deg
    assert cfg.angle_tol_rad == pytest.approx(math.radians(cfg.angle_tol_deg))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"decision_period": 0}, "decision_period"),
        ({"max_episode_decisions": 0}, "max_episode_decisions"),
        ({"plan_length": -1}, "plan_length"),
        ({"partner_mode": "full"}, "mode must be one of"),
        ({"slip_prob_range": (0.2, 0.1)}, "slip_prob_range"),
        ({"slip_prob_range": (-0.1, 0.2)}, "slip_prob_range"),
        ({"angle_tol_deg": 0.0}, "angle_tol_deg"),
        ({"lock_write_tol_deg": 5.0}, "lock_write_tol_deg"),
        ({"lock_pull_step_deg": 0.0}, "lock_pull_step_deg"),
        ({"lock_pull_step_deg": 4.0}, "lock_pull_step_deg"),
        ({"body_pos_tol": 0.0}, "body_pos_tol"),
        ({"body_settle_qvel": -1.0}, "body_settle_qvel"),
        (lambda: {"reward": SkillsRewardConfig(success_bonus=-1.0)}, "success_bonus"),
        (lambda: {"reward": SkillsRewardConfig(step_penalty=2.0)}, "step_penalty"),
    ],
)
def test_config_rejects_bad_values(overrides, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        kwargs = overrides() if callable(overrides) else overrides
        FaceTurnSkillsConfig(**kwargs)


def test_reward_config_requires_penalty_below_bonus() -> None:
    with pytest.raises(ValueError, match="below success_bonus"):
        SkillsRewardConfig(success_bonus=0.01, step_penalty=0.01)


# --- decision / interruption machine -------------------------------------------------------


def test_apply_decision_continues_reselected_primitives() -> None:
    current = np.array([1, 2, 0])
    new, changed = apply_decision(current, np.array([1, 2, 3]))
    assert changed.tolist() == [False, False, True]
    assert new.tolist() == [1, 2, 3]
    assert current.tolist() == [1, 2, 0]  # the input is not mutated


class _FakePrimitive:
    """Records start/step calls; `ok` controls whether starts succeed."""

    def __init__(self, ok: bool = True) -> None:
        self.started: list[list[int]] = []
        self.stepped: list[list[int]] = []
        self.ok = ok

    def start(self, ctx, arm: int, envs) -> np.ndarray:
        self.started.append(list(map(int, envs)))
        return np.full(len(envs), self.ok, dtype=bool)

    def step(self, ctx, arm: int, envs) -> np.ndarray:
        self.stepped.append(list(map(int, envs)))
        return np.zeros(len(envs), dtype=bool)


def _view(n: int, t: int = 0, needed: list | None = None) -> RuntimeView:
    zeros = np.zeros(n)
    return RuntimeView(
        qpos=[np.zeros((n, 5)) for _ in range(2)],
        arm_speed=[zeros.copy() for _ in range(2)],
        jaw_speed=[zeros.copy() for _ in range(2)],
        base_pos=[np.zeros((n, 3)) for _ in range(2)],
        base_rot=[np.eye(3)[None].repeat(n, axis=0) for _ in range(2)],
        cube_pos=np.zeros((n, 3)),
        cube_quat=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        layer_angle=zeros.copy(),
        turn_target=zeros.copy(),
        layer_at_target=zeros.astype(bool),
        needed_flip=needed if needed is not None else [None] * n,
        t=t,
    )


class _RecordingRuntime(SkillsRuntime):
    """A runtime whose primitives are test fakes instead of IK planners."""

    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.fakes: dict[tuple[int, int], _FakePrimitive] = {}

    def _build_primitive(self, arm: int, primitive_id: int):
        fake = _FakePrimitive(ok=self.fake_ok.get((arm, primitive_id), True))
        self.fakes[(arm, primitive_id)] = fake
        return fake

    fake_ok: dict = None


def _recording_runtime(n: int, ok: dict | None = None) -> _RecordingRuntime:
    runtime = _RecordingRuntime(n)
    object.__setattr__(runtime, "fake_ok", ok or {})
    runtime.reset(_view(n), np.ones((2, n)), np.zeros((2, n), dtype=int))
    return runtime


def test_runtime_starts_new_primitive_and_interrupts_old() -> None:
    runtime = _recording_runtime(2)
    runtime.set_decision(0, np.array([1, 1]), _view(2))
    assert runtime.fakes[(0, 1)].started == [[0, 1]]

    runtime.substep(_view(2, t=1))
    assert runtime.fakes[(0, 1)].stepped == [[0, 1]]

    # Env 0 keeps primitive 1 (continues, no restart); env 1 switches to 3 (interrupt).
    runtime.set_decision(0, np.array([1, 3]), _view(2, t=1))
    assert runtime.fakes[(0, 3)].started == [[1]]
    runtime.substep(_view(2, t=2))
    stepped = [call for call in runtime.fakes[(0, 1)].stepped if call != [0, 1]]
    # Env 0 still steps primitive 1; env 1 does not (it was interrupted).
    assert runtime.fakes[(0, 1)].stepped[-1] == [0]
    assert stepped == [] or stepped[-1] == [0]
    assert runtime.fakes[(0, 3)].stepped[-1] == [1]
    assert runtime.selections()[0].tolist() == [1, 3]


def test_runtime_reselecting_the_same_primitive_does_not_restart() -> None:
    runtime = _recording_runtime(1)
    runtime.set_decision(0, np.array([2]), _view(1))
    runtime.set_decision(0, np.array([2]), _view(1, t=1))
    assert runtime.fakes[(0, 2)].started == [[0]]  # started exactly once


def test_runtime_failed_start_falls_back_to_wait() -> None:
    runtime = _recording_runtime(2, ok={(1, 3): False})
    runtime.set_decision(1, np.array([3, 3]), _view(2))
    assert runtime.selections()[1].tolist() == [0, 0]


def test_runtime_start_delay_holds_the_hand() -> None:
    runtime = _recording_runtime(1)
    runtime.reset(_view(1, t=0), np.ones((2, 1)), np.full((2, 1), 5, dtype=int))
    runtime.set_decision(0, np.array([1]), _view(1, t=0))
    runtime.substep(_view(1, t=1))  # before the delay: nothing steps
    assert runtime.fakes[(0, 1)].stepped == []
    runtime.substep(_view(1, t=5))
    assert runtime.fakes[(0, 1)].stepped == [[0]]


def test_runtime_reuses_running_instance_for_joining_envs() -> None:
    runtime = _recording_runtime(2)
    runtime.set_decision(0, np.array([1, 0]), _view(2))
    runtime.substep(_view(2, t=1))
    # Env 1 joins primitive 1 while env 0 keeps running it: same instance, start only
    # for the joining env (the other one's progress must survive).
    runtime.set_decision(0, np.array([1, 1]), _view(2, t=1))
    assert runtime.fakes[(0, 1)].started == [[0], [1]]


def test_runtime_actions_shape_and_grip_passthrough() -> None:
    runtime = _recording_runtime(3)
    actions = runtime.actions(_view(3))
    assert len(actions) == 2
    for arm_action in actions:
        assert arm_action.shape == (3, 6)
        assert np.all(arm_action[:, :5] == 0.0)  # hold: zero arm deltas
        np.testing.assert_allclose(arm_action[:, 5], 1.0)  # gripper open (+1)


def test_holder_grasp_is_a_close_then_home_drag_macro() -> None:
    """The holder's grasp ends by dragging the clamped cube back to the home XY
    along the table: the flip's release-drop lands it 2-4 cm off centre, past
    the rotator's vertical-finger zone."""
    from callosum.experts.skills_runtime import Sequence

    macro = SkillsRuntime(1)._holder_primitive("grasp")
    assert isinstance(macro, Sequence)
    assert [child.name for child in macro.children] == ["grasp close", "grasp home"]


def test_flip_macro_holds_the_cube_through_the_flip() -> None:
    """The flip keeps the grip through lift/roll/lower (the loaded wrist cannot
    press the cube onto the table), re-centres it, then RELEASES it onto the
    lower jaw: the tine ride keeps the layer lock open through contact and
    gives the rotator the low rim where its vertical-finger grasp is feasible."""
    from callosum.experts.skills_runtime import Sequence

    runtime = SkillsRuntime(1)
    for name in ("flip90", "flip180"):
        macro = runtime._holder_primitive(name)
        assert isinstance(macro, Sequence)
        assert [child.name for child in macro.children] == [
            "flip lift",
            "flip roll",
            "flip lower",
            "flip release",
        ]


# --- useless-action classifier ---------------------------------------------------------------


def _facts(**overrides) -> DecisionFacts:
    ones = np.ones(1, dtype=bool)
    zeros = np.zeros(1, dtype=bool)
    values = {
        "holder_grasp": ones,
        "rotator_grasp": zeros,
        "holder_near_core": ones,
        "rotator_near_layer": ones,
        "up_face_is_move_face": ones,
    }
    values.update(overrides)
    return DecisionFacts(**values)


def test_holder_useless_rules() -> None:
    release, grasp, lift, flip90 = (
        HOLDER_ID["release"],
        HOLDER_ID["grasp"],
        HOLDER_ID["lift"],
        HOLDER_ID["flip90"],
    )
    # release while not holding / lift while not holding
    assert useless_actions(Role.HOLDER, [release], _facts(holder_grasp=np.zeros(1, bool)))
    assert useless_actions(Role.HOLDER, [lift], _facts(holder_grasp=np.zeros(1, bool)))
    assert not useless_actions(Role.HOLDER, [release], _facts())
    # flip while the rotator holds the layer
    assert useless_actions(Role.HOLDER, [flip90], _facts(rotator_grasp=np.ones(1, bool)))
    assert not useless_actions(Role.HOLDER, [flip90], _facts())
    # grasp air far from the core, or re-grasp what is already held
    assert useless_actions(Role.HOLDER, [grasp], _facts(holder_near_core=np.zeros(1, bool)))
    assert useless_actions(Role.HOLDER, [grasp], _facts())
    assert not useless_actions(
        Role.HOLDER, [grasp], _facts(holder_grasp=np.zeros(1, bool))
    )
    # preparation is never classified
    for name in ("wait", "approach_body", "retreat", "lower"):
        assert not useless_actions(Role.HOLDER, [HOLDER_ID[name]], _facts())


def test_rotator_useless_rules() -> None:
    turn90, release, grasp = ROTATOR_ID["turn90"], ROTATOR_ID["release"], ROTATOR_ID["grasp"]
    both = _facts(rotator_grasp=np.ones(1, bool))
    # a turn needs the layer held AND the lock open (the holder grasping)
    assert useless_actions(Role.ROTATOR, [turn90], _facts())  # layer not held
    assert useless_actions(
        Role.ROTATOR,
        [turn90],
        _facts(rotator_grasp=np.ones(1, bool), holder_grasp=np.zeros(1, bool)),
    )  # locked: the holder does not grasp the core
    assert not useless_actions(Role.ROTATOR, [turn90], both)
    assert useless_actions(Role.ROTATOR, [release], _facts())  # releasing nothing
    assert not useless_actions(Role.ROTATOR, [release], both)
    assert useless_actions(
        Role.ROTATOR, [grasp], _facts(rotator_near_layer=np.zeros(1, bool))
    )
    assert useless_actions(Role.ROTATOR, [grasp], both)  # re-grasping what is held
    for name in ("wait", "approach_face", "retreat"):
        assert not useless_actions(Role.ROTATOR, [ROTATOR_ID[name]], _facts())


# --- layer targets and orientation ------------------------------------------------------------


def test_turn_target_maps_into_joint_limits() -> None:
    half_pi = math.pi / 2
    assert turn_target_rad(0.0, 1) == pytest.approx(half_pi)
    assert turn_target_rad(0.0, 2) == pytest.approx(math.pi)
    assert turn_target_rad(half_pi, 1) == pytest.approx(math.pi)
    # 90 + 180 = 270 deg wraps to the equivalent resting configuration 90 deg.
    assert turn_target_rad(half_pi, 2) == pytest.approx(half_pi)
    # 180 + 90 = 270 -> 90 deg (the counter-clockwise realization of a quarter turn).
    assert turn_target_rad(math.pi, 1) == pytest.approx(half_pi)
    # The current angle snaps to its nearest detent first.
    assert turn_target_rad(half_pi - 0.05, 1) == pytest.approx(math.pi)
    with pytest.raises(ValueError):
        turn_target_rad(0.0, 3)


def test_up_face_and_orientation_from_quat() -> None:
    identity = (0.0, 0.0, 0.0, 1.0)
    assert up_face_from_quat(identity) == "U"
    assert orientation_from_quat(identity) == CubeOrientation.upright()
    # Rotation about +x by 90 deg: the front face (+y) comes up (right-hand rule).
    half = math.sqrt(0.5)
    assert up_face_from_quat((half, 0.0, 0.0, half)) == "F"
    assert orientation_from_quat((half, 0.0, 0.0, half)).up == "F"
    # Rotation about +y by 90 deg: the left face (-x) comes up.
    assert up_face_from_quat((0.0, half, 0.0, half)) == "L"
    # The orientation model and the quaternion agree with _plans' own flip algebra.
    flipped = CubeOrientation.upright().flipped(Flip("R", 1))
    assert orientation_from_quat((half, 0.0, 0.0, half)) == flipped


def test_own_block_key_layout_matches_the_contract() -> None:
    from callosum.envs._skills_obs import own_block_fields

    for role, n_own in ((Role.HOLDER, N_HOLDER), (Role.ROTATOR, N_ROTATOR)):
        layout = own_block_key_layout(role)
        fields = own_block_fields(n_own)
        prefix = "holder_" if role is Role.HOLDER else "rotator_"
        assert [name for name, _ in layout] == [prefix + name for name, _ in fields]
        assert [size for _, size in layout] == [size for _, size in fields]
    total = sum(size for _, size in own_block_key_layout(Role.HOLDER))
    total += sum(size for _, size in own_block_key_layout(Role.ROTATOR))
    assert total == own_block_size(N_HOLDER) + own_block_size(N_ROTATOR)


def test_own_block_assembly_through_the_trainer_contract() -> None:
    """Marker values written per `own_block_slices` survive the trainer's slicing."""
    from callosum.training._skills_agent_obs import build_agent_obs, extract, shared_vec_size

    shared = np.zeros((1, shared_vec_size()), dtype=np.float32)
    for role, n_own, face in ((Role.HOLDER, N_HOLDER, 3), (Role.ROTATOR, N_ROTATOR, 1)):
        block = extract(shared, role)
        slices = own_block_slices(n_own)
        block[..., slices["plan_face_onehot"]][..., face] = 1.0
        block[..., slices["own_primitive_progress"]] = 0.25
        block[..., slices["cube_position"]] = (0.1, -0.2, 0.3)
    for role in (Role.HOLDER, Role.ROTATOR):
        own = build_agent_obs(shared, None, None, "none", role)[0]
        slices = own_block_slices(num_primitives(role))
        np.testing.assert_allclose(own[..., slices["cube_position"]], (0.1, -0.2, 0.3))


# --- conductor ------------------------------------------------------------------------------


def _obs(
    role: Role,
    *,
    face: str | None = "F",
    quarter: int = 1,
    quat=(0.0, 0.0, 0.0, 1.0),
    progress: float = 0.0,
    angle_rad: float = 0.0,
    gripper: float = 0.6,
    cube_z: float = 0.046,
) -> np.ndarray:
    n_own = num_primitives(role)
    vec = np.zeros(own_block_size(n_own))
    slices = own_block_slices(n_own)
    if face is not None:
        vec[slices["plan_face_onehot"]][FACES.index(face)] = 1.0
        vec[slices["plan_quarter_turns"]] = quarter / 2.0
        vec[slices["plan_moves_remaining"]] = 1.0
    vec[slices["cube_position"]] = [0.0, 0.0, cube_z]
    vec[slices["cube_orientation_quat"]] = quat
    vec[slices["active_layer_angle"]] = angle_rad / math.pi
    vec[slices["own_gripper_opening"]] = gripper
    vec[slices["own_primitive_progress"]] = progress
    vec[slices["own_primitive_onehot"]][0] = 1.0
    return vec


def _info(holder=0, rotator=0, angle_deg=0.0, plan_index=0) -> dict:
    return {
        "holder_primitive": holder,
        "rotator_primitive": rotator,
        "active_layer_angle_deg": angle_deg,
        "plan_index": plan_index,
        "success": False,
    }


def test_conductor_holder_walks_to_the_flip() -> None:
    conductor = SkillsConductor(settle_decisions=1)
    quat_upright = (0.0, 0.0, 0.0, 1.0)
    obs = _obs(Role.HOLDER, face="F", quat=quat_upright)
    info = _info()
    assert conductor.act("holder", obs, info) == HOLDER_ID["approach_body"]  # wait -> approach
    done = _obs(Role.HOLDER, face="F", quat=quat_upright, progress=1.0)
    assert conductor.act("holder", done, info) == HOLDER_ID["grasp"]  # approach settled
    assert conductor.act("holder", done, info) == HOLDER_ID["wait"]  # grasp settled -> serving
    # F is not up: the holder (still holding) flips a quarter turn about R/L.
    assert conductor.act("holder", done, info) == HOLDER_ID["flip90"]


def test_conductor_holder_recovers_a_slipped_flip() -> None:
    conductor = SkillsConductor(settle_decisions=1)
    quat_upright = (0.0, 0.0, 0.0, 1.0)
    done = _obs(Role.HOLDER, face="F", quat=quat_upright, progress=1.0)
    info = _info()
    for expected in (
        HOLDER_ID["approach_body"],
        HOLDER_ID["grasp"],
        HOLDER_ID["wait"],
        HOLDER_ID["flip90"],
    ):
        assert conductor.act("holder", done, info) == expected
    # The flip "finished" but the orientation did not change (a slipped grasp): the
    # escape hatch resynchronizes -- release, retreat, re-approach.
    assert conductor.act("holder", done, info) == HOLDER_ID["release"]
    assert conductor.act("holder", done, info) == HOLDER_ID["release"]  # held 2 decisions
    assert conductor.act("holder", done, info) == HOLDER_ID["retreat"]
    assert conductor.act("holder", done, info) == HOLDER_ID["retreat"]
    assert conductor.act("holder", done, info) == HOLDER_ID["approach_body"]


def test_conductor_holder_regrasps_after_the_flip_release() -> None:
    conductor = SkillsConductor(settle_decisions=1)
    half = math.sqrt(0.5)
    quat_f_up = (half, 0.0, 0.0, half)  # rotation about +x: F up
    done = _obs(Role.HOLDER, face="F", quat=(0.0, 0.0, 0.0, 1.0), progress=1.0)
    info = _info()
    for expected in (
        HOLDER_ID["approach_body"],
        HOLDER_ID["grasp"],
        HOLDER_ID["wait"],
        HOLDER_ID["flip90"],
    ):
        assert conductor.act("holder", done, info) == expected
    # The flip landed and the macro released the cube: re-approach and re-grasp
    # (the grasp macro then drags it back home).
    riding = _obs(Role.HOLDER, face="F", quat=quat_f_up, progress=1.0, gripper=1.0)
    assert conductor.act("holder", riding, info) == HOLDER_ID["approach_body"]
    assert conductor.act("holder", riding, info) == HOLDER_ID["grasp"]
    assert conductor.act("holder", riding, info) == HOLDER_ID["wait"]  # serving holds
    # Plan finished: let go and retreat -- the cube is already on the table.
    finished = _obs(Role.HOLDER, face=None, quat=quat_f_up, progress=1.0)
    assert conductor.act("holder", finished, info) == HOLDER_ID["release"]


def test_conductor_holder_redoes_a_failed_grasp() -> None:
    """A grasp that failed reads as open jaws (an injected failure, 1.0) or fully
    closed on air (a miss, 0.0): serving must notice through the own gripper
    opening and re-approach instead of holding air."""
    for gripper in (1.0, 0.0):
        conductor = SkillsConductor(settle_decisions=1)
        failed = _obs(
            Role.HOLDER, face="U", quat=(0.0, 0.0, 0.0, 1.0), progress=1.0, gripper=gripper
        )
        info = _info()
        for expected in (HOLDER_ID["approach_body"], HOLDER_ID["grasp"], HOLDER_ID["wait"]):
            assert conductor.act("holder", failed, info) == expected
        # U is already up (no flip needed) but the jaws hold nothing: re-approach.
        assert conductor.act("holder", failed, info) == HOLDER_ID["approach_body"]
        assert conductor.act("holder", failed, info) == HOLDER_ID["grasp"]
        # The retry clamped the core (gripper reads in-band): serving holds.
        holding = _obs(Role.HOLDER, face="U", quat=(0.0, 0.0, 0.0, 1.0), progress=1.0)
        assert conductor.act("holder", holding, info) == HOLDER_ID["wait"]
        assert conductor.act("holder", holding, info) == HOLDER_ID["wait"]


def test_conductor_rotator_turns_and_recovers() -> None:
    conductor = SkillsConductor(settle_decisions=1)
    half = math.sqrt(0.5)
    quat_f_up = (half, 0.0, 0.0, half)
    obs = _obs(Role.ROTATOR, face="F", quat=quat_f_up)
    info = _info(holder=HOLDER_ID["grasp"], angle_deg=0.0)
    assert conductor.act("rotator", obs, info) == ROTATOR_ID["wait"]  # partner streak 1
    assert conductor.act("rotator", obs, info) == ROTATOR_ID["approach_face"]  # streak 2
    done = _obs(Role.ROTATOR, face="F", quat=quat_f_up, progress=1.0)
    assert conductor.act("rotator", done, info) == ROTATOR_ID["grasp"]
    assert conductor.act("rotator", done, info) == ROTATOR_ID["turn90"]
    # Turn done with the layer at the 90 deg target: release and stand by.
    ok = _info(holder=HOLDER_ID["grasp"], angle_deg=90.0)
    assert conductor.act("rotator", done, ok) == ROTATOR_ID["release"]
    assert conductor.act("rotator", done, ok) == ROTATOR_ID["release"]  # held 2 decisions
    assert conductor.act("rotator", done, ok) == ROTATOR_ID["retreat"]
    assert conductor.act("rotator", done, ok) == ROTATOR_ID["retreat"]


def test_conductor_rotator_retries_a_slipped_turn() -> None:
    conductor = SkillsConductor(settle_decisions=1)
    half = math.sqrt(0.5)
    quat_f_up = (half, 0.0, 0.0, half)
    obs = _obs(Role.ROTATOR, face="F", quat=quat_f_up, gripper=0.6)
    done = _obs(Role.ROTATOR, face="F", quat=quat_f_up, progress=1.0, gripper=0.6)
    slipped = _info(holder=HOLDER_ID["grasp"], angle_deg=30.0)
    assert conductor.act("rotator", obs, slipped) == ROTATOR_ID["wait"]
    assert conductor.act("rotator", obs, slipped) == ROTATOR_ID["approach_face"]
    assert conductor.act("rotator", done, slipped) == ROTATOR_ID["grasp"]
    assert conductor.act("rotator", done, slipped) == ROTATOR_ID["turn90"]
    # The turn "finished" at 30 deg (a slip). A retry must INTERRUPT the completed
    # turn (one `wait` decision) before re-issuing it: re-selecting the same id
    # would only hold the finished primitive.
    assert conductor.act("rotator", done, slipped) == ROTATOR_ID["wait"]
    assert conductor.act("rotator", done, slipped) == ROTATOR_ID["turn90"]
    assert conductor.act("rotator", done, slipped) == ROTATOR_ID["wait"]
    assert conductor.act("rotator", done, slipped) == ROTATOR_ID["turn90"]
    # Retries exhausted: resync (release, retreat, approach).
    assert conductor.act("rotator", done, slipped) == ROTATOR_ID["release"]


def test_conductor_rotator_redoes_a_missed_grasp() -> None:
    """A grasp that missed the layer (jaws fully closed on air read 0.0) or never
    closed (injected failure, reads 1.0) is resynchronized, never turned."""
    for gripper in (0.0, 1.0):
        conductor = SkillsConductor(settle_decisions=1)
        half = math.sqrt(0.5)
        quat_f_up = (half, 0.0, 0.0, half)
        obs = _obs(Role.ROTATOR, face="F", quat=quat_f_up, gripper=0.6)
        missed = _obs(Role.ROTATOR, face="F", quat=quat_f_up, progress=1.0, gripper=gripper)
        info = _info(holder=HOLDER_ID["grasp"])
        assert conductor.act("rotator", obs, info) == ROTATOR_ID["wait"]
        assert conductor.act("rotator", obs, info) == ROTATOR_ID["approach_face"]
        assert conductor.act("rotator", missed, info) == ROTATOR_ID["grasp"]
        # The grasp "completed" holding nothing: release/retreat/approach, not turn.
        assert conductor.act("rotator", missed, info) == ROTATOR_ID["release"]


def test_conductor_rotator_waits_while_face_not_up() -> None:
    conductor = SkillsConductor(settle_decisions=1)
    quat_upright = (0.0, 0.0, 0.0, 1.0)
    obs = _obs(Role.ROTATOR, face="F", quat=quat_upright)
    info = _info(holder=HOLDER_ID["grasp"])
    assert conductor.act("rotator", obs, info) == ROTATOR_ID["wait"]
    assert conductor.act("rotator", obs, info) == ROTATOR_ID["wait"]  # F not up yet


def test_conductor_rotator_hold_belief_sets_and_clears() -> None:
    """The partner-hold belief: set by a grasp streak, kept across the holder's
    flips (the flip macro holds through by design), cleared by any selection
    that opens the jaws or parks them away (release/retreat/approach). While a
    flip RUNS, the rotator still waits -- the cube is a moving target."""
    conductor = SkillsConductor(settle_decisions=1)
    upright = (0.0, 0.0, 0.0, 1.0)
    obs = _obs(Role.ROTATOR, face="F", quat=upright)  # F not up yet
    info = _info(holder=HOLDER_ID["grasp"])
    assert conductor.act("rotator", obs, info) == ROTATOR_ID["wait"]  # streak builds
    assert conductor.act("rotator", obs, info) == ROTATOR_ID["wait"]  # belief set, gate shut
    half = math.sqrt(0.5)
    f_up = _obs(Role.ROTATOR, face="F", quat=(half, 0.0, 0.0, half))
    # A running flip blocks the approach even with the face up and the belief set.
    assert conductor.act("rotator", f_up, _info(holder=HOLDER_ID["flip90"])) == ROTATOR_ID["wait"]
    # The holder settles into serving: approach.
    assert conductor.act("rotator", f_up, _info(holder=HOLDER_ID["wait"])) == ROTATOR_ID[
        "approach_face"
    ]
    # The holder releases: the belief clears, and re-opening needs a fresh streak.
    reset = SkillsConductor(settle_decisions=1)
    assert reset.act("rotator", obs, info) == ROTATOR_ID["wait"]
    assert reset.act("rotator", obs, info) == ROTATOR_ID["wait"]
    assert reset.act("rotator", f_up, _info(holder=HOLDER_ID["release"])) == ROTATOR_ID["wait"]
    assert reset.act("rotator", f_up, _info(holder=HOLDER_ID["retreat"])) == ROTATOR_ID["wait"]
    assert reset.act("rotator", f_up, info) == ROTATOR_ID["wait"]  # streak 1
    assert reset.act("rotator", f_up, info) == ROTATOR_ID["approach_face"]


def test_conductor_rotator_trusts_a_standing_hold_between_moves() -> None:
    """After a completed turn the holder idles in `wait` while still holding: the
    next move on the same up face must be approachable without a fresh streak."""
    conductor = SkillsConductor(settle_decisions=1)
    half = math.sqrt(0.5)
    quat_f_up = (half, 0.0, 0.0, half)
    obs = _obs(Role.ROTATOR, face="F", quat=quat_f_up)
    done = _obs(Role.ROTATOR, face="F", quat=quat_f_up, progress=1.0)
    info = _info(holder=HOLDER_ID["grasp"], angle_deg=0.0)
    assert conductor.act("rotator", obs, info) == ROTATOR_ID["wait"]
    assert conductor.act("rotator", obs, info) == ROTATOR_ID["approach_face"]
    assert conductor.act("rotator", done, info) == ROTATOR_ID["grasp"]
    assert conductor.act("rotator", done, info) == ROTATOR_ID["turn90"]
    ok = _info(holder=HOLDER_ID["grasp"], angle_deg=90.0)
    assert conductor.act("rotator", done, ok) == ROTATOR_ID["release"]
    assert conductor.act("rotator", done, ok) == ROTATOR_ID["release"]
    assert conductor.act("rotator", done, ok) == ROTATOR_ID["retreat"]
    assert conductor.act("rotator", done, ok) == ROTATOR_ID["retreat"]
    assert conductor.act("rotator", done, ok) == ROTATOR_ID["retreat"]  # enters wait
    idle = _info(holder=HOLDER_ID["wait"])  # the holder serves: still holding
    assert conductor.act("rotator", done, idle) == ROTATOR_ID["approach_face"]


def test_conductor_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="holder or rotator"):
        SkillsConductor().act("arm_c", np.zeros(own_block_size(N_HOLDER)), _info())


# --- plan/orientation integration with the runtime helpers ------------------------------------


def test_needed_flip_logic_end_to_end() -> None:
    """flips_to_bring_up -> up_face_from_quat round trip for every face."""
    half = math.sqrt(0.5)
    quarter_quats = {
        Flip("R", 1): (half, 0.0, 0.0, half),
        Flip("L", 1): (-half, 0.0, 0.0, half),
        Flip("F", 1): (0.0, half, 0.0, half),
        Flip("B", 1): (0.0, -half, 0.0, half),
    }
    for flip, quat in quarter_quats.items():
        orientation = CubeOrientation.upright().flipped(flip)
        assert up_face_from_quat(quat) == orientation.up
        assert orientation.up != "U"  # a real flip happened
    # A 180 deg flip about any horizontal axis brings D up.
    pi_quat = (1.0, 0.0, 0.0, 0.0)  # 180 deg about +x
    assert up_face_from_quat(pi_quat) == "D"
    assert CubeOrientation.upright().flipped(Flip("R", 2)).up == "D"


def test_plan_move_face_matches_onehot_order() -> None:
    assert FACES == ("U", "D", "L", "R", "F", "B")  # the obs one-hot order is frozen
    move = PlanMove("B", 2)
    assert FACES.index(move.face) == 5


def test_env_module_imports_resolve() -> None:
    """Every `from X import name` in face_turn_skills.py must exist in X.

    face_turn_skills.py imports mani_skill at module level, so macOS never executes
    its imports in tests -- this static check catches wrong-module imports (the
    review blocker: `own_block_key_layout` imported from `_skills_obs` instead of
    `skills_runtime`) before the cloud run does.
    """
    import ast
    import importlib
    from pathlib import Path

    import callosum.envs

    env_path = Path(callosum.envs.__file__).parent / "face_turn_skills.py"
    tree = ast.parse(env_path.read_text())
    checked = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.module):
            continue
        try:
            module = importlib.import_module(node.module)
        except ImportError:
            # Server-only deps (mani_skill & co) are not importable on macOS.
            continue
        for alias in node.names:
            assert hasattr(module, alias.name), (
                f"face_turn_skills imports {alias.name!r} from {node.module}, "
                f"but the module has no such attribute"
            )
            checked += 1
    assert checked > 10, "expected to statically resolve a meaningful number of imports"


def test_orientation_from_quat_degenerate_45deg() -> None:
    """A mid-flip 45-deg quaternion must still yield a valid permutation."""
    import numpy as np

    from callosum.experts.skills_runtime import orientation_from_quat

    half = np.cos(np.pi / 4)
    quat = (half, half, 0.0, 0.0)  # 90 deg about +x == two 45-deg assignments meet
    ori = orientation_from_quat(quat)
    assert sorted(ori.slots) == ["B", "D", "F", "L", "R", "U"]
    # Exactly on the diagonal between two faces the greedy tie-break is stable.
    diag = (1.0, np.sin(np.pi / 8), 0.0, 0.0)
    ori2 = orientation_from_quat(diag)
    assert sorted(ori2.slots) == ["B", "D", "F", "L", "R", "U"]
