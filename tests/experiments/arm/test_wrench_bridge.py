"""Unit tests of the intent->skills bridge pure logic (phases, schedule, timing metrics).

Only `callosum.experiments.arm.wrench_bridge`'s pure paths are exercised here: phase
classification, phase segmentation, the phase->primitive mapping, the arm-decision
schedule and the timing comparison. The rollout itself needs `mujoco_warp`+CUDA and runs
on the pod (`scripts/experiments/arm/bridge_wrench_to_skills.py`); the module imports
torch optionally, so these tests run on macOS/CI without the `train` extra.
"""

import math
from pathlib import Path

import pytest

from callosum.experiments.arm.wrench_bridge import (
    ARM_DECISION_SECONDS,
    HOLDER_FORCE_ON_N,
    HOLDER_TORQUE_ON_NM,
    PHASES,
    ROTATOR_TAU_ON_NM,
    ArmDecision,
    PhaseSegment,
    PhaseSignals,
    ReducedStep,
    build_primitive_schedule,
    classify_arm_phase,
    classify_phase,
    compare_phase_timing,
    intended_primitives,
    schedule_json,
    segment_phases,
)


def _step(
    t: int,
    *,
    phase: str = "",
    holder_force: float = 0.0,
    holder_torque: float = 0.0,
    rotator_tau_z: float = 0.0,
    rise_deg: float = 0.0,
    angle_deg: float | None = None,
) -> ReducedStep:
    """A synthetic reduced step with the given physical signals."""
    force = (holder_force, 0.0, 0.0)
    torque = (0.0, 0.0, holder_torque)
    rotator = (0.0, 0.0, rotator_tau_z)
    return ReducedStep(
        t=t,
        time_s=t * 0.04,
        layer_angle_rad=math.radians(angle_deg if angle_deg is not None else t),
        layer_ang_vel=0.0,
        core_drift_pos_m=0.0,
        core_drift_rot_deg=0.0,
        holder_force=force,
        holder_torque=torque,
        rotator_force=(0.0, 0.0, 0.0),
        rotator_torque=rotator,
        angle_rise_deg=rise_deg,
        phase=phase,
    )


# --- phase classification -----------------------------------------------------------------


def test_turn_needs_torque_and_rising_layer():
    turning = PhaseSignals(
        holder_force_norm_n=0.0,
        holder_torque_norm_nm=0.0,
        rotator_tau_z_nm=ROTATOR_TAU_ON_NM,
        angle_rise_deg=1.0,
    )
    assert classify_phase(turning) == "turn"


def test_turn_needs_rise_even_with_full_torque():
    held = PhaseSignals(
        holder_force_norm_n=0.5,
        holder_torque_norm_nm=5e-3,
        rotator_tau_z_nm=2e-3,
        angle_rise_deg=0.0,
    )
    assert classify_phase(held) == "hold"


def test_hold_from_force_and_from_torque_alone():
    force_only = PhaseSignals(1.0, 0.0, 0.0, 0.0)
    torque_only = PhaseSignals(0.0, 1e-3, 0.0, 0.0)
    assert classify_phase(force_only) == "hold"
    assert classify_phase(torque_only) == "hold"


def test_idle_below_every_threshold():
    idle = PhaseSignals(
        holder_force_norm_n=HOLDER_FORCE_ON_N / 2,
        holder_torque_norm_nm=HOLDER_TORQUE_ON_NM / 2,
        rotator_tau_z_nm=ROTATOR_TAU_ON_NM / 2,
        angle_rise_deg=10.0,
    )
    assert classify_phase(idle) == "idle"


def test_turn_takes_precedence_over_a_holding_holder():
    both = PhaseSignals(1.0, 5e-3, 2e-3, 1.0)
    assert classify_phase(both) == "turn"


def test_classify_thresholds_are_overridable():
    weak = PhaseSignals(0.02, 0.0, 0.0, 0.0)
    assert classify_phase(weak) == "idle"
    assert classify_phase(weak, holder_force_on=0.01) == "hold"


# --- segmentation --------------------------------------------------------------------------


def test_segment_merges_single_step_flicker():
    steps = [
        _step(0, holder_force=1.0),
        _step(1, holder_force=1.0),
        _step(2),  # one-step dropout
        _step(3, holder_force=1.0),
        _step(4, holder_force=1.0),
    ]
    segments = segment_phases(steps, control_dt_s=0.04)
    assert [segment.phase for segment in segments] == ["hold"]
    assert segments[0].t_start == 0
    assert segments[0].t_end == 4
    assert segments[0].duration_s == pytest.approx(5 * 0.04)


def test_segment_hold_then_turn_keeps_order_and_durations():
    steps = [
        _step(0, holder_force=1.0),
        _step(1, holder_force=1.0),
        _step(2, holder_force=1.0, rotator_tau_z=1e-3, rise_deg=2.0),
        _step(3, holder_force=1.0, rotator_tau_z=1e-3, rise_deg=2.0),
        _step(4, holder_force=1.0, rotator_tau_z=1e-3, rise_deg=2.0),
    ]
    segments = segment_phases(steps, control_dt_s=0.04)
    assert [segment.phase for segment in segments] == ["hold", "turn"]
    assert segments[0].steps == 2
    assert segments[1].steps == 3
    assert segments[1].duration_s == pytest.approx(3 * 0.04)


def test_success_relabels_the_trailing_non_turn_run_as_release():
    steps = [_step(0, holder_force=1.0), _step(1, holder_force=1.0), _step(2), _step(3)]
    segments = segment_phases(steps, control_dt_s=0.04, success=True)
    assert segments[-1].phase == "release"
    assert segments[0].phase == "hold"


def test_failure_keeps_the_trailing_run_idle():
    steps = [_step(0), _step(1)]
    segments = segment_phases(steps, control_dt_s=0.04, success=False)
    assert [segment.phase for segment in segments] == ["idle"]


def test_segment_classifies_unlabelled_steps():
    steps = [_step(0), _step(1, holder_force=1.0), _step(2, holder_force=1.0)]
    segments = segment_phases(steps, control_dt_s=0.01)
    assert [segment.phase for segment in segments] == ["idle", "hold"]


# --- phase -> primitive mapping -------------------------------------------------------------


@pytest.mark.parametrize("phase", PHASES)
def test_every_phase_maps_to_a_registered_primitive(phase):
    from callosum.experiments.arm.skills import (
        HOLDER_PRIMITIVES,
        ROTATOR_PRIMITIVES,
    )

    holder_entry, rotator_entry = intended_primitives(phase, entry=True)
    holder_held, rotator_held = intended_primitives(phase, entry=False)
    holder_names = {primitive.name for primitive in HOLDER_PRIMITIVES}
    rotator_names = {primitive.name for primitive in ROTATOR_PRIMITIVES}
    assert {holder_entry, holder_held} <= holder_names
    assert {rotator_entry, rotator_held} <= rotator_names


def test_intent_sequence_for_hold_and_turn():
    assert intended_primitives("hold", entry=True) == ("grasp", "grasp")
    assert intended_primitives("hold", entry=False) == ("wait", "wait")
    assert intended_primitives("turn", entry=True) == ("wait", "turn90")
    assert intended_primitives("release", entry=False) == ("retreat", "retreat")


def test_unknown_phase_is_rejected():
    with pytest.raises(ValueError):
        intended_primitives("nope")


# --- schedule ------------------------------------------------------------------------------


def _segments() -> list[PhaseSegment]:
    return [
        PhaseSegment(phase="hold", t_start=0, t_end=4, duration_s=5 * 0.04),
        PhaseSegment(phase="turn", t_start=5, t_end=24, duration_s=20 * 0.04),
    ]


def test_dt_mode_collapses_the_reduced_episode_to_a_few_decisions():
    schedule = build_primitive_schedule(_segments(), timing_mode="dt", engage_decisions=1)
    reduced = [segment for segment in schedule if segment.source == "reduced"]
    # 0.2 s hold -> 1 decision, 0.8 s turn -> round(1.6) = 2 decisions at 0.5 s each.
    assert [segment.decisions for segment in reduced] == [1, 2]
    assert [segment.phase for segment in reduced] == ["hold", "turn"]


def test_share_mode_keeps_the_reduced_duration_shares():
    schedule = build_primitive_schedule(_segments(), timing_mode="share", total_decisions=25)
    reduced = [segment for segment in schedule if segment.source == "reduced"]
    assert [segment.phase for segment in reduced] == ["hold", "turn"]
    # 1:4 duration share of 25 decisions -> 5 hold, 20 turn.
    assert [segment.decisions for segment in reduced] == [5, 20]


def test_schedule_is_bracketed_by_arm_native_engage_and_release():
    schedule = build_primitive_schedule(_segments(), engage_decisions=6, release_decisions=2)
    assert schedule[0].phase == "idle"
    assert schedule[0].source == "arm-native"
    assert schedule[-1].phase == "release"
    assert schedule[-1].source == "arm-native"
    assert [segment.phase for segment in schedule] == ["idle", "hold", "turn", "release"]


def test_reduced_idle_runs_are_absorbed_into_the_engage_segment():
    segments = [
        PhaseSegment(phase="idle", t_start=0, t_end=3, duration_s=4 * 0.04),
        PhaseSegment(phase="hold", t_start=4, t_end=8, duration_s=5 * 0.04),
    ]
    schedule = build_primitive_schedule(segments, timing_mode="dt")
    sources = [segment.source for segment in schedule]
    assert sources.count("arm-native") == 2  # engage + release only
    assert [segment.phase for segment in schedule] == ["idle", "hold", "release"]


def test_share_mode_gives_every_reduced_phase_at_least_one_decision():
    segments = [
        PhaseSegment(phase="hold", t_start=0, t_end=0, duration_s=0.04),
        PhaseSegment(phase="turn", t_start=1, t_end=2, duration_s=0.08),
    ]
    schedule = build_primitive_schedule(segments, timing_mode="share", total_decisions=2)
    reduced = [segment for segment in schedule if segment.source == "reduced"]
    assert all(segment.decisions >= 1 for segment in reduced)


def test_bad_timing_mode_is_rejected():
    with pytest.raises(ValueError):
        build_primitive_schedule(_segments(), timing_mode="magic")


def test_schedule_json_totals_and_constants():
    schedule = build_primitive_schedule(_segments(), engage_decisions=4, release_decisions=2)
    payload = schedule_json(schedule)
    assert payload["total_decisions"] == sum(segment.decisions for segment in schedule)
    assert payload["arm_decision_s"] == pytest.approx(0.5)
    entry = payload["segments"][1]
    assert entry["holder_entry"] == "grasp"
    assert entry["rotator_entry"] == "grasp"


# --- arm decisions and the timing comparison -----------------------------------------------


def _arm_decisions(phases: list[str]) -> list[ArmDecision]:
    primitives = {
        "idle": ("approach_body", "wait"),
        "hold": ("wait", "grasp"),
        "turn": ("wait", "turn90"),
        "release": ("release", "retreat"),
    }
    decisions = []
    for index, phase in enumerate(phases):
        holder, rotator = primitives[phase]
        decisions.append(
            ArmDecision(
                decision=index,
                holder=holder,
                rotator=rotator,
                layer_angle_deg=float(index),
                angle_rise_deg=2.0 if phase == "turn" else 0.0,
                holder_holding=phase in ("hold", "turn"),
                rotator_grasp=phase in ("turn", "release"),
                plan_index=1 if phase == "turn" else 0,
                success=False,
                phase=phase,
            )
        )
    return decisions


def test_arm_phase_classification():
    base = ArmDecision(0, "approach_body", "wait", 0.0, 0.0, False, False, 0, False)
    assert classify_arm_phase(base) == "idle"
    hold = ArmDecision(0, "wait", "grasp", 0.0, 0.0, True, False, 0, False)
    assert classify_arm_phase(hold) == "hold"
    turning = ArmDecision(0, "wait", "turn90", 30.0, 3.0, True, True, 0, False)
    assert classify_arm_phase(turning) == "turn"
    stalled = ArmDecision(0, "wait", "turn90", 30.0, 0.0, True, True, 0, False)
    assert classify_arm_phase(stalled) == "hold"
    release = ArmDecision(0, "release", "release", 90.0, 0.0, False, False, 1, True)
    assert classify_arm_phase(release) == "release"


def test_identical_phase_sequences_have_zero_error():
    segments = [
        PhaseSegment(phase="hold", t_start=0, t_end=3, duration_s=4 * 0.04),
        PhaseSegment(phase="turn", t_start=4, t_end=7, duration_s=4 * 0.04),
    ]
    arms = _arm_decisions(["idle", "hold", "hold", "hold", "turn", "turn", "turn", "release"])
    timing = compare_phase_timing(
        segments,
        arms,
        reduced_control_dt_s=0.04,
        reduced_success=True,
        arm_success=True,
    )
    assert timing["decision_over_control_step"] == pytest.approx(ARM_DECISION_SECONDS / 0.04)
    # The reduced run is 4 hold + 4 turn steps; the arm run is 8 decisions with the same
    # turn onset position (decision 4), so the normalized onsets agree exactly.
    assert timing["turn_onset_fraction_reduced"] == pytest.approx(4 / 7)
    assert timing["turn_onset_fraction_arm"] == pytest.approx(4 / 7)
    assert timing["turn_onset_offset"] == pytest.approx(0.0)
    assert timing["phase_share_l1_error"] > 0.0  # the arm's release/idle tail is extra
    assert timing["reduced_success_fraction"] == 1.0
    assert timing["arm_success_fraction"] == 1.0


def test_missing_turn_in_arm_trace_reports_none_onset():
    segments = [PhaseSegment(phase="turn", t_start=0, t_end=1, duration_s=2 * 0.04)]
    arms = _arm_decisions(["idle", "hold"])
    timing = compare_phase_timing(
        segments,
        arms,
        reduced_control_dt_s=0.04,
        reduced_success=True,
        arm_success=False,
    )
    assert timing["turn_onset_fraction_arm"] is None
    assert timing["turn_onset_offset"] is None
    assert timing["arm_success"] is False
    assert timing["arm_success_fraction"] == 0.0


def test_failed_episode_has_zero_success_fraction():
    timing = compare_phase_timing(
        [PhaseSegment(phase="hold", t_start=0, t_end=0, duration_s=0.04)],
        _arm_decisions(["idle"]),
        reduced_control_dt_s=0.04,
        reduced_success=False,
        arm_success=False,
    )
    assert timing["reduced_success_fraction"] == 0.0
    assert timing["reduced_success"] is False


# --- the bridge script's executor helpers ---------------------------------------------------

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/experiments/arm/bridge_wrench_to_skills.py"


def _load_bridge_script():
    """Import `bridge_wrench_to_skills.py` without the simulator (all sim imports are lazy)."""
    import importlib.util
    import sys

    name = "bridge_wrench_to_skills_under_test"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclasses resolve `cls.__module__` through sys.modules
    spec.loader.exec_module(module)
    return module


def _facts(script, **overrides):
    """A `_Facts` with the executor's defaults, overridable per test."""
    values = {
        "holder_primitive": "wait",
        "rotator_primitive": "wait",
        "holder_progress": 0.0,
        "rotator_progress": 0.0,
        "holder_opening": 1.0,
        "rotator_grasp": False,
        "angle_deg": 0.0,
        "plan_index": 0,
        "success": False,
    }
    values.update(overrides)
    return script._Facts(**values)


def test_script_executor_engages_then_holds():
    script = _load_bridge_script()
    memory: dict = {}
    # Fresh episode: approach with the holder, wait with the rotator.
    holder, rotator, done = script._engage_choice(
        memory, _facts(script, holder_primitive="approach_body", holder_progress=0.5)
    )
    assert (holder, rotator, done) == ("approach_body", "wait", False)
    # Approach finished -> grasp.
    holder, rotator, done = script._engage_choice(
        memory, _facts(script, holder_primitive="approach_body", holder_progress=1.0)
    )
    assert (holder, rotator, done) == ("grasp", "wait", False)
    # Jaw closed on the core -> the rotator starts approaching.
    holder, rotator, done = script._engage_choice(
        memory, _facts(script, holder_primitive="grasp", holder_progress=1.0, holder_opening=0.7)
    )
    assert (holder, rotator, done) == ("wait", "approach_face", False)
    # Rotator approach finished -> grasp.
    holder, rotator, done = script._engage_choice(
        memory,
        _facts(
            script,
            holder_primitive="wait",
            holder_progress=1.0,
            holder_opening=0.7,
            rotator_primitive="approach_face",
            rotator_progress=1.0,
        ),
    )
    assert (holder, rotator, done) == ("wait", "grasp", False)
    # Grasp fact established -> engagement done.
    holder, rotator, done = script._engage_choice(
        memory,
        _facts(
            script,
            holder_primitive="wait",
            holder_progress=1.0,
            holder_opening=0.7,
            rotator_primitive="grasp",
            rotator_progress=1.0,
            rotator_grasp=True,
        ),
    )
    assert (holder, rotator, done) == ("wait", "wait", True)


def test_script_executor_hold_and_turn_choices():
    script = _load_bridge_script()
    # An unengaged hand set keeps driving the engagement FSM (never asserts a hold).
    holder, rotator, done = script._hold_choice({}, _facts(script))
    assert done is False
    assert (holder, rotator) == ("approach_body", "wait")
    assert script._hold_choice({}, _facts(script, holder_grasp_env=True, rotator_grasp=True)) == (
        "wait",
        "wait",
        True,
    )
    assert script._turn_choice(_facts(script), 90.0, 4.0) == ("grasp", "turn90", False)
    assert script._turn_choice(_facts(script, holder_opening=0.7), 90.0, 4.0) == (
        "wait",
        "turn90",
        False,
    )
    assert script._turn_choice(_facts(script, plan_index=1), 90.0, 4.0)[2] is True
    assert script._turn_choice(_facts(script, angle_deg=88.0), 90.0, 4.0)[2] is True


def test_script_executor_releases_then_retreats():
    script = _load_bridge_script()
    memory: dict = {}
    first = script._release_choice(memory, _facts(script))
    assert first[:2] == ("release", "release")
    second = script._release_choice(memory, _facts(script))
    assert second[:2] == ("retreat", "retreat")
    final = script._release_choice(
        memory, _facts(script, holder_primitive="retreat", holder_progress=1.0)
    )
    assert final[2] is True


def test_derived_shares_ignore_the_arm_native_brackets():
    segments = [
        PhaseSegment(phase="hold", t_start=0, t_end=1, duration_s=2 * 0.04),
        PhaseSegment(phase="turn", t_start=2, t_end=9, duration_s=8 * 0.04),
    ]
    arms = _arm_decisions(["idle"] * 4 + ["hold"] * 2 + ["turn"] * 8 + ["release"] * 3)
    timing = compare_phase_timing(
        segments,
        arms,
        reduced_control_dt_s=0.04,
        reduced_success=True,
        arm_success=True,
    )
    assert timing["derived_phase_shares_reduced"] == {"hold": 0.2, "turn": 0.8}
    assert timing["derived_phase_shares_arm"] == {"hold": 0.2, "turn": 0.8}
    assert timing["derived_phase_share_l1_error"] == pytest.approx(0.0)
    assert timing["derived_turn_onset_fraction_reduced"] == pytest.approx(2 / 9)
    assert timing["derived_turn_onset_fraction_arm"] == pytest.approx(2 / 9)
    # The all-phase shares also count the arm-native idle/release brackets.
    assert timing["phase_share_l1_error"] > 0.0
