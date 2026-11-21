"""Coordination metrics of the skill-level cube assembly on synthetic episode records.

Pure-logic tests (tasks 4.1-4.4): known `EpisodeRecord` sets must give exact metric
values, the cross-play matrix must keep its shape/diagonal/means on fake 2x2 results,
recovery must contrast injected vs clean runs, and the empty/all-success edges must not
explode. No mani_skill, no torch.
"""

import math

import pytest

from callosum.envs._skills import Role
from callosum.common.metrics import (
    CrossPlayMatrix,
    EpisodeRecord,
    crossplay_matrix,
    partner_prediction_accuracy,
    recovery_summary,
    success_rate,
    summarize,
    summary_dict,
    time_to_success_stats,
    useless_rate,
)


def rec(
    success: bool = True,
    time: int | None = 10,
    decisions: int = 10,
    steps: int | None = 100,
    uh: int = 0,
    ur: int = 0,
    pred: tuple[int, ...] = (),
    true: tuple[int, ...] = (),
) -> EpisodeRecord:
    """A record with the success/time invariant already applied."""
    return EpisodeRecord(
        success=success,
        time_to_success=time if success else None,
        decisions=decisions,
        sim_steps_to_success=steps if success else None,
        useless_holder=uh,
        useless_rotator=ur,
        predicted_partner=pred,
        true_partner=true,
    )


# --- record invariants ---------------------------------------------------------------------


def test_record_requires_time_exactly_on_success() -> None:
    with pytest.raises(ValueError, match="time_to_success"):
        EpisodeRecord(success=True, time_to_success=None, decisions=5)
    with pytest.raises(ValueError, match="time_to_success"):
        EpisodeRecord(success=False, time_to_success=5, decisions=5)


def test_record_rejects_negative_and_ragged_predictions() -> None:
    with pytest.raises(ValueError, match="decisions"):
        EpisodeRecord(success=False, time_to_success=None, decisions=-1)
    with pytest.raises(ValueError, match="useless"):
        EpisodeRecord(success=False, time_to_success=None, decisions=1, useless_holder=-1)
    with pytest.raises(ValueError, match="equal length"):
        EpisodeRecord(
            success=False, time_to_success=None, decisions=1, predicted_partner=(0,)
        )


# --- per-run aggregators -------------------------------------------------------------------


def test_success_rate_empty_and_mixed() -> None:
    assert success_rate([]) == 0.0
    records = [rec(), rec(time=5), rec(success=False)]
    assert success_rate(records) == pytest.approx(2 / 3)


def test_time_to_success_stats_only_over_successes() -> None:
    records = [
        rec(time=10, decisions=10),
        rec(time=20, decisions=20),
        rec(time=30, decisions=30),
        rec(success=False, decisions=40),
    ]
    mean, std = time_to_success_stats(records)
    assert mean == pytest.approx(20.0)
    assert std == pytest.approx(math.sqrt(200 / 3))  # population std (ddof=0)


def test_time_to_success_stats_none_without_successes() -> None:
    assert time_to_success_stats([]) is None
    assert time_to_success_stats([rec(success=False), rec(success=False)]) is None


def test_time_to_success_stats_all_success_zero_std() -> None:
    mean, std = time_to_success_stats([rec(time=7, decisions=7)])
    assert mean == 7.0 and std == 0.0


def test_useless_rate_pools_decisions_per_role() -> None:
    records = [
        rec(success=False, decisions=10, uh=2, ur=0),
        rec(success=False, decisions=30, uh=0, ur=6),
    ]
    assert useless_rate(records, Role.HOLDER) == pytest.approx(2 / 40)
    assert useless_rate(records, Role.ROTATOR) == pytest.approx(6 / 40)
    assert useless_rate([], Role.HOLDER) == 0.0
    assert useless_rate(
        [EpisodeRecord(success=False, time_to_success=None, decisions=0)], Role.HOLDER
    ) == 0.0


def test_partner_prediction_accuracy_over_recorded_predictions() -> None:
    records = [
        rec(pred=(0, 1, 2), true=(0, 1, 0)),  # 2 of 3
        rec(pred=(1,), true=(0,)),  # 0 of 1
        rec(success=False),  # nothing recorded
    ]
    assert partner_prediction_accuracy(records) == pytest.approx(2 / 4)
    assert partner_prediction_accuracy([rec(), rec(success=False)]) is None
    assert partner_prediction_accuracy([]) is None


def test_summarize_exact_fields() -> None:
    records = [
        rec(time=10, decisions=20, uh=1, pred=(0, 1, 2), true=(0, 1, 0)),
        rec(time=20, decisions=30, ur=3, pred=(1,), true=(0,)),
        rec(success=False, decisions=40, uh=3, ur=1),
    ]
    summary = summarize(records)
    assert summary.episodes == 3
    assert summary.successes == 2
    assert summary.success_rate == pytest.approx(2 / 3)
    assert summary.decisions == 90
    assert summary.useless_holder == 4 and summary.useless_rotator == 4
    assert summary.useless_rate_holder == pytest.approx(4 / 90)
    assert summary.useless_rate_rotator == pytest.approx(4 / 90)
    assert summary.time_to_success_mean == pytest.approx(15.0)
    assert summary.time_to_success_std == pytest.approx(5.0)
    assert summary.partner_prediction_accuracy == pytest.approx(0.5)
    as_dict = summary_dict(records)
    assert as_dict["episodes"] == 3 and as_dict["success_rate"] == pytest.approx(2 / 3)
    assert set(as_dict) == set(summary.__dict__)


def test_summarize_empty_run() -> None:
    summary = summarize([])
    assert summary.episodes == 0 and summary.success_rate == 0.0
    assert summary.time_to_success_mean is None
    assert summary.useless_rate_holder == 0.0 and summary.useless_rate_rotator == 0.0
    assert summary.partner_prediction_accuracy is None


# --- cross-play ----------------------------------------------------------------------------


def test_crossplay_matrix_full_2x2() -> None:
    matrix = crossplay_matrix(
        {("s1", "s1"): 0.9, ("s1", "s2"): 0.4, ("s2", "s1"): 0.6, ("s2", "s2"): 0.8}
    )
    assert matrix.holders == ("s1", "s2")
    assert matrix.rotators == ("s1", "s2")
    assert matrix.cell("s1", "s1") == pytest.approx(0.9)
    assert matrix.cell("s2", "s1") == pytest.approx(0.6)
    assert matrix.self_play_mean == pytest.approx(0.85)
    assert matrix.cross_play_mean == pytest.approx(0.5)
    assert matrix.cells_where(diagonal=True) == [0.9, 0.8]
    assert matrix.cells_where(diagonal=False) == [0.4, 0.6]
    assert matrix.to_dict() == {
        "holders": ["s1", "s2"],
        "rotators": ["s1", "s2"],
        "matrix": [[0.9, 0.4], [0.6, 0.8]],
        "self_play_mean": pytest.approx(0.85),
        "cross_play_mean": pytest.approx(0.5),
    }


def test_crossplay_matrix_sorted_labels_and_missing_cells() -> None:
    matrix = crossplay_matrix({("b", "b"): 0.8, ("b", "a"): 0.4, ("a", "a"): 0.9})
    assert matrix.holders == ("a", "b")  # label-sorted, insertion order irrelevant
    assert matrix.rotators == ("a", "b")
    assert matrix.cell("a", "b") is None  # pair never evaluated
    assert matrix.self_play_mean == pytest.approx(0.85)
    assert matrix.cross_play_mean == pytest.approx(0.4)  # only the present off-diagonal
    assert matrix.to_dict()["matrix"] == [[0.9, None], [0.4, 0.8]]


def test_crossplay_matrix_edges_and_validation() -> None:
    empty = crossplay_matrix({})
    assert empty.holders == () and empty.self_play_mean is None and empty.cross_play_mean is None
    only_self = crossplay_matrix({("a", "a"): 1.0})
    assert only_self.cross_play_mean is None
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        CrossPlayMatrix.from_results({("a", "b"): 1.5})


# --- recovery ------------------------------------------------------------------------------


def test_recovery_contrast_between_injected_and_clean_runs() -> None:
    with_failures = [
        rec(time=30, decisions=30),
        rec(time=40, decisions=40),
        rec(success=False, decisions=50),
    ]
    without_failures = [
        rec(time=10, decisions=10),
        rec(time=20, decisions=20),
        rec(time=30, decisions=30),
        rec(success=False, decisions=40),
    ]
    recovery = recovery_summary(with_failures, without_failures)
    assert recovery.episodes_with == 3 and recovery.episodes_without == 4
    assert recovery.success_with == pytest.approx(2 / 3)
    assert recovery.success_without == pytest.approx(3 / 4)
    assert recovery.success_delta == pytest.approx(2 / 3 - 3 / 4)
    assert recovery.time_with == pytest.approx(35.0)
    assert recovery.time_without == pytest.approx(20.0)
    assert recovery.time_delta == pytest.approx(15.0)
    assert recovery.recovered_rate == recovery.success_with


def test_recovery_edges() -> None:
    nothing = recovery_summary([], [])
    assert nothing.success_delta == 0.0 and nothing.time_delta is None
    failing = recovery_summary([rec(success=False)], [rec(time=10, decisions=10)])
    assert failing.success_delta == -1.0
    assert failing.time_with is None and failing.time_delta is None
