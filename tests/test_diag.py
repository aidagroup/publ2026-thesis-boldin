"""Averaging of the env's `diag_*` info keys (pure torch)."""

import pytest

torch = pytest.importorskip("torch")  # CI installs only the dev extra; run with `make dev`

from callosum.envs._diag import diag_name
from callosum.training._diag import DiagStats


def test_diag_name_only_matches_the_prefix() -> None:
    assert diag_name("diag_face_angle") == "face_angle"
    assert diag_name("face_angle") is None and diag_name("diag_") is None


def _infos(grasp, angle, **extra) -> dict:
    return {
        "success": torch.zeros(len(grasp), dtype=torch.bool),  # not a diagnostic: ignored
        "diag_holder_grasp": torch.tensor(grasp),
        "diag_face_angle": torch.tensor(angle),
        **extra,
    }


def test_means_maxes_and_no_diagnostics() -> None:
    stats = DiagStats(2, "cpu")
    stats.add({"success": torch.zeros(2, dtype=torch.bool)})
    assert stats.take() == ({}, {})
    stats.add(_infos([True, False], [0.5, 1.0]))
    stats.add(_infos([True, True], [1.5, 0.0]))
    means, maxes = stats.take()
    assert means["holder_grasp"] == pytest.approx(0.75)
    assert means["face_angle"] == pytest.approx(0.75)
    assert maxes == {"face_angle": pytest.approx(1.5)}  # flags have no max
    assert stats.take() == ({}, {})  # the sums were cleared


def test_final_info_replaces_the_reset_values_of_ended_episodes() -> None:
    stats = DiagStats(2, "cpu")
    infos = _infos([False, False], [0.0, 0.0])  # the reset state of env 0, a plain step of env 1
    infos["_final_info"] = torch.tensor([True, False])
    infos["final_info"] = _infos([True, True], [2.0, 9.0])  # env 1's entry is not used
    stats.add(infos)
    means, _ = stats.take()
    assert means["holder_grasp"] == pytest.approx(0.5)  # env 0 final True, env 1 current False
    assert means["face_angle"] == pytest.approx(1.0)


def test_ever_rate_is_per_finished_episode_and_spans_takes() -> None:
    stats = DiagStats(2, "cpu")
    stats.add(_infos([True, False], [0.0, 0.0]))
    stats.take()  # an iteration boundary in the middle of the episodes
    infos = _infos([False, False], [0.0, 0.0])
    infos["_final_info"] = torch.tensor([True, True])
    infos["final_info"] = _infos([False, False], [0.0, 0.0])
    stats.add(infos)
    means, _ = stats.take()
    assert means["ever_holder_grasp"] == pytest.approx(0.5)  # env 0 grasped once, env 1 never
    # A new episode starts with cleared flags.
    stats.add(infos)
    assert stats.take()[0]["ever_holder_grasp"] == 0.0


def test_invalid_envs_are_left_out() -> None:
    stats = DiagStats(2, "cpu")
    valid = torch.tensor([True, False])
    stats.add(_infos([True, True], [1.0, float("nan")]), valid)
    means, maxes = stats.take()
    assert means["holder_grasp"] == 1.0 and means["face_angle"] == 1.0
    assert maxes["face_angle"] == 1.0
