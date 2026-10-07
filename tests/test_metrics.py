"""Episode statistics, scalar logging and the progress line of the IPPO trainer (no torch)."""

import numpy as np

from callosum.training._metrics import (
    EpisodeStats,
    MetricLogger,
    eval_diag_summary,
    progress_line,
)


def test_episode_stats_average_only_finished_envs() -> None:
    stats = EpisodeStats()
    assert stats.add({}) == 0 and stats.means() == {}
    infos = {
        "_final_info": np.array([True, False, True]),
        "final_info": {
            "episode": {
                "return": np.array([1.0, 100.0, 3.0]),
                "success_once": np.array([True, True, False]),
            }
        },
    }
    assert stats.add(infos) == 2
    assert stats.means() == {"return": 2.0, "success_once": 0.5}
    assert stats.num_episodes == 2
    assert stats.add(infos) == 2 and stats.num_episodes == 4  # accumulates over steps


def test_logger_and_progress_line() -> None:
    class Writer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, float, int]] = []

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.calls.append((tag, value, step))

    writer = Writer()
    logger = MetricLogger(writer)
    logger.log("train/return", np.float32(0.5), 10)
    logger.log("losses/agent_a/entropy", float("nan"), 10)  # kept locally, not written
    logger.log_many({"success_once": 0.25}, 10, prefix="eval/")
    assert writer.calls == [("train/return", 0.5, 10), ("eval/success_once", 0.25, 10)]
    logger.log("charts/learning_rate", 1.5e-4, 10)
    line = progress_line(logger, 3, 10, 7680, 1234.4, train_episodes=4)
    assert "iter 3/10" in line and "step 7680" in line and "sps 1234" in line
    assert "ret 0.5" in line and "(n=4)" in line and "eval_succ 0.250" in line
    assert "ent a/b nan/-" in line and "lr 1.50e-04" in line


def test_episode_stats_exclude_leaves_out_poisoned_envs() -> None:
    stats = EpisodeStats()
    infos = {
        "_final_info": np.array([True, True, False]),
        "final_info": {"episode": {"return": np.array([1.0, np.nan, 5.0])}},
    }
    assert stats.add(infos, exclude=np.array([False, True, False])) == 1
    assert stats.means() == {"return": 1.0}


def test_progress_line_optional_fields() -> None:
    logger = MetricLogger()
    plain = progress_line(logger, 1, 10, 100, 5.0, 0)
    assert "hold a/b" not in plain and "NONFINITE" not in plain
    logger.log("diag/holder_grasp", 0.123, 100)
    logger.log("diag/rotator_grasp", 0.05, 100)
    line = progress_line(logger, 1, 10, 100, 5.0, 0, nonfinite=(2, 5))
    assert "hold a/b 0.12/0.05" in line and "NONFINITE 2 (total 5)" in line
    assert "NONFINITE" not in progress_line(logger, 1, 10, 100, 5.0, 0, nonfinite=(0, 0))


def test_eval_diag_summary() -> None:
    assert eval_diag_summary({}, {}) == ""  # envs without diagnostics
    means = {
        "holder_grasp": 0.5,
        "rotator_grasp": 0.25,
        "both_grasp": 0.125,
        "ever_holder_grasp": 1.0,
        "face_angle": 3.14159265 / 2,
    }
    line = eval_diag_summary(means, {"face_angle": 3.14159265})
    assert "grasp hold 0.50 rot 0.25 both 0.12" in line
    assert "ever hold 1.00 rot - both -" in line
    assert "face 90.0 deg mean, 180.0 max" in line
