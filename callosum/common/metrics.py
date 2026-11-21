"""Coordination metrics of the skill-level cube assembly (openspec tasks 4.1-4.4): pure logic.

This module is the metric layer behind `python -m callosum.experiments.arm.skills_eval`: it aggregates finished
episodes (`EpisodeRecord`), never environments -- the [cloud] harness records episodes,
this layer turns them into the numbers the coordination-metrics spec demands (success
rate, time to success, useless-action rate, partner-primitive prediction accuracy) plus
the three report structures: run summaries, cross-play matrices and recovery summaries.

Pure stdlib + numpy on purpose (the same macOS-testability rule as
`callosum.experiments.arm.skills`): no mani_skill, no torch, deterministic outputs, JSON-friendly
types. The standard deviation is the population one (ddof=0), so a single successful
episode has std 0.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from callosum.experiments.arm.skills import Role


@dataclass(frozen=True)
class EpisodeRecord:
    """One finished episode, as recorded by the eval harness (not by the env).

    Attributes:
        success: the plan was executed within tolerances.
        time_to_success: decisions until success (None exactly when the episode failed).
        decisions: decisions taken in the episode (both hands decide at every decision).
        sim_steps_to_success: simulation steps until success (None if failed or unknown).
        useless_holder / useless_rotator: pointless choices summed over the episode per
            hand (turn of a locked layer, premature release, needless re-grasp, ...),
            the env's per-decision `useless_action_*` flags of `_skills_obs`.
        predicted_partner / true_partner: the recorded predictions of the partner's
            current primitive and their ground truth, one entry per recorded decision
            (empty when nothing was recorded -- modes without a wired predictor).
    """

    success: bool
    time_to_success: int | None
    decisions: int
    sim_steps_to_success: int | None = None
    useless_holder: int = 0
    useless_rotator: int = 0
    predicted_partner: tuple[int, ...] = ()
    true_partner: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.success != (self.time_to_success is not None):
            raise ValueError(
                "time_to_success must be set exactly when the episode succeeded "
                f"(success={self.success}, time_to_success={self.time_to_success})"
            )
        if self.decisions < 0:
            raise ValueError(f"decisions must be >= 0, got {self.decisions}")
        if self.time_to_success is not None and self.time_to_success < 0:
            raise ValueError(f"time_to_success must be >= 0, got {self.time_to_success}")
        if self.sim_steps_to_success is not None and self.sim_steps_to_success < 0:
            raise ValueError(f"sim_steps_to_success must be >= 0, got {self.sim_steps_to_success}")
        if self.useless_holder < 0 or self.useless_rotator < 0:
            raise ValueError(
                f"useless counts must be >= 0, got ({self.useless_holder}, {self.useless_rotator})"
            )
        if len(self.predicted_partner) != len(self.true_partner):
            raise ValueError(
                f"predicted/true partner sequences must have equal length, got "
                f"{len(self.predicted_partner)} != {len(self.true_partner)}"
            )

    def useless(self, role: Role) -> int:
        """The useless-action count of one hand."""
        return self.useless_holder if role is Role.HOLDER else self.useless_rotator

    def predictions_recorded(self) -> int:
        """How many predictions the episode recorded."""
        return len(self.predicted_partner)

    def prediction_hits(self) -> int:
        """How many recorded predictions matched the partner's true primitive."""
        return sum(
            1
            for predicted, true in zip(self.predicted_partner, self.true_partner)
            if predicted == true
        )


# --- per-run aggregators -----------------------------------------------------------------


def success_rate(records: Sequence[EpisodeRecord]) -> float:
    """Share of successful episodes (0.0 when there are none)."""
    if not records:
        return 0.0
    return sum(1 for record in records if record.success) / len(records)


def time_to_success_stats(records: Sequence[EpisodeRecord]) -> tuple[float, float] | None:
    """`(mean, population std)` of the decision count until success.

    Only successful episodes have a time to success; None when no episode succeeded
    (including the empty input).
    """
    times = [record.time_to_success for record in records if record.success]
    if not times:
        return None
    return float(np.mean(times)), float(np.std(times))


def useless_rate(records: Sequence[EpisodeRecord], role: Role) -> float:
    """Useless actions per decision of one hand, over all episodes.

    The denominator pools the decisions of every episode, so long episodes weigh
    accordingly; 0.0 when no decisions were taken at all.
    """
    useless = sum(record.useless(role) for record in records)
    decisions = sum(record.decisions for record in records)
    return useless / decisions if decisions else 0.0


def partner_prediction_accuracy(records: Sequence[EpisodeRecord]) -> float | None:
    """Recorded predictions that matched the partner's true primitive.

    None when nothing was recorded (the run had no wired predictor): the accuracy of
    zero predictions is undefined, not zero.
    """
    hits = sum(record.prediction_hits() for record in records)
    total = sum(record.predictions_recorded() for record in records)
    return hits / total if total else None


@dataclass(frozen=True)
class RunSummary:
    """Every metric of one run (one checkpoint pair under one condition).

    `time_to_success_mean/std` are over successful episodes only (None without any);
    the useless counts are episode sums, the rates divide them by the pooled decisions.
    """

    episodes: int
    successes: int
    success_rate: float
    decisions: int
    useless_holder: int
    useless_rotator: int
    useless_rate_holder: float
    useless_rate_rotator: float
    time_to_success_mean: float | None
    time_to_success_std: float | None
    partner_prediction_accuracy: float | None


def summarize(records: Sequence[EpisodeRecord]) -> RunSummary:
    """All per-run aggregators of `records` in one JSON-ready dataclass."""
    time = time_to_success_stats(records)
    return RunSummary(
        episodes=len(records),
        successes=sum(1 for record in records if record.success),
        success_rate=success_rate(records),
        decisions=sum(record.decisions for record in records),
        useless_holder=sum(record.useless_holder for record in records),
        useless_rotator=sum(record.useless_rotator for record in records),
        useless_rate_holder=useless_rate(records, Role.HOLDER),
        useless_rate_rotator=useless_rate(records, Role.ROTATOR),
        time_to_success_mean=None if time is None else time[0],
        time_to_success_std=None if time is None else time[1],
        partner_prediction_accuracy=partner_prediction_accuracy(records),
    )


def summary_dict(records: Sequence[EpisodeRecord]) -> dict[str, float | int | None]:
    """`summarize(records)` as a plain dict (what `--out results.json` stores per pair)."""
    return asdict(summarize(records))


# --- cross-play --------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossPlayMatrix:
    """Success of every `(holder, rotator)` checkpoint pair, as a matrix.

    Rows are holder checkpoints, columns rotator checkpoints, both label-sorted, so the
    matrix is deterministic regardless of dict order. The diagonal (holder and rotator
    from the same checkpoint) is self-play; off-diagonal cells are zero-shot cross-play
    with an unfamiliar partner (the thesis's main claim: partner prediction should raise
    the cross-play mean towards the self-play one).
    """

    holders: tuple[str, ...]
    rotators: tuple[str, ...]
    cells: dict[tuple[str, str], float]

    @classmethod
    def from_results(cls, results: Mapping[tuple[str, str], float]) -> "CrossPlayMatrix":
        """Build the matrix from `{(holder_label, rotator_label): success_rate}`."""
        for (holder, rotator), success in results.items():
            if not 0.0 <= success <= 1.0:
                raise ValueError(
                    f"success rate of ({holder!r}, {rotator!r}) must be within [0, 1], "
                    f"got {success}"
                )
        holders = tuple(sorted({holder for holder, _ in results}))
        rotators = tuple(sorted({rotator for _, rotator in results}))
        return cls(holders=holders, rotators=rotators, cells=dict(results))

    def cell(self, holder: str, rotator: str) -> float | None:
        """One matrix cell, or None where the pair was not evaluated."""
        return self.cells.get((holder, rotator))

    def cells_where(self, diagonal: bool) -> list[float]:
        """The success rates of the self-play (diagonal) or cross-play (other) cells."""
        return [value for (h, r), value in self.cells.items() if (h == r) == diagonal]

    @property
    def self_play_mean(self) -> float | None:
        """Mean success over the diagonal cells (None without any)."""
        cells = self.cells_where(diagonal=True)
        return float(np.mean(cells)) if cells else None

    @property
    def cross_play_mean(self) -> float | None:
        """Mean success over the off-diagonal cells (None without any)."""
        cells = self.cells_where(diagonal=False)
        return float(np.mean(cells)) if cells else None

    def to_dict(self) -> dict:
        """JSON-ready form: labels, the full matrix (None for missing cells), the means."""
        return {
            "holders": list(self.holders),
            "rotators": list(self.rotators),
            "matrix": [
                [self.cell(holder, rotator) for rotator in self.rotators] for holder in self.holders
            ],
            "self_play_mean": self.self_play_mean,
            "cross_play_mean": self.cross_play_mean,
        }


def crossplay_matrix(results: Mapping[tuple[str, str], float]) -> CrossPlayMatrix:
    """The cross-play matrix of `{(holder_label, rotator_label): success_rate}`."""
    return CrossPlayMatrix.from_results(results)


# --- recovery ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoverySummary:
    """Failure-injected vs clean episodes of the same checkpoint pair.

    `success_delta` / `time_delta` are `with - without` (a positive time delta means the
    injections cost extra decisions even when the plan still completed). Every episode
    of the with-run received the configured injections, so its success rate IS the share
    of episodes that recovered (`recovered_rate`, the spec's "доля восстановившихся").
    """

    episodes_with: int
    episodes_without: int
    success_with: float
    success_without: float
    success_delta: float
    recovered_rate: float
    time_with: float | None
    time_without: float | None
    time_delta: float | None


def recovery_summary(
    with_failures: Sequence[EpisodeRecord], without_failures: Sequence[EpisodeRecord]
) -> RecoverySummary:
    """Compare a failure-injected run against a clean run of the same pair."""
    time_with = time_to_success_stats(with_failures)
    time_without = time_to_success_stats(without_failures)
    success_with = success_rate(with_failures)
    success_without = success_rate(without_failures)
    return RecoverySummary(
        episodes_with=len(with_failures),
        episodes_without=len(without_failures),
        success_with=success_with,
        success_without=success_without,
        success_delta=success_with - success_without,
        recovered_rate=success_with,
        time_with=None if time_with is None else time_with[0],
        time_without=None if time_without is None else time_without[0],
        time_delta=(
            None if time_with is None or time_without is None else time_with[0] - time_without[0]
        ),
    )
