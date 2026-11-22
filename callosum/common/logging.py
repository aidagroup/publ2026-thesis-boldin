"""Episode-metric collection, TensorBoard scalars and the per-iteration stdout line.

Pure Python (no mani_skill, no torch import; it works on whatever tensors/arrays it is handed), so it is testable on
macOS/CI. Runs on the lab server are detached and watched with `tail -f`, so every number of
interest must also reach stdout, not only the TensorBoard event file.
"""

import math
from collections import defaultdict
from collections.abc import Mapping


def _as_float(value) -> float:
    """A float from a Python number, numpy scalar or 0-d tensor."""
    item = getattr(value, "item", None)
    return float(item()) if callable(item) else float(value)


class EpisodeStats:
    """Running mean of ManiSkillVectorEnv's per-episode metrics over finished episodes.

    `ManiSkillVectorEnv(record_metrics=True)` puts the metrics of the episodes that just ended
    into `infos["final_info"]["episode"]` (tensors of shape `(num_envs,)`: `return`,
    `episode_len`, `reward`, `success_once`, `success_at_end` only when terminations are
    ignored, ...) and flags those envs in `infos["_final_info"]`. `add` accumulates only the
    flagged entries, so the mean is over finished episodes, not over all envs.
    """

    def __init__(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self.num_episodes = 0

    def add(self, infos: Mapping, exclude=None) -> int:
        """Accumulate the episodes that ended in this step; returns how many were counted.

        Args:
            infos: the vector env's info dict.
            exclude: optional bool array/tensor `(num_envs,)`; envs that are True are left out
                (episodes poisoned by a non-finite simulation state, whose return is NaN).
        """
        if "final_info" not in infos:
            return 0
        mask = infos["_final_info"]
        if exclude is not None:
            mask = mask & ~exclude
        episode = infos["final_info"]["episode"]
        finished = int(mask.sum())
        for key, value in episode.items():
            selected = value[mask]
            self._sums[key] += float(selected.sum())
            self._counts[key] += finished
        self.num_episodes += finished
        return finished

    def means(self) -> dict[str, float]:
        """Mean of every metric over the episodes added so far (empty if there were none)."""
        return {k: self._sums[k] / self._counts[k] for k in self._sums if self._counts[k] > 0}


class MetricLogger:
    """Writes scalars to TensorBoard and remembers the latest value of each tag.

    Args:
        writer: anything with `add_scalar(tag, value, step)` (a `SummaryWriter`) or `None`.
    """

    def __init__(self, writer=None) -> None:
        self._writer = writer
        self.latest: dict[str, float] = {}

    def log(self, tag: str, value, step: int) -> None:
        """Log one scalar (non-finite values are kept in the dict but not sent to TensorBoard)."""
        number = _as_float(value)
        self.latest[tag] = number
        if self._writer is not None and math.isfinite(number):
            self._writer.add_scalar(tag, number, step)

    def log_many(self, scalars: Mapping[str, object], step: int, prefix: str = "") -> None:
        """Log every `prefix + key` scalar of a dict."""
        for key, value in scalars.items():
            self.log(prefix + key, value, step)

    def get(self, tag: str) -> float | None:
        """The most recent value of `tag`, or `None` if it was never logged."""
        return self.latest.get(tag)


def format_value(value: float | None, spec: str = ".3g") -> str:
    """`value` formatted with `spec`, or `-` if it is missing."""
    return "-" if value is None else format(value, spec)


def eval_diag_summary(means: Mapping[str, float], maxes: Mapping[str, float]) -> str:
    """Compact stdout summary of an evaluation's FaceTurn diagnostics ("" if the env has none).

    `means` / `maxes` are `DiagStats.take()` results: grasp rates over all env-steps (holder,
    rotator, both at once), the share of episodes with a grasp at least once, and the face angle
    (mean and max, converted from radians to degrees).
    """
    parts = []
    grasps = [
        ("hold", means.get("holder_grasp")),
        ("rot", means.get("rotator_grasp")),
        ("both", means.get("both_grasp")),
    ]
    if any(v is not None for _, v in grasps):
        parts.append("grasp " + " ".join(f"{k} {format_value(v, '.2f')}" for k, v in grasps))
    ever = [
        ("hold", means.get("ever_holder_grasp")),
        ("rot", means.get("ever_rotator_grasp")),
        ("both", means.get("ever_both_grasp")),
    ]
    if any(v is not None for _, v in ever):
        parts.append("ever " + " ".join(f"{k} {format_value(v, '.2f')}" for k, v in ever))
    if "face_angle" in means:
        top = maxes.get("face_angle", means["face_angle"])
        parts.append(
            f"face {math.degrees(means['face_angle']):.1f} deg mean, {math.degrees(top):.1f} max"
        )
    return " | ".join(parts)


def progress_line(
    logger: MetricLogger,
    iteration: int,
    num_iterations: int,
    global_step: int,
    sps: float,
    train_episodes: int,
    nonfinite: tuple[int, int] = (0, 0),
) -> str:
    """One compact stdout line per iteration (the latest value of each headline metric).

    `nonfinite` is `(envs this iteration, envs in total)` of the non-finite-env guard; it is only
    shown when the total is positive. The `hold a/b` field (holder / rotator grasp rate over the
    rollout) only appears for envs that publish those diagnostics.
    """
    entropy = [
        format_value(logger.get(f"losses/{r}/entropy"), ".2f") for r in ("agent_a", "agent_b")
    ]
    kl = [format_value(logger.get(f"losses/{r}/approx_kl"), ".3f") for r in ("agent_a", "agent_b")]
    fields = [
        f"iter {iteration}/{num_iterations}",
        f"step {global_step}",
        f"sps {sps:.0f}",
        f"ret {format_value(logger.get('train/return'))}",
        f"succ {format_value(logger.get('train/success_once'), '.3f')} (n={train_episodes})",
        f"ent a/b {entropy[0]}/{entropy[1]}",
        f"kl a/b {kl[0]}/{kl[1]}",
        f"lr {format_value(logger.get('charts/learning_rate'), '.2e')}",
        f"eval_succ {format_value(logger.get('eval/success_once'), '.3f')}",
        f"eval_ret {format_value(logger.get('eval/return'))}",
    ]
    hold = [logger.get("diag/holder_grasp"), logger.get("diag/rotator_grasp")]
    if all(v is not None for v in hold):
        fields.insert(5, f"hold a/b {hold[0]:.2f}/{hold[1]:.2f}")
    if nonfinite[1] > 0:
        fields.append(f"NONFINITE {nonfinite[0]} (total {nonfinite[1]})")
    return " | ".join(fields)
