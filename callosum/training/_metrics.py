"""Metric collection: TensorBoard, stdout, and an end-of-run summary.

Pure Python (no torch, no mani_skill import), so it is unit-testable on
macOS/CI like callosum.training._ppo_core and _agent_obs -- see
docs/implementation-plan.md section 0.

Why this exists: the trainer used to write metrics ONLY to TensorBoard, so a
run's log showed nothing but `SPS=...` and every question about it needed a
separate script reading the event files. Runs here are minutes long and watched
through `tail -f` over a JupyterHub terminal, so the numbers have to be in the
log itself.

`log()` mirrors each scalar three ways: to TensorBoard, into the current
iteration's line, and into a history used for the closing summary.
"""

from collections.abc import Iterable

# Shown on the progress bar and in each iteration's line, in this order, when
# present. Everything else still reaches TensorBoard and the final summary.
HEADLINE_TAGS: tuple[str, ...] = (
    "train/return",
    "train/success_once",
    "eval/success_once",
    "losses/jepa_loss",
)

_SHORT = {
    "train/return": "return",
    "train/success_once": "succ",
    "eval/success_once": "eval_succ",
    "losses/jepa_loss": "jepa",
}


def _as_float(value) -> float:
    """Coerce a scalar, 0-d tensor or numpy scalar to float."""
    item = getattr(value, "item", None)
    return float(item()) if callable(item) else float(value)


class MetricLogger:
    """Collects scalars for TensorBoard, the console, and a final summary.

    Args:
        writer: anything with `add_scalar(tag, value, step)` -- a
            `SummaryWriter`, or None to collect without writing (tests).
        headline: tags surfaced on the progress bar / iteration line.
    """

    def __init__(self, writer=None, headline: Iterable[str] = HEADLINE_TAGS):
        self._writer = writer
        self._headline = tuple(headline)
        self._history: dict[str, list[float]] = {}

    def log(self, tag: str, value, step: int) -> None:
        v = _as_float(value)
        if self._writer is not None:
            self._writer.add_scalar(tag, v, step)
        self._history.setdefault(tag, []).append(v)

    def headline(self) -> dict[str, str]:
        """Short name -> formatted value, for a progress-bar postfix.

        Headline tags are NOT cleared between iterations: episode metrics only
        appear on iterations where an episode finished, and blanking them would
        make the bar flicker between a number and nothing.
        """
        out = {}
        for tag in self._headline:
            if tag in self._history:
                out[_SHORT.get(tag, tag)] = f"{self._history[tag][-1]:.4g}"
        return out

    def iteration_line(self, iteration: int, total: int, global_step: int, sps: float) -> str:
        """One compact line per iteration, for logs where the bar is disabled."""
        parts = [f"iter {iteration}/{total}", f"step {global_step}", f"sps {sps:.0f}"]
        parts += [f"{k} {v}" for k, v in self.headline().items()]
        return " | ".join(parts)

    def summary(self) -> str:
        """Closing table: first/last/min/max/n for every tag ever logged.

        Replaces re-reading the event files with a separate script, which is
        what every past run needed.
        """
        if not self._history:
            return "no metrics recorded"
        width = max(len(t) for t in self._history)
        lines = [
            "",
            "=" * (width + 58),
            f"{'metric'.ljust(width)}   first     last      min       max       n",
        ]
        for tag in sorted(self._history):
            v = self._history[tag]
            lines.append(
                f"{tag.ljust(width)} {v[0]:+9.4g} {v[-1]:+9.4g} "
                f"{min(v):+9.4g} {max(v):+9.4g} {len(v):6d}"
            )
        lines.append("=" * (width + 58))
        return "\n".join(lines)
