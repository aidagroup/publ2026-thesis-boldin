"""Summarise the two-robot coop-suite held-out evaluations.

Reads `results_coop/<run>_base<base>.json` files produced by
`python -m callosum.experiments.coop_tasks.eval` and prints a markdown table:
per task/mode, the mean success rate over training seeds and eval seed bases
(with min–max), the seed-to-seed spread and the `oracle - none` / `predicted - none`
gaps. Also writes a JSON summary next to the results.

Usage:
    python scripts/analysis/summarize_coop.py --results-dir results_coop --out results_coop/summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

MODES = ("none", "oracle", "pred")
TASKS = ("cartpole", "pendulum", "toss")


def _parse_name(path: Path) -> tuple[str, str, int, int] | None:
    """`coop_<task>_<mode>_s<seed>_base<base>.json` -> (task, mode, seed, base)."""
    stem = path.stem
    if not stem.startswith("coop_"):
        return None
    try:
        head, base = stem.rsplit("_base", 1)
        task, mode, seed_part = head[len("coop_") :].rsplit("_s", 2)
        return task, mode, int(seed_part), int(base)
    except ValueError:
        return None


def collect(results_dir: Path) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(results_dir.glob("coop_*_base*.json")):
        parsed = _parse_name(path)
        if parsed is None:
            continue
        task, mode, seed, base = parsed
        payload = json.loads(path.read_text())
        grouped[(task, mode)].append(
            {
                "seed": seed,
                "base": base,
                "success_rate": float(payload["success_rate"]),
                "steps_mean": float(payload.get("steps_mean", 0.0)),
                "diagnostic": float(payload.get("mean_diagnostic", 0.0)),
            }
        )
    return grouped


def summarize(grouped: dict[tuple[str, str], list[dict]]) -> dict:
    summary: dict = {"tasks": {}}
    for task in TASKS:
        task_summary: dict = {}
        for mode in MODES:
            rows = grouped.get((task, mode), [])
            if not rows:
                continue
            # per training seed: mean over the three held-out eval bases
            per_seed: dict[int, float] = defaultdict(float)
            counts: dict[int, int] = defaultdict(int)
            for row in rows:
                per_seed[row["seed"]] += row["success_rate"]
                counts[row["seed"]] += 1
            seed_means = {seed: per_seed[seed] / counts[seed] for seed in sorted(per_seed)}
            values = list(seed_means.values())
            task_summary[mode] = {
                "per_seed": seed_means,
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "n_evals": len(rows),
            }
        if "oracle" in task_summary and "none" in task_summary:
            task_summary["gap_oracle_none"] = (
                task_summary["oracle"]["mean"] - task_summary["none"]["mean"]
            )
        if "pred" in task_summary and "none" in task_summary:
            task_summary["gap_pred_none"] = (
                task_summary["pred"]["mean"] - task_summary["none"]["mean"]
            )
        summary["tasks"][task] = task_summary
    return summary


def render_markdown(summary: dict) -> str:
    lines = [
        "| task | mode | success (mean) | per-seed | min–max | n eval |",
        "|---|---|---|---|---|---|",
    ]
    for task, task_summary in summary["tasks"].items():
        for mode in MODES:
            if mode not in task_summary:
                continue
            row = task_summary[mode]
            per_seed = " / ".join(f"{value:.3f}" for value in row["per_seed"].values())
            lines.append(
                f"| {task} | {mode} | **{row['mean']:.3f}** | {per_seed} | "
                f"{row['min']:.3f}–{row['max']:.3f} | {row['n_evals']} |"
            )
    gaps = []
    for task, task_summary in summary["tasks"].items():
        if "gap_oracle_none" in task_summary:
            gaps.append(f"{task}: oracle−none = {task_summary['gap_oracle_none']:+.3f}")
        if "gap_pred_none" in task_summary:
            gaps.append(f"{task}: predicted−none = {task_summary['gap_pred_none']:+.3f}")
    if gaps:
        lines += ["", "**Gaps:** " + "; ".join(gaps)]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results_coop")
    parser.add_argument("--out", default="results_coop/summary.json")
    args = parser.parse_args()
    grouped = collect(Path(args.results_dir))
    if not grouped:
        raise SystemExit(f"no coop eval JSON files in {args.results_dir}")
    summary = summarize(grouped)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(render_markdown(summary))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
