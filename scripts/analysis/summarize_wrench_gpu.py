"""Summarise a CubeWrench-v0 GPU experiment grid (logs + `results_gpu/*.json`).

Reads the per-run evaluation JSONs written by `scripts/experiments/cube_wrench/run_wrench_gpu_grid.sh eval` (128 greedy
episodes x eval seeds 0/1/2) and the training logs of the runs, and prints the tables the
ablation needs:

* per run: greedy success / steps / drift (mean over the eval seeds) and the eval curve;
* the oracle-vs-none gap per preset at every eval point (asymptotic gap = mean of the last
  three evals, max gap, and the eval update at which each mode crosses 0.5 / 0.8 / 0.9);
* the predicted sweep (aux_coef x history_len): its success at updates 50-200 against the
  `none` baseline, so the best early-budget predictor config is visible.

The numbers are written to `<results>/summary.json` and the tables are printed to stdout.
Pure stdlib (runs on the Mac too, given the files) and no simulator import.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

_GRID_RE = re.compile(r"^wrench_gpu_(default|harsh)_(none|oracle|predicted)_s(\d+)$")
_SWEEP_RE = re.compile(r"^wrench_gpu_sweep_h(\d+)_aux(\d+)_s(\d+)$")
_EVAL_LINE = re.compile(
    r"^eval @ update (\d+): \d+ episodes \| success_rate=([0-9.]+) return=(-?[0-9.]+) "
    r"angle_deg=(-?[0-9.]+) len=([0-9.]+)"
)
_UPDATE_LINE = re.compile(
    r"^update (\d+)/\d+ \| step \d+ \| (\d+) sps \| \d+ eps \| return (-?[0-9.]+) "
    r"success ([0-9.]+) len ([0-9.]+)"
)
_TARGETS = (0.5, 0.8, 0.9)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI (results directory)."""
    parser = argparse.ArgumentParser(description="Summarise a CubeWrench-v0 GPU grid.")
    parser.add_argument("--results", default="results_gpu", help="directory of eval JSONs")
    parser.add_argument("--runs", default="runs", help="directory of the run logs/checkpoints")
    parser.add_argument(
        "--out", default=None, help="summary JSON (default: <results>/summary.json)"
    )
    return parser.parse_args(argv)


def run_identity(name: str) -> dict | None:
    """`{"kind", "preset", "mode", "seed"}` of a grid/sweep run name, or None."""
    match = _GRID_RE.match(name)
    if match:
        preset, mode, seed = match.groups()
        return {"kind": "grid", "preset": preset, "mode": mode, "seed": int(seed)}
    match = _SWEEP_RE.match(name)
    if match:
        history, aux_tag, seed = match.groups()
        aux = 0.03 if aux_tag == "003" else 0.1
        return {
            "kind": "sweep",
            "preset": "default",
            "mode": "predicted",
            "seed": int(seed),
            "history_len": int(history),
            "aux_coef": aux,
        }
    return None


def parse_log(path: Path) -> dict:
    """The eval and per-update train curves of one run log."""
    eval_curve: list[dict] = []
    train_curve: list[dict] = []
    sps: list[int] = []
    if not path.is_file():
        return {"eval_curve": eval_curve, "train_curve": train_curve, "sps_mean": None}
    for line in path.read_text(errors="replace").splitlines():
        match = _EVAL_LINE.match(line)
        if match:
            update, success, ret, angle, length = match.groups()
            eval_curve.append(
                {
                    "update": int(update),
                    "success_rate": float(success),
                    "return": float(ret),
                    "angle_deg": float(angle),
                    "len": float(length),
                }
            )
            continue
        match = _UPDATE_LINE.match(line)
        if match:
            update, rate, ret, success, length = match.groups()
            train_curve.append(
                {
                    "update": int(update),
                    "sps": int(rate),
                    "return": float(ret),
                    "success_rate": float(success),
                    "len": float(length),
                }
            )
            sps.append(int(rate))
    return {
        "eval_curve": eval_curve,
        "train_curve": train_curve,
        "sps_mean": int(statistics.fmean(sps)) if sps else None,
    }


def thresholds(points: list[dict], key: str = "success_rate") -> dict[str, int | None]:
    """First update at which `success_rate` reaches each of 0.5/0.8/0.9."""
    result: dict[str, int | None] = {}
    for target in _TARGETS:
        hit = next((point["update"] for point in points if point[key] >= target), None)
        result[f"{target:.1f}"] = hit
    return result


def eval_summary(results: Path, name: str) -> dict:
    """Aggregate the 128-episode evals of one run over all eval-seed files found."""
    rows = []
    for path in sorted(results.glob(f"{name}_evalseed*.json")):
        rows.append(json.loads(path.read_text()))
    if not rows:
        return {}
    success = [float(row["success_rate"]) for row in rows]
    return {
        "eval_seeds": [int(row["seed"]) for row in rows],
        "episodes": int(rows[0]["episodes"]),
        "randomization": rows[0].get("randomization"),
        "success_mean": statistics.fmean(success),
        "success_std": statistics.pstdev(success) if len(success) > 1 else 0.0,
        "success_min": min(success),
        "success_max": max(success),
        "steps_mean": statistics.fmean(float(row["steps_mean"]) for row in rows),
        "time_to_success_mean": statistics.fmean(
            float(row["steps_to_success_mean"])
            for row in rows
            if row["steps_to_success_mean"] is not None
        ),
        "drift_pos_mm": statistics.fmean(1e3 * float(row["mean_drift_pos_m"]) for row in rows),
        "drift_rot_deg": statistics.fmean(
            57.29577951308232 * float(row["mean_drift_rot_rad"]) for row in rows
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Build the summary JSON and print the tables."""
    args = parse_args(argv)
    results = Path(args.results)
    runs_dir = Path(args.runs)
    out = Path(args.out) if args.out else results / "summary.json"

    names = set()
    for path in results.glob("*_evalseed*.json"):
        names.add(path.name.split("_evalseed")[0])
    for path in runs_dir.glob("wrench_gpu_*/latest.pt"):
        names.add(path.parent.name)

    identities = {}
    for name in sorted(names):
        identity = run_identity(name)
        if identity is None:
            print(f"skip unknown run name {name}")
            continue
        identities[name] = identity

    summary: dict[str, dict] = {}
    for name, identity in identities.items():
        entry = dict(identity)
        entry["checkpoint"] = str(runs_dir / name / "latest.pt")
        entry["log"] = str(runs_dir / f"{name}.log")
        log = parse_log(Path(entry["log"]))
        entry["eval_curve"] = log["eval_curve"]
        entry["train_curve"] = log["train_curve"]
        entry["sps_mean"] = log["sps_mean"]
        entry["eval_thresholds"] = thresholds(log["eval_curve"])
        entry["train_thresholds"] = thresholds(log["train_curve"])
        entry["final"] = eval_summary(results, name)
        summary[name] = entry

    # --- grid tables ------------------------------------------------------------------
    grid = {name: entry for name, entry in summary.items() if entry["kind"] == "grid"}
    print("\n## 128-episode greedy eval (mean over eval seeds 0/1/2)\n")
    print("| preset | mode | seed | success | steps | drift pos (mm) | drift rot (deg) | ckpt |")
    print("|---|---|---|---|---|---|---|---|")
    for preset in ("default", "harsh"):
        for mode in ("none", "oracle", "predicted"):
            rows = sorted(
                (name, entry)
                for name, entry in grid.items()
                if entry["preset"] == preset and entry["mode"] == mode
            )
            for name, entry in rows:
                final = entry["final"]
                if not final:
                    print(f"| {preset} | {mode} | {entry['seed']} | - | - | - | - | {name} |")
                    continue
                print(
                    f"| {preset} | {mode} | {entry['seed']} | "
                    f"{final['success_mean']:.3f} ± {final['success_std']:.3f} | "
                    f"{final['steps_mean']:.1f} | {final['drift_pos_mm']:.1f} | "
                    f"{final['drift_rot_deg']:.1f} | {name} |"
                )

    print("\n## Eval curves (greedy, 32 episodes per point; success by update)\n")
    print("| preset | mode | seed | u50 | u100 | u150 | u200 | u250 | u300 | eval 0.5/0.8/0.9 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for preset in ("default", "harsh"):
        for mode in ("none", "oracle", "predicted"):
            for name, entry in sorted(
                (name, entry)
                for name, entry in grid.items()
                if entry["preset"] == preset and entry["mode"] == mode
            ):
                curve = {point["update"]: point["success_rate"] for point in entry["eval_curve"]}
                cells = " | ".join(
                    f"{curve[u]:.2f}" if u in curve else "-" for u in (50, 100, 150, 200, 250, 300)
                )
                crossed = entry["eval_thresholds"]
                marks = "/".join(
                    "-" if crossed[t] is None else str(crossed[t]) for t in ("0.5", "0.8", "0.9")
                )
                print(f"| {preset} | {mode} | {entry['seed']} | {cells} | {marks} |")

    # --- oracle vs none gap -----------------------------------------------------------
    print("\n## oracle - none gap (greedy eval success)\n")
    print("| preset | seed | u50 | u100 | u150 | u200 | u250 | u300 | max gap | asymptotic |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    gaps: dict[str, dict] = {}
    for preset in ("default", "harsh"):
        seeds = sorted(
            {
                entry["seed"]
                for entry in grid.values()
                if entry["preset"] == preset and entry["mode"] == "oracle"
            }
        )
        for seed in seeds:
            oracle = next(
                (
                    entry
                    for entry in grid.values()
                    if entry["preset"] == preset
                    and entry["mode"] == "oracle"
                    and entry["seed"] == seed
                ),
                None,
            )
            baseline = next(
                (
                    entry
                    for entry in grid.values()
                    if entry["preset"] == preset
                    and entry["mode"] == "none"
                    and entry["seed"] == seed
                ),
                None,
            )
            if oracle is None or baseline is None:
                continue
            oracle_curve = {
                point["update"]: point["success_rate"] for point in oracle["eval_curve"]
            }
            none_curve = {
                point["update"]: point["success_rate"] for point in baseline["eval_curve"]
            }
            common = sorted(set(oracle_curve) & set(none_curve))
            if not common:
                continue
            deltas = {u: oracle_curve[u] - none_curve[u] for u in common}
            cells = " | ".join(
                f"{deltas[u]:+.2f}" if u in deltas else "-" for u in (50, 100, 150, 200, 250, 300)
            )
            asymptotic = statistics.fmean(deltas[u] for u in common[-3:])
            print(
                f"| {preset} | {seed} | {cells} | {max(deltas.values()):+.2f} | {asymptotic:+.2f} |"
            )
            gaps[f"{preset}_s{seed}"] = {
                "preset": preset,
                "seed": seed,
                "deltas": {str(u): deltas[u] for u in common},
                "max_gap": max(deltas.values()),
                "asymptotic_gap": asymptotic,
            }

    # --- predicted sweep --------------------------------------------------------------
    sweep = {name: entry for name, entry in summary.items() if entry["kind"] == "sweep"}
    if sweep:
        print("\n## predicted sweep (aux_coef x history_len; greedy eval, 32 episodes)\n")
        print("| history_len | aux_coef | u50 | u100 | u150 | u200 | final 128-ep success |")
        print("|---|---|---|---|---|---|---|")
        for name, entry in sorted(
            sweep.items(), key=lambda item: (item[1]["history_len"], item[1]["aux_coef"])
        ):
            curve = {point["update"]: point["success_rate"] for point in entry["eval_curve"]}
            cells = " | ".join(
                f"{curve[u]:.3f}" if u in curve else "-" for u in (50, 100, 150, 200)
            )
            final = entry["final"]
            final_text = f"{final['success_mean']:.3f}" if final else "-"
            print(f"| {entry['history_len']} | {entry['aux_coef']} | {cells} | {final_text} |")
        none_curve: dict[int, float] = {}
        for entry in grid.values():
            if entry["preset"] == "default" and entry["mode"] == "none" and entry["seed"] == 0:
                none_curve = {
                    point["update"]: point["success_rate"] for point in entry["eval_curve"]
                }
        if none_curve:
            print(
                "\nnone baseline (default preset, seed 0): "
                + " | ".join(
                    f"u{u} {none_curve[u]:.3f}" for u in (50, 100, 150, 200) if u in none_curve
                )
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": summary, "gaps": gaps}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
