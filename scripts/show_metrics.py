"""Print the scalar metrics of past runs from their TensorBoard event files.

The trainer now prints a summary when it finishes, so this is for runs that
already ended, were interrupted, or need comparing against each other -- and
for when the TensorBoard web UI is unreachable, which is the normal case behind
a JupyterHub proxy.

    uv run python scripts/show_metrics.py                  # every run under runs/
    uv run python scripts/show_metrics.py FaceTurn         # runs matching a substring
    uv run python scripts/show_metrics.py --last 3         # the 3 most recent
"""

import argparse
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pattern", nargs="?", default="", help="substring of the run directory name")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--last", type=int, default=0, help="only the N most recently modified")
    args = ap.parse_args()

    dirs = [d for d in sorted(glob.glob(f"{args.runs_dir}/*/")) if args.pattern in d]
    if args.last:
        dirs = sorted(dirs, key=os.path.getmtime)[-args.last :]
    if not dirs:
        print(f"no runs under {args.runs_dir}/ matching {args.pattern!r}")
        return 1

    for d in dirs:
        ea = EventAccumulator(d)
        ea.Reload()
        tags = ea.Tags()["scalars"]
        print(f"\n{d}")
        if not tags:
            print("  (no scalars — the run died before its first iteration?)")
            continue
        width = max(len(t) for t in tags)
        print(f"  {'metric'.ljust(width)}   first     last      min       max       n")
        for t in sorted(tags):
            v = [e.value for e in ea.Scalars(t)]
            print(
                f"  {t.ljust(width)} {v[0]:+9.4g} {v[-1]:+9.4g} "
                f"{min(v):+9.4g} {max(v):+9.4g} {len(v):6d}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
