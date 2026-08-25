"""Step 3.3: Bi-JEPA partner-input ablation launcher.

Runs the three `partner_input` modes (none / oracle / predicted) of
``callosum.training.ippo`` as **isolated subprocesses** so that a crash or
hang in one mode cannot waste an entire paid GPU pod session (see
``docs/server-runbook.md`` -- the GPU is billed by the hour).

Design notes:
- The launcher itself is stdlib-only (no torch / no mani_skill), so it is
  importable and `--help`-able on macOS/CI. Only the spawned ``ippo``
  subprocesses are server-only.
- The SOLE ablation variable is ``partner_input`` -- that is the single
  switch that changes the policy's partner channel while keeping network
  width, seed, and Bi-JEPA hyperparams identical, which is what makes the
  triple valid (docs/implementation-plan.md step 3.2). The Bi-JEPA module
  knobs (``latent_dim``, ``aux_weight``, ...) live in the nested
  ``BiJEPAConfig`` and are deliberately NOT CLI-exposed (see
  callosum.configs.ippo.parse_args); they use the defaults documented in
  callosum/configs/ablation.yaml. To sweep them, add YAML loading to
  callosum.training.ippo first.
- All three modes in a *round* share the SAME seed (review Б1: otherwise
  "oracle beats none" could be a variance artifact, not an information
  effect). Use `--seed <S>` for the first round and `--seeds <N>` for a
  multi-round robustness sweep; each round s uses seed = --seed + s.
- Each mode gets a distinct ``exp_name``
  (``<env>__ablation_<mode>__s<seed>``) so TensorBoard logs are
  disambiguated under ``runs/``.
- A JSONL summary (``runs/<ablation-dir>/ablation_summary.jsonl``) records
  mode, seed, run_name, return-code, elapsed time, and the per-mode stdout
  log path for post-hoc comparison.

Example (directional ablation budget; raise for the full gate run):
    uv run python scripts/run_ablation.py --env-id FaceTurn-v0 --total-timesteps 2000000 --seed 1
    uv run python scripts/run_ablation.py --env-id FaceTurn-v0 --total-timesteps 2000000 --seed 1 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MODES: tuple[str, ...] = ("none", "oracle", "predicted")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-id", default="FaceTurn-v0", help="env to ablate on")
    ap.add_argument(
        "--total-timesteps",
        type=int,
        default=2_000_000,
        help="per-mode budget (directional ablation default; raise to the "
        "full 10M gate for final numbers)",
    )
    ap.add_argument("--seed", type=int, default=1, help="base seed for round 1")
    ap.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="number of seed rounds (>=1); all three modes share the SAME seed "
        "within a round so the ablation is comparable (review Б1). Round s "
        "uses seed = --seed + s. Default 1 (single round).",
    )
    ap.add_argument(
        "--num-envs", type=int, default=None, help="override num_envs forwarded to ippo"
    )
    ap.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="token list forwarded verbatim to ippo, e.g. '--control-mode pd_joint_delta_pos'",
    )
    ap.add_argument(
        "--ablation-dir",
        default=None,
        help="under runs/ -- defaults to ablation-<timestamp>",
    )
    args = ap.parse_args()

    ablation_dir = args.ablation_dir or f"ablation-{int(time.time())}"
    runs_root = Path("runs") / ablation_dir
    runs_root.mkdir(parents=True, exist_ok=True)
    summary_path = runs_root / "ablation_summary.jsonl"

    rc_overall = 0
    seeds = range(args.seed, args.seed + max(args.seeds, 1))
    n_total = len(seeds) * len(MODES)
    step = 0
    for s in seeds:
        for idx, mode in enumerate(MODES):
            step += 1
            run_name = f"{args.env_id}__ablation_{mode}__s{s}"
            # Seed is part of the log filename so multi-round sweeps don't
            # clobber each other's stdout.
            log_path = runs_root / f"{mode}__s{s}.log"

            cmd = [
                sys.executable,
                "-m",
                "callosum.training.ippo",
                "--env-id",
                args.env_id,
                "--partner-input",
                mode,
                "--total-timesteps",
                str(args.total_timesteps),
                "--seed",
                str(s),  # SAME seed for all three modes in this round (Б1)
                "--exp-name",
                run_name,
            ]
            if args.num_envs is not None:
                cmd += ["--num-envs", str(args.num_envs)]
            cmd += list(args.extra)

            print(
                f"\n=== Bi-JEPA ablation: partner_input={mode!r} seed={s} "
                f"(run {step}/{n_total}, idx in round {idx + 1}/{len(MODES)}) ==="
            )
            print(f"tb log: runs/{run_name}")
            print(f"stdout: {log_path}")
            print(f"cmd: {' '.join(cmd)}")

            t0 = time.time()
            with log_path.open("w") as logf:
                try:
                    proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)
                except FileNotFoundError as e:
                    proc = None
                    logf.write(f"launcher error: {e}\n")
                    logf.flush()
            dt = time.time() - t0

            record = {
                "mode": mode,
                "seed": s,
                "run_name": run_name,
                "cmd": cmd,
                "returncode": proc.returncode if proc is not None else -1,
                "elapsed_s": round(dt, 1),
                "stdout_log": str(log_path),
            }
            with summary_path.open("a") as sf:
                sf.write(json.dumps(record) + "\n")
            print(f"  -> rc={record['returncode']} elapsed={dt:.0f}s (summary -> {summary_path})")

            if proc is None or proc.returncode != 0:
                rc_overall = 1
                print("  !! this mode failed; see its log before trusting the ablation.")

    print(f"\nAblation complete. Summary: {summary_path}")
    print("Compare success-rate curves across the three runs/<run_name> dirs in TensorBoard.")
    return rc_overall


if __name__ == "__main__":
    raise SystemExit(main())
