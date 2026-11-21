"""Evaluate a trained `wrench_ppo` checkpoint on CubeWrench-v0 (PLAN.md step 2).

Loads a `latest.pt` checkpoint of `callosum.training.wrench_ppo` (weights + config), runs
the two deterministic (greedy) policies for `--episodes` episodes of the reduced
wrench-level cube and writes the metrics the plan's ablation table needs: success rate,
final layer angle, core position / rotation drift, steps to success and the failure count.

Greedy: the actions are the policies' distribution means, clipped to the env's `[-1, 1]`
wrench box. The episode seeds are `--seed + i`, so a run is reproducible, and the env's
randomisation (latency, gain, authority, slip, start delay, impulse) is the training one.

The aggregation reuses `callosum.eval.metrics` (`EpisodeRecord`, `success_rate`,
`time_to_success_stats`) -- one record per episode, with the decision count equal to the
control-step count of this env.

Run it on the server (MuJoCo is CPU-only but the plan forbids training/sim on the laptop
outside the small smoke checks):

    python scripts/eval_wrench_cube.py --checkpoint runs/wrench_oracle_s0/latest.pt \\
        --partner-mode oracle --episodes 128 --seed 0 --out results_wrench_oracle_s0.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from callosum.envs._wrench_logic import AGENT_KEYS, PARTNER_MODES, obs_size
from callosum.envs.wrench_cube import WrenchCubeEnv
from callosum.eval.metrics import EpisodeRecord, success_rate, time_to_success_stats
from callosum.training.wrench_ppo import load_checkpoint, load_weights, make_policies


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate a wrench_ppo checkpoint (greedy policies) on CubeWrench-v0."
    )
    parser.add_argument("--checkpoint", required=True, help="path to latest.pt")
    parser.add_argument(
        "--partner-mode",
        choices=PARTNER_MODES,
        required=True,
        help="env observation mode; must be the one the checkpoint was trained with",
    )
    parser.add_argument("--episodes", type=int, default=64, help="episodes to run")
    parser.add_argument("--seed", type=int, default=0, help="base episode seed")
    parser.add_argument("--out", default="results.json", help="JSON output file of the metrics")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error(f"--episodes must be >= 1, got {args.episodes}")
    if args.seed < 0:
        parser.error(f"--seed must be >= 0, got {args.seed}")
    return args


def check_checkpoint_compat(payload: dict, partner_mode: str) -> None:
    """Raise if the checkpoint was trained on another observation layout than `partner_mode`."""
    saved_mode = payload.get("partner_mode")
    if saved_mode is not None and saved_mode != partner_mode:
        raise ValueError(
            f"checkpoint was made with partner_mode={saved_mode!r}, this run uses "
            f"{partner_mode!r} (the policy input width differs: "
            f"{payload.get('obs_size')} vs {obs_size(partner_mode)})"
        )
    expected = obs_size(partner_mode)
    saved_size = payload.get("obs_size")
    if saved_size is not None and int(saved_size) != expected:
        raise ValueError(
            f"checkpoint policies take inputs of width {saved_size} but {partner_mode!r} "
            f"observations are {expected} wide"
        )


def run_episode(
    env: WrenchCubeEnv, policies: list[torch.nn.Module], seed: int
) -> tuple[EpisodeRecord, dict]:
    """Run one greedy episode; returns its record and the env's final info dict."""
    obs, info = env.reset(seed=seed)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        actions = {}
        for name, policy in zip(AGENT_KEYS, policies):
            with torch.no_grad():
                action = policy.get_action(
                    torch.as_tensor(obs[name]).unsqueeze(0), deterministic=True
                )
            actions[name] = torch.clamp(action, -1.0, 1.0).squeeze(0).numpy()
        obs, _reward, terminated, truncated, info = env.step(actions)
        steps += 1
    succeeded = bool(info["success"])
    record = EpisodeRecord(
        success=succeeded,
        time_to_success=steps if succeeded else None,
        decisions=steps,
    )
    return record, info


def summarize(records: list[EpisodeRecord], infos: list[dict]) -> dict:
    """The JSON-ready metric block of one evaluation."""
    steps = np.asarray([record.decisions for record in records], dtype=np.float64)
    time = time_to_success_stats(records)
    successes = sum(1 for record in records if record.success)
    return {
        "episodes": len(records),
        "successes": successes,
        "failures": len(records) - successes,
        "success_rate": success_rate(records),
        "mean_angle_deg": float(np.mean([info["angle_deg"] for info in infos])),
        "mean_drift_pos_m": float(np.mean([info["drift_pos"] for info in infos])),
        "mean_drift_rot_rad": float(np.mean([info["drift_rot"] for info in infos])),
        "steps_mean": float(np.mean(steps)),
        "steps_to_success_mean": None if time is None else time[0],
        "steps_to_success_std": None if time is None else time[1],
    }


def main(argv: list[str] | None = None) -> int:
    """Evaluate the checkpoint, print the summary and write the JSON report."""
    args = parse_args(argv)
    payload = load_checkpoint(args.checkpoint)
    check_checkpoint_compat(payload, args.partner_mode)
    policies = make_policies(args.partner_mode, "cpu")
    load_weights(payload, policies)

    records: list[EpisodeRecord] = []
    infos: list[dict] = []
    env = WrenchCubeEnv(partner_mode=args.partner_mode)
    try:
        print(
            f"eval {args.checkpoint} | partner_mode {args.partner_mode} | "
            f"{args.episodes} episodes from seed {args.seed} | greedy policies",
            flush=True,
        )
        for episode in range(args.episodes):
            record, info = run_episode(env, policies, args.seed + episode)
            records.append(record)
            infos.append(info)
    finally:
        env.close()

    summary = summarize(records, infos)
    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "partner_mode": args.partner_mode,
        "seed": args.seed,
        "format": payload.get("format"),
        "checkpoint_update": payload.get("update"),
        **summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    time = summary["steps_to_success_mean"]
    print(
        f"success_rate={summary['success_rate']:.3f} "
        f"({summary['successes']}/{summary['episodes']}, {summary['failures']} failures) | "
        f"angle_deg={summary['mean_angle_deg']:.1f} | "
        f"drift_pos_mm={summary['mean_drift_pos_m'] * 1e3:.1f} | "
        f"drift_rot_deg={math.degrees(summary['mean_drift_rot_rad']):.1f} | "
        f"steps={summary['steps_mean']:.1f} | "
        f"steps_to_success={'-' if time is None else f'{time:.1f}'}"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
