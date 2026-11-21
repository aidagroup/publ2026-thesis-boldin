"""Evaluate a trained `wrench_ppo` checkpoint on CubeWrench-v0 (PLAN.md steps 2 and 3).

Loads a `latest.pt` checkpoint of `callosum.training.wrench_ppo` (weights + config) and runs
the two deterministic (greedy) policies for `--episodes` episodes of the reduced wrench-level
cube, writing the metrics the plan's ablation table needs: success rate, final layer angle,
core position / rotation drift, steps to success and the failure count.

The ablation's three policy **input modes** are supported:

* `none` -- the env observation only (`WrenchCubeEnv(partner_mode="none")`);
* `oracle` -- the env's privileged partner block (`WrenchCubeEnv(partner_mode="oracle")`);
* `predicted` -- Bi-JEPA: every step the agent's own 26-d observation history is encoded and
  the partner predictor produces `ẑ_j`, appended to the own observation (26 + 32 = 58).
  The env still runs with `partner_mode="none"` and the agent never sees the partner's
  current observation -- the same path as in training.

The input mode is taken from `--input-mode`, or inferred from the checkpoint when the flag is
omitted (`--partner-mode` is kept as the old `none`/`oracle` spelling); a mismatch between the
checkpoint and the requested mode fails before anything is built
(`wrench_ppo.check_checkpoint_compat`), including the policy input width.

Greedy: the actions are the policies' distribution means, clipped to the env's `[-1, 1]`
wrench box. The episode seeds are `--seed + i`, so a run is reproducible, and the env's
randomisation (latency, gain, authority, slip, start delay, impulse) is the training one.

The aggregation reuses `callosum.eval.metrics` (`EpisodeRecord`, `success_rate`,
`time_to_success_stats`) -- one record per episode, with the decision count equal to the
control-step count of this env.

Run it on the server (MuJoCo is CPU-only but the plan forbids training/sim on the laptop
outside the small smoke checks):

    python scripts/eval_wrench_cube.py --checkpoint runs/wrench_oracle_s0/latest.pt \\
        --input-mode oracle --episodes 128 --seed 0 --out results_wrench_oracle_s0.json
    python scripts/eval_wrench_cube.py --checkpoint runs/wrench_pred_s0/latest.pt \\
        --input-mode predicted --episodes 128 --seed 0 --out results_wrench_pred_s0.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from callosum.configs.wrench_ppo import INPUT_MODES
from callosum.envs._wrench_logic import AGENT_KEYS, PARTNER_MODES, obs_size
from callosum.envs.wrench_cube import WrenchCubeEnv
from callosum.eval.metrics import EpisodeRecord, success_rate, time_to_success_stats
from callosum.training.wrench_ppo import (
    LatentInputBuilder,
    check_checkpoint_compat,
    config_from_payload,
    load_bijepa_weights,
    load_checkpoint,
    load_weights,
    make_bijepa,
    make_policies,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate a wrench_ppo checkpoint (greedy policies) on CubeWrench-v0."
    )
    parser.add_argument("--checkpoint", required=True, help="path to latest.pt")
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default=None,
        help="policy input mode: none, oracle or predicted (default: the checkpoint's mode)",
    )
    parser.add_argument(
        "--partner-mode",
        choices=PARTNER_MODES,
        default=None,
        help="deprecated alias of --input-mode none|oracle (kept for the step-2 commands)",
    )
    parser.add_argument("--episodes", type=int, default=64, help="episodes to run")
    parser.add_argument("--seed", type=int, default=0, help="base episode seed")
    parser.add_argument("--out", default="results.json", help="JSON output file of the metrics")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error(f"--episodes must be >= 1, got {args.episodes}")
    if args.seed < 0:
        parser.error(f"--seed must be >= 0, got {args.seed}")
    if (
        args.input_mode is not None
        and args.partner_mode is not None
        and args.input_mode != args.partner_mode
    ):
        parser.error(
            f"--input-mode {args.input_mode!r} and --partner-mode "
            f"{args.partner_mode!r} disagree; pass only --input-mode"
        )
    return args


def resolve_input_mode(payload: dict, args: argparse.Namespace) -> str:
    """The evaluation's policy input mode: explicit flag, else the checkpoint's saved mode."""
    explicit = args.input_mode or args.partner_mode
    saved = payload.get("input_mode")
    if explicit is None:
        if saved is not None:
            return str(saved)
        # Step-2 checkpoints predate `input_mode`: their partner mode *is* their input mode.
        legacy = payload.get("partner_mode")
        if legacy is not None:
            return str(legacy)
        raise ValueError(
            f"cannot infer the policy input mode of {args.checkpoint}; pass --input-mode "
            f"(one of {INPUT_MODES})"
        )
    if saved is not None and saved != explicit:
        raise ValueError(
            f"checkpoint was made with input_mode={saved!r}, --input-mode is {explicit!r}"
        )
    return explicit


def _batch_obs(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """A single-env numpy observation as a `(1, obs_dim)` batch of float32 tensors."""
    return {
        key: torch.as_tensor(value, dtype=torch.float32).unsqueeze(0) for key, value in obs.items()
    }


def run_episode(
    env: WrenchCubeEnv,
    policies: list[torch.nn.Module],
    seed: int,
    builder: LatentInputBuilder | None = None,
) -> tuple[EpisodeRecord, dict]:
    """Run one greedy episode; returns its record and the env's final info dict.

    Under `predicted` (`builder` is not `None`) the policy input is built from the agent's own
    observation history exactly as in training: `[own observation, ẑ_partner]`, with the
    partner never observed; otherwise the env observation itself is the policy input.
    """
    obs, info = env.reset(seed=seed)
    obs_batch = _batch_obs(obs)
    policy_obs = builder.reset(obs_batch) if builder is not None else obs_batch
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        actions = {}
        for name, policy in zip(AGENT_KEYS, policies):
            with torch.no_grad():
                action = policy.get_action(policy_obs[name], deterministic=True)
            actions[name] = torch.clamp(action, -1.0, 1.0).squeeze(0).numpy()
        obs, _reward, terminated, truncated, info = env.step(actions)
        steps += 1
        if terminated or truncated:
            break
        obs_batch = _batch_obs(obs)
        policy_obs = builder.step(obs_batch, torch.zeros(1)) if builder is not None else obs_batch
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
    input_mode = resolve_input_mode(payload, args)
    env_mode, latent_dim = check_checkpoint_compat(payload, input_mode)
    cfg = config_from_payload(payload)
    if input_mode == "predicted" and (not cfg.use_bijepa or cfg.latent_dim != latent_dim):
        raise ValueError(
            "checkpoint is inconsistent: it stores Bi-JEPA weights (latent_dim="
            f"{latent_dim}) but its saved config has use_bijepa={cfg.use_bijepa} and "
            f"latent_dim={cfg.latent_dim}"
        )
    policies = make_policies(env_mode, "cpu", latent_dim=latent_dim)
    load_weights(payload, policies)
    builder: LatentInputBuilder | None = None
    bijepa = make_bijepa(cfg, "cpu")
    if bijepa is not None:
        load_bijepa_weights(payload, bijepa)
        bijepa.eval()
        builder = LatentInputBuilder(bijepa, device="cpu")

    records: list[EpisodeRecord] = []
    infos: list[dict] = []
    env = WrenchCubeEnv(partner_mode=env_mode)
    try:
        print(
            f"eval {args.checkpoint} | input_mode {input_mode} (env partner_mode {env_mode}) | "
            f"policy input {obs_size(env_mode) + latent_dim} | {args.episodes} episodes from "
            f"seed {args.seed} | greedy policies",
            flush=True,
        )
        for episode in range(args.episodes):
            record, info = run_episode(env, policies, args.seed + episode, builder)
            records.append(record)
            infos.append(info)
    finally:
        env.close()

    summary = summarize(records, infos)
    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "input_mode": input_mode,
        "partner_mode": env_mode,
        "policy_input_size": obs_size(env_mode) + latent_dim,
        "latent_dim": latent_dim,
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
