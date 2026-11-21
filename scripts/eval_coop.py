"""Evaluate a trained `wrench_ppo` checkpoint on the reduced two-agent coop suite.

Loads a `latest.pt` checkpoint of `callosum.training.wrench_ppo` (weights + config) and runs
the two deterministic (greedy) policies for `--episodes` episodes of the checkpoint's coop
task (`--task cartpole`/`--task pendulum` is recorded in the checkpoint config; the cube has
its own `scripts/eval_wrench_cube.py`). The metrics the task-suite comparison needs are
written as JSON: success rate, steps / steps to success and the task diagnostics (the final
pole angle for the cart-pole, the achieved amplitude and completed hold cycles for the
pendulum).

The ablation's three policy **input modes** are supported exactly as in the cube evaluator:

* `none` -- the env observation only (`partner_mode="none"`);
* `oracle` -- the env's privileged partner block (`partner_mode="oracle"`);
* `predicted` -- Bi-JEPA: the agent's own observation history is encoded and the partner
  predictor produces `ẑ_j`, appended to the own observation. The env still runs with
  `partner_mode="none"`, so the agent never sees the partner block it is asked to predict.

The input mode is taken from `--input-mode`, or inferred from the checkpoint when the flag is
omitted; a mismatch fails before anything is built (`wrench_ppo.check_checkpoint_compat`).

Greedy: the actions are the policies' distribution means, clipped to the env's `[-1, 1]`
box. The evaluation is a FIXED held-out seed set: episode `k` uses the seed
`--seed + EVAL_SEED_OFFSET + k` (`EVAL_SEED_OFFSET = 10_000_019`, outside the training seed
stream). The env's task config is read from the checkpoint's config, so a run is evaluated
with the same task parameters it was trained on.

Run it on the server (the coop suite is CPU MuJoCo, but the plan forbids training/sim on the
laptop outside the small smoke checks):

    python scripts/eval_coop.py --checkpoint runs/coop_cartpole_oracle_s0/latest.pt \\
        --input-mode oracle --episodes 128 --seed 0 --out results_coop_cartpole_oracle_s0.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from callosum.common.metrics import EpisodeRecord, success_rate, time_to_success_stats
from callosum.configs.wrench_ppo import BACKENDS, INPUT_MODES, TASKS
from callosum.envs.coop import make_coop_env
from callosum.training.wrench_ppo import (
    EVAL_SEED_OFFSET,
    LatentInputBuilder,
    check_checkpoint_compat,
    config_from_payload,
    load_bijepa_weights,
    load_checkpoint,
    load_weights,
    make_bijepa,
    make_policies,
)

_COOP_TASKS = tuple(task for task in TASKS if task != "cube")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate a wrench_ppo checkpoint (greedy policies) on a coop task."
    )
    parser.add_argument("--checkpoint", required=True, help="path to latest.pt")
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default=None,
        help="policy input mode: none, oracle or predicted (default: the checkpoint's mode)",
    )
    parser.add_argument("--episodes", type=int, default=128, help="episodes to run")
    parser.add_argument("--seed", type=int, default=0, help="base episode seed")
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="cpu",
        help="evaluation backend; the coop suite is CPU-only at this step (warp is cube-only)",
    )
    parser.add_argument("--out", default="results_coop.json", help="JSON output file")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error(f"--episodes must be >= 1, got {args.episodes}")
    if args.seed < 0:
        parser.error(f"--seed must be >= 0, got {args.seed}")
    if args.backend != "cpu":
        parser.error("the coop suite has no warp backend yet; evaluate with --backend cpu")
    return args


def resolve_input_mode(payload: dict, args: argparse.Namespace) -> str:
    """The evaluation's policy input mode: the explicit flag, else the checkpoint's mode."""
    if args.input_mode is not None:
        saved = payload.get("input_mode")
        if saved is not None and saved != args.input_mode:
            raise ValueError(
                f"checkpoint was made with input_mode={saved!r}, --input-mode is "
                f"{args.input_mode!r}"
            )
        return args.input_mode
    saved = payload.get("input_mode")
    if saved is not None:
        return str(saved)
    legacy = payload.get("partner_mode")
    if legacy is not None:
        return str(legacy)
    raise ValueError(
        f"cannot infer the policy input mode of {args.checkpoint}; pass --input-mode "
        f"(one of {INPUT_MODES})"
    )


def _batch_obs(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """A single-env numpy observation as a `(1, obs_dim)` batch of float32 tensors."""
    return {
        key: torch.as_tensor(value, dtype=torch.float32).unsqueeze(0) for key, value in obs.items()
    }


def run_episode(
    env,
    policies: list[torch.nn.Module],
    seed: int,
    agent_keys: tuple[str, ...],
    builder: LatentInputBuilder | None = None,
) -> tuple[EpisodeRecord, dict]:
    """Run one greedy episode; returns its record and the env's final info dict.

    Under `predicted` (`builder` is not `None`) the policy input is built from the agent's own
    observation history exactly as in training: `[own observation, ẑ_partner]`; otherwise the
    env observation itself is the policy input.
    """
    obs, _info = env.reset(seed=seed)
    obs_batch = _batch_obs(obs)
    policy_obs = builder.reset(obs_batch) if builder is not None else obs_batch
    steps = 0
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        actions = {}
        for name, policy in zip(agent_keys, policies):
            with torch.no_grad():
                action = policy.get_action(policy_obs[name], deterministic=True)
            actions[name] = torch.clamp(action, -1.0, 1.0).squeeze(0).numpy()
        obs, _reward, terminated, truncated, info = env.step(actions)
        steps += 1
        if terminated or truncated:
            break
        obs_batch = _batch_obs(obs)
        policy_obs = builder.step(obs_batch, torch.zeros(1)) if builder is not None else obs_batch
    record = EpisodeRecord(
        success=bool(info["success"]),
        time_to_success=steps if info["success"] else None,
        decisions=steps,
    )
    return record, info


def summarize(records: list[EpisodeRecord], infos: list[dict]) -> dict:
    """The JSON-ready metric block of one evaluation."""
    steps = np.asarray([record.decisions for record in records], dtype=np.float64)
    successes = sum(1 for record in records if record.success)
    time = time_to_success_stats(records)
    summary = {
        "episodes": len(records),
        "successes": successes,
        "failures": len(records) - successes,
        "success_rate": success_rate(records),
        "mean_angle_deg": float(np.mean([info["angle_deg"] for info in infos])),
        "steps_mean": float(np.mean(steps)),
        "steps_to_success_mean": None if time is None else time[0],
        "steps_to_success_std": None if time is None else time[1],
    }
    if infos and "amplitude_deg" in infos[0]:
        summary["mean_amplitude_deg"] = float(np.mean([info["amplitude_deg"] for info in infos]))
        summary["mean_completed_cycles"] = float(
            np.mean([info["completed_cycles"] for info in infos])
        )
        summary["hold_cycles"] = float(infos[0]["hold_cycles"])
    return summary


def main(argv: list[str] | None = None) -> int:
    """Evaluate the checkpoint, print the summary and write the JSON report."""
    args = parse_args(argv)
    payload = load_checkpoint(args.checkpoint)
    cfg = config_from_payload(payload)
    if cfg.task not in _COOP_TASKS:
        raise ValueError(
            f"checkpoint task is {cfg.task!r}; scripts/eval_coop.py evaluates the coop tasks "
            f"{_COOP_TASKS} (the cube has scripts/eval_wrench_cube.py)"
        )
    input_mode = resolve_input_mode(payload, args)
    env_mode, latent_dim = check_checkpoint_compat(payload, input_mode)
    env_config = cfg.coop_env_config()
    policies = make_policies(
        env_mode,
        "cpu",
        latent_dim=latent_dim,
        input_size=cfg.env_obs_size,
        action_dim=cfg.action_size,
    )
    load_weights(payload, policies, agent_keys=cfg.agent_keys)
    builder: LatentInputBuilder | None = None
    bijepa = make_bijepa(cfg, "cpu")
    if bijepa is not None:
        load_bijepa_weights(payload, bijepa)
        bijepa.eval()
        builder = LatentInputBuilder(bijepa, device="cpu", agent_keys=cfg.agent_keys)

    records: list[EpisodeRecord] = []
    infos: list[dict] = []
    print(
        f"eval {args.checkpoint} | task {cfg.task} | input_mode {input_mode} (env "
        f"partner_mode {env_mode}) | policy input {cfg.env_obs_size + latent_dim} | "
        f"{args.episodes} episodes from held-out seed {args.seed + EVAL_SEED_OFFSET} | "
        "greedy policies",
        flush=True,
    )
    env = make_coop_env(cfg.task, config=env_config, partner_mode=env_mode)
    try:
        for episode in range(args.episodes):
            record, info = run_episode(
                env,
                policies,
                args.seed + EVAL_SEED_OFFSET + episode,
                cfg.agent_keys,
                builder,
            )
            records.append(record)
            infos.append(info)
    finally:
        env.close()

    summary = summarize(records, infos)
    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "task": cfg.task,
        "input_mode": input_mode,
        "partner_mode": env_mode,
        "backend": "cpu",
        "policy_input_size": cfg.env_obs_size + latent_dim,
        "latent_dim": latent_dim,
        "seed": args.seed,
        "eval_seed_offset": EVAL_SEED_OFFSET,
        "episode_seed_range": [
            args.seed + EVAL_SEED_OFFSET,
            args.seed + EVAL_SEED_OFFSET + args.episodes - 1,
        ],
        "format": payload.get("format"),
        "checkpoint_update": payload.get("update"),
        **summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    time = summary["steps_to_success_mean"]
    extra = ""
    if "mean_amplitude_deg" in summary:
        extra = (
            f" | amplitude_deg={summary['mean_amplitude_deg']:.1f} "
            f"cycles={summary['mean_completed_cycles']:.2f}/{summary['hold_cycles']:.0f}"
        )
    print(
        f"success_rate={summary['success_rate']:.3f} "
        f"({summary['successes']}/{summary['episodes']}, {summary['failures']} failures) | "
        f"angle_deg={summary['mean_angle_deg']:.1f} | "
        f"steps={summary['steps_mean']:.1f} | "
        f"steps_to_success={'-' if time is None else f'{time:.1f}'}{extra}"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
