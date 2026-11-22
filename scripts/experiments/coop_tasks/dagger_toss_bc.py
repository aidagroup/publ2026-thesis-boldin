"""DAgger refinement of the `TossCatch-v0` behaviour-cloning baseline.

Plain behaviour cloning on the scripted oracle's own trajectories plateaus at a very low
held-out success: the oracle is a closed-loop planner, so a small action error changes the
ball trajectory, and the next state is one the demonstrations never cover (the classical
covariate shift of imitation learning). This script runs DAgger (Ross et al., 2011) on top of
`scripts/experiments/coop_tasks/train_toss_bc.py`:

1. roll out the CURRENT BC policy (greedy) on the batched GPU physics env and, at every
   visited state, ask the scripted oracle for its action (the label); with probability
   `--oracle-prob` the oracle's action is executed instead of the policy's, which keeps the
   rollouts deep enough to cover the later phases of a rally;
2. aggregate the new (state -> oracle action) pairs with all demonstrations collected so far
   (beta = 0 aggregation: all iterations are kept);
3. retrain the BC policies from scratch on the aggregated dataset with the existing trainer
   (held-out demo episodes for the validation MSE, fresh seeds for the held-out success);
4. use the new checkpoint for the next iteration's rollouts.

The states are labelled by the oracle with its own knowledge of the latency queues the policy
built, so the labels are the CPU oracle's exact actions for that GPU trajectory. The output is
the same `wrench_ppo` checkpoint as the plain BC run (`<runs-dir>/<exp-name>/latest.pt`), plus
`dagger_log.json` with the per-iteration held-out success.

Run on the pod from the repo root:

    MUJOCO_GL=egl /root/.callosum/venv/bin/python \\
        scripts/experiments/coop_tasks/dagger_toss_bc.py \\
        --demos demos/toss_bc.npz --exp-name toss_bc_s0 --iterations 3

Requires that `train_toss_bc.py` has already produced `<runs-dir>/<exp-name>/latest.pt`
(the starting policy); pass `--skip-initial-train` only for debugging.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# `sys.path[0]` is this script's directory when run directly, so the sibling collector's GPU
# adapter can be reused instead of duplicating the oracle plumbing.
from collect_toss_bc_demos import (
    PARTNER_MODE,
    TASK,
    TOSS_SPEC,
    WorldView,
    new_latency_queues,
    observation_slices,
    read_gpu_step,
    world_state,
)
from train_toss_bc import AGENT_KEYS

from callosum.experiments.coop_tasks import make_coop_env, make_oracle
from callosum.experiments.coop_tasks.config import coop_config
from callosum.experiments.coop_tasks.env_gpu import CoopBatchGPU
from callosum.experiments.coop_tasks.logic import sample_toss_randomization
from callosum.experiments.cube_wrench.train import (
    evaluate_greedy,
    load_checkpoint,
    load_weights,
    make_policies,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the DAgger CLI."""
    parser = argparse.ArgumentParser(
        description="DAgger refinement of the TossCatch BC baseline on the GPU env.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--demos", type=Path, default=Path("demos/toss_bc.npz"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--exp-name", default="toss_bc_s0", help="run directory of the BC run")
    parser.add_argument("--iterations", type=int, default=3, help="DAgger iterations")
    parser.add_argument("--num-envs", type=int, default=128, help="GPU worlds per iteration")
    parser.add_argument("--max-steps", type=int, default=150, help="per-iteration rollout step cap")
    parser.add_argument(
        "--base-seed",
        type=int,
        default=5_100_000,
        help="first DAgger rollout seed (kept clear of the demo/eval seed blocks)",
    )
    parser.add_argument(
        "--oracle-prob",
        type=float,
        default=0.25,
        help="probability of executing the oracle action during a rollout (state coverage)",
    )
    parser.add_argument("--epochs", type=int, default=150, help="BC epochs per iteration")
    parser.add_argument("--eval-every", type=int, default=50, help="success-eval period (epochs)")
    parser.add_argument("--patience", type=int, default=40, help="BC early-stop patience")
    parser.add_argument(
        "--eval-episodes", type=int, default=128, help="fresh held-out success episodes"
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=5_500_000,
        help="base seed of the fresh held-out success evaluation",
    )
    parser.add_argument("--device", default="cuda", help="CUDA device")
    parser.add_argument("--seed", type=int, default=0, help="mixing/split seed")
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.num_envs < 1 or args.max_steps < 1:
        parser.error("--iterations, --num-envs and --max-steps must be >= 1")
    if not 0.0 <= args.oracle_prob <= 1.0:
        parser.error(f"--oracle-prob must be in [0, 1], got {args.oracle_prob}")
    return args


def load_policy_checkpoint(path: Path, device: torch.device) -> list[torch.nn.Module]:
    """The checkpoint's two greedy policies (the raw-observation actor-critics of the eval)."""
    payload = load_checkpoint(path)
    policies = make_policies(
        PARTNER_MODE,
        device,
        latent_dim=0,
        input_size=int(payload["policy_input_size"]),
        action_dim=3,
    )
    load_weights(payload, policies, agent_keys=AGENT_KEYS)
    for policy in policies:
        policy.eval()
    return policies


def rollout_iteration(
    args: argparse.Namespace,
    policies: list[torch.nn.Module],
    base_seed: int,
) -> dict[str, Any]:
    """Roll out the current policy for one GPU batch and label every state with the oracle."""
    cfg = coop_config(TASK)
    geometry = make_coop_env(TASK, config=cfg, partner_mode=PARTNER_MODE)
    env = CoopBatchGPU(
        TASK,
        args.num_envs,
        partner_mode=PARTNER_MODE,
        config=cfg,
        seed=base_seed,
        device=args.device,
    )
    slices = observation_slices(TOSS_SPEC, PARTNER_MODE)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed + base_seed)
    obs_left_rows: list[np.ndarray] = []
    obs_right_rows: list[np.ndarray] = []
    label_left_rows: list[np.ndarray] = []
    label_right_rows: list[np.ndarray] = []
    episode_rows: list[np.ndarray] = []
    episode_outcome = np.zeros(args.num_envs, dtype=bool)
    episode_done = np.zeros(args.num_envs, dtype=bool)
    start = time.time()
    try:
        observations = env.reset(seed=base_seed)
        records = [
            sample_toss_randomization(
                cfg.randomization,
                control_dt=cfg.control_dt,
                max_control_steps=cfg.max_control_steps,
                seed=base_seed + world,
            )
            for world in range(args.num_envs)
        ]
        oracles = [make_oracle(TASK, cfg) for _ in range(args.num_envs)]
        queues = new_latency_queues(records)
        episode_ids = np.arange(args.num_envs, dtype=np.int64)
        active = np.ones(args.num_envs, dtype=bool)
        active_tensor = torch.as_tensor(active, device=device)
        actions_left = np.zeros((args.num_envs, TOSS_SPEC.action_dim), dtype=np.float32)
        actions_right = np.zeros_like(actions_left)
        for step in range(args.max_steps):
            active_rows = np.flatnonzero(active)
            if active_rows.size == 0:
                break
            obs_left, obs_right, shared, pad_targets, contact_flags = read_gpu_step(
                env, observations, slices["shared"]
            )
            # The policy's own (greedy) action on every active world.
            with torch.no_grad():
                policy_left = torch.clamp(
                    policies[0].get_action(observations["left"][active_tensor], deterministic=True),
                    -1.0,
                    1.0,
                )
                policy_right = torch.clamp(
                    policies[1].get_action(
                        observations["right"][active_tensor], deterministic=True
                    ),
                    -1.0,
                    1.0,
                )
            policy_left_np = policy_left.cpu().numpy()
            policy_right_np = policy_right.cpu().numpy()
            # The oracle's label for the same states (same pending commands -> exact labels).
            labels_left = np.zeros((active_rows.size, TOSS_SPEC.action_dim), dtype=np.float32)
            labels_right = np.zeros_like(labels_left)
            for position, world in enumerate(active_rows.tolist()):
                view = WorldView(
                    geometry=geometry,
                    config=cfg,
                    randomization=records[world],
                    state=world_state(
                        step,
                        TOSS_SPEC.shared_fields,
                        shared[world],
                        pad_targets[world],
                        contact_flags[world],
                    ),
                    queues=queues[world],
                )
                action = oracles[world].act(view)
                labels_left[position] = np.clip(action["left"], -1.0, 1.0)
                labels_right[position] = np.clip(action["right"], -1.0, 1.0)
            # Execute a mix of the learner's and the oracle's actions (DAgger with coverage).
            use_oracle = rng.random(active_rows.size) < args.oracle_prob
            actions_left[active_rows] = np.where(use_oracle[:, None], labels_left, policy_left_np)
            actions_right[active_rows] = np.where(
                use_oracle[:, None], labels_right, policy_right_np
            )
            obs_left_rows.append(obs_left[active_rows])
            obs_right_rows.append(obs_right[active_rows])
            label_left_rows.append(labels_left)
            label_right_rows.append(labels_right)
            episode_rows.append(episode_ids[active_rows])
            for world in active_rows.tolist():
                for role, action in zip(AGENT_KEYS, (actions_left[world], actions_right[world])):
                    queue = queues[world][role]
                    queue.append(action.astype(np.float64))
                    queue.popleft()
            result = env.step(
                {
                    "left": torch.as_tensor(actions_left, dtype=torch.float32, device=device),
                    "right": torch.as_tensor(actions_right, dtype=torch.float32, device=device),
                }
            )
            done = (result.terminated | result.truncated).detach().cpu().numpy()
            done_rows = np.flatnonzero(done)
            if len(done_rows) != len(result.finished_successes):  # pragma: no cover
                raise RuntimeError(
                    "GPU env contract broken: "
                    f"{len(done_rows)} worlds finished but "
                    f"{len(result.finished_successes)} finished statistics reported"
                )
            for position, world in enumerate(done_rows.tolist()):
                if not active[world]:
                    continue
                episode_outcome[world] = bool(result.finished_successes[position])
                episode_done[world] = True
                active[world] = False
                active_tensor[world] = False
            observations = result.obs
    finally:
        env.close()
        geometry.close()
    total = int(sum(row.shape[0] for row in obs_left_rows))
    return {
        "obs": np.stack(
            (np.concatenate(obs_left_rows), np.concatenate(obs_right_rows)), axis=1
        ).astype(np.float32),
        "actions": np.stack(
            (np.concatenate(label_left_rows), np.concatenate(label_right_rows)), axis=1
        ).astype(np.float32),
        "episode": np.concatenate(episode_rows).astype(np.int32),
        "episode_success": np.ones(args.num_envs, dtype=bool),
        "episode_complete": np.ones(args.num_envs, dtype=bool),
        "episode_seed": base_seed + np.arange(args.num_envs, dtype=np.int64),
        "episode_length": np.zeros(args.num_envs, dtype=np.int32),
        "transitions": total,
        "policy_success_rate": float(episode_outcome.mean()),
        "episodes_finished": int(episode_done.sum()),
        "wall_seconds": time.time() - start,
    }


def merge_datasets(
    base: dict[str, np.ndarray], extra: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    """Concatenate the base demos with every DAgger iteration (beta = 0 aggregation).

    The episode ids of the DAgger parts are shifted past the base ids, and the extra episodes
    are marked successful/complete because their ACTIONS are exact oracle labels (the rollout
    outcome of the behavioural policy is irrelevant for the supervision).
    """
    obs = [base["obs"]]
    actions = [base["actions"]]
    episodes = [base["episode"].astype(np.int64)]
    success = [base["episode_success"].astype(bool)]
    complete = [base["episode_complete"].astype(bool)]
    seeds = [base["episode_seed"].astype(np.int64)]
    lengths = [base["episode_length"].astype(np.int32)]
    next_id = int(base["episode"].max()) + 1 if base["episode"].size else 0
    for part in extra:
        obs.append(part["obs"])
        actions.append(part["actions"])
        episodes.append(part["episode"].astype(np.int64) + next_id)
        success.append(part["episode_success"].astype(bool))
        complete.append(part["episode_complete"].astype(bool))
        seeds.append(part["episode_seed"].astype(np.int64))
        lengths.append(part["episode_length"].astype(np.int32))
        next_id += int(part["episode"].max()) + 1
    return {
        "obs": np.concatenate(obs).astype(np.float32),
        "actions": np.concatenate(actions).astype(np.float32),
        "episode": np.concatenate(episodes).astype(np.int32),
        "episode_success": np.concatenate(success),
        "episode_complete": np.concatenate(complete),
        "episode_seed": np.concatenate(seeds),
        "episode_length": np.concatenate(lengths),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the DAgger iterations and leave the final checkpoint in the BC run directory."""
    args = parse_args(argv)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    run_dir = Path(args.runs_dir) / args.exp_name
    checkpoint = run_dir / "latest.pt"
    if not checkpoint.exists():
        raise SystemExit(f"{checkpoint} does not exist; run train_toss_bc.py first")
    trainer = Path(__file__).with_name("train_toss_bc.py")
    base_meta = json.loads(args.demos.with_suffix(".json").read_text())
    base = {key: value for key, value in np.load(args.demos).items()}
    log: dict[str, Any] = {
        "demos": str(args.demos),
        "iterations": [],
        "hyperparameters": {
            "iterations": args.iterations,
            "num_envs": args.num_envs,
            "max_steps": args.max_steps,
            "base_seed": args.base_seed,
            "oracle_prob": args.oracle_prob,
            "epochs": args.epochs,
            "eval_episodes": args.eval_episodes,
            "eval_seed": args.eval_seed,
        },
        "base_transitions": int(base["obs"].shape[0]),
    }
    eval_env = CoopBatchGPU(
        TASK,
        min(args.eval_episodes, 128),
        partner_mode=PARTNER_MODE,
        config=coop_config(TASK),
        seed=args.eval_seed,
        device=str(device),
    )
    extra: list[dict[str, Any]] = []
    aggregated_path = args.demos.with_name(args.demos.stem + "_dagger.npz")
    try:
        for iteration in range(1, args.iterations + 1):
            seed = args.base_seed + (iteration - 1) * args.num_envs
            policies = load_policy_checkpoint(checkpoint, device)
            rollout = rollout_iteration(args, policies, seed)
            extra.append(rollout)
            aggregated = merge_datasets(base, extra)
            metadata = dict(base_meta)
            metadata.update(
                {
                    "source": "oracle demos + DAgger relabelling",
                    "dagger_iterations": iteration,
                    "dagger_transitions": sum(part["transitions"] for part in extra),
                    "dagger_policy_success_rates": [part["policy_success_rate"] for part in extra],
                    "transitions": int(aggregated["obs"].shape[0]),
                    "successful_transitions": int(
                        np.isin(
                            aggregated["episode"], np.flatnonzero(aggregated["episode_success"])
                        ).sum()
                    ),
                    "complete_episodes": int(aggregated["episode_complete"].sum()),
                    "episode_seed_range": [
                        int(aggregated["episode_seed"].min()),
                        int(aggregated["episode_seed"].max()),
                    ],
                }
            )
            np.savez_compressed(aggregated_path, **aggregated)
            aggregated_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            # Retrain from scratch on the aggregated dataset (the trainer's own model selection
            # keeps the weights with the best fresh held-out success).
            command = [
                sys.executable,
                str(trainer),
                "--demos",
                str(aggregated_path),
                "--runs-dir",
                str(args.runs_dir),
                "--exp-name",
                args.exp_name,
                "--epochs",
                str(args.epochs),
                "--eval-every",
                str(args.eval_every),
                "--patience",
                str(args.patience),
                "--eval-episodes",
                str(args.eval_episodes),
                "--eval-seed",
                str(args.eval_seed),
                "--device",
                str(device),
            ]
            print(f"--- DAgger iteration {iteration}: retraining on aggregated demos:", flush=True)
            print(" ".join(command), flush=True)
            code = subprocess.run(command, cwd=Path.cwd(), check=False).returncode
            if code != 0:
                raise SystemExit(f"BC retraining failed with exit code {code}")
            policies = load_policy_checkpoint(checkpoint, device)
            summary = evaluate_greedy(
                policies,
                eval_env,
                args.eval_episodes,
                device=device,
                seed=args.eval_seed,
                agent_keys=AGENT_KEYS,
            )
            log["iterations"].append(
                {
                    "iteration": iteration,
                    "seed": seed,
                    "rollout_transitions": rollout["transitions"],
                    "rollout_policy_success_rate": rollout["policy_success_rate"],
                    "rollout_episodes_finished": rollout["episodes_finished"],
                    "aggregated_transitions": int(aggregated["obs"].shape[0]),
                    "held_out_success_rate": summary.success_rate,
                    "held_out_mean_length": summary.mean_length,
                }
            )
            print(
                f"DAgger iteration {iteration}: rollout success "
                f"{rollout['policy_success_rate']:.3f} | aggregated "
                f"{aggregated['obs'].shape[0]} transitions | held-out success "
                f"{summary.success_rate:.3f}",
                flush=True,
            )
    finally:
        eval_env.close()
    (run_dir / "dagger_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"wrote {checkpoint} | wrote {run_dir / 'dagger_log.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
