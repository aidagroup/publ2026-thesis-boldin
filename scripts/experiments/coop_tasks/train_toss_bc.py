"""Behaviour-cloning baseline for `TossCatch-v0` from the scripted-oracle demos.

Trains the two IPPO actor-critics (`callosum.common.ppo_core.ActorCritic`, the exact class the
trainer/evaluator build) to clone the scripted oracle's paddle commands from the observations
of `scripts/experiments/coop_tasks/collect_toss_bc_demos.py`. The result is a normal
`wrench_ppo` checkpoint (`<run dir>/latest.pt`, written by
`callosum.experiments.cube_wrench.train.save_checkpoint`), so it can be judged by
`callosum/experiments/coop_tasks/eval.py` and rendered by
`callosum.experiments.coop_tasks.render --scene policy-vs-oracle` unchanged -- this is the
learned `none`-input baseline the IPPO run is compared against.

What is cloned: only the actor mean (MSE to the demonstrated normalized 3-D command); the
critic is untrained (BC has no value target) and `actor_logstd` stays at its initial value.
Observations are standardized per feature with the TRAIN split's statistics, and the
standardization is then folded into the first linear layer of both heads, so the saved policy
consumes the RAW environment observation (the evaluator does not normalize).

Validation: whole demo episodes are held out (`--val-fraction`), plus a separate fresh
held-out seed block for the success metric. During training the script logs the validation
action MSE each epoch; every `--eval-every` epochs (and once at the end) it also runs the
greedy policy on `--eval-episodes` fresh held-out GPU episodes and logs the success rate. The
checkpoint that maximizes the held-out success (tie-break: lowest validation MSE) is the one
saved as `latest.pt`; the loss/success curve goes to `<run dir>/bc_log.json`.

Run on the pod from the repo root (GPU only for the periodic success evaluations; the BC
optimization itself is a tiny MLP fit):

    /root/.callosum/venv/bin/python scripts/experiments/coop_tasks/train_toss_bc.py \\
        --demos demos/toss_bc.npz --exp-name toss_bc_s0 --epochs 300

Then evaluate the checkpoint exactly like an IPPO run:

    /root/.callosum/venv/bin/python -m callosum.experiments.coop_tasks.eval \\
        --checkpoint runs/toss_bc_s0/latest.pt --input-mode none --backend warp \\
        --num-envs 128 --episodes 128 --seed 0 --out results_toss_bc/bc_base0.json
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from callosum.common.ppo_core import bc_mse
from callosum.experiments.cube_wrench.ppo_config import WrenchPPOConfig
from callosum.experiments.cube_wrench.train import (
    EvalSummary,
    evaluate_greedy,
    make_policies,
    save_checkpoint,
)

TASK = "toss"
PARTNER_MODE = "none"
AGENT_KEYS = ("left", "right")
EVAL_ENV_WORLDS = 128
"""Worlds of the held-out success-evaluation batch (one episode per world per batch)."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the BC CLI."""
    parser = argparse.ArgumentParser(
        description="Behaviour-clone the TossCatch scripted oracle from GPU demos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--demos", type=Path, default=Path("demos/toss_bc.npz"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--exp-name", default="toss_bc_s0", help="run directory name")
    parser.add_argument("--seed", type=int, default=0, help="split/shuffle seed")
    parser.add_argument("--epochs", type=int, default=300, help="maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="minibatch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument(
        "--val-fraction", type=float, default=0.15, help="held-out demo episode fraction"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="stop after this many epochs without a validation-MSE improvement",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="also clone the oracle's failed episodes (default: successful episodes only)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="held-out success evaluation period in epochs (0 = only at the end)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=128,
        help="fresh held-out episodes of a success evaluation (GPU)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="base seed of the fresh held-out evaluation block (default: demo base + 500000)",
    )
    parser.add_argument("--device", default="cuda", help="CUDA device of the policies/env")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.eval_episodes < 1:
        parser.error("--epochs, --batch-size and --eval-episodes must be >= 1")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error(f"--val-fraction must be in (0, 1), got {args.val_fraction}")
    if args.patience < 1:
        parser.error(f"--patience must be >= 1, got {args.patience}")
    return args


def load_demos(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read the demo `.npz` and its `.json` metadata sidecar."""
    data = {key: value for key, value in np.load(path).items()}
    required = {"obs", "actions", "episode", "episode_success", "episode_complete", "episode_seed"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"{path} is missing arrays {sorted(missing)}")
    meta_path = path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return data, metadata


def split_episodes(
    episodes: np.ndarray, episode_success: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the successful demo episodes deterministically into train/validation sets.

    The split is by EPISODE, not by transition, so every step of a held-out episode stays out
    of training (otherwise neighbouring frames of one trajectory would leak across the split).
    """
    usable = np.flatnonzero(episode_success)
    if usable.size < 2:
        raise ValueError(f"need at least 2 successful episodes, got {usable.size}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(usable)
    val_count = max(1, round(val_fraction * usable.size))
    val_episodes = np.sort(order[:val_count])
    train_episodes = np.sort(order[val_count:])
    train_mask = np.isin(episodes, train_episodes)
    val_mask = np.isin(episodes, val_episodes)
    return train_mask, val_mask, train_episodes, val_episodes


def fold_obs_normalization(agent: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
    """Fold an input standardization into the first linear layer of both heads (in place).

    After the fold the agent's heads consume the raw observation:
    `W (x - μ) / σ + b == (W / σ) x + (b - W μ / σ)`. This is exact for the linear-first heads
    this repo builds (`ActorCritic.actor_mean`/`critic` start with a `nn.Linear`), so the
    evaluator's unnormalized environment observations are the correct inputs.
    """
    for head in (agent.actor_mean, agent.critic):
        first = head[0]
        if not isinstance(first, nn.Linear):  # pragma: no cover - the repo's MLP layout
            raise TypeError(f"expected a linear first layer, got {type(first).__name__}")
        with torch.no_grad():
            first.weight.mul_(1.0 / std)
            first.bias.sub_(first.weight @ mean)


def _normalized(obs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """The standardized policy input of one agent."""
    return (obs - mean) / std


def _clone_states(policies: list[nn.Module]) -> dict[str, dict[str, torch.Tensor]]:
    """Snapshot the policies' parameters (a plain-state copy for checkpoint selection)."""
    return {
        role: {key: value.detach().clone() for key, value in policy.state_dict().items()}
        for role, policy in zip(AGENT_KEYS, policies)
    }


def _load_states(states: dict[str, dict[str, torch.Tensor]], policies: list[nn.Module]) -> None:
    """Restore a `_clone_states` snapshot into the (training) policies."""
    for role, policy in zip(AGENT_KEYS, policies):
        policy.load_state_dict(states[role])


def main(argv: list[str] | None = None) -> int:
    """Train the BC policies, log the curve and write the `wrench_ppo` checkpoint."""
    args = parse_args(argv)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    data, metadata = load_demos(args.demos)
    episodes = data["episode"].astype(np.int64)
    episode_success = data["episode_success"].astype(bool)
    episode_complete = data["episode_complete"].astype(bool)
    obs, actions = data["obs"], data["actions"]
    if obs.ndim != 3 or obs.shape[1] != 2 or actions.shape != (obs.shape[0], 2, 3):
        raise ValueError(
            f"unexpected demo shapes: obs {obs.shape}, actions {actions.shape} (expected "
            "(transitions, 2 agents, 26) and (transitions, 2 agents, 3))"
        )
    # Drop the episodes the collection cap cut off; their actions have no terminal label.
    keep = episode_complete[episodes]
    if not args.include_failed:
        # The oracle's own failures are ~14% of the episodes; cloning their (best-effort)
        # actions would teach recoveries the script has not actually solved.
        keep &= episode_success[episodes]
    obs, actions, episodes = obs[keep], actions[keep], episodes[keep]
    train_mask, val_mask, train_episodes, val_episodes = split_episodes(
        episodes, episode_success, args.val_fraction, args.seed
    )
    print(
        f"demos {args.demos} | task {metadata.get('task', TASK)} | transitions "
        f"{obs.shape[0]} ({int(train_mask.sum())} train / {int(val_mask.sum())} val) | "
        f"episodes {train_episodes.size} train / {val_episodes.size} val | "
        f"dims {obs.shape[2]} obs -> {actions.shape[2]} act | seed {args.seed}",
        flush=True,
    )
    obs_train = {
        role: torch.as_tensor(obs[train_mask, index], dtype=torch.float32, device=device)
        for index, role in enumerate(AGENT_KEYS)
    }
    act_train = {
        role: torch.as_tensor(actions[train_mask, index], dtype=torch.float32, device=device)
        for index, role in enumerate(AGENT_KEYS)
    }
    obs_val = {
        role: torch.as_tensor(obs[val_mask, index], dtype=torch.float32, device=device)
        for index, role in enumerate(AGENT_KEYS)
    }
    act_val = {
        role: torch.as_tensor(actions[val_mask, index], dtype=torch.float32, device=device)
        for index, role in enumerate(AGENT_KEYS)
    }
    mean = {role: obs_train[role].mean(dim=0) for role in AGENT_KEYS}
    std = {role: obs_train[role].std(dim=0).clamp(min=1e-6) for role in AGENT_KEYS}
    policies = make_policies(
        PARTNER_MODE, device, latent_dim=0, input_size=int(obs.shape[2]), action_dim=3
    )
    optimizers = [torch.optim.Adam(policy.parameters(), lr=args.lr) for policy in policies]
    train_total = int(train_mask.sum())
    batches = max(1, train_total // args.batch_size)
    eval_seed = (
        int(metadata.get("base_seed", 5_000_000)) + 500_000
        if args.eval_seed is None
        else args.eval_seed
    )
    cfg = WrenchPPOConfig(
        exp_name=args.exp_name,
        runs_dir=str(args.runs_dir),
        seed=args.seed,
        total_updates=max(1, args.epochs),
        task=TASK,
        partner_mode=PARTNER_MODE,
        backend="warp",
        device=str(device),
        num_envs=128,
        rollout_steps=64,
    )
    log: dict[str, Any] = {
        "demos": str(args.demos),
        "demos_metadata": metadata,
        "exp_name": args.exp_name,
        "seed": args.seed,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "val_fraction": args.val_fraction,
            "patience": args.patience,
            "include_failed": args.include_failed,
            "eval_every": args.eval_every,
            "eval_episodes": args.eval_episodes,
            "eval_seed": int(eval_seed),
        },
        "episodes": {
            "train": train_episodes.tolist(),
            "val": val_episodes.tolist(),
        },
        "dataset": {
            "transitions": int(obs.shape[0]),
            "train_transitions": train_total,
            "val_transitions": int(val_mask.sum()),
        },
        "epochs": [],
        "evaluations": [],
    }
    # The evaluation env (batched GPU) and the selection bookkeeping.
    from callosum.experiments.coop_tasks.config import coop_config
    from callosum.experiments.coop_tasks.env_gpu import CoopBatchGPU

    eval_env = CoopBatchGPU(
        TASK,
        min(args.eval_episodes, EVAL_ENV_WORLDS),
        partner_mode=PARTNER_MODE,
        config=coop_config(TASK),
        seed=eval_seed,
        device=str(device),
    )

    def run_held_out_eval(epoch: int, val_mse: float) -> tuple[float, float]:
        """Greedy held-out success of the CURRENT policies (normalization folded into a copy)."""
        candidate = [copy.deepcopy(policy) for policy in policies]
        for index, policy in enumerate(candidate):
            fold_obs_normalization(policy, mean[AGENT_KEYS[index]], std[AGENT_KEYS[index]])
        summary: EvalSummary = evaluate_greedy(
            candidate,
            eval_env,
            args.eval_episodes,
            device=device,
            seed=eval_seed,
            agent_keys=AGENT_KEYS,
        )
        log["evaluations"].append(
            {
                "epoch": epoch,
                "val_success_rate": summary.success_rate,
                "val_mean_return": summary.mean_return,
                "val_mean_length": summary.mean_length,
                "val_mse_mean": val_mse,
            }
        )
        print(
            f"epoch {epoch:4d}/{args.epochs} | val_mse {val_mse:.5f} | "
            f"held-out success {summary.success_rate:.3f} | "
            f"mean length {summary.mean_length:.1f} | {time.time() - start:.0f}s",
            flush=True,
        )
        return summary.success_rate, val_mse

    best_state: dict[str, dict[str, torch.Tensor]] | None = None
    best_epoch = -1
    best_success = -1.0
    best_eval_mse = float("inf")
    best_curve_mse = float("inf")
    curve_state: dict[str, dict[str, torch.Tensor]] | None = None
    curve_epoch = -1
    stale = 0
    start = time.time()
    epochs_run = 0
    last_eval_epoch = -1
    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        order = torch.randperm(train_total, device=device)
        train_batch_mse = {role: [] for role in AGENT_KEYS}
        for batch in range(batches):
            indices = order[batch * args.batch_size : (batch + 1) * args.batch_size]
            if indices.numel() < 2:
                continue
            for index, (role, policy, optimizer) in enumerate(
                zip(AGENT_KEYS, policies, optimizers)
            ):
                features = _normalized(obs_train[role][indices], mean[role], std[role])
                loss = bc_mse(policy, features, act_train[role][indices])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_batch_mse[role].append(float(loss.detach()))
        val_mse: dict[str, float] = {}
        val_per_dim: dict[str, list[float]] = {}
        with torch.no_grad():
            for role, policy in zip(AGENT_KEYS, policies):
                error = (
                    policy.actor_mean(_normalized(obs_val[role], mean[role], std[role]))
                    - act_val[role]
                )
                val_mse[role] = float((error**2).mean())
                val_per_dim[role] = [float(value) for value in (error**2).mean(dim=0)]
        mean_val_mse = float(np.mean(list(val_mse.values())))
        log["epochs"].append(
            {
                "epoch": epoch,
                "train_mse": {role: float(np.mean(train_batch_mse[role])) for role in AGENT_KEYS},
                "val_mse": val_mse,
                "val_mse_per_dim": val_per_dim,
                "val_mse_mean": mean_val_mse,
            }
        )
        if mean_val_mse < best_curve_mse - 1e-9:
            best_curve_mse = mean_val_mse
            curve_epoch = epoch
            curve_state = _clone_states(policies)
            stale = 0
        else:
            stale += 1
        evaluate = (args.eval_every > 0 and epoch % args.eval_every == 0) or epoch == args.epochs
        if evaluate:
            success, evaluated_mse = run_held_out_eval(epoch, mean_val_mse)
            last_eval_epoch = epoch
            if success > best_success or (
                success == best_success and evaluated_mse < best_eval_mse - 1e-9
            ):
                best_success = success
                best_eval_mse = evaluated_mse
                best_epoch = epoch
                best_state = _clone_states(policies)
        elif epoch % 10 == 0:
            print(
                f"epoch {epoch:4d}/{args.epochs} | train_mse "
                f"{np.mean([np.mean(train_batch_mse[role]) for role in AGENT_KEYS]):.5f} | "
                f"val_mse {mean_val_mse:.5f} | {time.time() - start:.0f}s",
                flush=True,
            )
        if stale >= args.patience:
            print(
                f"early stop: no validation-MSE improvement for {args.patience} epochs "
                f"(best curve MSE {best_curve_mse:.5f} at epoch {curve_epoch})",
                flush=True,
            )
            break
    # Early stop can land between two success evaluations: always evaluate the final weights
    # once so the checkpoint choice is not based on stale weights.
    if last_eval_epoch != epochs_run:
        final_mse = log["epochs"][-1]["val_mse_mean"]
        success, evaluated_mse = run_held_out_eval(epochs_run, final_mse)
        last_eval_epoch = epochs_run
        if success > best_success or (success == best_success and evaluated_mse < best_eval_mse):
            best_success = success
            best_eval_mse = evaluated_mse
            best_epoch = epochs_run
            best_state = _clone_states(policies)
    eval_env.close()
    if best_state is None:
        # No success evaluation happened (impossible with the default flags, kept safe).
        best_state = curve_state if curve_state is not None else _clone_states(policies)
        best_epoch = curve_epoch
        best_eval_mse = best_curve_mse
    _load_states(best_state, policies)
    for index, policy in enumerate(policies):
        fold_obs_normalization(policy, mean[AGENT_KEYS[index]], std[AGENT_KEYS[index]])
    run_dir = Path(args.runs_dir) / args.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "latest.pt"
    save_checkpoint(
        checkpoint_path,
        policies,
        optimizers,
        cfg,
        update=epochs_run,
        global_step=int(obs.shape[0]),
        eval_metrics={
            "val_action_mse": float(best_eval_mse),
            "val_success_rate": float(best_success),
            "best_epoch": float(best_epoch),
        },
        agent_keys=AGENT_KEYS,
    )
    (run_dir / "bc_log.json").write_text(json.dumps(log, indent=2) + "\n")
    config = copy.deepcopy(dict(cfg.__dict__))
    config.update(
        {
            "bc": log["hyperparameters"],
            "checkpoint_epoch": epochs_run,
            "best_epoch": best_epoch,
            "best_val_mse": float(best_eval_mse),
            "best_val_success_rate": float(best_success),
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Keep a stable `bc.pt` alias next to `latest.pt` (the arm BC line's checkpoint name).
    shutil.copyfile(checkpoint_path, run_dir / "bc.pt")
    print(
        f"wrote {checkpoint_path} | epochs {epochs_run} (best epoch {best_epoch}) | "
        f"best val_mse {best_eval_mse:.5f} | best held-out success {best_success:.3f} | "
        f"{time.time() - start:.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
