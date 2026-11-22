"""[cloud] Collect scripted-conductor demos for the skill-level line and train categorical BC.

The must-have E1 check of the skill-level (FaceTurnSkills-v0) line: two SO-ARM101 agents
each pick ONE discrete primitive per decision, so the policies are categorical
(`callosum.common.categorical.CategoricalPolicy`) and behaviour cloning is plain
cross-entropy of the conductor's primitive ids under the per-agent inputs
(`callosum.experiments.arm.skills_agent_obs.SkillsObsBuilder`).

Pipeline (all in one process, one pod_job):
  1. COLLECT: run the scripted conductor on a FIXED plan (`--plan-notation`, default the
     one-move `"U"` = turn the up face by 90 deg) for a range of seeds; save per-decision
     `(holder obs, rotator obs, holder primitive id, rotator primitive id)` of SUCCESSFUL
     episodes only (failure trajectories are never used for BC).
  2. TRAIN: two categorical policies, cross-entropy to the conductor's ids, episode-wise
     train/validation split, validation accuracy per epoch.
  3. SAVE: a format-1 weights checkpoint via `common.checkpoint.save_weights_checkpoint`
     (the format `arm/trainer.py --checkpoint` and `skills_eval` load: per-agent state
     dicts, obs_dims, obs_fields, agent_uids).
  4. EVAL: run the LEARNED policies (argmax logits) for `--eval-episodes` episodes on seeds
     OUTSIDE the collection seeds; report success rate / decisions / timeouts.

The conductor itself remains the upper bound and the demo source; the learned policy is the
claim ("learned skill-level policy for one face turn"). `--plan-notation "R U R' U'"` runs
the same pipeline on the full pif-paf plan (six executable moves), which needs the flips to
work first.

Usage (pod, repo root, CPU sim is fine and faster than the one-env GPU fetch path):
```
/root/.callosum/venv/bin/python scripts/experiments/arm/train_skills_bc.py \
  --plan-notation U --collect-episodes 32 --eval-episodes 64 \
  --sim-backend cpu --out runs/skills_bc_U --exp-name skills_bc_U
```

Never run this on the Mac (needs the simulator); the pure torch parts would run, but the
collection does not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from callosum.common.categorical import CategoricalPolicy
from callosum.common.checkpoint import save_weights_checkpoint
from callosum.experiments.arm.skills import Role, num_primitives
from callosum.experiments.arm.skills_conductor import SkillsConductor
from callosum.experiments.arm.skills_config import FaceTurnSkillsConfig
from callosum.experiments.arm.skills_obs import INFO_SUCCESS

ROLES = (Role.HOLDER, Role.ROTATOR)
ROLE_NAMES = ("holder", "rotator")
N_ACTIONS = tuple(num_primitives(role) for role in ROLES)


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _extra_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-id", default="FaceTurnSkills-v0")
    parser.add_argument(
        "--plan-notation",
        default="U",
        help='Fixed plan in standard face-turn notation. (default: "U" = one 90 deg turn)',
    )
    parser.add_argument(
        "--partner-mode",
        default="oracle",
        help="Partner-visibility mode the policies are trained/evaluated in. (default: oracle)",
    )
    parser.add_argument("--collect-episodes", type=int, default=32)
    parser.add_argument("--collect-seed-start", type=int, default=0)
    parser.add_argument(
        "--max-collect-seeds",
        type=int,
        default=40,
        help="Stop collecting after this many reset seeds even if fewer successes appeared.",
    )
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-seed-start", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0, help="Torch seed of the BC training.")
    parser.add_argument("--max-episode-decisions", type=int, default=200)
    parser.add_argument("--sim-backend", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("runs/skills_bc"))
    parser.add_argument("--exp-name", default="skills_bc")


def _make_env(args: argparse.Namespace):
    """The fixed-plan skills env (one process, state observations, no renderer)."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers the stock ManiSkill envs)

    import callosum.experiments.arm.skills_env  # noqa: F401  (registers FaceTurnSkills-v0)
    from callosum.experiments.arm.sim_compat import prepare_sim_backend
    from callosum.experiments.arm.skills_agent_obs import SkillsObsBuilder

    prepare_sim_backend(args.sim_backend)
    config = FaceTurnSkillsConfig(
        plan_notation=args.plan_notation,
        max_episode_decisions=int(args.max_episode_decisions),
    )
    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="state",
        sim_backend=args.sim_backend,
        render_backend="none",
        render_mode=None,
        partner_mode=args.partner_mode,
        skills_config=config,
    )
    base = env.unwrapped
    obs, _infos = env.reset(seed=args.collect_seed_start)
    builder = SkillsObsBuilder.from_env_obs(
        base.get_obs(unflattened=True), obs, args.partner_mode, tuple(env.action_space.spaces)
    )
    args._obs_fields = [list(fields) for fields in builder.fields]
    args._agent_uids = list(env.action_space.spaces)
    return env, base, builder, config


def _rows(infos: dict, n: int) -> list[dict]:
    return [
        {
            key: (_np(value).reshape(-1)[e] if hasattr(value, "reshape") else value)
            for key, value in infos.items()
        }
        for e in range(n)
    ]


def rollout(args: argparse.Namespace, actor, *, seed_start: int, episodes_wanted: int):
    """Run `actor` and return `(successful_episodes, eval_records)`.

    `actor(inputs, infos) -> (holder_ids, rotator_ids)` gives one primitive id per env.
    Successful episodes are returned as sample dicts (`obs_holder`, `obs_rotator`,
    `act_holder`, `act_rotator`); every episode (successful or not) adds an eval record.
    """
    import torch

    env, base, builder, _config = _make_env(args)
    n = base.num_envs
    uids = tuple(env.action_space.spaces)
    conductor = SkillsConductor()
    successful: list[dict] = []
    records: list[dict] = []
    seed = seed_start
    seed_cap = seed_start + max(args.max_collect_seeds, episodes_wanted + 2)
    try:
        with torch.no_grad():
            while len(records) < episodes_wanted and seed < seed_cap:
                obs, infos = env.reset(seed=seed)
                buffers = [{"h": [], "r": [], "ah": [], "ar": []} for _ in range(n)]
                decisions = np.zeros(n, dtype=np.int64)
                step = 0
                while step < args.max_episode_decisions and len(records) < episodes_wanted:
                    inputs = builder(obs, infos)
                    info_rows = _rows(infos, n)
                    if actor is None:  # scripted conductor
                        holder_ids = [
                            conductor.act("holder", inputs[0][e].cpu().numpy(), info_rows[e])
                            for e in range(n)
                        ]
                        rotator_ids = [
                            conductor.act("rotator", inputs[1][e].cpu().numpy(), info_rows[e])
                            for e in range(n)
                        ]
                    else:
                        holder_ids, rotator_ids = actor(inputs, infos)
                    for e in range(n):
                        buffers[e]["h"].append(inputs[0][e].cpu().numpy().copy())
                        buffers[e]["r"].append(inputs[1][e].cpu().numpy().copy())
                        buffers[e]["ah"].append(int(holder_ids[e]))
                        buffers[e]["ar"].append(int(rotator_ids[e]))
                    action = {
                        uids[0]: torch.as_tensor(holder_ids, dtype=torch.long, device=base.device),
                        uids[1]: torch.as_tensor(rotator_ids, dtype=torch.long, device=base.device),
                    }
                    obs, _reward, terminated, truncated, infos = env.step(action)
                    step += 1
                    decisions += 1
                    success = _np(infos.get(INFO_SUCCESS, 0)).astype(bool)
                    done = _np(terminated).astype(bool) | _np(truncated).astype(bool)
                    if done.any():
                        for e in np.nonzero(done)[0]:
                            ok = bool(success[e])
                            records.append(
                                {
                                    "success": ok,
                                    "decisions": int(decisions[e]),
                                    "timeout": bool(not ok and _np(truncated)[e]),
                                }
                            )
                            if ok:
                                successful.append(
                                    {
                                        "obs_holder": np.stack(buffers[e]["h"]),
                                        "obs_rotator": np.stack(buffers[e]["r"]),
                                        "act_holder": np.asarray(buffers[e]["ah"], dtype=np.int64),
                                        "act_rotator": np.asarray(buffers[e]["ar"], dtype=np.int64),
                                        "seed": seed,
                                    }
                                )
                        # A finished (raw, unvectorised) ManiSkill episode keeps reporting
                        # `terminated` on every later step -- future steps belong to NO
                        # episode. Reset explicitly for the next episode instead of
                        # recording one bogus "success" record per leftover step.
                        obs, infos = env.reset(seed=seed + 1 + len(records))
                        buffers = [{"h": [], "r": [], "ah": [], "ar": []} for _ in range(n)]
                        decisions = np.zeros(n, dtype=np.int64)
                        conductor = SkillsConductor()
                        step = 0
                seed += 1
    finally:
        env.close()
    return successful, records


def train_bc(args: argparse.Namespace, episodes: list[dict]) -> tuple[list, dict]:
    """Cross-entropy BC of the two categorical actors; returns the policies and stats."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Episode-wise split so validation measures generalisation to unseen trajectories.
    order = np.random.permutation(len(episodes))
    n_val = max(1, round(args.val_fraction * len(episodes))) if len(episodes) > 2 else 0
    val_ids = set(order[:n_val].tolist())
    train_ids = [i for i in range(len(episodes)) if i not in val_ids]
    splits = {"train": train_ids, "val": sorted(val_ids)}

    def tensors(ids):
        data: dict = {}
        for role in ROLE_NAMES:
            data[f"{role}_obs"] = torch.as_tensor(
                np.concatenate([episodes[i][f"obs_{role}"] for i in ids]), dtype=torch.float32
            )
            data[f"{role}_act"] = torch.as_tensor(
                np.concatenate([episodes[i][f"act_{role}"] for i in ids]), dtype=torch.long
            )
        data["size"] = sum(len(episodes[i]["act_holder"]) for i in ids)
        return data

    data = {name: tensors(ids) for name, ids in splits.items()}
    obs_dims = [data["train"][f"{role}_obs"].shape[1] for role in ROLE_NAMES]
    policies = [CategoricalPolicy(obs_dim, n_act) for obs_dim, n_act in zip(obs_dims, N_ACTIONS)]
    optimizers = [
        torch.optim.Adam(policy.actor_logits.parameters(), lr=args.lr) for policy in policies
    ]
    loss_fn = torch.nn.CrossEntropyLoss()
    stats = {
        "obs_dims": obs_dims,
        "train_size": data["train"]["size"],
        "epochs": [],
        "best_val": 0.0,
    }
    best_state = [policy.state_dict() for policy in policies]
    for epoch in range(args.epochs):
        for policy in policies:
            policy.train()
        total_loss = 0.0
        for role, policy, optimizer in zip(ROLE_NAMES, policies, optimizers):
            obs_t, act_t = data["train"][f"{role}_obs"], data["train"][f"{role}_act"]
            permutation = torch.randperm(obs_t.shape[0])
            for start in range(0, len(permutation), args.batch_size):
                index = permutation[start : start + args.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(policy.actor_logits(obs_t[index]), act_t[index])
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * len(index)
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            val_accuracy = {}
            for role, policy in zip(ROLE_NAMES, policies):
                policy.eval()
                if data["val"]["size"]:
                    with torch.no_grad():
                        logits = policy.actor_logits(data["val"][f"{role}_obs"])
                        val_accuracy[role] = float(
                            (logits.argmax(-1) == data["val"][f"{role}_act"]).float().mean()
                        )
                else:
                    val_accuracy[role] = float("nan")
            stats["epochs"].append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / max(data["train"]["size"], 1),
                    "val_accuracy": val_accuracy,
                }
            )
            print(
                f"  epoch {epoch:3d}: train loss {total_loss / max(data['train']['size'], 1):.4f} "
                f"| val acc h/r {val_accuracy['holder']:.3f}/{val_accuracy['rotator']:.3f}",
                flush=True,
            )
            mean_val = float(np.nanmean(list(val_accuracy.values())))
            if mean_val >= stats["best_val"]:
                stats["best_val"] = mean_val
                best_state = [policy.state_dict() for policy in policies]
    for policy, state in zip(policies, best_state, strict=True):
        policy.load_state_dict(state)
        policy.eval()
    return policies, stats


def evaluate(args: argparse.Namespace, policies: list) -> dict:
    """Run the learned policies (argmax) on eval seeds outside the collection range."""

    def actor(inputs, infos):
        del infos
        with torch.no_grad():
            return (
                policies[0].get_action(inputs[0], deterministic=True).tolist(),
                policies[1].get_action(inputs[1], deterministic=True).tolist(),
            )

    _samples, records = rollout(
        args, actor, seed_start=args.eval_seed_start, episodes_wanted=args.eval_episodes
    )
    successes = [record for record in records if record["success"]]
    decisions = np.array([record["decisions"] for record in records], dtype=float)
    return {
        "episodes": len(records),
        "success_rate": len(successes) / max(len(records), 1),
        "timeout_rate": float(np.mean([record["timeout"] for record in records]))
        if records
        else 0.0,
        "decisions_mean": float(decisions.mean()) if len(records) else None,
        "decisions_to_success_mean": (
            float(np.mean([record["decisions"] for record in successes])) if successes else None
        ),
    }


def main(argv: list[str] | None = None) -> None:
    if sys.platform == "darwin":
        print("this pipeline needs the simulator; run it on the compute box")
        return
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _extra_flags(parser)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(
        f"collect: plan {args.plan_notation!r} | mode {args.partner_mode} | "
        f"up to {args.collect_episodes} successful episodes from seeds "
        f"{args.collect_seed_start}..",
        flush=True,
    )
    episodes, collect_records = rollout(
        args, None, seed_start=args.collect_seed_start, episodes_wanted=args.collect_episodes
    )
    print(
        f"  collected {len(episodes)} successful of {len(collect_records)} episodes "
        f"({time.time() - started:.1f}s)",
        flush=True,
    )
    if len(episodes) < 2:
        raise SystemExit(f"not enough successful conductor episodes ({len(episodes)}) for BC")
    policies, bc_stats = train_bc(args, episodes)
    checkpoint = args.out / "bc.pt"
    save_weights_checkpoint(
        checkpoint,
        list(policies),
        {
            "partner_mode": args.partner_mode,
            "plan_notation": args.plan_notation,
            "env_id": args.env_id,
        },
        bc_stats["obs_dims"],
        getattr(args, "_agent_uids", ["so101_pg-0", "so101_pg-1"]),
        getattr(args, "_obs_fields", [["own/*", "partner/*"], ["own/*", "partner/*"]]),
        extra={"bc": {k: v for k, v in bc_stats.items() if k != "obs_dims"}},
    )
    print(f"  saved {checkpoint} (val acc {bc_stats['best_val']:.3f})", flush=True)
    evaluation = evaluate(args, policies)
    report = {
        "plan_notation": args.plan_notation,
        "partner_mode": args.partner_mode,
        "collect": {
            "episodes": len(collect_records),
            "success_rate": float(np.mean([r["success"] for r in collect_records])),
            "successful_kept": len(episodes),
        },
        "bc": {
            "train_size": bc_stats["train_size"],
            "best_val_accuracy": bc_stats["best_val"],
            "history": bc_stats["epochs"],
        },
        "eval": evaluation,
        "checkpoint": str(checkpoint),
        "elapsed_s": time.time() - started,
    }
    (args.out / "bc_report.json").write_text(json.dumps(report, indent=2, default=float))
    print("=== BC summary ===")
    print(f"plan                 : {args.plan_notation!r} (mode {args.partner_mode})")
    print(f"collect episodes     : {len(collect_records)} ({len(episodes)} successful kept)")
    print(f"BC train size        : {bc_stats['train_size']} decisions")
    print(f"BC best val accuracy : {bc_stats['best_val']:.3f}")
    print(
        f"LEARNED eval         : {evaluation['success_rate']:.3f} over "
        f"{evaluation['episodes']} episodes | decisions "
        f"{evaluation['decisions_mean']:.1f} | timeouts {evaluation['timeout_rate']:.3f}"
    )
    print(f"checkpoint           : {checkpoint}")
    print(f"report               : {args.out / 'bc_report.json'}")


if __name__ == "__main__":
    main()
