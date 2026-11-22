"""[cloud] Behaviour + partner-prediction measurement harness of FaceTurnSkills-v0.

Two measurements over the FIXED pif-paf plan (`R U R' U'`, 6 executable moves):

1. **Behaviour** -- run a policy on the fixed plan under each partner-visibility mode and
   report: success rate, decisions to success, flips executed, timeouts, useless-action
   rates and PHASE DURATIONS (decisions spent on each primitive, per hand). The policy is
   either the scripted conductor (`--policy conductor`, the skill-level upper bound / env
   validation) or IPPO checkpoints (`--policy checkpoints --checkpoints ...`).
2. **Prediction** -- from the same rollouts (the `oracle` run exposes the partner's
   primitive stream in `info`) fit ridge probes for the PARTNER'S NEXT PRIMITIVE:
   - from the agent's OWN observation block only (the information a `none`-mode policy
     has), and
   - from own + the oracle partner block (the information upper bound).
   Report top-1 accuracy vs the majority-class chance, plus ANTICIPATION of the partner's
   next flip / turn / release at horizons 1/2/4 decisions (ROC-AUC and 10-bin ECE).

BI-JEPA CAVEAT (read this): the skill-level `predicted` partner block in
`callosum.experiments.arm.skills_agent_obs` is a ZERO STUB of the contract width -- the real
Bi-JEPA latent/predictor of `callosum.common.bijepa` is NOT wired into the skill env or the
IPPO trainer yet (only the joint-level line integrates Bi-JEPA). The harness therefore
measures the OBSERVABLE partner-information gap (own obs vs oracle info) and the ceiling a
Bi-JEPA latent could fill; it does NOT evaluate a trained Bi-JEPA predictor. A
`--partner-modes predicted` run drives a policy whose latent block is all zeros and the
output says so.

Usage (pod, GPU sim; CPU works for 1 env but is ~5-10x slower):
```
/root/.callosum/venv/bin/python scripts/experiments/arm/measure_skills_coordination.py \
  --policy conductor --partner-modes none oracle --episodes 8 --num-envs 4 \
  --plan-notation "R U R' U'" --max-episode-decisions 600 \
  --out results_skills_measure.json
```
Checkpoints (IPPO trainer run dirs or .pt files):
```
... --policy checkpoints --checkpoints 'runs/skills_s1' 'runs/skills_s2' \
    --partner-modes none oracle predicted
```

Never run this on the Mac (needs the simulator). Pure index/metric helpers are unit-tested
on macOS where they have no simulator dependency.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from callosum.experiments.arm.skills import (
    PARTNER_VISIBILITY_MODES,
    Role,
    num_primitives,
)
from callosum.experiments.arm.skills_obs import (
    INFO_PRIMITIVE_HOLDER,
    INFO_PRIMITIVE_ROTATOR,
    INFO_SUCCESS,
    INFO_USELESS_HOLDER,
    INFO_USELESS_ROTATOR,
    own_block_size,
)
from callosum.experiments.arm.skills_plans import FACES  # noqa: F401  (kept for the docs example)

HOLDER = Role.HOLDER
ROTATOR = Role.ROTATOR
PARTNER_KEY = {HOLDER: INFO_PRIMITIVE_ROTATOR, ROTATOR: INFO_PRIMITIVE_HOLDER}
EVENT_SETS = {
    "flip": ("flip90", "flip180"),
    "turn": ("turn90", "turn180"),
    "release": ("release",),
}
"""Partner events anticipated at a horizon: primitive names of the partner's registry."""


def _primitive_names(role: Role) -> tuple[str, ...]:
    from callosum.experiments.arm.skills import HOLDER_PRIMITIVES, ROTATOR_PRIMITIVES

    registry = HOLDER_PRIMITIVES if role is HOLDER else ROTATOR_PRIMITIVES
    return tuple(primitive.name for primitive in registry)


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _rows(infos: dict, n: int) -> list[dict]:
    """Per-env dicts of the batched info dict of one decision."""
    return [
        {
            key: (
                _np(value).reshape(-1)[e] if hasattr(value, "reshape") or np.ndim(value) else value
            )
            for key, value in infos.items()
        }
        for e in range(n)
    ]


# --- probes (pure numpy; unit-tested on macOS) ---------------------------------------------


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score with the TRAIN mean/std (constant columns become zero columns)."""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (test - mean) / std


def ridge_fit(features: np.ndarray, targets: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Closed-form ridge regression with an intercept column appended; `(D+1, K)` weights."""
    design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ targets)


def ridge_predict(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Predictions of a `ridge_fit` model for `(N, D)` features."""
    return np.concatenate([features, np.ones((features.shape[0], 1))], axis=1) @ weights


def majority_share(labels: np.ndarray) -> float:
    """Chance accuracy of a classifier that always predicts the most frequent class."""
    if labels.size == 0:
        return float("nan")
    return Counter(labels.tolist()).most_common(1)[0][1] / labels.size


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC from ranks (ties averaged); NaN when only one class is present."""
    if labels.size == 0 or labels.min() == labels.max():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks of tied scores, so a constant predictor gives AUC 0.5.
    sorted_scores = scores[order]
    start = 0
    for end in range(1, len(sorted_scores) + 1):
        if end == len(sorted_scores) or sorted_scores[end] != sorted_scores[start]:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def calibration_ece(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of sigmoid(scores) against binary labels."""
    if labels.size == 0:
        return float("nan")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (probabilities >= lo) & (probabilities < hi if hi < 1.0 else probabilities <= 1.0)
        if not mask.any():
            continue
        total += mask.sum() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(total / labels.size)


def probe_next_primitive(
    features_by_kind: dict[str, np.ndarray],
    labels: np.ndarray,
    train_mask: np.ndarray,
    n_classes: int,
) -> dict:
    """Top-1 next-primitive accuracy of a ridge one-hot probe, per feature kind."""
    out = {}
    for kind, features in features_by_kind.items():
        train, test = features[train_mask], features[~train_mask]
        y_train, y_test = labels[train_mask], labels[~train_mask]
        if len(np.unique(y_train)) < 2 or len(y_test) == 0:
            out[kind] = {"accuracy": float("nan"), "chance": float("nan")}
            continue
        train_s, test_s = standardize(train.astype(float), test.astype(float))
        targets = np.eye(n_classes, dtype=float)[y_train]
        weights = ridge_fit(train_s, targets)
        predictions = ridge_predict(weights, test_s).argmax(axis=1)
        out[kind] = {
            "accuracy": float((predictions == y_test).mean()),
            "chance": float(majority_share(y_test)),
        }
    return out


def probe_events(
    features_by_kind: dict[str, np.ndarray],
    event_labels: dict[str, np.ndarray],
    train_mask: np.ndarray,
) -> dict:
    """Anticipation metrics (AUC + ECE) of binary partner events, per feature kind/event."""
    out: dict = {}
    for event, labels in event_labels.items():
        out[event] = {}
        for kind, features in features_by_kind.items():
            train, test = features[train_mask], features[~train_mask]
            y_train, y_test = labels[train_mask], labels[~train_mask]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                out[event][kind] = {"auc": float("nan"), "ece": float("nan")}
                continue
            train_s, test_s = standardize(train.astype(float), test.astype(float))
            weights = ridge_fit(train_s, y_train.astype(float)[:, None])
            scores = ridge_predict(weights, test_s)[:, 0]
            out[event][kind] = {
                "auc": roc_auc(scores, y_test),
                "ece": calibration_ece(scores, y_test),
                "base_rate": float(y_test.mean()),
            }
    return out


# --- rollouts ------------------------------------------------------------------------------


def _make_env(args: argparse.Namespace, partner_mode: str):
    """The single state-observation skills env with the fixed plan and decision budget."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers the stock ManiSkill envs)

    import callosum.experiments.arm.skills_env  # noqa: F401  (registers FaceTurnSkills-v0)
    from callosum.experiments.arm.sim_compat import prepare_sim_backend
    from callosum.experiments.arm.skills_config import FaceTurnSkillsConfig

    prepare_sim_backend(args.sim_backend)
    config = FaceTurnSkillsConfig(
        plan_notation=args.plan_notation,
        max_episode_decisions=int(args.max_episode_decisions),
    )
    env = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        obs_mode="state",
        sim_backend=args.sim_backend,
        render_backend="none",
        render_mode=None,
        partner_mode=partner_mode,
        skills_config=config,
    )
    return env, config


def _load_actor(args: argparse.Namespace, base_env, builder):
    """`act(inputs, infos) -> (holder_ids, rotator_ids)` for the requested policy.

    `base_env` is the UNWRAPPED env (`env.unwrapped`): the gym `TimeLimit` wrapper does
    not forward `num_envs`/`device`, so the raw env is the honest source of those.
    """
    from callosum.experiments.arm.skills_conductor import SkillsConductor

    if args.policy == "conductor":
        conductors = [SkillsConductor() for _ in range(base_env.num_envs)]

        def act(inputs, infos):
            decision_info = _rows(infos, base_env.num_envs)
            holder, rotator = [], []
            for e, conductor in enumerate(conductors):
                holder.append(conductor.act("holder", inputs[0][e].cpu().numpy(), decision_info[e]))
                rotator.append(
                    conductor.act("rotator", inputs[1][e].cpu().numpy(), decision_info[e])
                )
            return holder, rotator, conductors

        return act

    if args.policy == "checkpoints":
        import torch

        from callosum.experiments.arm import skills_eval

        checkpoints = skills_eval.expand_checkpoints(args.checkpoints or [])
        if len(checkpoints) < 1:
            raise SystemExit("--policy checkpoints needs --checkpoints")
        holder_ckpt = checkpoints[0]
        rotator_ckpt = checkpoints[1] if len(checkpoints) > 1 else checkpoints[0]
        holder, rotator = skills_eval._load_pair(
            holder_ckpt[1], rotator_ckpt[1], base_env.device, builder.obs_dims
        )

        def act(inputs, infos):
            del infos
            with torch.no_grad():
                holder_ids = holder.get_action(inputs[0], deterministic=True).tolist()
                rotator_ids = rotator.get_action(inputs[1], deterministic=True).tolist()
            return holder_ids, rotator_ids, None

        return act

    raise SystemExit(f"unknown policy {args.policy!r}")


def collect_rollout(args: argparse.Namespace, partner_mode: str, label: str) -> dict:
    """Run `args.episodes` episodes; return behaviour records and probe samples."""
    import torch

    from callosum.experiments.arm.skills_agent_obs import SkillsObsBuilder

    env, config = _make_env(args, partner_mode)
    base_env = env.unwrapped
    try:
        n = base_env.num_envs
        device = base_env.device
        uids = tuple(env.action_space.spaces)
        obs, infos = env.reset(seed=args.seed)
        builder = SkillsObsBuilder.from_env_obs(
            env.unwrapped.get_obs(unflattened=True), obs, partner_mode, uids
        )
        own_sizes = [own_block_size(num_primitives(role)) for role in (HOLDER, ROTATOR)]
        holder_primitive = _primitive_names(HOLDER)
        rotator_primitive = _primitive_names(ROTATOR)
        names = {HOLDER: holder_primitive, ROTATOR: rotator_primitive}
        flip_ids = {i for i, name in enumerate(holder_primitive) if name in EVENT_SETS["flip"]}
        act = _load_actor(args, base_env, builder)

        successes = np.zeros(n, dtype=bool)
        decisions = np.zeros(n, dtype=np.int64)
        flips = np.zeros(n, dtype=np.int64)
        useless_h = np.zeros(n, dtype=np.int64)
        useless_r = np.zeros(n, dtype=np.int64)
        phase_counts: dict[str, Counter] = defaultdict(Counter)
        previous_holder = np.full(n, -1, dtype=np.int64)
        records: list[dict] = []
        # Probe samples, per agent role: features, partner ids per decision, episode ids.
        samples = {
            role: {"own": [], "oracle": [], "partner": [], "episode": []}
            for role in (HOLDER, ROTATOR)
        }
        episode_counter = 0
        guard = (math.ceil(args.episodes / max(1, n)) + 2) * int(args.max_episode_decisions)
        step = 0
        with torch.no_grad():
            while len(records) < args.episodes and step < guard:
                inputs = builder(obs, infos)
                holder_ids, rotator_ids, conductors = act(inputs, infos)
                for role_index, role in enumerate((HOLDER, ROTATOR)):
                    samples[role]["own"].append(
                        inputs[role_index][:, : own_sizes[role_index]].cpu().numpy().copy()
                    )
                    samples[role]["oracle"].append(inputs[role_index].cpu().numpy().copy())
                    samples[role]["partner"].append(_np(infos[PARTNER_KEY[role]]).astype(np.int64))
                    samples[role]["episode"].append(
                        np.arange(episode_counter, episode_counter + n, dtype=np.int64)
                    )
                for e in range(n):
                    phase_counts[f"holder/{names[HOLDER][int(holder_ids[e])]}"]["count"] += 1
                    phase_counts[f"rotator/{names[ROTATOR][int(rotator_ids[e])]}"]["count"] += 1
                    if int(holder_ids[e]) in flip_ids and previous_holder[e] != int(holder_ids[e]):
                        flips[e] += 1
                    previous_holder[e] = int(holder_ids[e])
                action = {
                    uids[0]: torch.as_tensor(holder_ids, dtype=torch.long, device=device),
                    uids[1]: torch.as_tensor(rotator_ids, dtype=torch.long, device=device),
                }
                obs, _reward, terminated, truncated, infos = env.step(action)
                step += 1
                decisions += 1
                useless_h += _np(infos.get(INFO_USELESS_HOLDER, 0)).astype(np.int64)
                useless_r += _np(infos.get(INFO_USELESS_ROTATOR, 0)).astype(np.int64)
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
                                "flips": int(flips[e]),
                                "useless_holder": int(useless_h[e]),
                                "useless_rotator": int(useless_r[e]),
                            }
                        )
                        successes |= ok
                        decisions[e] = 0
                        flips[e] = 0
                        useless_h[e] = 0
                        useless_r[e] = 0
                        episode_counter += 1
                    # A finished raw ManiSkill episode keeps reporting `terminated` on
                    # further steps (no vector-env auto-reset): reset explicitly, or every
                    # leftover step would be recorded as a one-decision "success episode".
                    obs, infos = env.reset(seed=args.seed + episode_counter)
                    previous_holder = np.full(n, -1, dtype=np.int64)
                    if conductors is not None:
                        from callosum.experiments.arm.skills_conductor import SkillsConductor

                        conductors[:] = [SkillsConductor() for _ in range(n)]
        if len(records) < args.episodes:
            print(
                f"WARNING: {label}: only {len(records)} of {args.episodes} episodes finished "
                f"within the step guard ({step} decisions; budget "
                f"{config.max_episode_decisions})"
            )
        probe = _build_probe(args, samples, partner_mode)
        return {
            "label": label,
            "partner_mode": partner_mode,
            "plan_notation": config.plan_notation,
            "plan_length": config.plan_length,
            "max_episode_decisions": config.max_episode_decisions,
            "records": records,
            "phase_counts": {key: value["count"] for key, value in phase_counts.items()},
            "probe": probe,
        }
    finally:
        env.close()


def _build_probe(args: argparse.Namespace, samples: dict, partner_mode: str) -> dict | None:
    """Fit the next-primitive / anticipation probes from the collected `oracle` samples."""
    if not args.probe:
        return None
    if partner_mode != "oracle":
        return {"skipped": "probes need the oracle partner stream (--partner-modes oracle)"}
    result: dict = {"events": {}, "next_primitive": {}}
    for role in (HOLDER, ROTATOR):
        data = samples[role]
        if len(data["partner"]) < 2:
            continue
        features = {
            "own": np.concatenate(data["own"], axis=0),
            "oracle": np.concatenate(data["oracle"], axis=0),
        }
        partner = np.concatenate(data["partner"], axis=0)
        episode = np.concatenate(data["episode"], axis=0)
        partner_role = ROTATOR if role is HOLDER else HOLDER
        names = _primitive_names(partner_role)
        n_classes = num_primitives(partner_role)
        # Episode split: first 60% of episodes train, the rest validate.
        unique_episodes = np.unique(episode)
        split = max(1, round(0.6 * len(unique_episodes)))
        train_episodes = set(unique_episodes[:split].tolist())
        train_mask = np.array([e in train_episodes for e in episode])
        # Next decision's partner primitive, within one episode (the last row is dropped).
        next_partner = np.full_like(partner, -1)
        next_partner[:-1] = partner[1:]
        keep = episode[:-1] == episode[1:]
        kept_features = {kind: value[:-1][keep] for kind, value in features.items()}
        result["next_primitive"][role.value] = probe_next_primitive(
            kept_features, next_partner[:-1][keep], train_mask[:-1][keep], n_classes
        )
        # Anticipation: the partner's primitive h decisions later is in an event set.
        for horizon in args.probe_horizons:
            if horizon >= len(partner):
                continue
            aligned = episode[:-horizon] == episode[horizon:]
            future = partner[horizon:]
            mask = train_mask[:-horizon][aligned]
            valid_features = {kind: value[:-horizon][aligned] for kind, value in features.items()}
            for event, event_names in EVENT_SETS.items():
                ids = [i for i, name in enumerate(names) if name in event_names]
                if not ids:
                    continue  # this partner registry has no such primitive
                labels = np.isin(future[aligned], ids).astype(int)
                metrics = probe_events(valid_features, {role.value: labels}, mask)
                result["events"].setdefault(f"h{horizon}/{event}", {}).update(metrics)
    return result


# --- reporting -----------------------------------------------------------------------------


def summarize_records(records: list[dict]) -> dict:
    """Aggregate episode records into the behaviour numbers of the report."""
    if not records:
        return {}
    successes = [record for record in records if record["success"]]
    decisions = np.array([record["decisions"] for record in records], dtype=float)
    return {
        "episodes": len(records),
        "success_rate": len(successes) / len(records),
        "timeout_rate": float(np.mean([record["timeout"] for record in records])),
        "decisions_mean": float(decisions.mean()),
        "decisions_to_success_mean": (
            float(np.mean([record["decisions"] for record in successes])) if successes else None
        ),
        "flips_mean": float(np.mean([record["flips"] for record in records])),
        "useless_holder_per_decision": float(
            sum(record["useless_holder"] for record in records) / max(decisions.sum(), 1.0)
        ),
        "useless_rotator_per_decision": float(
            sum(record["useless_rotator"] for record in records) / max(decisions.sum(), 1.0)
        ),
    }


def _print_summary(summary: dict, phase_counts: dict, n_decisions: int) -> None:
    print(
        f"  episodes {summary['episodes']} | success {summary['success_rate']:.3f} | "
        f"timeout {summary['timeout_rate']:.3f} | "
        f"decisions {summary['decisions_mean']:.1f}"
        + (
            f" (success {summary['decisions_to_success_mean']:.1f})"
            if summary["decisions_to_success_mean"] is not None
            else ""
        )
        + f" | flips {summary['flips_mean']:.1f} | useless h/r "
        f"{summary['useless_holder_per_decision']:.3f}/"
        f"{summary['useless_rotator_per_decision']:.3f}"
    )
    groups: dict[str, list[str]] = defaultdict(list)
    for key in sorted(phase_counts):
        role, name = key.split("/", 1)
        groups[role].append(f"{name} {phase_counts[key] / max(n_decisions, 1):.2f}")
    for role, parts in groups.items():
        print(f"  phase decisions/decision [{role}]: {', '.join(parts)}")


def main(argv: list[str] | None = None) -> None:
    if sys.platform == "darwin":
        print("this harness needs the simulator; run it on the compute box")
        return
    sys.stdout.reconfigure(line_buffering=True)
    args = _parse_args(argv)
    payload: dict = {"config": dict(vars(args)), "runs": {}}
    for partner_mode in args.partner_modes:
        if partner_mode in ("predicted", "predicted-direct"):
            print(
                f"WARNING: partner mode {partner_mode!r} supplies a ZERO stub block "
                "(no Bi-JEPA latent is wired into the skill env/trainer yet): the run "
                "measures a no-information policy input, not a predictor."
            )
        for seed in args.seeds:
            args.seed = int(seed)
            label = f"{args.policy}_{partner_mode}_s{seed}"
            print(f"run {label}", flush=True)
            result = collect_rollout(args, partner_mode, label)
            summary = summarize_records(result["records"])
            payload["runs"][label] = {
                "behaviour": summary,
                "phase_counts": result["phase_counts"],
                "probe": result["probe"],
                "config": {
                    "policy": args.policy,
                    "partner_mode": partner_mode,
                    "seed": seed,
                    "plan_notation": result["plan_notation"],
                    "plan_length": result["plan_length"],
                    "max_episode_decisions": result["max_episode_decisions"],
                    "episodes": args.episodes,
                },
            }
            if summary:
                _print_summary(
                    summary,
                    result["phase_counts"],
                    int(sum(record["decisions"] for record in result["records"])),
                )
            if result["probe"]:
                _print_probe(result["probe"])
    Path(args.out).write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {args.out}")
    print(
        "NOTE: 'predicted' partner input is a zero stub in this repository -- the Bi-JEPA "
        "predictor is not wired into the skill line, so no true Bi-JEPA numbers are reported."
    )


def _print_probe(probe: dict) -> None:
    if "skipped" in probe:
        print(f"  probe skipped: {probe['skipped']}")
        return
    for role, values in probe.get("next_primitive", {}).items():
        parts = [
            f"{kind} acc {data['accuracy']:.3f} (chance {data['chance']:.3f})"
            for kind, data in values.items()
            if not math.isnan(data["accuracy"])
        ]
        print(f"  probe next partner primitive [{role}]: " + ("; ".join(parts) or "n/a"))
    for event, per_role in probe.get("events", {}).items():
        for role, kinds in per_role.items():
            parts = [
                f"{kind} AUC {data['auc']:.3f} ECE {data['ece']:.3f} "
                f"(base {data.get('base_rate', float('nan')):.3f})"
                for kind, data in kinds.items()
                if not math.isnan(data["auc"])
            ]
            if parts:
                print(f"  probe {event} [{role}]: " + "; ".join(parts))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-id", default="FaceTurnSkills-v0")
    parser.add_argument("--policy", choices=("conductor", "checkpoints"), default="conductor")
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument(
        "--partner-modes",
        nargs="+",
        choices=PARTNER_VISIBILITY_MODES,
        default=["none", "oracle"],
        help="Partner-visibility modes to evaluate. (default: none oracle)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Reset seeds.")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--plan-notation", default="R U R' U'", help="Fixed plan (pif-paf). (default: %(default)r)"
    )
    parser.add_argument("--max-episode-decisions", type=int, default=600)
    parser.add_argument("--sim-backend", default="gpu", help="([cloud] gpu).")
    parser.add_argument("--probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--probe-horizons",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="Anticipation horizons in decisions. (default: 1 2 4)",
    )
    parser.add_argument("--out", default="results_skills_measure.json")
    args = parser.parse_args(argv)
    if args.policy == "checkpoints" and not args.checkpoints:
        parser.error("--policy checkpoints needs --checkpoints")
    return args


if __name__ == "__main__":
    main()
