"""Evaluate a trained `wrench_ppo` checkpoint on the reduced two-agent coop suite.

Loads a `latest.pt` checkpoint of `callosum.experiments.cube_wrench.train` (weights + config) and runs
the two deterministic (greedy) policies for `--episodes` episodes of the checkpoint's coop
task (`--task cartpole`/`pendulum`/`toss` is recorded in the checkpoint config; the cube has
its own `callosum/experiments/cube_wrench/eval.py`). The metrics the task-suite comparison needs are
written as JSON: success rate, steps / steps to success and the task diagnostics (the final
pole angle for the cart-pole, the achieved amplitude and completed hold cycles for the
pendulum, the pass count for the toss-catch).

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

The evaluation backend is selectable: `--backend warp` (the default on a CUDA host) runs the
episodes as one batched `CoopBatchGPU` (`mujoco_warp`), `--backend cpu` uses the single-instance
MuJoCo loop (parity/CI). Both produce the same JSON report; the warp path evaluates many
held-out episodes in parallel, which is what makes the 128-episode held-out protocol cheap.

Run it on the server (the plan forbids training/sim on the laptop outside the small smoke
checks):

    python -m callosum.experiments.coop_tasks.eval --checkpoint runs/coop_cartpole_oracle_s0/latest.pt \\
        --input-mode oracle --episodes 128 --seed 0 --out results_coop_cartpole_oracle_s0.json
    python -m callosum.experiments.coop_tasks.eval --backend cpu ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from callosum.common.metrics import EpisodeRecord, success_rate, time_to_success_stats
from callosum.experiments.coop_tasks import make_coop_env
from callosum.experiments.coop_tasks.logic import task_diagnostic
from callosum.experiments.cube_wrench.ppo_config import (
    BACKENDS,
    INPUT_MODES,
    TASKS,
    cuda_visible,
)
from callosum.experiments.cube_wrench.train import (
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
        choices=("auto",) + BACKENDS,
        default="auto",
        help=(
            "evaluation backend: auto (default: warp when a CUDA device is visible, else cpu), "
            "warp (batched mujoco_warp GPU) or cpu (single-instance MuJoCo loop)"
        ),
    )
    parser.add_argument(
        "--device", default=None, help="CUDA device of the warp backend (default: cuda)"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="worlds of the batched warp evaluation (the CPU path ignores it)",
    )
    parser.add_argument("--out", default="results_coop.json", help="JSON output file")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error(f"--episodes must be >= 1, got {args.episodes}")
    if args.seed < 0:
        parser.error(f"--seed must be >= 0, got {args.seed}")
    if args.backend == "auto":
        args.backend = "warp" if cuda_visible() else "cpu"
    if args.backend == "warp" and args.device is None:
        args.device = "cuda"
    if args.backend == "cpu":
        args.device = args.device or "cpu"
    if args.num_envs < 1:
        parser.error(f"--num-envs must be >= 1, got {args.num_envs}")
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


def summarize(records: list[EpisodeRecord], infos: list[dict], task: str) -> dict:
    """The JSON-ready metric block of one evaluation (task-aware diagnostics)."""
    steps = np.asarray([record.decisions for record in records], dtype=np.float64)
    successes = sum(1 for record in records if record.success)
    time = time_to_success_stats(records)
    summary = {
        "episodes": len(records),
        "successes": successes,
        "failures": len(records) - successes,
        "success_rate": success_rate(records),
        "mean_diagnostic": float(np.mean([task_diagnostic(task, info) for info in infos])),
        "steps_mean": float(np.mean(steps)),
        "steps_to_success_mean": None if time is None else time[0],
        "steps_to_success_std": None if time is None else time[1],
    }
    if task == "cartpole":
        summary["mean_angle_deg"] = float(np.mean([info["angle_deg"] for info in infos]))
    if infos and "amplitude_deg" in infos[0]:
        summary["mean_amplitude_deg"] = float(np.mean([info["amplitude_deg"] for info in infos]))
        summary["mean_completed_cycles"] = float(
            np.mean([info["completed_cycles"] for info in infos])
        )
        summary["hold_cycles"] = float(infos[0]["hold_cycles"])
    if task == "toss":
        summary["mean_passes"] = float(np.mean([info["passes"] for info in infos]))
        summary["target_passes"] = float(infos[0]["target_passes"])
        # Honesty diagnostics: a success may only be granted while the ball is above the
        # floor, and the pass gate requires several control steps per counted pass, so the
        # success ball heights and the success step counts pin both down in the JSON report.
        success_ball_z = [info["ball_z"] for info in infos if info["success"]]
        success_steps = sorted(record.decisions for record in records if record.success)
        summary["mean_terminal_ball_z"] = float(np.mean([info["ball_z"] for info in infos]))
        summary["min_success_ball_z"] = float(min(success_ball_z)) if success_ball_z else None
        summary["min_success_steps"] = float(success_steps[0]) if success_steps else None
        summary["p10_success_steps"] = (
            float(success_steps[len(success_steps) // 10]) if success_steps else None
        )
        # Physical-paddle evidence: the plates are servo-driven rigid bodies now, so a held-out
        # report must show that their contact shells never let them interpenetrate. The
        # geometry is the SAT measure of `paddle_model.paddle_pair_geometry` (0 penetration
        # means no overlap); `pad_min_gap` is a lower bound of the measured plate distance.
        if "pad_penetration" in infos[0]:
            summary["max_pad_penetration"] = float(
                np.max([info["pad_penetration"] for info in infos])
            )
            summary["mean_pad_contact_steps"] = float(
                np.mean([info["pad_contact_steps"] for info in infos])
            )
            summary["episodes_with_pad_contact"] = float(
                np.mean([1.0 if info["pad_contact_steps"] > 0.0 else 0.0 for info in infos])
            )
            summary["min_pad_gap"] = float(np.min([info["pad_min_gap"] for info in infos]))
    return summary


@torch.no_grad()
def evaluate_warp(
    cfg,
    env_mode: str,
    latent_dim: int,
    policies: list[torch.nn.Module],
    builder: LatentInputBuilder | None,
    agent_keys: tuple[str, ...],
    args: argparse.Namespace,
    device: torch.device | str,
) -> tuple[list[EpisodeRecord], list[dict]]:
    """Greedy evaluation of `--episodes` held-out episodes on the batched GPU env.

    Mirrors `callosum.experiments.cube_wrench.train.evaluate_greedy`: world `i` of a batch is
    reset with exactly one seed (`args.seed + EVAL_SEED_OFFSET + episode`) and only that
    world's FIRST episode is recorded, so fast successes cannot dominate the number. The
    batches run `args.num_envs` worlds at a time; `finished_infos` carries the CPU-compatible
    terminal info dicts the JSON summary needs.
    """
    from callosum.experiments.coop_tasks.env_gpu import CoopBatchGPU

    vec = CoopBatchGPU(
        cfg.task,
        args.num_envs,
        partner_mode=env_mode,
        config=cfg.coop_env_config(),
        seed=args.seed + EVAL_SEED_OFFSET,
        device=args.device,
    )
    max_control_steps = int(vec.config.max_control_steps)
    records: list[EpisodeRecord] = []
    infos: list[dict] = []
    try:
        while len(records) < args.episodes:
            batch = min(vec.num_envs, args.episodes - len(records))
            obs = vec.reset(seed=args.seed + EVAL_SEED_OFFSET + len(records))
            policy_obs = builder.reset(obs) if builder is not None else obs
            recorded = [False] * vec.num_envs
            for _step in range(max_control_steps + 1):
                actions = {}
                for name, policy in zip(agent_keys, policies):
                    action = policy.get_action(policy_obs[name], deterministic=True)
                    actions[name] = torch.clamp(action, -1.0, 1.0)
                result = vec.step(actions)
                done = result.terminated | result.truncated
                policy_obs = (
                    builder.step(result.obs, done.float()) if builder is not None else result.obs
                )
                done_rows = done.nonzero(as_tuple=False).flatten().cpu().tolist()
                if len(done_rows) != len(result.finished_returns):
                    raise RuntimeError(
                        "GPU env contract broken: "
                        f"{len(done_rows)} worlds finished but "
                        f"{len(result.finished_returns)} finished statistics reported"
                    )
                for position, world in enumerate(done_rows):
                    if world >= batch or recorded[world]:
                        continue
                    recorded[world] = True
                    success = bool(result.finished_successes[position])
                    length = int(result.finished_lengths[position])
                    records.append(
                        EpisodeRecord(
                            success=success,
                            time_to_success=length if success else None,
                            decisions=length,
                        )
                    )
                    infos.append(result.finished_infos[position])
                if all(recorded[:batch]):
                    break
            else:
                raise RuntimeError(
                    "the warp evaluation did not finish the first episode of every world "
                    f"within {max_control_steps + 1} steps; the env's episode budget is broken"
                )
    finally:
        vec.close()
    return records, infos


def main(argv: list[str] | None = None) -> int:
    """Evaluate the checkpoint, print the summary and write the JSON report."""
    args = parse_args(argv)
    payload = load_checkpoint(args.checkpoint)
    cfg = config_from_payload(payload)
    if cfg.task not in _COOP_TASKS:
        raise ValueError(
            f"checkpoint task is {cfg.task!r}; callosum/experiments/coop_tasks/eval.py evaluates the coop tasks "
            f"{_COOP_TASKS} (the cube has callosum/experiments/cube_wrench/eval.py)"
        )
    input_mode = resolve_input_mode(payload, args)
    env_mode, latent_dim = check_checkpoint_compat(payload, input_mode)
    env_config = cfg.coop_env_config()
    # The hidden-partner variant is part of the run config (TossConfig.occlusion); the
    # checkpoint records it, so the report states which observation layout it was judged on.
    occlusion = str(getattr(env_config, "occlusion", "none"))
    device = torch.device(args.device)
    policies = make_policies(
        env_mode,
        device,
        latent_dim=latent_dim,
        input_size=cfg.env_obs_size,
        action_dim=cfg.action_size,
    )
    load_weights(payload, policies, agent_keys=cfg.agent_keys)
    builder: LatentInputBuilder | None = None
    bijepa = make_bijepa(cfg, device)
    if bijepa is not None:
        load_bijepa_weights(payload, bijepa)
        bijepa.eval()
        builder = LatentInputBuilder(bijepa, device=device, agent_keys=cfg.agent_keys)

    records: list[EpisodeRecord] = []
    infos: list[dict] = []
    print(
        f"eval {args.checkpoint} | task {cfg.task} | input_mode {input_mode} (env "
        f"partner_mode {env_mode}, occlusion {occlusion}) | backend {args.backend} | "
        f"policy input {cfg.env_obs_size + latent_dim} | "
        f"{args.episodes} episodes from held-out seed {args.seed + EVAL_SEED_OFFSET} | "
        "greedy policies",
        flush=True,
    )
    if args.backend == "warp":
        records, infos = evaluate_warp(
            cfg,
            env_mode,
            latent_dim,
            policies,
            builder,
            cfg.agent_keys,
            args,
            device,
        )
    else:
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

    summary = summarize(records, infos, cfg.task)
    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "task": cfg.task,
        "input_mode": input_mode,
        "partner_mode": env_mode,
        "occlusion": occlusion,
        "backend": args.backend,
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
    if cfg.task == "cartpole":
        diagonal = f"angle_deg={summary['mean_angle_deg']:.1f}"
    elif cfg.task == "toss":
        diagonal = f"passes={summary['mean_passes']:.2f}/{summary['target_passes']:.0f}"
    else:
        diagonal = f"angle_deg={summary['mean_diagnostic']:.1f}"
    print(
        f"success_rate={summary['success_rate']:.3f} "
        f"({summary['successes']}/{summary['episodes']}, {summary['failures']} failures) | "
        f"{diagonal} | "
        f"steps={summary['steps_mean']:.1f} | "
        f"steps_to_success={'-' if time is None else f'{time:.1f}'}{extra}"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
